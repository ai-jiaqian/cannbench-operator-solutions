/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_plugin.cpp
 * \brief MLA API layer - torch bindings (compiled with g++).
 *
 * Only shape / dtype metadata is inspected here; every arithmetic and data movement step runs
 * inside the custom device kernels.  The device scratch (the f32 score matrix, the attention
 * weights and the Matmul system workspace) is allocated as ordinary device tensors and every byte
 * of it is produced by a kernel.  No host readback, no host-side runtime call.
 *
 * Layout handling: with numKVHeads == 1 the per-batch query / output arrays of BSND ([B,S,Nq,D])
 * and BNSD ([B,Nq,S,D]) are the same contiguous [S*Nq, D] block, and k_nope / k_rope / v are plain
 * [Skv, D] matrices per batch.  The kernels therefore need no permutation at all: the flat row
 * index is identical for input and output, and only the causal mask needs the row -> key position
 * decoding (BSND s = r / Nq, BNSD s = r % S).
 */

#include <tuple>
#include <string>
#include <cmath>
#include <algorithm>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "../op_kernel/mla_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("mla(Tensor q_nope, Tensor q_rope, Tensor k_nope, Tensor k_rope, Tensor v, "
          "int numKVHeads=1, float scaleValue=-1.0, str inputLayout=\"BSND\", "
          "bool is_causal=False) -> Tensor");
}

namespace {

struct MlaDims {
    int64_t B = 0;
    int64_t S = 0;
    int64_t Nq = 0;
    int64_t Skv = 0;
    int64_t Dn = 0;
    int64_t Dr = 0;
    int64_t bnMode = 0; // 0 = BSND, 1 = BNSD
};

bool mla_is_bsnd(const std::string& layout)
{
    return layout == "BSND" || layout == "bsnd";
}

/*! \brief band height for the cube stages.
 *
 *  Every band owns RQ consecutive query rows of ONE batch and re-reads that batch's whole K / V,
 *  so the K / V traffic is proportional to totalRows / RQ.  RQ is therefore made as large as it
 *  reasonably can be: the largest power-of-two divisor of R (which keeps bands aligned to batch
 *  boundaries) that does not exceed both a hard cap and totalRows / aic, so at least one band per
 *  hardware block remains.  A larger RQ also amortises the Matmul per-call setup over more
 *  baseM tiles.  R is S * Nq with Nq in {64, 128}, so a power-of-two divisor up to 64 always exists. */
int64_t mla_pick_rq(int64_t R, int64_t totalRows, int64_t aic)
{
    const int64_t rqMax = 1024;
    if (aic < 1) {
        aic = 1;
    }
    int64_t cap = totalRows / aic;
    if (cap < 16) {
        cap = 16;
    }
    if (cap > rqMax) {
        cap = rqMax;
    }
    if (cap > R) {
        cap = R;
    }
    int64_t rq = 1;
    while (rq * 2 <= cap && (R % (rq * 2)) == 0) {
        rq *= 2;
    }
    if ((R % rq) != 0) {
        rq = R;
    }
    return rq;
}

MlaDims mla_resolve(const torch::Tensor& q_nope, const torch::Tensor& q_rope,
                    const torch::Tensor& k_nope, const torch::Tensor& k_rope,
                    const torch::Tensor& v, int64_t numKVHeads, const std::string& inputLayout)
{
    MlaDims d;
    TORCH_CHECK(q_nope.dim() == 4 && q_rope.dim() == 4 && k_nope.dim() == 4 &&
                    k_rope.dim() == 4 && v.dim() == 4,
                "mla: all inputs must be 4-D.");
    const bool bsnd = mla_is_bsnd(inputLayout);
    TORCH_CHECK(bsnd || inputLayout == "BNSD" || inputLayout == "bnsd",
                "mla: unsupported inputLayout.");
    TORCH_CHECK(numKVHeads == 1, "mla: only numKVHeads == 1 is supported.");
    const auto dt = q_nope.scalar_type();
    TORCH_CHECK(dt == torch::kFloat16 || dt == torch::kBFloat16,
                "mla: only float16 / bfloat16 are supported.");
    TORCH_CHECK(q_rope.scalar_type() == dt && k_nope.scalar_type() == dt &&
                    k_rope.scalar_type() == dt && v.scalar_type() == dt,
                "mla: all inputs must share one dtype.");

    d.bnMode = bsnd ? 0 : 1;
    d.B = q_nope.size(0);
    d.Dn = q_nope.size(3);
    d.Dr = q_rope.size(3);
    if (bsnd) {
        d.S = q_nope.size(1);
        d.Nq = q_nope.size(2);
        d.Skv = k_nope.size(1);
        TORCH_CHECK(k_nope.size(2) == 1 && k_rope.size(2) == 1 && v.size(2) == 1,
                    "mla: only numKVHeads == 1 is supported.");
    } else {
        d.Nq = q_nope.size(1);
        d.S = q_nope.size(2);
        d.Skv = k_nope.size(2);
        TORCH_CHECK(k_nope.size(1) == 1 && k_rope.size(1) == 1 && v.size(1) == 1,
                    "mla: only numKVHeads == 1 is supported.");
    }
    TORCH_CHECK(q_rope.size(0) == d.B && q_rope.size(3) == d.Dr, "mla: q_rope shape mismatch.");
    TORCH_CHECK(q_rope.size(d.bnMode == 0 ? 1 : 2) == d.S &&
                    q_rope.size(d.bnMode == 0 ? 2 : 1) == d.Nq,
                "mla: q_rope shape does not match q_nope.");
    TORCH_CHECK(k_rope.size(0) == d.B && k_rope.size(3) == d.Dr &&
                    k_rope.size(d.bnMode == 0 ? 1 : 2) == d.Skv,
                "mla: k_rope shape mismatch.");
    TORCH_CHECK(k_nope.size(3) == d.Dn && v.size(3) == d.Dn, "mla: head dim mismatch.");
    return d;
}

} // namespace

torch::Tensor mla_meta(const torch::Tensor& q_nope, const torch::Tensor& q_rope,
                       const torch::Tensor& k_nope, const torch::Tensor& k_rope,
                       const torch::Tensor& v, int64_t numKVHeads, double scaleValue,
                       const std::string& inputLayout, bool is_causal)
{
    (void)q_rope;
    (void)k_nope;
    (void)k_rope;
    (void)numKVHeads;
    (void)scaleValue;
    (void)is_causal;
    MlaDims d = mla_resolve(q_nope, q_rope, k_nope, k_rope, v, numKVHeads, inputLayout);
    (void)d;
    return torch::empty_like(q_nope);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("mla", mla_meta);
}

torch::Tensor mla_npu(const torch::Tensor& q_nope, const torch::Tensor& q_rope,
                      const torch::Tensor& k_nope, const torch::Tensor& k_rope,
                      const torch::Tensor& v, int64_t numKVHeads, double scaleValue,
                      const std::string& inputLayout, bool is_causal)
{
    const c10::OptionalDeviceGuard guard(q_nope.device());

    auto qn = q_nope.contiguous();
    auto qr = q_rope.contiguous();
    auto kn = k_nope.contiguous();
    auto kr = k_rope.contiguous();
    auto vv = v.contiguous();

    MlaDims d = mla_resolve(qn, qr, kn, kr, vv, numKVHeads, inputLayout);
    auto y = torch::empty_like(qn);

    const int64_t R = d.S * d.Nq;
    const int64_t totalRows = d.B * R;
    if (totalRows <= 0 || d.Skv <= 0 || d.Dn <= 0 || d.Dr <= 0) {
        return y;
    }

    const float scale = (scaleValue > 0.0)
                            ? static_cast<float>(scaleValue)
                            : static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.Dn + d.Dr)));

    int64_t aic = mla_aic_num();
    if (aic <= 0) {
        aic = 20;
    }
    const int64_t rq = mla_pick_rq(R, totalRows, aic);
    const int64_t nBandsTot = totalRows / rq;

    int64_t nBlk = (nBandsTot < aic) ? nBandsTot : aic;
    if (nBlk < 1) {
        nBlk = 1;
    }

    int64_t perWs = mla_sys_ws_bytes();
    if (perWs <= 0) {
        perWs = 16 * 1024 * 1024;
    }
    // each hardware block owns its own slice of the Matmul system workspace
    const int64_t wsCap = (int64_t)768 * 1024 * 1024;
    if (nBlk * perWs > wsCap) {
        nBlk = wsCap / perWs;
        if (nBlk < 1) {
            nBlk = 1;
        }
        if (nBlk > nBandsTot) {
            nBlk = nBandsTot;
        }
    }
    const int64_t wsBytes = perWs * nBlk;

    auto sBuf = torch::empty({totalRows * d.Skv}, qn.options().dtype(torch::kFloat));
    auto pBuf = torch::empty({totalRows * d.Skv}, qn.options());
    auto wsBuf = torch::empty({wsBytes / 4 + 8}, qn.options().dtype(torch::kFloat));

    const int64_t dt = (qn.scalar_type() == torch::kBFloat16) ? 1 : 0;
    MlaTiling tqn;
    MlaTiling tqr;
    MlaTiling tpv;
    TORCH_CHECK(mla_tiling_mm(dt, rq, d.Skv, d.Dn, 1, 1, &tqn) == 0, "mla: qk(nope) tiling failed.");
    TORCH_CHECK(mla_tiling_mm(dt, rq, d.Skv, d.Dr, 1, 1, &tqr) == 0, "mla: qk(rope) tiling failed.");
    TORCH_CHECK(mla_tiling_mm(dt, rq, d.Dn, d.Skv, 0, 0, &tpv) == 0, "mla: pv tiling failed.");

    /* The softmax tile owns RG consecutive query rows (RG always divides R, so a tile never
     * straddles a batch boundary).  The per-tile cost -- two UB<->GM copies, the scale pass, the
     * cast and the two pipe synchronisations -- is paid once whatever the tile width, so the tile
     * is sized from a target element count instead of a fixed row count.  That removes most of the
     * per-tile dispatch on the small-key-count shapes, while the UB footprint stays at the same
     * ~140 KB the largest key count already required (RG is capped so the tile never grows past
     * 8192 elements). */
    int64_t br = 1;
    {
        int64_t want = 8192 / d.Skv;
        if (want > 32) {
            want = 32;
        }
        if (want < 1) {
            want = 1;
        }
        while (br * 2 <= want) {
            br *= 2;
        }
    }
    while (br > 1 && (R % br) != 0) {
        br /= 2;
    }

    int64_t aiv = mla_aiv_num();
    if (aiv <= 0) {
        aiv = 40;
    }
    int64_t nTiles = totalRows / br;
    int64_t nBlkS = (nTiles < aiv) ? nTiles : aiv;
    if (nBlkS < 1) {
        nBlkS = 1;
    }

    GM_ADDR qnPtr = reinterpret_cast<GM_ADDR>(qn.data_ptr());
    GM_ADDR qrPtr = reinterpret_cast<GM_ADDR>(qr.data_ptr());
    GM_ADDR knPtr = reinterpret_cast<GM_ADDR>(kn.data_ptr());
    GM_ADDR krPtr = reinterpret_cast<GM_ADDR>(kr.data_ptr());
    GM_ADDR vvPtr = reinterpret_cast<GM_ADDR>(vv.data_ptr());
    GM_ADDR yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());
    GM_ADDR sPtr = reinterpret_cast<GM_ADDR>(sBuf.data_ptr());
    GM_ADDR pPtr = reinterpret_cast<GM_ADDR>(pBuf.data_ptr());
    GM_ADDR wsPtr = reinterpret_cast<GM_ADDR>(wsBuf.data_ptr());

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    const int64_t B = d.B;
    const int64_t Skv = d.Skv;
    const int64_t Nq = d.Nq;
    const int64_t S = d.S;
    const int64_t bnMode = d.bnMode;
    const int64_t causal = is_causal ? 1 : 0;
    const int64_t Dn = d.Dn;
    const int64_t Dr = d.Dr;

    if (dt == 0) {
        launch_mla_qk_half(qnPtr, knPtr, sPtr, wsPtr, tqn, B, R, Skv, Dn, rq, perWs, nBlk,
                           nBandsTot, 0, stream);
        launch_mla_qk_half(qrPtr, krPtr, sPtr, wsPtr, tqr, B, R, Skv, Dr, rq, perWs, nBlk,
                           nBandsTot, 1, stream);
        launch_mla_softmax_half(sPtr, pPtr, totalRows, R, Skv, S, Nq, bnMode, causal, scale, br,
                                nBlkS, stream);
        launch_mla_pv_half(pPtr, vvPtr, yPtr, wsPtr, tpv, B, R, Skv, Dn, rq, perWs, nBlk, nBandsTot,
                           stream);
    } else {
        launch_mla_qk_bfloat16(qnPtr, knPtr, sPtr, wsPtr, tqn, B, R, Skv, Dn, rq, perWs, nBlk,
                               nBandsTot, 0, stream);
        launch_mla_qk_bfloat16(qrPtr, krPtr, sPtr, wsPtr, tqr, B, R, Skv, Dr, rq, perWs, nBlk,
                               nBandsTot, 1, stream);
        launch_mla_softmax_bfloat16(sPtr, pPtr, totalRows, R, Skv, S, Nq, bnMode, causal, scale, br,
                                    nBlkS, stream);
        launch_mla_pv_bfloat16(pPtr, vvPtr, yPtr, wsPtr, tpv, B, R, Skv, Dn, rq, perWs, nBlk,
                               nBandsTot, stream);
    }
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("mla", mla_npu);
}

} // namespace cann_bench
