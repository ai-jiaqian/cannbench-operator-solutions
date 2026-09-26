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
 * \file gqa_plugin.cpp
 * \brief GQA API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <algorithm>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/gqa_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def(
        "gqa(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0, bool is_causal=False) "
        "-> Tensor");
}

torch::Tensor gqa_meta(const torch::Tensor &query, const torch::Tensor &key,
                       const torch::Tensor &value, double scaleValue, bool is_causal)
{
    (void)key;
    (void)value;
    (void)scaleValue;
    (void)is_causal;
    TORCH_CHECK(query.dim() == 4, "gqa: query must be [B, S, Nq, D].");
    return torch::empty_like(query);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("gqa", gqa_meta);
}

namespace {

/* Row-tile sizing for the two cube stages.
 *
 * Each work item owns `MT` consecutive rows of one (batch, kv-head) block, so the total number of
 * work items is BG * ceil(RS / MT).  A small MT turns a large prefill into thousands of very short
 * Matmul invocations whose per-invocation pipeline setup dwarfs the tile arithmetic, so the tile is
 * grown until the grid is a small multiple of the physical cube core count.  RS itself is the upper
 * bound (one work item per block), and MT is never allowed below a floor that keeps a sensible
 * amount of work per invocation.
 */
constexpr int64_t GQA_TARGET_BLOCKS = 192;
constexpr int64_t GQA_MT_MIN = 128;

int64_t GqaPickRowsPerTile(int64_t RS, int64_t BG, int64_t &nMT)
{
    int64_t want = 1;
    if (BG < GQA_TARGET_BLOCKS) {
        want = (GQA_TARGET_BLOCKS + BG - 1) / BG;
        const int64_t maxByRows = (RS + GQA_MT_MIN - 1) / GQA_MT_MIN;
        if (want > maxByRows) {
            want = maxByRows;
        }
    }
    if (want < 1) {
        want = 1;
    }
    int64_t mt = (RS + want - 1) / want;
    mt = ((mt + 15) / 16) * 16;
    if (mt > RS) {
        mt = RS;
    }
    if (mt < 1) {
        mt = 1;
    }
    nMT = (RS + mt - 1) / mt;
    return mt;
}

struct GqaPlan {
    int64_t B, S, Nq, Nkv, Skv, D, G, BG, RS, MT, nMT;
    float scale;
    int32_t causal;
    int32_t useMax;
    int64_t isBf16;
    int32_t flat;
    GqaCubeTiling qk, pv;
    int64_t totalElems;
    int64_t totalRows;
    int64_t chunkRows;
    int64_t smBlocks, smRowsPerCore;
    int64_t packBase[3]; /* element offsets of qp/kp/vp inside the scratch region (packed pipeline) */
};

bool GqaMakePlan(const torch::Tensor &q, const torch::Tensor &k, int64_t Nq, int64_t D, int64_t Skv,
                 int64_t Nkv, double scaleValue, bool is_causal, GqaPlan &p)
{
    p.B = q.size(0);
    p.S = q.size(1);
    p.Nq = Nq;
    p.D = D;
    p.Skv = Skv;
    p.Nkv = Nkv;
    if (Nkv <= 0 || Nq % Nkv != 0) {
        return false;
    }
    p.G = Nq / Nkv;
    p.BG = p.B * Nkv;
    p.RS = p.G * p.S;
    p.scale = static_cast<float>(scaleValue);
    if (!(p.scale > 0.0f)) {
        p.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    }
    p.causal = is_causal ? 1 : 0;
    p.isBf16 = (q.scalar_type() == torch::kBFloat16) ? 1 : 0;
    (void)k;

    /* Softmax is shift invariant, so the row-maximum subtraction is only needed to bound the
     * exponent.  The scores entering the exponentiation are |q.k| * scale <= D * |scale| (the
     * operator's own value range is [-1, 1]), and exp(60) = 1.1e26 is still comfortably inside
     * fp32, so below that bound the subtraction - and with it one vector/scalar barrier per row -
     * is provably redundant.  Above it the classic max-subtracted form is used, and the flat
     * formulation (which cannot mask before its maximum) is not eligible. */
    const double bound = static_cast<double>(D) * std::fabs((double)p.scale);
    p.useMax = (bound > 60.0) ? 1 : 0;

    const int64_t L = p.Skv * p.Nkv;

    /* Which formulation moves fewer bytes?
     *
     * packed = 3Q + 3K + 3V + 3S + 2Y      (pack in, pack out, unpack out)
     * flat   =  Q +  K +  V + 3*Nkv*S + Y  (native tensors, expanded score matrix)
     * where S is the [S*Nq, Skv] score matrix, so the flat score matrix is Nkv times wider.  With
     * attention's numbers (|K| = |V| and the score matrix quadratic in the sequence length) the
     * flat form wins for every short-query shape - decode and MTP, and the MQA degenerate - and
     * loses for prefill. */
    p.flat = 0;
    {
        const int64_t qE = p.B * p.S * p.Nq * p.D;
        const int64_t kE = p.B * p.Skv * p.Nkv * p.D;
        const int64_t sE = p.B * p.S * p.Nq * p.Skv;
        const int64_t packedCost = 3 * qE + 6 * kE + 3 * sE + 2 * qE;
        const int64_t flatCost = 2 * qE + 2 * kE + 3 * sE * Nkv;
        if (p.useMax == 0 && (L % 16) == 0 && L >= 64 && (64 % Nkv) == 0 &&
            flatCost < packedCost) {
            p.flat = 1;
        }
    }

    const int64_t qSzP = p.BG * p.RS * p.D;
    const int64_t kSzP = p.BG * p.Skv * p.D;
    const int64_t scSzP = p.BG * p.RS * p.Skv;

    if (p.flat != 0) {
        /* Same cube stages, pointed at the native tensors: rows (s, h) of one batch, and a score
         * column count of Skv*Nkv because B is the whole flat K[b]. */
        p.BG = p.B;
        p.RS = p.S * p.Nq;
        p.MT = GqaPickRowsPerTile(p.RS, p.BG, p.nMT);
        if (!gqa_build_qk_tiling(p.MT, L, p.D, p.isBf16, p.qk)) {
            return false;
        }
        if (!gqa_build_pv_tiling(p.MT, p.D, L, p.isBf16, p.pv)) {
            return false;
        }
        p.totalRows = p.B * p.RS;
        p.totalElems = p.totalRows * L;
        p.packBase[0] = p.packBase[1] = p.packBase[2] = 0;
    } else {
        p.MT = GqaPickRowsPerTile(p.RS, p.BG, p.nMT);
        if (!gqa_build_qk_tiling(p.MT, p.Skv, p.D, p.isBf16, p.qk)) {
            return false;
        }
        if (!gqa_build_pv_tiling(p.MT, p.D, p.Skv, p.isBf16, p.pv)) {
            return false;
        }
        p.totalRows = p.BG * p.RS;
        p.totalElems = qSzP + kSzP + kSzP + scSzP;
        p.packBase[0] = 0;
        p.packBase[1] = qSzP;
        p.packBase[2] = qSzP + kSzP;
    }

    const int64_t esz = static_cast<int64_t>(q.element_size());
    int64_t chunk = (48 * 1024) / (D * esz);
    if (chunk < 1) {
        chunk = 1;
    }
    p.chunkRows = chunk;

    int64_t cores = gqa_aiv_core_num();
    p.smBlocks = std::min<int64_t>(cores, p.totalRows);
    if (p.smBlocks < 1) {
        p.smBlocks = 1;
    }
    p.smRowsPerCore = (p.totalRows + p.smBlocks - 1) / p.smBlocks;
    p.smBlocks = (p.totalRows + p.smRowsPerCore - 1) / p.smRowsPerCore;
    return true;
}

}  // namespace

torch::Tensor gqa_npu(const torch::Tensor &query, const torch::Tensor &key,
                      const torch::Tensor &value, double scaleValue, bool is_causal)
{
    const c10::OptionalDeviceGuard guard(query.device());
    auto q = query.is_contiguous() ? query : query.contiguous();
    auto k = key.is_contiguous() ? key : key.contiguous();
    auto v = value.is_contiguous() ? value : value.contiguous();

    const int64_t Nq = q.size(2);
    const int64_t D = q.size(3);
    const int64_t Skv = k.size(1);
    const int64_t Nkv = k.size(2);

    auto y = gqa_meta(q, k, v, scaleValue, is_causal);

    GqaPlan p{};
    TORCH_CHECK(GqaMakePlan(q, k, Nq, D, Skv, Nkv, scaleValue, is_causal, p),
                "gqa: unsupported shape or tiling failure.");

    const int64_t esz = static_cast<int64_t>(q.element_size());
    auto byteOpt = q.options().dtype(torch::kUInt8);
    const int64_t swBytes = gqa_lib_workspace_size();
    const int64_t swPad = ((swBytes + 511) / 512) * 512;
    /* One allocation carries both the Matmul system workspace (first, 512B aligned) and the
     * pipeline scratch, so a single caching-allocator round trip serves the whole call.
     *
     * Packed layout : [Qp | Kp | Vp | scores]  (C of the PV stage reuses Qp, which is dead by then)
     * Flat   layout : [scores / P]             (only the expanded score matrix is ever materialised)
     */
    auto wsAll = torch::empty({swPad + p.totalElems * esz}, byteOpt);
    char *wsBase = (char *)wsAll.data_ptr();
    GM_ADDR sysWs = (GM_ADDR)wsBase;
    char *scratch = wsBase + swPad;
    GM_ADDR qp = (GM_ADDR)(scratch + p.packBase[0] * esz);
    GM_ADDR kp = (GM_ADDR)(scratch + p.packBase[1] * esz);
    GM_ADDR vp = (GM_ADDR)(scratch + p.packBase[2] * esz);
    GM_ADDR yp = qp; /* the packed query tile is dead once stage 2 has consumed it */
    GM_ADDR scFlat = (GM_ADDR)scratch;
    GM_ADDR scPack = (GM_ADDR)(scratch +
                               (p.packBase[2] + p.BG * p.Skv * p.D) * esz); /* after Vp */

    auto qPtr = (GM_ADDR)q.data_ptr();
    auto kPtr = (GM_ADDR)k.data_ptr();
    auto vPtr = (GM_ADDR)v.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const int64_t B = p.B;
    const int64_t S = p.S;
    const int64_t NqC = p.Nq;
    const int64_t NkvC = p.Nkv;
    const int64_t SkvC = p.Skv;
    const int64_t DC = p.D;
    const int64_t GC = p.G;
    const int64_t chunkRows = p.chunkRows;
    const int64_t mt = p.MT;
    const int64_t nmt = p.nMT;
    const int64_t bg = p.BG;
    const int64_t rs = p.RS;
    const int64_t isBf16 = p.isBf16;
    const float scale = p.scale;
    const int32_t causal = p.causal;
    const int32_t useMax = p.useMax;
    const int64_t flat = p.flat;
    const int64_t smBlocks = p.smBlocks;
    const int64_t smRows = p.smRowsPerCore;
    const int64_t totalRows = p.totalRows;
    const int64_t L = SkvC * NkvC;
    const GqaCubeTiling tQk = p.qk;
    const GqaCubeTiling tPv = p.pv;

    auto acl_call = [=]() -> int {
        if (flat != 0) {
            if (isBf16 != 0) {
                launch_gqa_qk_bf16(qPtr, kPtr, scFlat, sysWs, B, rs, L, DC, mt, nmt, tQk, stream);
                launch_gqa_softmax_flat_bf16(scFlat, totalRows, S, NqC, GC, NkvC, SkvC, scale,
                                             causal, smBlocks, smRows, stream);
                launch_gqa_pv_bf16(scFlat, vPtr, yPtr, sysWs, B, rs, L, DC, mt, nmt, tPv, stream);
            } else {
                launch_gqa_qk_f16(qPtr, kPtr, scFlat, sysWs, B, rs, L, DC, mt, nmt, tQk, stream);
                launch_gqa_softmax_flat_f16(scFlat, totalRows, S, NqC, GC, NkvC, SkvC, scale,
                                            causal, smBlocks, smRows, stream);
                launch_gqa_pv_f16(scFlat, vPtr, yPtr, sysWs, B, rs, L, DC, mt, nmt, tPv, stream);
            }
        } else if (isBf16 != 0) {
            launch_gqa_pack_bf16(qPtr, kPtr, vPtr, qp, kp, vp, B, S, NqC, NkvC, SkvC, DC, chunkRows,
                                 stream);
            launch_gqa_qk_bf16(qp, kp, scPack, sysWs, bg, rs, SkvC, DC, mt, nmt, tQk, stream);
            launch_gqa_softmax_bf16(scPack, totalRows, rs, GC, S, SkvC, scale, causal, useMax,
                                    smBlocks, smRows, stream);
            launch_gqa_pv_bf16(scPack, vp, yp, sysWs, bg, rs, SkvC, DC, mt, nmt, tPv, stream);
            launch_gqa_unpack_bf16(yp, yPtr, B, S, NqC, NkvC, DC, chunkRows, stream);
        } else {
            launch_gqa_pack_f16(qPtr, kPtr, vPtr, qp, kp, vp, B, S, NqC, NkvC, SkvC, DC, chunkRows,
                                stream);
            launch_gqa_qk_f16(qp, kp, scPack, sysWs, bg, rs, SkvC, DC, mt, nmt, tQk, stream);
            launch_gqa_softmax_f16(scPack, totalRows, rs, GC, S, SkvC, scale, causal, useMax,
                                   smBlocks, smRows, stream);
            launch_gqa_pv_f16(scPack, vp, yp, sysWs, bg, rs, SkvC, DC, mt, nmt, tPv, stream);
            launch_gqa_unpack_f16(yp, yPtr, B, S, NqC, NkvC, DC, chunkRows, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Gqa", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("gqa", gqa_npu);
}

}  // namespace cann_bench
