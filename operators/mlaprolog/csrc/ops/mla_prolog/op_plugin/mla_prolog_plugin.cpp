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
 * \file mla_prolog_plugin.cpp
 * \brief MlaProlog API layer - torch bindings (compiled with g++).
 *
 * The API layer inspects metadata only.  Every arithmetic operation, dtype conversion, layout transform and
 * workspace initialisation happens inside the custom device kernels launched from here.
 */

#include <tuple>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/mla_prolog_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("mla_prolog(Tensor token_x, Tensor w_dq, Tensor w_uq_qr, Tensor w_uk, Tensor w_dkv_kr, "
          "Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, "
          "int n_heads, float rmsnorm_epsilon_cq=1e-5, float rmsnorm_epsilon_ckv=1e-5) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
}

namespace {

struct MlaDims {
    int64_t B = 0;
    int64_t S = 0;
    int64_t He = 0;
    int64_t Hcq = 0;
    int64_t NW = 0;
    int64_t N = 0;
    int64_t PH = 0;
    int64_t D = 0;
    int64_t Dr = 0;
    int64_t Hckv = 0;
    int64_t HkvDr = 0;
    int64_t M = 0;
};

void mla_check(const torch::Tensor& t, const char* what, int64_t ndim)
{
    TORCH_CHECK(t.dim() == ndim, "mla_prolog: ", what, " must be ", ndim, "-D.");
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, "mla_prolog: ", what, " must be bfloat16.");
}

MlaDims mla_resolve(const torch::Tensor& token_x, const torch::Tensor& w_dq, const torch::Tensor& w_uq_qr,
                    const torch::Tensor& w_uk, const torch::Tensor& w_dkv_kr,
                    const torch::Tensor& g_cq, const torch::Tensor& g_ckv,
                    const torch::Tensor& rope_sin, const torch::Tensor& rope_cos, int64_t n_heads)
{
    MlaDims d;
    mla_check(token_x, "token_x", 3);
    mla_check(w_dq, "w_dq", 2);
    mla_check(w_uq_qr, "w_uq_qr", 2);
    mla_check(w_uk, "w_uk", 3);
    mla_check(w_dkv_kr, "w_dkv_kr", 2);
    mla_check(g_cq, "rmsnorm_gamma_cq", 1);
    mla_check(g_ckv, "rmsnorm_gamma_ckv", 1);
    mla_check(rope_sin, "rope_sin", 3);
    mla_check(rope_cos, "rope_cos", 3);

    d.B = token_x.size(0);
    d.S = token_x.size(1);
    d.He = token_x.size(2);
    d.Hcq = w_uq_qr.size(0);
    d.NW = w_uq_qr.size(1);
    d.N = n_heads;
    TORCH_CHECK(d.N > 0 && d.NW % d.N == 0, "mla_prolog: w_uq_qr[1] must be n_heads*(D+Dr).");
    d.PH = d.NW / d.N;
    d.D = w_uk.size(1);
    d.Hckv = w_uk.size(2);
    TORCH_CHECK(d.PH > d.D, "mla_prolog: bad D/Dr split.");
    d.Dr = d.PH - d.D;
    d.HkvDr = w_dkv_kr.size(1);
    d.M = d.B * d.S;

    TORCH_CHECK(w_dq.size(0) == d.He && w_dq.size(1) == d.Hcq, "mla_prolog: w_dq shape mismatch.");
    TORCH_CHECK(w_dkv_kr.size(0) == d.He, "mla_prolog: w_dkv_kr shape mismatch.");
    TORCH_CHECK(d.HkvDr == d.Hckv + d.Dr, "mla_prolog: w_dkv_kr[1] must be Hckv+Dr.");
    TORCH_CHECK(w_uk.size(0) == d.N, "mla_prolog: w_uk[0] must be n_heads.");
    TORCH_CHECK(g_cq.size(0) == d.Hcq && g_ckv.size(0) == d.Hckv, "mla_prolog: gamma length mismatch.");
    TORCH_CHECK(rope_sin.sizes() == rope_cos.sizes(), "mla_prolog: rope_sin/cos shape mismatch.");
    TORCH_CHECK(rope_sin.size(0) == d.B && rope_sin.size(1) == d.S && rope_sin.size(2) == d.Dr,
                "mla_prolog: rope table must be [B,S,Dr].");

    /* the device kernels size their Unified Buffer scratch with these compile-time bounds */
    TORCH_CHECK(d.Hcq <= MLA_MAX_HCQ && d.HkvDr <= MLA_MAX_HKVDR && d.Hckv <= MLA_MAX_HCKV &&
                    d.Dr <= MLA_MAX_DR,
                "mla_prolog: shape outside the supported range.");
    TORCH_CHECK(d.Dr % 2 == 0, "mla_prolog: Dr must be even.");
    TORCH_CHECK(d.D % 16 == 0 && d.Dr % 16 == 0 && d.Hckv % 16 == 0 && d.Hcq % 16 == 0 &&
                    d.HkvDr % 16 == 0 && (3 * d.D) % 16 == 0,
                "mla_prolog: dimensions must be multiples of 16.");
    return d;
}

/*! \brief pick the largest per-block column count that evenly divides n and is a multiple of 16 */
int64_t PickSplitN(int64_t n, int64_t targetBlocks, int64_t* outBlocks)
{
    int64_t bestS = -1;
    int64_t bestScore = -1;
    for (int64_t s = 16; s <= n; s += 16) {
        if (n % s != 0) {
            continue;
        }
        int64_t nb = n / s;
        int64_t score = nb > targetBlocks ? nb - targetBlocks : targetBlocks - nb;
        if (bestS < 0 || score < bestScore || (score == bestScore && s > bestS)) {
            bestS = s;
            bestScore = score;
        }
    }
    if (bestS < 0) {
        bestS = n;
    }
    if (bestS <= 0) {
        bestS = n;
    }
    *outBlocks = n / bestS;
    return bestS;
}

inline int64_t AlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mla_prolog_meta(
    const torch::Tensor& token_x, const torch::Tensor& w_dq, const torch::Tensor& w_uq_qr,
    const torch::Tensor& w_uk, const torch::Tensor& w_dkv_kr, const torch::Tensor& rmsnorm_gamma_cq,
    const torch::Tensor& rmsnorm_gamma_ckv, const torch::Tensor& rope_sin, const torch::Tensor& rope_cos,
    int64_t n_heads, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv)
{
    (void)rmsnorm_epsilon_cq;
    (void)rmsnorm_epsilon_ckv;
    const MlaDims d = mla_resolve(token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, rmsnorm_gamma_cq,
                                  rmsnorm_gamma_ckv, rope_sin, rope_cos, n_heads);
    auto opts = token_x.options();
    return std::make_tuple(torch::empty({d.B, d.S, d.N, d.Hckv}, opts),
                           torch::empty({d.B, d.S, d.N, d.Dr}, opts),
                           torch::empty({d.B, d.S, d.Hckv}, opts),
                           torch::empty({d.B, d.S, d.Dr}, opts));
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("mla_prolog", mla_prolog_meta);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mla_prolog_npu(
    const torch::Tensor& token_x, const torch::Tensor& w_dq, const torch::Tensor& w_uq_qr,
    const torch::Tensor& w_uk, const torch::Tensor& w_dkv_kr, const torch::Tensor& rmsnorm_gamma_cq,
    const torch::Tensor& rmsnorm_gamma_ckv, const torch::Tensor& rope_sin, const torch::Tensor& rope_cos,
    int64_t n_heads, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv)
{
    const c10::OptionalDeviceGuard guard(token_x.device());
    const MlaDims d = mla_resolve(token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, rmsnorm_gamma_cq,
                                  rmsnorm_gamma_ckv, rope_sin, rope_cos, n_heads);

    auto opts = token_x.options();
    auto query = torch::empty({d.B, d.S, d.N, d.Hckv}, opts);
    auto queryRope = torch::empty({d.B, d.S, d.N, d.Dr}, opts);
    auto ckv = torch::empty({d.B, d.S, d.Hckv}, opts);
    auto krope = torch::empty({d.B, d.S, d.Dr}, opts);

    const int64_t M = d.M;
    if (M == 0) {
        return std::make_tuple(query, queryRope, ckv, krope);
    }

    const int64_t aic = mla_aic_num();
    const int64_t aiv = mla_aiv_num();

    // ---- workspace ----
    const int64_t szCq = M * d.Hcq * 4;
    const int64_t szDkv = M * d.HkvDr * 4;
    const int64_t szSplit = 3 * M * d.Hcq * 2;
    const int64_t szQr2 = 3 * M * d.NW * 4;
    const int64_t szQcs = d.N * M * 3 * d.D * 2;
    const int64_t szQrope = M * d.N * d.Dr * 4;
    const int64_t szQtmp3 = d.N * 3 * M * d.Hckv * 4;
    const int64_t szQtmp = d.N * M * d.Hckv * 2;
    const int64_t sysWs = mla_sys_workspace_bytes();

    int64_t off = 0;
    auto bump = [&off](int64_t sz) {
        int64_t o = off;
        off = AlignUp(off + sz, 512);
        return o;
    };
    const int64_t oCq = bump(szCq);
    const int64_t oDkv = bump(szDkv);
    const int64_t oSplit = bump(szSplit);
    const int64_t oQr2 = bump(szQr2);
    const int64_t oQcs = bump(szQcs);
    const int64_t oQrope = bump(szQrope);
    const int64_t oQtmp3 = bump(szQtmp3);
    const int64_t oQtmp = bump(szQtmp);
    const int64_t oSys = bump(sysWs);
    const int64_t total = off;

    auto ws = torch::empty({total}, opts.dtype(torch::kInt8));
    uint8_t* wsBase = (uint8_t*)ws.data_ptr();
    GM_ADDR wsPtr = (GM_ADDR)(wsBase + oSys);

    // ---- tilings ----
    int64_t nb1 = 1;
    int64_t nb2 = 1;
    int64_t nb3 = 1;
    const int64_t tgt1 = aic * d.Hcq / (d.Hcq + d.HkvDr);
    const int64_t sN1 = PickSplitN(d.Hcq, tgt1 > 1 ? tgt1 : 1, &nb1);
    const int64_t sN2 = PickSplitN(d.HkvDr, (aic - nb1) > 1 ? (aic - nb1) : 1, &nb2);
    const int64_t sN3 = PickSplitN(d.NW, aic > 1 ? aic : 1, &nb3);

    MlaTiling t1, t2, t3, t4;
    TORCH_CHECK(mla_make_tiling(M, sN1, d.He, M, d.Hcq, d.He, d.He, 0, &t1) == 0,
                "mla_prolog: tiling 1 failed.");
    TORCH_CHECK(mla_make_tiling(M, sN2, d.He, M, d.HkvDr, d.He, d.He, 0, &t2) == 0,
                "mla_prolog: tiling 2 failed.");
    TORCH_CHECK(mla_make_tiling(3 * M, sN3, d.Hcq, 3 * M, d.NW, d.Hcq, d.Hcq, 0, &t3) == 0,
                "mla_prolog: tiling 3 failed.");
    TORCH_CHECK(mla_make_tiling(3 * M, d.Hckv, d.D, 3 * M, d.Hckv, d.D, d.D, 0, &t4) == 0,
                "mla_prolog: tiling 4 failed.");

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    GM_ADDR xP = (GM_ADDR)token_x.data_ptr();
    GM_ADDR wdqP = (GM_ADDR)w_dq.data_ptr();
    GM_ADDR wuqP = (GM_ADDR)w_uq_qr.data_ptr();
    GM_ADDR wukP = (GM_ADDR)w_uk.data_ptr();
    GM_ADDR wdkvP = (GM_ADDR)w_dkv_kr.data_ptr();
    GM_ADDR gcqP = (GM_ADDR)rmsnorm_gamma_cq.data_ptr();
    GM_ADDR gckvP = (GM_ADDR)rmsnorm_gamma_ckv.data_ptr();
    GM_ADDR sinP = (GM_ADDR)rope_sin.data_ptr();
    GM_ADDR cosP = (GM_ADDR)rope_cos.data_ptr();
    GM_ADDR qP = (GM_ADDR)query.data_ptr();
    GM_ADDR qrP = (GM_ADDR)queryRope.data_ptr();
    GM_ADDR ckvP = (GM_ADDR)ckv.data_ptr();
    GM_ADDR krP = (GM_ADDR)krope.data_ptr();

    GM_ADDR cqW = (GM_ADDR)(wsBase + oCq);
    GM_ADDR dkvW = (GM_ADDR)(wsBase + oDkv);
    GM_ADDR splitW = (GM_ADDR)(wsBase + oSplit);
    GM_ADDR qr2W = (GM_ADDR)(wsBase + oQr2);
    GM_ADDR qcsW = (GM_ADDR)(wsBase + oQcs);
    GM_ADDR qropeW = (GM_ADDR)(wsBase + oQrope);
    GM_ADDR qtmp3W = (GM_ADDR)(wsBase + oQtmp3);
    GM_ADDR qtmpW = (GM_ADDR)(wsBase + oQtmp);

    auto acl_call = [=]() -> int {
        // K1  cq_raw = x @ W_DQ ; dkv = x @ W_DKV_KR
        launch_mla_mm_pair(xP, wdqP, cqW, wdkvP, dkvW, wsPtr, &t1, &t2, M, d.He, d.Hcq, d.HkvDr, sN1,
                           nb1, sN2, stream);
        // K2  c_q split, c_kv, k_rope
        launch_mla_v_norm(cqW, gcqP, dkvW, gckvP, cosP, sinP, splitW, ckvP, krP, M, d.Hcq, d.HkvDr,
                          d.Hckv, d.Dr, (float)rmsnorm_epsilon_cq, (float)rmsnorm_epsilon_ckv, aiv, stream);
        // K3  (removed) w_uk is consumed untripled by K6; the 3 q_c terms are stacked along M
        // K4  qr2 = csplit @ W_UQ_QR
        launch_mla_mm_big(splitW, wuqP, qr2W, wsPtr, &t3, sN3, nb3, stream);
        // K5  fold + split(q_c) into qcs[N][3M][D] + q_r_raw
        launch_mla_v_fold(qr2W, qcsW, qropeW, M, d.N, d.D, d.Dr, d.NW, aiv, stream);
        // K6  qtmp3 = qcs @ wuk (per head, K = D, weight read once)
        launch_mla_mm_qk(qcsW, wukP, qtmp3W, wsPtr, &t4, 3 * M, d.D, d.Hckv, d.N, stream);
        // K6b fold the three per-term products in fp32
        launch_mla_v_foldq(qtmp3W, qtmpW, M, d.N, d.Hckv, aiv, stream);
        // K7  transpose + RoPE
        launch_mla_v_final(qtmpW, qropeW, cosP, sinP, qP, qrP, M, d.N, d.Hckv, d.Dr, aiv, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MlaProlog", acl_call);

    return std::make_tuple(query, queryRope, ckv, krope);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("mla_prolog", mla_prolog_npu);
}

}  // namespace cann_bench
