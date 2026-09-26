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
 * \file gqa_qk_kernel.cpp
 * \brief GQA stage 2 (pure cube): scores[bg] = Qp[bg] @ Kp[bg]^T
 *
 *   scores[bg][m, j] = sum_d Qp[bg][m, d] * Kp[bg][j, d]      m in [0, G*S)   j in [0, S_kv)
 *
 * One block owns one (batch, kv-head) pair and one row tile of it.  The accumulator is fp32 in L0C
 * and the fixpipe narrows it to the operator dtype on the way out, which is exactly the reference's
 * "matmul in fp32, round the scores once to T" behaviour.
 *
 * A is stored [M, K] (not transposed) and B is stored [N, K] (transposed), both plain contiguous
 * matrices produced by the pack stage.
 */

#define ASCENDC_CUBE_ONLY

#include <algorithm>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#if defined(__has_include)
#if __has_include("lib/matmul_intf.h")
#include "lib/matmul_intf.h"
#elif __has_include("matmul_intf.h")
#include "matmul_intf.h"
#endif
#if __has_include("tiling/tiling_api.h")
#include "tiling/tiling_api.h"
#elif __has_include("tiling_api.h")
#include "tiling_api.h"
#endif
#endif

#include "gqa_launch.h"
#include "gqa_dev_tiling.h"

using namespace AscendC;

namespace {

void GqaCopyHostTiling(optiling::TCubeTiling &h, GqaCubeTiling &d)
{
    d.usedCoreNum = static_cast<int32_t>(h.get_usedCoreNum());
    d.M = static_cast<int32_t>(h.get_M());
    d.N = static_cast<int32_t>(h.get_N());
    d.Ka = static_cast<int32_t>(h.get_Ka());
    d.Kb = static_cast<int32_t>(h.get_Kb());
    d.singleCoreM = static_cast<int32_t>(h.get_singleCoreM());
    d.singleCoreN = static_cast<int32_t>(h.get_singleCoreN());
    d.singleCoreK = static_cast<int32_t>(h.get_singleCoreK());
    d.baseM = static_cast<int32_t>(h.get_baseM());
    d.baseN = static_cast<int32_t>(h.get_baseN());
    d.baseK = static_cast<int32_t>(h.get_baseK());
    d.depthA1 = static_cast<int32_t>(h.get_depthA1());
    d.depthB1 = static_cast<int32_t>(h.get_depthB1());
    d.stepM = static_cast<int32_t>(h.get_stepM());
    d.stepN = static_cast<int32_t>(h.get_stepN());
    d.isBias = static_cast<int32_t>(h.get_isBias());
    d.transLength = static_cast<int32_t>(h.get_transLength());
    d.iterateOrder = static_cast<int32_t>(h.get_iterateOrder());
    d.shareMode = static_cast<int32_t>(h.get_shareMode());
    d.shareL1Size = static_cast<int32_t>(h.get_shareL1Size());
    d.shareL0CSize = static_cast<int32_t>(h.get_shareL0CSize());
    d.shareUbSize = static_cast<int32_t>(h.get_shareUbSize());
    d.batchM = static_cast<int32_t>(h.get_batchM());
    d.batchN = static_cast<int32_t>(h.get_batchN());
    d.batchNum = static_cast<int32_t>(h.get_BatchNum());
    d.stepKa = static_cast<int32_t>(h.get_stepKa());
    d.stepKb = static_cast<int32_t>(h.get_stepKb());
    d.pad0 = 0;
    d.pad1 = 0;
}

/* One single-core Matmul tiling for      C[m, n] = sum_k A[m, k] * B[n, k]   (bTransposed)
 *                                  or    C[m, n] = sum_k A[m, k] * B[k, n]   (not transposed) */
bool GqaMakeTiling(int64_t m, int64_t n, int64_t k, int64_t bTransposed, int64_t isBf16,
                   GqaCubeTiling &out)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MatmulApiTiling ti(*plat);

    const matmul_tiling::DataType opDt =
        (isBf16 != 0) ? matmul_tiling::DataType::DT_BFLOAT16 : matmul_tiling::DataType::DT_FLOAT16;

    ti.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, opDt, false);
    ti.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, opDt,
                bTransposed != 0);
    ti.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, opDt);
    ti.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   matmul_tiling::DataType::DT_FLOAT);
    ti.SetBias(false);
    ti.SetShape(m, n, k);
    ti.SetOrgShape(m, n, k);

    optiling::TCubeTiling hostTiling;
    if (ti.GetTiling(hostTiling) == -1) {
        return false;
    }
    GqaCopyHostTiling(hostTiling, out);
    return true;
}

}  // namespace

bool gqa_build_qk_tiling(int64_t m, int64_t n, int64_t k, int64_t isBf16, GqaCubeTiling &out)
{
    return GqaMakeTiling(m, n, k, 1, isBf16, out);
}

bool gqa_build_pv_tiling(int64_t m, int64_t n, int64_t k, int64_t isBf16, GqaCubeTiling &out)
{
    return GqaMakeTiling(m, n, k, 0, isBf16, out);
}

int64_t gqa_lib_workspace_size()
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    return static_cast<int64_t>(plat->GetLibApiWorkSpaceSize());
}

int64_t gqa_aiv_core_num()
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t n = static_cast<int64_t>(plat->GetCoreNumAiv());
    if (n <= 0) {
        n = 1;
    }
    return n;
}

template <typename T>
__global__ __aicore__ void gqa_qk_kernel(GM_ADDR qpAddr, GM_ADDR kpAddr, GM_ADDR scAddr,
                                         GM_ADDR sysWs, int64_t BG, int64_t RS, int64_t Skv,
                                         int64_t D, int64_t MT, int64_t nMT, GqaCubeTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    if (blk >= BG * nMT) {
        return;
    }
    const int64_t bg = blk / nMT;
    const int64_t mi = blk % nMT;
    const int64_t m0 = mi * MT;
    int64_t mlen = RS - m0;
    if (mlen > MT) {
        mlen = MT;
    }
    if (mlen <= 0) {
        return;
    }

    AscendC::SetSysWorkspace(sysWs);

    AscendC::TPipe pipe;
    using A_T = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using B_T = MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using C_T = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BIAS_T = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    Matmul<A_T, B_T, C_T, BIAS_T> mm;
    AscendC::tiling::TCubeTiling devTiling;
    GqaDevFill(t, devTiling);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &devTiling);

    AscendC::GlobalTensor<T> aG;
    AscendC::GlobalTensor<T> bG;
    AscendC::GlobalTensor<T> cG;
    aG.SetGlobalBuffer((__gm__ T *)qpAddr + (bg * RS + m0) * D);
    bG.SetGlobalBuffer((__gm__ T *)kpAddr + bg * Skv * D);
    cG.SetGlobalBuffer((__gm__ T *)scAddr + (bg * RS + m0) * Skv);

    mm.SetTensorA(aG, false);
    mm.SetTensorB(bG, true);
    mm.SetOrgShape(static_cast<int32_t>(mlen), static_cast<int32_t>(Skv), static_cast<int32_t>(D));
    mm.SetSingleShape(static_cast<int32_t>(mlen), static_cast<int32_t>(Skv), static_cast<int32_t>(D));
    mm.IterateAll(cG, 0);
    mm.End();
}

extern "C" void launch_gqa_qk_f16(GM_ADDR qp, GM_ADDR kp, GM_ADDR scores, GM_ADDR sysWs,
                                  int64_t BG, int64_t RS, int64_t Skv, int64_t D, int64_t MT,
                                  int64_t nMT, GqaCubeTiling t, void *stream)
{
    gqa_qk_kernel<half><<<BG * nMT, nullptr, stream>>>(qp, kp, scores, sysWs, BG, RS, Skv, D, MT,
                                                       nMT, t);
}

extern "C" void launch_gqa_qk_bf16(GM_ADDR qp, GM_ADDR kp, GM_ADDR scores, GM_ADDR sysWs,
                                   int64_t BG, int64_t RS, int64_t Skv, int64_t D, int64_t MT,
                                   int64_t nMT, GqaCubeTiling t, void *stream)
{
    gqa_qk_kernel<bfloat16_t><<<BG * nMT, nullptr, stream>>>(qp, kp, scores, sysWs, BG, RS, Skv, D,
                                                             MT, nMT, t);
}
