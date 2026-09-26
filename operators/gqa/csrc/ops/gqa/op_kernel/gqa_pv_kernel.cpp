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
 * \file gqa_pv_kernel.cpp
 * \brief GQA stage 4 (pure cube): Yp[bg] = P[bg] @ Vp[bg]
 *
 *   Yp[bg][m, :] = sum_j P[bg][m, j] * Vp[bg][j, :]
 *
 * Same structure as the QK stage; here B is a plain [K, N] matrix (not transposed).  The fp32 L0C
 * accumulator is narrowed to the operator dtype by the fixpipe, matching the reference's single
 * rounding of the attention output.
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
#endif

#include "gqa_launch.h"
#include "gqa_dev_tiling.h"

using namespace AscendC;

template <typename T>
__global__ __aicore__ void gqa_pv_kernel(GM_ADDR pAddr, GM_ADDR vpAddr, GM_ADDR ypAddr,
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
    using B_T = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using C_T = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BIAS_T = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    Matmul<A_T, B_T, C_T, BIAS_T> mm;
    AscendC::tiling::TCubeTiling devTiling;
    GqaDevFill(t, devTiling);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &devTiling);

    AscendC::GlobalTensor<T> aG;
    AscendC::GlobalTensor<T> bG;
    AscendC::GlobalTensor<T> cG;
    aG.SetGlobalBuffer((__gm__ T *)pAddr + (bg * RS + m0) * Skv);
    bG.SetGlobalBuffer((__gm__ T *)vpAddr + bg * Skv * D);
    cG.SetGlobalBuffer((__gm__ T *)ypAddr + (bg * RS + m0) * D);

    mm.SetTensorA(aG, false);
    mm.SetTensorB(bG, false);
    mm.SetOrgShape(static_cast<int32_t>(mlen), static_cast<int32_t>(D), static_cast<int32_t>(Skv));
    mm.SetSingleShape(static_cast<int32_t>(mlen), static_cast<int32_t>(D), static_cast<int32_t>(Skv));
    mm.IterateAll(cG, 0);
    mm.End();
}

extern "C" void launch_gqa_pv_f16(GM_ADDR p, GM_ADDR vp, GM_ADDR yp, GM_ADDR sysWs, int64_t BG,
                                  int64_t RS, int64_t Skv, int64_t D, int64_t MT, int64_t nMT,
                                  GqaCubeTiling t, void *stream)
{
    gqa_pv_kernel<half><<<BG * nMT, nullptr, stream>>>(p, vp, yp, sysWs, BG, RS, Skv, D, MT, nMT, t);
}

extern "C" void launch_gqa_pv_bf16(GM_ADDR p, GM_ADDR vp, GM_ADDR yp, GM_ADDR sysWs, int64_t BG,
                                   int64_t RS, int64_t Skv, int64_t D, int64_t MT, int64_t nMT,
                                   GqaCubeTiling t, void *stream)
{
    gqa_pv_kernel<bfloat16_t><<<BG * nMT, nullptr, stream>>>(p, vp, yp, sysWs, BG, RS, Skv, D, MT,
                                                             nMT, t);
}
