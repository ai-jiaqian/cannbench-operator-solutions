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
 * \file mla_prolog_mm.cpp
 * \brief MlaProlog cube stages (bisheng + -xasc). Pure-cube (AIC) kernels built on the high-level Matmul API
 *        with a serialized kernel-side TCubeTiling handed in through the launch ABI.
 *
 * Every launched block computes exactly ONE self-contained block sized [singleM, singleN] and derives its own
 * A/B/C base pointers from GetBlockIdx() (the tiling describes one block, usedCoreNum is 1).  The row strides
 * come from the tiling's M/Ka/N fields via SetOrgShape.
 */

#define ASCENDC_CUBE_ONLY

#include <cstdint>
#include "kernel_operator.h"
#include "mla_prolog_launch.h"

#if defined(__has_include)
#  if __has_include("lib/matmul_intf.h")
#    include "lib/matmul_intf.h"
#    define MLA_MM_HDR 1
#  elif __has_include("lib/matmul/matmul_intf.h")
#    include "lib/matmul/matmul_intf.h"
#    define MLA_MM_HDR 1
#  elif __has_include("adv_api/matmul/matmul_intf.h")
#    include "adv_api/matmul/matmul_intf.h"
#    define MLA_MM_HDR 1
#  elif __has_include("matmul_intf.h")
#    include "matmul_intf.h"
#    define MLA_MM_HDR 1
#  endif
#endif

#if !defined(MLA_MM_HDR)
#error "mla_prolog: no matmul_intf.h found in the kernel include path"
#endif

using namespace AscendC;

namespace mlac {

using TA = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using TCF = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using TCB = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using TBI = MatmulType<TPosition::GM, CubeFormat::ND, float>;

/*! \brief the kernel-side tiling arrives as raw bytes from the host pass; the two sides share the type. */
__aicore__ inline void PodToTiling(const MlaTiling& p, tiling::TCubeTiling& t)
{
    uint8_t* d = reinterpret_cast<uint8_t*>(&t);
    for (uint32_t i = 0; i < (uint32_t)sizeof(tiling::TCubeTiling); ++i) {
        d[i] = p.v[i];
    }
}

}  // namespace mlac

// ---------------------------------------------------------------------------
// cq[M,Hcq]   = x[M,He] @ w_dq[He,Hcq]      (fp32)
// dkv[M,HkvDr]= x[M,He] @ w_dkv_kr[He,HkvDr](fp32)
// Blocks [0, nb1) serve the first matmul, blocks [nb1, nb1+nb2) the second.
// ---------------------------------------------------------------------------
__global__ __aicore__ void mla_mm_pair_kernel(GM_ADDR xPtr, GM_ADDR wdqPtr, GM_ADDR cqPtr,
                                              GM_ADDR wdkvPtr, GM_ADDR dkvPtr, GM_ADDR wsPtr,
                                              MlaTiling pod1, MlaTiling pod2,
                                              int64_t sN1, int64_t nb1, int64_t sN2)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int64_t blk = (int64_t)GetBlockIdx();
    if (blk >= nb1) {
        // second matmul band
        const int64_t n0 = (blk - nb1) * sN2;
        tiling::TCubeTiling t2;
        mlac::PodToTiling(pod2, t2);
        TPipe pipe;
        Matmul<mlac::TA, mlac::TA, mlac::TCF, mlac::TBI> mm;
        REGIST_MATMUL_OBJ(&pipe, (__gm__ uint8_t*)wsPtr, mm, &t2);
        GlobalTensor<bfloat16_t> gA;
        gA.SetGlobalBuffer((__gm__ bfloat16_t*)xPtr);
        GlobalTensor<bfloat16_t> gB;
        gB.SetGlobalBuffer((__gm__ bfloat16_t*)wdkvPtr + n0);
        GlobalTensor<float> gC;
        gC.SetGlobalBuffer((__gm__ float*)dkvPtr + n0);
        mm.SetTensorA(gA);
        mm.SetTensorB(gB);
        mm.IterateAll(gC);
        mm.End();
        return;
    }

    const int64_t n0 = blk * sN1;
    tiling::TCubeTiling t1;
    mlac::PodToTiling(pod1, t1);
    TPipe pipe;
    Matmul<mlac::TA, mlac::TA, mlac::TCF, mlac::TBI> mm;
    REGIST_MATMUL_OBJ(&pipe, (__gm__ uint8_t*)wsPtr, mm, &t1);
    GlobalTensor<bfloat16_t> gA;
    gA.SetGlobalBuffer((__gm__ bfloat16_t*)xPtr);
    GlobalTensor<bfloat16_t> gB;
    gB.SetGlobalBuffer((__gm__ bfloat16_t*)wdqPtr + n0);
    GlobalTensor<float> gC;
    gC.SetGlobalBuffer((__gm__ float*)cqPtr + n0);
    mm.SetTensorA(gA);
    mm.SetTensorB(gB);
    mm.IterateAll(gC);
    mm.End();
}

// ---------------------------------------------------------------------------
// qr2[3M, NW]f32 = csplit[3M, Hcq] @ w_uq_qr[Hcq, NW]   (N split across blocks)
// ---------------------------------------------------------------------------
__global__ __aicore__ void mla_mm_big_kernel(GM_ADDR aPtr, GM_ADDR bPtr, GM_ADDR cPtr, GM_ADDR wsPtr,
                                             MlaTiling pod, int64_t sN)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int64_t blk = (int64_t)GetBlockIdx();
    tiling::TCubeTiling t;
    mlac::PodToTiling(pod, t);
    TPipe pipe;
    Matmul<mlac::TA, mlac::TA, mlac::TCF, mlac::TBI> mm;
    REGIST_MATMUL_OBJ(&pipe, (__gm__ uint8_t*)wsPtr, mm, &t);
    const int64_t n0 = blk * sN;
    GlobalTensor<bfloat16_t> gA;
    gA.SetGlobalBuffer((__gm__ bfloat16_t*)aPtr);
    GlobalTensor<bfloat16_t> gB;
    gB.SetGlobalBuffer((__gm__ bfloat16_t*)bPtr + n0);
    GlobalTensor<float> gC;
    gC.SetGlobalBuffer((__gm__ float*)cPtr + n0);
    mm.SetTensorA(gA);
    mm.SetTensorB(gB);
    mm.IterateAll(gC);
    mm.End();
}

// ---------------------------------------------------------------------------
// Per-head absorption:  qtmp3[N][3M,Hckv]f32 = qcs[N][3M,D] @ wuk[N][D,Hckv]
//
// The three bf16 terms of q_c are stacked along M (NOT along K), so the w_uk operand is consumed
// untripled at K = D: no w_uk duplication buffer, no duplication pass and no 3x re-read of the
// weight.  The price is an fp32 C with 3M rows, which is folded back to bf16 by mla_v_foldq_kernel.
// One block per head; B and C both use row stride Hckv (see mla_make_tiling call site).
// ---------------------------------------------------------------------------
__global__ __aicore__ void mla_mm_qk_kernel(GM_ADDR aPtr, GM_ADDR bPtr, GM_ADDR cPtr, GM_ADDR wsPtr,
                                            MlaTiling pod, int64_t M3, int64_t D, int64_t Hckv)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int64_t n = (int64_t)GetBlockIdx();
    tiling::TCubeTiling t;
    mlac::PodToTiling(pod, t);
    TPipe pipe;
    Matmul<mlac::TA, mlac::TA, mlac::TCF, mlac::TBI> mm;
    REGIST_MATMUL_OBJ(&pipe, (__gm__ uint8_t*)wsPtr, mm, &t);
    GlobalTensor<bfloat16_t> gA;
    gA.SetGlobalBuffer((__gm__ bfloat16_t*)aPtr + n * M3 * D);
    GlobalTensor<bfloat16_t> gB;
    gB.SetGlobalBuffer((__gm__ bfloat16_t*)bPtr + n * D * Hckv);
    GlobalTensor<float> gC;
    gC.SetGlobalBuffer((__gm__ float*)cPtr + n * M3 * Hckv);
    mm.SetTensorA(gA);
    mm.SetTensorB(gB);
    mm.IterateAll(gC);
    mm.End();
}

// ---------------------------------------------------------------------------
// launch wrappers (plain C so the g++ plugin can call them)
// ---------------------------------------------------------------------------
extern "C" {

void launch_mla_mm_pair(GM_ADDR x, GM_ADDR wdq, GM_ADDR cq, GM_ADDR wdkv, GM_ADDR dkv, GM_ADDR ws,
                        const MlaTiling* t1, const MlaTiling* t2,
                        int64_t M, int64_t He, int64_t Hcq, int64_t HkvDr,
                        int64_t sN1, int64_t nb1, int64_t sN2, void* stream)
{
    (void)M;
    (void)He;
    (void)Hcq;
    (void)HkvDr;
    const int64_t nb2 = (HkvDr + sN2 - 1) / sN2;
    const int64_t total = nb1 + nb2;
    mla_mm_pair_kernel<<<total, nullptr, stream>>>((GM_ADDR)x, (GM_ADDR)wdq, (GM_ADDR)cq,
                                                   (GM_ADDR)wdkv, (GM_ADDR)dkv, (GM_ADDR)ws,
                                                   *t1, *t2, sN1, nb1, sN2);
}

void launch_mla_mm_big(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const MlaTiling* t,
                       int64_t sN, int64_t nb, void* stream)
{
    mla_mm_big_kernel<<<nb, nullptr, stream>>>((GM_ADDR)a, (GM_ADDR)b, (GM_ADDR)c, (GM_ADDR)ws,
                                               *t, sN);
}

void launch_mla_mm_qk(GM_ADDR qcs, GM_ADDR wuk, GM_ADDR qtmp3, GM_ADDR ws, const MlaTiling* t,
                      int64_t M3, int64_t D, int64_t Hckv, int64_t nHeads, void* stream)
{
    mla_mm_qk_kernel<<<nHeads, nullptr, stream>>>((GM_ADDR)qcs, (GM_ADDR)wuk, (GM_ADDR)qtmp3,
                                                 (GM_ADDR)ws, *t, M3, D, Hckv);
}

}  // extern "C"
