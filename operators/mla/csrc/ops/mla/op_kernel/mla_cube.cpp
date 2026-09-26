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
 * \file mla_cube.cpp
 * \brief MLA cube stages (bisheng + -xasc): the QK and the PV matmuls.
 *
 * This translation unit is pure-cube (AIC only): ASCENDC_CUBE_ONLY is defined before the
 * Matmul interface header is included.
 *
 *   K1  s[totalRows, Skv] = qa[totalRows, Dc] * kb[Skv, Dc]^T   (float32)
 *   K3  y[totalRows, Dn]  = p[totalRows, Skv] * v[Skv, Dn]      (input dtype)
 *
 * Parallelism comes from the grid: every launched block walks a contiguous set of row
 * bands of RQ query rows each, and for each band performs one complete self-contained
 * single-core [M, N, K] matmul (the band is selected with explicit A/C pointer offsets).
 * The Matmul object therefore never needs cross-core state, and the tiling handed in
 * always describes exactly one complete single-core matmul: usedCoreNum = 1,
 * singleCoreM = M, singleCoreN = N, singleCoreK = K.
 */

// Pure-cube translation unit: the whole file is AIC code.
#define ASCENDC_CUBE_ONLY

#include <cstdint>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#if defined(__has_include)
#  if __has_include("lib/matmul/matmul_intf.h")
#    include "lib/matmul/matmul_intf.h"
#    define MLA_MM_INTF_OK 1
#  elif __has_include("matmul/matmul_intf.h")
#    include "matmul/matmul_intf.h"
#    define MLA_MM_INTF_OK 1
#  elif __has_include("adv_api/matmul/matmul_intf.h")
#    include "adv_api/matmul/matmul_intf.h"
#    define MLA_MM_INTF_OK 1
#  elif __has_include("matmul_intf.h")
#    include "matmul_intf.h"
#    define MLA_MM_INTF_OK 1
#  endif
#endif

#ifndef MLA_MM_INTF_OK
#error "mla: no matmul_intf.h found in the kernel include path"
#endif

#if defined(__has_include)
#  if __has_include("kernel_tiling/kernel_tiling.h")
#    include "kernel_tiling/kernel_tiling.h"
#  endif
#endif

#include "mla_launch.h"

namespace mlac {

/* CubeFormat is a global enum on this toolchain: never qualify it as AscendC::CubeFormat. */
template <typename T>
using AType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
template <typename T>
using BTrType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, true>;
template <typename T>
using BNormType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, false>;
template <typename T>
using CF32Type = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
template <typename T>
using CValType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
using BiasT = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

/*! \brief rebuild the kernel-side TCubeTiling from the POD.
 *
 *  The struct is fully zeroed first so that no word is left holding stack garbage; only
 *  the documented fields carry values, the remaining reserved words stay 0. */
__aicore__ inline void PodToTiling(const MlaTiling& p, AscendC::tiling::TCubeTiling& t)
{
    int32_t* w = (int32_t*)&t;
    const uint32_t n = (uint32_t)(sizeof(AscendC::tiling::TCubeTiling) / sizeof(int32_t));
    for (uint32_t i = 0; i < n; ++i) {
        w[i] = 0;
    }
    t.usedCoreNum = p.usedCoreNum;
    t.M = p.M;
    t.N = p.N;
    t.Ka = p.Ka;
    t.Kb = p.Kb;
    t.singleCoreM = p.singleCoreM;
    t.singleCoreN = p.singleCoreN;
    t.singleCoreK = p.singleCoreK;
    t.baseM = p.baseM;
    t.baseN = p.baseN;
    t.baseK = p.baseK;
    t.depthA1 = p.depthA1;
    t.depthB1 = p.depthB1;
    t.stepM = p.stepM;
    t.stepN = p.stepN;
    t.stepKa = p.stepKa;
    t.stepKb = p.stepKb;
    t.isBias = p.isBias;
    t.transLength = p.transLength;
    t.iterateOrder = p.iterateOrder;
    t.dbL0A = p.dbL0A;
    t.dbL0B = p.dbL0B;
    t.dbL0C = p.dbL0C;
    t.shareMode = p.shareMode;
    t.shareL1Size = p.shareL1Size;
    t.shareL0CSize = p.shareL0CSize;
    t.shareUbSize = p.shareUbSize;
    t.batchM = p.batchM;
    t.batchN = p.batchN;
    t.singleBatchM = p.singleBatchM;
    t.singleBatchN = p.singleBatchN;
    t.mxTypePara = p.mxTypePara;
}

/*! \brief reject a tiling that could send the Matmul addressing loops out of range. */
__aicore__ inline bool TilingSane(const AscendC::tiling::TCubeTiling& t)
{
    return t.usedCoreNum > 0 && t.M > 0 && t.N > 0 && t.Ka > 0 && t.Kb > 0 &&
           t.singleCoreM > 0 && t.singleCoreN > 0 && t.singleCoreK > 0 && t.baseM > 0 &&
           t.baseN > 0 && t.baseK > 0;
}

/*! \brief pin the tiling so it describes exactly ONE complete single-core [M, N, K] matmul.
 *
 *  Each launched block computes a full [M, N] matmul on its own row band, so the Matmul must
 *  never decompose M / N across cores: if it did, every block would compute (and write) some
 *  other block's sub-block, which shows up as unwritten rows and wild, shape-dependent error
 *  magnitudes.  The tiling is therefore pinned rather than merely defaulted. */
__aicore__ inline void SanitizeTiling(AscendC::tiling::TCubeTiling& t)
{
    t.usedCoreNum = 1;
    t.singleCoreM = t.M;
    t.singleCoreN = t.N;
    t.singleCoreK = t.Ka;
    if (t.iterateOrder != 0 && t.iterateOrder != 1) {
        t.iterateOrder = 0;
    }
    if (t.isBias != 0 && t.isBias != 1) {
        t.isBias = 0;
    }
    if (t.stepM <= 0) {
        t.stepM = 1;
    }
    if (t.stepN <= 0) {
        t.stepN = 1;
    }
    if (t.stepKa <= 0) {
        t.stepKa = 1;
    }
    if (t.stepKb <= 0) {
        t.stepKb = 1;
    }
    if (t.depthA1 <= 0) {
        t.depthA1 = t.stepM * t.stepKa;
    }
    if (t.depthB1 <= 0) {
        t.depthB1 = t.stepN * t.stepKb;
    }
    if (t.dbL0A <= 0) {
        t.dbL0A = 1;
    }
    if (t.dbL0B <= 0) {
        t.dbL0B = 1;
    }
    if (t.dbL0C <= 0) {
        t.dbL0C = 1;
    }
    if (t.baseM <= 0) {
        t.baseM = 16;
    }
    if (t.baseN <= 0) {
        t.baseN = 16;
    }
    if (t.baseK <= 0) {
        t.baseK = 16;
    }
    if (t.baseM > t.singleCoreM) {
        t.baseM = t.singleCoreM;
    }
    if (t.baseN > t.singleCoreN) {
        t.baseN = t.singleCoreN;
    }
    if (t.baseK > t.singleCoreK) {
        t.baseK = t.singleCoreK;
    }
}

} // namespace mlac

// =========================================================================
// K1:  s[band rows, 0..Skv) = qa[band rows, 0..Dc) * kb[0..Skv, 0..Dc)^T
// =========================================================================
template <typename T>
__global__ __aicore__ void mla_qk_kernel(GM_ADDR qaPtr, GM_ADDR kbPtr, GM_ADDR sPtr,
                                         GM_ADDR wsPtr, MlaTiling pod, int64_t B, int64_t R,
                                         int64_t Skv, int64_t Dc, int64_t RQ, int64_t perWs,
                                         int64_t nBlocks, int64_t nBandsTot, int64_t enAtomic)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using namespace AscendC;

    AscendC::tiling::TCubeTiling t;
    mlac::PodToTiling(pod, t);
    if (!mlac::TilingSane(t)) {
        return;
    }
    mlac::SanitizeTiling(t);

    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    // The Matmul is handed its system workspace explicitly: this is a standalone direct
    // launch, so no framework has bootstrapped the global system workspace, and with the
    // tiling pinned to usedCoreNum == 1 the library's internal workspace offset is constant.
    // Every hardware block therefore receives its own valid slice of the workspace buffer.
    GM_ADDR wsSlice = (GM_ADDR)((__gm__ uint8_t*)wsPtr + blk * perWs);

    int64_t per = (nBandsTot + nBlocks - 1) / nBlocks;
    if (per < 1) {
        per = 1;
    }
    int64_t g0 = blk * per;
    int64_t g1 = g0 + per;
    if (g1 > nBandsTot) {
        g1 = nBandsTot;
    }
    if (g0 >= g1) {
        return;
    }

    TPipe pipe;
    Matmul<mlac::AType<T>, mlac::BTrType<T>, mlac::CF32Type<T>, mlac::BiasT> mm;
    REGIST_MATMUL_OBJ(&pipe, wsSlice, mm, &t);

    for (int64_t g = g0; g < g1; ++g) {
        const int64_t row0 = g * RQ;
        const int64_t b = row0 / R;
        const int64_t r0 = row0 - b * R;

        GlobalTensor<T> aGm;
        aGm.SetGlobalBuffer((__gm__ T*)qaPtr + (b * R + r0) * Dc);
        GlobalTensor<T> bGm;
        bGm.SetGlobalBuffer((__gm__ T*)kbPtr + b * Skv * Dc);
        GlobalTensor<float> cGm;
        cGm.SetGlobalBuffer((__gm__ float*)sPtr + row0 * Skv);

        mm.SetTensorA(aGm, false);
        mm.SetTensorB(bGm, true);
        mm.IterateAll(cGm, (uint8_t)(enAtomic != 0 ? 1 : 0));
    }
    mm.End();
}

// =========================================================================
// K3:  y[band rows, 0..Dn) = p[band rows, 0..Skv) * v[0..Skv, 0..Dn)
// =========================================================================
template <typename T>
__global__ __aicore__ void mla_pv_kernel(GM_ADDR pPtr, GM_ADDR vPtr, GM_ADDR yPtr, GM_ADDR wsPtr,
                                         MlaTiling pod, int64_t B, int64_t R, int64_t Skv,
                                         int64_t Dn, int64_t RQ, int64_t perWs, int64_t nBlocks,
                                         int64_t nBandsTot)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using namespace AscendC;

    AscendC::tiling::TCubeTiling t;
    mlac::PodToTiling(pod, t);
    if (!mlac::TilingSane(t)) {
        return;
    }
    mlac::SanitizeTiling(t);

    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    GM_ADDR wsSlice = (GM_ADDR)((__gm__ uint8_t*)wsPtr + blk * perWs);

    int64_t per = (nBandsTot + nBlocks - 1) / nBlocks;
    if (per < 1) {
        per = 1;
    }
    int64_t g0 = blk * per;
    int64_t g1 = g0 + per;
    if (g1 > nBandsTot) {
        g1 = nBandsTot;
    }
    if (g0 >= g1) {
        return;
    }

    TPipe pipe;
    Matmul<mlac::AType<T>, mlac::BNormType<T>, mlac::CValType<T>, mlac::BiasT> mm;
    REGIST_MATMUL_OBJ(&pipe, wsSlice, mm, &t);

    for (int64_t g = g0; g < g1; ++g) {
        const int64_t row0 = g * RQ;
        const int64_t b = row0 / R;
        const int64_t r0 = row0 - b * R;

        GlobalTensor<T> aGm;
        aGm.SetGlobalBuffer((__gm__ T*)pPtr + row0 * Skv);
        GlobalTensor<T> bGm;
        bGm.SetGlobalBuffer((__gm__ T*)vPtr + b * Skv * Dn);
        GlobalTensor<T> cGm;
        cGm.SetGlobalBuffer((__gm__ T*)yPtr + row0 * Dn);

        mm.SetTensorA(aGm, false);
        mm.SetTensorB(bGm, false);
        mm.IterateAll(cGm, (uint8_t)0);
    }
    mm.End();
}

extern "C" {

void launch_mla_qk_half(GM_ADDR qa, GM_ADDR kb, GM_ADDR s, GM_ADDR ws, MlaTiling t, int64_t B,
                        int64_t R, int64_t Skv, int64_t Dc, int64_t RQ, int64_t perWs,
                        int64_t nBlocks, int64_t nBandsTot, int64_t enAtomic, void* stream)
{
    mla_qk_kernel<half><<<nBlocks, nullptr, stream>>>(qa, kb, s, ws, t, B, R, Skv, Dc, RQ, perWs,
                                                      nBlocks, nBandsTot, enAtomic);
}

void launch_mla_qk_bfloat16(GM_ADDR qa, GM_ADDR kb, GM_ADDR s, GM_ADDR ws, MlaTiling t, int64_t B,
                            int64_t R, int64_t Skv, int64_t Dc, int64_t RQ, int64_t perWs,
                            int64_t nBlocks, int64_t nBandsTot, int64_t enAtomic, void* stream)
{
    mla_qk_kernel<bfloat16_t><<<nBlocks, nullptr, stream>>>(qa, kb, s, ws, t, B, R, Skv, Dc, RQ,
                                                            perWs, nBlocks, nBandsTot, enAtomic);
}

void launch_mla_pv_half(GM_ADDR p, GM_ADDR v, GM_ADDR y, GM_ADDR ws, MlaTiling t, int64_t B,
                        int64_t R, int64_t Skv, int64_t Dn, int64_t RQ, int64_t perWs,
                        int64_t nBlocks, int64_t nBandsTot, void* stream)
{
    mla_pv_kernel<half><<<nBlocks, nullptr, stream>>>(p, v, y, ws, t, B, R, Skv, Dn, RQ, perWs,
                                                      nBlocks, nBandsTot);
}

void launch_mla_pv_bfloat16(GM_ADDR p, GM_ADDR v, GM_ADDR y, GM_ADDR ws, MlaTiling t, int64_t B,
                            int64_t R, int64_t Skv, int64_t Dn, int64_t RQ, int64_t perWs,
                            int64_t nBlocks, int64_t nBandsTot, void* stream)
{
    mla_pv_kernel<bfloat16_t><<<nBlocks, nullptr, stream>>>(p, v, y, ws, t, B, R, Skv, Dn, RQ,
                                                            perWs, nBlocks, nBandsTot);
}

} // extern "C"
