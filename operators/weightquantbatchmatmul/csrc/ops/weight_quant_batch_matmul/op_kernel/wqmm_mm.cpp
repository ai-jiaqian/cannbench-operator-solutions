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
 * \file wqmm_mm.cpp
 * \brief WeightQuantBatchMatmul: pure-cube (ASCENDC_CUBE_ONLY) Matmul over the dequantised weight.
 *
 *   y[m0:m1, n0:n1] = x[m0:m1, :] @ wd[:, n0:n1] + bias[n0:n1]
 *
 * The host tiling describes a single (singleCoreM x singleCoreN x K) block; the kernel owns the
 * per-core decomposition, exactly as the Matmul advanced-API programming guide prescribes:
 *   mCoreIndex = blockIdx % mBlocks ,  nCoreIndex = blockIdx / mBlocks
 * and the A / B / C operands are offset accordingly.  Tail blocks go through SetSingleShape.
 *
 * The C matrix is produced directly in the output dtype with the bias folded in by the fixpipe, so the
 * operator needs no vector epilogue: for fp16 the pair is (A=half, B=half, C=half, bias=half) and for
 * bfloat16 it is (A=bf16, B=bf16, C=bf16, bias=float), both of which are supported on DAV_2201.
 */

#define ASCENDC_CUBE_ONLY

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul/matmul_intf.h"
#include "wqmm_launch.h"

namespace {

/*!
 * \brief MDL template for the cube stage.
 *
 * The host tiling is produced by MatmulApiTiling with the default config type (MDL), so the kernel must
 * instantiate the same template or the engine's depth/step fields are not honoured.  MDL is the template
 * meant for large shapes: MTE2 moves one big packet of several basic blocks from GM to L1 instead of one
 * packet per basic block.
 *
 * enableKdimReorderLoad is the counter-measure for the access pattern of this operator: every cube core
 * reads the *same* rows of A (x is shared) while streaming its own B slab, so all cores would otherwise
 * hit the same GM addresses in lockstep.  The API card recommends it for large K with neither operand
 * fully loaded, which is exactly this case.
 */
constexpr MatmulConfig WQMM_MM_CFG = GetMDLConfig(
    /* intrinsicsLimit      */ false,
    /* batchLoop            */ false,
    /* doMTE2Preload        */ 0,
    /* isVecND2NZ           */ false,
    /* isPerTensor          */ false,
    /* hasAntiQuantOffset   */ false,
    /* enUnitFlag           */ false,
    /* isMsgReuse           */ true,
    /* enableUBReuse        */ true,
    /* enableL1CacheUB      */ false,
    /* enableMixDualMaster  */ false,
    /* enableKdimReorderLoad*/ true);

template <typename T, typename TBias>
__global__ __aicore__ void wqmm_mm_kernel(GM_ADDR xPtr, GM_ADDR wdPtr, GM_ADDR bPtr, GM_ADDR yPtr,
                                          GM_ADDR wsPtr, WqmmTilingPOD tp)
{
    using namespace AscendC;
    if ASCEND_IS_AIV {
        return;             /* pure cube mode: the vector cores have nothing to do */
    }

    const int64_t M = static_cast<int64_t>(tp.M);
    const int64_t N = static_cast<int64_t>(tp.N);
    const int64_t K = static_cast<int64_t>(tp.K);
    const int64_t scM = static_cast<int64_t>(tp.singleCoreM);
    const int64_t scN = static_cast<int64_t>(tp.singleCoreN);
    const int64_t mBlocks = static_cast<int64_t>(tp.mBlocks);
    if (M <= 0 || N <= 0 || K <= 0 || scM <= 0 || scN <= 0 || mBlocks <= 0) {
        return;
    }
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    if (blk >= mBlocks * static_cast<int64_t>(tp.nBlocks)) {
        return;
    }

    const int64_t mIdx = blk % mBlocks;
    const int64_t nIdx = blk / mBlocks;
    const int64_t m0 = mIdx * scM;
    const int64_t n0 = nIdx * scN;
    int64_t tailM = M - m0;
    if (tailM > scM) {
        tailM = scM;
    }
    int64_t tailN = N - n0;
    if (tailN > scN) {
        tailN = scN;
    }
    if (tailM <= 0 || tailN <= 0) {
        return;
    }

    AscendC::tiling::TCubeTiling cubeLocal;
    {
        const int32_t *src = tp.cube;
        int32_t *dst = reinterpret_cast<int32_t *>(&cubeLocal);
        const int32_t words = static_cast<int32_t>(sizeof(AscendC::tiling::TCubeTiling) / sizeof(int32_t));
        for (int32_t i = 0; i < words; ++i) {
            dst[i] = src[i];
        }
    }

    SetSysWorkspace(reinterpret_cast<__gm__ uint8_t *>(wsPtr));

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, TBias>;
    Matmul<aType, bType, cType, biasType, WQMM_MM_CFG> mm;

    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &cubeLocal);

    GlobalTensor<T> xGm, wdGm, yGm;
    GlobalTensor<TBias> bGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xPtr));
    wdGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(wdPtr));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(yPtr));
    if (tp.hasBias != 0) {
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ TBias *>(bPtr));
    }

    mm.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K),
                   static_cast<int32_t>(K));
    if (tailM != scM || tailN != scN) {
        mm.SetSingleShape(static_cast<int32_t>(tailM), static_cast<int32_t>(tailN), static_cast<int32_t>(K));
    }
    mm.SetTensorA(xGm[m0 * K], false);
    mm.SetTensorB(wdGm[n0], false);
    if (tp.hasBias != 0) {
        mm.SetBias(bGm[n0]);
    }
    mm.IterateAll(yGm[m0 * N + n0]);
    mm.End();
}

} // namespace

extern "C" {

void launch_wqmm_mm_half(GM_ADDR x, GM_ADDR wd, GM_ADDR bias, GM_ADDR y, GM_ADDR ws,
                         WqmmTilingPOD tp, int64_t numBlocks, void *stream)
{
    if (numBlocks <= 0) {
        return;
    }
    wqmm_mm_kernel<half, half><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(x, wd, bias, y, ws, tp);
}

void launch_wqmm_mm_bf16(GM_ADDR x, GM_ADDR wd, GM_ADDR bias, GM_ADDR y, GM_ADDR ws,
                         WqmmTilingPOD tp, int64_t numBlocks, void *stream)
{
    if (numBlocks <= 0) {
        return;
    }
    wqmm_mm_kernel<bfloat16_t, float><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(x, wd, bias, y, ws, tp);
}

} // extern "C"
