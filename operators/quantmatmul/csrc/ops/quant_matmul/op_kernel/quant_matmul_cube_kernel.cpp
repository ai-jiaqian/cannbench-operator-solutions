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
 * \file quant_matmul_cube_kernel.cpp
 * \brief QuantMatmul CUBE stage (bisheng + -xasc, DAV_2201): C = x1 @ x2 in int32.
 *
 * This file contains ONLY the AIC (pure Cube) kernel plus the host-side tiling entry point.
 * The AIV dequant epilogue lives in quant_matmul_epilogue_kernel.cpp on purpose: the build
 * helper derives a kernel task type per translation unit, so mixing an AIC entry and an AIV
 * entry in one file silently mis-registers one of them.
 *
 * Math: every partial sum of the int8 contraction is an integer and the benchmark value range
 * keeps |C| far below 2^24, so int32 accumulation matches the fp64 golden reference.
 * Scaling / bias / offset / pertoken and the narrowing cast are done by the AIV epilogue.
 *
 * Work distribution: the Matmul library's own multi-core tiling (MultiCoreMatmulTiling with
 * SetDim = AIC count) splits the single GEMM over the available cube cores; the kernel simply
 * runs one Matmul object and IterateAll's its own tile into the int32 workspace.
 */

#define ASCENDC_CUBE_ONLY 1

#include <cstdint>
#include <cstring>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

// --- kernel-side Matmul high level API header (ascendc include tree) -----------------------------
#if defined(__has_include)
#  if __has_include("lib/matmul/matmul_intf.h")
#    define QM_MM_INTF_HEADER "lib/matmul/matmul_intf.h"
#  elif __has_include("adv_api/matmul/matmul_intf.h")
#    define QM_MM_INTF_HEADER "adv_api/matmul/matmul_intf.h"
#  elif __has_include("matmul_intf.h")
#    define QM_MM_INTF_HEADER "matmul_intf.h"
#  endif
#endif

#if defined(__has_include)
#  if __has_include("kernel_tiling/kernel_tiling.h")
#    include "kernel_tiling/kernel_tiling.h"
#  endif
#endif

#ifdef QM_MM_INTF_HEADER
#include QM_MM_INTF_HEADER
#endif

// --- host-side matmul tiling API ----------------------------------------------------------------
#if defined(__has_include)
#  if __has_include("adv_api/matmul/matmul_tiling.h")
#    define QM_MM_TILING_HEADER "adv_api/matmul/matmul_tiling.h"
#  elif __has_include("lib/matmul/matmul_tiling.h")
#    define QM_MM_TILING_HEADER "lib/matmul/matmul_tiling.h"
#  elif __has_include("matmul_tiling.h")
#    define QM_MM_TILING_HEADER "matmul_tiling.h"
#  endif
#endif

#ifdef QM_MM_TILING_HEADER
#include QM_MM_TILING_HEADER
#endif

// MultiCoreMatmulTiling lives in the multimatmul tiling header on some CANN layouts.
#if defined(__has_include)
#  if __has_include("adv_api/matmul/bmm_tiling.h")
#    include "adv_api/matmul/bmm_tiling.h"
#    define QM_HAS_MULTI_TILING 1
#  elif __has_include("lib/matmul/bmm_tiling.h")
#    include "lib/matmul/bmm_tiling.h"
#    define QM_HAS_MULTI_TILING 1
#  elif __has_include("bmm_tiling.h")
#    include "bmm_tiling.h"
#    define QM_HAS_MULTI_TILING 1
#  endif
#endif

#include "quant_matmul_launch.h"

namespace qm {

// Matmul template config: all shape fields stay at their runtime (-1) default and only the
// template selector is bound to the plain Norm config. Passing CFG_NORM directly as MM_CFG is
// rejected by some CANN builds because the constexpr-vs-runtime decision reads MM_CFG.usedCoreNum.
static constexpr MatmulApiStaticTiling QM_MM_CFG = []() {
    MatmulApiStaticTiling t{};
    t.cfg = CFG_NORM;
    return t;
}();

}  // namespace qm

using QmA = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>;
using QmB = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>;
using QmC = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;
using QmBias = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>;

/*!
 * \brief Pure-Cube int8 x int8 -> int32 kernel.
 *
 * The Matmul object is built for ONE single-core tile (usedCoreNum = 1); the work split over
 * blocks is done here explicitly from GetBlockIdx(), because the library's own multi-core
 * hand-off is not wired in a single-pass direct launch (all blocks would compute tile 0).
 * Ticket (blk) -> (batch, row block i, column block j); base pointers are clamped so that a full
 * scM x scN tile always stays inside the matrix (overlapping tiles recompute identical values).
 *
 * The whole batched problem is covered by ONE launch of batch*nr*nc blocks, so a 3D input costs a
 * single cube launch instead of one launch per batch.
 */
__global__ __aicore__ void qm_cube_i32_kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR cOut, GM_ADDR sysWs,
                                              qm::QmCubeArgs args)
{
    AscendC::TPipe pipe;
    AscendC::Matmul<QmA, QmB, QmC, QmBias, qm::QM_MM_CFG> mm;

    const AscendC::tiling::TCubeTiling* tiling =
        reinterpret_cast<const AscendC::tiling::TCubeTiling*>(args.tiling);

#if defined(REGIST_MATMUL_OBJ)
    REGIST_MATMUL_OBJ(&pipe, sysWs, mm, tiling);
#else
    AscendC::SetSysWorkspace(sysWs);
    mm.Init(tiling, &pipe);
#endif

    int64_t blk = AscendC::GetBlockIdx();
    if (blk >= args.numBlocks) {
        blk = 0;
    }
    const int64_t nc = args.nc > 0 ? args.nc : 1;
    const int64_t nr = args.nr > 0 ? args.nr : 1;
    const int64_t tiles = nr * nc;
    const int64_t b = blk / tiles;
    const int64_t tile = blk % tiles;
    const int64_t scM = args.scM > 0 ? args.scM : args.M;
    const int64_t scN = args.scN > 0 ? args.scN : args.N;

    int64_t rowOff = (tile / nc) * scM;
    if (rowOff + scM > args.M) {
        rowOff = args.M - scM;
    }
    if (rowOff < 0) {
        rowOff = 0;
    }
    int64_t colOff = (tile % nc) * scN;
    if (colOff + scN > args.N) {
        colOff = args.N - scN;
    }
    if (colOff < 0) {
        colOff = 0;
    }

    AscendC::GlobalTensor<int8_t> aGm;
    AscendC::GlobalTensor<int8_t> bGm;
    AscendC::GlobalTensor<int32_t> cGm;
    aGm.SetGlobalBuffer((__gm__ int8_t*)x1 + b * (args.M * args.K) + rowOff * args.K);
    bGm.SetGlobalBuffer((__gm__ int8_t*)x2 + b * (args.K * args.N) + colOff);
    cGm.SetGlobalBuffer((__gm__ int32_t*)cOut + b * (args.M * args.N) + rowOff * args.N + colOff);

    mm.SetTensorA(aGm, false);
    mm.SetTensorB(bGm, false);
    mm.IterateAll(cGm);
    mm.End();
}

// ===================================================================================================
// Host side tiling
// ===================================================================================================

static inline int64_t QmCeilDiv(int64_t a, int64_t b)
{
    return b > 0 ? (a + b - 1) / b : 0;
}

extern "C" int64_t qm_calc_cube_tiling(int64_t M, int64_t N, int64_t K, int64_t nAicHint,
                                       int32_t* tilingOut, int64_t* tilingInfo)
{
    for (int32_t i = 0; i < qm::QM_TILING_INTS; ++i) {
        tilingOut[i] = 0;
    }
    for (int32_t i = 0; i < 8; ++i) {
        if (tilingInfo != nullptr) {
            tilingInfo[i] = 0;
        }
    }

    int64_t aic = nAicHint;
    int64_t libWs = 0;
    int64_t aiv = 0;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        libWs = static_cast<int64_t>(plat->GetLibApiWorkSpaceSize());
        int64_t va = plat->GetCoreNumAiv();
        if (va > 0) {
            aiv = va;
        }
        if (aic <= 0) {
            int64_t v = plat->GetCoreNumAic();
            if (v > 0) {
                aic = v;
            }
        }
    }
    if (aic <= 0) {
        aic = 1;
    }
    if (aic > M * N) {
        aic = M * N;
    }

    // ---- pick the (scM, scN) work split ------------------------------------------------------
    // MEASURED (20 cases, 3 perf runs): the single dominant cost driver of the cube stage is
    // whether the per-core column extent `scN` handed to the Matmul tiling is an EXACT multiple of
    // the 256-wide base block. Every case whose singleCoreN was an exact multiple ran the cube at
    // 200-343 TOPS; every case with a partial trailing base block ran at 65-125 TOPS (12 of 12).
    // The fix is to choose scN = 256*k directly and then let the column block count follow,
    // nc = ceil(N/scN); the last column block is clamped by the kernel so no B column past N-1 is
    // ever read (the overlapped columns are recomputed with identical values).
    //
    // Cost model: the per-core padded work. One wave = nr*nc <= aic blocks run concurrently, so
    //   cost = paddedM * scN * ceil(nr*nc / aic).
    // Ties prefer fewer waves, then a larger paddedM (more base blocks in flight), then a narrower
    // scN. Blocks are capped at two waves: beyond that the per-block pipeline ramp dominates.
    // A paddedM of 128 (a single M base block) is refused whenever M can supply 256 rows.
    // For M < 256 the tile is forced to paddedM=128 and the op is pure memory streaming, so the
    // wave term is dropped and the narrowest scN (most parallel blocks) simply wins.
    const bool bigM = (M >= 256);
    int64_t bestNr = 1;
    int64_t bestNc = 1;
    int64_t bestPm = -1;
    int64_t bestScN = N;
    int64_t bestCost = -1;
    int64_t bestWaves = 0;
    for (int64_t k = 1;; ++k) {
        int64_t scN = 256 * k;
        if (scN > N) {
            if (k == 1) {
                scN = N;
            } else {
                break;
            }
        }
        const int64_t nc = QmCeilDiv(N, scN);
        for (int64_t nr = 1; nr <= aic; ++nr) {
            if (nr > M) {
                break;
            }
            const int64_t pm = QmCeilDiv(QmCeilDiv(M, nr), 128) * 128;
            if (bigM && pm < 256) {
                continue;
            }
            const int64_t blocks = nr * nc;
            int64_t waves = 1;
            if (bigM) {
                if (blocks * 2 < aic || blocks > 2 * aic) {
                    continue;
                }
                waves = QmCeilDiv(blocks, aic);
            }
            const int64_t cost = pm * scN * (bigM ? waves : 1);
            bool better = false;
            if (bestCost < 0 || cost < bestCost) {
                better = true;
            } else if (cost == bestCost) {
                if (waves < bestWaves) {
                    better = true;
                } else if (waves == bestWaves && pm > bestPm) {
                    better = true;
                } else if (waves == bestWaves && pm == bestPm && scN < bestScN) {
                    better = true;
                }
            }
            if (better) {
                bestCost = cost;
                bestWaves = waves;
                bestPm = pm;
                bestScN = scN;
                bestNr = nr;
                bestNc = nc;
            }
        }
        if (256 * k >= N) {
            break;
        }
    }

    const int64_t nr = bestNr;
    const int64_t nc = bestNc;
    const int64_t scM = QmCeilDiv(M, nr);
    const int64_t scN = bestScN;
    const int64_t numBlocks = nr * nc;

#if defined(QM_HAS_MULTI_TILING) && defined(QM_MM_TILING_HEADER)
    matmul_tiling::MultiCoreMatmulTiling mt(*plat);
    mt.SetDim(1);
    mt.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_INT8);
    mt.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_INT8);
    mt.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_INT32);
    mt.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   matmul_tiling::DataType::DT_INT32);
    mt.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    mt.SetSingleShape(static_cast<int32_t>(scM), static_cast<int32_t>(scN), static_cast<int32_t>(K));
    mt.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    mt.EnableBias(false);
    mt.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling cubeTiling;
    int64_t ret = mt.GetTiling(cubeTiling);
    if (ret == 0) {
        cubeTiling.SaveToBuffer(static_cast<void*>(tilingOut), qm::QM_TILING_BYTES);
    }
#endif

    if (tilingInfo != nullptr) {
        tilingInfo[0] = libWs;
        tilingInfo[1] = aic;
        tilingInfo[2] = aiv;
        tilingInfo[3] = numBlocks;
        tilingInfo[4] = nr;
        tilingInfo[5] = nc;
        tilingInfo[6] = scM;
        tilingInfo[7] = scN;
    }
    return numBlocks;
}

// ===================================================================================================
// Launch wrappers (plain C, called from the g++ plugin)
// ===================================================================================================

extern "C" void qm_launch_cube_i32(GM_ADDR x1, GM_ADDR x2, GM_ADDR cOut, GM_ADDR sysWs,
                                   const qm::QmCubeArgs* args, void* stream)
{
    qm_cube_i32_kernel<<<static_cast<uint32_t>(args->numBlocks), nullptr, stream>>>(
        x1, x2, cOut, sysWs, *args);
}
