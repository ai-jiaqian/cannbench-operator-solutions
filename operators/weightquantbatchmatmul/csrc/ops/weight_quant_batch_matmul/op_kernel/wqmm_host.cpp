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
 * \file wqmm_host.cpp
 * \brief WeightQuantBatchMatmul host-side tiling (compiled by bisheng, linked into the plugin).
 *
 * Produces
 *   - a single-block Matmul tiling for one (singleCoreM x singleCoreN x K) slab,
 *   - the (mBlocks x nBlocks) cube-core decomposition the kernel uses for its operand offsets,
 *   - the AIV dequant launch geometry.
 *
 * The tiling is generated for a *single* slab whose N width is ceil(N / numAic) rounded up to 16, so a
 * one-dimensional N split over the cube cores fills the array; the kernel owns the block offsets
 * (mCoreIndex = blockIdx % mBlocks, nCoreIndex = blockIdx / mBlocks), the convention used by the Matmul
 * advanced-API programming guide.  singleCoreM/singleCoreN are read back from the engine's tiling and used
 * as authoritative, so nothing is assumed about the engine's choice; only usedCoreNum is rewritten so the
 * serialised blob describes the grid that is actually launched.
 *
 * The tiling struct is AscendC::tiling::TCubeTiling, the kernel-side plain-C++ TilingData, so the host and
 * the device see literally the same type and layout.
 */

#include <algorithm>
#include <cstring>
#include "platform/platform_ascendc.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul/matmul_tiling.h"
#include "wqmm_launch.h"

namespace {

constexpr int64_t WQMM_CUBE_ALIGN = 16;
constexpr int64_t WQMM_DQ_COL_TILE = 2048;
/* Minimum column slab a single cube core is given, in elements (512 byte per row of the weight stream),
   and the minimum number of cube cores kept busy. */
constexpr int64_t WQMM_MIN_SLAB = 256;
constexpr int64_t WQMM_MIN_BLOCKS = 4;

inline int64_t WqmmAlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

inline int64_t WqmmCeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

} // namespace

int64_t calc_wqmm_plan(int64_t M, int64_t N, int64_t K, int64_t isBf16, int64_t hasBias,
                       int64_t numAic, WqmmTilingPOD *out,
                       int64_t *dqCores, int64_t *dqRowsPerCore, int64_t *dqColTile,
                       int64_t *sysWsBytes)
{
    if (out == nullptr) {
        return -1;
    }
    std::memset(out, 0, sizeof(*out));
    out->M = static_cast<int32_t>(M);
    out->N = static_cast<int32_t>(N);
    out->K = static_cast<int32_t>(K);
    out->hasBias = hasBias != 0 ? 1 : 0;
    out->isBf16 = isBf16 != 0 ? 1 : 0;
    out->diag[3] = static_cast<int32_t>(sizeof(AscendC::tiling::TCubeTiling));
    out->diag[4] = static_cast<int32_t>(sizeof(WqmmTilingPOD));
    if (M <= 0 || N <= 0 || K <= 0) {
        out->diag[0] = -1;
        return -1;
    }

    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        out->diag[0] = -2;
        return -1;
    }

    int64_t aic = (numAic > 0) ? numAic : static_cast<int64_t>(plat->GetCoreNumAic());
    if (aic <= 0) {
        aic = 1;
    }
    if (aic > N) {
        aic = N;
    }
    out->diag[1] = static_cast<int32_t>(aic);

    const matmul_tiling::DataType dt =
        isBf16 ? matmul_tiling::DataType::DT_BFLOAT16 : matmul_tiling::DataType::DT_FLOAT16;
    const matmul_tiling::DataType dtBias =
        isBf16 ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;

    int64_t nSlab = WqmmAlignUp(WqmmCeilDiv(N, aic), WQMM_CUBE_ALIGN);
    if (nSlab < WQMM_CUBE_ALIGN) {
        nSlab = WQMM_CUBE_ALIGN;
    }
    /* Each cube core streams its own column slab of the dequantised weight with a row stride of the full
       N.  A slab narrower than ~512 byte per row makes that stream a short strided burst per row, and the
       measured cost per weight element grows sharply with shrinking slab width (the 20 cases sort almost
       monotonically with it).  Widen the slab to at least WQMM_MIN_SLAB elements, while keeping at least
       WQMM_MIN_BLOCKS cube cores busy. */
    {
        int64_t minSlab = WQMM_MIN_SLAB;
        const int64_t floorSlab = WqmmAlignUp(WqmmCeilDiv(N, WQMM_MIN_BLOCKS), WQMM_CUBE_ALIGN);
        if (minSlab > floorSlab) {
            minSlab = floorSlab;
        }
        if (nSlab < minSlab) {
            nSlab = minSlab;
        }
    }
    if (nSlab > N) {
        nSlab = N;
    }

    AscendC::tiling::TCubeTiling cube;
    int64_t scM = M;
    int64_t scN = nSlab;
    int64_t mBlocks = 1;
    int64_t nBlocks = WqmmCeilDiv(N, nSlab);
    bool ok = false;
    int32_t attempts = 0;
    for (int attempt = 0; attempt < 6; ++attempt) {
        ++attempts;
        matmul_tiling::MatmulApiTiling tiling(*plat);
        tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dt, false);
        tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dt, false);
        tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dt);
        tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dtBias);
        tiling.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(nSlab), static_cast<int32_t>(K));
        tiling.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
        tiling.EnableBias(hasBias != 0);
        /* The kernel instantiates the MDL template, so ask the engine for the matching tiling. */
        tiling.SetMatmulConfigParams(1);
        tiling.SetBufferSpace(-1, -1, -1);
        const int64_t ret = tiling.GetTiling(cube);
        if (ret == -1) {
            out->diag[0] = -3;
            out->diag[6] = static_cast<int32_t>(nSlab);
            out->diag[7] = attempts;
            return -1;
        }
        if (sizeof(cube) > sizeof(out->cube)) {
            out->diag[0] = -5;
            return -1;
        }
        out->diag[8] = cube.M;
        out->diag[9] = cube.N;
        out->diag[10] = cube.Ka;
        out->diag[11] = cube.Kb;
        out->diag[12] = cube.singleCoreM;
        out->diag[13] = cube.singleCoreN;
        out->diag[14] = cube.usedCoreNum;
        if (cube.M != M || cube.N != N || cube.Ka != K || cube.Kb != K) {
            out->diag[0] = -4;
            return -1;
        }
        scM = cube.singleCoreM;
        scN = cube.singleCoreN;
        if (scM <= 0) {
            scM = M;
        }
        if (scN <= 0) {
            scN = nSlab;
        }
        mBlocks = WqmmCeilDiv(M, scM);
        nBlocks = WqmmCeilDiv(N, scN);
        if (mBlocks < 1) {
            mBlocks = 1;
        }
        if (nBlocks < 1) {
            nBlocks = 1;
        }
        if (mBlocks * nBlocks <= aic) {
            ok = true;
            break;
        }
        /* the engine produced more blocks than the core budget: widen the slab and retry */
        if (scN >= N) {
            break;
        }
        nSlab = WqmmAlignUp(scN * WqmmCeilDiv(mBlocks * nBlocks, aic), WQMM_CUBE_ALIGN);
        if (nSlab > N) {
            nSlab = N;
        }
    }
    if (!ok) {
        out->diag[0] = -6;
        out->diag[6] = static_cast<int32_t>(nSlab);
        out->diag[7] = attempts;
        return -1;
    }
    out->diag[0] = 0;
    out->diag[2] = static_cast<int32_t>(nSlab);
    out->diag[7] = attempts;

    /* make the serialised blob describe the grid that will actually be launched */
    cube.usedCoreNum = static_cast<int32_t>(mBlocks * nBlocks);

    out->singleCoreM = static_cast<int32_t>(scM);
    out->singleCoreN = static_cast<int32_t>(scN);
    out->mBlocks = static_cast<int32_t>(mBlocks);
    out->nBlocks = static_cast<int32_t>(nBlocks);
    std::memcpy(out->cube, &cube, sizeof(cube));
    out->diag[3] = cube.baseM;
    out->diag[4] = cube.baseN;
    out->diag[5] = cube.baseK;
    out->diag[6] = cube.singleCoreN;
    out->diag[7] = cube.singleCoreK;
    out->diag[8] = cube.depthA1;
    out->diag[9] = cube.depthB1;
    out->diag[10] = cube.stepKa;
    out->diag[11] = cube.stepKb;
    out->diag[12] = cube.stepM;
    out->diag[13] = cube.stepN;
    out->diag[14] = cube.usedCoreNum;
    out->diag[15] = cube.transLength;

    /* ---- dequant launch geometry ---- */
    int64_t aiv = static_cast<int64_t>(plat->GetCoreNumAiv());
    if (aiv <= 0) {
        aiv = 1;
    }
    if (aiv > K) {
        aiv = K;
    }
    int64_t rowsPerCore = WqmmCeilDiv(K, aiv);
    /* One column chunk per core pass; the kernel keeps two in-flight row blocks of this width in UB. */
    int64_t colTile = (N < WQMM_DQ_COL_TILE) ? N : WQMM_DQ_COL_TILE;

    if (dqCores != nullptr) {
        *dqCores = aiv;
    }
    if (dqRowsPerCore != nullptr) {
        *dqRowsPerCore = rowsPerCore;
    }
    if (dqColTile != nullptr) {
        *dqColTile = colTile;
    }
    if (sysWsBytes != nullptr) {
        int64_t libWs = static_cast<int64_t>(plat->GetLibApiWorkSpaceSize());
        if (libWs < 16 * 1024 * 1024) {
            libWs = 16 * 1024 * 1024;
        }
        *sysWsBytes = libWs;
    }
    return 0;
}

/*!
 * \brief Tiling for the anti-quant path: A = half [M, K], B = int8 [K, N], C = float [M, N], no bias
 *        (the bias is applied by the epilogue, in fp32, exactly like the reference).
 *
 * The engine is told to use the MDL tiling so the produced parameters are the ones the MDL template the
 * kernel instantiates expects.
 */
#if 0 /* Investigated and abandoned on this target: the A16W8 anti-quant Matmul scenario
       * (SetAntiQuantVector) is documented for 310P only and is not present on the DAV_2201
       * Matmul client (the member does not exist), so the dequantisation cannot be fused into the
       * Matmul GM->L1 copy here.  See wqmm_qmm.cpp.  Kept out of the build for reference. */
int64_t calc_wqmm_quant_plan(int64_t M, int64_t N, int64_t K, int64_t nAic, WqmmQuantPOD *out)
{
    if (out == nullptr) {
        return -1;
    }
    std::memset(out, 0, sizeof(*out));
    out->M = static_cast<int32_t>(M);
    out->N = static_cast<int32_t>(N);
    out->K = static_cast<int32_t>(K);
    out->diag[3] = static_cast<int32_t>(sizeof(AscendC::tiling::TCubeTiling));
    if (M <= 0 || N <= 0 || K <= 0) {
        out->diag[0] = -1;
        return -1;
    }

    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        out->diag[0] = -2;
        return -1;
    }

    int64_t aic = (nAic > 0) ? nAic : static_cast<int64_t>(plat->GetCoreNumAic());
    if (aic <= 0) {
        aic = 1;
    }
    if (aic > N) {
        aic = N;
    }
    out->diag[1] = static_cast<int32_t>(aic);

    int64_t nSlab = WqmmAlignUp(WqmmCeilDiv(N, aic), WQMM_CUBE_ALIGN);
    if (nSlab < WQMM_CUBE_ALIGN) {
        nSlab = WQMM_CUBE_ALIGN;
    }
    if (nSlab > N) {
        nSlab = N;
    }

    AscendC::tiling::TCubeTiling cube;
    int64_t scM = M;
    int64_t scN = nSlab;
    int64_t mBlocks = 1;
    int64_t nBlocks = WqmmCeilDiv(N, nSlab);
    bool ok = false;
    int32_t attempts = 0;
    for (int attempt = 0; attempt < 6; ++attempt) {
        ++attempts;
        matmul_tiling::MatmulApiTiling tiling(*plat);
        tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16, false);
        tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_INT8, false);
        tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
        tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
        tiling.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(nSlab), static_cast<int32_t>(K));
        tiling.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
        tiling.EnableBias(false);
        tiling.SetMatmulConfigParams(1);     /* MDL template, matching the kernel side */
        tiling.SetBufferSpace(-1, -1, -1);
        const int64_t ret = tiling.GetTiling(cube);
        if (ret == -1) {
            out->diag[0] = -3;
            out->diag[4] = static_cast<int32_t>(nSlab);
            out->diag[5] = attempts;
            return -1;
        }
        if (sizeof(cube) > sizeof(out->cube)) {
            out->diag[0] = -5;
            return -1;
        }
        out->diag[2] = cube.singleCoreM;
        out->diag[6] = cube.singleCoreN;
        out->diag[7] = cube.usedCoreNum;
        if (cube.M != M || cube.N != N || cube.Ka != K || cube.Kb != K) {
            out->diag[0] = -4;
            return -1;
        }
        scM = cube.singleCoreM;
        scN = cube.singleCoreN;
        if (scM <= 0) {
            scM = M;
        }
        if (scN <= 0) {
            scN = nSlab;
        }
        mBlocks = WqmmCeilDiv(M, scM);
        nBlocks = WqmmCeilDiv(N, scN);
        if (mBlocks < 1) {
            mBlocks = 1;
        }
        if (nBlocks < 1) {
            nBlocks = 1;
        }
        if (mBlocks * nBlocks <= aic) {
            ok = true;
            break;
        }
        if (scN >= N) {
            break;
        }
        nSlab = WqmmAlignUp(scN * WqmmCeilDiv(mBlocks * nBlocks, aic), WQMM_CUBE_ALIGN);
        if (nSlab > N) {
            nSlab = N;
        }
    }
    if (!ok) {
        out->diag[0] = -6;
        out->diag[4] = static_cast<int32_t>(nSlab);
        out->diag[5] = attempts;
        return -1;
    }

    cube.usedCoreNum = static_cast<int32_t>(mBlocks * nBlocks);
    out->diag[0] = 0;
    out->singleCoreM = static_cast<int32_t>(scM);
    out->singleCoreN = static_cast<int32_t>(scN);
    out->mBlocks = static_cast<int32_t>(mBlocks);
    out->nBlocks = static_cast<int32_t>(nBlocks);
    out->coefLen = static_cast<int32_t>(WqmmAlignUp(scN, WQMM_CUBE_ALIGN));
    std::memcpy(out->cube, &cube, sizeof(cube));

    /* ---- helper-kernel launch geometry ---- */
    int64_t aiv = static_cast<int64_t>(plat->GetCoreNumAiv());
    if (aiv <= 0) {
        aiv = 1;
    }
    int64_t prepTotal = M * K;
    int64_t prepCores = (prepTotal + 4095) / 4096;
    if (prepCores > aiv) {
        prepCores = aiv;
    }
    if (prepCores < 1) {
        prepCores = 1;
    }
    out->prepCores = static_cast<int32_t>(prepCores);
    out->prepPerCore = static_cast<int32_t>((prepTotal + prepCores - 1) / prepCores);

    int64_t epiCores = M;
    if (epiCores > aiv) {
        epiCores = aiv;
    }
    if (epiCores < 1) {
        epiCores = 1;
    }
    out->epiCores = static_cast<int32_t>(epiCores);
    out->epiRowsPerCore = static_cast<int32_t>((M + epiCores - 1) / epiCores);
    int64_t colTile = N < 4096 ? N : 4096;
    if (colTile < 1) {
        colTile = 1;
    }
    out->epiColTile = static_cast<int32_t>(colTile);

    int64_t libWs = static_cast<int64_t>(plat->GetLibApiWorkSpaceSize());
    if (libWs < 16 * 1024 * 1024) {
        libWs = 16 * 1024 * 1024;
    }
    out->sysWs = static_cast<int32_t>(libWs);
    return 0;
}
#endif /* #if 0 anti-quant tiling */
