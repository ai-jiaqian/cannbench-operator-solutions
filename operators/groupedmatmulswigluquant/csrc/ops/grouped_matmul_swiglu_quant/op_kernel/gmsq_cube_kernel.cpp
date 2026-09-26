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
 * \file gmsq_cube_kernel.cpp
 * \brief GroupedMatmulSwigluQuant - grouped matmul stage (int8 x int8 -> int32).
 *
 * For every expert g described by the cumsum group_list:
 *     c[rows_g, :] = x[rows_g, :] @ w[g]        (int8 x int8, exact int32 accumulation)
 *
 * Task decomposition (single launch, grid stride over the cube cores):
 *     one task = (expert g, column chunk ni).  A task covers the FULL row range of its expert so that
 *     every element of the weight is streamed exactly once; only the activation rows are re-read, once
 *     per column chunk.
 *
 * One Matmul object is reused for every task:
 *     SetOrgShape(rows, N, K, K) -> SetSingleShape(rows, cols, K) -> SetTensorA/SetTensorB -> IterateAll
 * orgN stays the original column count so that it remains the row stride of both the weight and the
 * result; `cols` is only the width actually computed.  The Matmul high level API computes its output
 * relative to the pointers it is handed, so a task may place its own A/B/C base at any element offset.
 *
 * Pure cube mode (ASCENDC_CUBE_ONLY) is used: this stage only needs the matrix unit.
 */

#define ASCENDC_CUBE_ONLY

#include "kernel_operator.h"
#include "lib/matmul/matmul_intf.h"
#include "platform/platform_ascendc.h"
#include "lib/matmul/matmul_tiling.h"

#include "gmsq_launch.h"

using namespace AscendC;

using GmsqTiling = AscendC::tiling::TCubeTiling;

/*!
 * \brief Device kernel: grouped matmul over a manual (expert, column chunk) task grid.
 */
__global__ __aicore__ void gmsq_cube_kernel(GM_ADDR xPtr, GM_ADDR wPtr, GM_ADDR cPtr, GM_ADDR ws,
                                            int64_t M, int64_t K, int64_t N, int64_t E, int64_t chunkN,
                                            GmsqGroupArg gl, GmsqTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    GlobalTensor<int8_t> xGm;
    GlobalTensor<int8_t> wGm;
    GlobalTensor<int32_t> cGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(xPtr));
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(wPtr));
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cPtr));

    int64_t ends[GMSQ_MAX_E];
    for (int64_t g = 0; g < E; ++g) {
        ends[g] = static_cast<int64_t>(gl.ends[g]);
    }

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    Matmul<aType, bType, cType, biasType> mm;

    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, ws, mm, &tiling);

    const int64_t nCuts = (chunkN > 0) ? ((N + chunkN - 1) / chunkN) : 1;

    // The launch plan depends on the group boundaries, which only exist on the device, so the total
    // task count is derived here from the tiny group_list POD.
    int64_t totalTasks = 0;
    for (int64_t g = 0; g < E; ++g) {
        const int64_t s = (g == 0) ? 0 : ends[g - 1];
        if (ends[g] > s) {
            totalTasks += nCuts;
        }
    }

    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    const int64_t nb = static_cast<int64_t>(GetBlockNum());

    for (int64_t t = blk; t < totalTasks; t += nb) {
        int64_t acc = 0;
        int64_t foundG = -1;
        int64_t foundNi = 0;
        for (int64_t g = 0; g < E; ++g) {
            const int64_t s = (g == 0) ? 0 : ends[g - 1];
            const int64_t e = ends[g];
            if (e > s) {
                if (t < acc + nCuts) {
                    foundG = g;
                    foundNi = t - acc;
                    break;
                }
                acc += nCuts;
            }
        }
        if (foundG < 0) {
            continue;
        }
        const int64_t s = (foundG == 0) ? 0 : ends[foundG - 1];
        const int64_t e = ends[foundG];
        const int64_t rows = e - s;
        const int64_t n0 = foundNi * chunkN;
        int64_t cols = N - n0;
        if (cols > chunkN) {
            cols = chunkN;
        }
        if (rows <= 0 || cols <= 0) {
            continue;
        }

        mm.SetOrgShape(static_cast<int32_t>(rows), static_cast<int32_t>(N), static_cast<int32_t>(K),
                       static_cast<int32_t>(K));
        mm.SetSingleShape(static_cast<int32_t>(rows), static_cast<int32_t>(cols),
                          static_cast<int32_t>(K));
        mm.SetTensorA(xGm[s * K], false);
        mm.SetTensorB(wGm[foundG * K * N + n0], false);
        mm.IterateAll(cGm[s * N + n0]);
    }
    mm.End();
}

// ---------------------------------------------------------------------------
// Host side: tiling + launch
// ---------------------------------------------------------------------------

namespace {

/*! Choose the column chunk so the (expert, chunk) grid lands on roughly two waves of cube cores. */
int64_t GmsqAlignedChunkN(int64_t N, int64_t E, int64_t coreNum, int64_t baseN)
{
    if (N <= 0) {
        return 1;
    }
    int64_t experts = (E > 0) ? E : 1;
    int64_t want = (2 * coreNum + experts - 1) / experts;
    if (want < 1) {
        want = 1;
    }
    int64_t unit = (baseN > 0) ? baseN : 128;
    if (unit > N) {
        unit = N;
    }
    int64_t chunk = (N + want - 1) / want;
    chunk = ((chunk + unit - 1) / unit) * unit;
    if (chunk < unit) {
        chunk = unit;
    }
    if (chunk > N) {
        chunk = N;
    }
    return chunk;
}

/*! Fill a kernel side TCubeTiling for a single core int8 x int8 -> int32 matmul of shape [M,N]x[K]. */
void GmsqBuildTiling(int64_t M, int64_t K, int64_t N, GmsqTiling* out)
{
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT8);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT8);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_INT32);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_INT32);
    tiling.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    tiling.SetOrgShape(static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K));
    tiling.EnableBias(false);
    tiling.SetBufferSpace(-1, -1, -1);
    (void)tiling.GetTiling(*out);
}

}  // namespace

int64_t gmsq_calc_aic_num()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = 0;
    if (platform != nullptr) {
        coreNum = static_cast<int64_t>(platform->GetCoreNumAic());
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    return coreNum;
}

int64_t gmsq_calc_aiv_num()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = 0;
    if (platform != nullptr) {
        coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    return coreNum;
}

int64_t gmsq_calc_workspace_size()
{
    return 16 * 1024 * 1024;
}

/*! Number of (expert, column chunk) tasks that actually carry work, derived from the cumsum POD. */
int64_t GmsqCountTasks(const GmsqGroupArg& gl, int64_t E, int64_t nCuts)
{
    int64_t tasks = 0;
    for (int64_t g = 0; g < E; ++g) {
        const int64_t s = (g == 0) ? 0 : static_cast<int64_t>(gl.ends[g - 1]);
        if (static_cast<int64_t>(gl.ends[g]) > s) {
            tasks += nCuts;
        }
    }
    return tasks;
}

extern "C" void launch_gmsq_cube(GM_ADDR x, GM_ADDR w, GM_ADDR c, GM_ADDR ws, int64_t M, int64_t K,
                                 int64_t N, int64_t E, GmsqGroupArg gl, void* stream)
{
    const int64_t coreNum = gmsq_calc_aic_num();

    GmsqTiling tiling;
    GmsqBuildTiling(M, K, N, &tiling);

    int64_t baseN = static_cast<int64_t>(tiling.baseN);
    const int64_t chunkN = GmsqAlignedChunkN(N, E, coreNum, baseN);
    const int64_t nCuts = (chunkN > 0) ? ((N + chunkN - 1) / chunkN) : 1;

    // Launching blocks that own no task still costs the full per-block Matmul prologue, so the grid is
    // capped at the number of real tasks.
    int64_t blocks = GmsqCountTasks(gl, E, nCuts);
    if (blocks > coreNum) {
        blocks = coreNum;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    gmsq_cube_kernel<<<static_cast<uint32_t>(blocks), nullptr, stream>>>(x, w, c, ws, M, K, N, E,
                                                                         chunkN, gl, tiling);
}
