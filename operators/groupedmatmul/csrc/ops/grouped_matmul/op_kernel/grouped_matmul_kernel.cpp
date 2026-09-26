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
 * \file grouped_matmul_kernel.cpp
 * \brief GroupedMatmul device kernel + host tiling/launch wrappers (bisheng + -xasc, dav-2201).
 *
 * Math (per expert g, rows_g = [gl[g-1], gl[g]), gl[-1] = 0):
 *     y[rows_g, :] = x[rows_g, :] @ weight[g] (+ bias[g])
 *
 * Design
 * ------
 * The problem is decomposed into independent tasks, one task being "at most singleM rows AND at most
 * chunkCols columns of one expert".  A single kernel launch runs every task with a grid-stride loop over
 * the cube block index, so even highly unbalanced / mostly-empty group lists keep every core busy.
 *
 * For each task the Matmul object is reused:
 *     SetOrgShape(rows, N, K, K) -> SetSingleShape(rows, cols, K) -> SetTensorA/SetTensorB/SetBias -> IterateAll
 * `N` stays the original column count so that it remains the row stride of the weight and of the result;
 * `cols` is only the width of the block being computed.  A task places its own A/B/C base at the exact
 * element offset it needs (the Matmul high level API computes relative to the pointers it is given).
 *
 * Splitting M re-reads the expert weight once per row chunk; splitting N re-reads the activation once per
 * column chunk.  The host plan therefore chooses (singleM, chunkCols, nCuts) with a small traffic model
 * whose per-core cost is ceil(tasks / coreNum) * (singleM * K + K * chunkCols) elements.
 *
 * Pure cube mode (ASCENDC_CUBE_ONLY) is used: this operator only needs the matrix unit.
 * The Matmul high level API requires the system workspace, which is handed in by the API layer as an
 * explicit GM_ADDR kernel argument (the framework does not register one for a hand written launch).
 */

#define ASCENDC_CUBE_ONLY

#include <cstdint>

#include "kernel_operator.h"
#include "lib/matmul/matmul_intf.h"
#include "platform/platform_ascendc.h"
#include "lib/matmul/matmul_tiling.h"
#include "lib/matmul/bmm_tiling.h"

#include "grouped_matmul_launch.h"

using namespace AscendC;

using GmmTiling = AscendC::tiling::TCubeTiling;

/*!
 * \brief One (expert, M-chunk, N-chunk) GEMM.
 *
 * The chunk of A is [rows, K] starting at row m0, the chunk of B is the column block [n0, n0+cols), and the
 * chunk of C is [rows, cols] at (m0, n0).  `N` is the ORIGINAL column count and therefore the stride of
 * both B (row-major [K, N] / [N, K]) and C; `cols` is only the width actually computed.
 */
template <typename Tx, typename Tbias, bool TRANS_B>
__global__ __aicore__ void gmm_kernel(GM_ADDR xPtr, GM_ADDR wPtr, GM_ADDR bPtr, GM_ADDR yPtr, GM_ADDR ws,
                                      int32_t K, int32_t N, int32_t E, int32_t singleM, int32_t chunkCols,
                                      int32_t nCuts, int32_t totalTasks, GmmTiling tiling, GmmGroupArg gl)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    using aType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, Tx>;
    using bType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, Tx, TRANS_B>;
    using cType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, Tx>;
    using biasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, Tbias>;

    GlobalTensor<Tx> xGm;
    GlobalTensor<Tx> wGm;
    GlobalTensor<Tx> yGm;
    GlobalTensor<Tbias> bGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ Tx*>(xPtr));
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ Tx*>(wPtr));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ Tx*>(yPtr));
    const bool hasBias = (bPtr != nullptr);
    if (hasBias) {
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ Tbias*>(bPtr));
    }
    if (ws == nullptr) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::Matmul<aType, bType, cType, biasType> mm;
    REGIST_MATMUL_OBJ(&pipe, ws, mm, &tiling);

    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t nb = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t iK = static_cast<int64_t>(K);
    const int64_t iN = static_cast<int64_t>(N);

    for (int64_t t = blk; t < static_cast<int64_t>(totalTasks); t += nb) {
        int64_t acc = 0;
        int64_t g = -1;
        int64_t mi = 0;
        int64_t ni = 0;
        for (int64_t e = 0; e < static_cast<int64_t>(E); ++e) {
            const int64_t s = (e == 0) ? 0 : static_cast<int64_t>(gl.ends[e - 1]);
            const int64_t en = static_cast<int64_t>(gl.ends[e]);
            if (en <= s) {
                continue;
            }
            const int64_t mChunks = (en - s + static_cast<int64_t>(singleM) - 1) / static_cast<int64_t>(singleM);
            const int64_t cnt = mChunks * static_cast<int64_t>(nCuts);
            if (t < acc + cnt) {
                g = e;
                const int64_t r = t - acc;
                mi = r / static_cast<int64_t>(nCuts);
                ni = r % static_cast<int64_t>(nCuts);
                break;
            }
            acc += cnt;
        }
        if (g < 0) {
            continue;
        }
        const int64_t s = (g == 0) ? 0 : static_cast<int64_t>(gl.ends[g - 1]);
        const int64_t en = static_cast<int64_t>(gl.ends[g]);
        const int64_t m0 = s + mi * static_cast<int64_t>(singleM);
        int64_t rows = en - m0;
        if (rows > static_cast<int64_t>(singleM)) {
            rows = static_cast<int64_t>(singleM);
        }
        const int64_t n0 = ni * static_cast<int64_t>(chunkCols);
        int64_t cols = iN - n0;
        if (cols > static_cast<int64_t>(chunkCols)) {
            cols = static_cast<int64_t>(chunkCols);
        }
        if (rows <= 0 || cols <= 0) {
            continue;
        }

        mm.SetOrgShape(static_cast<int32_t>(rows), N, K, K);
        mm.SetSingleShape(static_cast<int32_t>(rows), static_cast<int32_t>(cols), K);
        mm.SetTensorA(xGm[m0 * iK], false);
        if (TRANS_B) {
            mm.SetTensorB(wGm[g * iN * iK + n0 * iK], true);
        } else {
            mm.SetTensorB(wGm[g * iK * iN + n0], false);
        }
        if (hasBias) {
            mm.SetBias(bGm[g * iN + n0]);
        }
        mm.IterateAll(yGm[m0 * iN + n0]);
    }
    mm.End();
}

// ---------------------------------------------------------------------------
// Host side: plan, tiling and launch (plain host code inside the bisheng TU)
// ---------------------------------------------------------------------------

namespace {

inline int64_t GmmAlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

int64_t GmmMaxRows(const GmmGroupArg& gl)
{
    int64_t mx = 0;
    for (int32_t g = 0; g < gl.count; ++g) {
        const int64_t s = (g == 0) ? 0 : gl.ends[g - 1];
        const int64_t e = gl.ends[g];
        if (e - s > mx) {
            mx = e - s;
        }
    }
    return mx;
}

int64_t GmmTasksAt(const GmmGroupArg& gl, int64_t singleM)
{
    int64_t t = 0;
    for (int32_t g = 0; g < gl.count; ++g) {
        const int64_t s = (g == 0) ? 0 : gl.ends[g - 1];
        const int64_t e = gl.ends[g];
        if (e > s) {
            t += (e - s + singleM - 1) / singleM;
        }
    }
    return t;
}

struct GmmPlan {
    int64_t singleM;
    int64_t chunkCols;
    int64_t nCuts;
    int64_t totalTasks;
};

/*!
 * \brief Modelled wall time of one (singleM, chunkCols, nCuts) plan, in seconds.
 *
 * Two resources are modelled:
 *   * the aggregate HBM traffic -- splitting M re-reads the expert weight, splitting N re-reads the
 *     activation, and the result is written once; a task count that is small cannot have more than
 *     tasks * per-core bandwidth, a task count that is large saturates the shared HBM bandwidth;
 *   * the cube time of the longest core -- ceil(tasks / coreNum) waves of singleM x chunkCols x K mma.
 * The plan minimises the maximum of the two (plus a small per-wave ramp cost).
 */
double GmmEstimate(const GmmGroupArg& gl, int64_t M, int64_t N, int64_t K, int64_t coreNum, int64_t esize,
                   int64_t sM, int64_t nCuts, int64_t chunkCols, int64_t& tasksOut)
{
    const int64_t t0 = GmmTasksAt(gl, sM);
    if (t0 <= 0) {
        return -1.0;
    }
    const int64_t tasks = t0 * nCuts;
    tasksOut = tasks;
    int64_t waves = (tasks + coreNum - 1) / coreNum;
    if (waves < 1) {
        waves = 1;
    }
    const double kBytesPerSecond = 1.1e12;   // shared HBM bandwidth
    const double kPerCoreBW = 8.0e10;        // bandwidth a single cube core can pull
    const double kMacPerCycle = 4096.0;      // 16x16x16 mma
    const double kFreq = 1.8e9;
    const double kRampSeconds = 3.0e-7;      // pipeline fill/drain per wave

    double bw = kPerCoreBW * static_cast<double>(tasks);
    if (bw > kBytesPerSecond) {
        bw = kBytesPerSecond;
    }
    if (bw <= 0.0) {
        bw = kPerCoreBW;
    }
    const double traffic = static_cast<double>(nCuts) * static_cast<double>(M) * static_cast<double>(K) *
                               static_cast<double>(esize) +
                           static_cast<double>(tasks) * static_cast<double>(K) *
                               static_cast<double>(chunkCols) * static_cast<double>(esize) +
                           static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(esize);
    const double tTraffic = traffic / bw;
    const double tCompute = static_cast<double>(waves) * static_cast<double>(sM) *
                            static_cast<double>(chunkCols) * static_cast<double>(K) / kMacPerCycle / kFreq;
    const double t = (tTraffic > tCompute) ? tTraffic : tCompute;
    return t + kRampSeconds * static_cast<double>(waves);
}

/*!
 * \brief Choose the task grid minimising the modelled wall time.
 *
 * Splitting M multiplies the weight traffic, splitting N multiplies the activation traffic; both multiply
 * the task count and therefore the parallelism.  A minimum column chunk keeps a narrow column block from
 * wasting the cube.
 */
GmmPlan GmmMakePlan(const GmmGroupArg& gl, int64_t M, int64_t N, int64_t K, int64_t coreNum, int64_t esize)
{
    const int64_t MIN_CHUNK = 256;
    const int64_t MAX_CUTS = 16;
    GmmPlan best;
    best.singleM = 16;
    best.chunkCols = (N > 0) ? N : 1;
    best.nCuts = 1;
    best.totalTasks = GmmTasksAt(gl, 16);

    const int64_t maxRows = GmmMaxRows(gl);
    if (maxRows <= 0 || N <= 0 || K <= 0 || M <= 0) {
        return best;
    }

    double bestCost = -1.0;
    for (int64_t nCuts = 1; nCuts <= MAX_CUTS; ++nCuts) {
        int64_t chunkCols = GmmAlignUp((N + nCuts - 1) / nCuts, 16);
        if (chunkCols > N) {
            chunkCols = N;
        }
        if (nCuts > 1 && chunkCols < MIN_CHUNK) {
            break;
        }
        // Candidate row chunks: maxRows / k for k = 1..96, aligned up to 16, plus the extremes.
        for (int64_t k = 1; k <= 96; ++k) {
            int64_t sM = GmmAlignUp((maxRows + k - 1) / k, 16);
            if (sM > maxRows) {
                sM = maxRows;
            }
            if (sM < 16) {
                sM = 16;
            }
            int64_t tasks = 0;
            const double cost = GmmEstimate(gl, M, N, K, coreNum, esize, sM, nCuts, chunkCols, tasks);
            if (cost < 0.0) {
                continue;
            }
            if (bestCost < 0.0 || cost < bestCost) {
                bestCost = cost;
                best.singleM = sM;
                best.chunkCols = chunkCols;
                best.nCuts = nCuts;
                best.totalTasks = tasks;
            }
        }
    }
    return best;
}

matmul_tiling::DataType GmmMmDataType(int32_t code)
{
    if (code == GMM_DT_BF16) {
        return matmul_tiling::DataType::DT_BFLOAT16;
    }
    if (code == GMM_DT_FLOAT) {
        return matmul_tiling::DataType::DT_FLOAT;
    }
    return matmul_tiling::DataType::DT_FLOAT16;
}

bool GmmBuildTiling(const platform_ascendc::PlatformAscendC& plat, const GmmPlan& plan, int64_t N, int64_t K,
                    bool hasBias, bool transB, int32_t xDtype, int32_t biasDtype, GmmTiling& out)
{
    matmul_tiling::MatmulApiTiling t(plat);
    const matmul_tiling::DataType dt = GmmMmDataType(xDtype);
    const matmul_tiling::DataType bdt = hasBias ? GmmMmDataType(biasDtype) : dt;
    t.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dt);
    t.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dt, transB);
    t.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dt);
    t.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bdt);
    t.SetShape(plan.singleM, plan.chunkCols, K);
    t.SetOrgShape(plan.singleM, N, K, K);
    t.EnableBias(hasBias);
    t.SetBufferSpace(-1, -1, -1);
    return (t.GetTiling(out) != -1);
}

}  // namespace

extern "C" int64_t calc_gmm_workspace_size()
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return 1024;
    }
    uint32_t size = plat->GetLibApiWorkSpaceSize();
    if (size == 0) {
        size = 1024;
    }
    return static_cast<int64_t>(size);
}

extern "C" void launch_gmm(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR ws, int64_t M, int64_t N,
                           int64_t K, GmmGroupArg gl, int32_t xDtype, int32_t biasDtype, int32_t hasBias,
                           int32_t transB, void* stream)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr || gl.count <= 0 || gl.count > GMM_MAX_E || N <= 0 || K <= 0) {
        return;
    }
    int64_t coreNum = static_cast<int64_t>(plat->GetCoreNumAic());
    if (coreNum <= 0) {
        coreNum = 1;
    }

    const int64_t esize = (xDtype == GMM_DT_FLOAT) ? 4 : 2;
    const GmmPlan plan = GmmMakePlan(gl, M, N, K, coreNum, esize);
    if (plan.totalTasks <= 0) {
        return;
    }
    const int64_t numBlocks = (plan.totalTasks < coreNum) ? plan.totalTasks : coreNum;

    GmmTiling tiling;
    if (!GmmBuildTiling(*plat, plan, N, K, hasBias != 0, transB != 0, xDtype, biasDtype, tiling)) {
        // Fall back to a tiling derived from the full worst case block; if that also fails there is
        // nothing sensible left to launch.
        GmmPlan fallback = plan;
        fallback.singleM = GmmMaxRows(gl);
        if (fallback.singleM <= 0) {
            return;
        }
        fallback.chunkCols = N;
        if (!GmmBuildTiling(*plat, fallback, N, K, hasBias != 0, transB != 0, xDtype, biasDtype, tiling)) {
            return;
        }
    }

    const int32_t iK = static_cast<int32_t>(K);
    const int32_t iN = static_cast<int32_t>(N);
    const int32_t iE = gl.count;
    const int32_t iSingleM = static_cast<int32_t>(plan.singleM);
    const int32_t iChunk = static_cast<int32_t>(plan.chunkCols);
    const int32_t iCuts = static_cast<int32_t>(plan.nCuts);
    const int32_t iTasks = static_cast<int32_t>(plan.totalTasks);

#define GMM_LAUNCH(TX, TB, TR)                                                          \
    gmm_kernel<TX, TB, TR><<<numBlocks, nullptr, stream>>>(x, w, b, y, ws, iK, iN, iE, iSingleM, \
                                                           iChunk, iCuts, iTasks, tiling, gl)

    if (transB != 0) {
        if (xDtype == GMM_DT_HALF) {
            if (biasDtype == GMM_DT_FLOAT) {
                GMM_LAUNCH(half, float, true);
            } else if (biasDtype == GMM_DT_BF16) {
                GMM_LAUNCH(half, bfloat16_t, true);
            } else {
                GMM_LAUNCH(half, half, true);
            }
        } else if (xDtype == GMM_DT_BF16) {
            if (biasDtype == GMM_DT_HALF) {
                GMM_LAUNCH(bfloat16_t, half, true);
            } else if (biasDtype == GMM_DT_FLOAT) {
                GMM_LAUNCH(bfloat16_t, float, true);
            } else {
                GMM_LAUNCH(bfloat16_t, bfloat16_t, true);
            }
        } else {
            if (biasDtype == GMM_DT_HALF) {
                GMM_LAUNCH(float, half, true);
            } else if (biasDtype == GMM_DT_BF16) {
                GMM_LAUNCH(float, bfloat16_t, true);
            } else {
                GMM_LAUNCH(float, float, true);
            }
        }
    } else {
        if (xDtype == GMM_DT_HALF) {
            if (biasDtype == GMM_DT_FLOAT) {
                GMM_LAUNCH(half, float, false);
            } else if (biasDtype == GMM_DT_BF16) {
                GMM_LAUNCH(half, bfloat16_t, false);
            } else {
                GMM_LAUNCH(half, half, false);
            }
        } else if (xDtype == GMM_DT_BF16) {
            if (biasDtype == GMM_DT_HALF) {
                GMM_LAUNCH(bfloat16_t, half, false);
            } else if (biasDtype == GMM_DT_FLOAT) {
                GMM_LAUNCH(bfloat16_t, float, false);
            } else {
                GMM_LAUNCH(bfloat16_t, bfloat16_t, false);
            }
        } else {
            if (biasDtype == GMM_DT_HALF) {
                GMM_LAUNCH(float, half, false);
            } else if (biasDtype == GMM_DT_BF16) {
                GMM_LAUNCH(float, bfloat16_t, false);
            } else {
                GMM_LAUNCH(float, float, false);
            }
        }
    }
#undef GMM_LAUNCH
}
