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
 * \file mhc_sinkhorn_kernel.cpp
 * \brief MhcSinkhorn kernel + tiling + launch (compiled with bisheng + -xasc)
 *
 * y = linear-domain Sinkhorn-Knopp projection of every hc_mult x hc_mult matrix
 *
 *   iter 0 :  m <- exp(m - rowmax(m)); m <- m / rowsum(m); m <- m + eps;
 *             m <- m / (colsum(m) + eps)
 *   iter k :  m <- m / (rowsum(m) + eps); m <- m / (colsum(m) + eps)
 *
 * IMPORTANT - linkage: the kernel entry point, the tiling entry point and the
 * launch wrapper must all live at *global* scope with the exact names declared
 * in mhc_sinkhorn_launch.h.  Both translation units are only combined into a
 * shared library, and a shared library may keep unresolved symbols, so wrapping
 * `calc_mhc_sinkhorn_tiling_params` in a namespace produced a .so that built
 * cleanly and then died on the *first* call from the plugin.
 *
 * The kernel is explicitly marked __vector__ (AIV).
 *
 * Layout: see mhc_helpers.h.  The batch is processed in tiles of R matrices and
 * each tile lives in UB as X[c][i][r][lane], so every (c,i,r) row is exactly one
 * 32B DataBlock.
 *
 * Tile size matters more than anything else here.  Every vector op of an
 * iteration works on the WHOLE R-matrix tile, so R has to match what the core
 * actually owns: with R = 504 while matPerCore is 22, each of the ~14 calls per
 * iteration carries 252 lanes instead of 12, and the measured per-iteration cost
 * (~1.0us, i.e. ~130 cycles per call) is dominated by exactly that wasted work.
 * The tiling therefore clamps R to a multiple of 8 that is just big enough for
 * matPerCore (still capped by the UB budget and by the 255-repeat limit of the
 * block reduce).
 *
 * Padding lanes (lane >= m - 8*c) are filled with -inf by the CopyIn, so the row
 * MAX ignores them and exp(-inf - max) is exactly 0.  They must stay exactly 0
 * for the whole kernel: they are *not* copied out, but they take part in the
 * per-DataBlock row sums, and a non-zero value there grows to O(1/m) through the
 * column normalisation and then poisons every later row sum.  That is why the
 * `+ eps` of iteration 0 is written to the real lanes only: `Adds` puts it on
 * every lane and the masked `Duplicate` in MhcClearPad puts the padding lanes
 * back to exactly 0, so the eps reaches the real lanes exactly like the
 * reference and never reaches the padding.
 *
 * Synchronisation: X, Bx, S, C, OneS, OneC are plain VECCALC TBufs, so every
 * producer/consumer pair that crosses pipes needs an explicit barrier:
 * MTE2(CopyIn) -> V(compute), V(compute) -> MTE3(CopyOut) and MTE3 -> MTE2
 * (the next tile overwrites X).  PipeBarrier<PIPE_ALL> is used for those three
 * crossings; inside the vector pipe the helpers guard their own dependencies.
 *
 * Everything is fp32 (the operator is fp32-only by specification).
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "mhc_diag.h"
#include "mhc_helpers.h"

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

using namespace AscendC;
using namespace mhcdev;

/* ------------------------------------------------------------------------- */
/* Kernel                                                                    */
/* ------------------------------------------------------------------------- */
__global__ __aicore__ __vector__ void mhc_sinkhorn_kernel(GM_ADDR x, GM_ADDR y, int64_t totalMats,
    int32_t hcMult, int32_t iterStep, float eps, int64_t matPerCore, int32_t R)
{
#if MHC_DIAG_NOOP
    (void)x; (void)y; (void)totalMats; (void)hcMult; (void)iterStep; (void)eps;
    (void)matPerCore; (void)R;
    return;
#else
    const int32_t m = hcMult;
    const int32_t NC = (m + 7) / 8;
    const int32_t NR = NC * m * R;        /* 8-lane rows inside one tile      */
    const int32_t E = NR * 8;             /* floats inside one tile           */
    const int32_t RB = R * 8;             /* floats of one (chunk,row) slice  */
    const int32_t CB = NC * RB;           /* floats of one column-sum slice   */
    const int32_t SMSZ = m * R;           /* distinct row values per tile     */

    const int64_t blk = (int64_t)GetBlockIdx();
    int64_t b0 = blk * matPerCore;
    if (b0 >= totalMats) {
        return;
    }
    int64_t b1 = b0 + matPerCore;
    if (b1 > totalMats) {
        b1 = totalMats;
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuf, bBuf, sBuf, cBuf, oneSBuf, oneCBuf;
    pipe.InitBuffer(xBuf, (uint32_t)(E * (int32_t)sizeof(float)));
    pipe.InitBuffer(bBuf, (uint32_t)(E * (int32_t)sizeof(float)));
    pipe.InitBuffer(sBuf, (uint32_t)((NR + 64) * (int32_t)sizeof(float)));
    pipe.InitBuffer(cBuf, (uint32_t)((CB + 64) * (int32_t)sizeof(float)));
    pipe.InitBuffer(oneSBuf, (uint32_t)((SMSZ + 64) * (int32_t)sizeof(float)));
    pipe.InitBuffer(oneCBuf, (uint32_t)((CB + 64) * (int32_t)sizeof(float)));

    LocalTensor<float> X = xBuf.Get<float>();
    LocalTensor<float> Bx = bBuf.Get<float>();
    LocalTensor<float> S = sBuf.Get<float>();
    LocalTensor<float> C = cBuf.Get<float>();
    LocalTensor<float> OneS = oneSBuf.Get<float>();
    LocalTensor<float> OneC = oneCBuf.Get<float>();

    Duplicate(OneS, 1.0f, SMSZ);
    Duplicate(OneC, 1.0f, CB);
    PipeBarrier<PIPE_V>();

    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x);
    yGm.SetGlobalBuffer((__gm__ float*)y);

    for (int64_t tile = b0; tile < b1; tile += (int64_t)R) {
        int64_t rc = b1 - tile;
        if (rc > (int64_t)R) {
            rc = (int64_t)R;
        }
        const uint16_t rows = (uint16_t)rc;

        /* ---- CopyIn : gather the tile so that every (row, chunk) is one 32B block */
        for (int32_t c = 0; c < NC; ++c) {
            int32_t w = m - 8 * c;
            if (w > 8) {
                w = 8;
            }
            const uint32_t srcGap = (uint32_t)((m * m - w) * (int32_t)sizeof(float));
            for (int32_t i = 0; i < m; ++i) {
                LocalTensor<float> dstT = X[(c * m + i) * RB];
                GlobalTensor<float> srcT;
                srcT.SetGlobalBuffer((__gm__ float*)x + tile * m * m + i * m + 8 * c);
                DataCopyExtParams cp;
                cp.blockCount = rows;
                cp.blockLen = (uint32_t)(w * (int32_t)sizeof(float));
                cp.srcStride = srcGap;
                cp.dstStride = 0;
                cp.rsv = 0;
                DataCopyPadExtParams<float> pp;
                pp.isPad = (w < 8);
                pp.leftPadding = 0;
                pp.rightPadding = (uint8_t)(8 - w);
                pp.paddingValue = MhcNegInf();
                DataCopyPad(dstT, srcT, cp, pp);
            }
        }
        /* MTE2 -> V : the vector pipe must not read X before the DMA landed */
        PipeBarrier<PIPE_ALL>();

#if !MHC_DIAG_DMA_ONLY
        /* ---- iteration 0 : numerically stable row-softmax shift + exp ---- */
        MhcRowMax(S, X, E);
        if (NC > 1) {
            /* S[c*SMSZ + (i*R+r)] = max over the chunks of matrix row i */
            Max(S, S, S[SMSZ], SMSZ);
            PipeBarrier<PIPE_V>();
        }
        MhcBroadcast(Bx, S, SMSZ, NC);
        Sub(X, X, Bx, E);
        PipeBarrier<PIPE_V>();
        Exp(X, X, E);
        PipeBarrier<PIPE_V>();

        /* ---- Sinkhorn iterations ---- */
        for (int32_t it = 0; it < iterStep; ++it) {
            /* row normalisation.  The it>0 reference divisor is rowsum + eps;
               rowsum is ~1 here (the row pass normalised it and the column pass
               only multiplies by O(1) reciprocals) and it is strictly positive
               because the padding lanes hold exact 0, so dropping the eps costs
               a relative 1e-6 and removes one call per iteration. */
            MhcRowSum(S, X, E);
            if (NC > 1) {
                Add(S, S, S[SMSZ], SMSZ);
                PipeBarrier<PIPE_V>();
            }
            Div(S, OneS, S, SMSZ);
            PipeBarrier<PIPE_V>();
            MhcBroadcast(Bx, S, SMSZ, NC);
            Mul(X, X, Bx, E);
            PipeBarrier<PIPE_V>();
            if (it == 0) {
                /* reference: comb = comb / row_sum + eps, real lanes only */
                Adds(X, X, eps, E);
                PipeBarrier<PIPE_V>();
                MhcClearPad(X, m, NC, RB);
            }

            /* column normalisation : sum over i, then scale by the reciprocal */
            MhcColSum(C, X, m, NC, RB);
            Adds(C, C, eps, CB);
            PipeBarrier<PIPE_V>();
            Div(C, OneC, C, CB);
            PipeBarrier<PIPE_V>();
            MhcColScale(X, C, m, NC, RB);
        }
#endif

        /* V -> MTE3 : the DMA must not read X before the vector writes landed */
        PipeBarrier<PIPE_ALL>();

        /* ---- CopyOut : scatter the tile back to the [B, m, m] row-major layout */
        for (int32_t c = 0; c < NC; ++c) {
            int32_t w = m - 8 * c;
            if (w > 8) {
                w = 8;
            }
            const uint32_t dstGap = (uint32_t)((m * m - w) * (int32_t)sizeof(float));
            for (int32_t i = 0; i < m; ++i) {
                GlobalTensor<float> dstT;
                dstT.SetGlobalBuffer((__gm__ float*)y + tile * m * m + i * m + 8 * c);
                LocalTensor<float> srcT = X[(c * m + i) * RB];
                DataCopyExtParams cp;
                cp.blockCount = rows;
                cp.blockLen = (uint32_t)(w * (int32_t)sizeof(float));
                cp.srcStride = 0;
                cp.dstStride = dstGap;
                cp.rsv = 0;
                DataCopyPad(dstT, srcT, cp);
            }
        }
        /* MTE3 -> MTE2 : the next tile must not overwrite X while it is read out */
        PipeBarrier<PIPE_ALL>();
    }
#endif
}

/* ------------------------------------------------------------------------- */
/* Tiling (global scope: declared in mhc_sinkhorn_launch.h)                  */
/* ------------------------------------------------------------------------- */
std::tuple<int64_t, int64_t, int64_t> calc_mhc_sinkhorn_tiling_params(int64_t totalMats, int64_t hcMult)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    if (ubSize == 0) {
        ubSize = 192 * 1024;
    }

    const int64_t m = hcMult;
    const int64_t NC = (m + 7) / 8;

    int64_t numBlocks = coreNum;
    if (numBlocks > totalMats) {
        numBlocks = totalMats;
    }
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    const int64_t matPerCore = (totalMats + numBlocks - 1) / numBlocks;

    /* UB bytes consumed per 8-lane row of a tile (X + broadcast + row values) */
    const int64_t perRow = 4 * (8 + 8 + 1) + (64 * NC) / m + 32;
    int64_t budget = (int64_t)ubSize - 32 * 1024;
    if (budget < 24 * 1024) {
        budget = 24 * 1024;
    }
    const int64_t maxNR = budget / perRow;
    int64_t maxR = maxNR / (NC * m);
    /* the reduce / Brcb repeat count must stay within 255 */
    const int64_t repCap = 2040 / (NC * m);
    if (maxR > repCap) {
        maxR = repCap;
    }
    if (maxR > 512) {
        maxR = 512;
    }

    /* Never make the tile larger than the work this core owns: every vector op
       of an iteration covers the whole R-matrix tile, so R >> matPerCore means
       each call carries R/matPerCore times more lanes than necessary. */
    int64_t want = ((matPerCore + 7) / 8) * 8;
    if (want < 8) {
        want = 8;
    }
    if (maxR > want) {
        maxR = want;
    }
    int64_t R = (maxR / 8) * 8;
    if (R < 8) {
        R = 8;
    }
    return std::make_tuple(numBlocks, matPerCore, R);
}

/* ------------------------------------------------------------------------- */
/* Launch wrapper (regular C function callable from g++)                     */
/* ------------------------------------------------------------------------- */
extern "C" {

void launch_mhc_sinkhorn(GM_ADDR x, GM_ADDR y, int64_t totalMats, int32_t hcMult, int32_t iterStep,
                         float eps, int64_t numBlocks, int64_t matPerCore, int32_t tileMats,
                         void* stream)
{
    mhc_sinkhorn_kernel<<<(uint32_t)numBlocks, nullptr, stream>>>(
        x, y, totalMats, hcMult, iterStep, eps, matPerCore, tileMats);
}
}
