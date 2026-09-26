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
 * \file gmsq_vec_kernel.cpp
 * \brief GroupedMatmulSwigluQuant - dequant + SwiGLU + per-token requant stage.
 *
 * For every token row i (expert g = g(i) taken from the cumsum group_list):
 *     deqL[j] = mm[i, j]         * xScale[i] * wScale[g, j]
 *     deqR[j] = mm[i, halfN + j]  * xScale[i] * wScale[g, halfN + j]
 *     act[j]  = silu(deqL[j]) * deqR[j]
 *     yScale[i] = max(GMSQ_TINY, max_j |act[j]|) / 127
 *     y[i, j]   = clamp(round(act[j] / yScale[i]), -128, 127)
 *
 * Every core owns a contiguous range of rows, so the per-row work is uniform and the core count does not
 * have to be guessed.  The rows of a group are contiguous as well, therefore the expert index only changes
 * a handful of times along a core's range: wScale[g, :] is cached in UB and re-fetched only when the expert
 * changes.  That removes M * N * 4 bytes of DRAM traffic (the whole per-channel row was otherwise read once
 * per token row) and cuts the per-row DMA count in half.
 *
 * Each row is walked in column chunks of GMSQ_CHUNK elements of the half width so the UB budget stays
 * bounded while the signed activation of the whole row is retained (two passes are needed: max first,
 * then quantize).  The per-token scales of all rows owned by a core are staged in UB and flushed with a
 * single contiguous burst, which keeps the tiny 4 byte stores off the DMA issue path.
 *
 * The per-row dependency on the scalar pipe is kept to a single scalar read: the whole x_scale range of
 * the block is fetched once up front instead of one GM scalar load per row, and the row magnitude is
 * obtained by reducing `abs(act)` per chunk (one vector->scalar sync per chunk) rather than by a
 * max/min reduction pair over the whole row.
 */

#include "kernel_operator.h"

#include "gmsq_launch.h"

using namespace AscendC;

namespace {

constexpr int64_t GMSQ_CHUNK = 2048;
constexpr float GMSQ_TINY = 1.1754944e-38f;  // std::numeric_limits<float>::min()
constexpr float GMSQ_INV127 = 1.0f / 127.0f;

__aicore__ inline int64_t GmsqUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

}  // namespace

__global__ __aicore__ void gmsq_vec_kernel(GM_ADDR mmPtr, GM_ADDR xsPtr, GM_ADDR wsPtr, GM_ADDR yPtr,
                                           GM_ADDR ysPtr, int64_t M, int64_t N, int64_t E,
                                           GmsqGroupArg gl)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const int64_t halfN = N / 2;
    if (M <= 0 || halfN <= 0 || E <= 0) {
        return;
    }

    GlobalTensor<int32_t> mmGm;
    GlobalTensor<float> xsGm;
    GlobalTensor<float> wsGm;
    GlobalTensor<float> ysGm;
    GlobalTensor<int8_t> yGm;
    mmGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(mmPtr), M * N);
    xsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xsPtr), M);
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsPtr), E * N);
    ysGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ysPtr), M);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(yPtr), M * halfN);

    int64_t ends[GMSQ_MAX_E];
    for (int64_t g = 0; g < E; ++g) {
        ends[g] = static_cast<int64_t>(gl.ends[g]);
    }

    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    const int64_t nb = static_cast<int64_t>(GetBlockNum());
    const int64_t r0 = (M * blk) / nb;
    const int64_t r1 = (M * (blk + 1)) / nb;
    if (r1 <= r0) {
        return;
    }
    const int64_t myRows = r1 - r0;

    const int64_t ch = (halfN < GMSQ_CHUNK) ? halfN : GMSQ_CHUNK;
    const int64_t nChunks = (halfN + ch - 1) / ch;
    const int64_t chB = GmsqUp(ch * 4, 32) + 128;
    const int64_t actB = GmsqUp(halfN * 4, 32) + 256;
    const int64_t yB = GmsqUp(halfN, 32) + 256;
    const int64_t hB = GmsqUp(ch * 2, 32) + 128;
    const int64_t hA = GmsqUp(halfN, 8);  // cached wScale half-row pitch, always 32 byte aligned
    const int64_t wsB = GmsqUp(hA * 2 * 4, 32) + 256;
    const int64_t ysB = GmsqUp(myRows * 4, 32) + 64;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> qMML;
    TQue<TPosition::VECIN, 1> qMMR;
    TQue<TPosition::VECOUT, 1> qY;
    TBuf<TPosition::VECCALC> bFL;
    TBuf<TPosition::VECCALC> bFR;
    TBuf<TPosition::VECCALC> bFT;
    TBuf<TPosition::VECCALC> bAct;
    TBuf<TPosition::VECCALC> bH;
    TBuf<TPosition::VECCALC> bWS;
    TBuf<TPosition::VECCALC> bXS;
    TBuf<TPosition::VECCALC> bYS;
    TBuf<TPosition::VECCALC> bWork;
    TBuf<TPosition::VECCALC> bRed;

    pipe.InitBuffer(qMML, 1, chB);
    pipe.InitBuffer(qMMR, 1, chB);
    pipe.InitBuffer(qY, 1, yB);
    pipe.InitBuffer(bFL, chB);
    pipe.InitBuffer(bFR, chB);
    pipe.InitBuffer(bFT, chB);
    pipe.InitBuffer(bAct, actB);
    pipe.InitBuffer(bH, hB);
    pipe.InitBuffer(bWS, wsB);
    pipe.InitBuffer(bXS, GmsqUp(myRows * 4, 32) + 64);
    pipe.InitBuffer(bYS, ysB);
    pipe.InitBuffer(bWork, 8192);
    pipe.InitBuffer(bRed, 128);

    LocalTensor<float> fL = bFL.Get<float>();
    LocalTensor<float> fR = bFR.Get<float>();
    LocalTensor<float> fT = bFT.Get<float>();
    LocalTensor<float> act = bAct.Get<float>();
    LocalTensor<half> hT = bH.Get<half>();
    LocalTensor<float> wsLoc = bWS.Get<float>();
    LocalTensor<float> xsLoc = bXS.Get<float>();
    LocalTensor<float> ysLoc = bYS.Get<float>();
    LocalTensor<float> work = bWork.Get<float>();
    LocalTensor<float> red = bRed.Get<float>();

    const DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
    const DataCopyPadExtParams<float> padF{false, 0, 0, 0};
    const DataCopyExtParams cpWL{1, static_cast<uint32_t>(halfN * 4), 0, 0, 0};
    const DataCopyExtParams cpXS{1, static_cast<uint32_t>(myRows * 4), 0, 0, 0};

    // One contiguous fetch of this block's per-token dequant factors instead of a GM scalar load per row.
    DataCopyPad(xsLoc, xsGm[r0], cpXS, padF);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

    int64_t curG = -1;
    int64_t curEnd = -1;
    int64_t cachedG = -1;
    int64_t ysCount = 0;

    for (int64_t row = r0; row < r1; ++row) {
        if (row >= curEnd) {
            int64_t gg = 0;
            while (gg < E && row >= ends[gg]) {
                ++gg;
            }
            curG = (gg < E) ? gg : (E - 1);
            curEnd = ends[curG];
        }
        if (curG != cachedG) {
            if (cachedG >= 0) {
                // make sure the previous row's reads of the cache are done before it is refilled
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            DataCopyPad(wsLoc, wsGm[curG * N], cpWL, padF);
            DataCopyPad(wsLoc[hA], wsGm[curG * N + halfN], cpWL, padF);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            cachedG = curG;
        }

        const int64_t mmBase = row * N;
        const float xs = xsLoc.GetValue(static_cast<int32_t>(row - r0));

        // ---- pass 1: dequant + SwiGLU, keep the signed activation, track abs(act) magnitude ----
        float amax = 0.0f;
        for (int64_t c = 0; c < nChunks; ++c) {
            const int64_t off = c * ch;
            int64_t cnt = halfN - off;
            if (cnt > ch) {
                cnt = ch;
            }
            if (cnt <= 0) {
                break;
            }
            const int32_t cnt32 = static_cast<int32_t>(cnt);
            const DataCopyExtParams cp{1, static_cast<uint32_t>(cnt32 * 4), 0, 0, 0};

            LocalTensor<int32_t> tL = qMML.AllocTensor<int32_t>();
            LocalTensor<int32_t> tR = qMMR.AllocTensor<int32_t>();
            DataCopyPad(tL, mmGm[mmBase + off], cp, padI);
            DataCopyPad(tR, mmGm[mmBase + halfN + off], cp, padI);
            qMML.EnQue(tL);
            qMMR.EnQue(tR);
            tL = qMML.DeQue<int32_t>();
            tR = qMMR.DeQue<int32_t>();

            Cast(fL, tL, RoundMode::CAST_NONE, cnt32);
            Cast(fR, tR, RoundMode::CAST_NONE, cnt32);
            Muls(fL, fL, xs, cnt32);
            Mul(fL, fL, wsLoc[off], cnt32);
            Muls(fR, fR, xs, cnt32);
            Mul(fR, fR, wsLoc[hA + off], cnt32);
            // silu(l) = l / (1 + exp(-l))
            Muls(fT, fL, -1.0f, cnt32);
            Exp(fT, fT, cnt32);
            Adds(fT, fT, 1.0f, cnt32);
            Div(fT, fL, fT, cnt32);
            Mul(act[off], fT, fR, cnt32);

            qMML.FreeTensor(tL);
            qMMR.FreeTensor(tR);

            Abs(fT, act[off], cnt32);
            ReduceMax(red, fT, work, cnt32, false);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float chunkMax = red.GetValue(0);
            if (chunkMax > amax) {
                amax = chunkMax;
            }
        }

        if (!(amax > GMSQ_TINY)) {
            amax = GMSQ_TINY;
        }
        ysLoc.SetValue(ysCount, amax * GMSQ_INV127);
        ++ysCount;

        // ---- pass 2: requantize the retained activation and store the row ----
        const float invScale = 127.0f / amax;
        LocalTensor<int8_t> yLoc = qY.AllocTensor<int8_t>();
        for (int64_t c = 0; c < nChunks; ++c) {
            const int64_t off = c * ch;
            int64_t cnt = halfN - off;
            if (cnt > ch) {
                cnt = ch;
            }
            if (cnt <= 0) {
                break;
            }
            const int32_t cnt32 = static_cast<int32_t>(cnt);
            LocalTensor<float> aChunk = act[off];
            Muls(aChunk, aChunk, invScale, cnt32);
            Mins(aChunk, aChunk, 127.0f, cnt32);
            Maxs(aChunk, aChunk, -128.0f, cnt32);
            Cast(hT, aChunk, RoundMode::CAST_RINT, cnt32);
            Cast(yLoc[off], hT, RoundMode::CAST_RINT, cnt32);
        }
        qY.EnQue(yLoc);
        yLoc = qY.DeQue<int8_t>();
        const DataCopyExtParams cpY{1, static_cast<uint32_t>(halfN), 0, 0, 0};
        DataCopyPad(yGm[row * halfN], yLoc, cpY);
        qY.FreeTensor(yLoc);
    }

    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    const DataCopyExtParams cpS{1, static_cast<uint32_t>(ysCount * 4), 0, 0, 0};
    DataCopyPad(ysGm[r0], ysLoc, cpS);
}

extern "C" void launch_gmsq_vec(GM_ADDR mm, GM_ADDR xScale, GM_ADDR wScale, GM_ADDR y, GM_ADDR yScale,
                                int64_t M, int64_t N, int64_t E, GmsqGroupArg gl, void* stream)
{
    const int64_t coreNum = gmsq_calc_aiv_num();
    // A block that owns no row still pays the kernel prologue, so the grid never exceeds the row count.
    int64_t blocks = (M < coreNum) ? M : coreNum;
    if (blocks < 1) {
        blocks = 1;
    }
    gmsq_vec_kernel<<<static_cast<uint32_t>(blocks), nullptr, stream>>>(mm, xScale, wScale, y, yScale,
                                                                        M, N, E, gl);
}
