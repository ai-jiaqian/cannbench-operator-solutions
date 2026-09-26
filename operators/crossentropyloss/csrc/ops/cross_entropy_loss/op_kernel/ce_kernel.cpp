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
 * \file ce_kernel.cpp
 * \brief CrossEntropyLoss Ascend C kernel (Ascend 910B / DAV-2201, CANN 9.0.0).
 *
 *   loss_r = logsumexp_c(x_r) - x_r[t_r]          (hard class-index targets)
 *
 *   x is viewed as (N, C, P) with P = prod(trailing dims); row r = n*P + p and
 *   the class element c of row r sits at n*C*P + c*P + p.
 *
 *   P == 1  -> the class axis is the innermost (contiguous) one: row kernel.
 *   P >  1  -> the class axis is strided: column kernel over tiles of columns.
 *
 *   All arithmetic happens in fp32 on device; fp16/bf16 are widened with CAST_NONE
 *   and narrowed with CAST_RINT (round-half-even), matching the torch reference.
 *
 *   reduction: 0=none, 1=mean, 2=sum.  Rows whose hard target equals ignore_index
 *   produce 0 and are excluded from the mean denominator.
 */

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "ce_launch.h"

using namespace AscendC;

namespace {

constexpr int32_t kTmpFloats = 4096;                     // 16 KB reduce workspace
constexpr float kNegBig = -3.0e38f;                      // neutral element of a running max

__aicore__ inline int64_t CeAlign(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

__aicore__ inline int64_t CeMinI(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

template <typename T>
struct CeIsF32 {
    static constexpr bool value = false;
};
template <>
struct CeIsF32<float> {
    static constexpr bool value = true;
};

/*!
 * \brief high-dim widening cast half/bf16 -> float.
 *        One repeat converts 64 elements in the "flattened 64 element chunk" space;
 *        src advances by 64*esz bytes (64*esz/32 datablocks) and dst by 64 floats (8 datablocks).
 *        Valid whenever every operand row pitch is a multiple of 64 elements.
 */
template <typename T>
__aicore__ inline void CeCastChunksUp(const LocalTensor<float>& d, const LocalTensor<T>& s, int32_t chunks)
{
    UnaryRepeatParams prm{1, 1, 8, static_cast<uint32_t>(64 * static_cast<int32_t>(sizeof(T)) / 32)};
    int32_t done = 0;
    while (done < chunks) {
        int32_t t = chunks - done;
        if (t > 255) {
            t = 255;
        }
        Cast(d[done * 64], s[done * 64], RoundMode::CAST_NONE, static_cast<uint64_t>(64),
             static_cast<uint8_t>(t), prm);
        done += t;
    }
}

/* =====================================================================================
 *  P == 1 : row kernel.  Rows are contiguous runs of C elements.
 * ===================================================================================== */
template <typename T, typename TI>
__global__ __aicore__ void ce_row_kernel(GM_ADDR xAddr, GM_ADDR tAddr, GM_ADDR outAddr, GM_ADDR partAddr,
                                         int64_t R, int64_t C, int64_t ignoreIdx, int64_t redMode,
                                         int64_t numBlocks, int64_t bs)
{
    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    int64_t per = (R + numBlocks - 1) / numBlocks;
    if (per < 1) {
        per = 1;
    }
    int64_t r0 = blk * per;
    int64_t r1 = r0 + per;
    if (r0 > R) {
        r0 = R;
    }
    if (r1 > R) {
        r1 = R;
    }
    if (bs < 1) {
        bs = 1;
    }

    GlobalTensor<T> xG;
    GlobalTensor<TI> tG;
    GlobalTensor<T> oG;
    GlobalTensor<float> pG;
    xG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xAddr));
    tG.SetGlobalBuffer(reinterpret_cast<__gm__ TI*>(tAddr));
    oG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outAddr));
    pG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partAddr));

    const int32_t c32 = static_cast<int32_t>(C);
    const int64_t rp = CeAlign(C, 64);
    const int64_t rowBytesT = C * static_cast<int64_t>(sizeof(T));
    int64_t padBlk = (rp * static_cast<int64_t>(sizeof(T)) - CeAlign(rowBytesT, 32)) / 32;
    if (padBlk < 0) {
        padBlk = 0;
    }
    const bool hasIgnore = (ignoreIdx >= 0 && ignoreIdx < C);
    const int64_t blkBytes = bs * rp * static_cast<int64_t>(sizeof(T));

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qIn;
    TQue<QuePosition::VECIN, 1> qT;
    TQue<QuePosition::VECOUT, 1> qO;
    TBuf<TPosition::VECCALC> bTmp, bMx, bSm, bSlot, bF;
    TBuf<TPosition::VECCALC> bMxA, bSmA, bLA, bMB, bXF, bXC, bOC, bVldC, bVldP, bMskR, bRamp, bRsum;

    pipe.InitBuffer(qIn, 1, static_cast<uint32_t>(CeAlign(blkBytes, 32) + 32));
    pipe.InitBuffer(qT, 1, static_cast<uint32_t>(CeAlign(bs * static_cast<int64_t>(sizeof(TI)), 32) + 32));
    pipe.InitBuffer(qO, 1, 512);
    pipe.InitBuffer(bTmp, kTmpFloats * 4);
    pipe.InitBuffer(bMx, 256);
    pipe.InitBuffer(bSm, 256);
    pipe.InitBuffer(bSlot, 512);
    if constexpr (!CeIsF32<T>::value) {
        pipe.InitBuffer(bF, static_cast<uint32_t>(CeAlign(bs * rp * 4, 32) + 32));
    }
    // Scratch for the vectorised reduced path: two per-row 8-float pitched
    // arrays, the per-row loss, the broadcast block, the compact target
    // vectors and the gather offsets.
    const int64_t nA = CeAlign(bs, 64);
    const int64_t nP8 = bs * 8;
    pipe.InitBuffer(bSmA, static_cast<uint32_t>(nP8 * 4 + 32));
    pipe.InitBuffer(bLA, static_cast<uint32_t>(nP8 * 4 + 32));
    pipe.InitBuffer(bMB, 512);
    pipe.InitBuffer(bXF, static_cast<uint32_t>(nA * 4 + 32));
    pipe.InitBuffer(bXC, static_cast<uint32_t>(nA * 4 + 32));
    pipe.InitBuffer(bOC, static_cast<uint32_t>(nA * 4 + 32));
    pipe.InitBuffer(bVldC, static_cast<uint32_t>(nA * 4 + 32));
    pipe.InitBuffer(bVldP, static_cast<uint32_t>(CeAlign(bs, 8) * 8 * 4 + 32));
    pipe.InitBuffer(bMskR, static_cast<uint32_t>(nA / 8 + 64));
    pipe.InitBuffer(bRamp, static_cast<uint32_t>(nA * 4 + 32));
    pipe.InitBuffer(bRsum, 64);
    if (redMode != 0) {
        pipe.InitBuffer(bMxA, static_cast<uint32_t>(nP8 * 4 + 32));
    } else {
        pipe.InitBuffer(bMxA, 512);
    }

    if (redMode != 0) {
        // ramp[r] = r * rp * sizeof(float): byte offset of the start of row r in
        // the fp32 row tile.  mxA holds per-row maxima at an 8-float pitch (the
        // reduce destination must be 32B aligned) with the padding slots at 0,
        // smA likewise holds the per-row exp sums with the padding slots at 1.0
        // so that Ln(smA) leaves exactly 0 there.
        LocalTensor<float> ramp = bRamp.Get<float>();
        const uint32_t nAc = static_cast<uint32_t>(nA);
        CreateVecIndex(ramp, 0.0f, nAc);
        Muls(ramp, ramp, static_cast<float>(rp * 4), nAc);
        LocalTensor<float> mxA = bMxA.Get<float>();
        Duplicate(mxA, 0.0f, static_cast<uint32_t>(nP8));
        LocalTensor<float> smA2 = bSmA.Get<float>();
        Duplicate(smA2, 1.0f, static_cast<uint32_t>(nP8));
    }

    float accSum = 0.0f;
    float accCnt = 0.0f;
    int64_t rr = r0;

    while (rr < r1) {
        int64_t nRows = CeMinI(bs, r1 - rr);

        LocalTensor<T> raw = qIn.AllocTensor<T>();
        DataCopyExtParams cp;
        cp.blockCount = static_cast<uint16_t>(nRows);
        cp.blockLen = static_cast<uint32_t>(rowBytesT);
        cp.srcStride = 0;
        cp.dstStride = static_cast<uint32_t>(padBlk);
        cp.rsv = 0;
        DataCopyPadExtParams<T> padx{false, 0, 0, 0};
        DataCopyPad(raw, xG[rr * C], cp, padx);
        qIn.EnQue(raw);
        raw = qIn.DeQue<T>();

        LocalTensor<TI> tRaw = qT.AllocTensor<TI>();
        DataCopyExtParams cpt;
        cpt.blockCount = 1;
        cpt.blockLen = static_cast<uint32_t>(nRows * static_cast<int64_t>(sizeof(TI)));
        cpt.srcStride = 0;
        cpt.dstStride = 0;
        cpt.rsv = 0;
        DataCopyPadExtParams<TI> padt{false, 0, 0, 0};
        DataCopyPad(tRaw, tG[rr], cpt, padt);
        qT.EnQue(tRaw);
        tRaw = qT.DeQue<TI>();

        LocalTensor<float> fRows;
        if constexpr (CeIsF32<T>::value) {
            fRows = raw.template ReinterpretCast<float>();
        } else {
            LocalTensor<float> fb = bF.Get<float>();
            CeCastChunksUp<T>(fb, raw, static_cast<int32_t>(nRows * (rp / 64)));
            fRows = fb;
        }

        LocalTensor<float> slot = bSlot.Get<float>();

        if (redMode == 0) {
            LocalTensor<float> mx = bMx.Get<float>();
            LocalTensor<float> sm = bSm.Get<float>();
            for (int64_t j = 0; j < nRows; ++j) {
                LocalTensor<float> rowF = fRows[j * rp];
                ReduceMax<float>(mx, rowF, bTmp.Get<float>(), c32, false);
                const float mv = mx.GetValue(0);
                const int32_t tv0 = static_cast<int32_t>(tRaw.GetValue(static_cast<uint32_t>(j)));
                const int32_t tv = (tv0 < 0) ? 0 : ((tv0 >= c32) ? (c32 - 1) : tv0);
                const float xt = rowF.GetValue(static_cast<uint32_t>(tv));
                Adds(rowF, rowF, -mv, c32);
                Exp(rowF, rowF, c32);
                ReduceSum<float>(sm, rowF, bTmp.Get<float>(), c32);
                Ln(sm, sm, 8);
                const float loss = mv + sm.GetValue(0) - xt;
                DataCopyExtParams co;
                co.blockCount = 1;
                co.blockLen = static_cast<uint32_t>(sizeof(T));
                co.srcStride = 0;
                co.dstStride = 0;
                co.rsv = 0;
                if constexpr (CeIsF32<T>::value) {
                    LocalTensor<float> ot = qO.AllocTensor<float>();
                    Duplicate(ot, loss, 8);
                    qO.EnQue(ot);
                    ot = qO.DeQue<float>();
                    DataCopyPad(oG[rr + j], ot, co);
                    qO.FreeTensor(ot);
                } else {
                    Duplicate(slot, loss, 8);
                    LocalTensor<T> ot = qO.AllocTensor<T>();
                    Cast(ot, slot, RoundMode::CAST_RINT, static_cast<uint32_t>(64));
                    qO.EnQue(ot);
                    ot = qO.DeQue<T>();
                    DataCopyPad(oG[rr + j], ot, co);
                    qO.FreeTensor(ot);
                }
            }
        } else {
            // Reduced mean/sum path.  No scalar is read back per row: the row
            // maxima and exp sums land in 8-float pitched arrays, the max shift
            // is broadcast with Brcb + a stride-0 high dimensional Sub, and the
            // captured logits x[target] come from one Gather for the whole tile.
            LocalTensor<float> mxA = bMxA.Get<float>();
            LocalTensor<float> smA = bSmA.Get<float>();
            LocalTensor<float> lossA = bLA.Get<float>();
            LocalTensor<float> bcast = bMB.Get<float>();
            LocalTensor<float> xf = bXF.Get<float>();
            LocalTensor<float> xc = bXC.Get<float>();
            LocalTensor<float> vldC = bVldC.Get<float>();
            LocalTensor<float> vldP = bVldP.Get<float>();
            LocalTensor<int32_t> oc = bOC.Get<int32_t>();
            LocalTensor<uint8_t> mskR = bMskR.Get<uint8_t>();
            LocalTensor<float> rampL = bRamp.Get<float>();
            LocalTensor<float> rsum = bRsum.Get<float>();
            LocalTensor<float> tmpR = bTmp.Get<float>();
            const uint32_t nAc = static_cast<uint32_t>(nA);
            const uint32_t nPc = static_cast<uint32_t>(nRows * 8);

            // x[target] for every row of the tile, before the rows are shifted:
            // byte offset = row*rp*4 + target*4, target clamped into [0, C-1] so
            // that no offset can leave the row tile.
            if constexpr (std::is_same<TI, int32_t>::value) {
                Cast(xf, tRaw, RoundMode::CAST_NONE, nAc);
            } else {
                Cast(xf, tRaw, RoundMode::CAST_RINT, nAc);
            }
            if (hasIgnore) {
                Compares<float, uint8_t>(mskR, xf, static_cast<float>(ignoreIdx), CMPMODE::EQ, nAc);
                Duplicate(vldC, 0.0f, nAc);
                Select(vldC, mskR, vldC, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, nAc);
            }
            Maxs(xf, xf, 0.0f, nAc);
            Mins(xf, xf, static_cast<float>(c32 - 1), nAc);
            Muls(xf, xf, 4.0f, nAc);
            Add(xf, xf, rampL, nAc);
            // Lanes at or beyond nRows hold an unclamped row index (the ramp is
            // built for the whole buffered block), so bound the byte offset by
            // the last element of the tile: gathered garbage can never leave it.
            Mins(xf, xf, static_cast<float>((nRows - 1) * rp * 4 + (c32 - 1) * 4), nAc);
            Cast(oc, xf, RoundMode::CAST_RINT, nAc);
            Gather(xc, fRows, oc.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), nAc);

            for (int64_t j = 0; j < nRows; ++j) {
                LocalTensor<float> rowF = fRows[j * rp];
                ReduceMax<float>(mxA[j * 8], rowF, tmpR, c32, false);
                // one 32B block holding the row maximum eight times: every
                // repeat of the Sub reads that same block (src1 stride 0).
                Brcb(bcast, mxA[j * 8], 1, {1, 8});
                int64_t off = 0;
                int64_t rem = rp;
                while (rem > 0) {
                    const int64_t seg = (rem > 16320) ? 16320 : rem;
                    Sub(rowF[off], rowF[off], bcast, static_cast<uint64_t>(64),
                        static_cast<uint8_t>(seg >> 6), {1, 1, 0, 8, 8, 0});
                    off += seg;
                    rem -= seg;
                }
                Exp(rowF, rowF, static_cast<uint32_t>(rp));
                ReduceSum<float>(smA[j * 8], rowF, tmpR, c32);
            }

            // loss = max + ln(exp sum) - x[target]; the padding slots of smA hold
            // 1.0 (Ln -> 0) and those of mxA hold 0, so the pitched sums below
            // equal the exact per-row totals.
            Ln(lossA, smA, nPc);
            Add(lossA, lossA, mxA, nPc);
            if (hasIgnore) {
                Brcb(vldP, vldC, static_cast<uint8_t>(CeAlign(nRows, 8) >> 3), {1, 8});
                Mul(lossA, lossA, vldP, nPc);
                Mul(xc, xc, vldC, nAc);
            }
            ReduceSum<float>(rsum, lossA, tmpR, static_cast<int32_t>(nPc));
            const float partM = rsum.GetValue(0);
            ReduceSum<float>(rsum, xc, tmpR, static_cast<int32_t>(nRows));
            accSum += partM - rsum.GetValue(0);
            if (redMode == 1) {
                if (hasIgnore) {
                    ReduceSum<float>(rsum, vldC, tmpR, static_cast<int32_t>(nRows));
                    accCnt += rsum.GetValue(0);
                } else {
                    accCnt += static_cast<float>(nRows);
                }
            }
        }

        qT.FreeTensor(tRaw);
        qIn.FreeTensor(raw);
        rr += nRows;
    }

    if (redMode != 0) {
        LocalTensor<float> ot = qO.AllocTensor<float>();
        Duplicate(ot, accSum, 8);
        qO.EnQue(ot);
        ot = qO.DeQue<float>();
        DataCopyExtParams co;
        co.blockCount = 1;
        co.blockLen = 4;
        co.srcStride = 0;
        co.dstStride = 0;
        co.rsv = 0;
        DataCopyPad(pG[blk], ot, co);
        qO.FreeTensor(ot);

        LocalTensor<float> oc = qO.AllocTensor<float>();
        Duplicate(oc, accCnt, 8);
        qO.EnQue(oc);
        oc = qO.DeQue<float>();
        DataCopyPad(pG[numBlocks + blk], oc, co);
        qO.FreeTensor(oc);
    }
}

/*!
 * \brief store one per-lane plane of the class-split partials (comp 0: running max,
 *        1: sum of exp, 2: captured target value, 3: valid-lane flag).
 *        The plane layout is [comp][cs][lane] so that a whole lane tile of one
 *        component is contiguous.
 */
__aicore__ inline void CePlaneStore(const GlobalTensor<float>& lpG, const LocalTensor<float>& src, int64_t base,
                                    int64_t w, int64_t comp, int64_t nsplit, int64_t cs, int64_t lanes,
                                    TQue<QuePosition::VECOUT, 1>& qO)
{
    LocalTensor<float> ot = qO.AllocTensor<float>();
    Adds(ot, src, 0.0f, static_cast<uint32_t>(w));
    qO.EnQue(ot);
    ot = qO.DeQue<float>();
    DataCopyExtParams cw;
    cw.blockCount = 1;
    cw.blockLen = static_cast<uint32_t>(w * 4);
    cw.srcStride = 0;
    cw.dstStride = 0;
    cw.rsv = 0;
    DataCopyPad(lpG[(comp * nsplit + cs) * lanes + base], ot, cw);
    qO.FreeTensor(ot);
}

/* =====================================================================================
 *  P > 1 : column kernel.  Reduce axis has stride P; work on tiles of W columns.
 * ===================================================================================== */
template <typename T, typename TI>
__global__ __aicore__ void ce_col_kernel(GM_ADDR xAddr, GM_ADDR tAddr, GM_ADDR outAddr, GM_ADDR partAddr,
                                         GM_ADDR lpAddr, int64_t N, int64_t C, int64_t P, int64_t ignoreIdx,
                                         int64_t redMode, int64_t numBlocks, int64_t W, int64_t cb,
                                         int64_t nsplit)
{
    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    const int64_t tiles = (P + W - 1) / W;
    const int64_t groups = N * tiles;
    if (nsplit < 1) {
        nsplit = 1;
    }
    const int64_t tasks = groups * nsplit;
    int64_t per = (tasks + numBlocks - 1) / numBlocks;
    if (per < 1) {
        per = 1;
    }
    int64_t t0 = blk * per;
    int64_t t1 = t0 + per;
    if (t0 > tasks) {
        t0 = tasks;
    }
    if (t1 > tasks) {
        t1 = tasks;
    }
    if (cb < 1) {
        cb = 1;
    }

    GlobalTensor<T> xG;
    GlobalTensor<TI> tG;
    GlobalTensor<T> oG;
    GlobalTensor<float> pG;
    GlobalTensor<float> lpG;
    xG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xAddr));
    tG.SetGlobalBuffer(reinterpret_cast<__gm__ TI*>(tAddr));
    oG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outAddr));
    pG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partAddr));
    lpG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lpAddr));

    const int64_t lanes = groups * W;
    const int64_t cStep = (C + nsplit - 1) / nsplit;

    const int64_t Wbuf = CeAlign(W, 64);
    const int64_t pitchf = Wbuf;                                   // f32 pitch (32B aligned)
    const int64_t pitchT = CeAlign(Wbuf * static_cast<int64_t>(sizeof(T)), 32) / static_cast<int64_t>(sizeof(T));
    const bool hasIgnore = (ignoreIdx >= 0 && ignoreIdx < C);
    const bool isF = CeIsF32<T>::value;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qIn;
    TQue<QuePosition::VECIN, 1> qT;
    TQue<QuePosition::VECOUT, 1> qO;
    TBuf<TPosition::VECCALC> bF, bT32, bMsk, bTmp;
    TBuf<TPosition::VECCALC> bAccM, bAccS, bCap, bCm, bCn, bT1, bLoss;

    if constexpr (!CeIsF32<T>::value) {
        pipe.InitBuffer(bF, static_cast<uint32_t>(CeAlign(cb * pitchf * 4, 32) + 32));
    }
    pipe.InitBuffer(qIn, 1, static_cast<uint32_t>(CeAlign(cb * pitchT * static_cast<int64_t>(sizeof(T)), 32) + 32));
    pipe.InitBuffer(qT, 1, static_cast<uint32_t>(CeAlign(Wbuf * static_cast<int64_t>(sizeof(TI)), 32) + 128));
    pipe.InitBuffer(qO, 1, static_cast<uint32_t>(CeAlign(Wbuf * 4, 32) + 128));
    pipe.InitBuffer(bT32, static_cast<uint32_t>(CeAlign(Wbuf * 4, 32) + 128));
    pipe.InitBuffer(bMsk, 256);
    pipe.InitBuffer(bTmp, kTmpFloats * 4);
    pipe.InitBuffer(bAccM, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bAccS, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bCap, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bCm, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bCn, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bT1, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bLoss, static_cast<uint32_t>(Wbuf * 4 + 32));

    LocalTensor<float> accM = bAccM.Get<float>();
    LocalTensor<float> accS = bAccS.Get<float>();
    LocalTensor<float> cap = bCap.Get<float>();
    LocalTensor<float> cm = bCm.Get<float>();
    LocalTensor<float> cn = bCn.Get<float>();
    LocalTensor<float> tv = bT1.Get<float>();
    LocalTensor<float> lossF = bLoss.Get<float>();
    LocalTensor<int32_t> t32 = bT32.Get<int32_t>();
    LocalTensor<uint8_t> msk = bMsk.Get<uint8_t>();

    float accSum = 0.0f;
    float accCnt = 0.0f;

    for (int64_t tk = t0; tk < t1; ++tk) {
        const int64_t cs = tk % nsplit;
        const int64_t g = tk / nsplit;
        const int64_t n = g / tiles;
        const int64_t tIdx = g % tiles;
        const int64_t p0 = tIdx * W;
        int64_t cBeg = cs * cStep;
        if (cBeg > C) {
            cBeg = C;
        }
        int64_t cEnd = cBeg + cStep;
        if (cEnd > C) {
            cEnd = C;
        }
        int64_t w = W;
        if (w > P - p0) {
            w = P - p0;
        }
        int64_t wc = CeAlign(w, 64);
        if (wc > Wbuf) {
            wc = Wbuf;
        }
        const int32_t w32 = static_cast<int32_t>(w);
        const int32_t wc32 = static_cast<int32_t>(wc);

        DataCopyExtParams cpt;
        cpt.blockCount = 1;
        cpt.blockLen = static_cast<uint32_t>(w * static_cast<int64_t>(sizeof(TI)));
        cpt.srcStride = 0;
        cpt.dstStride = 0;
        cpt.rsv = 0;
        DataCopyPadExtParams<TI> padt{false, 0, 0, 0};
        LocalTensor<TI> tRaw = qT.AllocTensor<TI>();
        DataCopyPad(tRaw, tG[n * P + p0], cpt, padt);
        qT.EnQue(tRaw);
        tRaw = qT.DeQue<TI>();
        if constexpr (std::is_same<TI, int32_t>::value) {
            Adds(t32, tRaw, static_cast<int32_t>(0), wc32);
        } else {
            Cast(t32, tRaw, RoundMode::CAST_NONE, static_cast<uint32_t>(wc32));
        }
        qT.FreeTensor(tRaw);

        Duplicate(accM, kNegBig, wc32);
        Duplicate(accS, 0.0f, wc32);
        Duplicate(cap, 0.0f, wc32);

        for (int64_t c0 = cBeg; c0 < cEnd; c0 += cb) {
            const int64_t cbN = CeMinI(cb, cEnd - c0);

            LocalTensor<T> raw = qIn.AllocTensor<T>();
            DataCopyExtParams cpx;
            cpx.blockCount = static_cast<uint16_t>(cbN);
            cpx.blockLen = static_cast<uint32_t>(w * static_cast<int64_t>(sizeof(T)));
            cpx.srcStride = static_cast<uint32_t>((P - w) * static_cast<int64_t>(sizeof(T)));
            cpx.dstStride = static_cast<uint32_t>(
                (pitchT * static_cast<int64_t>(sizeof(T)) - CeAlign(w * static_cast<int64_t>(sizeof(T)), 32)) / 32);
            cpx.rsv = 0;
            DataCopyPadExtParams<T> padx{false, 0, 0, 0};
            DataCopyPad(raw, xG[n * C * P + c0 * P + p0], cpx, padx);
            qIn.EnQue(raw);
            raw = qIn.DeQue<T>();

            LocalTensor<float> fChunk;
            if constexpr (CeIsF32<T>::value) {
                fChunk = raw.template ReinterpretCast<float>();
            } else {
                LocalTensor<float> fb = bF.Get<float>();
                CeCastChunksUp<T>(fb, raw, static_cast<int32_t>(cbN * (pitchT / 64)));
                fChunk = fb;
            }

            Duplicate(cm, kNegBig, wc32);
            for (int64_t r = 0; r < cbN; ++r) {
                const int32_t ccl = static_cast<int32_t>(c0 + r);
                LocalTensor<float> rowF = fChunk[r * pitchf];
                Max(cm, cm, rowF, w32);
                Compares<int32_t, uint8_t>(msk, t32, ccl, CMPMODE::EQ, static_cast<uint32_t>(wc32));
                Select(cap, msk, rowF, cap, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(wc32));
            }
            Max(cn, accM, cm, w32);
            Sub(tv, accM, cn, w32);
            Exp(tv, tv, w32);
            Mul(accS, accS, tv, w32);
            for (int64_t r = 0; r < cbN; ++r) {
                LocalTensor<float> rowF = fChunk[r * pitchf];
                Sub(tv, rowF, cn, w32);
                Exp(tv, tv, w32);
                Add(accS, accS, tv, w32);
            }
            Adds(accM, cn, 0.0f, w32);

            qIn.FreeTensor(raw);
        }

        if (nsplit > 1) {
            // Class axis split: publish the per-lane partials so that the merge kernel can fold the
            // log-sum-exp of this class slice together with the other slices.
            if (hasIgnore) {
                Compares<int32_t, uint8_t>(msk, t32, static_cast<int32_t>(ignoreIdx), CMPMODE::EQ,
                                           static_cast<uint32_t>(wc32));
                Duplicate(cn, 0.0f, wc32);
                Select(tv, msk, cn, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, static_cast<uint32_t>(wc32));
            } else {
                Duplicate(tv, 1.0f, wc32);
            }
            const int64_t lbase = g * W;
            CePlaneStore(lpG, accM, lbase, w, 0, nsplit, cs, lanes, qO);
            CePlaneStore(lpG, accS, lbase, w, 1, nsplit, cs, lanes, qO);
            CePlaneStore(lpG, cap, lbase, w, 2, nsplit, cs, lanes, qO);
            CePlaneStore(lpG, tv, lbase, w, 3, nsplit, cs, lanes, qO);
        } else {
        Ln(accS, accS, w32);
        Sub(lossF, accM, cap, w32);
        Add(lossF, lossF, accS, w32);

        // ignore_index handling.  On Atlas A2 `Compares` with an int32 source only
        // supports CMPMODE::EQ, so the mask marks the IGNORED lanes and both
        // consumers invert it inline (mode-2 Select picks 0 for masked lanes, and
        // mode-1 Select picks 1.0f for the unmasked ones).
        LocalTensor<float> lossOut = lossF;
        if (hasIgnore) {
            Compares<int32_t, uint8_t>(msk, t32, static_cast<int32_t>(ignoreIdx), CMPMODE::EQ,
                                       static_cast<uint32_t>(wc32));
            Duplicate(cn, 0.0f, wc32);
            Select(tv, msk, cn, lossF, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(wc32));
            lossOut = tv;
        }

        if (redMode == 0) {
            LocalTensor<T> ot = qO.AllocTensor<T>();
            if constexpr (CeIsF32<T>::value) {
                LocalTensor<float> otf = ot.template ReinterpretCast<float>();
                Adds(otf, lossOut, 0.0f, w32);
                qO.EnQue(ot);
                ot = qO.DeQue<T>();
                DataCopyExtParams co;
                co.blockCount = 1;
                co.blockLen = static_cast<uint32_t>(w * static_cast<int64_t>(sizeof(T)));
                co.srcStride = 0;
                co.dstStride = 0;
                co.rsv = 0;
                DataCopyPad(oG[n * P + p0], ot, co);
            } else {
                Cast(ot, lossOut, RoundMode::CAST_RINT, static_cast<uint32_t>(wc32));
                qO.EnQue(ot);
                ot = qO.DeQue<T>();
                DataCopyExtParams co;
                co.blockCount = 1;
                co.blockLen = static_cast<uint32_t>(w * static_cast<int64_t>(sizeof(T)));
                co.srcStride = 0;
                co.dstStride = 0;
                co.rsv = 0;
                DataCopyPad(oG[n * P + p0], ot, co);
            }
            qO.FreeTensor(ot);
        } else {
            LocalTensor<float> sSum = bTmp.Get<float>();
            ReduceSum<float>(sSum, lossOut, bTmp.Get<float>(), w32);
            accSum += sSum.GetValue(0);
            if (redMode == 1) {
                if (hasIgnore) {
                    Duplicate(cm, 0.0f, wc32);
                    Select(cm, msk, cm, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, static_cast<uint32_t>(wc32));
                    ReduceSum<float>(sSum, cm, bTmp.Get<float>(), w32);
                    accCnt += sSum.GetValue(0);
                } else {
                    accCnt += static_cast<float>(w);
                }
            }
        }
        }
    }

    if (redMode != 0 && nsplit <= 1) {
        LocalTensor<float> ot = qO.AllocTensor<float>();
        Duplicate(ot, accSum, 8);
        qO.EnQue(ot);
        ot = qO.DeQue<float>();
        DataCopyExtParams co;
        co.blockCount = 1;
        co.blockLen = 4;
        co.srcStride = 0;
        co.dstStride = 0;
        co.rsv = 0;
        DataCopyPad(pG[blk], ot, co);
        qO.FreeTensor(ot);

        LocalTensor<float> oc = qO.AllocTensor<float>();
        Duplicate(oc, accCnt, 8);
        qO.EnQue(oc);
        oc = qO.DeQue<float>();
        DataCopyPad(pG[numBlocks + blk], oc, co);
        qO.FreeTensor(oc);
    }
}

/* =====================================================================================
 *  merge: fold the class-axis partials of one lane tile back into per-lane losses.
 * ===================================================================================== */
template <typename T>
__global__ __aicore__ void ce_merge_kernel(GM_ADDR lpAddr, GM_ADDR outAddr, GM_ADDR partAddr, int64_t N, int64_t C,
                                           int64_t P, int64_t redMode, int64_t numBlocks, int64_t W,
                                           int64_t nsplit)
{
    (void)C;
    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    const int64_t tiles = (P + W - 1) / W;
    const int64_t groups = N * tiles;
    int64_t per = (groups + numBlocks - 1) / numBlocks;
    if (per < 1) {
        per = 1;
    }
    int64_t g0 = blk * per;
    int64_t g1 = g0 + per;
    if (g0 > groups) {
        g0 = groups;
    }
    if (g1 > groups) {
        g1 = groups;
    }
    if (nsplit < 1) {
        nsplit = 1;
    }

    GlobalTensor<float> lpG;
    GlobalTensor<T> oG;
    GlobalTensor<float> pG;
    lpG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lpAddr));
    oG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outAddr));
    pG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partAddr));

    const int64_t Wbuf = CeAlign(W, 64);
    const int64_t lanes = groups * W;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qL;
    TQue<QuePosition::VECOUT, 1> qO;
    TBuf<TPosition::VECCALC> bAcc, bS, bC, bK, bT, bV, bTmp;
    pipe.InitBuffer(qL, 1, static_cast<uint32_t>(4 * Wbuf * 4 + 128));
    pipe.InitBuffer(qO, 1, static_cast<uint32_t>(Wbuf * 4 + 128));
    pipe.InitBuffer(bAcc, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bS, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bC, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bK, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bT, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bV, static_cast<uint32_t>(Wbuf * 4 + 32));
    pipe.InitBuffer(bTmp, kTmpFloats * 4);

    LocalTensor<float> accM = bAcc.Get<float>();
    LocalTensor<float> sum = bS.Get<float>();
    LocalTensor<float> capS = bC.Get<float>();
    LocalTensor<float> tmp = bK.Get<float>();
    LocalTensor<float> res = bT.Get<float>();
    LocalTensor<float> vld = bV.Get<float>();

    float accSum = 0.0f;
    float accCnt = 0.0f;

    for (int64_t g = g0; g < g1; ++g) {
        const int64_t n = g / tiles;
        const int64_t tIdx = g % tiles;
        const int64_t p0 = tIdx * W;
        int64_t w = W;
        if (w > P - p0) {
            w = P - p0;
        }
        const int32_t w32 = static_cast<int32_t>(w);
        const int64_t base = g * W;

        for (int64_t cs = 0; cs < nsplit; ++cs) {
            LocalTensor<float> blk4 = qL.AllocTensor<float>();
            DataCopyExtParams cl;
            cl.blockCount = 4;
            cl.blockLen = static_cast<uint32_t>(w * 4);
            // the four components of one class slice live nsplit*lanes apart, so the
            // source stride is that gap (in bytes) minus the block length
            cl.srcStride = static_cast<uint32_t>((nsplit * lanes - w) * 4);
            cl.dstStride = 0;
            cl.rsv = 0;
            DataCopyPadExtParams<float> pp{false, 0, 0, 0};
            DataCopyPad(blk4, lpG[cs * lanes + base], cl, pp);
            qL.EnQue(blk4);
            blk4 = qL.DeQue<float>();
            const int64_t ptu = CeAlign(w * 4, 32) / 4;
            LocalTensor<float> pm = blk4[0];
            LocalTensor<float> ps = blk4[ptu];
            LocalTensor<float> pc = blk4[2 * ptu];
            LocalTensor<float> pv = blk4[3 * ptu];
            if (cs == 0) {
                Adds(accM, pm, 0.0f, w32);
                Adds(sum, ps, 0.0f, w32);
                Adds(capS, pc, 0.0f, w32);
                Adds(vld, pv, 0.0f, w32);
            } else {
                Max(res, accM, pm, w32);                                  // new running max
                Sub(tmp, accM, res, w32);                                 // rescale the accrued sum
                Exp(tmp, tmp, w32);
                Mul(sum, sum, tmp, w32);
                Sub(tmp, pm, res, w32);
                Exp(tmp, tmp, w32);
                Mul(tmp, tmp, ps, w32);
                Add(sum, sum, tmp, w32);
                Adds(accM, res, 0.0f, w32);
                Add(capS, capS, pc, w32);
            }
            qL.FreeTensor(blk4);
        }

        Ln(res, sum, w32);
        Add(res, res, accM, w32);
        Sub(res, res, capS, w32);
        Mul(res, res, vld, w32);                                          // drop the ignore_index lanes

        if (redMode == 0) {
            LocalTensor<T> ot = qO.AllocTensor<T>();
            if constexpr (CeIsF32<T>::value) {
                LocalTensor<float> otf = ot.template ReinterpretCast<float>();
                Adds(otf, res, 0.0f, w32);
            } else {
                Cast(ot, res, RoundMode::CAST_RINT, static_cast<uint32_t>(CeAlign(w, 64)));
            }
            qO.EnQue(ot);
            ot = qO.DeQue<T>();
            DataCopyExtParams co;
            co.blockCount = 1;
            co.blockLen = static_cast<uint32_t>(w * static_cast<int64_t>(sizeof(T)));
            co.srcStride = 0;
            co.dstStride = 0;
            co.rsv = 0;
            DataCopyPad(oG[n * P + p0], ot, co);
            qO.FreeTensor(ot);
        } else {
            LocalTensor<float> sSum = bTmp.Get<float>();
            ReduceSum<float>(sSum, res, bTmp.Get<float>(), w32);
            accSum += sSum.GetValue(0);
            if (redMode == 1) {
                ReduceSum<float>(sSum, vld, bTmp.Get<float>(), w32);
                accCnt += sSum.GetValue(0);
            }
        }
    }

    if (redMode != 0) {
        LocalTensor<float> ot = qO.AllocTensor<float>();
        Duplicate(ot, accSum, 8);
        qO.EnQue(ot);
        ot = qO.DeQue<float>();
        DataCopyExtParams co;
        co.blockCount = 1;
        co.blockLen = 4;
        co.srcStride = 0;
        co.dstStride = 0;
        co.rsv = 0;
        DataCopyPad(pG[blk], ot, co);
        qO.FreeTensor(ot);

        LocalTensor<float> oc = qO.AllocTensor<float>();
        Duplicate(oc, accCnt, 8);
        qO.EnQue(oc);
        oc = qO.DeQue<float>();
        DataCopyPad(pG[numBlocks + blk], oc, co);
        qO.FreeTensor(oc);
    }
}

/* =====================================================================================
 *  finalize: fold per-block partials into the scalar output.
 * ===================================================================================== */
template <typename T>
__global__ __aicore__ void ce_final_kernel(GM_ADDR partAddr, GM_ADDR outAddr, int64_t numBlocks, int64_t redMode)
{
    GlobalTensor<float> pG;
    GlobalTensor<T> oG;
    pG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partAddr));
    oG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outAddr));

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qP;
    TQue<QuePosition::VECOUT, 1> qO;
    TBuf<TPosition::VECCALC> bSlot;
    pipe.InitBuffer(qP, 1, 4096);
    pipe.InitBuffer(qO, 1, 512);
    pipe.InitBuffer(bSlot, 512);

    LocalTensor<float> pv = qP.AllocTensor<float>();
    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint32_t>(numBlocks * 2 * 4);
    cp.srcStride = 0;
    cp.dstStride = 0;
    cp.rsv = 0;
    DataCopyPadExtParams<float> pp{false, 0, 0, 0};
    DataCopyPad(pv, pG, cp, pp);
    qP.EnQue(pv);
    pv = qP.DeQue<float>();

    float s = 0.0f;
    float c = 0.0f;
    for (int64_t i = 0; i < numBlocks; ++i) {
        s += pv.GetValue(static_cast<uint32_t>(i));
        c += pv.GetValue(static_cast<uint32_t>(numBlocks + i));
    }
    qP.FreeTensor(pv);

    const float res = (redMode == 1) ? (s / c) : s;
    LocalTensor<float> slot = bSlot.Get<float>();
    Duplicate(slot, res, 8);

    if constexpr (CeIsF32<T>::value) {
        LocalTensor<float> ot = qO.AllocTensor<float>();
        Adds(ot, slot, 0.0f, 8);
        qO.EnQue(ot);
        ot = qO.DeQue<float>();
        DataCopyExtParams co;
        co.blockCount = 1;
        co.blockLen = 4;
        co.srcStride = 0;
        co.dstStride = 0;
        co.rsv = 0;
        DataCopyPad(oG, ot, co);
        qO.FreeTensor(ot);
    } else {
        LocalTensor<T> ot = qO.AllocTensor<T>();
        Cast(ot, slot, RoundMode::CAST_RINT, static_cast<uint32_t>(64));
        qO.EnQue(ot);
        ot = qO.DeQue<T>();
        DataCopyExtParams co;
        co.blockCount = 1;
        co.blockLen = static_cast<uint32_t>(sizeof(T));
        co.srcStride = 0;
        co.dstStride = 0;
        co.rsv = 0;
        DataCopyPad(oG, ot, co);
        qO.FreeTensor(ot);
    }
}

} // namespace

/* =====================================================================================
 *  host side tiling query
 * ===================================================================================== */
extern "C" void ce_query_plan(int64_t N, int64_t C, int64_t P, int64_t esz, int64_t eszT, int64_t redMode,
                              int64_t* plan)
{
    (void)redMode;
    int64_t coreNum = 48;
    uint64_t ubSize = 196608;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        int64_t cn = static_cast<int64_t>(plat->GetCoreNumAiv());
        if (cn > 0) {
            coreNum = cn;
        }
        uint64_t u = 0;
        plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, u);
        if (u >= 65536) {
            ubSize = u;
        }
    }
    // Keep headroom for the framework and for the 8KB Select scratch buffer.
    int64_t usable = static_cast<int64_t>(ubSize / 8 * 7);
    if (usable > 150 * 1024) {
        usable = 150 * 1024;
    }

    if (P <= 1) {
        const int64_t rp = (C + 63) / 64 * 64;
        int64_t perRow = rp * esz + eszT + 4;
        if (esz != 4) {
            perRow += rp * 4;
        }
        // The reduced-mean/sum path keeps per-row pitched (8-float stride) max /
        // log-sum arrays plus the compact target vectors alive; that is about
        // 160 bytes per buffered row, so account for it before choosing bs.
        perRow += 160;
        const int64_t fixed = 18 * 1024;
        int64_t budget = usable - fixed;
        if (budget < perRow) {
            budget = perRow;
        }
        int64_t bsMax = budget / perRow;
        if (bsMax < 1) {
            bsMax = 1;
        }
        // The gather offsets and the pitched reduction arrays are fp32 exact
        // only while row*rp*4 stays below 2^24, and small bs keeps those scratch
        // buffers inside the UB budget.
        if (bsMax > 256) {
            bsMax = 256;
        }
        int64_t nb = coreNum;
        if (nb > N) {
            nb = N;
        }
        if (nb < 1) {
            nb = 1;
        }
        int64_t rowsPer = (N + nb - 1) / nb;
        if (rowsPer < 1) {
            rowsPer = 1;
        }
        int64_t bs = bsMax < rowsPer ? bsMax : rowsPer;
        plan[0] = 0;
        plan[1] = nb;
        plan[2] = bs;
        plan[3] = coreNum;
        plan[4] = 0;
        plan[5] = 1;
        plan[6] = nb;
        plan[7] = 0;
        plan[8] = 0;
    } else {
        int64_t W = P < 512 ? P : 512;
        int64_t Wbuf = (W + 63) / 64 * 64;
        int64_t pitchT = (Wbuf * esz + 31) / 32 * 32 / esz;
        int64_t perRow = pitchT * esz;
        if (esz != 4) {
            perRow += Wbuf * 4;
        }
        int64_t fixed = 7 * Wbuf * 4 + 16 * 1024 + 256 + (Wbuf * 4 + 128) + (Wbuf * eszT + 128) +
                        (Wbuf * 4 + 128);
        int64_t budget = usable - fixed;
        if (budget < perRow) {
            budget = perRow;
        }
        int64_t cbMax = budget / perRow;
        if (cbMax < 1) {
            cbMax = 1;
        }
        if (cbMax > 4095) {
            cbMax = 4095;
        }
        if (cbMax > C) {
            cbMax = C;
        }
        int64_t tiles = (P + W - 1) / W;
        int64_t groups = N * tiles;
        int64_t nsplit = 1;
        if (groups < coreNum) {
            int64_t want = (coreNum + groups - 1) / groups;
            if (want > C) {
                want = C;
            }
            if (want > 1) {
                nsplit = want;
            }
        }
        int64_t tasks = groups * nsplit;
        int64_t nb = coreNum;
        if (nb > tasks) {
            nb = tasks;
        }
        if (nb < 1) {
            nb = 1;
        }
        int64_t mb = coreNum;
        if (mb > groups) {
            mb = groups;
        }
        if (mb < 1) {
            mb = 1;
        }
        const int64_t lanes = groups * W;
        plan[0] = 1;
        plan[1] = nb;
        plan[2] = W;
        plan[3] = coreNum;
        plan[4] = cbMax;
        plan[5] = nsplit;
        plan[6] = mb;
        plan[7] = lanes;
        plan[8] = 4 * nsplit * lanes;
    }
}

/* =====================================================================================
 *  launch wrappers
 * ===================================================================================== */
extern "C" {

#define CE_ROW_LAUNCH(SFX, T, TI)                                                                          \
    void launch_ce_row_##SFX(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, int64_t R, int64_t C,             \
                             int64_t ignoreIdx, int64_t redMode, int64_t numBlocks, int64_t bs,            \
                             void* stream)                                                                 \
    {                                                                                                      \
        ce_row_kernel<T, TI><<<numBlocks, nullptr, stream>>>(x, t, o, p, R, C, ignoreIdx, redMode,         \
                                                             numBlocks, bs);                               \
    }

CE_ROW_LAUNCH(float_i32, float, int32_t)
CE_ROW_LAUNCH(float_i64, float, int64_t)
CE_ROW_LAUNCH(half_i32, half, int32_t)
CE_ROW_LAUNCH(half_i64, half, int64_t)
CE_ROW_LAUNCH(bf16_i32, bfloat16_t, int32_t)
CE_ROW_LAUNCH(bf16_i64, bfloat16_t, int64_t)

#define CE_COL_LAUNCH(SFX, T, TI)                                                                          \
    void launch_ce_col_##SFX(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, GM_ADDR lp, int64_t N, int64_t C,  \
                             int64_t PP, int64_t ignoreIdx, int64_t redMode, int64_t numBlocks, int64_t W,  \
                             int64_t cb, int64_t nsplit, void* stream)                                      \
    {                                                                                                      \
        ce_col_kernel<T, TI><<<numBlocks, nullptr, stream>>>(x, t, o, p, lp, N, C, PP, ignoreIdx, redMode, \
                                                             numBlocks, W, cb, nsplit);                    \
    }

CE_COL_LAUNCH(float_i32, float, int32_t)
CE_COL_LAUNCH(float_i64, float, int64_t)
CE_COL_LAUNCH(half_i32, half, int32_t)
CE_COL_LAUNCH(half_i64, half, int64_t)
CE_COL_LAUNCH(bf16_i32, bfloat16_t, int32_t)
CE_COL_LAUNCH(bf16_i64, bfloat16_t, int64_t)

#define CE_MERGE_LAUNCH(SFX, T)                                                                            \
    void launch_ce_merge_##SFX(GM_ADDR lp, GM_ADDR o, GM_ADDR p, int64_t N, int64_t C, int64_t PP,         \
                               int64_t redMode, int64_t numBlocks, int64_t W, int64_t nsplit,             \
                               void* stream)                                                              \
    {                                                                                                      \
        ce_merge_kernel<T><<<numBlocks, nullptr, stream>>>(lp, o, p, N, C, PP, redMode, numBlocks, W,       \
                                                           nsplit);                                        \
    }

CE_MERGE_LAUNCH(float, float)
CE_MERGE_LAUNCH(half, half)
CE_MERGE_LAUNCH(bf16, bfloat16_t)

void launch_ce_final_float(GM_ADDR part, GM_ADDR o, int64_t numBlocks, int64_t redMode, void* stream)
{
    ce_final_kernel<float><<<1, nullptr, stream>>>(part, o, numBlocks, redMode);
}
void launch_ce_final_half(GM_ADDR part, GM_ADDR o, int64_t numBlocks, int64_t redMode, void* stream)
{
    ce_final_kernel<half><<<1, nullptr, stream>>>(part, o, numBlocks, redMode);
}
void launch_ce_final_bf16(GM_ADDR part, GM_ADDR o, int64_t numBlocks, int64_t redMode, void* stream)
{
    ce_final_kernel<bfloat16_t><<<1, nullptr, stream>>>(part, o, numBlocks, redMode);
}
}
