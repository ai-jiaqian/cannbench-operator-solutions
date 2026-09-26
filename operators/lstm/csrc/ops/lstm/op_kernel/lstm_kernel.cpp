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
 * \file lstm_kernel.cpp
 * \brief LSTM device kernels + host launch wrappers (bisheng + -xasc, dav-2201).
 *
 * Two kernels on the caller's stream:
 *   1) lstm_prep_kernel<T> : cast x / weights / biases / h0 / c0 from dtype T into an fp32 workspace whose
 *                            rows are exactly the padded widths the recurrence consumes.  Each row is
 *                            either real (cast + right padded to the full destination width) or an
 *                            explicit zero row, so no workspace byte is left to chance.
 *   2) lstm_rec_kernel<T>  : fp32 recurrence for one layer.  Grid = B * D (one core per batch element and
 *                            direction) so there is no inter core synchronisation at all; the layer input
 *                            of layer l > 0 comes from the ping-pong sequence buffers written by the
 *                            launch for layer l - 1.  The kernel also emits y (last layer), hn and cn.
 *
 * The gate linear transform is a strip accumulate: for a 64 row weight block,
 *   acc[64][64] += W[64][k0..k0+64) .* broadcast(v[k0..k0+64))
 * with MulAddDst (src1RepStride = 0 broadcasts the 64 element vector slice over the 64 rows), followed by
 * WholeReduceSum which turns each 64 element row of acc into one dot product written contiguously.
 */

#define LSTM_DEVICE_TU 1

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_binary_intf.h"
#include "basic_api/kernel_operator_vec_reduce_intf.h"
#include "platform/platform_ascendc.h"
#include "adv_api/activation/sigmoid.h"
#include "adv_api/math/tanh.h"

#include "lstm_launch.h"

using namespace AscendC;

namespace {

constexpr int32_t LSTM_TILE_ROWS = 128;                       // gate rows per matvec / reduction block
constexpr int32_t LSTM_PR_ROWS = 64;                          // rows per LSTMP projection block
constexpr int32_t LSTM_TILE_ELEMS = LSTM_TILE_ROWS * 64;      // acc / weight tile elements
constexpr uint32_t LSTM_TILE_BYTES = LSTM_TILE_ELEMS * 4u;
constexpr uint32_t LSTM_ACT_TMP_BYTES = 16384u;               // shared tmp for Sigmoid / Tanh

__aicore__ inline int32_t LstmAlign8(int64_t x) { return static_cast<int32_t>((x + 7) / 8 * 8); }
__aicore__ inline int32_t LstmAlign64(int64_t x) { return static_cast<int32_t>((x + 63) / 64 * 64); }

/*! \brief workspace gate slot -> source gate index.
 *
 * The workspace stores the gates as [i | f | o | g] instead of the source [i | f | g | o] order so that
 * the three sigmoid gates occupy one contiguous 3G block and the tanh gate the trailing G block.  That
 * lets the recurrence run one Sigmoid and one Tanh call per time step instead of four calls. */
__aicore__ inline int64_t LstmGateSrc(int64_t slot)
{
    return (slot == 2) ? 3 : ((slot == 3) ? 2 : slot);
}

/*! \brief advance a device pointer by n float elements */
__aicore__ inline GM_ADDR LstmOffF(GM_ADDR base, int64_t n)
{
    return (GM_ADDR)((__gm__ float *)base + n);
}

/*! \brief advance a device pointer by n * sizeof(T) bytes */
__aicore__ inline GM_ADDR LstmOffBytes(GM_ADDR base, int64_t n, int64_t typeSize)
{
    return (GM_ADDR)((__gm__ uint8_t *)base + n * typeSize);
}

__aicore__ inline DataCopyPadExtParams<float> LstmPadF()
{
    DataCopyPadExtParams<float> p{false, 0, 0, 0.0f};
    return p;
}

/* ------------------------------------------------------------------ */
/* preparation kernel                                                  */
/* ------------------------------------------------------------------ */

/*! \brief one destination row: dst[0:dstLen) = cast(srcA[0:srcLen) [+ srcB]) padded with zeros */
template <typename T>
__aicore__ inline void LstmPrepRow(GM_ADDR srcA, GM_ADDR srcB, int64_t srcLen, GM_ADDR dst, int64_t dstLen,
                                   TQue<QuePosition::VECIN, 2> &qIn, TQue<QuePosition::VECOUT, 2> &qOut,
                                   TBuf<TPosition::VECCALC> &bTmp)
{
    if (dstLen <= 0) {
        return;
    }
    LocalTensor<float> f = qOut.AllocTensor<float>();
    if (srcLen <= 0) {
        Duplicate(f, 0.0f, static_cast<int32_t>(dstLen));
    } else {
        int32_t L = LstmAlign8(srcLen);
        if (static_cast<int64_t>(L) > dstLen) {
            L = static_cast<int32_t>(dstLen);
        }
        DataCopyExtParams cp{1, static_cast<uint32_t>(srcLen * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> pp{static_cast<bool>(L > srcLen), 0, static_cast<uint8_t>(L - srcLen),
                                   static_cast<T>(0)};
        LocalTensor<T> r = qIn.AllocTensor<T>();
        GlobalTensor<T> ga;
        ga.SetGlobalBuffer((__gm__ T *)srcA);
        DataCopyPad(r, ga[0], cp, pp);
        qIn.EnQue(r);
        r = qIn.DeQue<T>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(f, r, 0.0f, L);
        } else {
            Cast(f, r, RoundMode::CAST_NONE, L);
        }
        qIn.FreeTensor(r);

        if (srcB != nullptr) {
            LocalTensor<T> rb = qIn.AllocTensor<T>();
            GlobalTensor<T> gb;
            gb.SetGlobalBuffer((__gm__ T *)srcB);
            DataCopyPad(rb, gb[0], cp, pp);
            qIn.EnQue(rb);
            rb = qIn.DeQue<T>();
            LocalTensor<float> fb = bTmp.Get<float>();
            if constexpr (std::is_same_v<T, float>) {
                Adds(fb, rb, 0.0f, L);
            } else {
                Cast(fb, rb, RoundMode::CAST_NONE, L);
            }
            qIn.FreeTensor(rb);
            Add(f, f, fb, L);
        }
        if (static_cast<int64_t>(L) < dstLen) {
            Duplicate(f[L], 0.0f, static_cast<int32_t>(dstLen - L));
        }
    }
    qOut.EnQue(f);
    f = qOut.DeQue<float>();
    DataCopyExtParams cpo{1, static_cast<uint32_t>(dstLen * 4), 0, 0, 0};
    GlobalTensor<float> gd;
    gd.SetGlobalBuffer((__gm__ float *)dst);
    DataCopyPad(gd[0], f, cpo);
    qOut.FreeTensor(f);
}

template <typename T>
__global__ __aicore__ void lstm_prep_kernel(LstmArgs a)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qIn;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bTmp;
    int32_t pm = a.prepMaxElems;
    if (pm < LSTM_TILE_ROWS) {
        pm = LSTM_TILE_ROWS;
    }
    pipe.InitBuffer(qIn, 2, static_cast<uint32_t>(pm) * static_cast<uint32_t>(sizeof(T)));
    pipe.InitBuffer(qOut, 2, static_cast<uint32_t>(pm) * 4u);
    pipe.InitBuffer(bTmp, static_cast<uint32_t>(pm) * 4u);

    const int64_t nb = a.nbPrep > 0 ? static_cast<int64_t>(a.nbPrep) : 1;
    const int64_t bid = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t LD = a.LD;
    const int64_t G = a.G;
    const int64_t kHh = a.kHh;
    const int64_t H = a.H;
    const int64_t eff = a.eff;
    const int64_t tsz = static_cast<int64_t>(sizeof(T));

    /* ---- x: destination row (t*B + b) ---- */
    {
        const int64_t N = static_cast<int64_t>(a.S) * a.B;
        const int64_t w0 = N * bid / nb;
        const int64_t w1 = N * (bid + 1) / nb;
        for (int64_t i = w0; i < w1; ++i) {
            const int64_t t = i / a.B;
            const int64_t b = i - t * a.B;
            const int64_t srow = a.batchFirst ? (b * a.S + t) : i;
            LstmPrepRow<T>(LstmOffBytes(a.x, srow * a.inDim[0], tsz), nullptr, a.inDim[0],
                           LstmOffF(a.xW, i * a.inPad0), a.inPad0, qIn, qOut, bTmp);
        }
    }

    /* ---- weight_ih[ld], gate aware rows ---- */
    {
        const int64_t per = 4 * G;
        const int64_t N = LD * per;
        const int64_t w0 = N * bid / nb;
        const int64_t w1 = N * (bid + 1) / nb;
        for (int64_t i = w0; i < w1; ++i) {
            const int64_t ld = i / per;
            const int64_t r = i - ld * per;
            const int64_t g = r / G;
            const int64_t j = r - g * G;
            const int64_t inDim = a.inDim[ld];
            const int64_t inPad = a.inPad[ld];
            GM_ADDR src = nullptr;
            int64_t sl = 0;
            if (j < H) {
                src = LstmOffBytes(a.wih[ld], (LstmGateSrc(g) * H + j) * inDim, tsz);
                sl = inDim;
            }
            LstmPrepRow<T>(src, nullptr, sl, LstmOffF(a.wihW[ld], r * inPad), inPad, qIn, qOut, bTmp);
        }
    }

    /* ---- weight_hh[ld], gate aware rows ---- */
    {
        const int64_t per = 4 * G;
        const int64_t N = LD * per;
        const int64_t w0 = N * bid / nb;
        const int64_t w1 = N * (bid + 1) / nb;
        for (int64_t i = w0; i < w1; ++i) {
            const int64_t ld = i / per;
            const int64_t r = i - ld * per;
            const int64_t g = r / G;
            const int64_t j = r - g * G;
            GM_ADDR src = nullptr;
            int64_t sl = 0;
            if (j < H) {
                src = LstmOffBytes(a.whh[ld], (LstmGateSrc(g) * H + j) * eff, tsz);
                sl = eff;
            }
            LstmPrepRow<T>(src, nullptr, sl, LstmOffF(a.whhW[ld], r * kHh), kHh, qIn, qOut, bTmp);
        }
    }

    /* ---- weight_hr[ld] (LSTMP projection), plain rows padded to kHh ---- */
    if (a.useHr) {
        const int64_t N = LD * kHh;
        const int64_t w0 = N * bid / nb;
        const int64_t w1 = N * (bid + 1) / nb;
        for (int64_t i = w0; i < w1; ++i) {
            const int64_t ld = i / kHh;
            const int64_t r = i - ld * kHh;
            GM_ADDR src = nullptr;
            int64_t sl = 0;
            if (r < a.P) {
                src = LstmOffBytes(a.whr[ld], r * H, tsz);
                sl = H;
            }
            LstmPrepRow<T>(src, nullptr, sl, LstmOffF(a.whrW[ld], r * G), G, qIn, qOut, bTmp);
        }
    }

    /* ---- bias_ih + bias_hh, one G wide segment per gate ---- */
    {
        const int64_t N = LD * 4;
        const int64_t w0 = N * bid / nb;
        const int64_t w1 = N * (bid + 1) / nb;
        for (int64_t i = w0; i < w1; ++i) {
            const int64_t ld = i / 4;
            const int64_t g = i - ld * 4;
            GM_ADDR sa = nullptr;
            GM_ADDR sb = nullptr;
            int64_t sl = 0;
            if (a.hasBias) {
                sa = LstmOffBytes(a.bih[ld], LstmGateSrc(g) * H, tsz);
                sb = LstmOffBytes(a.bhh[ld], LstmGateSrc(g) * H, tsz);
                sl = H;
            }
            LstmPrepRow<T>(sa, sb, sl, LstmOffF(a.biasW[ld], g * G), G, qIn, qOut, bTmp);
        }
    }

    /* ---- h0 ---- */
    {
        const int64_t N = LD * a.B;
        const int64_t w0 = N * bid / nb;
        const int64_t w1 = N * (bid + 1) / nb;
        for (int64_t i = w0; i < w1; ++i) {
            GM_ADDR src = nullptr;
            int64_t sl = 0;
            if (a.hasH0) {
                src = LstmOffBytes(a.h0, i * eff, tsz);
                sl = eff;
            }
            LstmPrepRow<T>(src, nullptr, sl, LstmOffF(a.h0fW, i * kHh), kHh, qIn, qOut, bTmp);
        }
    }

    /* ---- c0 ---- */
    {
        const int64_t N = LD * a.B;
        const int64_t w0 = N * bid / nb;
        const int64_t w1 = N * (bid + 1) / nb;
        for (int64_t i = w0; i < w1; ++i) {
            GM_ADDR src = nullptr;
            int64_t sl = 0;
            if (a.hasC0) {
                src = LstmOffBytes(a.c0, i * H, tsz);
                sl = H;
            }
            LstmPrepRow<T>(src, nullptr, sl, LstmOffF(a.c0fW, i * G), G, qIn, qOut, bTmp);
        }
    }

    /* ---- zero the ping-pong layer buffers (their padding lanes are never rewritten) ---- */
    if (a.L > 1) {
        const int64_t NB = static_cast<int64_t>(a.S) * a.B;
        const int64_t N = 2 * NB;
        const int64_t w0 = N * bid / nb;
        const int64_t w1 = N * (bid + 1) / nb;
        for (int64_t i = w0; i < w1; ++i) {
            const int64_t which = i / NB;
            const int64_t r = i - which * NB;
            LstmPrepRow<T>(nullptr, nullptr, 0, LstmOffF(a.seqW[which], r * a.sw), a.sw, qIn, qOut, bTmp);
        }
    }
}

/* ------------------------------------------------------------------ */
/* recurrence kernel                                                   */
/* ------------------------------------------------------------------ */

/*! \brief acc[0:rows*64) (init on the first strip, else accumulate) += W tile .* broadcast(v slice)
 *
 * The very first strip of a gate block writes acc with Mul instead of MulAddDst so that no separate
 * zero-fill pass over acc is needed. */
__aicore__ inline void LstmAccum(LocalTensor<float> &acc, TQue<QuePosition::VECIN, 2> &qW,
                                 const LocalTensor<float> &v, GM_ADDR wBase, int64_t wOff, int64_t kStride,
                                 int64_t K, int32_t rows, bool &first)
{
    GlobalTensor<float> wg;
    wg.SetGlobalBuffer((__gm__ float *)wBase);
    const int64_t nStrip = K >> 6;
    for (int64_t s = 0; s < nStrip; ++s) {
        LocalTensor<float> wt = qW.AllocTensor<float>();
        DataCopyExtParams cp{static_cast<uint16_t>(rows), 256u,
                             static_cast<uint32_t>((kStride - 64) * 4), 0, 0};
        DataCopyPad(wt, wg[wOff + (s << 6)], cp, LstmPadF());
        qW.EnQue(wt);
        wt = qW.DeQue<float>();
        BinaryRepeatParams rp{1, 1, 1, 8, 8, 0};
        if (first) {
            Mul(acc, wt, v[0], static_cast<uint64_t>(64), static_cast<uint8_t>(rows), rp);
            first = false;
        } else {
            MulAddDst(acc, wt, v[static_cast<int32_t>(s << 6)], static_cast<uint64_t>(64),
                      static_cast<uint8_t>(rows), rp);
        }
        qW.FreeTensor(wt);
    }
}

/*! \brief same as LstmAccum but with the gate weights already resident in the Unified Buffer */
__aicore__ inline void LstmAccumUb(LocalTensor<float> &acc, const LocalTensor<float> &w, int64_t wOff,
                                   int64_t kStride, const LocalTensor<float> &v, int64_t K, int32_t rows,
                                   bool &first)
{
    const uint8_t rs = static_cast<uint8_t>(kStride >> 3);
    const int64_t nStrip = K >> 6;
    for (int64_t s = 0; s < nStrip; ++s) {
        BinaryRepeatParams rp{1, 1, 1, 8, rs, 0};
        if (first) {
            Mul(acc, w[static_cast<int32_t>(wOff)], v[0], static_cast<uint64_t>(64),
                static_cast<uint8_t>(rows), rp);
            first = false;
        } else {
            MulAddDst(acc, w[static_cast<int32_t>(wOff + (s << 6))], v[static_cast<int32_t>(s << 6)],
                      static_cast<uint64_t>(64), static_cast<uint8_t>(rows), rp);
        }
    }
}

template <typename T>
__global__ __aicore__ void lstm_rec_kernel(LstmArgs a, int64_t layer)
{
    const int64_t bid = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t nblk = static_cast<int64_t>(a.B) * a.D;
    if (bid >= nblk) {
        return;
    }
    const int64_t b = bid / a.D;
    const int64_t d = bid % a.D;
    const int64_t ld = layer * a.D + d;

    const int32_t G = a.G;
    const int32_t kHh = a.kHh;
    const int32_t fourG = 4 * G;
    const int32_t inPad = a.inPad[ld];
    const int32_t nrblk = fourG / LSTM_TILE_ROWS;
    const int32_t prblk = kHh / LSTM_PR_ROWS;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qW;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qV;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qO;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qF;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qCache;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qInit;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bAcc;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bRed;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bAct;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bT1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bT2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bHpr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bTmpS;

    /* Capacity driven residency decision: keep the per layer gate weights in UB when the remaining
     * Unified Buffer budget can hold them, so the per timestep weight traffic disappears entirely. */
    const int64_t needWih = static_cast<int64_t>(fourG) * inPad * 4;
    const int64_t needWhh = static_cast<int64_t>(fourG) * kHh * 4;
    const uint32_t qvB = static_cast<uint32_t>(inPad) * 4u;
    const uint32_t qoB = static_cast<uint32_t>(kHh) * static_cast<uint32_t>(sizeof(T));
    const uint32_t qfB = static_cast<uint32_t>(kHh) * 4u;
    const uint32_t redB = static_cast<uint32_t>(fourG) * 4u;
    const uint32_t actB = static_cast<uint32_t>(G) * 4u;
    const uint32_t hstB = static_cast<uint32_t>(kHh) * 4u;
    const uint32_t initB = redB + actB + hstB;   // bias(4G) | c(G) | h(kHh)
    const uint32_t otherB =
        qvB + qoB + qfB + LSTM_TILE_BYTES + redB + initB + 7u * actB + LSTM_ACT_TMP_BYTES;
    const uint32_t ubTotal = static_cast<uint32_t>(a.ubBytes > 0 ? a.ubBytes : 196608);
    bool cacheWih = false;
    bool cacheWhh = false;
    /* The LSTMP projection always streams W_hr through qW, so a projection case must reserve a
     * full size qW even when both gate weight streams are replaced by UB residency. */
    {
        const uint32_t qwBothB = (a.useHr != 0) ? (2u * LSTM_TILE_BYTES) : 256u;
        const uint32_t baseA = qwBothB + otherB;
        const uint32_t availA = (ubTotal > baseA + 4096u) ? (ubTotal - baseA - 4096u) : 0u;
        if (static_cast<int64_t>(availA) >= needWih + needWhh) {
            cacheWih = true;
            cacheWhh = true;
        } else {
            const uint32_t baseB = 2u * LSTM_TILE_BYTES + otherB;
            const uint32_t availB = (ubTotal > baseB + 4096u) ? (ubTotal - baseB - 4096u) : 0u;
            cacheWhh = (static_cast<int64_t>(availB) >= needWhh);
        }
    }
    const uint32_t cacheBytes = (cacheWih ? static_cast<uint32_t>(needWih) : 0u) +
                                (cacheWhh ? static_cast<uint32_t>(needWhh) : 0u);

    const bool qwFull = (a.useHr != 0) || !cacheWih;
    pipe.InitBuffer(qW, 2, qwFull ? LSTM_TILE_BYTES : 128u);
    pipe.InitBuffer(qV, 2, qvB);
    pipe.InitBuffer(qO, 2, qoB);
    pipe.InitBuffer(qF, 2, qfB);
    if (cacheBytes > 0u) {
        pipe.InitBuffer(qCache, 1, cacheBytes);
    }
    pipe.InitBuffer(bAcc, LSTM_TILE_BYTES);
    pipe.InitBuffer(bRed, redB);
    pipe.InitBuffer(qInit, 1, initB);
    pipe.InitBuffer(bAct, 4u * actB);
    pipe.InitBuffer(bT1, actB);
    pipe.InitBuffer(bT2, actB);
    pipe.InitBuffer(bHpr, actB);
    pipe.InitBuffer(bTmpS, LSTM_ACT_TMP_BYTES);

    LocalTensor<float> acc = bAcc.Get<float>();
    LocalTensor<float> red = bRed.Get<float>();
    LocalTensor<float> act = bAct.Get<float>();
    LocalTensor<float> t1 = bT1.Get<float>();
    LocalTensor<float> t2 = bT2.Get<float>();
    LocalTensor<float> hpr = bHpr.Get<float>();
    LocalTensor<uint8_t> actTmp = bTmpS.Get<uint8_t>();

    /* ---- initial state + bias + optional resident gate weights, all through queues (no manual flags) ---- */
    LocalTensor<float> initT = qInit.AllocTensor<float>();
    {
        GlobalTensor<float> gh, gc, gb;
        gh.SetGlobalBuffer((__gm__ float *)a.h0fW);
        gc.SetGlobalBuffer((__gm__ float *)a.c0fW);
        gb.SetGlobalBuffer((__gm__ float *)a.biasW[ld]);
        DataCopyExtParams cp{1, static_cast<uint32_t>(fourG * 4), 0, 0, 0};
        DataCopyPad(initT[0], gb[0], cp, LstmPadF());
        cp.blockLen = static_cast<uint32_t>(G * 4);
        DataCopyPad(initT[fourG], gc[(ld * a.B + b) * G], cp, LstmPadF());
        cp.blockLen = static_cast<uint32_t>(kHh * 4);
        DataCopyPad(initT[fourG + G], gh[(ld * a.B + b) * kHh], cp, LstmPadF());
    }
    qInit.EnQue(initT);
    initT = qInit.DeQue<float>();
    LocalTensor<float> bias = initT;
    LocalTensor<float> cBuf = initT[fourG];
    LocalTensor<float> hst = initT[fourG + G];

    LocalTensor<float> wihC;
    LocalTensor<float> whhC;
    const int32_t whhOffElems = cacheWih ? static_cast<int32_t>(fourG * inPad) : 0;
    if (cacheBytes > 0u) {
        LocalTensor<float> wbuf = qCache.AllocTensor<float>();
        if (cacheWih) {
            GlobalTensor<float> gw;
            gw.SetGlobalBuffer((__gm__ float *)a.wihW[ld]);
            DataCopyExtParams cpw{1, static_cast<uint32_t>(needWih), 0, 0, 0};
            DataCopyPad(wbuf[0], gw[0], cpw, LstmPadF());
        }
        if (cacheWhh) {
            GlobalTensor<float> gw;
            gw.SetGlobalBuffer((__gm__ float *)a.whhW[ld]);
            DataCopyExtParams cpw{1, static_cast<uint32_t>(needWhh), 0, 0, 0};
            DataCopyPad(wbuf[whhOffElems], gw[0], cpw, LstmPadF());
        }
        qCache.EnQue(wbuf);
        LocalTensor<float> wgot = qCache.DeQue<float>();
        if (cacheWih) {
            wihC = wgot[0];
        }
        if (cacheWhh) {
            whhC = wgot[whhOffElems];
        }
    }

    GM_ADDR inBase;
    int64_t inRowStride;
    if (layer == 0) {
        inBase = a.xW;
        inRowStride = a.inPad0;
    } else {
        inBase = a.seqW[(layer - 1) & 1];
        inRowStride = a.sw;
    }
    GlobalTensor<float> inGm;
    inGm.SetGlobalBuffer((__gm__ float *)inBase);

    const bool isLast = (layer == a.L - 1);

    for (int64_t step = 0; step < a.S; ++step) {
        const int64_t t = (d == 1) ? (a.S - 1 - step) : step;

        /* ---- layer input vector ---- */
        LocalTensor<float> vin = qV.AllocTensor<float>();
        {
            DataCopyExtParams cp{1, static_cast<uint32_t>(inPad * 4), 0, 0, 0};
            DataCopyPad(vin, inGm[(t * a.B + b) * inRowStride], cp, LstmPadF());
        }
        qV.EnQue(vin);
        vin = qV.DeQue<float>();

        /* ---- gates = W_ih x + W_hh h + bias ---- */
        for (int32_t rb = 0; rb < nrblk; ++rb) {
            const int32_t r0 = rb * LSTM_TILE_ROWS;
            bool first = true;
            if (cacheWih) {
                LstmAccumUb(acc, wihC, static_cast<int64_t>(r0) * inPad, inPad, vin, inPad, LSTM_TILE_ROWS,
                            first);
            } else {
                LstmAccum(acc, qW, vin, a.wihW[ld], static_cast<int64_t>(r0) * inPad, inPad, inPad,
                          LSTM_TILE_ROWS, first);
            }
            if (cacheWhh) {
                LstmAccumUb(acc, whhC, static_cast<int64_t>(r0) * kHh, kHh, hst, kHh, LSTM_TILE_ROWS, first);
            } else {
                LstmAccum(acc, qW, hst, a.whhW[ld], static_cast<int64_t>(r0) * kHh, kHh, kHh, LSTM_TILE_ROWS,
                          first);
            }
            WholeReduceSum<float>(red[r0], acc, static_cast<uint64_t>(64),
                                  static_cast<int32_t>(LSTM_TILE_ROWS), static_cast<int32_t>(1),
                                  static_cast<int32_t>(1), static_cast<int32_t>(8));
        }
        qV.FreeTensor(vin);

        Add(red, red, bias, fourG);

        /* ---- activations and cell update; workspace gate order is [i | f | o | g] ---- */
        Sigmoid<float>(act, red[0], actTmp, static_cast<uint32_t>(3 * G));
        Tanh<float>(act[3 * G], red[3 * G], actTmp, static_cast<uint32_t>(G));

        Mul<float>(t1, act[G], cBuf, G);
        Mul<float>(t2, act[0], act[3 * G], G);
        Add<float>(cBuf, t1, t2, G);
        Tanh<float>(t1, cBuf, actTmp, static_cast<uint32_t>(G));
        Mul<float>(hpr, act[2 * G], t1, G);

        /* ---- projection (LSTMP) or plain hidden state ---- */
        if (a.useHr) {
            for (int32_t rb = 0; rb < prblk; ++rb) {
                const int32_t r0 = rb * LSTM_PR_ROWS;
                bool pfirst = true;
                LstmAccum(acc, qW, hpr, a.whrW[ld], static_cast<int64_t>(r0) * G, G, G, LSTM_PR_ROWS, pfirst);
                WholeReduceSum<float>(hst[r0], acc, static_cast<uint64_t>(64),
                                      static_cast<int32_t>(LSTM_PR_ROWS), static_cast<int32_t>(1),
                                      static_cast<int32_t>(1), static_cast<int32_t>(8));
            }
        } else {
            Adds(hst, hpr, 0.0f, G);
        }

        /* ---- emit the layer output ---- */
        if (isLast) {
            LocalTensor<T> ot = qO.AllocTensor<T>();
            if constexpr (std::is_same_v<T, float>) {
                Adds(ot, hst, 0.0f, kHh);
            } else {
                Cast(ot, hst, RoundMode::CAST_RINT, kHh);
            }
            qO.EnQue(ot);
            ot = qO.DeQue<T>();
            GlobalTensor<T> gy;
            gy.SetGlobalBuffer((__gm__ T *)a.y);
            DataCopyExtParams cp{1, static_cast<uint32_t>(static_cast<int64_t>(a.eff) * sizeof(T)), 0, 0, 0};
            const int64_t off = a.batchFirst
                                    ? ((b * a.S + t) * a.D + d) * static_cast<int64_t>(a.eff)
                                    : ((t * a.B + b) * a.D + d) * static_cast<int64_t>(a.eff);
            DataCopyPad(gy[off], ot, cp);
            qO.FreeTensor(ot);
        } else {
            LocalTensor<float> of = qF.AllocTensor<float>();
            Adds(of, hst, 0.0f, kHh);
            qF.EnQue(of);
            of = qF.DeQue<float>();
            GlobalTensor<float> gs;
            gs.SetGlobalBuffer((__gm__ float *)a.seqW[layer & 1]);
            DataCopyExtParams cp{1, static_cast<uint32_t>(static_cast<int64_t>(a.eff) * 4), 0, 0, 0};
            const int64_t off = (t * a.B + b) * static_cast<int64_t>(a.sw) + static_cast<int64_t>(d) * a.eff;
            DataCopyPad(gs[off], of, cp);
            qF.FreeTensor(of);
        }
    }

    /* ---- final states ---- */
    {
        LocalTensor<T> ot = qO.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(ot, hst, 0.0f, kHh);
        } else {
            Cast(ot, hst, RoundMode::CAST_RINT, kHh);
        }
        qO.EnQue(ot);
        ot = qO.DeQue<T>();
        GlobalTensor<T> ghn;
        ghn.SetGlobalBuffer((__gm__ T *)a.hn);
        DataCopyExtParams cp{1, static_cast<uint32_t>(static_cast<int64_t>(a.eff) * sizeof(T)), 0, 0, 0};
        DataCopyPad(ghn[(ld * a.B + b) * static_cast<int64_t>(a.eff)], ot, cp);
        qO.FreeTensor(ot);
    }
    {
        LocalTensor<T> ot = qO.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(ot, cBuf, 0.0f, G);
        } else {
            Cast(ot, cBuf, RoundMode::CAST_RINT, G);
        }
        qO.EnQue(ot);
        ot = qO.DeQue<T>();
        GlobalTensor<T> gcn;
        gcn.SetGlobalBuffer((__gm__ T *)a.cn);
        DataCopyExtParams cp{1, static_cast<uint32_t>(static_cast<int64_t>(a.H) * sizeof(T)), 0, 0, 0};
        DataCopyPad(gcn[(ld * a.B + b) * static_cast<int64_t>(a.H)], ot, cp);
        qO.FreeTensor(ot);
    }
}

} // namespace

/* ------------------------------------------------------------------ */
/* host side helpers + launch wrappers                                 */
/* ------------------------------------------------------------------ */

int64_t lstm_core_num()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t n = static_cast<int64_t>(ascendcPlatform->GetCoreNumAiv());
    if (n <= 0) {
        n = 1;
    }
    if (n > 64) {
        n = 64;
    }
    return n;
}

int64_t lstm_ub_bytes()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (ub == 0) {
        ub = 196608;
    }
    return static_cast<int64_t>(ub);
}

extern "C" {

void launch_lstm_prep_float(LstmArgs a, void *stream)
{
    const int64_t nb = a.nbPrep > 0 ? a.nbPrep : 1;
    lstm_prep_kernel<float><<<nb, nullptr, stream>>>(a);
}

void launch_lstm_prep_half(LstmArgs a, void *stream)
{
    const int64_t nb = a.nbPrep > 0 ? a.nbPrep : 1;
    lstm_prep_kernel<half><<<nb, nullptr, stream>>>(a);
}

void launch_lstm_prep_bf16(LstmArgs a, void *stream)
{
    const int64_t nb = a.nbPrep > 0 ? a.nbPrep : 1;
    lstm_prep_kernel<bfloat16_t><<<nb, nullptr, stream>>>(a);
}

void launch_lstm_rec_float(LstmArgs a, int64_t layer, void *stream)
{
    lstm_rec_kernel<float><<<static_cast<int64_t>(a.B) * a.D, nullptr, stream>>>(a, layer);
}

void launch_lstm_rec_half(LstmArgs a, int64_t layer, void *stream)
{
    lstm_rec_kernel<half><<<static_cast<int64_t>(a.B) * a.D, nullptr, stream>>>(a, layer);
}

void launch_lstm_rec_bf16(LstmArgs a, int64_t layer, void *stream)
{
    lstm_rec_kernel<bfloat16_t><<<static_cast<int64_t>(a.B) * a.D, nullptr, stream>>>(a, layer);
}

} // extern "C"
