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
 * \file add_rms_norm_dynamic_quant_kernel.cpp
 * \brief AddRmsNormDynamicQuant kernel + tiling + launch (compiled with bisheng + -xasc)
 *
 *   xOut     = x1 + x2                                       (fp32 add, rounded to T on output)
 *   rms      = sqrt(mean(xOut^2) + eps)
 *   y_norm   = xOut / rms * gamma
 *   scaleOut = clamp(max|y_norm|, 1e-12) / 127               (fp32, shape = x1.shape[:-1])
 *   y        = round(y_norm / scaleOut)                      (int8, saturated)
 *
 * Algebra
 * -------
 * Quantization is evaluated through z = xOut * gamma.  Since y_norm = z * rstd,
 *     maxAbs   = max|z|
 *     scaleOut = clamp(maxAbs * rstd, 1e-12) / 127
 *     k        = rstd / scaleOut            (== 127/maxAbs when maxAbs > 0)
 *     y        = round(z * k)
 * The rms factor cancels out of `y` exactly while `scaleOut` keeps the golden's
 * clamp semantics (an all-zero row gives clamp(0,1e-12)/127, never
 * 1e-12*rstd/127).
 *
 * Two execution paths, chosen by the host tiling:
 *
 *  * BLOCK path (R rows per iteration, tileElems == 0): the whole row block is
 *    moved with ONE multi-row DataCopyPad per tensor, both inputs are widened to
 *    fp32, xOut is stored with one multi-row DMA, `prod = xOut^2` is computed
 *    block-wide, and only the gamma multiply (per-row, gamma is a single row) plus
 *    pass B stay per-row.  The two per-row reductions are replaced by ONE
 *    multi-row AR reduce each (srcShape = {Rb, P}).  Amortizing the per-row
 *    latency (small DMA + reduce + GetValue chain) over R rows is what makes this
 *    path win for short rows.
 *
 *  * CHUNK path (tileElems = S, rowBlock = 0): one row at a time, walked in
 *    chunks of S lanes with lane-wise accumulators and double-buffered queues.
 *    Used for rows too wide for a 2-row block footprint.
 *
 * Pitches (block path)
 * --------------------
 * The transfers keep the DMA's NATURAL row pitch -- AlignUp(N,16) lanes for
 * 2-byte T and AlignUp(N,32) for int8 -- so blockLen is the only DMA parameter
 * needed (no strides and, importantly, no custom padding: a rightPadding wider
 * than the 32B DMA pad granularity is rejected by the MTE unit).
 *
 * The fp32 workspace instead uses P = AlignUp(N,64) lanes per row, for two
 * reasons: every fp32 row start is then 32B aligned, and every element count
 * handed to a vector op (P for a row, Rb*P for the block) is a whole number of
 * 64-lane repeats -- a multi-row AR reduce whose inner axis is not a whole
 * repeat faults on this target.
 *
 * The two pitches only coincide when N is a multiple of 64 (P == N): only then
 * does the DMA leave NO pad lane at all.  That is the condition for the flat
 * block-wide cast; whenever N % 64 != 0 the pad lanes hold arbitrary DMA bytes,
 * so the widening/narrowing must be done per row (count = N) and the fp32 tails
 * keep their zeroed value.
 *
 * Tails
 * -----
 * All fp32 block buffers and the fp32 gamma are zeroed once per core, so every
 * unused tail lane is exactly 0 and the block-wide products/sums over
 * Rb*P lanes are exact: z = xOut*gamma, xOut^2, |z| and the two AR reductions
 * all see zero tails.  Summing zero is exact and a zero never affects a max of
 * absolute values, so no masking is required.
 *
 * int8 narrowing goes fp32 -> half -> int8 (both CAST_RINT): dav_c220 has no
 * fp32->int8 vector cast, and the worst-case double rounding is bounded by 1,
 * which the int8 tolerance admits.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "add_rms_norm_dynamic_quant_launch.h"

namespace {

constexpr int32_t ARDQ_DEPTH = 2;
constexpr int32_t ARDQ_WORK_BYTES = 4096;
constexpr int32_t ARDQ_RED_BYTES = 512;
constexpr int32_t ARDQ_SLOT_BYTES = 512;
constexpr int64_t ARDQ_TILE_MAX = 4096;
constexpr int32_t ARDQ_SCALE_BATCH = 32;

// Rows per block in the batched path.
constexpr int64_t ARDQ_RMAX = 64;
// sharedTmpBuffer for the multi-row AR reductions.
constexpr int32_t ARDQ_AR_WORK = 16384;
// lane slack added to the block buffers (keeps any partial-repeat access inside)
constexpr int64_t ARDQ_SLACK_LANES = 64;
// reduction staging: [0,64) sums, [64,128) maxes, then the scale row
constexpr int32_t ARDQ_SC_BYTES = 1024;

__aicore__ inline int64_t ArdqAlignUp(int64_t x, int64_t a)
{
    return ((x + a - 1) / a) * a;
}

}  // namespace

// ---------------------------------------------------------------------------
// Block path: R rows per iteration
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void ardq_block_kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
                                             GM_ADDR y, GM_ADDR xOut, GM_ADDR scaleOut,
                                             int64_t numRows, int64_t rowLen,
                                             int64_t rowsPerCore, int64_t Rrows,
                                             float epsilon)
{
    const int64_t N = rowLen;
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());

    int64_t rowStart = blk * rowsPerCore;
    if (rowStart >= numRows) {
        return;
    }
    int64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > numRows) {
        rowEnd = numRows;
    }

    const int64_t P = ArdqAlignUp(N, 64);     // fp32 lane pitch (whole 64-lane repeats)
    const int64_t nT = ArdqAlignUp(N, 16);    // natural T lane pitch of the transfer
    const int64_t n8 = ArdqAlignUp(N, 32);    // natural int8 lane pitch of the transfer
    const int64_t S = ARDQ_SLACK_LANES;
    // No DMA pad lane exists only when the natural pitch already is the fp32 pitch.
    const bool aligned = (N == P);
    const float invN = (N > 0) ? (1.0f / static_cast<float>(N)) : 0.0f;
    const float kInv127 = 1.0f / 127.0f;
    const int64_t R = Rrows;

    AscendC::GlobalTensor<T> x1Gm, x2Gm, gGm, xOutGm;
    AscendC::GlobalTensor<int8_t> yGm;
    AscendC::GlobalTensor<float> sGm;
    x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1) + rowStart * N);
    x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2) + rowStart * N);
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma));
    xOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xOut) + rowStart * N);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y) + rowStart * N);
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scaleOut) + rowStart);

    // Row slots plus exactly the slack a partial-repeat access of the last row
    // can reach: (R-1)*pitch + Align64(N) = R*pitch + (P - pitch).
    const int64_t Blane = R * P + S;
    const int64_t Tlane = R * nT + (P > nT ? (P - nT) : 0) + S;
    const int64_t Ylane = R * n8 + (P > n8 ? (P - n8) : 0) + S;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> x1Q, x2Q;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> oQ, yQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xoB, zB, gB, scB, hBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gTBuf, wkB;

    pipe.InitBuffer(x1Q, 1, static_cast<uint32_t>(Tlane * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(x2Q, 1, static_cast<uint32_t>(Tlane * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(oQ, 1, static_cast<uint32_t>(Tlane * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(yQ, 1, static_cast<uint32_t>(Ylane));
    pipe.InitBuffer(xoB, static_cast<uint32_t>(Blane * 4));
    pipe.InitBuffer(zB, static_cast<uint32_t>(Blane * 4));
    pipe.InitBuffer(gB, static_cast<uint32_t>(P * 4 + 256));
    pipe.InitBuffer(scB, ARDQ_SC_BYTES);
    pipe.InitBuffer(hBuf, static_cast<uint32_t>(P * 2 + 256));
    pipe.InitBuffer(gTBuf, static_cast<uint32_t>(nT * static_cast<int64_t>(sizeof(T)) + 256));
    pipe.InitBuffer(wkB, static_cast<uint32_t>(ARDQ_AR_WORK));

    auto xo = xoB.Get<float>();
    auto zb = zB.Get<float>();
    auto gf = gB.Get<float>();
    auto scv = scB.Get<float>();
    auto hf = hBuf.Get<half>();
    auto gT = gTBuf.Get<T>();
    auto work = wkB.Get<uint8_t>();

    constexpr AscendC::RoundMode kDownT =
        std::is_same<T, bfloat16_t>::value ? AscendC::RoundMode::CAST_RINT
                                           : AscendC::RoundMode::CAST_NONE;

    // ---- one-off init: zero the fp32 workspace and widen gamma ----
    // Every unused tail lane of the fp32 blocks stays 0 for the whole kernel, so
    // every Rb*P-wide op is exact: the sums only add zeros and the max of
    // absolute values cannot see a zero.
    {
        const int32_t zl = static_cast<int32_t>(R * P + S);
        AscendC::Duplicate(xo, 0.0f, zl);
        AscendC::Duplicate(zb, 0.0f, zl);
        AscendC::Duplicate(gf, 0.0f, static_cast<int32_t>(P + S));

        AscendC::DataCopyExtParams gcp{1, static_cast<uint32_t>(N * static_cast<int64_t>(sizeof(T))),
                                       0, 0, 0};
        AscendC::DataCopyPadExtParams<T> gpp{false, 0, 0, 0};
        AscendC::DataCopyPad(gT, gGm[0], gcp, gpp);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(gf, gT, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(N));
    }

    for (int64_t base = rowStart; base < rowEnd; base += R) {
        int64_t Rb = rowEnd - base;
        if (Rb > R) {
            Rb = R;
        }
        const int64_t rbase = base - rowStart;
        const uint16_t rbU = static_cast<uint16_t>(Rb);
        const uint32_t rowBytesT = static_cast<uint32_t>(N * static_cast<int64_t>(sizeof(T)));
        const int32_t blkElems = static_cast<int32_t>(Rb * P);
        const int32_t nU = static_cast<int32_t>(N);

        // ---------------- load x1 / x2 (one multi-row DMA each) ----------------
        {
            AscendC::DataCopyExtParams cp{rbU, rowBytesT, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            auto t1 = x1Q.AllocTensor<T>();
            auto t2 = x2Q.AllocTensor<T>();
            AscendC::DataCopyPad(t1, x1Gm[rbase * N], cp, pp);
            AscendC::DataCopyPad(t2, x2Gm[rbase * N], cp, pp);
            x1Q.EnQue(t1);
            x2Q.EnQue(t2);
            t1 = x1Q.DeQue<T>();
            t2 = x2Q.DeQue<T>();

            if (aligned) {
                // no DMA pad, identical pitches -> one flat cast for the block
                AscendC::Cast(xo, t1, AscendC::RoundMode::CAST_NONE, blkElems);
                AscendC::Cast(zb, t2, AscendC::RoundMode::CAST_NONE, blkElems);
            } else {
                // per-row widen: only [0,N) of a row slot is written, tails stay 0
                for (int64_t r = 0; r < Rb; ++r) {
                    AscendC::Cast(xo[r * P], t1[r * nT], AscendC::RoundMode::CAST_NONE, nU);
                }
                for (int64_t r = 0; r < Rb; ++r) {
                    AscendC::Cast(zb[r * P], t2[r * nT], AscendC::RoundMode::CAST_NONE, nU);
                }
            }
            x1Q.FreeTensor(t1);
            x2Q.FreeTensor(t2);
            AscendC::Add(xo, xo, zb, blkElems);  // xo = xOut (tails 0 + 0)
        }

        // ------------------------- xOut -> GM -------------------------
        {
            auto oT = oQ.AllocTensor<T>();
            if (aligned) {
                AscendC::Cast(oT, xo, kDownT, blkElems);
            } else {
                for (int64_t r = 0; r < Rb; ++r) {
                    AscendC::Cast(oT[r * nT], xo[r * P], kDownT, nU);
                }
            }
            oQ.EnQue(oT);
            auto oS = oQ.DeQue<T>();
            AscendC::DataCopyPad(xOutGm[rbase * N], oS,
                                 AscendC::DataCopyExtParams{rbU, rowBytesT, 0, 0, 0});
            oQ.FreeTensor(oS);
        }

        // ---- z = xOut * gamma (tails 0 * 0 = 0) ----
        for (int64_t r = 0; r < Rb; ++r) {
            AscendC::Mul(zb[r * P], xo[r * P], gf, static_cast<int32_t>(P));
        }

        // ---- prod = xOut^2 (in place: xOut is already stored), then |z| into
        //      the freed xo so that zb keeps the SIGNED z for pass B ----
        uint32_t shape[2] = {static_cast<uint32_t>(Rb), static_cast<uint32_t>(P)};
        AscendC::Mul(xo, xo, xo, blkElems);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(scv, xo, work, shape, true);
        AscendC::Abs(xo, zb, blkElems);
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR>(scv[64], xo, work, shape, true);

        AscendC::Muls(scv, scv, invN, static_cast<int32_t>(Rb));
        AscendC::Adds(scv, scv, epsilon, static_cast<int32_t>(Rb));
        AscendC::Sqrt(scv, scv, static_cast<int32_t>(Rb));

        // ---- scalar finalize (identical math to the chunk path) ----
        AscendC::PipeBarrier<PIPE_ALL>();
        float kArr[ARDQ_RMAX];
        float scArr[ARDQ_RMAX];
        for (int64_t r = 0; r < Rb; ++r) {
            const float rms = scv.GetValue(static_cast<int32_t>(r));
            const float maxAbs = scv[64].GetValue(static_cast<int32_t>(r));
            // No positive guard: NaN/inf must propagate into rstd (IEEE division).
            const float rstd = 1.0f / rms;
            const float m = maxAbs * rstd;
            // `(m < min)` propagates NaN (comparison false) and clamps only true zeros.
            const float sm = (m < 1e-12f) ? 1e-12f : m;
            scArr[r] = sm * kInv127;
            kArr[r] = rstd / scArr[r];
        }
        for (int64_t r = 0; r < Rb; ++r) {
            scv.SetValue(static_cast<int32_t>(r), scArr[r]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();  // scalar writes -> MTE3 / scalar -> vector

        AscendC::DataCopyPad(sGm[rbase], scv,
                             AscendC::DataCopyExtParams{1, static_cast<uint32_t>(Rb * 4), 0, 0, 0});

        // ------------------------- pass B -------------------------
        {
            auto yT = yQ.AllocTensor<int8_t>();
            for (int64_t r = 0; r < Rb; ++r) {
                AscendC::Muls(xo[r * P], zb[r * P], kArr[r], static_cast<int32_t>(P));
                AscendC::Cast(hf, xo[r * P], AscendC::RoundMode::CAST_RINT,
                              static_cast<int32_t>(P));
                AscendC::Cast(yT[r * n8], hf, AscendC::RoundMode::CAST_RINT, nU);
            }
            yQ.EnQue(yT);
            auto yS = yQ.DeQue<int8_t>();
            AscendC::DataCopyPad(yGm[rbase * N], yS,
                                 AscendC::DataCopyExtParams{rbU, static_cast<uint32_t>(N), 0, 0, 0});
            yQ.FreeTensor(yS);
        }
    }
}

// ---------------------------------------------------------------------------
// Chunk path: one row at a time, used when the row is too wide for a block
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void ardq_chunk_kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
                                             GM_ADDR y, GM_ADDR xOut, GM_ADDR scaleOut,
                                             int64_t numRows, int64_t rowLen,
                                             int64_t rowsPerCore, uint32_t tileElems,
                                             float epsilon)
{
    const int64_t N = rowLen;
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());

    int64_t rowStart = blk * rowsPerCore;
    if (rowStart >= numRows) {
        return;
    }
    int64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > numRows) {
        rowEnd = numRows;
    }

    const int64_t S = static_cast<int64_t>(tileElems);      // chunk length, multiple of 64
    const int64_t NF = ArdqAlignUp(N, 64);                  // fp32 full-row lanes
    const int64_t NT = ArdqAlignUp(N, 16);                  // 2-byte full-row lanes (32B)
    const float invN = (N > 0) ? (1.0f / static_cast<float>(N)) : 0.0f;
    const float kInv127 = 1.0f / 127.0f;

    AscendC::GlobalTensor<T> x1Gm, x2Gm, gGm, xOutGm;
    AscendC::GlobalTensor<int8_t> yGm;
    AscendC::GlobalTensor<float> sGm;
    x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1) + rowStart * N);
    x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2) + rowStart * N);
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma));
    xOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xOut) + rowStart * N);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y) + rowStart * N);
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scaleOut) + rowStart);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, ARDQ_DEPTH> inQ1, inQ2;
    AscendC::TQue<AscendC::TPosition::VECOUT, ARDQ_DEPTH> outQ, yQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaTb, zBuf, f1Buf, f2Buf, gfBuf,
        hBuf, accSumBuf, accMaxBuf, redBuf, workBuf, slotBuf;

    pipe.InitBuffer(inQ1, ARDQ_DEPTH, static_cast<uint32_t>(S * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(inQ2, ARDQ_DEPTH, static_cast<uint32_t>(S * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(outQ, ARDQ_DEPTH, static_cast<uint32_t>(S * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(yQ, ARDQ_DEPTH, static_cast<uint32_t>(ArdqAlignUp(S, 32)));
    pipe.InitBuffer(gammaTb, static_cast<uint32_t>(NT * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(zBuf, static_cast<uint32_t>(NF * 4));
    pipe.InitBuffer(f1Buf, static_cast<uint32_t>(S * 4));
    pipe.InitBuffer(f2Buf, static_cast<uint32_t>(S * 4));
    pipe.InitBuffer(gfBuf, static_cast<uint32_t>(S * 4));
    pipe.InitBuffer(hBuf, static_cast<uint32_t>(S * 2));
    pipe.InitBuffer(accSumBuf, static_cast<uint32_t>(S * 4));
    pipe.InitBuffer(accMaxBuf, static_cast<uint32_t>(S * 4));
    pipe.InitBuffer(redBuf, ARDQ_RED_BYTES);
    pipe.InitBuffer(workBuf, ARDQ_WORK_BYTES);
    pipe.InitBuffer(slotBuf, ARDQ_SLOT_BYTES);

    auto zf = zBuf.Get<float>();
    auto f1 = f1Buf.Get<float>();
    auto f2 = f2Buf.Get<float>();
    auto gf = gfBuf.Get<float>();
    auto hf = hBuf.Get<half>();
    auto aSum = accSumBuf.Get<float>();
    auto aMax = accMaxBuf.Get<float>();
    auto red = redBuf.Get<float>();
    auto work = workBuf.Get<float>();
    auto slot = slotBuf.Get<float>();
    auto gT = gammaTb.Get<T>();

    // ---- load gamma once per core (kept resident for the whole core range) ----
    {
        AscendC::DataCopyExtParams gcp{1, static_cast<uint32_t>(N * static_cast<int64_t>(sizeof(T))),
                                       0, 0, 0};
        AscendC::DataCopyPadExtParams<T> gpp{false, 0, 0, 0};
        AscendC::DataCopyPad(gT, gGm[0], gcp, gpp);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    constexpr AscendC::RoundMode kDownT =
        std::is_same<T, bfloat16_t>::value ? AscendC::RoundMode::CAST_RINT
                                           : AscendC::RoundMode::CAST_NONE;

    const int64_t localRows = rowEnd - rowStart;
    int64_t scaleIdx = 0;

    for (int64_t r = 0; r < localRows; ++r) {
        const int64_t base = r * N;

        AscendC::Duplicate(aSum, 0.0f, static_cast<int32_t>(S));
        AscendC::Duplicate(aMax, 0.0f, static_cast<int32_t>(S));

        // ------------------------- pass A -------------------------
        for (int64_t off = 0; off < N; off += S) {
            int64_t cnt = N - off;
            if (cnt > S) {
                cnt = S;
            }
            const uint32_t cntU = static_cast<uint32_t>(cnt);
            const uint32_t bytesT = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(T)));

            {
                AscendC::DataCopyExtParams cp{1, bytesT, 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};
                auto t1 = inQ1.AllocTensor<T>();
                auto t2 = inQ2.AllocTensor<T>();
                AscendC::DataCopyPad(t1, x1Gm[base + off], cp, pp);
                AscendC::DataCopyPad(t2, x2Gm[base + off], cp, pp);
                inQ1.EnQue(t1);
                inQ2.EnQue(t2);
            }
            auto t1 = inQ1.DeQue<T>();
            auto t2 = inQ2.DeQue<T>();
            AscendC::Cast(f1, t1, AscendC::RoundMode::CAST_NONE, cntU);
            AscendC::Cast(f2, t2, AscendC::RoundMode::CAST_NONE, cntU);
            inQ1.FreeTensor(t1);
            inQ2.FreeTensor(t2);
            AscendC::Add(f1, f1, f2, cntU);  // f1 = xOut

            // xOut -> GM (narrowed to T)
            {
                auto oT = outQ.AllocTensor<T>();
                AscendC::Cast(oT, f1, kDownT, cntU);
                outQ.EnQue(oT);
                auto oS = outQ.DeQue<T>();
                AscendC::DataCopyPad(xOutGm[base + off], oS,
                                     AscendC::DataCopyExtParams{1, bytesT, 0, 0, 0});
                outQ.FreeTensor(oS);
            }

            // sum(xOut^2) accumulated lane-wise over the chunk
            AscendC::Mul(f2, f1, f1, cntU);
            AscendC::Add(aSum, aSum, f2, cntU);

            // z = xOut * gamma, kept for pass B; max|z| accumulated lane-wise
            AscendC::Cast(gf, gT[off], AscendC::RoundMode::CAST_NONE, cntU);
            AscendC::Mul(zf[off], f1, gf, cntU);
            AscendC::Abs(f2, zf[off], cntU);
            AscendC::Max(aMax, aMax, f2, cntU);
        }

        // ---------------------- per-row finalize ----------------------
        AscendC::ReduceSum<float>(red, aSum, work, static_cast<int32_t>(S));
        AscendC::ReduceMax<float>(red[64], aMax, work, static_cast<int32_t>(S));
        AscendC::Muls(red, red, invN, 8);
        AscendC::Adds(red, red, epsilon, 8);
        AscendC::Sqrt(red, red, 8);
        AscendC::PipeBarrier<PIPE_ALL>();

        const float rms = red.GetValue(0);
        const float maxAbs = red[64].GetValue(0);
        // No positive guard: NaN/inf must propagate into rstd (IEEE division).
        const float rstd = 1.0f / rms;
        const float m = maxAbs * rstd;
        // `(m < min)` propagates NaN (comparison false) and clamps only true zeros.
        const float sm = (m < 1e-12f) ? 1e-12f : m;
        const float scale = sm * kInv127;
        const float k = rstd / scale;
        slot.SetValue(static_cast<int32_t>(scaleIdx), scale);

        // ------------------------- pass B -------------------------
        for (int64_t off = 0; off < N; off += S) {
            int64_t cnt = N - off;
            if (cnt > S) {
                cnt = S;
            }
            const uint32_t cntU = static_cast<uint32_t>(cnt);
            AscendC::Muls(f1, zf[off], k, cntU);
            auto yT = yQ.AllocTensor<int8_t>();
            AscendC::Cast(hf, f1, AscendC::RoundMode::CAST_RINT, cntU);
            AscendC::Cast(yT, hf, AscendC::RoundMode::CAST_RINT, cntU);
            yQ.EnQue(yT);
            auto yS = yQ.DeQue<int8_t>();
            AscendC::DataCopyPad(yGm[base + off], yS,
                                 AscendC::DataCopyExtParams{1, cntU, 0, 0, 0});
            yQ.FreeTensor(yS);
        }

        ++scaleIdx;
        if (scaleIdx == ARDQ_SCALE_BATCH || r + 1 == localRows) {
            AscendC::PipeBarrier<PIPE_ALL>();  // scalar writes -> MTE3
            AscendC::DataCopyPad(
                sGm[r + 1 - scaleIdx], slot,
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(scaleIdx * 4), 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();  // MTE3 -> scalar (slot reuse)
            scaleIdx = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Host tiling
// ---------------------------------------------------------------------------
std::tuple<int64_t, int64_t, int64_t, int64_t> calc_ardq_tiling(int64_t numRows, int64_t rowLen)
{
    int64_t ubBytes = 192 * 1024;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t queried = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, queried);
    if (queried > 0) {
        ubBytes = static_cast<int64_t>(queried);
    }
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    if (numRows < 1) {
        numRows = 1;
    }
    int64_t numBlocks = (numRows < coreNum) ? numRows : coreNum;
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t rowsPerCore = (numRows + numBlocks - 1) / numBlocks;

    // ---- try the block path first ----
    const int64_t P = ((rowLen + 63) / 64) * 64;    // fp32 lane pitch
    const int64_t nT = ((rowLen + 15) / 16) * 16;   // natural T lane pitch
    const int64_t n8 = ((rowLen + 31) / 32) * 32;   // natural int8 lane pitch
    // per row: x1 + x2 + oT T blocks (6*nT) + fp32 xo/zb (8*P) + int8 y (n8)
    const int64_t perRow = 6 * nT + 8 * P + n8 + 512;
    // per-core fixed: fp32 gamma + fp16 stage + AR work buffer + staging + the
    // T/y last-row slack (charged once, hence the -6*nT -n8 terms)
    const int64_t fixed = 15 * P - 6 * nT - n8 + 24000;
    int64_t avail = ubBytes - fixed;
    int64_t R = (perRow > 0) ? (avail / perRow) : 0;
    if (R > ARDQ_RMAX) {
        R = ARDQ_RMAX;
    }
    if (R >= 2) {
        return std::make_tuple(numBlocks, rowsPerCore, 0, R);
    }

    // ---- chunk path fallback (rows too wide for a 2-row block) ----
    const int64_t NF = ((rowLen + 63) / 64) * 64;
    const int64_t fixed2 = NF * 4 + nT * 2 + 512 + 4096 + 512 + 8192;
    int64_t avail2 = ubBytes - fixed2;
    // per S: inQ1/inQ2/outQ 4S (depth2), yQ 2S, f1/f2/gf/accSum/accMax 4S, h 2S => 36S
    int64_t S = avail2 / 36;
    S = (S / 64) * 64;
    if (S < 64) {
        S = 64;
    }
    if (S > ARDQ_TILE_MAX) {
        S = ARDQ_TILE_MAX;
    }
    const int64_t SN = ((rowLen + 63) / 64) * 64;
    if (S > SN) {
        S = SN;
    }
    if (S < 64) {
        S = 64;
    }
    return std::make_tuple(numBlocks, rowsPerCore, S, 0);
}

// ---------------------------------------------------------------------------
// Launch wrappers (regular C functions callable from g++)
// ---------------------------------------------------------------------------
extern "C" {

void launch_ardq_half(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR xOut,
                      GM_ADDR scaleOut, int64_t numRows, int64_t rowLen, int64_t numBlocks,
                      int64_t rowsPerCore, uint32_t tileElems, int64_t rowBlock, float epsilon,
                      void* stream)
{
    if (rowBlock > 0) {
        ardq_block_kernel<half><<<numBlocks, nullptr, stream>>>(x1, x2, gamma, y, xOut, scaleOut,
                                                                numRows, rowLen, rowsPerCore,
                                                                rowBlock, epsilon);
    } else {
        ardq_chunk_kernel<half><<<numBlocks, nullptr, stream>>>(x1, x2, gamma, y, xOut, scaleOut,
                                                                numRows, rowLen, rowsPerCore,
                                                                tileElems, epsilon);
    }
}

void launch_ardq_bf16(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR xOut,
                      GM_ADDR scaleOut, int64_t numRows, int64_t rowLen, int64_t numBlocks,
                      int64_t rowsPerCore, uint32_t tileElems, int64_t rowBlock, float epsilon,
                      void* stream)
{
    if (rowBlock > 0) {
        ardq_block_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x1, x2, gamma, y, xOut,
                                                                      scaleOut, numRows, rowLen,
                                                                      rowsPerCore, rowBlock,
                                                                      epsilon);
    } else {
        ardq_chunk_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x1, x2, gamma, y, xOut,
                                                                      scaleOut, numRows, rowLen,
                                                                      rowsPerCore, tileElems,
                                                                      epsilon);
    }
}

}  // extern "C"
