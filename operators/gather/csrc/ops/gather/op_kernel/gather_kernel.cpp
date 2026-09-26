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
 * \file gather_kernel.cpp
 * \brief Gather kernel (torch.gather semantics) - kernel + launch wrappers (bisheng + -xasc)
 *
 * y[c0,..,c_{n-1}] = x[..., idx, ...]   (index taken along `dim`)
 *
 * Fast path ("windowed vector gather")
 *   - a window of x, [xMid rows][W columns], is staged into UB,
 *   - one `Gather` per row block collects the requested elements with byte offsets
 *        off[i] = idx[i] * (W*sizeof(T)) + (i % W) * sizeof(T)
 *     (the (i % W)*sizeof(T) term is a periodic ramp precomputed into a UB pattern buffer),
 *   - the gathered rows are written back.
 *   W is chosen so that W*sizeof(T) and W*sizeof(index) are 32-byte multiples, which makes the
 *   implicit DataCopyPad UB row pitch (ceil32(blockLen)) equal to exactly W elements, so the
 *   index/offset/output row pitches all agree.
 *   Multi-block DataCopyPad is used with byte-unit srcStride/dstStride (a2 semantics), with the
 *   block count kept <= G_MAX_BLOCKS.
 *
 * Fast path for int64 x ("paired int32"): the vector Gather has no b64 form on this target, so
 *   the int64 elements are processed as their two little-endian int32 halves. The destination is
 *   a contiguous int32 array of 2*R values (out32[2*i+h] = half h of int64 element i) and the
 *   byte offset of destination position q is
 *        off[q] = idx(q>>1) * (W*8) + (q % 2W) * 4
 *   The index value is duplicated by a first in-UB gather using the byte offsets (q>>1)*sizeof(IT);
 *   the result buffer is then overwritten in place by the real gather and copied out as int64.
 *
 * Slow path ("per-element gather")
 *   General fallback: one element per 32-byte UB slot, GM->UB->GM, driven by an output odometer,
 *   with G_SUB_BATCH elements staged between two hardware-event barriers. Measured: the tiny
 *   copies themselves dominate (the barriers are not the bottleneck - raising G_SUB_BATCH from 64
 *   to 512 made c4/c6 slower), and batching the sub-32-byte stores into one multi-block copy is
 *   also slower, so per-element DataCopyPad is kept on both directions.
 *
 * int64 index values are narrowed with a vector `Gather` over the little-endian low words
 * (there is no int64 -> int32 Cast on this target).
 */

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "gather_launch.h"

constexpr static int64_t G_MAX_NDIM = 8;
constexpr static int64_t G_SLOT_BYTES = 32;
constexpr static int64_t G_SUB_BATCH = 64;
constexpr static int64_t G_MAX_BLOCKS = 2048; // max blockCount accepted by multi-block DataCopyPad
constexpr static int32_t G_EVT_M2M3 = 5;
constexpr static int32_t G_EVT_M3M2 = 6;

__aicore__ inline int64_t gGcd(int64_t a, int64_t b)
{
    if (a < 0) { a = -a; }
    if (b < 0) { b = -b; }
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

__aicore__ inline int64_t gLcm(int64_t a, int64_t b)
{
    if (a == 0 || b == 0) { return 0; }
    return a / gGcd(a, b) * b;
}

// pat[i] = (i % period) * unitBytes, for i in [0, n)
// Built with vector ops only: one period via CreateVecIndex/Muls, then doubling with Adds by 0.
__aicore__ inline void gBuildPeriodicRamp(
    AscendC::LocalTensor<int32_t> pat, int64_t period, int64_t unitBytes, int64_t n)
{
    if (n <= 0) { return; }
    if (period <= 0) { period = 1; }
    if (period == 1) {
        AscendC::Duplicate(pat, static_cast<int32_t>(0), static_cast<uint32_t>(n));
        return;
    }
    const int64_t base = (period < n) ? period : n;
    AscendC::CreateVecIndex(pat, static_cast<int32_t>(0), static_cast<uint32_t>(base));
    AscendC::Muls(pat, pat, static_cast<int32_t>(unitBytes), static_cast<uint32_t>(base));
    if (base < period) { return; } // single partial ramp is already the answer
    if ((period * unitBytes) % 32 != 0) { // safety net, host keeps this unreachable
        for (int64_t i = 0; i < n; ++i) {
            pat.SetValue(i, static_cast<int32_t>((i % period) * unitBytes));
        }
        return;
    }
    AscendC::PipeBarrier<PIPE_V>();
    int64_t done = base; // always a multiple of period
    while (done < n) {
        int64_t chunk = n - done;
        if (chunk > done) { chunk = done; }
        AscendC::Adds(pat[done], pat, static_cast<int32_t>(0), static_cast<uint32_t>(chunk));
        AscendC::PipeBarrier<PIPE_V>();
        done += chunk;
    }
}

// d[i] = (i >> 1) * step, for i in [0, n)
__aicore__ inline void gBuildDupRamp(AscendC::LocalTensor<int32_t> d, int64_t step, int64_t n)
{
    if (n <= 0) { return; }
    AscendC::CreateVecIndex(d, static_cast<int32_t>(0), static_cast<uint32_t>(n));
    AscendC::ShiftRight(d, d, static_cast<int32_t>(1), static_cast<uint32_t>(n));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(d, d, static_cast<int32_t>(step), static_cast<uint32_t>(n));
}

// ============================================================================================
// Fast path
// ============================================================================================
template <typename T, typename IT>
__global__ __aicore__ void gather_fast_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t n, int64_t dim,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,
    int64_t xMid, int64_t xRowStride,
    int64_t outerOut, int64_t mid, int64_t innerOut,
    int64_t Cc, int64_t RB, int64_t unitsPerA, int64_t numTasks,
    int64_t numBlocks)
{
    (void)n;
    (void)outerOut;
    const int64_t uB = static_cast<int64_t>(sizeof(T));
    const int64_t iB = static_cast<int64_t>(sizeof(IT));
    const int64_t g = gLcm(32 / gGcd(uB, 32), 32 / gGcd(iB, 32));

    int64_t Ctail = innerOut - (unitsPerA - 1) * Cc;
    if (Ctail <= 0 || Ctail > Cc) { Ctail = Cc; }
    int64_t Wtail = ((Ctail + g - 1) / g) * g;
    if (Wtail > Cc) { Wtail = Cc; }
    if (Wtail < Ctail) { Wtail = Ctail; }

    const int64_t Rmax = RB * Cc; // maximum number of elements handled per inner iteration

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> winBuf, idxBuf, offBuf, outBuf, patBuf, patTBuf,
        patNBuf;
    pipe.InitBuffer(winBuf, xMid * Cc * uB);
    pipe.InitBuffer(idxBuf, Rmax * iB);
    pipe.InitBuffer(offBuf, Rmax * 4);
    pipe.InitBuffer(outBuf, Rmax * uB);
    pipe.InitBuffer(patBuf, Rmax * 4);
    pipe.InitBuffer(patTBuf, Rmax * 4);
    pipe.InitBuffer(patNBuf, Rmax * 4);

    AscendC::LocalTensor<T> winL = winBuf.Get<T>();
    AscendC::LocalTensor<int8_t> idx8 = idxBuf.Get<int8_t>();
    AscendC::LocalTensor<int32_t> idx32 = idxBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> offI = offBuf.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> offU = offBuf.Get<uint32_t>();
    AscendC::LocalTensor<T> outL = outBuf.Get<T>();
    AscendC::LocalTensor<int32_t> patA = patBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> patTA = patTBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> patNI = patNBuf.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> patNU = patNBuf.Get<uint32_t>();

    // column patterns: period Cc and period Wtail, values (i % period) * sizeof(T)
    gBuildPeriodicRamp(patA, Cc, uB, Rmax);
    gBuildPeriodicRamp(patTA, Wtail, uB, Rmax);
    // patNI[i] = i*sizeof(index): picks the little-endian low word of the i-th index element
    AscendC::CreateVecIndex(patNI, static_cast<int32_t>(0), static_cast<uint32_t>(Rmax));
    AscendC::Muls(patNI, patNI, static_cast<int32_t>(iB), static_cast<uint32_t>(Rmax));
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<int8_t> idxGm8;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    idxGm8.SetGlobalBuffer((__gm__ int8_t *)index);

    const int64_t xs[G_MAX_NDIM] = {xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7};
    const int64_t os[G_MAX_NDIM] = {os0, os1, os2, os3, os4, os5, os6, os7};

    AscendC::DataCopyPadExtParams<T> padT{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int8_t> padI8{false, 0, 0, 0};

    const int64_t bid = static_cast<int64_t>(AscendC::GetBlockIdx());

    for (int64_t tsk = bid; tsk < numTasks; tsk += numBlocks) {
        const int64_t a = tsk / unitsPerA;
        const int64_t uu = tsk - a * unitsPerA;
        const int64_t c0 = uu * Cc;
        int64_t Cct = innerOut - c0;
        if (Cct > Cc) { Cct = Cc; }
        const bool isTail = (Cct != Cc);
        const int64_t W = isTail ? Wtail : Cc;   // strip width in elements
        const int64_t cw = c0 + Cct - W;         // strip start column
        const bool packed = (W == innerOut);     // rows contiguous in GM
        const bool fused = packed && (W == 1);   // single contiguous block, offsets = idx*sizeof(T)
        const int64_t pitchBytes = W * uB;
        const int64_t idxBytes = W * iB;

        int64_t baseA = 0;
        int64_t tt = a;
        for (int64_t j = dim - 1; j >= 0; --j) {
            const int64_t cc = tt % os[j];
            tt /= os[j];
            baseA += cc * xs[j];
        }

        // ---------------- stage the window of x into UB ----------------
        if (packed) {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(xMid * W * uB), 0, 0, 0};
            AscendC::DataCopyPad(winL, xGm[baseA + cw], cp, padT);
        } else {
            const int64_t gapBytes = (xRowStride - W) * uB;
            int64_t done = 0;
            while (done < xMid) {
                int64_t cnt = xMid - done;
                if (cnt > G_MAX_BLOCKS) { cnt = G_MAX_BLOCKS; }
                AscendC::DataCopyExtParams cp{
                    static_cast<uint16_t>(cnt),
                    static_cast<uint32_t>(pitchBytes),
                    static_cast<uint32_t>(gapBytes), 0, 0};
                AscendC::DataCopyPad(winL[done * W],
                                     xGm[baseA + done * xRowStride + cw], cp, padT);
                done += cnt;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        for (int64_t b = 0; b < mid; b += RB) {
            int64_t rbs = mid - b;
            if (rbs > RB) { rbs = RB; }
            const int64_t row0 = a * mid + b;
            const int64_t cell0 = row0 * innerOut + cw;

            if (fused) {
                const int64_t R = rbs;
                AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(R * iB), 0, 0, 0};
                AscendC::DataCopyPad(idx8, idxGm8[cell0 * iB], cp, padI8);
                AscendC::PipeBarrier<PIPE_ALL>();
                if constexpr (std::is_same<IT, int32_t>::value) {
                    AscendC::Muls(offI, idx32, static_cast<int32_t>(uB), static_cast<uint32_t>(R));
                } else {
                    AscendC::Gather(offI, idx32, patNU, static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(R));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(offI, offI, static_cast<int32_t>(uB), static_cast<uint32_t>(R));
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(outL, winL, offU, static_cast<uint32_t>(0), static_cast<uint32_t>(R));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams wcp{1, static_cast<uint32_t>(R * uB), 0, 0, 0};
                AscendC::DataCopyPad(yGm[cell0], outL, wcp);
                AscendC::PipeBarrier<PIPE_ALL>();
                continue;
            }

            const int64_t R = rbs * W;

            // ---- index block (rows land at pitch W elements) ----
            {
                AscendC::DataCopyExtParams cp{
                    static_cast<uint16_t>(rbs),
                    static_cast<uint32_t>(idxBytes),
                    static_cast<uint32_t>((innerOut - W) * iB), 0, 0};
                AscendC::DataCopyPad(idx8, idxGm8[cell0 * iB], cp, padI8);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // ---- byte offsets into the window for the whole block ----
            if constexpr (std::is_same<IT, int32_t>::value) {
                AscendC::Muls(offI, idx32, static_cast<int32_t>(pitchBytes), static_cast<uint32_t>(R));
            } else {
                AscendC::Gather(offI, idx32, patNU, static_cast<uint32_t>(0),
                                static_cast<uint32_t>(R));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(offI, offI, static_cast<int32_t>(pitchBytes), static_cast<uint32_t>(R));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(offI, offI, isTail ? patTA : patA, static_cast<uint32_t>(R));
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Gather(outL, winL, offU, static_cast<uint32_t>(0), static_cast<uint32_t>(R));
            AscendC::PipeBarrier<PIPE_ALL>();

            // ---- write back ----
            {
                AscendC::DataCopyExtParams cp{
                    static_cast<uint16_t>(rbs),
                    static_cast<uint32_t>(pitchBytes), 0,
                    static_cast<uint32_t>((innerOut - W) * uB), 0};
                AscendC::DataCopyPad(yGm[cell0], outL, cp);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ============================================================================================
// Fast path for int64 x (paired int32 halves)
// ============================================================================================
template <typename IT>
__global__ __aicore__ void gather_fast64_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t n, int64_t dim,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,
    int64_t xMid, int64_t xRowStride,
    int64_t outerOut, int64_t mid, int64_t innerOut,
    int64_t Cc, int64_t RB, int64_t unitsPerA, int64_t numTasks,
    int64_t numBlocks)
{
    (void)n;
    (void)outerOut;
    const int64_t uB = 8;  // int64 elements in the window / output
    const int64_t iB = static_cast<int64_t>(sizeof(IT));
    const int64_t g = gLcm(32 / gGcd(uB, 32), 32 / gGcd(iB, 32));

    int64_t Ctail = innerOut - (unitsPerA - 1) * Cc;
    if (Ctail <= 0 || Ctail > Cc) { Ctail = Cc; }
    int64_t Wtail = ((Ctail + g - 1) / g) * g;
    if (Wtail > Cc) { Wtail = Cc; }
    if (Wtail < Ctail) { Wtail = Ctail; }

    const int64_t RmaxE = RB * Cc;      // int64 elements per inner iteration
    const int64_t Rmax32 = 2 * RmaxE;   // int32 halves per inner iteration

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> winBuf, idxBuf, offBuf, outBuf, patBuf, patTBuf,
        patDBuf;
    pipe.InitBuffer(winBuf, xMid * Cc * uB);
    pipe.InitBuffer(idxBuf, RmaxE * iB);
    pipe.InitBuffer(offBuf, Rmax32 * 4);
    pipe.InitBuffer(outBuf, Rmax32 * 4);
    pipe.InitBuffer(patBuf, Rmax32 * 4);
    pipe.InitBuffer(patTBuf, Rmax32 * 4);
    pipe.InitBuffer(patDBuf, Rmax32 * 4);

    AscendC::LocalTensor<int64_t> winL = winBuf.Get<int64_t>();
    AscendC::LocalTensor<int32_t> win32 = winBuf.Get<int32_t>();
    AscendC::LocalTensor<int8_t> idx8 = idxBuf.Get<int8_t>();
    AscendC::LocalTensor<int32_t> idx32 = idxBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> offI = offBuf.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> offU = offBuf.Get<uint32_t>();
    AscendC::LocalTensor<int32_t> out32 = outBuf.Get<int32_t>();
    AscendC::LocalTensor<int64_t> out64 = outBuf.Get<int64_t>();
    AscendC::LocalTensor<int32_t> patA = patBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> patTA = patTBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> patD = patDBuf.Get<int32_t>();
    AscendC::LocalTensor<uint32_t> patDU = patDBuf.Get<uint32_t>();

    // patA[q] = (q % 2W)*4  (byte offset of the half inside the row), patD[q] = (q>>1)*iB
    gBuildPeriodicRamp(patA, 2 * Cc, 4, Rmax32);
    gBuildPeriodicRamp(patTA, 2 * Wtail, 4, Rmax32);
    gBuildDupRamp(patD, iB, Rmax32);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::GlobalTensor<int64_t> xGm;
    AscendC::GlobalTensor<int64_t> yGm;
    AscendC::GlobalTensor<int8_t> idxGm8;
    xGm.SetGlobalBuffer((__gm__ int64_t *)x);
    yGm.SetGlobalBuffer((__gm__ int64_t *)y);
    idxGm8.SetGlobalBuffer((__gm__ int8_t *)index);

    const int64_t xs[G_MAX_NDIM] = {xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7};
    const int64_t os[G_MAX_NDIM] = {os0, os1, os2, os3, os4, os5, os6, os7};

    AscendC::DataCopyPadExtParams<int64_t> padT{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int8_t> padI8{false, 0, 0, 0};

    const int64_t bid = static_cast<int64_t>(AscendC::GetBlockIdx());

    for (int64_t tsk = bid; tsk < numTasks; tsk += numBlocks) {
        const int64_t a = tsk / unitsPerA;
        const int64_t uu = tsk - a * unitsPerA;
        const int64_t c0 = uu * Cc;
        int64_t Cct = innerOut - c0;
        if (Cct > Cc) { Cct = Cc; }
        const bool isTail = (Cct != Cc);
        const int64_t W = isTail ? Wtail : Cc;   // strip width in int64 elements
        const int64_t cw = c0 + Cct - W;
        const bool packed = (W == innerOut);
        const int64_t pitchBytes = W * uB;
        const int64_t idxBytes = W * iB;

        int64_t baseA = 0;
        int64_t tt = a;
        for (int64_t j = dim - 1; j >= 0; --j) {
            const int64_t cc = tt % os[j];
            tt /= os[j];
            baseA += cc * xs[j];
        }

        // ---------------- stage the window of x into UB ----------------
        if (packed) {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(xMid * W * uB), 0, 0, 0};
            AscendC::DataCopyPad(winL, xGm[baseA + cw], cp, padT);
        } else {
            const int64_t gapBytes = (xRowStride - W) * uB;
            int64_t done = 0;
            while (done < xMid) {
                int64_t cnt = xMid - done;
                if (cnt > G_MAX_BLOCKS) { cnt = G_MAX_BLOCKS; }
                AscendC::DataCopyExtParams cp{
                    static_cast<uint16_t>(cnt),
                    static_cast<uint32_t>(pitchBytes),
                    static_cast<uint32_t>(gapBytes), 0, 0};
                AscendC::DataCopyPad(winL[done * W],
                                     xGm[baseA + done * xRowStride + cw], cp, padT);
                done += cnt;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        for (int64_t b = 0; b < mid; b += RB) {
            int64_t rbs = mid - b;
            if (rbs > RB) { rbs = RB; }
            const int64_t row0 = a * mid + b;
            const int64_t cell0 = row0 * innerOut + cw;

            const int64_t R = rbs * W;       // int64 elements in this block
            const int64_t R32 = 2 * R;       // int32 halves in this block

            // ---- index block: rows land at pitch W*sizeof(IT) bytes ----
            {
                AscendC::DataCopyExtParams cp{
                    static_cast<uint16_t>(rbs),
                    static_cast<uint32_t>(idxBytes),
                    static_cast<uint32_t>((innerOut - W) * iB), 0, 0};
                AscendC::DataCopyPad(idx8, idxGm8[cell0 * iB], cp, padI8);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // ---- duplicate the per-element index value over its two halves ----
            AscendC::Gather(out32, idx32, patDU, static_cast<uint32_t>(0),
                            static_cast<uint32_t>(R32));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(out32, out32, static_cast<int32_t>(pitchBytes), static_cast<uint32_t>(R32));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(offI, out32, isTail ? patTA : patA, static_cast<uint32_t>(R32));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(out32, win32, offU, static_cast<uint32_t>(0),
                            static_cast<uint32_t>(R32));
            AscendC::PipeBarrier<PIPE_ALL>();

            // ---- write back (int64 rows) ----
            {
                AscendC::DataCopyExtParams cp{
                    static_cast<uint16_t>(rbs),
                    static_cast<uint32_t>(pitchBytes), 0,
                    static_cast<uint32_t>((innerOut - W) * uB), 0};
                AscendC::DataCopyPad(yGm[cell0], out64, cp);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

// ============================================================================================
// Slow path: general per-element gather
// ============================================================================================
template <typename T, typename IT>
__global__ __aicore__ void gather_slow_kernel(
    GM_ADDR x, GM_ADDR index, GM_ADDR y,
    int64_t n, int64_t dim,
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,
    int64_t xMid, int64_t xNumel, int64_t totalOut,
    int64_t blockLength, int64_t tileElems, int64_t numBlocks)
{
    (void)numBlocks;
    const int64_t slotElems = G_SLOT_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t bid = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t start = bid * blockLength;
    if (start >= totalOut) { return; }
    int64_t end = start + blockLength;
    if (end > totalOut) { end = totalOut; }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> idxQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> slotBuf;
    pipe.InitBuffer(idxQ, 1, tileElems * static_cast<int64_t>(sizeof(IT)));
    pipe.InitBuffer(slotBuf, G_SUB_BATCH * G_SLOT_BYTES);

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<IT> idxGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);
    idxGm.SetGlobalBuffer((__gm__ IT *)index);

    const int64_t xs[G_MAX_NDIM] = {xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7};
    const int64_t os[G_MAX_NDIM] = {os0, os1, os2, os3, os4, os5, os6, os7};

    int64_t Os[G_MAX_NDIM];
    int64_t c[G_MAX_NDIM];
    Os[n - 1] = 1;
    for (int64_t j = n - 2; j >= 0; --j) { Os[j] = Os[j + 1] * os[j + 1]; }
    int64_t rem = start;
    for (int64_t j = 0; j < n; ++j) {
        c[j] = rem / Os[j];
        rem -= c[j] * Os[j];
    }
    for (int64_t j = n; j < G_MAX_NDIM; ++j) { c[j] = 0; }
    int64_t baseSrc = 0;
    for (int64_t j = 0; j < n; ++j) {
        if (j != dim) { baseSrc += c[j] * xs[j]; }
    }

    const int64_t xStrideDim = xs[dim];
    AscendC::DataCopyPadExtParams<IT> padI{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padT{false, 0, 0, 0};
    AscendC::DataCopyExtParams oneElem{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};

    AscendC::LocalTensor<T> slotL = slotBuf.Get<T>();

    for (int64_t o0 = start; o0 < end; o0 += tileElems) {
        int64_t t = end - o0;
        if (t > tileElems) { t = tileElems; }

        AscendC::LocalTensor<IT> idxTile = idxQ.AllocTensor<IT>();
        AscendC::DataCopyExtParams icp{
            1, static_cast<uint32_t>(t * static_cast<int64_t>(sizeof(IT))), 0, 0, 0};
        AscendC::DataCopyPad(idxTile, idxGm[o0], icp, padI);
        idxQ.EnQue(idxTile);
        idxTile = idxQ.DeQue<IT>();

        for (int64_t s = 0; s < t; s += G_SUB_BATCH) {
            int64_t b = t - s;
            if (b > G_SUB_BATCH) { b = G_SUB_BATCH; }
            int64_t offs[G_SUB_BATCH];
            for (int64_t j = 0; j < b; ++j) {
                int64_t v = static_cast<int64_t>(idxTile.GetValue(s + j));
                if (v < 0 || v >= xMid) { v = 0; }
                int64_t off = baseSrc + v * xStrideDim;
                if (off < 0 || off >= xNumel) { off = 0; }
                offs[j] = off;
                for (int64_t d = n - 1; d >= 0; --d) {
                    if (c[d] < os[d] - 1) {
                        c[d] += 1;
                        if (d != dim) { baseSrc += xs[d]; }
                        break;
                    }
                    c[d] = 0;
                    if (d != dim) { baseSrc -= (os[d] - 1) * xs[d]; }
                }
            }

            for (int64_t j = 0; j < b; ++j) {
                AscendC::DataCopyPad(slotL[j * slotElems], xGm[offs[j]], oneElem, padT);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(G_EVT_M2M3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(G_EVT_M2M3);
            for (int64_t j = 0; j < b; ++j) {
                AscendC::DataCopyPad(yGm[o0 + s + j], slotL[j * slotElems], oneElem);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(G_EVT_M3M2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(G_EVT_M3M2);
        }

        idxQ.FreeTensor(idxTile);
    }
}

// ============================================================================================
// Launch wrappers
// ============================================================================================
extern "C" {

#define G_DEF_FAST(NAME, T, IT)                                                              \
    void gather_launch_fast_##NAME(G_ARGS_DECL, void* stream)                                \
    {                                                                                        \
        gather_fast_kernel<T, IT><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(    \
            G_ARGS_CALL);                                                                    \
    }

#define G_DEF_FAST64(NAME, IT)                                                               \
    void gather_launch_fast64_##NAME(G_ARGS_DECL, void* stream)                              \
    {                                                                                        \
        gather_fast64_kernel<IT><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(     \
            G_ARGS_CALL);                                                                    \
    }

#define G_DEF_SLOW2(NAME, T, IT)                                                             \
    void gather_launch_slow2_##NAME(G_SLOW_ARGS_DECL, void* stream)                           \
    {                                                                                        \
        gather_slow_kernel<T, IT><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(    \
            G_SLOW_ARGS_CALL);                                                               \
    }

G_DEF_FAST(f32_i32, float, int32_t)
G_DEF_FAST(f32_i64, float, int64_t)
G_DEF_FAST(f16_i32, half, int32_t)
G_DEF_FAST(f16_i64, half, int64_t)
G_DEF_FAST(bf16_i32, bfloat16_t, int32_t)
G_DEF_FAST(bf16_i64, bfloat16_t, int64_t)
G_DEF_FAST(i8_i32, int8_t, int32_t)
G_DEF_FAST(i8_i64, int8_t, int64_t)
G_DEF_FAST(i32_i32, int32_t, int32_t)
G_DEF_FAST(i32_i64, int32_t, int64_t)

G_DEF_FAST64(i32, int32_t)
G_DEF_FAST64(i64, int64_t)

G_DEF_SLOW2(f32_i8, float, int8_t)
G_DEF_SLOW2(f32_i32, float, int32_t)
G_DEF_SLOW2(f32_i64, float, int64_t)
G_DEF_SLOW2(f16_i8, half, int8_t)
G_DEF_SLOW2(f16_i32, half, int32_t)
G_DEF_SLOW2(f16_i64, half, int64_t)
G_DEF_SLOW2(bf16_i8, bfloat16_t, int8_t)
G_DEF_SLOW2(bf16_i32, bfloat16_t, int32_t)
G_DEF_SLOW2(bf16_i64, bfloat16_t, int64_t)
G_DEF_SLOW2(i8_i8, int8_t, int8_t)
G_DEF_SLOW2(i8_i32, int8_t, int32_t)
G_DEF_SLOW2(i8_i64, int8_t, int64_t)
G_DEF_SLOW2(i32_i8, int32_t, int8_t)
G_DEF_SLOW2(i32_i32, int32_t, int32_t)
G_DEF_SLOW2(i32_i64, int32_t, int64_t)
G_DEF_SLOW2(i64_i8, int64_t, int8_t)
G_DEF_SLOW2(i64_i32, int64_t, int32_t)
G_DEF_SLOW2(i64_i64, int64_t, int64_t)

#undef G_DEF_FAST
#undef G_DEF_FAST64
#undef G_DEF_SLOW2
}
