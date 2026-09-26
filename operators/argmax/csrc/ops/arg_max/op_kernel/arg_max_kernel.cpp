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
 * \file arg_max_kernel.cpp
 * \brief ArgMax device kernels + host tiling (compiled with bisheng / -xasc).
 *
 *   indices = argmax(input, dim)
 *
 * The ND tensor is viewed as (outer, reduceLen, inner) with `dim` the reduce axis.
 * The output viewed as (outer, inner) is contiguous for both keepdim settings, so the
 * kernels only ever write that flattened index tensor (int64).
 *
 *   mode 0 (inner <= 1): the reduce axis is innermost, so a row is a contiguous run of
 *       reduceLen elements.  Rows are cut into chunks of at most kSegRow elements.  Every
 *       (row, chunk) pair is one work unit.  A unit loads its chunk contiguously in
 *       fp32 and reduces it with ReduceMax(calIndex = true), which returns both the
 *       maximum value and the raw bit pattern of the smallest index holding it.  When a
 *       row spans exactly one chunk the final int64 index is written directly; otherwise
 *       the per-chunk (value, index) pairs go to a small device workspace and a second
 *       sweep merges them (strictly greater keeps the earlier chunk).
 *
 *   mode 1 (inner > 1): the reduce axis is a middle axis.  Every work unit owns wTile
 *       columns of one outer slice and a slice of the reduce axis.  Rows are streamed in
 *       and a running (value, index) vector pair is advanced with Compare / Select / Max,
 *       so every column is its own argmax instance.  Partial results of the reduce-axis
 *       split are merged by a second sweep with the very same vector pattern.
 *
 * All vector arithmetic happens in fp32.  half / bfloat16 are widened with CAST_NONE and
 * int32 / int64 with CAST_RINT.  Before comparing, every element is replaced by a ranking
 * key: NaN -> +inf, otherwise the value itself.  ReduceMax already propagates NaN as the
 * maximum, so the row path needs no key; the scalar merge maps NaN to +inf so that a NaN
 * chunk compares as the maximum.  The column path builds the key with
 * Compares(src, -inf, GE) + Select(..., +inf) : the mask is TRUE for every ordered value
 * (including +-inf) and FALSE for a NaN, so ordered lanes keep their value while a NaN
 * lane takes the +inf scalar.
 *
 * int32 / int64 are converted through fp32, which is exact while |value| < 2^24.  Larger
 * integer magnitudes are rounded to the nearest fp32 and can therefore lose the strict
 * ordering between two very close integers (documented limitation, no host fallback).
 */

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "arg_max_launch.h"

using namespace AscendC;

namespace {

constexpr int64_t kSegRow = 4096;   // row mode chunk width (multiple of 64)
constexpr int64_t kBorder = 1024;   // column mode: maximum column tile width

__aicore__ inline float ArgMaxNegInf()
{
    uint32_t b = 0xFF800000u;
    return *reinterpret_cast<float*>(&b);
}

__aicore__ inline float ArgMaxPosInf()
{
    uint32_t b = 0x7F800000u;
    return *reinterpret_cast<float*>(&b);
}

/*! \brief NaN -> +inf ranking key for a scalar value (row mode merge). */
__aicore__ inline float ArgMaxKeyFromValue(float v)
{
    uint32_t b = *reinterpret_cast<uint32_t*>(&v);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) {
        return ArgMaxPosInf();
    }
    return v;
}

__aicore__ inline int64_t ArgMaxUp(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

/*!
 * \brief dtype -> fp32 working buffer, cnt elements.  cnt is a multiple of 8 for float
 *        and a multiple of 64 otherwise.
 */
template <typename T>
__aicore__ inline void ArgMaxToFloat(const LocalTensor<float>& dst, const LocalTensor<T>& src, uint32_t cnt)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy(dst, src, cnt);
    } else if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        Cast(dst, src, RoundMode::CAST_NONE, cnt);
    } else {
        Cast(dst, src, RoundMode::CAST_RINT, cnt);
    }
}

/*!
 * \brief reduce one contiguous chunk of a row.  Writes the maximum value and the raw
 *        index bit pattern into `dp`.
 */
template <typename T>
__aicore__ inline void ArgMaxReduceChunk(const GlobalTensor<T>& xG, int64_t srcOff, int64_t len,
                                         TQue<QuePosition::VECIN, 2>& inQ,
                                         const LocalTensor<float>& fBuf,
                                         const LocalTensor<float>& workBuf,
                                         const LocalTensor<float>& dp)
{
    LocalTensor<T> tile = inQ.AllocTensor<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(tile, xG[srcOff], cp, pp);
    inQ.EnQue(tile);
    tile = inQ.DeQue<T>();
    if constexpr (std::is_same<T, float>::value) {
        ArgMaxToFloat<T>(fBuf, tile, static_cast<uint32_t>(ArgMaxUp(len, 8)));
    } else {
        ArgMaxToFloat<T>(fBuf, tile, static_cast<uint32_t>(ArgMaxUp(len, 64)));
    }
    inQ.FreeTensor(tile);
    ReduceMax<float>(dp, fBuf, workBuf, static_cast<int32_t>(len), true);
    PipeBarrier<PIPE_ALL>();
}

// ---------------------------------------------------------------------------
// row mode, single chunk per row: write the int64 index directly
//
// A batch of `rbRow` consecutive rows is a contiguous run of GM, so it is fetched with
// a single strided DataCopyPad (blockCount = rows, blockLen = one row, srcStride = 0).
// Every block of a non 32B aligned blockLen is grown to 32B by the hardware, so the UB
// rows land at pitch = ALIGN32(reduceLen * sizeof(T)) / sizeof(T) and each row starts on
// a 32B boundary - exactly what ReduceMax needs.  One batch therefore costs one DMA
// latency instead of one per row, which is what dominates short-row shapes.
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void argmax_row_direct_kernel(GM_ADDR xGm_, GM_ADDR yGm_,
                                                    int64_t outer, int64_t reduceLen,
                                                    int64_t pitch, int64_t rbRow,
                                                    int64_t batches, int64_t upb)
{
#ifdef KERNEL_TASK_TYPE_DEFAULT
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    GlobalTensor<T> xG;
    xG.SetGlobalBuffer((__gm__ T*)xGm_);
    GlobalTensor<int64_t> yG;
    yG.SetGlobalBuffer((__gm__ int64_t*)yGm_);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TBuf<TPosition::VECCALC> fBuf, workBuf, pairBuf;
    TBuf<TPosition::VECOUT> outBuf;
    pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(rbRow * pitch * static_cast<int64_t>(sizeof(T))));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(fBuf, static_cast<uint32_t>(rbRow * pitch * 4));
    }
    pipe.InitBuffer(workBuf, 8192);
    pipe.InitBuffer(pairBuf, static_cast<uint32_t>(rbRow * 32));
    pipe.InitBuffer(outBuf, static_cast<uint32_t>(rbRow * 8 + 32));

    LocalTensor<float> work = workBuf.Get<float>();
    LocalTensor<float> dp = pairBuf.Get<float>();
    LocalTensor<uint32_t> dpu = dp.ReinterpretCast<uint32_t>();
    LocalTensor<int64_t> ob = outBuf.Get<int64_t>();

    int64_t b = static_cast<int64_t>(GetBlockIdx());
    int64_t b0 = b * upb;
    int64_t b1 = b0 + upb;
    if (b1 > batches) {
        b1 = batches;
    }
    if (b0 >= batches) {
        return;
    }

    for (int64_t bi = b0; bi < b1; ++bi) {
        int64_t r0 = bi * rbRow;
        int64_t nb = outer - r0;
        if (nb > rbRow) {
            nb = rbRow;
        }
        LocalTensor<T> tile = inQ.AllocTensor<T>();
        DataCopyExtParams cp{static_cast<uint16_t>(nb),
                             static_cast<uint32_t>(reduceLen * static_cast<int64_t>(sizeof(T))),
                             0, 0, 0};
        DataCopyPadExtParams<T> pp{false, 0, 0, 0};
        DataCopyPad(tile, xG[r0 * reduceLen], cp, pp);
        inQ.EnQue(tile);
        tile = inQ.DeQue<T>();

        LocalTensor<float> fsrc;
        if constexpr (std::is_same<T, float>::value) {
            fsrc = tile;
        } else {
            LocalTensor<float> f = fBuf.Get<float>();
            ArgMaxToFloat<T>(f, tile, static_cast<uint32_t>(nb * pitch));
            inQ.FreeTensor(tile);
            fsrc = f;
        }

        for (int64_t k = 0; k < nb; ++k) {
            ReduceMax<float>(dp[k * 8], fsrc[k * pitch], work, static_cast<int32_t>(reduceLen), true);
        }
        PipeBarrier<PIPE_ALL>();
        for (int64_t k = 0; k < nb; ++k) {
            ob.SetValue(k, static_cast<int64_t>(dpu.GetValue(k * 8 + 1)));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams op{1, static_cast<uint32_t>(nb * 8), 0, 0, 0};
        DataCopyPad(yG[r0], ob, op);
        PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same<T, float>::value) {
            inQ.FreeTensor(tile);
        }
    }
}

// ---------------------------------------------------------------------------
// row mode, multi chunk per row: per (row, chunk) partial -> workspace
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void argmax_row_partial_kernel(GM_ADDR xGm_, GM_ADDR wsGm_,
                                                     int64_t reduceLen, int64_t nChunk,
                                                     int64_t segRow, int64_t units,
                                                     int64_t upb)
{
#ifdef KERNEL_TASK_TYPE_DEFAULT
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    constexpr int64_t kBatch = 32;

    GlobalTensor<T> xG;
    xG.SetGlobalBuffer((__gm__ T*)xGm_);
    GlobalTensor<float> wsG;
    wsG.SetGlobalBuffer((__gm__ float*)wsGm_);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TBuf<TPosition::VECCALC> fBuf, workBuf, pairBuf;
    TBuf<TPosition::VECOUT> wsBuf;
    pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(segRow * static_cast<int64_t>(sizeof(T))));
    pipe.InitBuffer(fBuf, static_cast<uint32_t>(segRow * 4));
    pipe.InitBuffer(workBuf, static_cast<uint32_t>(segRow * 4));
    pipe.InitBuffer(pairBuf, 64);
    pipe.InitBuffer(wsBuf, static_cast<uint32_t>(kBatch * 8));

    LocalTensor<float> f = fBuf.Get<float>();
    LocalTensor<float> work = workBuf.Get<float>();
    LocalTensor<float> dp = pairBuf.Get<float>();
    LocalTensor<float> wl = wsBuf.Get<float>();

    int64_t b = static_cast<int64_t>(GetBlockIdx());
    int64_t u0 = b * upb;
    int64_t u1 = u0 + upb;
    if (u1 > units) {
        u1 = units;
    }
    if (u0 >= units) {
        return;
    }

    int64_t k = 0;
    int64_t startU = u0;
    for (int64_t u = u0; u < u1; ++u) {
        int64_t r = u / nChunk;
        int64_t c = u - r * nChunk;
        int64_t off = c * segRow;
        int64_t len = reduceLen - off;
        if (len > segRow) {
            len = segRow;
        }
        ArgMaxReduceChunk<T>(xG, r * reduceLen + off, len, inQ, f, work, dp);
        float v = ArgMaxKeyFromValue(dp.GetValue(0));
        // ReduceMax returns the index inside this chunk; the merge works on global indices.
        int64_t idx = off + static_cast<int64_t>(dp.ReinterpretCast<uint32_t>().GetValue(1));
        wl.SetValue(2 * k, v);
        wl.SetValue(2 * k + 1, static_cast<float>(idx));
        ++k;
        if (k == kBatch || u == u1 - 1) {
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams op{1, static_cast<uint32_t>(k * 8), 0, 0, 0};
            DataCopyPad(wsG[startU * 2], wl, op);
            PipeBarrier<PIPE_ALL>();
            k = 0;
            startU = u + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// row mode merge: (row, chunk) partials -> int64 index
// ---------------------------------------------------------------------------
__global__ __aicore__ void argmax_row_combine_kernel(GM_ADDR wsGm_, GM_ADDR yGm_,
                                                     int64_t outer, int64_t nChunk,
                                                     int64_t upb, int64_t batchRows)
{
#ifdef KERNEL_TASK_TYPE_DEFAULT
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    constexpr int64_t kChunkPairs = 512;

    GlobalTensor<float> wsG;
    wsG.SetGlobalBuffer((__gm__ float*)wsGm_);
    GlobalTensor<int64_t> yG;
    yG.SetGlobalBuffer((__gm__ int64_t*)yGm_);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TBuf<TPosition::VECOUT> outBuf;
    pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(kChunkPairs * 8));
    pipe.InitBuffer(outBuf, 512);

    LocalTensor<int64_t> ob = outBuf.Get<int64_t>();

    int64_t b = static_cast<int64_t>(GetBlockIdx());
    int64_t r0 = b * upb;
    int64_t r1 = r0 + upb;
    if (r1 > outer) {
        r1 = outer;
    }
    if (r0 >= outer) {
        return;
    }

    int64_t k = 0;
    int64_t startRow = r0;
    for (int64_t r = r0; r < r1; ++r) {
        float bestV = ArgMaxNegInf();
        int64_t bestI = 0;
        for (int64_t cs = 0; cs < nChunk; cs += kChunkPairs) {
            int64_t cnt = nChunk - cs;
            if (cnt > kChunkPairs) {
                cnt = kChunkPairs;
            }
            LocalTensor<float> t = inQ.AllocTensor<float>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * 8), 0, 0, 0};
            DataCopyPadExtParams<float> pp{false, 0, 0, 0};
            DataCopyPad(t, wsG[(r * nChunk + cs) * 2], cp, pp);
            inQ.EnQue(t);
            t = inQ.DeQue<float>();
            PipeBarrier<PIPE_ALL>();
            for (int64_t j = 0; j < cnt; ++j) {
                float v = t.GetValue(2 * j);
                if (v > bestV) {
                    bestV = v;
                    bestI = static_cast<int64_t>(t.GetValue(2 * j + 1));
                }
            }
            PipeBarrier<PIPE_ALL>();
            inQ.FreeTensor(t);
        }
        ob.SetValue(k, bestI);
        ++k;
        if (k == batchRows || r == r1 - 1) {
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams op{1, static_cast<uint32_t>(k * 8), 0, 0, 0};
            DataCopyPad(yG[startRow], ob, op);
            PipeBarrier<PIPE_ALL>();
            k = 0;
            startRow = r + 1;
        }
    }
}

/*!
 * \brief Column-tile geometry.
 *
 * Tile t of the inner axis starts at t*wTile.  The LAST tile is right-aligned on
 * `inner` and rounded up to a multiple of 64 columns so that (a) no tile is a
 * tiny tail (a 1-column tail would degenerate into one 2-byte DMA per row) and
 * (b) every tile width is a multiple of 64, which makes the row pitch of the
 * batched DataCopyPad exactly wAct and keeps every vector count a multiple of
 * 64.  Adjacent tiles may therefore overlap by up to 63 columns; the duplicate
 * writes carry identical values, which is benign.
 *
 * Only when inner < 64 can a tile width not be a multiple of 64; that case
 * falls back to a per-row DMA with a full-wTile row pitch.
 */
__aicore__ inline void ArgMaxColTile(int64_t t, int64_t wTile, int64_t inner, int64_t& colStart,
                                     int64_t& wAct)
{
    colStart = t * wTile;
    if (colStart + wTile > inner && inner >= wTile) {
        int64_t need = inner - colStart;
        int64_t w2 = (need + 63) / 64 * 64;
        if (w2 > wTile) {
            w2 = wTile;
        }
        if (w2 < 64) {
            w2 = 64;
        }
        colStart = inner - w2;
        wAct = w2;
        return;
    }
    wAct = inner - colStart;
    if (wAct > wTile) {
        wAct = wTile;
    }
}

// ---------------------------------------------------------------------------
// column mode stage 1: partial (value, index) per (outer, column tile, reduce segment)
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void argmax_col_partial_kernel(GM_ADDR xGm_, GM_ADDR wsGm_, GM_ADDR yGm_,
                                                     int64_t reduceLen, int64_t inner,
                                                     int64_t wTile, int64_t nColTile,
                                                     int64_t nseg, int64_t segD,
                                                     int64_t units, int64_t upb, int64_t rb,
                                                     int64_t directOut)
{
#ifdef KERNEL_TASK_TYPE_DEFAULT
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    GlobalTensor<T> xG;
    xG.SetGlobalBuffer((__gm__ T*)xGm_);
    GlobalTensor<float> wsG;
    wsG.SetGlobalBuffer((__gm__ float*)wsGm_);
    GlobalTensor<int64_t> yG;
    yG.SetGlobalBuffer((__gm__ int64_t*)yGm_);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TBuf<TPosition::VECCALC> fBuf, keyBuf, cmBuf, ciBuf;
    TBuf<TPosition::VECCALC> mBuf, m2Buf, selTmp, i32Buf, i64Buf;
    pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(rb * wTile * static_cast<int64_t>(sizeof(T))));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(fBuf, static_cast<uint32_t>(rb * wTile * 4));
    }
    pipe.InitBuffer(keyBuf, static_cast<uint32_t>(wTile * 4));
    pipe.InitBuffer(cmBuf, static_cast<uint32_t>(wTile * 4));
    pipe.InitBuffer(ciBuf, static_cast<uint32_t>(wTile * 4));
    pipe.InitBuffer(mBuf, static_cast<uint32_t>(wTile / 8 + 32));
    pipe.InitBuffer(m2Buf, static_cast<uint32_t>(wTile / 8 + 32));
    pipe.InitBuffer(selTmp, 8192);
    pipe.InitBuffer(i32Buf, static_cast<uint32_t>(wTile * 4));
    pipe.InitBuffer(i64Buf, static_cast<uint32_t>(wTile * 8));

    LocalTensor<float> fTile;
    if constexpr (!std::is_same<T, float>::value) {
        fTile = fBuf.Get<float>();
    }
    LocalTensor<float> key = keyBuf.Get<float>();
    LocalTensor<float> curMax = cmBuf.Get<float>();
    LocalTensor<float> curIdx = ciBuf.Get<float>();
    LocalTensor<uint8_t> maskNaN = mBuf.Get<uint8_t>();
    LocalTensor<uint8_t> maskLE = m2Buf.Get<uint8_t>();
    LocalTensor<int32_t> o32 = i32Buf.Get<int32_t>();
    LocalTensor<int64_t> o64 = i64Buf.Get<int64_t>();

    const float negInf = ArgMaxNegInf();
    const float posInf = ArgMaxPosInf();

    int64_t b = static_cast<int64_t>(GetBlockIdx());
    int64_t u0 = b * upb;
    int64_t u1 = u0 + upb;
    if (u1 > units) {
        u1 = units;
    }
    if (u0 >= units) {
        return;
    }

    for (int64_t u = u0; u < u1; ++u) {
        int64_t g = u / nseg;
        int64_t s = u - g * nseg;
        int64_t o = g / nColTile;
        int64_t t = g - o * nColTile;
        int64_t colStart = 0;
        int64_t wAct = 0;
        ArgMaxColTile(t, wTile, inner, colStart, wAct);
        int64_t redStart = s * segD;
        int64_t rows = reduceLen - redStart;
        if (rows > segD) {
            rows = segD;
        }
        if (wAct <= 0) {
            continue;
        }
        const bool full = (wAct % 64 == 0);
        const uint32_t lanes = static_cast<uint32_t>(full ? wAct : wTile);
        const int64_t pitch = full ? wAct : wTile;
        int64_t base = o * reduceLen * inner + redStart * inner + colStart;

        Duplicate(curMax, negInf, lanes);
        Duplicate(curIdx, 0.0f, lanes);
        if (rows <= 0) {
            // empty slice of the reduce axis: publish a neutral entry so the merge never
            // reads uninitialised workspace.
            if (directOut == 0) {
                PipeBarrier<PIPE_ALL>();
                DataCopyExtParams zcp{1, static_cast<uint32_t>(wTile * 4), 0, 0, 0};
                DataCopyPad(wsG[u * 2 * wTile], curMax, zcp);
                DataCopyPad(wsG[u * 2 * wTile + wTile], curIdx, zcp);
                PipeBarrier<PIPE_ALL>();
            }
            continue;
        }

        for (int64_t i = 0; i < rows; i += rb) {
            int64_t nb = rows - i;
            if (nb > rb) {
                nb = rb;
            }
            LocalTensor<T> tq = inQ.AllocTensor<T>();
            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            if (full) {
                DataCopyExtParams cp{static_cast<uint16_t>(nb),
                                     static_cast<uint32_t>(wAct * static_cast<int64_t>(sizeof(T))),
                                     static_cast<uint32_t>((inner - wAct) * static_cast<int64_t>(sizeof(T))),
                                     0, 0};
                DataCopyPad(tq, xG[base + i * inner], cp, pp);
            } else {
                for (int64_t j = 0; j < nb; ++j) {
                    DataCopyExtParams cp{1,
                                         static_cast<uint32_t>(wAct * static_cast<int64_t>(sizeof(T))),
                                         0, 0, 0};
                    DataCopyPad(tq[j * wTile], xG[base + (i + j) * inner], cp, pp);
                }
            }
            inQ.EnQue(tq);
            tq = inQ.DeQue<T>();
            LocalTensor<float> fsrc;
            if constexpr (std::is_same<T, float>::value) {
                // fp32 rows are consumed straight out of the queued tile: no UB->UB copy.
                fsrc = tq;
            } else {
                ArgMaxToFloat<T>(fTile, tq, static_cast<uint32_t>(nb * pitch));
                fsrc = fTile;
            }

            for (int64_t j = 0; j < nb; ++j) {
                LocalTensor<float> row = fsrc[j * pitch];
                float pos = static_cast<float>(redStart + i + j);
                Compares(maskNaN, row, negInf, CMPMODE::GE, lanes);
                Select(key, maskNaN, row, posInf, SELMODE::VSEL_TENSOR_SCALAR_MODE, lanes);
                Compare(maskLE, key, curMax, CMPMODE::LE, lanes);
                Select(curIdx, maskLE, curIdx, pos, SELMODE::VSEL_TENSOR_SCALAR_MODE, lanes);
                Max(curMax, curMax, key, lanes);
            }
            inQ.FreeTensor(tq);
        }

        if (directOut != 0) {
            // single reduce slice: the per-column index is final, write it out directly
            // (skips the whole merge kernel and its workspace round trip).
            Cast<int32_t, float>(o32, curIdx, RoundMode::CAST_RINT, lanes);
            Cast<int64_t, int32_t>(o64, o32, RoundMode::CAST_NONE, lanes);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams op{1, static_cast<uint32_t>(wAct * 8), 0, 0, 0};
            DataCopyPad(yG[o * inner + colStart], o64, op);
            PipeBarrier<PIPE_ALL>();
            continue;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams wcp{1, static_cast<uint32_t>(wTile * 4), 0, 0, 0};
        DataCopyPad(wsG[u * 2 * wTile], curMax, wcp);
        DataCopyPad(wsG[u * 2 * wTile + wTile], curIdx, wcp);
        // curMax / curIdx are reused by the next unit: wait for the outgoing DMA to be
        // consumed before the next Duplicate overwrites them (MTE3 -> V hazard).
        PipeBarrier<PIPE_ALL>();
    }
}

// ---------------------------------------------------------------------------
// column mode stage 2: merge the reduce-axis partials and write int64 indices
// ---------------------------------------------------------------------------
__global__ __aicore__ void argmax_col_combine_kernel(GM_ADDR wsGm_, GM_ADDR yGm_,
                                                     int64_t reduceLen, int64_t inner, int64_t wTile,
                                                     int64_t nColTile, int64_t nseg, int64_t segD,
                                                     int64_t units, int64_t upb)
{
#ifdef KERNEL_TASK_TYPE_DEFAULT
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    GlobalTensor<float> wsG;
    wsG.SetGlobalBuffer((__gm__ float*)wsGm_);
    GlobalTensor<int64_t> yG;
    yG.SetGlobalBuffer((__gm__ int64_t*)yGm_);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TBuf<TPosition::VECCALC> cmBuf, ciBuf, mBuf, i32Buf, i64Buf, selTmp;
    pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(2 * wTile * 4));
    pipe.InitBuffer(cmBuf, static_cast<uint32_t>(wTile * 4));
    pipe.InitBuffer(ciBuf, static_cast<uint32_t>(wTile * 4));
    pipe.InitBuffer(mBuf, static_cast<uint32_t>(wTile / 8 + 32));
    pipe.InitBuffer(i32Buf, static_cast<uint32_t>(wTile * 4));
    pipe.InitBuffer(i64Buf, static_cast<uint32_t>(wTile * 8));
    pipe.InitBuffer(selTmp, 8192);

    LocalTensor<float> curMax = cmBuf.Get<float>();
    LocalTensor<float> curIdx = ciBuf.Get<float>();
    LocalTensor<uint8_t> maskLE = mBuf.Get<uint8_t>();
    LocalTensor<int32_t> i32 = i32Buf.Get<int32_t>();
    LocalTensor<int64_t> i64 = i64Buf.Get<int64_t>();

    const float negInf = ArgMaxNegInf();

    int64_t b = static_cast<int64_t>(GetBlockIdx());
    int64_t u0 = b * upb;
    int64_t u1 = u0 + upb;
    if (u1 > units) {
        u1 = units;
    }
    if (u0 >= units) {
        return;
    }

    for (int64_t u = u0; u < u1; ++u) {
        int64_t o = u / nColTile;
        int64_t t = u - o * nColTile;
        int64_t colStart = 0;
        int64_t wAct = 0;
        ArgMaxColTile(t, wTile, inner, colStart, wAct);
        if (wAct <= 0) {
            continue;
        }
        const uint32_t lanes =
            static_cast<uint32_t>((wAct % 64 == 0) ? wAct : wTile);
        int64_t sEnd = (reduceLen + segD - 1) / segD;
        if (sEnd > nseg) {
            sEnd = nseg;
        }
        Duplicate(curMax, negInf, lanes);
        Duplicate(curIdx, 0.0f, lanes);
        for (int64_t s = 0; s < sEnd; ++s) {
            LocalTensor<float> tq = inQ.AllocTensor<float>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(2 * wTile * 4), 0, 0, 0};
            DataCopyPadExtParams<float> pp{false, 0, 0, 0};
            DataCopyPad(tq, wsG[(u * nseg + s) * 2 * wTile], cp, pp);
            inQ.EnQue(tq);
            tq = inQ.DeQue<float>();
            LocalTensor<float> vals = tq;
            LocalTensor<float> idxs = tq[wTile];
            Compare(maskLE, vals, curMax, CMPMODE::LE, lanes);
            Select(curIdx, maskLE, curIdx, idxs, SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
            Max(curMax, curMax, vals, lanes);
            inQ.FreeTensor(tq);
        }
        Cast<int32_t, float>(i32, curIdx, RoundMode::CAST_RINT, lanes);
        Cast<int64_t, int32_t>(i64, i32, RoundMode::CAST_NONE, lanes);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams op{1, static_cast<uint32_t>(wAct * 8), 0, 0, 0};
        DataCopyPad(yG[o * inner + colStart], i64, op);
        // i64 is rewritten by the next unit: wait for the outgoing DMA (MTE3 -> V hazard).
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace

// ---------------------------------------------------------------------------
// host tiling
// ---------------------------------------------------------------------------
static int64_t ArgMaxTypeSize(int64_t dtypeCode)
{
    switch (dtypeCode) {
        case AM_HALF:
        case AM_BF16:
            return 2;
        case AM_INT64:
            return 8;
        default:
            return 4;
    }
}

ArgMaxPlan calc_argmax_plan(int64_t outer, int64_t reduceLen, int64_t inner, int64_t dtypeCode)
{
    ArgMaxPlan p;
    p.mode = 0;
    p.outer = outer;
    p.reduceLen = reduceLen;
    p.inner = inner;
    p.segRow = kSegRow;
    p.nChunk = 1;
    p.unitsRow = 1;
    p.blocksRow = 1;
    p.upbRow = 1;
    p.unitsRow2 = 0;
    p.blocksRow2 = 0;
    p.upbRow2 = 0;
    p.batchRows = 8;
    p.wTile = 64;
    p.nColTile = 1;
    p.nseg = 1;
    p.segD = 1;
    p.unitsCol = 0;
    p.blocksCol = 0;
    p.upbCol = 0;
    p.unitsCol2 = 0;
    p.blocksCol2 = 0;
    p.upbCol2 = 0;
    p.rb = 8;
    p.wsFloats = 0;
    p.pitchRow = 1;
    p.rbRow = 1;
    p.directCol = 0;

    int64_t coreNum = 1;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform != nullptr) {
        coreNum = platform->GetCoreNumAiv();
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    if (outer < 1) {
        outer = 1;
    }
    if (reduceLen < 1) {
        reduceLen = 1;
    }

    if (inner <= 1) {
        p.mode = 0;
        p.nChunk = (reduceLen + kSegRow - 1) / kSegRow;
        if (p.nChunk < 1) {
            p.nChunk = 1;
        }
        if (p.nChunk == 1) {
            int64_t tsize = ArgMaxTypeSize(dtypeCode);
            int64_t pitch = ((reduceLen * tsize + 31) / 32 * 32) / tsize;
            if (pitch < 1) {
                pitch = 1;
            }
            p.pitchRow = pitch;
            int64_t rb = 32768 / (pitch * tsize);
            if (dtypeCode != AM_FLOAT) {
                int64_t rb2 = 49152 / (pitch * 4);
                if (rb > rb2) {
                    rb = rb2;
                }
            }
            if (rb > 32) {
                rb = 32;
            }
            if (rb < 1) {
                rb = 1;
            }
            p.rbRow = rb;
            int64_t batches = (outer + rb - 1) / rb;
            p.unitsRow = batches;
            p.blocksRow = batches < coreNum ? batches : coreNum;
            p.upbRow = (batches + p.blocksRow - 1) / p.blocksRow;
        } else {
            p.unitsRow = outer * p.nChunk;
            p.blocksRow = p.unitsRow < coreNum ? p.unitsRow : coreNum;
            p.upbRow = (p.unitsRow + p.blocksRow - 1) / p.blocksRow;
            p.unitsRow2 = outer;
            p.blocksRow2 = outer < coreNum ? outer : coreNum;
            p.upbRow2 = (outer + p.blocksRow2 - 1) / p.blocksRow2;
            p.batchRows = 16;
            p.wsFloats = outer * p.nChunk * 2;
        }
    } else {
        p.mode = 1;
        int64_t wt = inner < kBorder ? inner : kBorder;
        int64_t wTile = (wt / 64) * 64;
        if (wTile < 64) {
            wTile = 64;
        }
        p.wTile = wTile;
        p.nColTile = (inner + wTile - 1) / wTile;
        int64_t base = outer * p.nColTile;
        if (base < 1) {
            base = 1;
        }
        int64_t nseg = (2 * coreNum + base - 1) / base;
        if (nseg < 1) {
            nseg = 1;
        }
        if (nseg > reduceLen) {
            nseg = reduceLen;
        }
        p.nseg = nseg;
        p.segD = (reduceLen + nseg - 1) / nseg;
        p.unitsCol = base * nseg;
        p.blocksCol = p.unitsCol < coreNum ? p.unitsCol : coreNum;
        p.upbCol = (p.unitsCol + p.blocksCol - 1) / p.blocksCol;
        p.unitsCol2 = base;
        p.blocksCol2 = base < coreNum ? base : coreNum;
        p.upbCol2 = (base + p.blocksCol2 - 1) / p.blocksCol2;
        int64_t tsize = ArgMaxTypeSize(dtypeCode);
        int64_t rb = 32768 / (wTile * tsize);
        if (rb > 8) {
            rb = 8;
        }
        if (rb < 1) {
            rb = 1;
        }
        p.rb = rb;
        p.directCol = (nseg == 1) ? 1 : 0;
        p.wsFloats = p.directCol != 0 ? 0 : p.unitsCol * 2 * wTile;
    }
    return p;
}

// ---------------------------------------------------------------------------
// launch wrappers (host)
// ---------------------------------------------------------------------------
extern "C" {

#define ARGMAX_ROW_DIRECT(NAME, TY)                                                                  \
    void NAME(GM_ADDR x, GM_ADDR y, const ArgMaxPlan* p, void* stream)                               \
    {                                                                                                \
        argmax_row_direct_kernel<TY><<<static_cast<uint32_t>(p->blocksRow), nullptr, stream>>>(       \
            x, y, p->outer, p->reduceLen, p->pitchRow, p->rbRow, p->unitsRow, p->upbRow);             \
    }

#define ARGMAX_ROW_PARTIAL(NAME, TY)                                                                 \
    void NAME(GM_ADDR x, GM_ADDR ws, const ArgMaxPlan* p, void* stream)                              \
    {                                                                                                \
        argmax_row_partial_kernel<TY><<<static_cast<uint32_t>(p->blocksRow), nullptr, stream>>>(      \
            x, ws, p->reduceLen, p->nChunk, p->segRow, p->unitsRow, p->upbRow);                       \
    }

#define ARGMAX_COL_PARTIAL(NAME, TY)                                                                 \
    void NAME(GM_ADDR x, GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream)                    \
    {                                                                                                \
        argmax_col_partial_kernel<TY><<<static_cast<uint32_t>(p->blocksCol), nullptr, stream>>>(      \
            x, ws, y, p->reduceLen, p->inner, p->wTile, p->nColTile, p->nseg, p->segD, p->unitsCol,   \
            p->upbCol, p->rb, p->directCol);                                                          \
    }

ARGMAX_ROW_DIRECT(launch_argmax_row_direct_f16, half)
ARGMAX_ROW_DIRECT(launch_argmax_row_direct_f32, float)
ARGMAX_ROW_DIRECT(launch_argmax_row_direct_bf16, bfloat16_t)
ARGMAX_ROW_DIRECT(launch_argmax_row_direct_i32, int32_t)
ARGMAX_ROW_DIRECT(launch_argmax_row_direct_i64, int64_t)

ARGMAX_ROW_PARTIAL(launch_argmax_row_partial_f16, half)
ARGMAX_ROW_PARTIAL(launch_argmax_row_partial_f32, float)
ARGMAX_ROW_PARTIAL(launch_argmax_row_partial_bf16, bfloat16_t)
ARGMAX_ROW_PARTIAL(launch_argmax_row_partial_i32, int32_t)
ARGMAX_ROW_PARTIAL(launch_argmax_row_partial_i64, int64_t)

ARGMAX_COL_PARTIAL(launch_argmax_col_partial_f16, half)
ARGMAX_COL_PARTIAL(launch_argmax_col_partial_f32, float)
ARGMAX_COL_PARTIAL(launch_argmax_col_partial_bf16, bfloat16_t)
ARGMAX_COL_PARTIAL(launch_argmax_col_partial_i32, int32_t)
ARGMAX_COL_PARTIAL(launch_argmax_col_partial_i64, int64_t)

void launch_argmax_row_combine(GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream)
{
    argmax_row_combine_kernel<<<static_cast<uint32_t>(p->blocksRow2), nullptr, stream>>>(
        ws, y, p->outer, p->nChunk, p->upbRow2, p->batchRows);
}

void launch_argmax_col_combine(GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream)
{
    argmax_col_combine_kernel<<<static_cast<uint32_t>(p->blocksCol2), nullptr, stream>>>(
        ws, y, p->reduceLen, p->inner, p->wTile, p->nColTile, p->nseg, p->segD, p->unitsCol2,
        p->upbCol2);
}

} // extern "C"
