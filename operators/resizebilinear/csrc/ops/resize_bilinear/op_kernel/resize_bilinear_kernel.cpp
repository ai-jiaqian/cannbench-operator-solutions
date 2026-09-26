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
 * \file resize_bilinear_kernel.cpp
 * \brief ResizeBilinear device kernel + host tiling/launch (bisheng + -xasc, dav-2201 / Ascend 910B).
 *
 * Semantics identical to torch.nn.functional.interpolate(mode='bilinear'):
 *
 *   align_corners == true :  src = scale * dst
 *   align_corners == false:  src = scale * (dst + 0.5f) - 0.5f
 *   src = max(src, 0)                      (torch clamps the lower bound only)
 *   i0 = (int)src ; i1 = min(i0 + 1, in - 1) ; lambda = src - i0
 *
 *   innerLow  = (1 - b) * x[h0, w0] + b * x[h0, w1]
 *   innerHigh = (1 - b) * x[h1, w0] + b * x[h1, w1]
 *   y         = (1 - a) * innerLow + a * innerHigh
 *
 * The summation grouping is exactly the torch CPU reference's, and all arithmetic
 * runs in fp32 with a single final rounding to the input dtype, matching torch's
 * accscalar_t (float) accumulation for fp16 / bf16 inputs.
 *
 * Structure (per AIV core):
 *   - The N*C*H_out output rows are split into contiguous ranges, one per core.
 *   - W-direction tables depend only on W_out and are built once per core:
 *       * a 4 x Wq zero-padded uint32 byte-offset table indexing a FIXED two-slot
 *         row buffer, laid out [x_h0[w0] | x_h1[w0] | x_h0[w1] | x_h1[w1]]
 *         (slot pitch `slotElems`; every segment padded to Wq = align_up(W_out, 8)
 *         so all intra-table slices are 32B aligned);
 *       * the matching weight table W2 = [1-b | 1-b | b | b].
 *   - Per output row the two source rows are copied into the two slots, cast once,
 *     and a single Gather + one Mul produce the four weighted neighbours; a single
 *     Add over 2*W_out yields [innerLow | innerHigh].
 *   - Upsampling makes many consecutive output rows share the same (h0, h1) pair;
 *     the pair and its [innerLow | innerHigh] are cached, so those rows only pay
 *     Muls + Axpy.  Reusing an already computed inner term is value-identical.
 *   - Output rows are accumulated into a small fp32 tile and flushed in batches, so
 *     the per-row cost carries no queue synchronisation.
 */

#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

namespace {

constexpr static int64_t RB_PIPELINE_DEPTH = 2;
/*! Extra zero-filled slots appended to every W_out sized buffer.  Gather may
 *  operate on whole vector registers; zero offsets can only ever select element 0
 *  and the destination padding keeps any over-wide store inside UB. */
constexpr static int64_t RB_PAD = 64;
constexpr static int64_t RB_GATHER_CHUNK = 4096;
constexpr static int64_t RB_TILE_ROWS_SMALL = 8;   /* Wq <= 512   */
constexpr static int64_t RB_TILE_ROWS_LARGE = 4;   /* Wq >  512   */

}  // namespace

__aicore__ inline int64_t RbAlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

/*! \brief Gather `count` fp32 elements from `src` using byte offsets `offs`. */
__aicore__ inline void RbGatherChunked(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::LocalTensor<float>& src,
                                       const AscendC::LocalTensor<uint32_t>& offs, int64_t count)
{
    for (int64_t done = 0; done < count; done += RB_GATHER_CHUNK) {
        int64_t c = count - done;
        if (c > RB_GATHER_CHUNK) {
            c = RB_GATHER_CHUNK;
        }
        AscendC::Gather(dst[done], src, offs[done], static_cast<uint32_t>(0),
                        static_cast<uint32_t>(c));
    }
}

/*!
 * \brief ResizeBilinear kernel.
 *
 * \param totalRows    N * C * H_out
 * \param rowsPerCore  number of contiguous output rows owned by one core
 * \param alignCorners 0 / 1
 */
template <typename T>
__global__ __aicore__ void resize_bilinear_kernel(GM_ADDR x, GM_ADDR y, int64_t totalRows,
                                                  int64_t rowsPerCore, int64_t hIn, int64_t wIn,
                                                  int64_t hOut, int64_t wOut, float scaleH,
                                                  float scaleW, int32_t alignCorners)
{
    using namespace AscendC;

    const int64_t blockIdx = GetBlockIdx();
    const int64_t rowBegin = blockIdx * rowsPerCore;
    if (rowBegin >= totalRows) {
        return;
    }
    int64_t rowEnd = rowBegin + rowsPerCore;
    if (rowEnd > totalRows) {
        rowEnd = totalRows;
    }
    const int64_t nRows = rowEnd - rowBegin;
    if (nRows <= 0) {
        return;
    }

    const int32_t wInI = static_cast<int32_t>(wIn);
    const int32_t wOutI = static_cast<int32_t>(wOut);
    const int32_t hInMax = static_cast<int32_t>(hIn) - 1;
    const float wInM1 = static_cast<float>(wIn - 1);
    /* segment stride: padded so that every intra-table slice is 32B aligned */
    const int64_t wq = RbAlignUp(wOut, 8);
    const int32_t wqI = static_cast<int32_t>(wq);
    /* UB pitch of one row slot (elements; identical index space for T and fp32) */
    const int64_t slotElems = RbAlignUp(wIn * static_cast<int64_t>(sizeof(T)), 32) / sizeof(T);
    const int64_t tableElems = 4 * wq;
    const int64_t tileRows = (wq <= 512) ? RB_TILE_ROWS_SMALL : RB_TILE_ROWS_LARGE;

    TPipe pipe;
    TQue<QuePosition::VECIN, RB_PIPELINE_DEPTH> qIn;
    TQue<QuePosition::VECOUT, RB_PIPELINE_DEPTH> qOut;
    TBuf<TPosition::VECCALC> bF32;
    TBuf<TPosition::VECCALC> bOff;
    TBuf<TPosition::VECCALC> bW2;
    TBuf<TPosition::VECCALC> bD4;
    TBuf<TPosition::VECCALC> bInner;
    TBuf<TPosition::VECCALC> bScr;
    TBuf<TPosition::VECCALC> bTile;

    pipe.InitBuffer(qIn, RB_PIPELINE_DEPTH,
                    static_cast<uint32_t>(2 * slotElems * sizeof(T) + 32));
    pipe.InitBuffer(qOut, RB_PIPELINE_DEPTH,
                    static_cast<uint32_t>(tileRows * wq * static_cast<int64_t>(sizeof(T)) + 32));
    if constexpr (!std::is_same<T, float>::value) {
        pipe.InitBuffer(bF32, static_cast<uint32_t>(2 * slotElems * sizeof(float) + 32));
        pipe.InitBuffer(bTile, static_cast<uint32_t>(tileRows * wq * 4 + 32));
    }
    const uint32_t tblBytes = static_cast<uint32_t>((tableElems + RB_PAD) * 4);
    pipe.InitBuffer(bOff, tblBytes);
    pipe.InitBuffer(bW2, tblBytes);
    pipe.InitBuffer(bD4, tblBytes);
    pipe.InitBuffer(bInner, static_cast<uint32_t>((2 * wq + RB_PAD) * 4));
    pipe.InitBuffer(bScr, static_cast<uint32_t>((wq + RB_PAD) * 4));

    /* ------------------------------------------------------------------
     * W-direction tables, built once per core.
     *   segment 0 : x[h0, w0]   (slot 0)
     *   segment 1 : x[h1, w0]   (slot 1)
     *   segment 2 : x[h0, w1]   (slot 0)
     *   segment 3 : x[h1, w1]   (slot 1)
     * so one Add over 2*W_out collapses segments (0,2) into innerLow and
     * (1,3) into innerHigh.
     * ------------------------------------------------------------------ */
    {
        LocalTensor<float> wF = bD4.Get<float>();  // scratch (rewritten per row pair)
        LocalTensor<float> w0F = bScr.Get<float>();  // scratch
        LocalTensor<int32_t> scrI = bInner.Get<int32_t>();  // scratch
        LocalTensor<int32_t> offI = bOff.Get<int32_t>();
        LocalTensor<float> w2 = bW2.Get<float>();

        CreateVecIndex(wF, 0.0f, static_cast<uint32_t>(wOutI));
        if (alignCorners != 0) {
            Muls(wF, wF, scaleW, wOutI);
        } else {
            Adds(wF, wF, 0.5f, wOutI);
            Muls(wF, wF, scaleW, wOutI);
            Adds(wF, wF, -0.5f, wOutI);
            Maxs(wF, wF, 0.0f, wOutI);
        }
        Mins(wF, wF, wInM1, wOutI);

        Duplicate<int32_t>(offI, 0, static_cast<int32_t>(tableElems + RB_PAD));
        Duplicate<float>(w2, 0.0f, static_cast<int32_t>(tableElems + RB_PAD));

        const int32_t f32Bytes = static_cast<int32_t>(sizeof(float));
        const int32_t slotBytes = static_cast<int32_t>(slotElems * sizeof(float));

        Cast(scrI, wF, RoundMode::CAST_FLOOR, static_cast<uint32_t>(wOutI));  // w0
        Cast(w0F, scrI, RoundMode::CAST_NONE, static_cast<uint32_t>(wOutI));
        Sub(w2[2 * wq], wF, w0F, wOutI);                       // beta in segment 2
        Muls(offI, scrI, f32Bytes, wOutI);                     // segment 0
        Adds(offI[wq], offI, slotBytes, wOutI);                // segment 1
        Muls(w2, w2[2 * wq], -1.0f, wOutI);
        Adds(w2, w2, 1.0f, wOutI);                             // 1 - beta in segment 0
        Adds(w2[wq], w2, 0.0f, wOutI);                         // 1 - beta in segment 1

        Adds(w0F, w0F, 1.0f, wOutI);
        Mins(w0F, w0F, wInM1, wOutI);
        Cast(scrI, w0F, RoundMode::CAST_FLOOR, static_cast<uint32_t>(wOutI));  // w1
        Muls(offI[2 * wq], scrI, f32Bytes, wOutI);             // segment 2
        Adds(offI[3 * wq], offI[2 * wq], slotBytes, wOutI);    // segment 3
        Adds(w2[3 * wq], w2[2 * wq], 0.0f, wOutI);             // beta in segment 3
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<uint32_t> offU = bOff.Get<uint32_t>();
    LocalTensor<float> w2 = bW2.Get<float>();
    LocalTensor<float> d4 = bD4.Get<float>();
    LocalTensor<float> inner = bInner.Get<float>();

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    const DataCopyExtParams cpRow{1, static_cast<uint32_t>(wIn * static_cast<int64_t>(sizeof(T))),
                                  0, 0, 0};
    const DataCopyExtParams cpRow2{2, static_cast<uint32_t>(wIn * static_cast<int64_t>(sizeof(T))),
                                   0, 0, 0};
    const DataCopyExtParams cpOut{1, static_cast<uint32_t>(wOut * static_cast<int64_t>(sizeof(T))),
                                  0, 0, 0};
    const DataCopyPadExtParams<T> padT{false, 0, 0, 0};
    /* The two source rows are adjacent in GM whenever h1 == h0 + 1, and they are
     * adjacent in UB whenever the slot pitch has no padding.  Only then can both
     * be fetched with one two-block DataCopyPad (multi-block copies constrain the
     * source block addresses to 32B, which is exactly slotElems == wIn). */
    const bool mergeRows = (slotElems == wIn);
    /* A padded tile makes the rows non-contiguous in UB, so the chunk can only be
     * stored with one call when there is no padding at all. */
    const bool mergeStore = (wq == wOut);

    int64_t i = 0;
    int64_t nc = rowBegin / hOut;
    int64_t oh = rowBegin - nc * hOut;
    int64_t cachedOff0 = -1;
    int64_t cachedOff1 = -1;

    while (i < nRows) {
        int64_t chunk = nRows - i;
        if (chunk > tileRows) {
            chunk = tileRows;
        }
        const int64_t firstRow = i;
        LocalTensor<float> tile;
        if constexpr (std::is_same<T, float>::value) {
            tile = qOut.AllocTensor<float>();
        } else {
            tile = bTile.Get<float>();
        }

        for (int64_t t = 0; t < chunk; ++t, ++i) {
            float h;
            if (alignCorners != 0) {
                h = scaleH * static_cast<float>(oh);
            } else {
                h = scaleH * (static_cast<float>(oh) + 0.5f) - 0.5f;
            }
            if (h < 0.0f) {
                h = 0.0f;
            }
            int32_t h0 = static_cast<int32_t>(h);
            if (h0 > hInMax) {
                h0 = hInMax;
            }
            int32_t h1 = h0 + 1;
            if (h1 > hInMax) {
                h1 = hInMax;
            }
            const float a = h - static_cast<float>(h0);
            const float oma = 1.0f - a;

            const int64_t planeBase = nc * hIn * wIn;
            const int64_t gOff0 = planeBase + static_cast<int64_t>(h0) * wIn;
            const int64_t gOff1 = planeBase + static_cast<int64_t>(h1) * wIn;

            if (gOff0 != cachedOff0 || gOff1 != cachedOff1) {
                auto raw = qIn.AllocTensor<T>();
                if (mergeRows && (h1 == h0 + 1)) {
                    DataCopyPad(raw, xGm[gOff0], cpRow2, padT);
                } else {
                    DataCopyPad(raw, xGm[gOff0], cpRow, padT);
                    DataCopyPad(raw[slotElems], xGm[gOff1], cpRow, padT);
                }
                qIn.EnQue(raw);
                raw = qIn.DeQue<T>();

                if constexpr (std::is_same<T, float>::value) {
                    RbGatherChunked(d4, raw, offU, tableElems);
                    qIn.FreeTensor(raw);
                } else {
                    LocalTensor<float> f32 = bF32.Get<float>();
                    Cast(f32, raw, RoundMode::CAST_NONE,
                         static_cast<uint32_t>(slotElems + wIn));
                    qIn.FreeTensor(raw);
                    RbGatherChunked(d4, f32, offU, tableElems);
                }
                Mul(d4, d4, w2, static_cast<int32_t>(tableElems));
                /* Covers [0, W_out) for innerLow and [wq, wq + W_out) for
                 * innerHigh; note wq >= W_out so the count is wq + W_out. */
                Add(inner, d4, d4[2 * wq], wqI + wOutI);
                cachedOff0 = gOff0;
                cachedOff1 = gOff1;
            }

            LocalTensor<float> yRow = tile[t * wq];
            Muls(yRow, inner, oma, wOutI);
            Axpy(yRow, inner[wq], a, wOutI);

            ++oh;
            if (oh == hOut) {
                oh = 0;
                ++nc;
            }
        }

        /* ---- flush the chunk ---- */
        const int64_t gmBase = (rowBegin + firstRow) * wOut;
        const bool oneStore = mergeStore && (chunk > 1);
        const DataCopyExtParams cpChunk{static_cast<uint16_t>(chunk),
                                        static_cast<uint32_t>(wOut * static_cast<int64_t>(sizeof(T))),
                                        0, 0, 0};
        if constexpr (std::is_same<T, float>::value) {
            qOut.EnQue(tile);
            auto d = qOut.DeQue<float>();
            if (oneStore) {
                DataCopyPad(yGm[gmBase], d, cpChunk);
            } else {
                for (int64_t t = 0; t < chunk; ++t) {
                    DataCopyPad(yGm[gmBase + t * wOut], d[t * wq], cpOut);
                }
            }
            qOut.FreeTensor(d);
        } else {
            auto outT = qOut.AllocTensor<T>();
            Cast(outT, tile, RoundMode::CAST_RINT, static_cast<uint32_t>(chunk * wq));
            qOut.EnQue(outT);
            auto d = qOut.DeQue<T>();
            if (oneStore) {
                DataCopyPad(yGm[gmBase], d, cpChunk);
            } else {
                for (int64_t t = 0; t < chunk; ++t) {
                    DataCopyPad(yGm[gmBase + t * wOut], d[t * wq], cpOut);
                }
            }
            qOut.FreeTensor(d);
        }
    }
    (void)wInI;
}

/* ------------------------------------------------------------------
 * Host side
 * ------------------------------------------------------------------ */

std::tuple<int64_t, int64_t> calc_resize_bilinear_tiling(int64_t totalRows)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    if (totalRows <= 0) {
        return std::make_tuple(static_cast<int64_t>(1), static_cast<int64_t>(1));
    }
    int64_t numBlocks = std::min(coreNum, totalRows);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t rowsPerCore = (totalRows + numBlocks - 1) / numBlocks;
    if (rowsPerCore < 1) {
        rowsPerCore = 1;
    }
    numBlocks = (totalRows + rowsPerCore - 1) / rowsPerCore;
    return std::make_tuple(numBlocks, rowsPerCore);
}

extern "C" {

void launch_resize_bilinear_float(GM_ADDR x, GM_ADDR y, int64_t totalRows, int64_t numBlocks,
                                  int64_t rowsPerCore, int64_t hIn, int64_t wIn, int64_t hOut,
                                  int64_t wOut, float scaleH, float scaleW, int32_t alignCorners,
                                  void* stream)
{
    resize_bilinear_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, totalRows, rowsPerCore, hIn,
                                                                  wIn, hOut, wOut, scaleH, scaleW,
                                                                  alignCorners);
}

void launch_resize_bilinear_half(GM_ADDR x, GM_ADDR y, int64_t totalRows, int64_t numBlocks,
                                 int64_t rowsPerCore, int64_t hIn, int64_t wIn, int64_t hOut,
                                 int64_t wOut, float scaleH, float scaleW, int32_t alignCorners,
                                 void* stream)
{
    resize_bilinear_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, totalRows, rowsPerCore, hIn,
                                                                 wIn, hOut, wOut, scaleH, scaleW,
                                                                 alignCorners);
}

void launch_resize_bilinear_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalRows, int64_t numBlocks,
                                     int64_t rowsPerCore, int64_t hIn, int64_t wIn, int64_t hOut,
                                     int64_t wOut, float scaleH, float scaleW, int32_t alignCorners,
                                     void* stream)
{
    resize_bilinear_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(
        x, y, totalRows, rowsPerCore, hIn, wIn, hOut, wOut, scaleH, scaleW, alignCorners);
}

}  // extern "C"
