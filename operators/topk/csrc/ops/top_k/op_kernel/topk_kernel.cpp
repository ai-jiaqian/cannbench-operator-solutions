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
 * \file topk_kernel.cpp
 * \brief TopK device kernel + host tiling (compiled with bisheng, -xasc).
 *
 * Semantics: y, idx = torch.topk(x, k, dim, largest)
 *
 * The ND tensor is viewed as [outer, reduce, inner] with the reduction axis in the middle:
 *
 *   mode 0 (inner == 1): the reduction axis is the innermost one, a unit is one contiguous run of
 *                        `reduce` elements.  Cs == 1 and the "column" is already contiguous.
 *
 *   mode 1 (inner > 1) : the reduction axis is a middle axis.  A unit owns a group of `Cs` adjacent
 *                        inner indices of one outer slice.  The tile is read as `R` rows of `Cs`
 *                        contiguous elements (blockCount = R, blockLen = Cs * sizeof(T)) so that the
 *                        UB layout is a plain row major [R][Cs] matrix; the per-column data is then
 *                        produced with a vector Gather into a column major [Cs][R] buffer, which is
 *                        exactly what Sort32/MrgSort need in order to process all Cs columns in one
 *                        go (each column occupies R = multiple of 32 consecutive elements).
 *
 * Per reduce tile:
 *   1. load the tile, widen it to fp32 (all supported dtypes are exact images in fp32 over the
 *      tested value ranges; int64 beyond 2^24 is a declared limitation),
 *   2. Sort32 sorts every 32-element group of every column in descending order,
 *   3. a 4-way MrgSort tree merges the groups of each column (the final nq == 2 level is done per
 *      column so that only the documented validBit = 15 / repeatTimes combination is used),
 *   4. the tile top (min(R, k) entries) is merged into the running per-column k-sized sorted
 *      accumulator with a MrgSort of two queues.
 *
 * `largest == 0` is implemented by negating the values on load and negating them back on output.
 * Padding slots of a partial tile are filled with -inf BEFORE the values are negated, so they can
 * never displace a real element regardless of the sign convention.
 *
 * Output: a chunk of rows of the sorted accumulators is gathered into [t][c] order, cast to the
 * output dtype / int64 and written with a single DataCopyPad (blockCount = rows, blockLen = Cs*sz,
 * dstStride = (inner - Cs) * sz).
 */

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "topk_launch.h"

using namespace AscendC;

namespace {

constexpr int64_t TK_DEPTH = 1;

__aicore__ inline float TkNegInf()
{
    uint32_t b = 0xFF800000u;
    float f = 0.0f;
    __builtin_memcpy(&f, &b, sizeof(f));
    return f;
}

__aicore__ inline int64_t TkMinI(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline int64_t TkAlI(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

/*!
 * \brief Return the 256 byte aligned view of a buffer (Sort32 requires a 256B aligned destination,
 *        MrgSort/Gather 32B aligned, Sort32/Gather sources 128B aligned).
 */
template <typename U>
__aicore__ inline LocalTensor<U> TkAl(const LocalTensor<U> &t)
{
    uint64_t a = (uint64_t)t.GetPhyAddr();
    uint32_t off = (uint32_t)(((256u - (uint32_t)(a & 255u)) & 255u) / (uint32_t)sizeof(U));
    if (off == 0u) {
        return t;
    }
    return t[off];
}

/*!
 * \brief tbl[p] = p * mult + floor(p / div) * add, computed with an exact float division
 *        (div is a power of two and n < 2^24) followed by a FLOOR float -> int32 cast.  Floor (not
 *        round-to-nearest) is mandatory: the quotient is only integral for multiples of div.
 */
__aicore__ inline void TkBuildTable(const LocalTensor<int32_t> &tbl, const LocalTensor<int32_t> &ti,
                                    const LocalTensor<float> &tf, int64_t n, int64_t divi,
                                    int64_t mult, int64_t add)
{
    if (n <= 0) {
        return;
    }
    CreateVecIndex(tbl, (int32_t)0, (uint32_t)n);
    Cast(tf, tbl, RoundMode::CAST_NONE, (uint32_t)n);
    Muls(tf, tf, 1.0f / (float)divi, (int32_t)n);
    Cast(ti, tf, RoundMode::CAST_FLOOR, (uint32_t)n);
    Muls(ti, ti, (int32_t)add, (int32_t)n);
    Muls(tbl, tbl, (int32_t)mult, (int32_t)n);
    Add(tbl, tbl, ti, (int32_t)n);
}

/*!
 * \brief Cast `rows` rows of `elems` elements from a packed source (row pitch srcRepBlk data blocks)
 *        to a packed destination (row pitch dstRepBlk data blocks).  repeatTime is capped at 255.
 */
template <typename TD, typename TS>
__aicore__ inline void TkCastRows(const LocalTensor<TD> &dst, const LocalTensor<TS> &src, int64_t rows,
                                  int64_t elems, uint16_t srcRepBlk, uint16_t dstRepBlk)
{
    int64_t done = 0;
    int64_t so = 0;
    int64_t dof = 0;
    const int64_t sStep = (int64_t)srcRepBlk * 32 / (int64_t)sizeof(TS);
    const int64_t dStep = (int64_t)dstRepBlk * 32 / (int64_t)sizeof(TD);
    while (done < rows) {
        int64_t r = TkMinI(rows - done, 255);
        if constexpr (std::is_same<TD, TS>::value && std::is_same<TD, float>::value) {
            // Same-type Cast is not a copy (see TkTileToFloat): copy row by row instead.
            for (int64_t i = 0; i < r; ++i) {
                Muls(dst[(uint32_t)(dof + i * dStep)], src[(uint32_t)(so + i * sStep)], 1.0f,
                     (int32_t)elems);
            }
        } else {
            UnaryRepeatParams prm{1, 1, (uint8_t)dstRepBlk, (uint8_t)srcRepBlk};
            Cast(dst[(uint32_t)dof], src[(uint32_t)so], RoundMode::CAST_NONE, (uint64_t)elems, (uint8_t)r, prm);
        }
        done += r;
        so += r * sStep;
        dof += r * dStep;
    }
}

/*!
 * \brief Merge the R/32 sorted 32-struct queues of every column (R = power-of-two multiple of 32)
 *        into one sorted R-struct list per column.  cbuf and obuf are the two ping-pong buffers, each
 *        holding 2*R*Cs floats; the result is returned.
 */
__aicore__ inline LocalTensor<float> TkMergeTree(LocalTensor<float> cbuf, LocalTensor<float> obuf,
                                                 int64_t R, int64_t Cs)
{
    int64_t nq = R / 32;
    int64_t ql = 32;
    while (nq > 1) {
        if ((nq % 4) == 0) {
            int64_t gpc = nq / 4;
            int64_t cstep = Cs;
            if (gpc * Cs > 255) {
                cstep = 255 / gpc;
                if (cstep < 1) {
                    cstep = 1;
                }
            }
            for (int64_t c0 = 0; c0 < Cs; c0 += cstep) {
                int64_t cb = TkMinI(cstep, Cs - c0);
                MrgSort4Info pi;
                pi.elementLengths[0] = (uint16_t)ql;
                pi.elementLengths[1] = (uint16_t)ql;
                pi.elementLengths[2] = (uint16_t)ql;
                pi.elementLengths[3] = (uint16_t)ql;
                pi.ifExhaustedSuspension = false;
                pi.validBit = 0xF;
                pi.repeatTimes = (uint8_t)(gpc * cb);
                uint32_t base = (uint32_t)(c0 * 2 * R);
                MrgSortSrcList<float> sl(cbuf[base], cbuf[base + (uint32_t)(ql * 2)],
                                         cbuf[base + (uint32_t)(ql * 4)], cbuf[base + (uint32_t)(ql * 6)]);
                MrgSort<float>(obuf[base], sl, pi);
            }
            LocalTensor<float> tmp = cbuf;
            cbuf = obuf;
            obuf = tmp;
            nq = gpc;
            ql *= 4;
        } else {
            for (int64_t c = 0; c < Cs; ++c) {
                MrgSort4Info pi;
                pi.elementLengths[0] = (uint16_t)ql;
                pi.elementLengths[1] = (uint16_t)ql;
                pi.elementLengths[2] = 0;
                pi.elementLengths[3] = 0;
                pi.ifExhaustedSuspension = false;
                pi.validBit = 0x3;
                pi.repeatTimes = 1;
                uint32_t base = (uint32_t)(c * 2 * R);
                MrgSortSrcList<float> sl(cbuf[base], cbuf[base + (uint32_t)(ql * 2)], cbuf[base], cbuf[base]);
                MrgSort<float>(obuf[base], sl, pi);
            }
            LocalTensor<float> tmp = cbuf;
            cbuf = obuf;
            obuf = tmp;
            nq = 1;
            ql *= 2;
        }
    }
    return cbuf;
}

/*!
 * \brief Widen the loaded tile to fp32 and store it in fT.
 *        The tile geometry is (R rows) x (Cs columns); when the rows are not packed (the raw block
 *        length is not a multiple of 32 bytes) the row pitch is srcPitch and the fp32 row pitch Pf,
 *        otherwise the whole thing is a plain contiguous copy of rows*Cs elements.
 */
template <typename T>
__aicore__ inline void TkTileToFloat(const LocalTensor<float> &fT, const LocalTensor<T> &rt,
                                     const LocalTensor<half> &hT, int64_t rows, int64_t elems,
                                     int64_t Pf, int64_t srcPitch, bool packed)
{
    if (packed) {
        const int64_t n = rows * elems;
        if constexpr (std::is_same<T, float>::value) {
            // There is no float -> float Cast: a Cast whose source and destination types are
            // identical does NOT copy anything, which would leave fT at its -inf pre-fill.
            // Multiplying by 1.0f is an exact copy (sign of zero, inf and NaN are preserved).
            Muls(fT, rt, 1.0f, (int32_t)n);
        } else if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            Cast(fT, rt, RoundMode::CAST_NONE, (uint32_t)n);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            Cast(fT, rt, RoundMode::CAST_RINT, (uint32_t)n);
        } else if constexpr (std::is_same<T, int64_t>::value) {
            Cast(fT, rt, RoundMode::CAST_RINT, (uint32_t)n);
        } else {
            Cast(hT, rt, RoundMode::CAST_NONE, (uint32_t)n);
            Cast(fT, hT, RoundMode::CAST_NONE, (uint32_t)n);
        }
    } else {
        const uint16_t sRep = (uint16_t)(srcPitch / 32);
        const uint16_t dRep = (uint16_t)(Pf / 32);
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            const int64_t hPitch = TkAlI(elems * 2, 32);
            TkCastRows<half, T>(hT, rt, rows, elems, sRep, (uint16_t)(hPitch / 32));
            TkCastRows<float, half>(fT, hT, rows, elems, (uint16_t)(hPitch / 32), dRep);
        } else {
            TkCastRows<float, T>(fT, rt, rows, elems, sRep, dRep);
        }
    }
}

/*!
 * \brief Negate the real (loaded) part of the fp32 tile when largest == 0.
 */
__aicore__ inline void TkNegReal(const LocalTensor<float> &fT, int64_t rows, int64_t elems, int64_t Pf,
                                 bool packed)
{
    if (packed) {
        Muls(fT, fT, -1.0f, (int32_t)(rows * elems));
    } else {
        const int64_t step = Pf / 4;
        for (int64_t r = 0; r < rows; ++r) {
            Muls(fT[(uint32_t)(r * step)], fT[(uint32_t)(r * step)], -1.0f, (int32_t)elems);
        }
    }
}

template <typename T>
__global__ __aicore__ void topk_kernel(GM_ADDR xPtr, GM_ADDR yPtr, GM_ADDR iPtr, GM_ADDR ixPtr,
                                       int64_t reduce, int64_t inner, int64_t k, int64_t largest,
                                       int64_t Cs, int64_t R, int64_t groups, int64_t numUnits,
                                       int64_t numBlocks, int64_t ch, int64_t mode, int64_t accPitch,
                                       int64_t chunkRem)
{
    const bool colMode = (mode == 1);
    const int64_t sz = (int64_t)sizeof(T);
    const int64_t N = R * Cs;
    const int64_t Pf = colMode ? TkAlI(Cs * 4, 32) : 4;
    const int64_t srcPitch = TkAlI(Cs * sz, 32);
    const bool packedTile = (!colMode) || ((Cs * sz) % 32 == 0);
    const int64_t accFloats = accPitch / 4;

    GlobalTensor<T> xG;
    GlobalTensor<T> yG;
    GlobalTensor<int64_t> iG;
    GlobalTensor<int64_t> ixG;
    xG.SetGlobalBuffer((__gm__ T *)xPtr);
    yG.SetGlobalBuffer((__gm__ T *)yPtr);
    iG.SetGlobalBuffer((__gm__ int64_t *)iPtr);
    ixG.SetGlobalBuffer((__gm__ int64_t *)ixPtr);

    TPipe pipe;
    TQue<QuePosition::VECIN, TK_DEPTH> rawQ;
    TBuf<TPosition::VECCALC> bH;
    TBuf<TPosition::VECCALC> bF;
    TBuf<TPosition::VECCALC> bC;
    TBuf<TPosition::VECCALC> bA;
    TBuf<TPosition::VECCALC> bB;
    TBuf<TPosition::VECCALC> bAcc;
    TBuf<TPosition::VECCALC> bMo;
    TBuf<TPosition::VECCALC> bCF;
    TBuf<TPosition::VECCALC> bOF;
    TBuf<TPosition::VECCALC> bVF;
    TBuf<TPosition::VECCALC> bIF;
    TBuf<TPosition::VECCALC> bVT;
    TBuf<TPosition::VECCALC> bI64;
    TBuf<TPosition::VECCALC> bHO;
    TBuf<TPosition::VECCALC> bIU;

    int64_t rawBytes = (colMode ? (R * srcPitch) : TkAlI(R * sz, 32)) + 512;
    int64_t hBytes = 512;
    if (sz == 1) {
        hBytes = (colMode ? (R * TkAlI(Cs * 2, 32)) : TkAlI(N * 2, 32)) + 512;
    }
    int64_t fBytes = (colMode ? (R * Pf) : TkAlI(N * 4, 32)) + 512;
    int64_t cBytes = (colMode ? TkAlI(N * 4, 32) : 64) + 512;
    int64_t abBytes = TkAlI(2 * N * 4, 32) + 512;
    int64_t accBytes = Cs * accPitch + 512;
    int64_t moBytes = TkAlI(8 * (k + R), 32) + 512;
    int64_t cfBytes = TkAlI(N * 4, 32) + 512;
    int64_t ofBytes = TkAlI(ch * Cs * 4, 32) + 512;
    int64_t vfBytes = TkAlI(ch * Cs * 4, 32) + 512;
    int64_t vtBytes = TkAlI(ch * Cs * sz, 32) + 512;
    int64_t i64Bytes = TkAlI(ch * Cs * 8, 32) + 512;
    int64_t hoBytes = TkAlI(ch * Cs * 2, 32) + 512;
    int64_t iuBytes = TkAlI(R * 4, 32) + 512;

    pipe.InitBuffer(rawQ, TK_DEPTH, (int32_t)rawBytes);
    pipe.InitBuffer(bH, (int32_t)hBytes);
    pipe.InitBuffer(bF, (int32_t)fBytes);
    pipe.InitBuffer(bC, (int32_t)cBytes);
    pipe.InitBuffer(bA, (int32_t)abBytes);
    pipe.InitBuffer(bB, (int32_t)abBytes);
    pipe.InitBuffer(bAcc, (int32_t)accBytes);
    pipe.InitBuffer(bMo, (int32_t)moBytes);
    pipe.InitBuffer(bCF, (int32_t)cfBytes);
    pipe.InitBuffer(bOF, (int32_t)ofBytes);
    pipe.InitBuffer(bVF, (int32_t)vfBytes);
    pipe.InitBuffer(bIF, (int32_t)vfBytes);
    pipe.InitBuffer(bVT, (int32_t)vtBytes);
    pipe.InitBuffer(bI64, (int32_t)i64Bytes);
    pipe.InitBuffer(bHO, (int32_t)hoBytes);
    pipe.InitBuffer(bIU, (int32_t)iuBytes);

    LocalTensor<float> fT = TkAl(bF.Get<float>());
    LocalTensor<float> colF = colMode ? TkAl(bC.Get<float>()) : fT;
    LocalTensor<float> sA = TkAl(bA.Get<float>());
    LocalTensor<float> sB = TkAl(bB.Get<float>());
    LocalTensor<float> accF = bAcc.Get<float>();
    LocalTensor<float> moF = TkAl(bMo.Get<float>());
    LocalTensor<int32_t> cfI = TkAl(bCF.Get<int32_t>());
    LocalTensor<int32_t> ofI = TkAl(bOF.Get<int32_t>());
    LocalTensor<float> vF = TkAl(bVF.Get<float>());
    LocalTensor<float> iF = TkAl(bIF.Get<float>());
    LocalTensor<half> hT = TkAl(bH.Get<half>());
    LocalTensor<half> hoT = TkAl(bHO.Get<half>());
    LocalTensor<uint32_t> cfU = TkAl(bCF.Get<uint32_t>());
    LocalTensor<uint32_t> ofU = TkAl(bOF.Get<uint32_t>());
    LocalTensor<int32_t> idxI = TkAl(bIU.Get<int32_t>());
    LocalTensor<uint32_t> idxU = TkAl(bIU.Get<uint32_t>());
    LocalTensor<int32_t> i32V = iF.ReinterpretCast<int32_t>();

    // Offset tables (constant for the whole kernel; built once per core).
    {
        LocalTensor<int32_t> ti = sA.ReinterpretCast<int32_t>();
        LocalTensor<float> tf = sB;
        TkBuildTable(cfI, ti, tf, N, R, Pf, 4 - R * Pf);
        TkBuildTable(ofI, ti, tf, ch * Cs, Cs, accPitch, 8 - Cs * accPitch);
    }

    const float ne = TkNegInf();
    const int64_t nTile = (reduce + R - 1) / R;
    const int64_t take = TkMinI(R, k);
    const int64_t per = R / 32;
    const int64_t blk = (int64_t)AscendC::GetBlockIdx();

    for (int64_t u = blk; u < numUnits; u += numBlocks) {
        int64_t o = u;
        int64_t i0 = 0;
        if (colMode) {
            o = u / groups;
            int64_t g = u - o * groups;
            i0 = g * Cs;
            if (i0 + Cs > inner) {
                i0 = inner - Cs;
            }
        }
        // mode 2 (row split): unit u owns the reduce-axis slice
        // [u*reduce + min(u,chunkRem), +unitLen), where `reduce` carries the base chunk length.
        // mode 3 (merge): a single unit over a contiguous value array plus its index array.
        int64_t unitBase = o * reduce * inner + i0;
        int64_t outBase = o * k * inner + i0;
        int64_t unitLen = reduce;
        if (mode == 2) {
            unitBase = u * reduce + TkMinI(u, chunkRem);
            unitLen = reduce + ((u < chunkRem) ? 1 : 0);
            outBase = u * k;
        }
        const int64_t uTiles = (mode == 2) ? ((unitLen + R - 1) / R) : nTile;
        int64_t accLen = 0;

        for (int64_t t = 0; t < uTiles; ++t) {
            const int64_t tstart = t * R;
            const int64_t realLen = TkMinI(R, unitLen - tstart);

            // ---- 1. load and widen
            {
                LocalTensor<T> rt = rawQ.AllocTensor<T>();
                if (colMode) {
                    DataCopyExtParams cp;
                    cp.blockCount = (uint16_t)realLen;
                    cp.blockLen = (uint32_t)(Cs * sz);
                    cp.srcStride = (uint32_t)((inner - Cs) * sz);
                    cp.dstStride = 0;
                    cp.rsv = 0;
                    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
                    DataCopyPad(rt, xG[unitBase + tstart * inner], cp, pp);
                } else {
                    DataCopyExtParams cp;
                    cp.blockCount = 1;
                    cp.blockLen = (uint32_t)(realLen * sz);
                    cp.srcStride = 0;
                    cp.dstStride = 0;
                    cp.rsv = 0;
                    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
                    DataCopyPad(rt, xG[unitBase + tstart], cp, pp);
                }
                rawQ.EnQue(rt);
                rt = rawQ.DeQue<T>();

                // Only the slots past the real data must hold -inf, and a full tile needs no fill
                // at all: the sort can never promote a padding slot above a real one when the unit
                // has at least k real elements.  The fill must START on a 32-byte boundary, so the
                // start is rounded down and a few real slots are refilled - harmless because the
                // load below overwrites them.
                if (realLen < R) {
                    const int64_t tailStart = (realLen * Pf / 4) & ~(int64_t)7;
                    Duplicate(fT[(uint32_t)tailStart], ne, (int32_t)(R * Pf / 4 - tailStart));
                }
                TkTileToFloat<T>(fT, rt, hT, realLen, Cs, Pf, srcPitch, packedTile);
                if (largest == 0) {
                    TkNegReal(fT, realLen, Cs, Pf, packedTile);
                }
                if (colMode) {
                    Gather(colF, fT, cfU, (uint32_t)0, (uint32_t)N);
                }
                rawQ.FreeTensor(rt);
            }

            // ---- 2. reduce index vector (identical for every column of the group)
            if (mode == 3) {
                // merge phase: the indices come from the row-split phase's workspace.  Load the
                // int64 tile through the (unused at this point) merge buffer and narrow it.
                Duplicate(idxI, (int32_t)0, (int32_t)R);
                LocalTensor<int64_t> i64In = TkAl(bMo.Get<int64_t>());
                DataCopyExtParams ciIn;
                ciIn.blockCount = 1;
                ciIn.blockLen = (uint32_t)(realLen * 8);
                ciIn.srcStride = 0;
                ciIn.dstStride = 0;
                ciIn.rsv = 0;
                DataCopyPadExtParams<int64_t> ppIn{false, 0, 0, 0};
                DataCopyPad(i64In, ixG[unitBase + tstart * inner], ciIn, ppIn);
                PipeBarrier<PIPE_ALL>();
                Cast(idxI, i64In, RoundMode::CAST_NONE, (uint32_t)realLen);
            } else if (mode == 2) {
                // Row-split phase: the index stored in the workspace must be the ORIGINAL index in
                // the whole row, i.e. the chunk start (unitBase, with inner == 1) plus the in-chunk
                // position.  Padding slots are clamped to the last element of the chunk.
                CreateVecIndex(idxI, (int32_t)0, (uint32_t)R);
                Adds(idxI, idxI, (int32_t)(unitBase + tstart), (int32_t)R);
                Mins(idxI, idxI, (int32_t)(unitBase + unitLen - 1), (int32_t)R);
            } else {
                // tstart == t * R: build the ramp once per unit and then just advance it, and only
                // clamp on the partial (last) tile - full tiles are already in range.
                if (t == 0) {
                    CreateVecIndex(idxI, (int32_t)0, (uint32_t)R);
                } else {
                    Adds(idxI, idxI, (int32_t)R, (int32_t)R);
                }
                if (realLen < R) {
                    Mins(idxI, idxI, (int32_t)(unitLen - 1), (int32_t)R);
                }
            }

            // ---- 3. Sort32 per column, then the merge tree
            for (int64_t c = 0; c < Cs; ++c) {
                Sort32<float>(sA[(uint32_t)(c * 2 * R)], colF[(uint32_t)(c * R)], idxU, (int32_t)per);
            }
            PipeBarrier<PIPE_V>();
            LocalTensor<float> sorted = TkMergeTree(sA, sB, R, Cs);
            PipeBarrier<PIPE_V>();

            // ---- 4. merge the tile top into the per-column accumulator
            for (int64_t c = 0; c < Cs; ++c) {
                LocalTensor<float> acc = accF[(uint32_t)(c * accFloats)];
                LocalTensor<float> tile = sorted[(uint32_t)(c * 2 * R)];
                if (accLen == 0) {
                    const int64_t tl = TkMinI(R, k);
                    Adds(acc, tile, 0.0f, (int32_t)(2 * tl));
                    if (c == Cs - 1) {
                        accLen = tl;
                    }
                } else {
                    MrgSort4Info pi;
                    pi.elementLengths[0] = (uint16_t)accLen;
                    pi.elementLengths[1] = (uint16_t)take;
                    pi.elementLengths[2] = 0;
                    pi.elementLengths[3] = 0;
                    pi.ifExhaustedSuspension = false;
                    pi.validBit = 0x3;
                    pi.repeatTimes = 1;
                    MrgSortSrcList<float> sl(acc, tile, acc, acc);
                    MrgSort<float>(moF, sl, pi);
                    const int64_t nl = TkMinI(k, accLen + take);
                    Adds(acc, moF, 0.0f, (int32_t)(2 * nl));
                    if (c == Cs - 1) {
                        accLen = nl;
                    }
                }
            }
            PipeBarrier<PIPE_V>();
        }

        // ---- 5. store
        for (int64_t t0 = 0; t0 < k; t0 += ch) {
            const int64_t rows = TkMinI(ch, k - t0);
            const int64_t n = rows * Cs;
            const bool outContig = (!colMode) || (Cs == inner);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            Gather(vF, accF, ofU, (uint32_t)(t0 * 8), (uint32_t)n);
            Gather(iF, accF, ofU, (uint32_t)(t0 * 8 + 4), (uint32_t)n);
            if (largest == 0) {
                Muls(vF, vF, -1.0f, (int32_t)n);
            }
            LocalTensor<T> vT;
            if constexpr (std::is_same<T, float>::value) {
                vT = vF;
            } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                vT = TkAl(bVT.Get<T>());
                Cast(hoT, vF, RoundMode::CAST_RINT, (uint32_t)n);
                Cast(vT, hoT, RoundMode::CAST_RINT, (uint32_t)n);
            } else {
                vT = TkAl(bVT.Get<T>());
                Cast(vT, vF, RoundMode::CAST_RINT, (uint32_t)n);
            }
            LocalTensor<int64_t> i64T = TkAl(bI64.Get<int64_t>());
            Cast(i64T, i32V, RoundMode::CAST_NONE, (uint32_t)n);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            DataCopyExtParams cv;
            cv.srcStride = 0;
            cv.rsv = 0;
            if (outContig) {
                cv.blockCount = 1;
                cv.blockLen = (uint32_t)(rows * Cs * sz);
                cv.dstStride = 0;
            } else {
                cv.blockCount = (uint16_t)rows;
                cv.blockLen = (uint32_t)(Cs * sz);
                cv.dstStride = (uint32_t)((inner - Cs) * sz);
            }
            DataCopyPad(yG[outBase + t0 * inner], vT, cv);

            DataCopyExtParams ci;
            ci.srcStride = 0;
            ci.rsv = 0;
            if (outContig) {
                ci.blockCount = 1;
                ci.blockLen = (uint32_t)(rows * Cs * 8);
                ci.dstStride = 0;
            } else {
                ci.blockCount = (uint16_t)rows;
                ci.blockLen = (uint32_t)(Cs * 8);
                ci.dstStride = (uint32_t)((inner - Cs) * 8);
            }
            DataCopyPad(iG[outBase + t0 * inner], i64T, ci);
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------------
// Host tiling
// ---------------------------------------------------------------------------------------------------

static inline int64_t HAl(int64_t v, int64_t a) { return (v + a - 1) / a * a; }
static inline int64_t HMin(int64_t a, int64_t b) { return a < b ? a : b; }

static int64_t TopkUbNeeded(int64_t R, int64_t Cs, int64_t k, int64_t sz, int64_t ch, int64_t mode)
{
    const int64_t N = R * Cs;
    const bool colMode = (mode == 1);
    const int64_t Pf = colMode ? HAl(Cs * 4, 32) : 4;
    const int64_t srcPitch = HAl(Cs * sz, 32);
    int64_t accPitch = HAl(8 * k, 32);
    if (accPitch < 32) {
        accPitch = 32;
    }

    int64_t need = 0;
    need += (colMode ? (R * srcPitch) : HAl(R * sz, 32)) + 512;
    need += (sz == 1) ? ((colMode ? (R * HAl(Cs * 2, 32)) : HAl(N * 2, 32)) + 512) : 512;
    need += (colMode ? (R * Pf) : HAl(N * 4, 32)) + 512;
    need += (colMode ? HAl(N * 4, 32) : 64) + 512;
    need += 2 * (HAl(2 * N * 4, 32) + 512);
    need += Cs * accPitch + 512;
    need += HAl(8 * (k + R), 32) + 512;
    need += HAl(N * 4, 32) + 512;
    need += HAl(ch * Cs * 4, 32) + 512;
    need += 2 * (HAl(ch * Cs * 4, 32) + 512);
    need += HAl(ch * Cs * sz, 32) + 512;
    need += HAl(ch * Cs * 8, 32) + 512;
    need += HAl(ch * Cs * 2, 32) + 512;
    need += HAl(R * 4, 32) + 512;
    return need;
}

TopkPlan calc_topk_plan(int64_t outer, int64_t reduce, int64_t inner, int64_t k, int64_t largest,
                        int64_t sz)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 192 * 1024;
    int64_t coreNum = 1;
    if (plat != nullptr) {
        uint64_t tmp = 0;
        plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, tmp);
        if (tmp > 0) {
            ubSize = tmp;
        }
        coreNum = plat->GetCoreNumAiv();
        if (coreNum <= 0) {
            coreNum = 1;
        }
    }
    const int64_t budget = (int64_t)ubSize - (int64_t)(ubSize / 8);

    TopkPlan p;
    p.outer = outer;
    p.reduce = reduce;
    p.inner = inner;
    p.k = k;
    p.largest = (largest != 0) ? 1 : 0;
    p.mode = (inner <= 1) ? 0 : 1;

    int64_t csUnit = (sz == 1) ? 32 : ((sz == 2) ? 16 : 8);

    // candidate R values (largest first).  Measured per-tile cost is ~0.7us fixed + ~0.9ns per
    // element, so a bigger tile amortises the fixed part; the search picks the cheapest feasible.
    int64_t rCand[8] = {4096, 2048, 1024, 512, 256, 128, 64, 32};
    const int nR = 8;

    double bestCost = -1.0;
    int64_t bestCs = 1;
    int64_t bestR = 32;
    int64_t bestCh = 1;

    auto eval = [&](int64_t Cs, int64_t R) {
        int64_t ch = (int64_t)(8192 / (Cs * 8));
        if (ch < 1) {
            ch = 1;
        }
        if (ch > k) {
            ch = k;
        }
        if (ch > 2 * R) {
            ch = 2 * R;
        }
        if (ch < 1) {
            ch = 1;
        }
        int64_t need = TopkUbNeeded(R, Cs, k, sz, ch, p.mode);
        if (need > budget) {
            return;
        }
        int64_t ntiles = (reduce + R - 1) / R;
        int64_t tk = HMin(R, k);
        double lg = 0.0;
        {
            int64_t v = R / 32;
            while (v > 1) {
                lg += 1.0;
                v /= 2;
            }
        }
        // TOTAL cost (ns) over the whole case, not a per-unit cost: the number of units depends on
        // Cs, so a per-unit cost cannot see that a larger Cs removes whole units' worth of DMA
        // transactions and per-tile pipeline drains.  Calibrated from the measured baseline:
        //   ~700ns fixed per tile (5 pipeline drains), ~(0.30+0.15*log2(R/32)) ns per element
        //   processed, ~0.30ns per merge struct plus ~20ns per MrgSort instruction, ~6ns per DMA
        //   transaction (mode 1 issues `reduce` transactions per unit, mode 0 one per tile).
        const bool colM = (p.mode == 1);
        const double csE = colM ? (double)Cs : 1.0;
        const double perUnit = 700.0 * (double)ntiles +
                               (0.30 + 0.15 * lg) * (double)reduce * csE +
                               (0.30 * (double)(k + tk) + 20.0) * (double)ntiles * csE +
                               6.0 * (colM ? (double)reduce : (double)ntiles);
        const double nUnits = colM ? (double)outer * (double)((inner + Cs - 1) / Cs) : (double)outer;
        double cost = perUnit * nUnits;
        if (bestCost < 0.0 || cost < bestCost) {
            bestCost = cost;
            bestCs = Cs;
            bestR = R;
            bestCh = ch;
        }
    };

    if (p.mode == 0) {
        for (int i = 0; i < nR; ++i) {
            eval(1, rCand[i]);
        }
    } else {
        int64_t csList[8];
        int nCs = 0;
        if (inner < csUnit) {
            csList[nCs++] = inner;
        } else {
            int64_t c = csUnit;
            while (c <= inner && c <= 256 && nCs < 8) {
                csList[nCs++] = c;
                c *= 2;
            }
        }
        for (int ci = nCs - 1; ci >= 0; --ci) {
            for (int i = 0; i < nR; ++i) {
                eval(csList[ci], rCand[i]);
            }
        }
        if (bestCost < 0.0) {
            // last resort: minimal tile
            bestCs = (inner < csUnit) ? inner : csUnit;
            if (bestCs > inner) {
                bestCs = inner;
            }
            bestR = 32;
            bestCh = (k < 1) ? 1 : ((k < 64) ? k : 64);
        }
    }

    int64_t accPitch = HAl(8 * k, 32);
    if (accPitch < 32) {
        accPitch = 32;
    }
    p.Cs = bestCs;
    p.R = bestR;
    p.ch = bestCh;
    p.accPitch = accPitch;
    p.coreNum = coreNum;
    if (p.mode == 0) {
        p.groups = 1;
    } else {
        p.groups = (inner + bestCs - 1) / bestCs;
    }
    p.numUnits = outer * p.groups;
    p.numBlocks = HMin(coreNum, p.numUnits);
    if (p.numBlocks < 1) {
        p.numBlocks = 1;
    }
    p.ubNeeded = TopkUbNeeded(bestR, bestCs, k, sz, bestCh, p.mode);
    return p;
}

// ---------------------------------------------------------------------------------------------------
// Launch wrappers
// ---------------------------------------------------------------------------------------------------

extern "C" {

static void topk_launch_impl(int64_t typeCode, GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR ix,
                             int64_t reduce, int64_t inner, int64_t k, int64_t largest, int64_t Cs,
                             int64_t R, int64_t groups, int64_t numUnits, int64_t numBlocks, int64_t ch,
                             int64_t mode, int64_t accPitch, int64_t chunkRem, void *stream)
{
    switch (typeCode) {
        case TK_T_F32:
            topk_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, idx, ix, reduce, inner, k, largest, Cs,
                                                               R, groups, numUnits, numBlocks, ch, mode,
                                                               accPitch, chunkRem);
            break;
        case TK_T_F16:
            topk_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, idx, ix, reduce, inner, k, largest, Cs,
                                                              R, groups, numUnits, numBlocks, ch, mode,
                                                              accPitch, chunkRem);
            break;
        case TK_T_BF16:
            topk_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, idx, ix, reduce, inner, k, largest,
                                                                    Cs, R, groups, numUnits, numBlocks, ch,
                                                                    mode, accPitch, chunkRem);
            break;
        case TK_T_I8:
            topk_kernel<int8_t><<<numBlocks, nullptr, stream>>>(x, y, idx, ix, reduce, inner, k, largest, Cs,
                                                                R, groups, numUnits, numBlocks, ch, mode,
                                                                accPitch, chunkRem);
            break;
        case TK_T_U8:
            topk_kernel<uint8_t><<<numBlocks, nullptr, stream>>>(x, y, idx, ix, reduce, inner, k, largest, Cs,
                                                                 R, groups, numUnits, numBlocks, ch, mode,
                                                                 accPitch, chunkRem);
            break;
        case TK_T_I32:
            topk_kernel<int32_t><<<numBlocks, nullptr, stream>>>(x, y, idx, ix, reduce, inner, k, largest, Cs,
                                                                 R, groups, numUnits, numBlocks, ch, mode,
                                                                 accPitch, chunkRem);
            break;
        default:
            topk_kernel<int64_t><<<numBlocks, nullptr, stream>>>(x, y, idx, ix, reduce, inner, k, largest, Cs,
                                                                 R, groups, numUnits, numBlocks, ch, mode,
                                                                 accPitch, chunkRem);
            break;
    }
}

void launch_topk(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR ix, int64_t typeCode, int64_t outer,
                 int64_t reduce, int64_t inner, int64_t k, int64_t largest, int64_t Cs, int64_t R,
                 int64_t groups, int64_t numUnits, int64_t numBlocks, int64_t ch, int64_t mode,
                 int64_t accPitch, int64_t chunkRem, void *stream)
{
    (void)outer;
    topk_launch_impl(typeCode, x, y, idx, ix, reduce, inner, k, largest, Cs, R, groups, numUnits, numBlocks,
                     ch, mode, accPitch, chunkRem, stream);
}

} // extern "C"
