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
 * \file strided_slice_kernel.cpp
 * \brief StridedSlice (TensorFlow style slice with masks) direct-launch kernel, tiling and launch.
 *
 *   y = x[begin[0]:end[0]:strides[0], begin[1]:end[1]:strides[1], ...]  with the
 *   begin_mask / end_mask / ellipsis_mask / shrink_axis_mask / new_axis_mask modifiers.
 *
 * The output is contiguous and every output dimension maps to one input axis with a fixed step, so
 * the operator is a strided gather from a contiguous source:
 *     in_off(k) = baseOffset + sum_d idx_d(k) * step[d]        (element units)
 * The host side reduces the request to the "effective" dims (sliced axes with output length > 1;
 * shrunk axes collapse into baseOffset, new axes have length 1) and merges neighbours while the
 * mapping stays an arithmetic progression (step[d-1] == size[d]*step[d]).
 *
 * Execution paths
 *   path 1 -- work horse for whole contiguous rows: a unit is `u` rows of one uniformly strided
 *             group, moved by one multi block load (blockCount = u, blockLen = rowBytes,
 *             GM gap = (rowGap-innerLen)*es) and one multi block store (contiguous output).  Both
 *             directions see the same UB row pitch align32(blockLen), so the two are consistent.
 *             Collapses the many-tiny-DMA shapes (a 127 wide row with a 128 gap was 1032256 single
 *             row units and 19 ms) into one unit per >100 rows.
 *   path 0 -- R == 1 and inner step 1: pure contiguous copy.
 *   path 5 -- rows whose inner dim is strided (inner step > 1, element <= 4 byte), batched: the unit
 *             loads `rn` complete row spans with one multi block DMA (blockLen = the span the row's
 *             taps need, GM gap = rowGap - span), compacts them with the vector Gather, and stores the
 *             packed rows.  The Gather table has two forms, both indexed by the packed output index
 *             i = r*perRowTbl + c into a table of int32 offsets off[i]:
 *               compact (padded span == innerLen*step*es, perRowTbl == innerLen): ONE global ramp
 *                 off[i] = i*step*es over all rnMax*innerLen entries, and the store is a single block;
 *               padded (otherwise, perRowTbl == align32(innerLen*es)/es): the output rows sit at the
 *                 natural hardware pitch align32(innerLen*es) - the pitch a single block store gets
 *                 anyway - so the table is laid out one block of perRowTbl int32 per row with a byte
 *                 pitch perRowTbl*4 that is a multiple of 32, and entry (r,c) = r*pitch + c*step*es.
 *                 It is built as a row 0 ramp plus one vector Add per row (dst offset r*perRowTbl, 32B
 *                 aligned).  The entries past the row's last tap keep ramping, so the staging buffer is
 *                 sized with that slack.  The store is then a plain multi block DMA with BOTH strides
 *                 left at 0, which is what makes it independent of the UB stride unit.  This is what
 *                 lets a row whose span is not a multiple of 32 bytes (e.g. 250 taps x 2 x 4 = 1996 ->
 *                 pitch 2016 against a packed 2000) still batch many rows per DMA.
 *   path 6 -- 8 byte elements with a strided inner dim (SS_ENABLE_I64GATHER).  The vector Gather is
 *             not dependable for 8 byte types, so the element is split: the same source is viewed as
 *             int32 and the LOWER AND UPPER half of every output element are gathered with one table
 *             entry each, off[2j+b] = j*step*8 + b*4.  That table is not a plain ramp, but it is a
 *             doubling construct: seed 8 entries and each doubling adds step*4*len over a block of len
 *             entries (the pair index advances by len/2, so the byte offset advances by step*4*len).
 *             The gathered int32 stream IS the packed int64 output (little endian), so the store is
 *             one contiguous block.  Path 6 works on its own int64 typed global/local views, because
 *             the enclosing kernel is a template over the operator's own element type.  Measured: case 5
 *             6761 us -> 141.5 us (perf 0.04 -> 0.99), i.e. the metric floor.  The scalar alternative
 *             (path 3) measures ~35 cycles per element; a two blocks per element DMA variant is even
 *             worse (~21 cycles per block per core, calibrated from level3 case 20).
 *   path 2 -- contiguous span read per row tile (single block) + vector Gather + single block store;
 *             no inter-block UB pitch at all, so it is the universal fallback and what produced the
 *             first 20/20 run.  Gather wants byte offsets and srcBaseAddr = 0 for a LocalTensor
 *             source (runbook runbooks/precision/localtensor_gather_base_address_not_double_counted.md).
 *   path 3 -- element wise scalar fallback: negative inner steps, and shapes whose row structure fits
 *             no DMA form (including every 8 byte case that path 6 declined).
 *   path 4 -- multi block strided 1D gather, one element per block (SS_ENABLE_GRANULE).  Measured
 *             SLOWER than the span path on its only candidate (level3/strided_slice_15: 42.8 us
 *             against 14.8 us): a multi block DMA costs tens of cycles per block while the span path
 *             pays one descriptor per tile.  Hence disabled.
 *
 * READ LENGTH (SS_WIDE_READ): a multi block load whose blockLen is not a multiple of 32 bytes forces a
 * partial 32B granule at every block boundary (level3 case 2 moves 2047 blocks of 8188 bytes with a 4
 * byte gap).  When the row pitch has room (rowGap*es >= align32(rowBytes)) the load is issued with the
 * ALIGNED row length and the gap recomputed as rowGap*es - align32(rowBytes), which makes the whole run
 * contiguous.  Measured: case 2 27.96 us -> 20.48 us.  The padding reads only data that is still inside
 * the row's own pitch, it is never stored (the store keeps moving innerLen elements per row), and it
 * cannot leave the tensor by more than 31 bytes even in the worst case.
 *
 * STAGING COPY (SS_VEC_COPY): the load goes into a VECIN queue and the store must leave from a VECOUT
 * queue (a single queue loses the store - see below), so the data is moved between the two.  A UB->UB
 * DataCopy does that in 32B granules at roughly 0.5 TB/s, which is what dominates the many row cases
 * (level3 case 20 streams ~526 MB through it).  Adds performs the same transfer on the vector pipe,
 * which is far quicker.  Adds only accepts half/float/int16/int32, so the buffers are viewed as a 16
 * or 32 bit integer type of the same width - an integer add of zero is a bit exact copy for every
 * operator element type (including -0.0 and NaN patterns), and an 8 byte element is simply two int32
 * lanes.  1 byte elements go through int16 when the count is even, otherwise the DataCopy is kept.
 *
 * Both staging paths use the exact structure of the known good `add` scaffold - a VECIN queue for the
 * load, a vector step, a VECOUT queue for the store, the store buffer freed only after the store has
 * been issued.  A prior revision loaded and stored through ONE VECIN queue: the store never became
 * visible (garbage on the large cases, the NaN signature MERE=MARE=0.000000 on the small fp16/bf16
 * ones), i.e. the staging buffer was reused before the asynchronous store had drained.
 *
 * UB row pitch: only the value 0 is ever passed as a UB side stride, and the hardware then spaces the
 * rows by align32(blockLen) itself.  Writing a gap explicitly is not safe: the UB side stride of
 * DataCopyExtParams counts 32 byte datablocks, not bytes, so a byte sized gap is interpreted 32x too
 * large - measured "The read address of the MTE instruction is out of range"
 * (level3/strided_slice_8) and, earlier, a hard MTE write address out of range.  Every path that needs
 * a row pitch on both sides therefore derives it from align32(blockLen) as well.
 *
 * The __global__ entry is explicitly marked AIV-only: the automatic kernel-type derivation does not
 * cover a signature carrying an aggregate argument and ld.lld then rejects the derived type
 * ("auto derivate failed, please manually add the function type attribute").
 */

#include <algorithm>
#include <tuple>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "strided_slice_launch.h"

/* diagnostic switches */
#ifndef SS_FORCE_SCALAR
#define SS_FORCE_SCALAR 0
#endif
#ifndef SS_ENABLE_DMA_PATHS
#define SS_ENABLE_DMA_PATHS 1
#endif
#ifndef SS_ENABLE_MULTIROW
#define SS_ENABLE_MULTIROW 1
#endif
#ifndef SS_ENABLE_PADDED_TABLE
#define SS_ENABLE_PADDED_TABLE 1
#endif
#ifndef SS_ENABLE_I64GATHER
#define SS_ENABLE_I64GATHER 1
#endif
#ifndef SS_ENABLE_GRANULE
#define SS_ENABLE_GRANULE 0
#endif
/* 1: load align32(rowBytes) per row when the row pitch has room (see READ LENGTH in the header) */
#ifndef SS_WIDE_READ
#define SS_WIDE_READ 1
#endif
/* 1: move the staging data between the queues with a vector Adds instead of a UB->UB DataCopy */
#ifndef SS_VEC_COPY
#define SS_VEC_COPY 1
#endif

namespace stridedslice_ns {

using namespace AscendC;

constexpr int64_t SS_ALIGN = 32;
constexpr int64_t SS_COPY_CHUNK = 8192;
/* the vector copy chunks in units of the viewed integer type; keep the count well inside the 255
 * repeat window for the widest type (255 * 64 = 16320 lanes) */
constexpr int64_t SS_VEC_CHUNK = 4096;
constexpr int64_t SS_MAX_GATHER = 8192;
#define SS_ALIGN_UP(v, a) (((v) + (a) - 1) / (a) * (a))
#define SS_ALIGN_DN(v, a) ((v) / (a) * (a))
/* Extra staging bytes a padded table can address past the last row of the batch: the tail entries of
 * a row keep ramping (c*step*es) instead of stopping at the row's last tap, so a reference into the
 * row after the last one is possible.  Zero when the ramp cannot leave the span.  A macro because it
 * is used from both the host tiling function and the aicore kernel (an __aicore__ helper cannot be
 * called from host code and a plain helper is not callable from device code). */
#define SS_PADDED_SLACK(tabEnt, step, es, pitch) \
    ((((tabEnt) - 1) * (step) * (es) + (es)) > (pitch) \
         ? (((tabEnt) - 1) * (step) * (es) + (es)) - (pitch) \
         : 0)

/* mixed radix -> element offset of the odometer index `idx` over dims [0, nd) */
__aicore__ inline int64_t SsOd(int64_t idx, const StridedSliceTiling &t)
{
    int64_t off = 0;
    for (int64_t d = t.nd - 1; d >= 0; --d) {
        int64_t s = t.sz[d];
        if (s > 1) {
            int64_t i = idx % s;
            idx /= s;
            off += i * t.st[d];
        }
    }
    return off;
}

/* build off[i] = i * stepBytes for i in [0, n) on a 32B aligned int32 table */
__aicore__ inline void SsBuildRamp(LocalTensor<int32_t> offI, int64_t n, int64_t stepBytes)
{
    const int64_t seed = (n < 8) ? n : 8;
    for (int64_t i = 0; i < seed; ++i) {
        offI.SetValue(i, static_cast<int32_t>(i * stepBytes));
    }
    PipeBarrier<PIPE_ALL>();
    int64_t len = 8;
    while (len < n) {
        int64_t m = n - len;
        if (m > len) {
            m = len;
        }
        m = SS_ALIGN_DN(m, 8);
        if (m <= 0) {
            break;
        }
        Adds(offI[len], offI, static_cast<int32_t>(len * stepBytes), static_cast<int32_t>(m));
        PipeBarrier<PIPE_V>();
        len += m;
    }
    PipeBarrier<PIPE_ALL>();
}

/* build the int64 half table off[2j+b] = j*step*8 + b*4 for j < n pairs.
 * A doubling block of len entries spans len/2 pairs, so its constant is step*4*len. */
__aicore__ inline void SsBuildHalfTable(LocalTensor<int32_t> offI, int64_t n, int64_t step)
{
    const int64_t cnt = 2 * n;
    const int64_t seed = (cnt < 8) ? cnt : 8;
    for (int64_t i = 0; i < seed; ++i) {
        const int64_t j = i / 2;
        offI.SetValue(i, static_cast<int32_t>(j * step * 8 + (i % 2) * 4));
    }
    PipeBarrier<PIPE_ALL>();
    int64_t len = 8;
    while (len < cnt) {
        int64_t m = cnt - len;
        if (m > len) {
            m = len;
        }
        m = SS_ALIGN_DN(m, 8);
        if (m <= 0) {
            break;
        }
        Adds(offI[len], offI, static_cast<int32_t>(step * 4 * len), static_cast<int32_t>(m));
        PipeBarrier<PIPE_V>();
        len += m;
    }
    PipeBarrier<PIPE_ALL>();
}

/* raw staging move with the plain UB->UB DataCopy */
template <typename T>
__aicore__ inline void SsRawCopy(LocalTensor<T> dst, LocalTensor<T> src, int64_t total)
{
    for (int64_t k = 0; k < total; k += SS_COPY_CHUNK) {
        int64_t nn = total - k;
        if (nn > SS_COPY_CHUNK) {
            nn = SS_COPY_CHUNK;
        }
        DataCopy(dst[k], src[k], static_cast<uint32_t>(nn));
    }
}

/* vector staging move on an integer view of the same bytes (an integer add of zero is bit exact) */
template <typename V>
__aicore__ inline void SsVecCopyTyped(LocalTensor<V> d, LocalTensor<V> s, int64_t cnt)
{
    for (int64_t k = 0; k < cnt; k += SS_VEC_CHUNK) {
        int64_t nn = cnt - k;
        if (nn > SS_VEC_CHUNK) {
            nn = SS_VEC_CHUNK;
        }
        Adds(d[k], s[k], static_cast<V>(0), static_cast<int32_t>(nn));
    }
    PipeBarrier<PIPE_V>();
}

/* move `total` elements from a VECIN staging buffer to a VECOUT staging buffer.  The UB->UB DataCopy
 * runs in 32 byte granules and dominates the many row shapes, so the vector pipe is used whenever the
 * element width can be viewed as a type that Adds accepts (half/float/int16/int32). */
template <typename T>
__aicore__ inline void SsStageCopy(LocalTensor<T> dst, LocalTensor<T> src, int64_t total)
{
#if SS_VEC_COPY
    if constexpr (sizeof(T) == 4) {
        SsVecCopyTyped<int32_t>(dst.template ReinterpretCast<int32_t>(),
                                src.template ReinterpretCast<int32_t>(), total);
        return;
    } else if constexpr (sizeof(T) == 8) {
        SsVecCopyTyped<int32_t>(dst.template ReinterpretCast<int32_t>(),
                                src.template ReinterpretCast<int32_t>(), total * 2);
        return;
    } else if constexpr (sizeof(T) == 2) {
        SsVecCopyTyped<int16_t>(dst.template ReinterpretCast<int16_t>(),
                                src.template ReinterpretCast<int16_t>(), total);
        return;
    } else {
        if ((total % 2) == 0) {
            SsVecCopyTyped<int16_t>(dst.template ReinterpretCast<int16_t>(),
                                    src.template ReinterpretCast<int16_t>(), total / 2);
            return;
        }
    }
#endif
    SsRawCopy<T>(dst, src, total);
}

/* ------------------------------------------------------------------ kernel */

template <typename T>
__global__ __aicore__ void strided_slice_kernel(GM_ADDR x, GM_ADDR y, StridedSliceTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (t.outNumel <= 0) {
        return;
    }
    const int64_t blk = GetBlockIdx();
    if (blk >= t.numBlocks) {
        return;
    }
    int64_t per = (t.numUnits + t.numBlocks - 1) / t.numBlocks;
    int64_t u0 = blk * per;
    int64_t u1 = u0 + per;
    if (u1 > t.numUnits) {
        u1 = t.numUnits;
    }
    if (u0 >= u1) {
        return;
    }

    TPipe pipe;
    GlobalTensor<T> xG;
    GlobalTensor<T> yG;
    xG.SetGlobalBuffer((__gm__ T *)x);
    yG.SetGlobalBuffer((__gm__ T *)y);

    if (t.path == 3) {
        for (int64_t u = u0; u < u1; ++u) {
            const int64_t rowBase = t.baseOffset + SsOd(u, t);
            const int64_t dst = u * t.innerLen;
            for (int64_t j = 0; j < t.innerLen; ++j) {
                yG.SetValue(dst + j, xG.GetValue(rowBase + j * t.innerStep));
            }
        }
        return;
    }

    if (t.path == 0 || t.path == 1) {
        const bool rows = (t.path == 1);
        const int64_t es = (int64_t)sizeof(T);
        const int64_t rowBytes = rows ? (t.innerLen * es) : SS_ALIGN_UP(t.unitLen * es, SS_ALIGN);
        const int64_t padRowBytes = SS_ALIGN_UP(rowBytes, SS_ALIGN);
        const int64_t u = t.unitLen;
        const int64_t rowsPerUnit = rows ? u : 1;
        const int64_t bufBytes = padRowBytes * rowsPerUnit;
        const int64_t padElems = padRowBytes / es;
        /* both strides stay 0: the hardware spaces each block by align32(blockLen) itself, so the
         * load and the store see the same pitch and no gap has to be written. */
        const int64_t ubGap = 0;
        /* the aligned read length and the matching GM gap (see READ LENGTH in the file header) */
        const int64_t wideRead = (SS_WIDE_READ && rows && padRowBytes <= t.rowGap * es) ? 1 : 0;
        const int64_t readBytes = wideRead ? padRowBytes : rowBytes;

        TQue<QuePosition::VECIN, 2> inQ;
        TQue<QuePosition::VECOUT, 2> outQ;
        pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(bufBytes));
        pipe.InitBuffer(outQ, 2, static_cast<uint32_t>(bufBytes));

        for (int64_t i = u0; i < u1; ++i) {
            int64_t srcOff = 0;
            int64_t dstOff = 0;
            int64_t rn = 1;
            int64_t blkBytes = rowBytes;
            int64_t rdBytes = rows ? readBytes : rowBytes;
            int64_t gmGap = 0;
            if (rows) {
                const int64_t g = i / t.chunksPerGroup;
                const int64_t c = i - g * t.chunksPerGroup;
                const int64_t r0 = c * u;
                rn = t.nRowsPerGroup - r0;
                if (rn > u) {
                    rn = u;
                }
                if (rn <= 0) {
                    continue;
                }
                srcOff = t.baseOffset + SsOd(g, t) + r0 * t.rowGap;
                dstOff = g * t.nRowsPerGroup * t.innerLen + r0 * t.innerLen;
                gmGap = wideRead ? (t.rowGap * es - padRowBytes) : ((t.rowGap - t.innerLen) * es);
            } else {
                const int64_t off = i * t.unitLen;
                int64_t n = t.outNumel - off;
                if (n > t.unitLen) {
                    n = t.unitLen;
                }
                srcOff = t.baseOffset + off;
                dstOff = off;
                rn = 1;
                blkBytes = n * es;
                rdBytes = blkBytes;
            }

            auto inL = inQ.AllocTensor<T>();
            DataCopyExtParams cp{static_cast<uint16_t>(rn), static_cast<uint32_t>(rdBytes),
                                 static_cast<uint32_t>(gmGap), static_cast<uint32_t>(ubGap), 0};
            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            DataCopyPad(inL, xG[srcOff], cp, pp);
            inQ.EnQue(inL);
            inL = inQ.DeQue<T>();

            auto outL = outQ.AllocTensor<T>();
            const int64_t total = rows ? (rn * padElems) : (blkBytes / es);
            SsStageCopy<T>(outL, inL, total);
            inQ.FreeTensor(inL);
            outQ.EnQue(outL);
            outL = outQ.DeQue<T>();

            DataCopyExtParams co{static_cast<uint16_t>(rn), static_cast<uint32_t>(blkBytes),
                                 static_cast<uint32_t>(ubGap), 0, 0};
            DataCopyPad(yG[dstOff], outL, co);
            outQ.FreeTensor(outL);
        }
        return;
    }

    if (t.path == 5) {
        /* batched strided rows: one multi block span load, one Gather, one packed store */
        const int64_t as = t.innerStep;
        const int64_t es = (int64_t)sizeof(T);
        const int64_t rowSpan = (t.innerLen - 1) * as + 1;
        const int64_t rowSpanBytes = rowSpan * es;
        const int64_t padRowSpanBytes = SS_ALIGN_UP(rowSpanBytes, SS_ALIGN);
        const int64_t rnMax = t.unitLen;
        const int64_t tabEnt = t.ubGap; /* 0 => compact global ramp, else int32 entries per row */
        const bool compact = (tabEnt == 0);
        const int64_t perRowTbl = compact ? t.innerLen : tabEnt;
        const int64_t cntMax = rnMax * perRowTbl;
        const int64_t slack = compact ? 0 : SS_PADDED_SLACK(tabEnt, as, es, padRowSpanBytes);
        const int64_t inBytes = padRowSpanBytes * rnMax + slack;
        const int64_t outRowBytes = SS_ALIGN_UP(t.innerLen * es, SS_ALIGN);
        const int64_t outBytes = compact ? SS_ALIGN_UP(cntMax * es, SS_ALIGN)
                                         : (outRowBytes * rnMax);
        const int64_t offBytes = SS_ALIGN_UP(cntMax * 4, SS_ALIGN);
        const int64_t rowOutBytes = t.innerLen * es;
        const int64_t stepBytes = as * es;
        /* the aligned read length and the matching GM gap (see READ LENGTH in the file header) */
        const int64_t wideSpan = (SS_WIDE_READ && padRowSpanBytes <= t.rowGap * es) ? 1 : 0;
        const int64_t readSpanBytes = wideSpan ? padRowSpanBytes : rowSpanBytes;
        const int64_t gmGap = wideSpan ? (t.rowGap * es - padRowSpanBytes)
                                       : ((t.rowGap - rowSpan) * es);

        TQue<QuePosition::VECIN, 2> inQ;
        TQue<QuePosition::VECOUT, 2> outQ;
        TBuf<TPosition::VECCALC> offBuf;
        pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(inBytes));
        pipe.InitBuffer(outQ, 2, static_cast<uint32_t>(outBytes));
        pipe.InitBuffer(offBuf, static_cast<uint32_t>(offBytes));

        /* the compact table is ONE global ramp over every entry the Gather will index; the padded
         * table uses perRowTbl entries per row (a row 0 ramp plus one Add per row), so filling cntMax
         * entries covers both and never leaves an uninitialised offset behind */
        LocalTensor<int32_t> offI = offBuf.Get<int32_t>();
        SsBuildRamp(offI, cntMax, stepBytes);
        if (!compact) {
            /* row r of the table is row 0 shifted by r * the staging row pitch; the destination
             * r*perRowTbl is 4*perRowTbl bytes in, itself a multiple of 32 */
            for (int64_t r = 1; r < rnMax; ++r) {
                const int32_t add = static_cast<int32_t>(r * padRowSpanBytes);
                for (int64_t k = 0; k < perRowTbl; k += SS_COPY_CHUNK) {
                    int64_t nn = perRowTbl - k;
                    if (nn > SS_COPY_CHUNK) {
                        nn = SS_COPY_CHUNK;
                    }
                    Adds(offI[r * perRowTbl + k], offI[k], add, static_cast<int32_t>(nn));
                }
                PipeBarrier<PIPE_V>();
            }
            PipeBarrier<PIPE_ALL>();
        }
        LocalTensor<uint32_t> offU = offI.ReinterpretCast<uint32_t>();

        for (int64_t i = u0; i < u1; ++i) {
            const int64_t g = i / t.chunksPerGroup;
            const int64_t c = i - g * t.chunksPerGroup;
            const int64_t r0 = c * rnMax;
            int64_t rn = t.nRowsPerGroup - r0;
            if (rn > rnMax) {
                rn = rnMax;
            }
            if (rn <= 0) {
                continue;
            }
            const int64_t cnt = rn * perRowTbl;
            const int64_t srcOff = t.baseOffset + SsOd(g, t) + r0 * t.rowGap;
            const int64_t dstOff = g * t.nRowsPerGroup * t.innerLen + r0 * t.innerLen;

            auto inL = inQ.AllocTensor<T>();
            DataCopyExtParams cp{static_cast<uint16_t>(rn), static_cast<uint32_t>(readSpanBytes),
                                 static_cast<uint32_t>(gmGap), 0, 0};
            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            DataCopyPad(inL, xG[srcOff], cp, pp);
            inQ.EnQue(inL);
            inL = inQ.DeQue<T>();

            auto outL = outQ.AllocTensor<T>();
            const int64_t cntv = SS_ALIGN_DN(cnt, 8);
            if (cntv > 0) {
                Gather(outL, inL, offU, static_cast<uint32_t>(0), static_cast<uint32_t>(cntv));
            }
            if (compact) {
                for (int64_t k = cntv; k < cnt; ++k) {
                    outL.SetValue(k, inL.GetValue(k * as));
                }
            }
            PipeBarrier<PIPE_ALL>();
            outQ.EnQue(outL);
            outL = outQ.DeQue<T>();

            if (compact) {
                DataCopyExtParams co{static_cast<uint16_t>(1), static_cast<uint32_t>(cnt * es), 0, 0,
                                     0};
                DataCopyPad(yG[dstOff], outL, co);
            } else {
                /* the output rows already sit at align32(innerLen*es), the pitch a multi block DMA
                 * uses by itself, so both strides can stay 0 */
                DataCopyExtParams co{static_cast<uint16_t>(rn), static_cast<uint32_t>(rowOutBytes),
                                     0, 0, 0};
                DataCopyPad(yG[dstOff], outL, co);
            }
            inQ.FreeTensor(inL);
            outQ.FreeTensor(outL);
        }
        return;
    }

    if (t.path == 6) {
        /* 8 byte elements: gather the lower and the upper half of every output element through an
         * int32 view, which yields the packed int64 output in one contiguous store.  This path is
         * reached for every element type of the enclosing template, so it needs its own int64 views
         * of the buffers and of the global tensors. */
        GlobalTensor<int64_t> xG64;
        GlobalTensor<int64_t> yG64;
        xG64.SetGlobalBuffer((__gm__ int64_t *)x);
        yG64.SetGlobalBuffer((__gm__ int64_t *)y);
        const int64_t as = t.innerStep;
        const int64_t TL = t.unitLen;
        const int64_t inBytes = ((TL - 1) * as + 1) * 8;
        const int64_t outBytes = TL * 8;
        const int64_t offBytes = SS_ALIGN_UP(2 * TL * 4, SS_ALIGN);

        TQue<QuePosition::VECIN, 2> inQ;
        TQue<QuePosition::VECOUT, 2> outQ;
        TBuf<TPosition::VECCALC> offBuf;
        pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(SS_ALIGN_UP(inBytes, SS_ALIGN)));
        pipe.InitBuffer(outQ, 2, static_cast<uint32_t>(SS_ALIGN_UP(outBytes, SS_ALIGN)));
        pipe.InitBuffer(offBuf, static_cast<uint32_t>(offBytes));

        LocalTensor<int32_t> offI = offBuf.Get<int32_t>();
        SsBuildHalfTable(offI, TL, as);
        LocalTensor<uint32_t> offU = offI.ReinterpretCast<uint32_t>();

        for (int64_t u = u0; u < u1; ++u) {
            const int64_t r = u / t.chunksPerRow;
            const int64_t c = u - r * t.chunksPerRow;
            const int64_t j0 = c * TL;
            int64_t ln = t.innerLen - j0;
            if (ln > TL) {
                ln = TL;
            }
            const int64_t srcOff = t.baseOffset + SsOd(r, t) + j0 * as;
            const int64_t inSpan = (ln - 1) * as + 1;

            auto inL = inQ.AllocTensor<int64_t>();
            DataCopyExtParams cp{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(inSpan * (int64_t)sizeof(int64_t)), 0, 0, 0};
            DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
            DataCopyPad(inL, xG64[srcOff], cp, pp);
            inQ.EnQue(inL);
            inL = inQ.DeQue<int64_t>();

            auto outL = outQ.AllocTensor<int64_t>();
            LocalTensor<int32_t> in32 = inL.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> out32 = outL.template ReinterpretCast<int32_t>();
            const int64_t cnt = 2 * ln;
            const int64_t cntv = SS_ALIGN_DN(cnt, 8);
            if (cntv > 0) {
                Gather(out32, in32, offU, static_cast<uint32_t>(0), static_cast<uint32_t>(cntv));
            }
            for (int64_t k = cntv; k < cnt; ++k) {
                out32.SetValue(k, in32.GetValue(offI.GetValue(k) / 4));
            }
            PipeBarrier<PIPE_ALL>();
            outQ.EnQue(outL);
            outL = outQ.DeQue<int64_t>();

            DataCopyExtParams co{static_cast<uint16_t>(1),
                                 static_cast<uint32_t>(ln * (int64_t)sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(yG64[r * t.innerLen + j0], outL, co);
            inQ.FreeTensor(inL);
            outQ.FreeTensor(outL);
        }
        return;
    }

    /* path 2: contiguous span + Gather, one row tile per work item */
    {
        const int64_t as = t.innerStep;
        const int64_t es = (int64_t)sizeof(T);
        const int64_t TL = t.unitLen;
        const int64_t spanElems = (TL - 1) * as + 1;
        const int64_t spanBytes = SS_ALIGN_UP(spanElems * es, SS_ALIGN);
        const int64_t outBytes = SS_ALIGN_UP(TL * es, SS_ALIGN);
        int64_t offBytes = SS_ALIGN_UP(TL * 4, SS_ALIGN);
        if (offBytes < SS_ALIGN) {
            offBytes = SS_ALIGN;
        }

        TQue<QuePosition::VECIN, 2> inQ;
        TQue<QuePosition::VECOUT, 2> outQ;
        TBuf<TPosition::VECCALC> offBuf;
        pipe.InitBuffer(inQ, 2, static_cast<uint32_t>(spanBytes));
        pipe.InitBuffer(outQ, 2, static_cast<uint32_t>(outBytes));
        pipe.InitBuffer(offBuf, static_cast<uint32_t>(offBytes));

        LocalTensor<int32_t> offI = offBuf.Get<int32_t>();
        SsBuildRamp(offI, TL, as * es);
        LocalTensor<uint32_t> offU = offI.ReinterpretCast<uint32_t>();

        for (int64_t u = u0; u < u1; ++u) {
            const int64_t r = u / t.chunksPerRow;
            const int64_t c = u - r * t.chunksPerRow;
            const int64_t j0 = c * TL;
            int64_t ln = t.innerLen - j0;
            if (ln > TL) {
                ln = TL;
            }
            const int64_t rowBase = t.baseOffset + SsOd(r, t);
            const int64_t srcOff = rowBase + j0 * as;
            const int64_t inSpan = (ln - 1) * as + 1;

            auto inL = inQ.AllocTensor<T>();
            DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(inSpan * es), 0, 0,
                                 0};
            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            DataCopyPad(inL, xG[srcOff], cp, pp);
            inQ.EnQue(inL);
            inL = inQ.DeQue<T>();

            auto outL = outQ.AllocTensor<T>();
            const int64_t lnv = SS_ALIGN_DN(ln, 8);
            if (lnv > 0) {
                Gather(outL, inL, offU, static_cast<uint32_t>(0), static_cast<uint32_t>(lnv));
            }
            for (int64_t k = lnv; k < ln; ++k) {
                outL.SetValue(k, inL.GetValue(k * as));
            }
            PipeBarrier<PIPE_ALL>();
            outQ.EnQue(outL);
            outL = outQ.DeQue<T>();

            DataCopyExtParams co{static_cast<uint16_t>(1), static_cast<uint32_t>(ln * es), 0, 0, 0};
            DataCopyPad(yG[r * t.innerLen + j0], outL, co);
            inQ.FreeTensor(inL);
            outQ.FreeTensor(outL);
        }
    }
}

/* ------------------------------------------------------------------ tiling */

}  // namespace stridedslice_ns

StridedSliceTiling calc_strided_slice_tiling(int64_t baseOffset, int64_t outNumel, int64_t rank,
                                             const int64_t *size, const int64_t *step,
                                             int64_t elemBytes)
{
    using namespace stridedslice_ns;
    StridedSliceTiling t;
    t.outNumel = outNumel;
    t.baseOffset = baseOffset;
    if (outNumel <= 0) {
        return t;
    }

    int64_t sz[8] = {0};
    int64_t st[8] = {0};
    int64_t R = rank;
    if (R > 8) {
        R = 8;
    }
    for (int64_t i = 0; i < R; ++i) {
        sz[i] = size[i];
        st[i] = step[i];
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (int64_t d = R - 1; d >= 1; --d) {
            if (sz[d] > 0 && st[d - 1] == sz[d] * st[d]) {
                sz[d - 1] = sz[d - 1] * sz[d];
                st[d - 1] = st[d];
                for (int64_t k = d; k < R - 1; ++k) {
                    sz[k] = sz[k + 1];
                    st[k] = st[k + 1];
                }
                --R;
                changed = true;
                break;
            }
        }
    }
    if (R == 0) {
        R = 1;
        sz[0] = 1;
        st[0] = 1;
    }
    const int64_t innerLen = sz[R - 1];
    const int64_t innerStep = st[R - 1];
    int64_t rowCount = 1;
    for (int64_t d = 0; d < R - 1; ++d) {
        rowCount *= sz[d];
    }

    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = plat->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    /* path 2/6 share the UB with the offset table; the DMA paths own it except for one staging pair.
     * Keep a reserve for the framework bookkeeping but otherwise use the buffer: a larger tile means
     * fewer and bigger descriptors, and the descriptor latency dominates the transfer time. */
    int64_t ubBudget = static_cast<int64_t>(ubSize) - 64 * 1024;
    if (ubBudget < 8 * 1024) {
        ubBudget = 8 * 1024;
    }
    int64_t ubDmaBudget = static_cast<int64_t>(ubSize) - 32 * 1024;
    if (ubDmaBudget < 8 * 1024) {
        ubDmaBudget = 8 * 1024;
    }

    t.innerLen = innerLen;
    t.innerStep = innerStep;
    t.rowCount = rowCount;

    /* negative inner steps and genuinely unsupported element widths go straight to the scalar path.
     * 8 byte elements keep going through the path chain (path 6 may handle them, path 1 handles the
     * contiguous row case, and the elemBytes > 4 branch at the end catches whatever is left). */
    const bool scalarOnly = (innerStep < 0) || (innerStep > 1 && elemBytes > 8);

    bool planned = false;
    if (SS_FORCE_SCALAR || scalarOnly) {
        t.path = 3;
        t.nd = R - 1;
        for (int64_t d = 0; d < R - 1; ++d) {
            t.sz[d] = sz[d];
            t.st[d] = st[d];
        }
        t.numUnits = rowCount;
        t.unitLen = 1;
        planned = true;
    } else if (SS_ENABLE_DMA_PATHS && innerStep == 1 && R == 1) {
        /* contiguous copy; in and out staging buffers both count */
        int64_t chunk = ubDmaBudget / 4 / elemBytes;
        if (chunk < 1) {
            chunk = 1;
        }
        if (chunk > 65535) {
            chunk = 65535;
        }
        t.path = 0;
        t.unitLen = chunk;
        t.ubGap = 0;
        t.numUnits = (outNumel + chunk - 1) / chunk;
        planned = true;
    } else if (SS_ENABLE_DMA_PATHS && innerStep == 1 && R >= 2) {
        /* u whole contiguous rows per multi block DMA */
        const int64_t rowBytes = innerLen * elemBytes;
        const int64_t padRowBytes = SS_ALIGN_UP(rowBytes, SS_ALIGN);
        const int64_t nR = sz[R - 2];
        const int64_t rowGap = st[R - 2];
        int64_t uMax = ubDmaBudget / 4 / padRowBytes;
        if (uMax > 8192) {
            uMax = 8192;
        }
        if (uMax >= 1 && nR >= 1 && (rowGap >= innerLen || nR == 1)) {
            int64_t groups = 1;
            for (int64_t d = 0; d < R - 2; ++d) {
                groups *= sz[d];
            }
            int64_t u = (uMax < nR) ? uMax : nR;
            if (u < 1) {
                u = 1;
            }
            t.path = 1;
            t.unitLen = u;
            t.nRowsPerGroup = nR;
            t.rowGap = rowGap;
            t.ubGap = 0;
            t.chunksPerGroup = (nR + u - 1) / u;
            t.nd = R - 2;
            for (int64_t d = 0; d < R - 2; ++d) {
                t.sz[d] = sz[d];
                t.st[d] = st[d];
            }
            t.numUnits = groups * t.chunksPerGroup;
            planned = true;
        }
    }

    if (!planned && SS_ENABLE_MULTIROW && SS_ENABLE_DMA_PATHS && R >= 2 && innerStep > 1 &&
        innerStep <= 65535 && elemBytes <= 4) {
        /* batched strided rows; compact when the row span fills its aligned pitch exactly, otherwise
         * a per row table block sized for the output row pitch */
        const int64_t rowSpan = (innerLen - 1) * innerStep + 1;
        const int64_t rowSpanBytes = rowSpan * elemBytes;
        const int64_t padRowSpanBytes = SS_ALIGN_UP(rowSpanBytes, SS_ALIGN);
        const int64_t rowOutBytes = innerLen * elemBytes;
        const int64_t padOutRowBytes = SS_ALIGN_UP(rowOutBytes, SS_ALIGN);
        const int64_t nR = sz[R - 2];
        const int64_t rowGap = st[R - 2];
        const bool compact = (padRowSpanBytes == innerLen * innerStep * elemBytes);
        const int64_t tabEnt = padOutRowBytes / elemBytes; /* int32 table entries per output row */
        const bool padded = (!compact) && SS_ENABLE_PADDED_TABLE && (tabEnt >= innerLen);
        if ((compact || padded) && rowGap >= rowSpan && nR >= 1) {
            int64_t rnMax = 0;
            const int64_t perRowTbl = compact ? innerLen : tabEnt;
            if (compact) {
                const int64_t unitCost = 2 * padRowSpanBytes + innerLen * (2 * elemBytes + 4);
                rnMax = ubBudget / unitCost;
            } else {
                const int64_t unitCost = 2 * padRowSpanBytes + 2 * padOutRowBytes + tabEnt * 4;
                const int64_t slack =
                    SS_PADDED_SLACK(tabEnt, innerStep, elemBytes, padRowSpanBytes);
                const int64_t room = ubBudget - 2 * slack;
                rnMax = (room > unitCost) ? (room / unitCost) : 0;
            }
            if (rnMax > nR) {
                rnMax = nR;
            }
            const int64_t byGather = (perRowTbl > 0) ? (SS_MAX_GATHER / perRowTbl) : 65535;
            if (rnMax > byGather) {
                rnMax = byGather;
            }
            if (rnMax > 65535) {
                rnMax = 65535;
            }
            if (rnMax >= 1) {
                int64_t groups = 1;
                for (int64_t d = 0; d < R - 2; ++d) {
                    groups *= sz[d];
                }
                t.path = 5;
                t.unitLen = rnMax;
                t.nRowsPerGroup = nR;
                t.rowGap = rowGap;
                t.ubGap = compact ? 0 : tabEnt;
                t.chunksPerGroup = (nR + rnMax - 1) / rnMax;
                t.nd = R - 2;
                for (int64_t d = 0; d < R - 2; ++d) {
                    t.sz[d] = sz[d];
                    t.st[d] = st[d];
                }
                t.numUnits = groups * t.chunksPerGroup;
                planned = true;
            }
        }
    }

    if (!planned && SS_ENABLE_I64GATHER && SS_ENABLE_DMA_PATHS && R >= 2 && elemBytes == 8 &&
        innerStep > 1 && innerStep <= 65535) {
        /* int64 with a strided inner dim: gather the two 4 byte halves through an int32 view.
         * tl solves 2*((tl-1)*as+1)*8 + 2*tl*8 + 2*tl*4 <= ubBudget, then the exact footprint is
         * verified (and halved if the alignment terms push it over). */
        const int64_t as = innerStep;
        const int64_t cost = 2 * as * 8 + 2 * 8 + 8;
        int64_t tl = ubBudget / cost;
        if (tl > innerLen) {
            tl = innerLen;
        }
        if (tl >= 64) {
            tl = SS_ALIGN_DN(tl, 64);
        }
        if (tl < 1) {
            tl = 1;
        }
        while (tl > 1) {
            const int64_t inB = SS_ALIGN_UP(((tl - 1) * as + 1) * 8, SS_ALIGN);
            const int64_t outB = SS_ALIGN_UP(tl * 8, SS_ALIGN);
            const int64_t offB = SS_ALIGN_UP(2 * tl * 4, SS_ALIGN);
            if (2 * inB + 2 * outB + offB <= ubBudget) {
                break;
            }
            tl = tl / 2;
        }
        const int64_t inB = SS_ALIGN_UP(((tl - 1) * as + 1) * 8, SS_ALIGN);
        const int64_t outB = SS_ALIGN_UP(tl * 8, SS_ALIGN);
        const int64_t offB = SS_ALIGN_UP(2 * tl * 4, SS_ALIGN);
        if (2 * inB + 2 * outB + offB <= ubBudget) {
            t.path = 6;
            t.unitLen = tl;
            t.nd = R - 1;
            for (int64_t d = 0; d < R - 1; ++d) {
                t.sz[d] = sz[d];
                t.st[d] = st[d];
            }
            t.chunksPerRow = (innerLen + tl - 1) / tl;
            t.numUnits = rowCount * t.chunksPerRow;
            planned = true;
        }
    }

    if (!planned && SS_ENABLE_GRANULE && SS_ENABLE_DMA_PATHS && rowCount == 1 &&
        innerStep * elemBytes > SS_ALIGN && innerStep <= 65535) {
        const int64_t padRowBytes = SS_ALIGN_UP(elemBytes, SS_ALIGN);
        int64_t uMax = ubDmaBudget / 4 / padRowBytes;
        if (uMax > 65535) {
            uMax = 65535;
        }
        if (uMax >= 1) {
            int64_t u = (uMax < innerLen) ? uMax : innerLen;
            if (u < 1) {
                u = 1;
            }
            t.path = 1;
            t.unitLen = u;
            t.nRowsPerGroup = innerLen;
            t.rowGap = innerStep;
            t.ubGap = 0;
            t.chunksPerGroup = (innerLen + u - 1) / u;
            t.nd = 0;
            t.innerLen = 1;
            t.numUnits = t.chunksPerGroup;
            planned = true;
        }
    }

    if (!planned && elemBytes > 4) {
        /* 8 byte elements that no DMA form above could take: element wise scalar */
        t.path = 3;
        t.nd = R - 1;
        for (int64_t d = 0; d < R - 1; ++d) {
            t.sz[d] = sz[d];
            t.st[d] = st[d];
        }
        t.numUnits = rowCount;
        t.unitLen = 1;
        planned = true;
    }

    if (!planned) {
        /* path 2: one row tile per work item, single block DMAs + Gather.
         * tl solves 2*((tl-1)*as+1)*es + 2*tl*es + 4*tl <= ubBudget; the exact footprint is then
         * verified (and halved if the alignment terms push it over). */
        const int64_t as = innerStep;
        const int64_t cost = 2 * as * elemBytes + 2 * elemBytes + 4;
        int64_t tl = ubBudget / cost;
        if (tl > innerLen) {
            tl = innerLen;
        }
        if (tl >= 64) {
            tl = SS_ALIGN_DN(tl, 64);
        }
        if (tl < 1) {
            tl = 1;
        }
        while (tl > 1) {
            const int64_t spanElems = (tl - 1) * as + 1;
            const int64_t total = 2 * SS_ALIGN_UP(spanElems * elemBytes, SS_ALIGN) +
                                  2 * SS_ALIGN_UP(tl * elemBytes, SS_ALIGN) + SS_ALIGN_UP(tl * 4, SS_ALIGN);
            if (total <= ubBudget) {
                break;
            }
            tl = tl / 2;
        }
        if (tl < 1) {
            tl = 1;
        }
        t.path = 2;
        t.unitLen = tl;
        t.nd = R - 1;
        for (int64_t d = 0; d < R - 1; ++d) {
            t.sz[d] = sz[d];
            t.st[d] = st[d];
        }
        t.chunksPerRow = (innerLen + tl - 1) / tl;
        t.numUnits = rowCount * t.chunksPerRow;
    }

    if (t.numUnits < 1) {
        t.numUnits = 1;
    }
    t.numBlocks = (coreNum < t.numUnits) ? coreNum : t.numUnits;
    if (t.numBlocks < 1) {
        t.numBlocks = 1;
    }
    return t;
}

/* ------------------------------------------------------------------ launch */

extern "C" {

void launch_strided_slice_i8(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream)
{
    stridedslice_ns::strided_slice_kernel<int8_t><<<static_cast<uint32_t>(t.numBlocks), nullptr, stream>>>(x, y,
                                                                                                           t);
}

void launch_strided_slice_u8(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream)
{
    stridedslice_ns::strided_slice_kernel<uint8_t><<<static_cast<uint32_t>(t.numBlocks), nullptr, stream>>>(x, y,
                                                                                                            t);
}

void launch_strided_slice_i32(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream)
{
    stridedslice_ns::strided_slice_kernel<int32_t><<<static_cast<uint32_t>(t.numBlocks), nullptr, stream>>>(x, y,
                                                                                                            t);
}

void launch_strided_slice_i64(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream)
{
    stridedslice_ns::strided_slice_kernel<int64_t><<<static_cast<uint32_t>(t.numBlocks), nullptr, stream>>>(x, y,
                                                                                                            t);
}

void launch_strided_slice_f16(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream)
{
    stridedslice_ns::strided_slice_kernel<half><<<static_cast<uint32_t>(t.numBlocks), nullptr, stream>>>(x, y, t);
}

void launch_strided_slice_bf16(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream)
{
    stridedslice_ns::strided_slice_kernel<bfloat16_t><<<static_cast<uint32_t>(t.numBlocks), nullptr, stream>>>(x,
                                                                                                               y,
                                                                                                               t);
}

void launch_strided_slice_f32(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream)
{
    stridedslice_ns::strided_slice_kernel<float><<<static_cast<uint32_t>(t.numBlocks), nullptr, stream>>>(x, y, t);
}

}  // extern "C"
