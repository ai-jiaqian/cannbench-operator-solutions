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
 * \file adaptive_avg_pool_3d_kernel.cpp
 * \brief AdaptiveAvgPool3D direct-launch kernel (Ascend C, dav-2201).
 *
 *   y[n,c,od,oh,ow] = mean of x[n,c,d,h,w] over
 *       d in [floor(od*D/OD), ceil((od+1)*D/OD))
 *       h in [floor(oh*H/OH), ceil((oh+1)*H/OH))
 *       w in [floor(ow*W/OW), ceil((ow+1)*W/OW))
 *
 * A work item ("task") is one (n, c, od) triple: one output slab y[n,c,od,:,:]
 * of OH*OW values whose input region (d in [d0,d1), all h, all w) is a
 * contiguous run of nd*H*W elements streamed exactly once.
 *
 * ===========================================================================
 * WHY THERE IS NO PADDED ROW LAYOUT
 * ---------------------------------------------------------------------------
 * Re-pitching every row of a loaded block (so that each row starts on a 32B
 * boundary) needs one DataCopyPad block per row, and a multi-block descriptor
 * costs a fixed ~20-75 cycles per block on top of the transfer.  For narrow
 * rows (prime W, 16 bit data) that per-block cost dominated every kernel
 * measured here - cases 7/8/10/11/15/18 spent 80-97% of their time there -
 * while replacing it with one contiguous burst made case 3 three times faster.
 *
 * So rows are never re-pitched.  A group's rows are loaded as ONE contiguous
 * run into a packed staging buffer (destination packed, so one descriptor),
 * accumulated over d into a packed fp32 row buffer F (pitch W), and the two
 * spatial reductions are byte-offset Gathers, which address any byte offset
 * and therefore do not care about the row pitch at all.
 *
 * ===========================================================================
 * ALIGNMENT AND GATHER RULES (violating any one is a wrong result or a fatal
 * device VEC exception; all of them were observed here):
 *  * every LocalTensor offset used as a vector operand must be a multiple of
 *    8 fp32 lanes (32 bytes) and the operand length a multiple of
 *    32/sizeof(element) elements.  No tail is ever "topped up" with a
 *    Duplicate on a misaligned address; the shared zero slots are placed at
 *    8-lane aligned indices for exactly this reason, and OWS = alignUp(OW, 8)
 *    keeps every window-sums row start vector-aligned.
 *  * Gather(dst, src, offsets, srcBaseAddr, count) must be called with
 *    srcBaseAddr == 0.  A non-zero value was measured to give wrong results
 *    even though the matching offsets were in range (cases 1/3/5/6 with a
 *    base of t*4 failed while every base-0 gather was correct).  A shift along
 *    the gather source is therefore expressed as an extra offset-table block
 *    per term, never as a base: block t is read as offsets[t*cnt] with base 0.
 *  * every offset in a table is a BYTE offset that must land inside the source
 *    tensor for every lane that will actually be read; a gather cannot be
 *    bounds checked from the kernel, so a shared zero slot lives inside F and
 *    inside S and every "don't care" lane points at one of them.
 *  * a table may only be written where it has been allocated: the optional
 *    extra block is written only when it is part of the table layout.
 *  * the staging slices used to batch several d-copies are pitched by a whole
 *    number of 32B blocks, so the Cast/Add operands of every slice stay
 *    vector-aligned.
 *  * a bfloat16_t / half scalar -> float cast is not supported by the
 *    compiler, so scalar tails are read out of a small fp32 Cast instead.
 * ===========================================================================
 *
 * ---------------------------------------------------------------------------
 * Global pooling (OH == OW == 1): one flat pass
 * ---------------------------------------------------------------------------
 * The task region is one contiguous run consumed as a flat vector: cut into
 * passes of `chunk` elements, DMA'd in large sub-blocks into the packed staging
 * buffer and accumulated elementwise into F.  Lane j of F holds the sum of
 * every element whose pass offset is j, so F is a bag of partial sums and one
 * balanced halving tree plus 8 scalar reads finish it.  `chunk` is the largest
 * flat run the UB can hold (a small chunk costs one barrier round trip per
 * pass and dominated the two global cases), but never wider than the widest run
 * that can actually occur, so the closing fold stays short.
 *
 * ---------------------------------------------------------------------------
 * Group decomposition
 * ---------------------------------------------------------------------------
 * A group of consecutive output rows [ohCur, ohEnd) reads input rows
 * [hBase, hBase+rows), hBase = floor(ohCur*H/OH); those row ranges are
 * disjoint between groups so every input element is read once.  rows is
 * bounded by capRows (a UB budget) and the output lane count by
 * AAP_MAX_GROUP_NL.  Groups are the outer loop and tasks the inner one, so the
 * offset tables are built once per core instead of once per task.
 *
 * ---------------------------------------------------------------------------
 * Stage 1: d-reduction (packed, several planes in flight)
 * ---------------------------------------------------------------------------
 * A group's rows are loaded once per d in [d0,d1) and added elementwise into
 * F[0..rows*W).  The loop dominates the small shapes and it is not the
 * transfer: every d used to need its own barrier pair and a barrier costs
 * ~100 cycles, so case 15 spent 3.2k of its 6.9k cycles waiting.  KD planes are
 * therefore copied into KD staging slices (pitched by a 32B multiple) ahead of
 * one barrier and then accumulated with KD vector ops, so the serialising
 * d-copies number nd/KD instead of nd.  KD comes from a fixed UB share, so
 * shapes whose single slab is already large keep KD = 1 and are unaffected.
 *
 * ---------------------------------------------------------------------------
 * Stage 2: W window reduction (before the H reduction!)
 * ---------------------------------------------------------------------------
 * Window widths are exactly {qnW, qnW+1}, qnW = ceil(W/OW).  (With
 * W = OW*q + r, nw(j) = q + ceil(frac(j) + r/OW) - 1, and the last bracket is
 * 1 for every j when r == 0, so nw = q there and q or q+1 otherwise.)
 *      Sw[h][ow] = sum_{t<qnW} F[h][w0(ow)+t] + [nw(ow) > qnW]*F[h][w0(ow)+qnW]
 * The first sum has the same term count on every lane, so one offset-table
 * block per t covers all qnW terms with no masks; the optional second term uses
 * a block whose "no extra element" lanes point at a shared zero slot, and it is
 * skipped entirely when no window needs it (the real test is max width > qnW -
 * "W % OW != 0" is only sufficient, e.g. 31 = 5*6+1 has every window of width
 * 7 = qnW).  The table is
 *      offW[t][i][j] = ((i*W) + w0(j) + t)*4                (base 0)
 * so block t>0 is block t=0 plus 4*t on every lane: only block 0 and the extra
 * block are materialised lane by lane, the rest come from one Adds each -
 * building all qnW+1 blocks scalar-wise cost more than the DMA on the shapes
 * with few tasks per core (case 12 lost 50us to it).
 * A gather costs ~100 cycles of setup no matter how few lanes it moves, so when
 * the table has a single chunk (CH == 1) its t-blocks are contiguous with the
 * chunk stride and up to BATCH of them are fetched by ONE gather and folded
 * block-wise in place; otherwise (multi-chunk tables) the per-t loop is used.
 *
 * ---------------------------------------------------------------------------
 * Stage 3: H window reduction
 * ---------------------------------------------------------------------------
 * Window heights are exactly {qnH, qnH+1}, qnH = ceil(H/OH), so
 *      R[oh][ow] = sum_{t<qnH} S[rowRel(oh)+t][ow] + [nh(oh) > qnH]*S[rowRel(oh)+qnH][ow]
 * The table is [t][lane] with a UNIFORM block stride of gNlPad, so all qnH(+1)
 * terms come out of ONE gather whose destination uses the table's own layout,
 * and the terms then collapse with qnH in-place block adds over gNlPad lanes.
 * Issuing qnH separate gathers instead costs ~100 cycles of setup each, which
 * is a sixth of the per-task budget on the small-output shapes.  The second
 * term is dropped when every window height is <= qnH (case 4 has H=16 < OH=32,
 * so each output row takes exactly one input row and a gather was being
 * wasted).  The group result is compact, so it is stored with ONE descriptor.
 *
 * Doing W before H is what makes the packed layout possible: the W stage is a
 * gather (pitch agnostic), its result S is produced in whatever pitch we like,
 * and the H stage reads S by byte offset - so no reduction ever needs a
 * re-pitched row.  The op count also drops: a doubling fold inside every h
 * window costs sum(nh-1) ~ H-OH small vector ops, while the gather form costs
 * 2*(qnH+1) ~ 2*maxNh ops.
 *
 * ---------------------------------------------------------------------------
 * Stage 4: scale and store
 * ---------------------------------------------------------------------------
 * Every window is divided by its own volume (nd*nh*nw): 1/(nh*nw) comes from a
 * per-group lane table and 1/nd is applied per task with one Muls, so the
 * groups are independent and no cross-group normalisation state is kept.
 *
 * ---------------------------------------------------------------------------
 * Accuracy
 * ---------------------------------------------------------------------------
 * All accumulation is fp32 and hierarchical.  NaN / +-inf inputs propagate
 * like the reference: the mean of an all-inf window is inf and the mean of a
 * window containing NaN is NaN, because the final scaling is a plain multiply
 * on the raw accumulation and nothing clamps a value.
 */

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

using namespace AscendC;

namespace {

/*! \brief start index of window `o` when `n` inputs are split into `out` windows */
__aicore__ inline int64_t aap_lo(int64_t o, int64_t n, int64_t out)
{
    return (o * n) / out;
}

/*! \brief end index (exclusive) of window `o` when `n` inputs are split into `out` windows */
__aicore__ inline int64_t aap_hi(int64_t o, int64_t n, int64_t out)
{
    return ((o + 1) * n + out - 1) / out;
}

/*! \brief upper bound on the number of output lanes batched into one group */
constexpr int64_t AAP_MAX_GROUP_NL = 1024;

/*! \brief largest W-reduction chunk, in lanes */
constexpr int64_t AAP_W_CHUNK_NL = 1024;

/*! \brief largest number of W terms fetched by one gather */
constexpr int64_t AAP_W_BATCH = 4;

/*! \brief end row of the group starting at `ohCur` and its row count */
__aicore__ inline int64_t aap_group_end(int64_t H, int64_t OH, int64_t OW, int64_t ohCur,
                                        int64_t capRows, int64_t* rowsOut)
{
    const int64_t hBase = aap_lo(ohCur, H, OH);
    int64_t ohEnd = ohCur;
    int64_t rows = 0;
    while (ohEnd < OH) {
        const int64_t r2 = aap_hi(ohEnd, H, OH) - hBase;
        const int64_t cnt = (((ohEnd - ohCur + 1) * OW + 7) / 8) * 8;
        if (ohEnd > ohCur && (r2 > capRows || cnt > AAP_MAX_GROUP_NL)) {
            break;
        }
        rows = r2;
        ++ohEnd;
    }
    *rowsOut = rows;
    return ohEnd;
}

/*! \brief scan the whole group decomposition for a given row budget */
__aicore__ inline void aap_group_scan(int64_t H, int64_t OH, int64_t OW, int64_t capRows,
                                      int64_t* maxRows, int64_t* maxNlPad)
{
    int64_t mr = 1;
    int64_t mn = 8;
    int64_t ohCur = 0;
    while (ohCur < OH) {
        int64_t rows = 0;
        const int64_t ohEnd = aap_group_end(H, OH, OW, ohCur, capRows, &rows);
        const int64_t gNlPad = ((((ohEnd - ohCur) * OW) + 7) / 8) * 8;
        if (rows > mr) {
            mr = rows;
        }
        if (gNlPad > mn) {
            mn = gNlPad;
        }
        ohCur = ohEnd;
    }
    *maxRows = mr;
    *maxNlPad = mn;
}

}  // namespace

template <typename T>
__global__ __aicore__ void aap3d_kernel(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D,
                                        int64_t H, int64_t W, int64_t OD, int64_t OH, int64_t OW,
                                        int64_t tasksPerCore, int64_t ubBytes)
{
    constexpr int64_t ES = static_cast<int64_t>(sizeof(T));
    constexpr bool IS16 = (sizeof(T) < 4u);
    /* elements spanning exactly one 32B vector block */
    constexpr int64_t AS = 32 / ES;

    const int64_t totalTasks = N * C * OD;
    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    int64_t t0 = blk * tasksPerCore;
    int64_t t1 = t0 + tasksPerCore;
    if (t1 > totalTasks) {
        t1 = totalTasks;
    }
    if (t0 >= t1) {
        return;
    }

    const int64_t qnW = (W + OW - 1) / OW;
    const int64_t qnH = (H + OH - 1) / OH;
    const int64_t OWS = ((OW + 7) / 8) * 8;
    const bool globalPool = (OH == 1 && OW == 1);
    /* first row of a task and the row pitch are 32B aligned -> one burst is legal */
    const bool rowAligned = (((W * ES) % 32) == 0);

    /* Every window is of width qnW or qnW+1 and of height qnH or qnH+1.  The
       "+1" term is only needed when at least one window really is wider/taller,
       which is what these two tests answer; "W % OW != 0" is merely sufficient
       and would waste a gather and an add on shapes like 31 = 5*6+1 (all
       windows 7 = qnW) or 16/32 (all windows 1 = qnW). */
    int64_t maxNw = 0;
    for (int64_t j = 0; j < OW; ++j) {
        const int64_t nw = aap_hi(j, W, OW) - aap_lo(j, W, OW);
        if (nw > maxNw) {
            maxNw = nw;
        }
    }
    int64_t maxNh = 0;
    for (int64_t oh = 0; oh < OH; ++oh) {
        const int64_t nh = aap_hi(oh, H, OH) - aap_lo(oh, H, OH);
        if (nh > maxNh) {
            maxNh = nh;
        }
    }
    const bool needXW = (maxNw > qnW);
    const bool needXH = (maxNh > qnH);

    /* number of W offset-table blocks and of H terms */
    const int64_t nWBlk = needXW ? (qnW + 1) : qnW;
    const int64_t nHTerm = needXH ? (qnH + 1) : qnH;

    /* ------------------------------------------------------------------
     * pass 1: group geometry + UB fit
     * ------------------------------------------------------------------ */
    int64_t maxRows = 1;
    int64_t maxNlPad = 8;
    int64_t RC = 1;
    int64_t CH = 1;
    int64_t KD = 1;
    int64_t BAT = 1;
    int64_t twLanes = 8;
    int64_t stagePitch = 1;
    int64_t capRows = H;
    if (capRows < maxNh) {
        capRows = maxNh;
    }
    int64_t capFlat = 1;
    const int64_t ubLimit = ubBytes * 3 / 4;
    const int64_t maxNd = (D + OD - 1) / OD + 1;
    /* fixed UB shares: several d-planes in flight, and several W terms per gather */
    const int64_t stageShare = ubBytes / 6;
    const int64_t twShare = ubBytes / 16;

    if (globalPool) {
        /* the widest flat run that can occur, so the closing fold stays short */
        capFlat = ((maxNd * H * W + AS - 1) / AS) * AS;
        maxRows = capRows;
        maxNlPad = 8;
        twLanes = maxRows * 8 + 24;
        for (int32_t it = 0; it < 24; ++it) {
            int64_t cp = capFlat;
            if (cp > 32768) {
                cp = 32768;
            }
            const int64_t need = (cp + 72) * 4 + (cp * ES + 64) +
                                 (IS16 ? (cp * 4 + 64) : 0) + (twLanes * 4) + 64 + 2048;
            if (need <= ubLimit) {
                break;
            }
            int64_t nr = cp * 3 / 4;
            if (nr >= capFlat) {
                break;
            }
            capFlat = (nr / AS) * AS;
            if (capFlat < AS) {
                capFlat = AS;
            }
        }
        if (capFlat > 32768) {
            capFlat = 32768;
        }
        stagePitch = 0;
    } else {
        for (int32_t it = 0; it < 24; ++it) {
            aap_group_scan(H, OH, OW, capRows, &maxRows, &maxNlPad);
            RC = AAP_W_CHUNK_NL / OWS;
            if (RC < 1) {
                RC = 1;
            }
            if (RC > maxRows) {
                RC = maxRows;
            }
            CH = (maxRows + RC - 1) / RC;
            capFlat = maxRows * W;
            /* one gather can only span several t-blocks when the table has a
               single chunk; otherwise the per-t loop is used */
            BAT = (CH == 1) ? AAP_W_BATCH : 1;
            twLanes = BAT * RC * OWS + 16;
            if ((BAT > 1) && (twLanes * 4 > twShare)) {
                BAT = 1;
                twLanes = RC * OWS + 16;
            }
            stagePitch = ((maxRows * W * ES + 31) / 32) * 32;
            KD = stageShare / stagePitch;
            if (KD < 1) {
                KD = 1;
            }
            if (KD > maxNd) {
                KD = maxNd;
            }
            if (KD > 8) {
                KD = 8;
            }
            const int64_t need =
                (capFlat + 72) * 4 + (maxRows * OWS + 16) * 4 + twLanes * 4 +
                ((qnH + 1) * maxNlPad + 8) * 4 +
                (KD * stagePitch + 64) + (IS16 ? (maxRows * W * 4 + 64) : 0) +
                nWBlk * CH * RC * OWS * 4 + ((qnH + 1) * maxNlPad) * 4 + 64 +
                (maxNlPad * 4 + 32) + (IS16 ? (maxNlPad * ES + 32) : 0) + 4096;
            if (need <= ubLimit) {
                break;
            }
            int64_t nr = capRows * 3 / 4;
            if (nr < maxNh) {
                nr = maxNh;
            }
            if (nr >= capRows) {
                break;
            }
            capRows = nr;
        }
        aap_group_scan(H, OH, OW, capRows, &maxRows, &maxNlPad);
        RC = AAP_W_CHUNK_NL / OWS;
        if (RC < 1) {
            RC = 1;
        }
        if (RC > maxRows) {
            RC = maxRows;
        }
        CH = (maxRows + RC - 1) / RC;
        capFlat = maxRows * W;
        BAT = (CH == 1) ? AAP_W_BATCH : 1;
        twLanes = BAT * RC * OWS + 16;
        if ((BAT > 1) && (twLanes * 4 > twShare)) {
            BAT = 1;
            twLanes = RC * OWS + 16;
        }
        stagePitch = ((maxRows * W * ES + 31) / 32) * 32;
        KD = stageShare / stagePitch;
        if (KD < 1) {
            KD = 1;
        }
        if (KD > maxNd) {
            KD = maxNd;
        }
        if (KD > 8) {
            KD = 8;
        }
    }
    const int64_t WBlkSz = CH * RC * OWS;

    /* ------------------------------------------------------------------
     * buffers
     * ------------------------------------------------------------------ */
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufF;
    TBuf<TPosition::VECCALC> bufS;
    TBuf<TPosition::VECCALC> bufTW;
    TBuf<TPosition::VECCALC> bufTH;
    TBuf<TPosition::VECCALC> bufStage;
    TBuf<TPosition::VECCALC> bufCast;
    TBuf<TPosition::VECCALC> bufOffW;
    TBuf<TPosition::VECCALC> bufOffH;
    TBuf<TPosition::VECCALC> bufSc;
    TBuf<TPosition::VECCALC> bufOT;

    const int64_t stageBytes = globalPool ? (capFlat * ES + 64) : (KD * stagePitch + 64);
    const int64_t thLanes = globalPool ? (maxRows * 8 + 24) : ((qnH + 1) * maxNlPad + 8);
    pipe.InitBuffer(bufF, static_cast<uint32_t>((capFlat + 72) * 4));
    pipe.InitBuffer(bufS, static_cast<uint32_t>((maxRows * OWS + 16) * 4));
    pipe.InitBuffer(bufTW, static_cast<uint32_t>(twLanes * 4));
    pipe.InitBuffer(bufTH, static_cast<uint32_t>(thLanes * 4));
    pipe.InitBuffer(bufStage, static_cast<uint32_t>(stageBytes));
    if (IS16) {
        pipe.InitBuffer(bufCast, static_cast<uint32_t>(capFlat * 4 + 64));
    }
    pipe.InitBuffer(bufOffW, static_cast<uint32_t>(globalPool ? 64 : (nWBlk * WBlkSz * 4 + 64)));
    pipe.InitBuffer(bufOffH, static_cast<uint32_t>(globalPool ? 64 : (((qnH + 1) * maxNlPad) * 4 + 64)));
    pipe.InitBuffer(bufSc, static_cast<uint32_t>(globalPool ? 64 : (maxNlPad * 4 + 32)));
    if (IS16) {
        pipe.InitBuffer(bufOT, static_cast<uint32_t>(globalPool ? 64 : (maxNlPad * ES + 32)));
    }

    auto F = bufF.Get<float>();
    auto S = bufS.Get<float>();
    auto TW = bufTW.Get<float>();
    auto TH = bufTH.Get<float>();
    auto stage = bufStage.Get<T>();
    /* int32 storage for the W tables: Adds exists for int32 and the gather
       reinterprets that same buffer as uint32 (every offset is positive and far
       below 2^31). */
    auto offWi = bufOffW.Get<int32_t>();
    auto offH = bufOffH.Get<uint32_t>();
    auto scB = bufSc.Get<float>();

    /* zero slots one past the used area of F and of S; both 8-lane aligned so
       the Duplicate and every gather address that lands on them stays aligned */
    const int64_t fZeroIdx = ((maxRows * W + 7) / 8) * 8 + 8;
    const int64_t sZeroIdx = maxRows * OWS;
    const int32_t fZeroOff = static_cast<int32_t>(fZeroIdx * 4);
    const uint32_t sZeroOff = static_cast<uint32_t>(sZeroIdx * 4);
    const int64_t chunkGlobal = (capFlat / AS) * AS;
    const int64_t stagePitchEl = globalPool ? 0 : (stagePitch / ES);

    Duplicate(F[static_cast<uint32_t>(fZeroIdx)], 0.0f, 8);
    Duplicate(S[static_cast<uint32_t>(sZeroIdx)], 0.0f, 8);
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    /* ==================================================================
     * global pooling: OH == OW == 1
     * ================================================================== */
    if (globalPool) {
        const int64_t chunk = chunkGlobal > AS ? chunkGlobal : AS;
        int64_t perCopy = (16384 / ES / AS) * AS;
        if (perCopy > chunk) {
            perCopy = chunk;
        }
        if (perCopy < AS) {
            perCopy = AS;
        }
        for (int64_t task = t0; task < t1; ++task) {
            const int64_t nc = task / OD;
            const int64_t od = task - nc * OD;
            const int64_t d0 = aap_lo(od, D, OD);
            const int64_t d1 = aap_hi(od, D, OD);
            const int64_t nd = d1 - d0;
            const int64_t xBase = (nc * D + d0) * H * W;
            const bool srcAligned = ((xBase * ES) % 32) == 0;
            const int64_t flatLen = nd * H * W;

            Duplicate(F, 0.0f, static_cast<int32_t>(chunkGlobal));
            PipeBarrier<PIPE_ALL>();
            float tailAcc = 0.0f;
            int64_t done = 0;
            while (done < flatLen) {
                int64_t n = flatLen - done;
                if (n > chunk) {
                    n = chunk;
                }
                int64_t off = 0;
                while (off < n) {
                    int64_t cnt = n - off;
                    if (cnt > perCopy) {
                        cnt = perCopy;
                    }
                    if (srcAligned && ((cnt % AS) == 0)) {
                        DataCopyParams dp{1, static_cast<uint16_t>(cnt * ES / 32), 0u, 0u};
                        DataCopy(stage[static_cast<uint32_t>(off)],
                                 xGm[static_cast<uint32_t>(xBase + done + off)], dp);
                    } else {
                        DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * ES), 0u, 0u, 0u};
                        DataCopyPadExtParams<T> pp{false, 0, 0, 0};
                        DataCopyPad(stage[static_cast<uint32_t>(off)],
                                    xGm[static_cast<uint32_t>(xBase + done + off)], cp, pp);
                    }
                    off += cnt;
                }
                PipeBarrier<PIPE_ALL>();
                const int64_t pad = (n / AS) * AS;
                if (pad > 0) {
                    if constexpr (IS16) {
                        auto cst = bufCast.Get<float>();
                        Cast(cst, stage, RoundMode::CAST_NONE, static_cast<int32_t>(pad));
                        Add(F, F, cst, static_cast<int32_t>(pad));
                    } else {
                        Add(F, F, stage, static_cast<int32_t>(pad));
                    }
                }
                if (pad < n) {
                    if constexpr (IS16) {
                        auto cst = bufCast.Get<float>();
                        Cast(cst, stage[static_cast<uint32_t>(pad)], RoundMode::CAST_NONE,
                             static_cast<int32_t>(AS));
                        PipeBarrier<PIPE_ALL>();
                        for (int64_t i = pad; i < n; ++i) {
                            tailAcc += cst.GetValue(static_cast<uint32_t>(i - pad));
                        }
                    } else {
                        for (int64_t i = pad; i < n; ++i) {
                            tailAcc += stage.GetValue(static_cast<uint32_t>(i));
                        }
                    }
                }
                PipeBarrier<PIPE_ALL>();
                done += n;
            }
            int64_t len = chunkGlobal;
            while (len > 8) {
                int64_t half = ((len / 2 + 7) / 8) * 8;
                if (half >= len) {
                    half = (len / 8) * 8;
                }
                if (half < 8) {
                    half = 8;
                }
                if (half >= len) {
                    break;
                }
                Add(F, F, F[static_cast<uint32_t>(half)], static_cast<int32_t>(len - half));
                len = half;
            }
            PipeBarrier<PIPE_ALL>();
            float ssum = 0.0f;
            for (int64_t i = 0; i < 8; ++i) {
                ssum += F.GetValue(static_cast<uint32_t>(i));
            }
            ssum += tailAcc;
            ssum *= 1.0f / static_cast<float>(nd * H * W);
            F.SetValue(0, ssum);
            PipeBarrier<PIPE_ALL>();
            if constexpr (IS16) {
                auto oT = bufOT.Get<T>();
                Cast(oT, F, RoundMode::CAST_RINT, 8);
                PipeBarrier<PIPE_ALL>();
                DataCopyExtParams co{1, static_cast<uint32_t>(ES), 0u, 0u, 0u};
                DataCopyPad(yGm[static_cast<uint32_t>(task)], oT, co);
            } else {
                DataCopyExtParams co{1, static_cast<uint32_t>(ES), 0u, 0u, 0u};
                DataCopyPad(yGm[static_cast<uint32_t>(task)], F, co);
            }
            PipeBarrier<PIPE_ALL>();
        }
        return;
    }

    /* ==================================================================
     * W offset tables: one block per t, every gather uses base 0.
     * Block t > 0 is block 0 plus 4*t, so only block 0 and the optional extra
     * block need one scalar store per lane.
     * ================================================================== */
    for (int64_t c = 0; c < CH; ++c) {
        for (int64_t i = 0; i < RC; ++i) {
            const int64_t rowAbs = c * RC + i;
            const int64_t rowUse = (rowAbs < maxRows) ? rowAbs : maxRows - 1;
            for (int64_t j = 0; j < OWS; ++j) {
                const int64_t wl = (j < OW) ? aap_lo(j, W, OW) : 0;
                const uint32_t lane = static_cast<uint32_t>(c * RC * OWS + i * OWS + j);
                /* block 0: a legal address on every lane (gap lanes are never read) */
                offWi.SetValue(lane, static_cast<int32_t>((rowUse * W + wl) * 4));
                if (needXW) {
                    int32_t xo = fZeroOff;
                    if (j < OW) {
                        const int64_t nw = aap_hi(j, W, OW) - wl;
                        if (nw > qnW) {
                            xo = static_cast<int32_t>((rowUse * W + wl + qnW) * 4);
                        }
                    }
                    offWi.SetValue(static_cast<int32_t>(qnW * WBlkSz + lane), xo);
                }
            }
        }
    }
    for (int64_t t = 1; t < qnW; ++t) {
        Adds(offWi[static_cast<uint32_t>(t * WBlkSz)], offWi, static_cast<int32_t>(4 * t),
             static_cast<int32_t>(WBlkSz));
    }
    PipeBarrier<PIPE_ALL>();

    /* ==================================================================
     * general path: group outer loop, task inner loop
     * ================================================================== */
    int64_t ohCur = 0;
    while (ohCur < OH) {
        int64_t rows = 0;
        const int64_t ohEnd = aap_group_end(H, OH, OW, ohCur, capRows, &rows);
        const int64_t hBase = aap_lo(ohCur, H, OH);
        const int64_t gOh = ohEnd - ohCur;
        const int64_t gNl = gOh * OW;
        const int64_t gNlPad = ((gNl + 7) / 8) * 8;

        /* --- per-group H table + scale table (shape algebra only) --- */
        for (int64_t lo = 0; lo < gOh; ++lo) {
            const int64_t oh = ohCur + lo;
            const int64_t h0 = aap_lo(oh, H, OH);
            const int64_t nh = aap_hi(oh, H, OH) - h0;
            const int64_t rowRel = h0 - hBase;
            const float invNh = 1.0f / static_cast<float>(nh);
            for (int64_t j = 0; j < OW; ++j) {
                const int64_t w0 = aap_lo(j, W, OW);
                const int64_t nw = aap_hi(j, W, OW) - w0;
                const uint32_t lane = static_cast<uint32_t>(lo * OW + j);
                for (int64_t t = 0; t < qnH; ++t) {
                    offH.SetValue(static_cast<uint32_t>(t * gNlPad + lane),
                                  static_cast<uint32_t>(((rowRel + t) * OWS + j) * 4));
                }
                if (needXH) {
                    uint32_t xo = sZeroOff;
                    if (nh > qnH) {
                        xo = static_cast<uint32_t>(((rowRel + qnH) * OWS + j) * 4);
                    }
                    offH.SetValue(static_cast<uint32_t>(qnH * gNlPad + lane), xo);
                }
                scB.SetValue(lane, invNh / static_cast<float>(nw));
            }
        }
        for (int64_t lane = gNl; lane < gNlPad; ++lane) {
            for (int64_t t = 0; t < qnH; ++t) {
                offH.SetValue(static_cast<uint32_t>(t * gNlPad + lane), sZeroOff);
            }
            if (needXH) {
                offH.SetValue(static_cast<uint32_t>(qnH * gNlPad + lane), sZeroOff);
            }
            scB.SetValue(static_cast<uint32_t>(lane), 0.0f);
        }
        PipeBarrier<PIPE_ALL>();

        for (int64_t task = t0; task < t1; ++task) {
            const int64_t nc = task / OD;
            const int64_t od = task - nc * OD;
            const int64_t d0 = aap_lo(od, D, OD);
            const int64_t d1 = aap_hi(od, D, OD);
            const int64_t nd = d1 - d0;
            const int64_t xBase = (nc * D + d0) * H * W;
            const int64_t yBase = task * OH * OW;
            const float invNd = 1.0f / static_cast<float>(nd);
            const int64_t cntF = ((rows * W + AS - 1) / AS) * AS;
            const int64_t totalBytes = rows * W * ES;

            /* ---- stage 1: d reduction of the group's rows (packed, KD in flight) ---- */
            Duplicate(F, 0.0f, static_cast<int32_t>(cntF));
            PipeBarrier<PIPE_ALL>();
            for (int64_t b0 = d0; b0 < d1; b0 += KD) {
                int64_t be = b0 + KD;
                if (be > d1) {
                    be = d1;
                }
                for (int64_t d = b0; d < be; ++d) {
                    const int64_t srcOff = xBase + (d - d0) * H * W + hBase * W;
                    const int32_t dstBase = static_cast<int32_t>((d - b0) * stagePitchEl);
                    int64_t done = 0;
                    while (done < totalBytes) {
                        int64_t chunkB = totalBytes - done;
                        if (chunkB > 16384) {
                            chunkB = 16384;
                        }
                        if (rowAligned && ((chunkB % 32) == 0)) {
                            DataCopyParams dp{1, static_cast<uint16_t>(chunkB / 32), 0u, 0u};
                            DataCopy(stage[static_cast<uint32_t>(dstBase + done / ES)],
                                     xGm[static_cast<uint32_t>(srcOff + done / ES)], dp);
                        } else {
                            DataCopyExtParams cp{1, static_cast<uint32_t>(chunkB), 0u, 0u, 0u};
                            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
                            DataCopyPad(stage[static_cast<uint32_t>(dstBase + done / ES)],
                                        xGm[static_cast<uint32_t>(srcOff + done / ES)], cp, pp);
                        }
                        done += chunkB;
                    }
                }
                PipeBarrier<PIPE_ALL>();
                for (int64_t d = b0; d < be; ++d) {
                    auto srcSlice = stage[static_cast<uint32_t>((d - b0) * stagePitchEl)];
                    if constexpr (IS16) {
                        auto cst = bufCast.Get<float>();
                        Cast(cst, srcSlice, RoundMode::CAST_NONE, static_cast<int32_t>(cntF));
                        Add(F, F, cst, static_cast<int32_t>(cntF));
                    } else {
                        Add(F, F, srcSlice, static_cast<int32_t>(cntF));
                    }
                }
                PipeBarrier<PIPE_ALL>();
            }

            /* ---- stage 2: W window reduction (packed rows, gather driven) ---- */
            for (int64_t r0 = 0; r0 < rows; r0 += RC) {
                int64_t rr = rows - r0;
                if (rr > RC) {
                    rr = RC;
                }
                const int64_t nl = rr * OWS;
                const int64_t c = r0 / RC;
                const uint32_t dstOff = static_cast<uint32_t>(r0 * OWS);
                const uint32_t cb = static_cast<uint32_t>(c * RC * OWS);
                if ((CH == 1) && (rr == RC) && (BAT > 1)) {
                    /* the t-blocks of a single-chunk table are contiguous */
                    int32_t first = 1;
                    int64_t t = 0;
                    while (t < nWBlk) {
                        int64_t nb = nWBlk - t;
                        if (nb > BAT) {
                            nb = BAT;
                        }
                        auto offT = offWi[static_cast<uint32_t>(t * WBlkSz + cb)]
                                        .ReinterpretCast<uint32_t>();
                        Gather(TW, F, offT, 0u, static_cast<uint32_t>(nb * nl));
                        for (int64_t jj = 1; jj < nb; ++jj) {
                            Add(TW, TW, TW[static_cast<uint32_t>(jj * nl)],
                                static_cast<int32_t>(nl));
                        }
                        if (first != 0) {
                            Adds(S[dstOff], TW, 0.0f, static_cast<int32_t>(nl));
                            first = 0;
                        } else {
                            Add(S[dstOff], S[dstOff], TW, static_cast<int32_t>(nl));
                        }
                        t += nb;
                    }
                } else {
                    for (int64_t t = 0; t < qnW; ++t) {
                        auto offT = offWi[static_cast<uint32_t>(t * WBlkSz + cb)]
                                        .ReinterpretCast<uint32_t>();
                        if (t == 0) {
                            Gather(S[dstOff], F, offT, 0u, static_cast<uint32_t>(nl));
                        } else {
                            Gather(TW, F, offT, 0u, static_cast<uint32_t>(nl));
                            Add(S[dstOff], S[dstOff], TW, static_cast<int32_t>(nl));
                        }
                    }
                    if (needXW) {
                        auto offT = offWi[static_cast<uint32_t>(qnW * WBlkSz + cb)]
                                        .ReinterpretCast<uint32_t>();
                        Gather(TW, F, offT, 0u, static_cast<uint32_t>(nl));
                        Add(S[dstOff], S[dstOff], TW, static_cast<int32_t>(nl));
                    }
                }
            }
            PipeBarrier<PIPE_ALL>();

            /* ---- stage 3: H window reduction: ONE gather then block-wise fold ----
               The table is [t][lane] with block stride gNlPad and the destination
               uses the very same layout, so a single gather fills all terms. */
            Gather(TH, S, offH, 0u, static_cast<uint32_t>(nHTerm * gNlPad));
            for (int64_t t = 1; t < nHTerm; ++t) {
                Add(TH, TH, TH[static_cast<uint32_t>(t * gNlPad)], static_cast<int32_t>(gNlPad));
            }
            PipeBarrier<PIPE_ALL>();

            /* ---- stage 4: scale and store ---- */
            Muls(TH, TH, invNd, static_cast<int32_t>(gNlPad));
            Mul(TH, TH, scB, static_cast<int32_t>(gNlPad));
            PipeBarrier<PIPE_ALL>();
            if constexpr (IS16) {
                auto oT = bufOT.Get<T>();
                Cast(oT, TH, RoundMode::CAST_RINT, static_cast<int32_t>(gNlPad));
                PipeBarrier<PIPE_ALL>();
                DataCopyExtParams co{1, static_cast<uint32_t>(gNl * ES), 0u, 0u, 0u};
                DataCopyPad(yGm[static_cast<uint32_t>(yBase + ohCur * OW)], oT, co);
            } else {
                DataCopyExtParams co{1, static_cast<uint32_t>(gNl * ES), 0u, 0u, 0u};
                DataCopyPad(yGm[static_cast<uint32_t>(yBase + ohCur * OW)], TH, co);
            }
            PipeBarrier<PIPE_ALL>();
        }
        ohCur = ohEnd;
    }
}

/* ---------------------------------------------------------------------------
 * host tiling + launch wrappers
 * ------------------------------------------------------------------------- */

extern "C" void calc_aap3d_tiling(int64_t totalTasks, int64_t* outBlocks, int64_t* outTasksPerCore,
                                  int64_t* outUbBytes)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    if (ascendcPlatform != nullptr) {
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    if (ubSize == 0) {
        ubSize = 192 * 1024;
    }
    int64_t coreNum = 40;
    if (ascendcPlatform != nullptr) {
        coreNum = ascendcPlatform->GetCoreNumAiv();
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    int64_t numBlocks = coreNum;
    if (numBlocks > totalTasks) {
        numBlocks = totalTasks;
    }
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    *outBlocks = numBlocks;
    *outTasksPerCore = (totalTasks + numBlocks - 1) / numBlocks;
    *outUbBytes = static_cast<int64_t>(ubSize);
}

extern "C" void launch_aap3d_kernel_float(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D,
                                          int64_t H, int64_t W, int64_t OD, int64_t OH, int64_t OW,
                                          int64_t numBlocks, int64_t tasksPerCore, int64_t ubBytes,
                                          void* stream)
{
    aap3d_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, N, C, D, H, W, OD, OH, OW,
                                                        tasksPerCore, ubBytes);
}

extern "C" void launch_aap3d_kernel_half(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D,
                                         int64_t H, int64_t W, int64_t OD, int64_t OH, int64_t OW,
                                         int64_t numBlocks, int64_t tasksPerCore, int64_t ubBytes,
                                         void* stream)
{
    aap3d_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, N, C, D, H, W, OD, OH, OW,
                                                       tasksPerCore, ubBytes);
}

extern "C" void launch_aap3d_kernel_bf16(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D,
                                         int64_t H, int64_t W, int64_t OD, int64_t OH, int64_t OW,
                                         int64_t numBlocks, int64_t tasksPerCore, int64_t ubBytes,
                                         void* stream)
{
    aap3d_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, N, C, D, H, W, OD, OH, OW,
                                                             tasksPerCore, ubBytes);
}
