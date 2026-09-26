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
 * \file conv2d_kernel.cpp
 * \brief Conv2D device kernel + host tiling (compiled with bisheng + -xasc).
 *
 * Math:
 *   y[n,co,ho,wo] = bias[co]
 *                 + sum_{ci,kh,kw} x[n,ci,ho*sh-pt+kh*dh, wo*sw-pl+kw*dw] * filter[co,ci,kh,kw]
 *
 * Output-stationary direct convolution on the vector (SIMD) unit.  One work item is
 * (n, co-block, ho-block) and holds a CT x HBT block of fp32 accumulators in UB.
 *
 * Every accumulator / staged-input row has the same element pitch `Pr`, so the contribution of one
 * (ci,kh,kw) tap to *all* rows of the block is a single fused multiply-add:
 *
 *     Axpy(acc[co], xRowBlock, weightScalar, rl * Pr)
 *
 * Rows whose input row lies outside the image contribute zero because their staged row is
 * explicitly cleared with Duplicate.  Column border handling is folded into the DataCopyPad
 * left/right padding fields, which write [leftPad zeros][payload][rightPad zeros] starting at the
 * destination base, so the destination index of a staged element is exactly its output column.
 *
 * Column stride sw > 1 is handled with the built-in GatherMask sampling patterns: the tap needs
 * x[sw*wo + (kw*dw-pl)], so the span is loaded starting at c0 = kw*dw-pl - ((kw*dw-pl) mod sw) and
 * a 1-of-sw sampling pattern starting at the same phase compacts the span into the required
 * Wout columns, again starting at destination index 0.
 *
 * No DataCopyPad in this kernel uses blockCount != 1 and no copy uses a non-zero stride, so no
 * assumption about the stride unit (bytes vs 32B blocks) is ever needed.
 *
 * Accumulation is always fp32 and is rounded once, when the row block is written out:
 *   float  -> stored straight from the fp32 accumulators
 *   half   -> Axpy<float, half> (fp16 source, fp32 destination)
 *   bf16   -> the staged bf16 rows / weight slab are Cast to fp32 first (there is no bf16 Axpy and
 *             no legal bf16 scalar arithmetic), then Axpy<float, float>.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "conv2d_launch.h"

using namespace AscendC;

namespace {
constexpr int64_t C2D_XQ_DEPTH = 2;
constexpr int64_t C2D_WSQ_DEPTH = 1;
constexpr int64_t C2D_YQ_DEPTH = 2;
constexpr int64_t C2D_UB_SLACK = 24576;
constexpr int64_t C2D_MAX_CT = 128;
constexpr int64_t C2D_MAX_HBT = 128;
} // namespace

// ---------------------------------------------------------------------------
// host side helpers (identical arithmetic is mirrored in the device body)
// ---------------------------------------------------------------------------
static inline int64_t C2DCeilDivH(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline int64_t C2DAlignUpH(int64_t v, int64_t a) { return C2DCeilDivH(v, a) * a; }
static inline int64_t C2DGcdH(int64_t a, int64_t b)
{
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) { int64_t t = a % b; a = b; b = t; }
    return a;
}

// ---------------------------------------------------------------------------
// multi-row staging geometry
//
// Tap kw needs, for every output row of the block and every kh, the input row
// base = ho0*sh - pt + kh*dh + row*sh.  For one group of `kg` consecutive kh the needed rows
// form the contiguous span [base, base + span) with
//   span(kg, rows) = (kg-1)*dh + (rows-1)*sh + 1.
// Staging each row of that span as one DataCopyPad block (blockLen = valid columns, GM
// srcStride = the *byte* gap W - valid, so the source advances by exactly one image row)
// puts every staged row at the shared row pitch Pr.  One Axpy per (co, kh) then covers the
// whole block of output rows, and the descriptor count per work item drops from
// Cin*Kh*Kw*rows to Cin*Kw*ceil(Kh/kg).
// ---------------------------------------------------------------------------
static inline int64_t C2DSpanRows(int64_t kg, int64_t rows, int64_t dh, int64_t sh)
{
    return (kg - 1) * dh + (rows - 1) * sh + 1;
}

Conv2dTiling calc_conv2d_tiling(int64_t N, int64_t Cin, int64_t H, int64_t W,
                                int64_t Cout, int64_t Kh, int64_t Kw,
                                int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                                int64_t pt, int64_t pb, int64_t pl, int64_t ppR,
                                int64_t Hout, int64_t Wout,
                                int64_t dtypeCode)
{
    (void)N; (void)H; (void)dh; (void)pt; (void)pb; (void)ppR;

    Conv2dTiling t;
    t.CT = 1; t.HBT = 1; t.Pr = 1; t.PitW = 1; t.ciChunk = 1;
    t.numCoBlk = 1; t.numHoBlk = 1; t.totalItems = 1; t.numBlocks = 1;
    t.gsP = 0; t.gsRep = 0; t.khGroup = 1;

    const int64_t ES = (dtypeCode == CONV2D_DTYPE_FLOAT) ? 4 : 2;
    const bool isBf16 = (dtypeCode == CONV2D_DTYPE_BF16);
    const int64_t K = Kh * Kw;
    if (K <= 0 || Cin <= 0 || Cout <= 0 || Wout <= 0 || Hout <= 0 || W <= 0 || ES <= 0) {
        return t;
    }

    uint64_t ubSize = 0;
    int64_t coreNum = 1;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform != nullptr) {
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        coreNum = (int64_t)platform->GetCoreNumAiv();
    }
    if (ubSize == 0) { ubSize = 192 * 1024; }
    int64_t ubBytes = (int64_t)ubSize;
    if (coreNum <= 0) { coreNum = 1; }

    // ---- row pitch (elements) shared by the accumulators and the staged input rows ----
    int64_t Pr = C2DAlignUpH(Wout * ES, 32) / ES;
    int64_t gsP = 0;
    int64_t gsRep = 0;
    if (sw > 1) {
        const int64_t span = sw * Wout;
        gsRep = C2DCeilDivH(span * ES, 256);
        const int64_t gatheredPerRepeat = 256 / (ES * sw); // 1-of-sw sampling of one 256B window
        const int64_t minPr = gsRep * gatheredPerRepeat;
        if (minPr > Pr) { Pr = minPr; }
        if (Pr <= 0) { Pr = 1; }
        const int64_t spanAligned = C2DAlignUpH(span * ES, 32) / ES;
        gsP = spanAligned;
        const int64_t spanNeed = gsRep * 256 / ES;
        if (spanNeed > gsP) { gsP = spanNeed; }
    }
    if (Pr <= 0) { Pr = 1; }

    // ---- weight slab tiling: make one co row 32B aligned so every copy stays blockCount == 1 ----
    int64_t ciChunk = 1;
    {
        const int64_t g = C2DGcdH(K * ES, 32);
        int64_t need = (g > 0) ? (32 / g) : 1;
        if (need < 1) { need = 1; }
        ciChunk = (Cin < need) ? Cin : need;
        if (ciChunk < 1) { ciChunk = 1; }
    }
    const int64_t PitW = C2DAlignUpH(ciChunk * K * ES, 32) / ES;

    // ---- UB search over (CT, HBT) ----
    const int64_t gtBytes = (sw > 1) ? C2DAlignUpH(gsP * ES, 32) : 0;

    // UB cost of one work item layout, in bytes.
    struct Cost {
        int64_t Pr, PitW, ES, gtBytes;
        bool isBf16, isF32;
        bool strided;
    } ctx;
    ctx.Pr = Pr; ctx.PitW = PitW; ctx.ES = ES; ctx.gtBytes = gtBytes;
    ctx.isBf16 = isBf16; ctx.isF32 = (ES == 4);
    ctx.strided = (sw > 1);

    auto costOf = [&ctx, &sh, &dh, &Kh](int64_t CTv, int64_t HBTv, int64_t kgv) -> int64_t {
        const int64_t xRows = C2DSpanRows(kgv, HBTv, dh, sh);
        int64_t c = 0;
        c += CTv * HBTv * ctx.Pr * 4 + 64;              // fp32 accumulators
        c += xRows * ctx.Pr * ctx.ES + 64;              // staged input rows
        c += CTv * ctx.PitW * ctx.ES + 64;              // staged weight slab
        c += CTv * ctx.ES + 64;                         // staged bias
        c += CTv * 4 + 64;                              // bias in fp32
        if (ctx.isBf16) {
            c += CTv * ctx.PitW * 4 + 64;                 // weight slab in fp32
            c += xRows * ctx.Pr * 4 + 64;                 // staged input rows in fp32
            c += CTv * HBTv * ctx.Pr * 4 + 64;            // Kahan compensation (one per co)
            c += 2 * HBTv * ctx.Pr * 4 + 64;              // Kahan scratch rows
        }
        if (!ctx.isF32) { c += HBTv * ctx.Pr * ctx.ES + 64; }   // output staging
        if (ctx.strided) { c += 2 * HBTv * ctx.gtBytes + 64; c += HBTv * ctx.Pr * ctx.ES + 64; }
        return c;
    };

    int64_t limit = ubBytes - C2D_UB_SLACK;
    if (limit < 8192) { limit = 8192; }

    int64_t ctMax = Cout < C2D_MAX_CT ? Cout : C2D_MAX_CT;
    int64_t hbMax = Hout < C2D_MAX_HBT ? Hout : C2D_MAX_HBT;
    int64_t bestCT = 1, bestHBT = 1, bestKG = 1;
    bool found = false;
    // ---- work-item shape search driven by an explicit cost model ----
    // Per item the dominant terms are (a) one scalar read + one vector issue per (co, tap) call,
    // (b) the vector work of that call (HBT*Pr lanes = HBT*Pr/64 repeats), (c) the copy descriptors
    // of the staged input rows, weight slab and store, and (d) a fixed per-item setup.  Items are
    // spread over the cores, so the modelled time is ceil(items/cores) * itemCost.  Shrinking HBT to
    // manufacture more items (as a pure occupancy heuristic does) is therefore only worth it when it
    // actually reduces that product; the model decides instead of a rule of thumb.
    const double callCost = 45.0;
    const double repCost = 2.5;
    const double dmaCost = 60.0;
    const double fixedCost = 400.0;
    const double opFactor = isBf16 ? 6.0 : 1.0;      // compensated (Kahan) accumulation
    const double castFactor = isBf16 ? 1.2 : 1.0;    // staged rows are also cast to fp32
    // The staged input rows are re-read once per output-channel block, so the GM traffic of a work
    // item is proportional to Cin*Kw*(staged rows)*Wout and the *total* traffic is proportional to
    // the item count, i.e. to 1/(CT*HBT).  That term is what keeps CT from being pushed to 1: with
    // CT = 1 the same image rows are fetched Cout times and the kernel becomes HBM bound.
    const double bwPerCore = 11.0;                   // effective GM bytes per cycle per core
    double bestCost = 1.0e30;
    for (int64_t ctv = 1; ctv <= ctMax; ctv <<= 1) {
        for (int64_t hb = 1; hb <= hbMax; hb <<= 1) {
            int64_t kg = Kh;
            while (kg > 1 && costOf(ctv, hb, kg) > limit) { kg -= 1; }
            if (costOf(ctv, hb, kg) > limit) { continue; }
            const int64_t items = N * C2DCeilDivH(Cout, ctv) * C2DCeilDivH(Hout, hb);
            const int64_t khGroups = C2DCeilDivH(Kh, kg);
            const int64_t spanH = C2DSpanRows(kg, hb, dh, sh);
            const int64_t vcCols = (sw == 1) ? Wout : (sw * Wout);
            const double stagedPerItem = (sw == 1)
                ? (double)Cin * (double)Kw * (double)khGroups * (double)spanH * (double)vcCols * (double)ES
                : (double)Cin * (double)K * (double)hb * (double)vcCols * (double)ES;
            const double perTap = repCost * ((double)hb * (double)Pr / 64.0) * opFactor * castFactor;
            const double itemCost =
                  (double)Cin * (double)K * (double)ctv * (callCost + perTap)
                + (double)Cin * (double)Kw * (double)khGroups * dmaCost
                + (double)C2DCeilDivH(Cin, ciChunk) * (double)ctv * dmaCost
                + (double)ctv * (double)hb * dmaCost
                + fixedCost;
            const double trafficCycles = (double)items * stagedPerItem / ((double)coreNum * bwPerCore);
            const double total = trafficCycles + (double)C2DCeilDivH(items, coreNum) * itemCost;
            if (total < bestCost * 0.999) {
                bestCost = total;
                bestCT = ctv;
                bestHBT = hb;
                bestKG = kg;
                found = true;
            }
        }
    }
    if (!found) { bestCT = 1; bestHBT = 1; bestKG = 1; }

    t.CT = bestCT;
    t.HBT = bestHBT;
    t.Pr = Pr;
    t.PitW = PitW;
    t.ciChunk = ciChunk;
    t.numCoBlk = C2DCeilDivH(Cout, bestCT);
    t.numHoBlk = C2DCeilDivH(Hout, bestHBT);
    t.totalItems = N * t.numCoBlk * t.numHoBlk;
    if (t.numHoBlk < 1) { t.numHoBlk = 1; }
    if (t.numCoBlk < 1) { t.numCoBlk = 1; }
    if (t.totalItems < 1) { t.totalItems = 1; }
    t.numBlocks = (t.totalItems < coreNum) ? t.totalItems : coreNum;
    if (t.numBlocks < 1) { t.numBlocks = 1; }
    t.gsP = gsP;
    t.gsRep = gsRep;
    t.khGroup = bestKG;
    return t;
}

// ---------------------------------------------------------------------------
// device kernel
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void conv2d_kernel(
    GM_ADDR xPtr, GM_ADDR wPtr, GM_ADDR bPtr, GM_ADDR yPtr,
    int64_t N, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t Kh, int64_t Kw,
    int64_t sh, int64_t sw, int64_t dh, int64_t dw,
    int64_t pt, int64_t pl, int64_t ppR, int64_t Hout, int64_t Wout,
    int64_t CT, int64_t HBT, int64_t Pr, int64_t PitW, int64_t ciChunk,
    int64_t numCoBlk, int64_t numHoBlk, int64_t totalItems,
    int64_t gsP, int64_t gsRep, int64_t kg)
{
    constexpr int64_t ES = (int64_t)sizeof(T);
    constexpr int64_t ES2 = ES / 2;
    constexpr bool IS_HALF = std::is_same<T, half>::value;
    constexpr bool IS_BF16 = std::is_same<T, bfloat16_t>::value;
    constexpr bool IS_F32 = std::is_same<T, float>::value;

    if (totalItems <= 0 || Cin <= 0 || Cout <= 0 || Wout <= 0 || Hout <= 0) { return; }

    const int64_t K = Kh * Kw;
    const int64_t coBlkSpan = numCoBlk * numHoBlk;
    if (kg < 1) { kg = 1; }
    if (kg > Kh) { kg = Kh; }
    const int64_t spanCap = (kg - 1) * dh + (HBT - 1) * sh + 1;

    TPipe pipe;
    TBuf<TPosition::VECCALC> accB;
    TBuf<TPosition::VECCALC> bfB;
    TBuf<TPosition::VECCALC> wfB;
    TBuf<TPosition::VECCALC> xfB;
    TBuf<TPosition::VECCALC> cmpB;
    TBuf<TPosition::VECCALC> yvB;
    TBuf<TPosition::VECCALC> tvB;
    TBuf<TPosition::VECCALC> xsB;
    TQue<QuePosition::VECIN, C2D_XQ_DEPTH> xQ;
    TQue<QuePosition::VECIN, C2D_XQ_DEPTH> gsQ;
    TQue<QuePosition::VECIN, C2D_WSQ_DEPTH> wQ;
    TQue<QuePosition::VECIN, C2D_WSQ_DEPTH> bQ;
    TQue<QuePosition::VECOUT, C2D_YQ_DEPTH> yQ;

    pipe.InitBuffer(accB, (uint32_t)(CT * HBT * Pr * 4 + 64));
    pipe.InitBuffer(wQ, C2D_WSQ_DEPTH, (uint32_t)(CT * PitW * ES + 64));
    pipe.InitBuffer(bQ, C2D_WSQ_DEPTH, (uint32_t)(CT * ES + 64));
    if constexpr (!IS_F32) {
        pipe.InitBuffer(bfB, (uint32_t)(CT * 4 + 64));
        pipe.InitBuffer(yQ, C2D_YQ_DEPTH, (uint32_t)(HBT * Pr * ES + 64));
    }
    if constexpr (IS_BF16) {
        pipe.InitBuffer(wfB, (uint32_t)(CT * PitW * 4 + 64));
        pipe.InitBuffer(xfB, (uint32_t)(spanCap * Pr * 4 + 64));
        pipe.InitBuffer(cmpB, (uint32_t)(CT * HBT * Pr * 4 + 64));
        pipe.InitBuffer(yvB, (uint32_t)(HBT * Pr * 4 + 64));
        pipe.InitBuffer(tvB, (uint32_t)(HBT * Pr * 4 + 64));
    }
    if (sw == 1) {
        pipe.InitBuffer(xQ, C2D_XQ_DEPTH, (uint32_t)(spanCap * Pr * ES + 64));
    } else {
        pipe.InitBuffer(gsQ, C2D_XQ_DEPTH, (uint32_t)(HBT * gsP * ES + 64));
        pipe.InitBuffer(xsB, (uint32_t)(HBT * Pr * ES + 64));
    }

    GlobalTensor<T> xGm;
    GlobalTensor<T> wGm;
    GlobalTensor<T> bGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T *)xPtr);
    wGm.SetGlobalBuffer((__gm__ T *)wPtr);
    bGm.SetGlobalBuffer((__gm__ T *)bPtr);
    yGm.SetGlobalBuffer((__gm__ T *)yPtr);

    LocalTensor<float> accF = accB.Get<float>();
    LocalTensor<float> bfF;
    LocalTensor<float> wfF;
    LocalTensor<float> xfF;
    LocalTensor<float> cmpF;
    LocalTensor<float> yvF;
    LocalTensor<float> tvF;
    LocalTensor<T> xsRow;
    LocalTensor<uint16_t> xsRowU;
    if constexpr (!IS_F32) { bfF = bfB.Get<float>(); }
    if constexpr (IS_BF16) {
        wfF = wfB.Get<float>();
        xfF = xfB.Get<float>();
        cmpF = cmpB.Get<float>();
        yvF = yvB.Get<float>();
        tvF = tvB.Get<float>();
    }
    if (sw != 1) {
        xsRow = xsB.Get<T>();
        xsRowU = xsRow.template ReinterpretCast<uint16_t>();
    }

    const int64_t blkNum = (int64_t)GetBlockNum();

    for (int64_t it = (int64_t)GetBlockIdx(); it < totalItems; it += blkNum) {
        const int64_t n = it / coBlkSpan;
        const int64_t rem = it - n * coBlkSpan;
        const int64_t coBlk = rem / numHoBlk;
        const int64_t hoBlk = rem - coBlk * numHoBlk;
        const int64_t co0 = coBlk * CT;
        int64_t ctl = Cout - co0;
        if (ctl > CT) { ctl = CT; }
        const int64_t ho0 = hoBlk * HBT;
        int64_t rl = Hout - ho0;
        if (rl > HBT) { rl = HBT; }
        if (ctl <= 0 || rl <= 0) { continue; }
        const int64_t lanes = rl * Pr;
        const int64_t xBase = n * Cin * H * W;
        const int64_t yBase = (n * Cout) * Hout * Wout;

        // ---------------- bias -> fp32 -> accumulator init ----------------
        {
            auto bb = bQ.AllocTensor<T>();
            DataCopyExtParams bc{1, (uint32_t)(ctl * ES), 0, 0, 0};
            DataCopyPadExtParams<T> bp{false, 0, 0, 0};
            DataCopyPad(bb, bGm[co0], bc, bp);
            bQ.EnQue(bb);
            bb = bQ.DeQue<T>();
            if constexpr (!IS_F32) {
                Cast(bfF, bb, RoundMode::CAST_NONE, (int32_t)ctl);
                PipeBarrier<PIPE_ALL>();
            }
            for (int64_t co = 0; co < ctl; ++co) {
                float bv = 0.0f;
                if constexpr (IS_F32) { bv = bb.GetValue((uint32_t)co); }
                else { bv = bfF.GetValue((uint32_t)co); }
                Duplicate(accF[co * HBT * Pr], bv, (int32_t)lanes);
                if constexpr (IS_BF16) { Duplicate(cmpF[co * HBT * Pr], 0.0f, (int32_t)lanes); }
            }
            bQ.FreeTensor(bb);
        }

        // ---------------- reduction over input channels ----------------
        for (int64_t ci0 = 0; ci0 < Cin; ci0 += ciChunk) {
            int64_t cic = Cin - ci0;
            if (cic > ciChunk) { cic = ciChunk; }

            auto wb = wQ.AllocTensor<T>();
            {
                DataCopyExtParams wc{1, (uint32_t)(cic * K * ES), 0, 0, 0};
                DataCopyPadExtParams<T> wp{false, 0, 0, 0};
                for (int64_t co = 0; co < ctl; ++co) {
                    DataCopyPad(wb[co * PitW], wGm[(co0 + co) * Cin * K + ci0 * K], wc, wp);
                }
            }
            wQ.EnQue(wb);
            wb = wQ.DeQue<T>();
            if constexpr (IS_BF16) {
                Cast(wfF, wb, RoundMode::CAST_NONE, (int32_t)(ctl * PitW));
                PipeBarrier<PIPE_ALL>();
            }

            for (int64_t cii = 0; cii < cic; ++cii) {
                const int64_t ci = ci0 + cii;
                for (int64_t kh = 0; kh < Kh; ++kh) {
                    const int64_t ihBase = ho0 * sh - pt + kh * dh;
                    for (int64_t kw = 0; kw < Kw; ++kw) {
                        // ---- tap geometry: span start column, valid length and padding ----
                        const int64_t d = kw * dw - pl;
                        int64_t q = 0;
                        int64_t span = Wout;
                        int64_t pattern = 1;
                        if (sw != 1) {
                            int64_t dd = d;
                            if (dd < 0) { dd += ((0 - dd + sw - 1) / sw) * sw; }
                            q = dd % sw;
                            span = sw * Wout;
                            pattern = (sw == 2) ? (1 + q) : (3 + q);
                        }
                        const int64_t c0 = d - q;
                        const int64_t vs = (c0 > 0) ? c0 : 0;
                        int64_t ve = c0 + span;
                        if (ve > W) { ve = W; }
                        int64_t vc = ve - vs;
                        if (vc < 0) { vc = 0; }
                        const int64_t lp = vs - c0;
                        const int64_t rp = (c0 + span) - ve;

                        LocalTensor<T> xT;
                        if (sw == 1) {
                          if ((kh % kg) == 0) {
                            // One descriptor stages every input row the whole kh group of this tap
                            // needs.  The rows land at the shared pitch Pr, so one Axpy per (co,kh)
                            // covers the entire block of output rows.
                            int64_t kgr = Kh - kh;
                            if (kgr > kg) { kgr = kg; }
                            const int64_t spn = (kgr - 1) * dh + (rl - 1) * sh + 1;
                            const int64_t base = ho0 * sh - pt + kh * dh;
                            int64_t r0 = 0 - base;
                            if (r0 < 0) { r0 = 0; }
                            if (r0 > spn) { r0 = spn; }
                            int64_t r1 = H - base;
                            if (r1 < 0) { r1 = 0; }
                            if (r1 > spn) { r1 = spn; }
                            auto xb = xQ.AllocTensor<T>();
                            auto xu = xb.template ReinterpretCast<uint16_t>();
                            const int64_t rowU = Pr * ES2;
                            if (r1 > r0 && vc > 0) {
                                DataCopyExtParams rc{(uint16_t)(r1 - r0), (uint32_t)(vc * ES),
                                                     (uint32_t)((W - vc) * ES), 0, 0};
                                DataCopyPadExtParams<T> pp{true, (uint8_t)lp, (uint8_t)rp, 0};
                                DataCopyPad(xb[r0 * Pr],
                                            xGm[xBase + (ci * H + base + r0) * W + vs], rc, pp);
                            }
                            xQ.EnQue(xb);
                            xb = xQ.DeQue<T>();
                            for (int64_t row = 0; row < r0; ++row) {
                                Duplicate(xu[row * rowU], (uint16_t)0, (int32_t)rowU);
                            }
                            for (int64_t row = r1; row < spn; ++row) {
                                Duplicate(xu[row * rowU], (uint16_t)0, (int32_t)rowU);
                            }
                            if constexpr (IS_BF16) {
                                Cast(xfF, xb, RoundMode::CAST_NONE, (int32_t)(spn * Pr));
                            }
                            for (int64_t kh2 = kh; kh2 < kh + kgr; ++kh2) {
                                const int64_t kl = kh2 - kh;
                                const int64_t wIdx = cii * K + kh2 * Kw + kw;
                                if (sh == 1) {
                                    const int64_t off = kl * dh * Pr;
                                    if constexpr (IS_BF16) {
                                        for (int64_t co = 0; co < ctl; ++co) {
                                            const float wv = wfF.GetValue((uint32_t)(co * PitW + wIdx));
                                            LocalTensor<float> ac = accF[co * HBT * Pr];
                                            LocalTensor<float> cc = cmpF[co * HBT * Pr];
                                            Muls(yvF, xfF[off], wv, (int32_t)lanes);
                                            Sub(yvF, yvF, cc, (int32_t)lanes);
                                            Add(tvF, ac, yvF, (int32_t)lanes);
                                            Sub(cc, tvF, ac, (int32_t)lanes);
                                            Sub(cc, cc, yvF, (int32_t)lanes);
                                            Muls(ac, tvF, 1.0f, (int32_t)lanes);
                                        }
                                    } else if constexpr (IS_HALF) {
                                        for (int64_t co = 0; co < ctl; ++co) {
                                            const half wv = wb.GetValue((uint32_t)(co * PitW + wIdx));
                                            Axpy(accF[co * HBT * Pr], xb[off], wv, (int32_t)lanes);
                                        }
                                    } else {
                                        for (int64_t co = 0; co < ctl; ++co) {
                                            const float wv = wb.GetValue((uint32_t)(co * PitW + wIdx));
                                            Axpy(accF[co * HBT * Pr], xb[off], wv, (int32_t)lanes);
                                        }
                                    }
                                } else {
                                    for (int64_t row = 0; row < rl; ++row) {
                                        const int64_t off = (row * sh + kl * dh) * Pr;
                                        if constexpr (IS_BF16) {
                                            for (int64_t co = 0; co < ctl; ++co) {
                                                const float wv = wfF.GetValue((uint32_t)(co * PitW + wIdx));
                                                LocalTensor<float> ac = accF[co * HBT * Pr + row * Pr];
                                                LocalTensor<float> cc = cmpF[co * HBT * Pr + row * Pr];
                                                Muls(yvF, xfF[off], wv, (int32_t)Pr);
                                                Sub(yvF, yvF, cc, (int32_t)Pr);
                                                Add(tvF, ac, yvF, (int32_t)Pr);
                                                Sub(cc, tvF, ac, (int32_t)Pr);
                                                Sub(cc, cc, yvF, (int32_t)Pr);
                                                Muls(ac, tvF, 1.0f, (int32_t)Pr);
                                            }
                                        } else if constexpr (IS_HALF) {
                                            for (int64_t co = 0; co < ctl; ++co) {
                                                const half wv = wb.GetValue((uint32_t)(co * PitW + wIdx));
                                                Axpy(accF[co * HBT * Pr + row * Pr], xb[off], wv, (int32_t)Pr);
                                            }
                                        } else {
                                            for (int64_t co = 0; co < ctl; ++co) {
                                                const float wv = wb.GetValue((uint32_t)(co * PitW + wIdx));
                                                Axpy(accF[co * HBT * Pr + row * Pr], xb[off], wv, (int32_t)Pr);
                                            }
                                        }
                                    }
                                }
                            }
                            xQ.FreeTensor(xb);
                          }
                        } else {
                            const int64_t gsRowU = gsP * ES2;
                            auto gb = gsQ.AllocTensor<T>();
                            auto gu = gb.template ReinterpretCast<uint16_t>();
                            for (int64_t row = 0; row < rl; ++row) {
                                const int64_t ih = ihBase + row * sh;
                                if (ih < 0 || ih >= H || vc <= 0) { continue; }
                                DataCopyExtParams rc{1, (uint32_t)(vc * ES), 0, 0, 0};
                                DataCopyPadExtParams<T> pp{true, (uint8_t)lp, (uint8_t)rp, 0};
                                DataCopyPad(gb[row * gsP],
                                            xGm[xBase + (ci * H + ih) * W + vs], rc, pp);
                            }
                            gsQ.EnQue(gb);
                            gb = gsQ.DeQue<T>();
                            for (int64_t row = 0; row < rl; ++row) {
                                const int64_t ih = ihBase + row * sh;
                                if (ih < 0 || ih >= H || vc <= 0) {
                                    Duplicate(gu[row * gsRowU], (uint16_t)0, (int32_t)gsRowU);
                                }
                            }
                            // 1-of-sw compaction: dst[wo] = span[sw*wo + phase]
                            for (int64_t row = 0; row < rl; ++row) {
                                uint64_t rsvd = 0;
                                GatherMaskParams gmp{1, (uint16_t)gsRep, 8, 0};
                                GatherMask(xsRow[row * Pr], gb[row * gsP], (uint8_t)pattern,
                                           false, 0u, gmp, rsvd);
                            }
                            for (int64_t row = 0; row < rl; ++row) {
                                const int64_t ih = ihBase + row * sh;
                                if (ih < 0 || ih >= H || vc <= 0) {
                                    // zero through the uint16 view: bf16 scalar literals are unsafe
                                    Duplicate(xsRowU[row * Pr * ES2], (uint16_t)0, (int32_t)(Pr * ES2));
                                }
                            }
                            if constexpr (IS_BF16) {
                                Cast(xfF, xsRow, RoundMode::CAST_NONE, (int32_t)lanes);
                                for (int64_t co = 0; co < ctl; ++co) {
                                    const float wv = wfF.GetValue((uint32_t)(co * PitW + cii * K + kh * Kw + kw));
                                    LocalTensor<float> ac = accF[co * HBT * Pr];
                                    LocalTensor<float> cc = cmpF[co * HBT * Pr];
                                    Muls(yvF, xfF, wv, (int32_t)lanes);
                                    Sub(yvF, yvF, cc, (int32_t)lanes);
                                    Add(tvF, ac, yvF, (int32_t)lanes);
                                    Sub(cc, tvF, ac, (int32_t)lanes);
                                    Sub(cc, cc, yvF, (int32_t)lanes);
                                    Muls(ac, tvF, 1.0f, (int32_t)lanes);
                                }
                            } else if constexpr (IS_HALF) {
                                for (int64_t co = 0; co < ctl; ++co) {
                                    const half wv = wb.GetValue((uint32_t)(co * PitW + cii * K + kh * Kw + kw));
                                    Axpy(accF[co * HBT * Pr], xsRow, wv, (int32_t)lanes);
                                }
                            } else {
                                for (int64_t co = 0; co < ctl; ++co) {
                                    const float wv = wb.GetValue((uint32_t)(co * PitW + cii * K + kh * Kw + kw));
                                    Axpy(accF[co * HBT * Pr], xsRow, wv, (int32_t)lanes);
                                }
                            }
                            gsQ.FreeTensor(gb);
                        }
                    }
                }
            }
            wQ.FreeTensor(wb);
        }

        if constexpr (IS_BF16) {
            // fold the accumulated low-order residual back into the fp32 sum
            for (int64_t co = 0; co < ctl; ++co) {
                Add(accF[co * HBT * Pr], accF[co * HBT * Pr], cmpF[co * HBT * Pr], (int32_t)lanes);
            }
        }

        // ---------------- write back ----------------
        if constexpr (IS_F32) {
            // V (Axpy) -> MTE3 (store)
            PipeBarrier<PIPE_ALL>();
            for (int64_t co = 0; co < ctl; ++co) {
                const int64_t coG = co0 + co;
                const int64_t rowBase = yBase + coG * Hout * Wout + ho0 * Wout;
                for (int64_t row = 0; row < rl; ++row) {
                    DataCopyExtParams oc{1, (uint32_t)(Wout * ES), 0, 0, 0};
                    DataCopyPad(yGm[rowBase + row * Wout], accF[co * HBT * Pr + row * Pr], oc);
                }
            }
            // MTE3 (store) -> V (next work item's bias Duplicate writes the same accumulator)
            PipeBarrier<PIPE_ALL>();
        } else {
            for (int64_t co = 0; co < ctl; ++co) {
                const int64_t coG = co0 + co;
                const int64_t rowBase = yBase + coG * Hout * Wout + ho0 * Wout;
                auto yb = yQ.AllocTensor<T>();
                Cast(yb, accF[co * HBT * Pr], RoundMode::CAST_RINT, (int32_t)lanes);
                yQ.EnQue(yb);
                yb = yQ.DeQue<T>();
                for (int64_t row = 0; row < rl; ++row) {
                    DataCopyExtParams oc{1, (uint32_t)(Wout * ES), 0, 0, 0};
                    DataCopyPad(yGm[rowBase + row * Wout], yb[row * Pr], oc);
                }
                yQ.FreeTensor(yb);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// launch wrappers (regular C functions callable from g++)
// ---------------------------------------------------------------------------
extern "C" {

void launch_conv2d_kernel_half(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               int64_t N, int64_t Cin, int64_t H, int64_t W,
                               int64_t Cout, int64_t Kh, int64_t Kw,
                               int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                               int64_t pt, int64_t pl, int64_t ppR, int64_t Hout, int64_t Wout,
                               int64_t CT, int64_t HBT, int64_t Pr, int64_t PitW, int64_t ciChunk,
                               int64_t numCoBlk, int64_t numHoBlk, int64_t totalItems,
                               int64_t gsP, int64_t gsRep, int64_t kg, int64_t numBlocks, void *stream)
{
    conv2d_kernel<half><<<(uint32_t)numBlocks, nullptr, stream>>>(
        x, w, b, y, N, Cin, H, W, Cout, Kh, Kw, sh, sw, dh, dw, pt, pl, ppR, Hout, Wout,
        CT, HBT, Pr, PitW, ciChunk, numCoBlk, numHoBlk, totalItems, gsP, gsRep, kg);
}

void launch_conv2d_kernel_bfloat16(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                   int64_t N, int64_t Cin, int64_t H, int64_t W,
                                   int64_t Cout, int64_t Kh, int64_t Kw,
                                   int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                                   int64_t pt, int64_t pl, int64_t ppR, int64_t Hout, int64_t Wout,
                                   int64_t CT, int64_t HBT, int64_t Pr, int64_t PitW, int64_t ciChunk,
                                   int64_t numCoBlk, int64_t numHoBlk, int64_t totalItems,
                                   int64_t gsP, int64_t gsRep, int64_t kg, int64_t numBlocks, void *stream)
{
    conv2d_kernel<bfloat16_t><<<(uint32_t)numBlocks, nullptr, stream>>>(
        x, w, b, y, N, Cin, H, W, Cout, Kh, Kw, sh, sw, dh, dw, pt, pl, ppR, Hout, Wout,
        CT, HBT, Pr, PitW, ciChunk, numCoBlk, numHoBlk, totalItems, gsP, gsRep, kg);
}

void launch_conv2d_kernel_float(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                int64_t N, int64_t Cin, int64_t H, int64_t W,
                                int64_t Cout, int64_t Kh, int64_t Kw,
                                int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                                int64_t pt, int64_t pl, int64_t ppR, int64_t Hout, int64_t Wout,
                                int64_t CT, int64_t HBT, int64_t Pr, int64_t PitW, int64_t ciChunk,
                                int64_t numCoBlk, int64_t numHoBlk, int64_t totalItems,
                                int64_t gsP, int64_t gsRep, int64_t kg, int64_t numBlocks, void *stream)
{
    conv2d_kernel<float><<<(uint32_t)numBlocks, nullptr, stream>>>(
        x, w, b, y, N, Cin, H, W, Cout, Kh, Kw, sh, sw, dh, dw, pt, pl, ppR, Hout, Wout,
        CT, HBT, Pr, PitW, ciChunk, numCoBlk, numHoBlk, totalItems, gsP, gsRep, kg);
}

} // extern "C"
