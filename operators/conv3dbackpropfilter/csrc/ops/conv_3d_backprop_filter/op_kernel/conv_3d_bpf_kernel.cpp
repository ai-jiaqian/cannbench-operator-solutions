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
 * \file conv_3d_bpf_kernel.cpp
 * \brief Conv3DBackpropFilter device kernels + host tiling (compiled with bisheng + -xasc)
 *
 *   y[co, cig, kd, kh, kw] =
 *       sum_{n, od, oh, ow} grad[n, co, od, oh, ow] *
 *           x[n, ci, od*sd - pd + kd*dd, oh*sh - ph + kh*dh, ow*sw - pw + kw*dw]
 *   (out of range x positions read as 0; ci = (co / Cout_g) * Cin_g + cig)
 *
 * The operator is computed as a blocked GEMM:
 *   reduction index  = (n, od, oh, ow)
 *   output columns   = (ci, kh, kw)
 *   output rows      = co
 *
 *   for every work item (kd, coTile, colTile):
 *     acc[CT][NT][chunkLen] = 0
 *     for every plane chunk (chunkLen = RT rows of the ow plane, padded to RW):
 *       for n, od:
 *         load the grad tile for the CT rows          (reused by all NT columns)
 *         load the NT x-column windows                (reused by all CT rows)
 *         acc += grad * window
 *     reduce acc -> scalar accumulator -> y
 *
 * Out of range x positions participate in the sum as explicit zeros (the golden
 * im2col buffer is zero filled), which reproduces the reference NaN placement for
 * infinite grad values.  The three-segment multiply-add below splits the chunk rows
 * into "before the valid row range" (uses a UB-resident zero vector), "valid"
 * (uses the loaded window) and "after" so no out-of-range row ever contributes
 * anything except the explicit zero product.
 *
 * For sw > 1 a de-interleave pre-pass materialises
 *   xSplit[n][ci][s][d][h][j] = x[n][ci][d][h][j*sw + s]   (0 when j*sw + s >= W)
 * so that every column window is a plain contiguous slice and the very same
 * window machinery applies.  sd / sh are handled by the DMA source stride.
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "conv_3d_bpf_launch.h"

using namespace AscendC;

__aicore__ inline int64_t C3Min(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline int64_t C3Max(int64_t a, int64_t b) { return a > b ? a : b; }
__aicore__ inline int64_t C3CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
__aicore__ inline int64_t C3RoundUp(int64_t v, int64_t m) { return ((v + m - 1) / m) * m; }

/*!
 * \brief geometry of one (kh, kw) column window inside one plane chunk.
 *
 * rows of the window are oh in [oh0, oh0+rt); the row is valid when
 *   0 <= oh*sh + (kh*dh - ph) < H
 * and the column is valid when 0 <= q + ow < rowStride, with q the first source
 * column of the window (q may be negative or beyond the row end, both mean zero).
 */
struct C3Win {
    int64_t a0;      // number of leading window rows that must be zero
    int64_t a1;      // end of the valid window rows
    int64_t colStart;
    int64_t validCols;
    int64_t leftPad;
    int64_t rightPad;
};

__aicore__ inline C3Win C3Window(int64_t kh, int64_t kw,
                                 int64_t oh0, int64_t rt,
                                 int64_t H, int64_t Wout, int64_t RW,
                                 int64_t rowStride, int64_t subPl,
                                 int64_t sh, int64_t ph, int64_t dh,
                                 int64_t pw, int64_t dw)
{
    C3Win w;
    int64_t c0 = kw * dw - pw;
    int64_t q;
    if (subPl == 1) {
        q = c0;
    } else {
        int64_t m = c0 % subPl;
        if (m < 0) {
            m += subPl;
        }
        q = (c0 - m) / subPl;
    }
    int64_t r0 = kh * dh - ph;          // ih = oh*sh + r0
    int64_t oa = C3CeilDiv(-r0, sh);    // smallest oh with oh*sh + r0 >= 0
    oa = C3Max(oa, (int64_t)0);
    int64_t ob = (H - 1 - r0) / sh;     // largest oh with oh*sh + r0 <= H-1
    ob += 1;
    ob = C3Min(ob, H);
    int64_t vA = C3Max(oh0, oa);
    int64_t vB = C3Min(oh0 + rt, ob);
    if (vB <= vA) {
        w.a0 = 0;
        w.a1 = 0;
    } else {
        w.a0 = vA - oh0;
        w.a1 = vB - oh0;
    }
    int64_t colStart = C3Max((int64_t)0, q);
    int64_t colEnd = C3Min(rowStride, q + Wout);
    int64_t validCols = colEnd - colStart;
    if (validCols < 0) {
        validCols = 0;
    }
    w.colStart = colStart;
    w.validCols = validCols;
    w.leftPad = colStart - q;
    w.rightPad = (q + Wout) - colEnd + (RW - Wout);
    return w;
}

/* --------------------------------------------------------------------------- */
/* de-interleave pre-pass for stride_w > 1                                      */
/* --------------------------------------------------------------------------- */
template <typename T>
__global__ __aicore__ void c3bpf_split_kernel(
    GM_ADDR xPtr, GM_ADDR xsPtr,
    int64_t rows, int64_t W, int64_t jmax, int64_t sw, int64_t DH,
    int64_t numBlocks)
{
    constexpr int64_t ES = (int64_t)sizeof(T);
    const int64_t offStride = C3RoundUp(jmax, 8);

    GlobalTensor<T> xGm;
    GlobalTensor<T> xsGm;
    xGm.SetGlobalBuffer((__gm__ T *)xPtr);
    xsGm.SetGlobalBuffer((__gm__ T *)xsPtr);

    TPipe pipe;
    TQue<TPosition::VECIN, 2> inQ;
    TQue<TPosition::VECOUT, 2> outQ;
    TBuf<TPosition::VECCALC> bOff;
    pipe.InitBuffer(inQ, 2, (uint32_t)C3RoundUp(W * ES, 32));
    pipe.InitBuffer(outQ, 2, (uint32_t)C3RoundUp(jmax * ES, 32));    pipe.InitBuffer(bOff, (uint32_t)(sw * offStride * 4 + 32));

    auto offU = bOff.Get<uint32_t>();
    for (int64_t s = 0; s < sw; ++s) {
        for (int64_t j = 0; j < jmax; ++j) {
            offU.SetValue((int32_t)(s * offStride + j), (uint32_t)((j * sw + s) * ES));
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    const int64_t blk = (int64_t)GetBlockIdx();
    const int64_t r0 = rows * blk / numBlocks;
    const int64_t r1 = rows * (blk + 1) / numBlocks;

    for (int64_t row = r0; row < r1; ++row) {
        const int64_t nc = row / DH;
        const int64_t dh = row - nc * DH;
        auto inR = inQ.AllocTensor<T>();
        DataCopyExtParams cpIn{1, (uint32_t)(W * ES), 0u, 0u, 0u};
        DataCopyPadExtParams<T> ppIn{false, 0, 0, (T)0};
        DataCopyPad(inR, xGm[row * W], cpIn, ppIn);
        inQ.EnQue(inR);
        auto srcR = inQ.DeQue<T>();
        for (int64_t s = 0; s < sw; ++s) {
            const int64_t js = (W > s) ? ((W - s + sw - 1) / sw) : 0;
            auto q = outQ.AllocTensor<T>();
            // Zero fill first: the gather destination must stay 32B aligned, and the
            // tail [js, jmax) has to be zero because it stands for out-of-range columns.
            Duplicate(q, (T)0, (int32_t)jmax);
            if (js > 0) {
                Gather(q, srcR, offU[(int32_t)(s * offStride)], (uint32_t)0, (uint32_t)js);
            }
            outQ.EnQue(q);
            auto qq = outQ.DeQue<T>();
            int64_t dstOff = ((nc * sw + s) * DH + dh) * jmax;
            DataCopyExtParams cpOut{1, (uint32_t)(jmax * ES), 0u, 0u, 0u};
            DataCopyPad(xsGm[dstOff], qq, cpOut);
            outQ.FreeTensor(qq);
        }
        inQ.FreeTensor(srcR);
    }
}

/* --------------------------------------------------------------------------- */
/* main blocked kernel                                                          */
/* --------------------------------------------------------------------------- */
template <typename T>
__global__ __aicore__ void c3bpf_main_kernel(
    GM_ADDR xPtr, GM_ADDR gPtr, GM_ADDR yPtr, GM_ADDR xsPtr,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Dout, int64_t Hout, int64_t Wout,
    int64_t Kd, int64_t Kh, int64_t Kw,
    int64_t sd, int64_t sh, int64_t sw,
    int64_t pd, int64_t ph, int64_t pw,
    int64_t dd, int64_t dh, int64_t dw,
    int64_t Cin_g, int64_t Cout_g, int64_t groups,
    int64_t rowStride, int64_t subPl, int64_t RW,
    int64_t CT, int64_t NT, int64_t RT,
    int64_t ncpg, int64_t ntpc, int64_t numItems, int64_t numBlocks,
    int64_t b1x1)
{
    constexpr int64_t ES = (int64_t)sizeof(T);
    constexpr bool ISBF = std::is_same<T, bfloat16_t>::value;
    constexpr RoundMode RM_IN = RoundMode::CAST_NONE;
    constexpr RoundMode RM_OUT = ISBF ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;

    // b1x1: Kd=Kh=Kw=1 with no padding and unit striding, so the reduction index
    // (od,oh,ow) maps identically onto the x tensor.  The row axis is therefore the
    // flattened row (od*Hout + oh) of length Dout*Hout and RT is expressed in those
    // flattened rows, which lets a single DataCopyPad cover RT contiguous rows for
    // both operands (srcStride = 0).  Every term is in range, so the zero padding
    // semantics of the general path are vacuous here.
    const int64_t flatRows = b1x1 ? (Dout * Hout) : Hout;
    const int64_t gradPlane = b1x1 ? flatRows : Hout;

    const int64_t KhKw = Kh * Kw;
    const int64_t chunkLen = RT * RW;
    const int64_t nHChunks = C3CeilDiv(flatRows, RT);
    const int64_t nColTiles = Cin_g * ntpc;
    const int64_t nCoTiles = groups * ncpg;
    const int64_t tilesPerKd = nCoTiles * nColTiles;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bAcc;
    TBuf<TPosition::VECCALC> bGf;
    TBuf<TPosition::VECCALC> bXf;
    TBuf<TPosition::VECCALC> bXz;
    TBuf<TPosition::VECCALC> bRed;
    TBuf<TPosition::VECCALC> bScal;
    TBuf<TPosition::VECCALC> bOff;
    TBuf<TPosition::VECCALC> bTmp;
    TBuf<TPosition::VECCALC> bCf;
    TQue<TPosition::VECIN, 2> qG;
    TQue<TPosition::VECIN, 2> qX;
    TQue<TPosition::VECOUT, 2> qO;

    pipe.InitBuffer(bAcc, (uint32_t)(CT * NT * chunkLen * 4));
    pipe.InitBuffer(bGf, (uint32_t)(CT * chunkLen * 4));
    pipe.InitBuffer(bXf, (uint32_t)(NT * chunkLen * 4));
    pipe.InitBuffer(bXz, (uint32_t)(chunkLen * 4 + 32));
    pipe.InitBuffer(bRed, (uint32_t)(CT * NT * 8 * 4));
    pipe.InitBuffer(bScal, (uint32_t)(CT * NT * 8 * 4));
    pipe.InitBuffer(bOff, (uint32_t)(CT * NT * 4 + 32));
    pipe.InitBuffer(bTmp, 1024);
    pipe.InitBuffer(bCf, 2048);
    pipe.InitBuffer(qG, 2, (uint32_t)C3RoundUp(CT * chunkLen * ES, 32));
    pipe.InitBuffer(qX, 2, (uint32_t)C3RoundUp(NT * chunkLen * ES, 32));
    pipe.InitBuffer(qO, 2, (uint32_t)C3RoundUp(256 * ES, 32));

    auto acc = bAcc.Get<float>();
    auto gf = bGf.Get<float>();
    auto xf = bXf.Get<float>();
    auto xz = bXz.Get<float>();
    auto red = bRed.Get<float>();
    auto scl = bScal.Get<float>();
    auto offU = bOff.Get<uint32_t>();
    auto tmp = bTmp.Get<float>();
    auto cf = bCf.Get<float>();

    for (int32_t i = 0; i < (int32_t)(CT * NT); ++i) {
        offU.SetValue(i, (uint32_t)(i * 32));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    Duplicate(xz, 0.0f, (int32_t)chunkLen);

    // xPtr is unused: xsPtr aliases x when sw == 1, otherwise the de-interleaved split.
    (void)xPtr;
    GlobalTensor<T> gGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> xUse;
    gGm.SetGlobalBuffer((__gm__ T *)gPtr);
    yGm.SetGlobalBuffer((__gm__ T *)yPtr);
    xUse.SetGlobalBuffer((__gm__ T *)xsPtr);

    const int64_t blk = (int64_t)GetBlockIdx();
    const int64_t w0 = numItems * blk / numBlocks;
    const int64_t w1 = numItems * (blk + 1) / numBlocks;

    for (int64_t item = w0; item < w1; ++item) {
        const int64_t kd = item / tilesPerKd;
        const int64_t rem1 = item - kd * tilesPerKd;
        const int64_t ctIdx = rem1 / nColTiles;
        const int64_t clIdx = rem1 - ctIdx * nColTiles;
        const int64_t grp = ctIdx / ncpg;
        const int64_t ctR = ctIdx - grp * ncpg;
        const int64_t coBase = grp * Cout_g + ctR * CT;
        const int64_t ct = C3Min(CT, Cout_g - ctR * CT);
        const int64_t cig = clIdx / ntpc;
        const int64_t wt = clIdx - cig * ntpc;
        const int64_t colBase = wt * NT;
        const int64_t nt = C3Min(NT, KhKw - colBase);
        const int64_t cci = grp * Cin_g + cig;
        const int64_t yRowBase = ((coBase * Cin_g + cig) * Kd + kd) * KhKw + colBase;

        Duplicate(scl, 0.0f, (int32_t)(ct * nt * 8));

        for (int64_t hc = 0; hc < nHChunks; ++hc) {
            const int64_t oh0 = hc * RT;
            const int64_t rt = C3Min(RT, flatRows - oh0);
            Duplicate(acc, 0.0f, (int32_t)(ct * nt * chunkLen));

            if (b1x1) {
                for (int64_t n = 0; n < N; ++n) {
                    // ---- grad tile for the ct rows (contiguous flattened rows) ----
                    auto gq = qG.AllocTensor<T>();
                    for (int64_t rr = 0; rr < ct; ++rr) {
                        const int64_t co = coBase + rr;
                        const int64_t gOff = ((n * Cout + co) * gradPlane + oh0) * Wout;
                        if (Wout == RW) {
                            // Row pitch already matches the destination pitch, so the
                            // whole rt-row slice is one contiguous block.
                            DataCopyExtParams cp{(uint16_t)1, (uint32_t)(rt * Wout * ES), 0u, 0u, 0u};
                            DataCopyPadExtParams<T> pp{true, 0, 0, (T)0};
                            DataCopyPad(gq[(int32_t)(rr * chunkLen)], gGm[gOff], cp, pp);
                        } else {
                            DataCopyExtParams cp{(uint16_t)rt, (uint32_t)(Wout * ES), 0u, 0u, 0u};
                            DataCopyPadExtParams<T> pp{true, 0, (uint8_t)(RW - Wout), (T)0};
                            DataCopyPad(gq[(int32_t)(rr * chunkLen)], gGm[gOff], cp, pp);
                        }
                    }
                    qG.EnQue(gq);
                    auto gq2 = qG.DeQue<T>();
                    for (int64_t rr = 0; rr < ct; ++rr) {
                        Cast(gf[(int32_t)(rr * chunkLen)], gq2[(int32_t)(rr * chunkLen)],
                             RM_IN, (int32_t)(rt * RW));
                    }

                    // ---- x windows (contiguous flattened rows) --------------------
                    auto xq = qX.AllocTensor<T>();
                    for (int64_t j = 0; j < nt; ++j) {
                        const int64_t xOff = ((n * Cin + cci) * gradPlane + oh0) * rowStride;
                        if (Wout == RW) {
                            DataCopyExtParams cp{(uint16_t)1, (uint32_t)(rt * Wout * ES), 0u, 0u, 0u};
                            DataCopyPadExtParams<T> pp{true, 0, 0, (T)0};
                            DataCopyPad(xq[(int32_t)(j * chunkLen)], xUse[xOff], cp, pp);
                        } else {
                            DataCopyExtParams cp{(uint16_t)rt, (uint32_t)(Wout * ES), 0u, 0u, 0u};
                            DataCopyPadExtParams<T> pp{true, 0, (uint8_t)(RW - Wout), (T)0};
                            DataCopyPad(xq[(int32_t)(j * chunkLen)], xUse[xOff], cp, pp);
                        }
                    }
                    qX.EnQue(xq);
                    auto xq2 = qX.DeQue<T>();
                    for (int64_t j = 0; j < nt; ++j) {
                        Cast(xf[(int32_t)(j * chunkLen)], xq2[(int32_t)(j * chunkLen)],
                             RM_IN, (int32_t)(rt * RW));
                    }

                    for (int64_t j = 0; j < nt; ++j) {
                        for (int64_t rr = 0; rr < ct; ++rr) {
                            MulAddDst(acc[(int32_t)((rr * nt + j) * chunkLen)],
                                      gf[(int32_t)(rr * chunkLen)],
                                      xf[(int32_t)(j * chunkLen)], (int32_t)(rt * RW));
                        }
                    }
                    qG.FreeTensor(gq2);
                    qX.FreeTensor(xq2);
                }
            } else {
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t od = 0; od < Dout; ++od) {
                    const int64_t id = od * sd - pd + kd * dd;
                    const bool idOk = (id >= 0) && (id < D);

                    // ---- grad tile for the ct rows -------------------------
                    auto gq = qG.AllocTensor<T>();
                    for (int64_t rr = 0; rr < ct; ++rr) {
                        const int64_t co = coBase + rr;
                        const int64_t gOff = (((n * Cout + co) * Dout + od) * Hout + oh0) * Wout;
                        if (Wout == RW) {
                            DataCopyExtParams cp{(uint16_t)1, (uint32_t)(rt * Wout * ES), 0u, 0u, 0u};
                            DataCopyPadExtParams<T> pp{true, 0, 0, (T)0};
                            DataCopyPad(gq[(int32_t)(rr * chunkLen)], gGm[gOff], cp, pp);
                        } else {
                            DataCopyExtParams cp{(uint16_t)rt, (uint32_t)(Wout * ES), 0u, 0u, 0u};
                            DataCopyPadExtParams<T> pp{true, 0, (uint8_t)(RW - Wout), (T)0};
                            DataCopyPad(gq[(int32_t)(rr * chunkLen)], gGm[gOff], cp, pp);
                        }
                    }
                    qG.EnQue(gq);
                    auto gq2 = qG.DeQue<T>();
                    for (int64_t rr = 0; rr < ct; ++rr) {
                        Cast(gf[(int32_t)(rr * chunkLen)], gq2[(int32_t)(rr * chunkLen)],
                             RM_IN, (int32_t)(rt * RW));
                    }

                    // ---- x windows -----------------------------------------
                    auto xq = qX.AllocTensor<T>();
                    if (idOk) {
                        for (int64_t j = 0; j < nt; ++j) {
                            const int64_t wh = colBase + j;
                            const int64_t kh = wh / Kw;
                            const int64_t kw = wh - kh * Kw;
                            C3Win w = C3Window(kh, kw, oh0, rt, H, Wout, RW, rowStride, subPl,
                                               sh, ph, dh, pw, dw);
                            if ((w.a1 > w.a0) && (w.validCols > 0)) {
                                const int64_t ihA = (oh0 + w.a0) * sh + (kh * dh - ph);
                                int64_t xOff;
                                if (subPl == 1) {
                                    xOff = (((n * Cin + cci) * D + id) * H + ihA) * rowStride + w.colStart;
                                } else {
                                    int64_t c0 = kw * dw - pw;
                                    int64_t mm = c0 % subPl;
                                    if (mm < 0) {
                                        mm += subPl;
                                    }
                                    xOff = ((((n * Cin + cci) * subPl + mm) * D + id) * H + ihA) * rowStride
                                           + w.colStart;
                                }
                                DataCopyExtParams cp{(uint16_t)(w.a1 - w.a0), (uint32_t)(w.validCols * ES),
                                                     (uint32_t)((sh * rowStride - w.validCols) * ES), 0u, 0u};
                                DataCopyPadExtParams<T> pp{true, (uint8_t)w.leftPad,
                                                           (uint8_t)w.rightPad, (T)0};
                                if (w.leftPad == 0 && (sh * rowStride - w.validCols) == 0 &&
                                    w.validCols == RW) {
                                    // No horizontal border and the source rows are
                                    // contiguous: fold all window rows into a single block.
                                    DataCopyExtParams cp1{(uint16_t)1,
                                                          (uint32_t)((w.a1 - w.a0) * w.validCols * ES),
                                                          0u, 0u, 0u};
                                    DataCopyPadExtParams<T> pp1{true, 0, 0, (T)0};
                                    DataCopyPad(xq[(int32_t)(j * chunkLen + w.a0 * RW)],
                                                xUse[xOff], cp1, pp1);
                                } else {
                                    DataCopyPad(xq[(int32_t)(j * chunkLen + w.a0 * RW)],
                                                xUse[xOff], cp, pp);
                                }
                            }
                        }
                    }
                    qX.EnQue(xq);
                    auto xq2 = qX.DeQue<T>();
                    if (idOk) {
                        for (int64_t j = 0; j < nt; ++j) {
                            Cast(xf[(int32_t)(j * chunkLen)], xq2[(int32_t)(j * chunkLen)],
                                 RM_IN, (int32_t)(rt * RW));
                        }
                    }

                    // ---- multiply add --------------------------------------
                    for (int64_t j = 0; j < nt; ++j) {
                        const int64_t wh = colBase + j;
                        const int64_t kh = wh / Kw;
                        const int64_t kw = wh - kh * Kw;
                        C3Win w = C3Window(kh, kw, oh0, rt, H, Wout, RW, rowStride, subPl,
                                           sh, ph, dh, pw, dw);
                        if (!idOk) {
                            w.a0 = 0;
                            w.a1 = 0;
                        }
                        const int64_t t0 = w.a0 * RW;
                        const int64_t t1 = w.a1 * RW;
                        const int64_t all = rt * RW;
                        for (int64_t rr = 0; rr < ct; ++rr) {
                            auto ac = acc[(int32_t)((rr * nt + j) * chunkLen)];
                            auto gr = gf[(int32_t)(rr * chunkLen)];
                            if (t0 > 0) {
                                MulAddDst(ac, gr, xz, (int32_t)t0);
                            }
                            if (t1 > t0) {
                                auto xr = xf[(int32_t)(j * chunkLen + t0)];
                                MulAddDst(ac[t0], gr[t0], xr, (int32_t)(t1 - t0));
                            }
                            if (t1 < all) {
                                MulAddDst(ac[t1], gr[t1], xz, (int32_t)(all - t1));
                            }
                        }
                    }
                    qG.FreeTensor(gq2);
                    qX.FreeTensor(xq2);
                }
            }
            }

            // ---- reduce ------------------------------------------------
            for (int64_t i = 0; i < ct * nt; ++i) {
                ReduceSum(red[(int32_t)(i * 8)], acc[(int32_t)(i * chunkLen)], tmp, (int32_t)chunkLen);
            }
            Add(scl, scl, red, (uint64_t)1, (uint8_t)(ct * nt), {1, 1, 1, 1, 1, 1});
        }

        // ---- compact + store -------------------------------------------
        for (int64_t rr = 0; rr < ct; ++rr) {
            Gather(cf, scl[(int32_t)(rr * nt * 8)], offU, (uint32_t)0, (uint32_t)nt);
            auto oT = qO.AllocTensor<T>();
            Cast(oT, cf, RM_OUT, (int32_t)nt);
            qO.EnQue(oT);
            auto oT2 = qO.DeQue<T>();
            DataCopyExtParams cp{(uint16_t)1, (uint32_t)(nt * ES), 0u, 0u, 0u};
            DataCopyPad(yGm[yRowBase + rr * Cin_g * Kd * KhKw], oT2, cp);
            qO.FreeTensor(oT2);
        }
    }
}

/* --------------------------------------------------------------------------- */
/* host tiling                                                                  */
/* --------------------------------------------------------------------------- */
C3BpfTiling calc_conv_3d_bpf_tiling(
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Dout, int64_t Hout, int64_t Wout,
    int64_t Kd, int64_t Kh, int64_t Kw,
    int64_t sd, int64_t sh, int64_t sw,
    int64_t pd, int64_t ph, int64_t pw,
    int64_t dd, int64_t dh, int64_t dw,
    int64_t groups, int64_t elemSize)
{
    (void)N; (void)D; (void)H; (void)W; (void)Wout;
    (void)sd; (void)sh; (void)pd; (void)ph; (void)dd; (void)dh; (void)dw;

    C3BpfTiling t;
    t.numBlocks = 1; t.CT = 1; t.NT = 1; t.RT = 1; t.RW = 16; t.chunkLen = 16;
    t.nColTilesPerCi = 1; t.ncpg = 1; t.numItems = 0;
    t.rowStride = W; t.subPl = 1; t.jmax = W;
    t.b1x1 = 0;

    const int64_t es = elemSize;
    const int64_t KhKw = Kh * Kw;
    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;
    if (Cin_g <= 0 || Cout_g <= 0 || KhKw <= 0 || Dout <= 0 || Hout <= 0 || Wout <= 0) {
        return t;
    }

    int64_t RW = ((Wout + 15) / 16) * 16;
    if (RW < 16) {
        RW = 16;
    }
    t.RW = RW;

    int64_t rowStride = W;
    int64_t subPl = 1;
    int64_t jmax = W;
    if (sw > 1) {
        subPl = sw;
        jmax = (W + sw - 1) / sw;
        rowStride = jmax;
    }
    t.rowStride = rowStride;
    t.subPl = subPl;
    t.jmax = jmax;

    // Pure contraction regime: 1x1x1 filter, no padding, unit stride.  Then the
    // reduction index (od, oh, ow) is exactly the x index, so the row axis can be
    // flattened over (od, oh) and whole planes can be moved with one DMA.
    int64_t b1x1 = 0;
    if (Kd == 1 && Kh == 1 && Kw == 1 && pd == 0 && ph == 0 && pw == 0 &&
        sd == 1 && sh == 1 && sw == 1 &&
        Dout == D && Hout == H && Wout == W) {
        b1x1 = 1;
    }
    t.b1x1 = b1x1;
    const int64_t totRows = b1x1 ? (Dout * Hout) : Hout;

    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = plat->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    const double budget = (double)((int64_t)((double)ubSize * 0.75));
    t.numBlocks = coreNum;

    const double macs = (double)Cout_g * (double)Cin_g * (double)Kd * (double)KhKw *
                        (double)N * (double)Dout * (double)Hout * (double)Wout;
    const double cores = (double)coreNum;
    // calibrated against the measured case19 baseline (0.819 s at RT=1/CT=18/NT=9)
    const double CLK = 1.8e9;
    const double BW = 1.2e12;
    const double C_DMA = 160.0;   // per DataCopyPad (issue + paired Cast when 16-bit)
    const double C_MAD = 18.0;    // fixed part of one MulAddDst call
    const double C_RED = 180.0;   // per ReduceSum call

    const int64_t rtCand[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256};
    const int64_t ntCand[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 48, 64};

    // Candidate RT values.  In the b1x1 regime RT counts flattened (od, oh) rows, so
    // only whole multiples of Hout keep the operands contiguous; in the general
    // regime RT is a plain plane-row count bounded by Hout.
    std::vector<int64_t> rtList;
    if (b1x1) {
        for (int64_t og = 1; og <= Dout; ++og) {
            if (og * Hout > 4095) {
                break;   // DataCopyPad blockCount limit
            }
            rtList.push_back(og * Hout);
        }
    } else {
        for (size_t i = 0; i < sizeof(rtCand) / sizeof(rtCand[0]); ++i) {
            rtList.push_back(rtCand[i]);
        }
    }

    double bestCost = 1e300;
    int64_t bestCT = 1, bestNT = 1, bestRT = 1;
    for (size_t ia = 0; ia < rtList.size(); ++ia) {
        const int64_t RT = rtList[ia];
        if (RT > totRows) {
            continue;
        }
        const double chunk = (double)(RT * RW);
        const double nHC = (double)((totRows + RT - 1) / RT);
        for (size_t ib = 0; ib < sizeof(ntCand) / sizeof(ntCand[0]); ++ib) {
            const int64_t NT = ntCand[ib];
            if (NT > KhKw) {
                continue;
            }
            // UB(bytes) = chunk*(4*CT*NT + 8*CT + 8*NT + 4) + 68*CT*NT + misc
            const double constTerm = chunk * (8.0 * (double)NT + 4.0) + 4096.0;
            const double ctCoef = chunk * (4.0 * (double)NT + 8.0) + 68.0 * (double)NT;
            const double num = budget - constTerm;
            if (num <= 0.0 || ctCoef <= 0.0) {
                continue;
            }
            int64_t CT = (int64_t)(num / ctCoef);
            if (CT > Cout_g) {
                CT = Cout_g;
            }
            if (NT > 0 && CT > 255 / NT) {
                CT = 255 / NT;
            }
            if (CT < 1) {
                continue;
            }
            const double nCoT = (double)((Cout_g + CT - 1) / CT);
            const double nColT = (double)Cin_g * (double)((KhKw + NT - 1) / NT);
            const double items = (double)Kd * nCoT * nColT;
            // reduction extent still walked inside one item
            const double outer = b1x1 ? 1.0 : (double)Dout;
            const double iterations = items * nHC * (double)N * outer;
            const double dmaCount = iterations * (double)(CT + NT);
            const double madCalls = iterations * (double)CT * (double)NT;
            const double redCalls = items * nHC * (double)CT * (double)NT;
            const double cycles = dmaCount * C_DMA +
                                  madCalls * (C_MAD + chunk / 32.0) +
                                  redCalls * C_RED +
                                  macs / 32.0;
            const double tComp = cycles / cores / CLK;
            const double tTraf = (macs / (double)CT + macs / (double)NT) * (double)es / BW;
            const double cost = tComp > tTraf ? tComp : tTraf;
            if (cost < bestCost) {
                bestCost = cost;
                bestCT = CT;
                bestNT = NT;
                bestRT = RT;
            }
        }
    }
    t.CT = bestCT;
    t.NT = bestNT;
    t.RT = bestRT;
    t.chunkLen = t.RT * t.RW;
    t.nColTilesPerCi = (KhKw + t.NT - 1) / t.NT;
    t.ncpg = (Cout_g + t.CT - 1) / t.CT;
    t.numItems = Kd * (groups * t.ncpg) * (Cin_g * t.nColTilesPerCi);
    return t;
}

/* --------------------------------------------------------------------------- */
/* launch wrappers                                                              */
/* --------------------------------------------------------------------------- */
extern "C" {

void launch_conv3d_bpf_main_half(
    GM_ADDR x, GM_ADDR g, GM_ADDR y, GM_ADDR xs,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Dout, int64_t Hout, int64_t Wout,
    int64_t Kd, int64_t Kh, int64_t Kw,
    int64_t sd, int64_t sh, int64_t sw,
    int64_t pd, int64_t ph, int64_t pw,
    int64_t dd, int64_t dh, int64_t dw,
    int64_t Cin_g, int64_t Cout_g, int64_t groups,
    int64_t rowStride, int64_t subPl, int64_t RW,
    int64_t CT, int64_t NT, int64_t RT,
    int64_t ncpg, int64_t ntpc, int64_t numItems, int64_t numBlocks,
    int64_t b1x1,
    void* stream)
{
    c3bpf_main_kernel<half><<<numBlocks, nullptr, stream>>>(
        x, g, y, xs, N, Cin, D, H, W, Cout, Dout, Hout, Wout, Kd, Kh, Kw,
        sd, sh, sw, pd, ph, pw, dd, dh, dw, Cin_g, Cout_g, groups,
        rowStride, subPl, RW, CT, NT, RT, ncpg, ntpc, numItems, numBlocks, b1x1);
}

void launch_conv3d_bpf_main_bf16(
    GM_ADDR x, GM_ADDR g, GM_ADDR y, GM_ADDR xs,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Dout, int64_t Hout, int64_t Wout,
    int64_t Kd, int64_t Kh, int64_t Kw,
    int64_t sd, int64_t sh, int64_t sw,
    int64_t pd, int64_t ph, int64_t pw,
    int64_t dd, int64_t dh, int64_t dw,
    int64_t Cin_g, int64_t Cout_g, int64_t groups,
    int64_t rowStride, int64_t subPl, int64_t RW,
    int64_t CT, int64_t NT, int64_t RT,
    int64_t ncpg, int64_t ntpc, int64_t numItems, int64_t numBlocks,
    int64_t b1x1,
    void* stream)
{
    c3bpf_main_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(
        x, g, y, xs, N, Cin, D, H, W, Cout, Dout, Hout, Wout, Kd, Kh, Kw,
        sd, sh, sw, pd, ph, pw, dd, dh, dw, Cin_g, Cout_g, groups,
        rowStride, subPl, RW, CT, NT, RT, ncpg, ntpc, numItems, numBlocks, b1x1);
}

void launch_conv3d_bpf_split_half(
    GM_ADDR x, GM_ADDR xs, int64_t rows, int64_t W, int64_t jmax,
    int64_t sw, int64_t DH, int64_t numBlocks, void* stream)
{
    c3bpf_split_kernel<half><<<numBlocks, nullptr, stream>>>(x, xs, rows, W, jmax, sw, DH, numBlocks);
}

void launch_conv3d_bpf_split_bf16(
    GM_ADDR x, GM_ADDR xs, int64_t rows, int64_t W, int64_t jmax,
    int64_t sw, int64_t DH, int64_t numBlocks, void* stream)
{
    c3bpf_split_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, xs, rows, W, jmax, sw, DH, numBlocks);
}

} // extern "C"
