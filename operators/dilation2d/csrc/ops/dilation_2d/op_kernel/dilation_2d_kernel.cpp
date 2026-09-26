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
 * \file dilation_2d_kernel.cpp
 * \brief Dilation2D kernel + tiling + launch (compiled with bisheng + -xasc)
 *
 *   y[n,oh,ow,c] = max_{dy,dx} ( x[n, oh*sH + rH*dy - padTop, ow*sW + rW*dx - padLeft, c]
 *                                + filter[dy,dx,c] )
 *
 * Out-of-range taps contribute -inf, which is exactly the reference's -inf padding.
 * Everything is evaluated in fp16, matching the reference (patches + filter in fp16, then max).
 *
 * Layout / tiling
 * ---------------
 * A "row" is one (n, oh) output plane row, i.e. outW * C output elements.  Rows are
 * flattened over (n, oh) and split contiguously over the vector cores.
 *
 * Within a row the data is processed in tiles of `tw` output columns x `pad` channels,
 * where pad = align16(C).  pad is what makes the UB layout legal: DataCopyPad pads every
 * block to 32B, so with dstStride = 0 the UB pitch of one column is align32(C*2) = pad*2
 * bytes and every per-column UB base is 32B aligned.  When C*2 is not a multiple of 32
 * (C % 16 != 0) the GM block starts k*(C*2) are not 32B aligned, so each block is split
 * across 32B granules and a column block costs ~32 cycles + bytes/9.5 instead of ~8 cycles
 * + bytes/18.8.  The tiling therefore minimises the *number of columns moved*:
 *   input columns = loads * (outW + nTiles*e),   output columns = outW per row,
 * with e the per-tile extra span the dilated taps need.
 *
 * Adding the structural element (filter) is a per-channel broadcast over the columns.  To
 * keep the vector ops plain count-based element-wise ops, each core pre-expands the
 * filter into `fexp[t]` = a [tw][pad] buffer whose `tw` columns are all the same tap
 * vector.  The expansion uses log2 doubling (one written column per tap, then
 * copy [0,cur) -> [cur,cur+n)), so it costs Tap*(1+ceil(log2(tw))) vector ops instead of
 * Tap*tw.
 *
 * Two scheduling modes
 * --------------------
 * plain   (cacheR == 2): rows are the outer loop and tiles the inner one.  Every tap's
 *          input tile is loaded, consumed and released immediately.
 *
 * ring    (cacheR  > 2): tiles are the outer loop and rows the inner one.  For sW == 1
 *          each input row is used by exactly fh output rows; the ring holds
 *          R = rH*(fh-1)+sH input rows keyed by (ih mod R) so a row is (re)loaded only when
 *          its slot does not already hold it, cutting the moved columns by ~fh.  The slot
 *          map can only cause extra reloads, never stale data, because each buffer also
 *          carries the row index it holds.  A deeper ring means more buffers in flight, so
 *          the tiling only switches to it for a clear cost-model win; the plain mode stays
 *          for every marginal case.
 *
 * Two data-path variants:
 *   sW == 1 : for each dy one contiguous span of (twc + rW*(fw-1)) input columns is moved
 *             once; each dx then uses a sub-view starting at column rW*dx.
 *   sW >  1 : the taps are grouped by the residue class of their first sampled column
 *             modulo sW.  Taps in one group differ only by a whole number of sW strides, so
 *             ONE strided span (blockCount = number of grid columns, srcStride =
 *             (sW-1)*C*2 bytes) serves the whole group and each tap reads it at a
 *             different column offset.  With rW divisible by sW there is a single group,
 *             so all fw taps share one load instead of fw loads.
 * The span is always computed from the *actual* tile width twc, so the last (short) tile
 * does not move the extra e columns the full tile would.
 *
 * In both variants out-of-range columns/rows of the tile are filled with -inf by
 * Duplicate, and an out-of-range tap is *not* skipped: the reference's -inf padding still
 * takes part in the max, and (-inf) + (+inf filter) is NaN there just like the reference.
 */

#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "dilation_2d_launch.h"

constexpr int64_t D2_TW_MAX = 64;
constexpr int64_t D2_MAX_R = 16;
constexpr int64_t D2_GMAX = 4;
constexpr int64_t D2_SENT = -1000000000LL;

namespace dilation2d_ns {

__aicore__ inline half D2NegInf()
{
    uint16_t u = 0xFC00u;  // fp16 -inf
    return *reinterpret_cast<half *>(&u);
}

__aicore__ inline int64_t D2CeilDiv(int64_t a, int64_t b)
{
    return (a > 0) ? ((a + b - 1) / b) : 0;
}

// Group the fw taps by the residue class of their first sampled column modulo sW.
// Returns the number of groups (<= D2_GMAX) or -1 if there would be more.
//   rOf[g]    residue of the group's grid
//   qOf[dx]   grid index of tap dx's first column *within its group*
//   grpOf[dx] group index of tap dx
//   spanOf[g] number of grid columns the group's single load must cover for a tile of tw
__aicore__ inline int64_t D2Groups(int64_t rW, int64_t sW, int64_t fw, int64_t tw,
                                    int64_t *rOf, int64_t *qOf, int64_t *grpOf, int64_t *spanOf)
{
    int64_t qMax[D2_GMAX];
    int64_t nG = 0;
    for (int64_t dx = 0; dx < fw; ++dx) {
        const int64_t r = (rW * dx) % sW;
        const int64_t q = (rW * dx) / sW;
        int64_t g = -1;
        for (int64_t k = 0; k < nG; ++k) {
            if (rOf[k] == r) {
                g = k;
                break;
            }
        }
        if (g < 0) {
            if (nG >= D2_GMAX) {
                return -1;
            }
            g = nG;
            rOf[nG] = r;
            qMax[nG] = 0;
            ++nG;
        }
        grpOf[dx] = g;
        qOf[dx] = q;
        if (q > qMax[g]) {
            qMax[g] = q;
        }
    }
    for (int64_t k = 0; k < nG; ++k) {
        spanOf[k] = tw + qMax[k];
    }
    return nG;
}

template <int DEPTH>
__aicore__ inline void D2RunTiles(
    GM_ADDR x, GM_ADDR y,
    int64_t H, int64_t W, int64_t C, int64_t outH, int64_t outW,
    int64_t fh, int64_t fw, int64_t sH, int64_t sW, int64_t rH, int64_t rW,
    int64_t padTop, int64_t padLeft,
    int64_t rowStart, int64_t rowEnd,
    int64_t tw, int64_t pad, int64_t cacheR,
    AscendC::LocalTensor<half> &fexp, AscendC::LocalTensor<half> &tmp,
    AscendC::TQue<AscendC::QuePosition::VECIN, DEPTH> &cq,
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> &accQ)
{
    const int64_t Tap = fh * fw;
    const int64_t eA = rW * (fw - 1);  // path A per-tile span overhead (columns)
    const uint32_t colBytes = static_cast<uint32_t>(C * sizeof(half));

    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    xGm.SetGlobalBuffer((__gm__ half *)x, 1);
    yGm.SetGlobalBuffer((__gm__ half *)y, 1);

    AscendC::DataCopyPadExtParams<half> padP;
    padP.isPad = false;
    padP.leftPadding = 0;
    padP.rightPadding = 0;
    padP.paddingValue = static_cast<half>(0);

    const half ninf = D2NegInf();
    const int64_t nTiles = (outW + tw - 1) / tw;
    const int64_t R = (cacheR < 2) ? 2 : cacheR;
    const bool useCache = (R > 2);

    int64_t rOf[D2_GMAX];
    int64_t qOf[8];
    int64_t grpOf[8];
    int64_t spanOf[D2_GMAX];
    int64_t nG = 1;
    if (sW > 1) {
        nG = D2Groups(rW, sW, fw, tw, rOf, qOf, grpOf, spanOf);
    }

    AscendC::LocalTensor<half> ct[D2_MAX_R];
    int64_t rowOf[D2_MAX_R];

    if (!useCache) {
        // ------------------------------------------------------------------
        // plain streaming: row-outer / tile-inner, immediate release
        // ------------------------------------------------------------------
        for (int64_t row = rowStart; row < rowEnd; ++row) {
            const int64_t n = row / outH;
            const int64_t oh = row - n * outH;
            const int64_t rowElemBase = (n * H) * W;  // * C below
            const int64_t yRowColBase = row * outW;

            for (int64_t ti = 0; ti < nTiles; ++ti) {
                const int64_t ow0 = ti * tw;
                int64_t twc = outW - ow0;
                if (twc > tw) {
                    twc = tw;
                }
                if (twc <= 0) {
                    continue;
                }
                const uint32_t nEl = static_cast<uint32_t>(twc * pad);

                AscendC::LocalTensor<half> acc = accQ.template AllocTensor<half>();
                AscendC::Duplicate(acc, ninf, nEl);

                if (sW == 1) {
                    const int64_t spanT = twc + eA;
                    const uint32_t spanElT = static_cast<uint32_t>(spanT * pad);
                    const int64_t a = ow0 - padLeft;  // logical input column of span slot 0
                    const int64_t c0 = (a < 0) ? 0 : a;
                    int64_t c1 = a + spanT - 1;
                    if (c1 > W - 1) {
                        c1 = W - 1;
                    }
                    const int64_t pre = c0 - a;
                    const int64_t suf = (a + spanT - 1) - c1;
                    const int64_t nblk = c1 - c0 + 1;

                    for (int64_t dy = 0; dy < fh; ++dy) {
                        const int64_t ih = oh * sH + rH * dy - padTop;
                        AscendC::LocalTensor<half> t = cq.template AllocTensor<half>();
                        if (ih < 0 || ih >= H) {
                            AscendC::Duplicate(t, ninf, spanElT);
                        } else {
                            if (pre > 0) {
                                AscendC::Duplicate(t, ninf, static_cast<uint32_t>(pre * pad));
                            }
                            if (suf > 0) {
                                AscendC::Duplicate(t[(spanT - suf) * pad], ninf,
                                                   static_cast<uint32_t>(suf * pad));
                            }
                            if (nblk > 0) {
                                AscendC::DataCopyExtParams cp;
                                cp.blockCount = static_cast<uint16_t>(nblk);
                                cp.blockLen = colBytes;
                                cp.srcStride = 0;
                                cp.dstStride = 0;
                                cp.rsv = 0;
                                AscendC::DataCopyPad(t[pre * pad],
                                                     xGm[(rowElemBase + ih * W + c0) * C], cp,
                                                     padP);
                            }
                        }
                        cq.template EnQue(t);
                        AscendC::LocalTensor<half> tl = cq.template DeQue<half>();
                        for (int64_t dx = 0; dx < fw; ++dx) {
                            const int64_t tap = dy * fw + dx;
                            AscendC::Add(tmp, tl[rW * dx * pad], fexp[tap * tw * pad], nEl);
                            AscendC::Max(acc, acc, tmp, nEl);
                        }
                        cq.FreeTensor(tl);
                    }
                } else if (nG > 0) {
                    const int64_t abase = ow0 * sW - padLeft;
                    for (int64_t dy = 0; dy < fh; ++dy) {
                        const int64_t ih = oh * sH + rH * dy - padTop;
                        const bool rowOob = (ih < 0 || ih >= H);
                        for (int64_t g = 0; g < nG; ++g) {
                            const int64_t sg = twc + (spanOf[g] - tw);  // columns for this tile
                            const int64_t base = abase + rOf[g];
                            const uint32_t nElG = static_cast<uint32_t>(sg * pad);
                            int64_t g0 = 0;
                            if (base < 0) {
                                g0 = D2CeilDiv(-base, sW);
                            }
                            int64_t g1 = sg - 1;
                            const int64_t lastOk = W - 1 - base;
                            if (lastOk < 0) {
                                g1 = -1;
                            } else {
                                const int64_t gg = lastOk / sW;
                                if (gg < g1) {
                                    g1 = gg;
                                }
                            }
                            if (g0 > g1) {
                                g0 = 0;
                                g1 = -1;  // whole strided span out of range
                            }
                            const bool allOob = (g1 < 0);
                            AscendC::LocalTensor<half> t = cq.template AllocTensor<half>();
                            if (rowOob || allOob) {
                                AscendC::Duplicate(t, ninf, nElG);
                            } else {
                                if (g0 > 0) {
                                    AscendC::Duplicate(t, ninf, static_cast<uint32_t>(g0 * pad));
                                }
                                const int64_t suf = sg - 1 - g1;
                                if (suf > 0) {
                                    AscendC::Duplicate(t[(sg - suf) * pad], ninf,
                                                       static_cast<uint32_t>(suf * pad));
                                }
                                const int64_t srcCol = base + g0 * sW;
                                const int64_t nblk = g1 - g0 + 1;
                                AscendC::DataCopyExtParams cp;
                                cp.blockCount = static_cast<uint16_t>(nblk);
                                cp.blockLen = colBytes;
                                cp.srcStride = static_cast<uint32_t>((sW - 1) * C * sizeof(half));
                                cp.dstStride = 0;
                                cp.rsv = 0;
                                AscendC::DataCopyPad(t[g0 * pad],
                                                     xGm[(rowElemBase + ih * W + srcCol) * C], cp,
                                                     padP);
                            }
                            cq.template EnQue(t);
                            AscendC::LocalTensor<half> tl = cq.template DeQue<half>();
                            for (int64_t dx = 0; dx < fw; ++dx) {
                                if (grpOf[dx] != g) {
                                    continue;
                                }
                                const int64_t tap = dy * fw + dx;
                                AscendC::Add(tmp, tl[qOf[dx] * pad], fexp[tap * tw * pad], nEl);
                                AscendC::Max(acc, acc, tmp, nEl);
                            }
                            cq.FreeTensor(tl);
                        }
                    }
                } else {
                    // fallback: one strided load per (dy,dx)
                    for (int64_t dy = 0; dy < fh; ++dy) {
                        const int64_t ih = oh * sH + rH * dy - padTop;
                        const bool rowOob = (ih < 0 || ih >= H);
                        for (int64_t dx = 0; dx < fw; ++dx) {
                            const int64_t tap = dy * fw + dx;
                            const int64_t col0 = ow0 * sW + rW * dx - padLeft;
                            int64_t jlo = 0;
                            if (col0 < 0) {
                                jlo = D2CeilDiv(-col0, sW);
                            }
                            int64_t jhi = twc - 1;
                            const int64_t lastOk = W - 1 - col0;
                            if (lastOk < 0) {
                                jhi = -1;
                            } else {
                                const int64_t jh = lastOk / sW;
                                if (jh < jhi) {
                                    jhi = jh;
                                }
                            }
                            if (jlo > jhi) {
                                jlo = 0;
                                jhi = -1;  // whole tile out of range -> pure -inf tap
                            }
                            const bool allOob = (jhi < 0);
                            AscendC::LocalTensor<half> t = cq.template AllocTensor<half>();
                            if (rowOob || allOob) {
                                AscendC::Duplicate(t, ninf, nEl);
                            } else {
                                if (jlo > 0) {
                                    AscendC::Duplicate(t, ninf, static_cast<uint32_t>(jlo * pad));
                                }
                                if (jhi < twc - 1) {
                                    AscendC::Duplicate(t[(jhi + 1) * pad], ninf,
                                                       static_cast<uint32_t>((twc - 1 - jhi) * pad));
                                }
                                const int64_t srcCol = col0 + jlo * sW;
                                const int64_t nb = jhi - jlo + 1;
                                AscendC::DataCopyExtParams cp;
                                cp.blockCount = static_cast<uint16_t>(nb);
                                cp.blockLen = colBytes;
                                cp.srcStride = static_cast<uint32_t>((sW - 1) * C * sizeof(half));
                                cp.dstStride = 0;
                                cp.rsv = 0;
                                AscendC::DataCopyPad(t[jlo * pad],
                                                     xGm[(rowElemBase + ih * W + srcCol) * C], cp,
                                                     padP);
                            }
                            cq.template EnQue(t);
                            AscendC::LocalTensor<half> tl = cq.template DeQue<half>();
                            AscendC::Add(tmp, tl, fexp[tap * tw * pad], nEl);
                            AscendC::Max(acc, acc, tmp, nEl);
                            cq.FreeTensor(tl);
                        }
                    }
                }

                accQ.template EnQue(acc);
                AscendC::LocalTensor<half> ao = accQ.template DeQue<half>();
                AscendC::DataCopyExtParams ocp;
                ocp.blockCount = static_cast<uint16_t>(twc);
                ocp.blockLen = colBytes;
                ocp.srcStride = 0;
                ocp.dstStride = 0;
                ocp.rsv = 0;
                AscendC::DataCopyPad(yGm[(yRowColBase + ow0) * C], ao, ocp);
                accQ.FreeTensor(ao);
            }
        }
        return;
    }

    // ----------------------------------------------------------------------
    // ring cache: tile-outer / row-inner (sW == 1 only)
    // ----------------------------------------------------------------------
    (void)Tap;
    for (int64_t ti = 0; ti < nTiles; ++ti) {
        const int64_t ow0 = ti * tw;
        int64_t twc = outW - ow0;
        if (twc > tw) {
            twc = tw;
        }
        if (twc <= 0) {
            continue;
        }
        const uint32_t nEl = static_cast<uint32_t>(twc * pad);

        const int64_t spanT = twc + eA;
        const uint32_t spanElT = static_cast<uint32_t>(spanT * pad);
        const int64_t a = ow0 - padLeft;  // logical input column of span slot 0
        const int64_t c0 = (a < 0) ? 0 : a;
        int64_t c1 = a + spanT - 1;
        if (c1 > W - 1) {
            c1 = W - 1;
        }
        const int64_t pre = c0 - a;
        const int64_t suf = (a + spanT - 1) - c1;
        const int64_t nblk = c1 - c0 + 1;

        for (int64_t s = 0; s < R; ++s) {
            rowOf[s] = D2_SENT;
        }

        for (int64_t row = rowStart; row < rowEnd; ++row) {
            const int64_t n = row / outH;
            const int64_t oh = row - n * outH;
            const int64_t rowElemBase = (n * H) * W;  // * C below
            const int64_t yRowColBase = row * outW;

            AscendC::LocalTensor<half> acc = accQ.template AllocTensor<half>();
            AscendC::Duplicate(acc, ninf, nEl);

            for (int64_t dy = 0; dy < fh; ++dy) {
                const int64_t ih = oh * sH + rH * dy - padTop;
                int64_t slot = ih % R;
                if (slot < 0) {
                    slot += R;
                }
                if (rowOf[slot] != ih) {
                    if (rowOf[slot] != D2_SENT) {
                        cq.FreeTensor(ct[slot]);
                    }
                    AscendC::LocalTensor<half> t = cq.template AllocTensor<half>();
                    if (ih < 0 || ih >= H) {
                        // Reference pads with -inf at this tap; the tap still takes part in
                        // the max (and becomes NaN when the filter tap is +inf).
                        AscendC::Duplicate(t, ninf, spanElT);
                    } else {
                        if (pre > 0) {
                            AscendC::Duplicate(t, ninf, static_cast<uint32_t>(pre * pad));
                        }
                        if (suf > 0) {
                            AscendC::Duplicate(t[(spanT - suf) * pad], ninf,
                                               static_cast<uint32_t>(suf * pad));
                        }
                        if (nblk > 0) {
                            AscendC::DataCopyExtParams cp;
                            cp.blockCount = static_cast<uint16_t>(nblk);
                            cp.blockLen = colBytes;
                            cp.srcStride = 0;
                            cp.dstStride = 0;
                            cp.rsv = 0;
                            AscendC::DataCopyPad(t[pre * pad],
                                                 xGm[(rowElemBase + ih * W + c0) * C], cp,
                                                 padP);
                        }
                    }
                    cq.template EnQue(t);
                    ct[slot] = cq.template DeQue<half>();
                    rowOf[slot] = ih;
                }
                AscendC::LocalTensor<half> tl = ct[slot];
                for (int64_t dx = 0; dx < fw; ++dx) {
                    const int64_t tap = dy * fw + dx;
                    AscendC::Add(tmp, tl[rW * dx * pad], fexp[tap * tw * pad], nEl);
                    AscendC::Max(acc, acc, tmp, nEl);
                }
            }

            accQ.template EnQue(acc);
            AscendC::LocalTensor<half> ao = accQ.template DeQue<half>();
            AscendC::DataCopyExtParams ocp;
            ocp.blockCount = static_cast<uint16_t>(twc);
            ocp.blockLen = colBytes;
            ocp.srcStride = 0;
            ocp.dstStride = 0;
            ocp.rsv = 0;
            AscendC::DataCopyPad(yGm[(yRowColBase + ow0) * C], ao, ocp);
            accQ.FreeTensor(ao);
        }

        for (int64_t s = 0; s < R; ++s) {
            if (rowOf[s] != D2_SENT) {
                cq.FreeTensor(ct[s]);
                rowOf[s] = D2_SENT;
            }
        }
    }
}

__global__ __aicore__ void dilation_2d_kernel_half(
    GM_ADDR x, GM_ADDR filter, GM_ADDR y,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t outH, int64_t outW, int64_t fh, int64_t fw,
    int64_t sH, int64_t sW, int64_t rH, int64_t rW,
    int64_t padTop, int64_t padLeft,
    int64_t rowsTotal, int64_t blockRows, int64_t tw, int64_t pad, int64_t cacheR)
{
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t rowStart = blk * blockRows;
    int64_t rowEnd = rowStart + blockRows;
    if (rowEnd > rowsTotal) {
        rowEnd = rowsTotal;
    }
    if (rowStart >= rowEnd) {
        return;
    }

    AscendC::GlobalTensor<half> fGm;
    fGm.SetGlobalBuffer((__gm__ half *)filter, 1);

    const int64_t Tap = fh * fw;
    const uint32_t colBytes = static_cast<uint32_t>(C * sizeof(half));
    int64_t R = (cacheR < 2) ? 2 : cacheR;
    if (R > D2_MAX_R) {
        R = D2_MAX_R;
    }

    // Input buffer size / queue depth: path A spans (tw + rW*(fw-1)) contiguous columns
    // with a ring of R; path B uses one strided span per tap group, loaded one at a time.
    int64_t bufSpan = (sW == 1) ? (tw + rW * (fw - 1)) : tw;
    int64_t qDepth = R;
    if (sW > 1) {
        qDepth = 2;
        int64_t rO[D2_GMAX];
        int64_t qO[8];
        int64_t gr[8];
        int64_t sp[D2_GMAX];
        const int64_t ng = D2Groups(rW, sW, fw, tw, rO, qO, gr, sp);
        if (ng > 0) {
            int64_t ms = 0;
            for (int64_t k = 0; k < ng; ++k) {
                if (sp[k] > ms) {
                    ms = sp[k];
                }
            }
            if (ms > bufSpan) {
                bufSpan = ms;
            }
        }
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, D2_MAX_R> cq;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> filtQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> accQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fexpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    pipe.InitBuffer(cq, static_cast<uint8_t>(qDepth),
                    static_cast<uint32_t>(bufSpan * pad * sizeof(half)));
    pipe.InitBuffer(filtQ, 1, static_cast<uint32_t>(Tap * pad * sizeof(half)));
    pipe.InitBuffer(accQ, 2, static_cast<uint32_t>(tw * pad * sizeof(half)));
    pipe.InitBuffer(fexpBuf, static_cast<uint32_t>(Tap * tw * pad * sizeof(half)));
    pipe.InitBuffer(tmpBuf, static_cast<uint32_t>(tw * pad * sizeof(half)));

    AscendC::LocalTensor<half> fexp = fexpBuf.Get<half>();
    AscendC::LocalTensor<half> tmp = tmpBuf.Get<half>();

    AscendC::DataCopyPadExtParams<half> padP;
    padP.isPad = false;
    padP.leftPadding = 0;
    padP.rightPadding = 0;
    padP.paddingValue = static_cast<half>(0);

    // ---- structural element -> UB, then broadcast over the tw tile columns ----------
    // log2-doubling build: column 0 of every tap first, then copy the already built
    // prefix onto the next non-overlapping block.
    {
        AscendC::LocalTensor<half> ft = filtQ.template AllocTensor<half>();
        for (int64_t t = 0; t < Tap; ++t) {
            AscendC::DataCopyExtParams cp;
            cp.blockCount = 1;
            cp.blockLen = colBytes;
            cp.srcStride = 0;
            cp.dstStride = 0;
            cp.rsv = 0;
            AscendC::DataCopyPad(ft[t * pad], fGm[t * C], cp, padP);
        }
        filtQ.template EnQue(ft);
        AscendC::LocalTensor<half> filt = filtQ.template DeQue<half>();
        for (int64_t t = 0; t < Tap; ++t) {
            const int64_t base = t * tw * pad;
            AscendC::Adds(fexp[base], filt[t * pad], static_cast<half>(0),
                          static_cast<uint32_t>(pad));
            int64_t cur = 1;
            while (cur < tw) {
                int64_t nn = tw - cur;
                if (nn > cur) {
                    nn = cur;
                }
                AscendC::Adds(fexp[base + cur * pad], fexp[base], static_cast<half>(0),
                              static_cast<uint32_t>(nn * pad));
                cur += nn;
            }
        }
        filtQ.FreeTensor(filt);
    }

    D2RunTiles<D2_MAX_R>(x, y, H, W, C, outH, outW, fh, fw, sH, sW, rH, rW, padTop, padLeft,
                         rowStart, rowEnd, tw, pad, R, fexp, tmp, cq, accQ);
}

}  // namespace dilation2d_ns

// ---------------------------------------------------------------------------
// Host tiling (bisheng host side, callable from the g++ plugin)
// ---------------------------------------------------------------------------

// Host-side mirror of D2Groups (no device intrinsics): only the group count and the sum of
// the per-group grid spans are needed for the cost model.
static inline void D2GroupsHost(int64_t rW, int64_t sW, int64_t fw, int64_t &nG, int64_t &qsum)
{
    int64_t rOf[D2_GMAX];
    int64_t qMax[D2_GMAX];
    nG = 0;
    for (int64_t dx = 0; dx < fw; ++dx) {
        const int64_t r = (rW * dx) % sW;
        const int64_t q = (rW * dx) / sW;
        int64_t g = -1;
        for (int64_t k = 0; k < nG; ++k) {
            if (rOf[k] == r) {
                g = k;
                break;
            }
        }
        if (g < 0) {
            if (nG >= D2_GMAX) {
                continue;  // conservative: ignore the extra group
            }
            g = nG;
            rOf[nG] = r;
            qMax[nG] = 0;
            ++nG;
        }
        if (q > qMax[g]) {
            qMax[g] = q;
        }
    }
    qsum = 0;
    for (int64_t k = 0; k < nG; ++k) {
        qsum += qMax[k];
    }
}

Dilation2dTiling calc_dilation_2d_tiling(int64_t N, int64_t outH, int64_t outW, int64_t C,
                                         int64_t fh, int64_t fw, int64_t sH, int64_t sW,
                                         int64_t rH, int64_t rW)
{
    Dilation2dTiling t;
    t.rowsTotal = N * outH;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = platform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    int64_t rows = t.rowsTotal;
    if (rows < 1) {
        rows = 1;
    }
    int64_t nb = rows < coreNum ? rows : coreNum;
    if (nb < 1) {
        nb = 1;
    }
    t.blockRows = (rows + nb - 1) / nb;

    const int64_t pad = ((C + 15) / 16) * 16;
    t.pad = pad;
    const int64_t Tap = fh * fw;
    const int64_t eA = (sW == 1) ? (rW * (fw - 1)) : 0;
    // path B: the per-group grid overhead, in columns
    int64_t nGB = 1;
    int64_t qsumB = 0;
    if (sW > 1) {
        D2GroupsHost(rW, sW, fw, nGB, qsumB);
        if (nGB < 1) {
            nGB = 1;
        }
    }
    const int64_t eB = (sW > 1) ? (qsumB + nGB - 1) / nGB : 0;

    // Half-element budget: 3/8 of UB (leaves head room for framework overhead).
    int64_t budget = static_cast<int64_t>(ubSize) * 3 / 8;
    if (budget < 1024) {
        budget = 1024;
    }

    int64_t limit = D2_TW_MAX;
    if (limit > outW) {
        limit = outW;
    }
    if (limit < 1) {
        limit = 1;
    }

    // Buffer budget: fexp Tap*tw*pad + input queue + acc 2*tw*pad + filt Tap*pad + tmp tw*pad
    //   sW == 1: input queue = R * (tw + eA) * pad
    //   sW >  1: input queue = 2 * (tw + eB) * pad   (one group span at a time)
    auto fitTw = [&](int64_t R) -> int64_t {
        for (int64_t cand = limit; cand >= 1; --cand) {
            const int64_t inq = (sW == 1) ? R * (cand + eA) * pad : 2 * (cand + eB) * pad;
            const int64_t total = Tap * cand * pad + inq + 2 * cand * pad + Tap * pad +
                                  cand * pad;
            if (total <= budget) {
                return cand;
            }
        }
        return 1;
    };

    // Per-core cost model in cycles.
    //   vector: ~73 fp16 element-ops per cycle per core (measured on this archive).
    //   dma:    a DataCopyPad column block costs ~8 cycles + bytes/18.8 when the GM block
    //           start is 32B aligned (C*2 % 32 == 0), and ~32 cycles + bytes/9.5 when it is
    //           not, because the engine then splits every block across 32B granules.
    // Columns moved per core (the span is per-tile, so only nTiles*e extra columns):
    //   input  = loads * (outW + nTiles*e),   output = blockRows * outW
    // (Both are already included in the block and byte totals below.)
    const bool aligned = ((C * 2) % 32) == 0;
    const double perBlock = aligned ? 8.0 : 62.0;

    auto estCost = [&](int64_t R, int64_t tw) -> double {
        if (tw < 1) {
            tw = 1;
        }
        const int64_t nTiles = (outW + tw - 1) / tw;
        const double vec = static_cast<double>(t.blockRows) * static_cast<double>(outW) *
                           static_cast<double>(C) * 2.0 * static_cast<double>(Tap) / 73.0;
        double blocks;
        if (sW == 1) {
            const double loads = (R > 2) ? static_cast<double>(t.blockRows + R - 1)
                                         : static_cast<double>(t.blockRows) * static_cast<double>(fh);
            blocks = loads * (static_cast<double>(outW) +
                              static_cast<double>(nTiles) * static_cast<double>(eA));
        } else {
            blocks = static_cast<double>(t.blockRows) * static_cast<double>(fh) *
                     (static_cast<double>(nGB) * static_cast<double>(outW) +
                      static_cast<double>(nTiles) * static_cast<double>(qsumB));
        }
        blocks += static_cast<double>(t.blockRows) * static_cast<double>(outW);  // output write
        const double bytes = blocks * static_cast<double>(C) * 2.0;
        const double dma = blocks * perBlock + bytes / 50.0;
        return (vec > dma ? vec : dma) + 0.3 * (vec < dma ? vec : dma);
    };

    // Ring cache only for sW == 1.  A deeper ring means more buffers in flight, so it is
    // only used when the model gives it a clear win; marginal cases keep the plain mode.
    int64_t rb = 2;
    if (sW == 1 && t.blockRows >= 4) {
        int64_t rmax = rH * (fh - 1) + sH;
        if (rmax > D2_MAX_R) {
            rmax = D2_MAX_R;
        }
        if (rmax >= 3) {
            rb = rmax;
        }
    }

    const int64_t twA = fitTw(2);
    if (rb > 2) {
        const int64_t twB = fitTw(rb);
        if (estCost(rb, twB) < 0.75 * estCost(2, twA)) {
            t.tw = twB;
            t.cacheR = rb;
            return t;
        }
    }
    t.cacheR = 2;
    t.tw = twA;
    return t;
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------
extern "C" {

void launch_dilation_2d_kernel_half(
    GM_ADDR x, GM_ADDR filter, GM_ADDR y,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t outH, int64_t outW, int64_t fh, int64_t fw,
    int64_t sH, int64_t sW, int64_t rH, int64_t rW,
    int64_t padTop, int64_t padLeft,
    int64_t rowsTotal, int64_t blockRows, int64_t tw, int64_t pad,
    int64_t cacheR, int64_t numBlocks, void* stream)
{
    dilation2d_ns::dilation_2d_kernel_half<<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(
        x, filter, y, N, H, W, C, outH, outW, fh, fw, sH, sW, rH, rW, padTop, padLeft,
        rowsTotal, blockRows, tw, pad, cacheR);
}

}  // extern "C"
