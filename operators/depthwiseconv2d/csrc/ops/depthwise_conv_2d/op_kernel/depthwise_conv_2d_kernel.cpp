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
 * \file depthwise_conv_2d_kernel.cpp
 * \brief DepthwiseConv2D device kernel + host tiling/launch (bisheng + -xasc, dav-2201).
 *
 * Math
 * ----
 *   y[n,c,ho,wo] = bias[c] + sum_{kh,kw} x[n,c,ho*sh+kh*dh-ph, wo*sw+kw*dw-pw] * weight[c,kh,kw]
 *
 * Shape of the work
 * -----------------
 * A work item is one (plane = n*C + c, block of R output rows); a grid-stride loop over all items
 * walks the task space. Within an item the accumulation runs on an *extended output width*
 *   Wz = (Wout-1)*sw + 1
 * on which every tap (kw) is one contiguous fused multiply-add:
 *   z[i]     += w[kh,kw] * xp[i + kw*dw]        (i in [0, Wz))
 * and the stride is applied once, at the very end, by y[wo] = z[wo*sw].
 *
 * Building a tap operand
 * ----------------------
 * A single DataCopyPad from GM produces the whole tap vector for every row of the block:
 * the left/right padding fields of DataCopyPad emit the zero border, so the destination element
 * index already equals the extended-width index.  Each destination row is `rowa` (a multiple of 16)
 * fp32 elements wide, hence every vector operand base used below is 32B aligned.
 * The out-of-image rows of a block (only ever the first / last block of a plane) are zero filled with
 * Duplicate, which keeps 0*inf = NaN parity with the golden (the padded taps are really multiplied).
 *
 * Two load modes
 * --------------
 *  - SLAB mode (row stride 1): one DMA per kw loads a vertical span of S = R + (Kh-1)*dh rows; the
 *    Kh taps then reuse the same buffer shifted by kh*dh rows (32B aligned shift), so the GM traffic
 *    is ~Kw*S rows instead of Kh*Kw*R.
 *  - TAP mode (row stride > 1, or a span that does not fit UB): one row-strided DMA per (kh,kw).
 *
 * Precision
 * ---------
 * Every operand is brought to fp32, accumulation is fp32 (Axpy), bias is added in fp32 and the result
 * is rounded to the input dtype exactly once, at store time.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "depthwise_conv_2d_launch.h"

using namespace AscendC;

namespace dwc {

constexpr int64_t PIPE_DEPTH = 2;
constexpr int64_t UB_SLACK = 24576;
constexpr int64_t MAX_ROWS = 255;
constexpr int64_t ROW_ALIGN = 16; // 16 fp32 elements == 64B
// Upper bound for a compensation residual that is still meaningful.  A finite residual is
// bounded by (taps * ulp(maxFinite)/2) < 1e32, so anything at or above this bound is inf or NaN.
constexpr float DW_COMP_BOUND = 1.0e37f;
// The bit-mask form of Compares / Select (mode 1 / 2) needs this much UB reserved on Atlas A2.
constexpr int64_t DW_SELECT_RSV = 8192;

union F32U {
    uint32_t u;
    float f;
};

/*! \brief IEEE binary16 bit pattern -> float, with integer arithmetic only (no 16-bit scalar cast). */
__aicore__ inline float HalfToF32(uint16_t h)
{
    const uint32_t s = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t e = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    F32U r;
    if (e == 0u) {
        if (m == 0u) {
            r.u = s;
        } else {
            uint32_t shift = 0u;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                shift += 1u;
            }
            m &= 0x3FFu;
            r.u = s | (((uint32_t)(127 - 15) - shift) << 23) | (m << 13);
        }
    } else if (e == 31u) {
        r.u = s | 0x7F800000u | (m << 13);
    } else {
        r.u = s | ((e + 112u) << 23) | (m << 13);
    }
    return r.f;
}

/*! \brief bfloat16 bit pattern -> float (a pure bit move). */
__aicore__ inline float Bf16ToF32(uint16_t b)
{
    F32U r;
    r.u = ((uint32_t)b) << 16;
    return r.f;
}

/*! \brief Scalar read of one element of a tensor held in GM, always returned as fp32. */
template <typename T>
__aicore__ inline float ReadScalarF32(GM_ADDR ptr, int64_t idx);

template <>
__aicore__ inline float ReadScalarF32<float>(GM_ADDR ptr, int64_t idx)
{
    GlobalTensor<float> g;
    g.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ptr));
    return g.GetValue((uint64_t)idx);
}

template <>
__aicore__ inline float ReadScalarF32<half>(GM_ADDR ptr, int64_t idx)
{
    GlobalTensor<uint16_t> g;
    g.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(ptr));
    return HalfToF32(g.GetValue((uint64_t)idx));
}

template <>
__aicore__ inline float ReadScalarF32<bfloat16_t>(GM_ADDR ptr, int64_t idx)
{
    GlobalTensor<uint16_t> g;
    g.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(ptr));
    return Bf16ToF32(g.GetValue((uint64_t)idx));
}

/*! \brief Zero value of T without a 16-bit scalar conversion. */
template <typename T>
__aicore__ inline T ZeroT();

template <>
__aicore__ inline float ZeroT<float>()
{
    return 0.0f;
}

template <>
__aicore__ inline half ZeroT<half>()
{
    half h;
    *reinterpret_cast<uint16_t*>(&h) = (uint16_t)0;
    return h;
}

template <>
__aicore__ inline bfloat16_t ZeroT<bfloat16_t>()
{
    bfloat16_t h;
    *reinterpret_cast<uint16_t*>(&h) = (uint16_t)0;
    return h;
}

__aicore__ inline int64_t AlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

__aicore__ inline int64_t Align32B(int64_t bytes)
{
    return ((bytes + 31) / 32) * 32;
}

/*! \brief Horizontal geometry of tap kw on the zero padded input row. */
__aicore__ inline void HGeom(int64_t kw, int64_t dw, int64_t pw, int64_t Wz, int64_t W,
                             int64_t& leftPad, int64_t& srcStart, int64_t& valid, int64_t& rightPad)
{
    const int64_t p0 = kw * dw - pw;
    leftPad = 0;
    srcStart = p0;
    if (srcStart < 0) {
        leftPad = -srcStart;
        srcStart = 0;
    }
    int64_t srcEnd = p0 + Wz;
    if (srcEnd > W) {
        srcEnd = W;
    }
    if (srcEnd < srcStart) {
        srcEnd = srcStart;
    }
    valid = srcEnd - srcStart;
    rightPad = Wz - leftPad - valid;
    if (rightPad < 0) {
        rightPad = 0;
    }
}

/*! \brief Copy the fp32 row block into the (T typed) store buffer. */
template <typename T>
__aicore__ inline void StoreCopy(const LocalTensor<T>& dst, const LocalTensor<float>& src,
                                 int32_t count);

template <>
__aicore__ inline void StoreCopy<float>(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                        int32_t count)
{
    Adds(dst, src, 0.0f, count);
}

template <>
__aicore__ inline void StoreCopy<half>(const LocalTensor<half>& dst, const LocalTensor<float>& src,
                                       int32_t count)
{
    Cast(dst, src, RoundMode::CAST_RINT, count);
}

template <>
__aicore__ inline void StoreCopy<bfloat16_t>(const LocalTensor<bfloat16_t>& dst,
                                             const LocalTensor<float>& src, int32_t count)
{
    Cast(dst, src, RoundMode::CAST_RINT, count);
}

/*! \brief One compensated accumulation step: cur += w*op, with the rounding error of the step
 * tracked in `comp`.  `cur` and `alt` ping-pong so the running sum never has to be copied.
 *
 * Only used for 32 bit data, where the accumulation rounding is the dominant error term and could
 * not be excused by the per-dtype threshold.  For 16 bit data the final cast to the storage dtype
 * already dominates, so the plain Axpy is used there. */
__aicore__ inline void CompTap(LocalTensor<float>& cur, LocalTensor<float>& alt,
                               LocalTensor<float>& comp, LocalTensor<float>& prod,
                               const LocalTensor<float>& op, int32_t off, float w, int32_t count)
{
    Muls(prod[off], op, w, count);        // prod = fl(w*op)
    Add(alt[off], cur[off], prod[off], count); // alt  = fl(cur + prod)
    Sub(cur[off], cur[off], alt[off], count);  // cur  = fl(cur - alt)
    Add(cur[off], cur[off], prod[off], count); // cur  = fl(cur + prod) == error of the step
    Add(comp[off], comp[off], cur[off], count);
    LocalTensor<float> tmp = cur;
    cur = alt;
    alt = tmp;
}

/*! \brief Bring the freshly DMA'd raw operand block to a 32B aligned fp32 tensor.
 *
 * For a 32 bit operand the raw block already holds fp32, so it is reused in place (no copy).
 * For a 16 bit operand the range outside [zLo, zHi) is zeroed, the valid rows are cast to fp32 and
 * the (unread) positions beyond Wz-1 keep the harmless dummy the DMA left there.
 */
template <typename T>
__aicore__ inline void PrepareOp(const LocalTensor<T>& rawLt, LocalTensor<float>& opBuf,
                                 LocalTensor<float>& opLt, int64_t bufRows, int64_t rowa, int64_t zLo,
                                 int64_t zHi)
{
    if (sizeof(T) == 4) {
        opLt = rawLt.template ReinterpretCast<float>();
        if (zLo > 0) {
            Duplicate(opLt, 0.0f, (int32_t)(zLo * rowa));
        }
        if (zHi < bufRows) {
            Duplicate(opLt[(int32_t)(zHi * rowa)], 0.0f, (int32_t)((bufRows - zHi) * rowa));
        }
    } else {
        if (zLo > 0) {
            Duplicate(opBuf, 0.0f, (int32_t)(zLo * rowa));
        }
        if (zHi < bufRows) {
            Duplicate(opBuf[(int32_t)(zHi * rowa)], 0.0f, (int32_t)((bufRows - zHi) * rowa));
        }
        if (zHi > zLo) {
            Cast(opBuf[(int32_t)(zLo * rowa)], rawLt[(int32_t)(zLo * rowa)], RoundMode::CAST_NONE,
                 (int32_t)((zHi - zLo) * rowa));
        }
        opLt = opBuf;
    }
}

template <typename T>
__global__ __aicore__ void depthwise_conv_2d_kernel(GM_ADDR xPtr, GM_ADDR wPtr, GM_ADDR bPtr,
                                                    GM_ADDR yPtr, DWConvParams p)
{
    const int64_t R = p.R;
    const int64_t rowa = p.rowa;
    const int64_t outa = p.outa;
    const int64_t bufRows = p.slabMode ? p.S : R;
    const int64_t elemSize = p.elemSize;

    GlobalTensor<T> xGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xPtr));
    GlobalTensor<T> yGm;
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(yPtr));

    TPipe pipe;
    TQue<QuePosition::VECIN, PIPE_DEPTH> inQ;
    TQue<QuePosition::VECOUT, 1> outQ;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> acc2Buf;
    TBuf<TPosition::VECCALC> compBuf;
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> opBuf;
    TBuf<TPosition::VECCALC> yBuf;
    TBuf<TPosition::VECCALC> offBuf;

    const int64_t compOn = (sizeof(T) == 4) ? 1 : 0;
    // Compares reports one bit per element and requires count*sizeof(T) to be 256B aligned, so the
    // element counters used by the masking path are rounded up; every buffer they touch carries
    // 256B of slack so the rounded tail stays inside the allocation.
    const int64_t nElem = R * rowa;
    const int64_t nPad = ((nElem + 63) / 64) * 64;
    const uint32_t inBlock = (uint32_t)Align32B(bufRows * rowa * elemSize);
    pipe.InitBuffer(inQ, PIPE_DEPTH, inBlock);
    pipe.InitBuffer(outQ, 1, (uint32_t)Align32B(R * rowa * elemSize));
    pipe.InitBuffer(accBuf, (uint32_t)Align32B(R * rowa * 4 + 256));
    if (compOn) {
        pipe.InitBuffer(acc2Buf, (uint32_t)Align32B(R * rowa * 4 + 256));
        pipe.InitBuffer(compBuf, (uint32_t)Align32B(R * rowa * 4 + 256));
        pipe.InitBuffer(prodBuf, (uint32_t)Align32B(R * rowa * 4 + 256));
        pipe.InitBuffer(maskBuf, (uint32_t)Align32B(nPad / 8 + 32));
    } else {
        pipe.InitBuffer(opBuf, (uint32_t)Align32B(bufRows * rowa * 4));
    }
    if (p.sw > 1) {
        pipe.InitBuffer(yBuf, (uint32_t)Align32B(R * outa * 4));
        pipe.InitBuffer(offBuf, (uint32_t)Align32B(outa * 4));
    }

    LocalTensor<float> opBufLt;
    LocalTensor<float> compLt;
    LocalTensor<float> prodLt;
    LocalTensor<float> altLt;
    if (compOn) {
        compLt = compBuf.Get<float>();
        prodLt = prodBuf.Get<float>();
        altLt = acc2Buf.Get<float>();
    } else {
        opBufLt = opBuf.Get<float>();
    }

    // Gather index vector for the stride-2 (and generally sw>1) decimation, built once per core.
    LocalTensor<uint32_t> offLt;
    if (p.sw > 1) {
        LocalTensor<float> tmpF = offBuf.Get<float>();
        CreateVecIndex(tmpF, 0.0f, (uint32_t)outa);
        Muls(tmpF, tmpF, (float)(p.sw * 4), (int32_t)outa); // byte offsets into the fp32 accumulator
        LocalTensor<int32_t> tmpI = offBuf.Get<int32_t>();
        Cast(tmpI, tmpF, RoundMode::CAST_RINT, (int32_t)outa);
        offLt = tmpI.ReinterpretCast<uint32_t>();
    }

    const int64_t storePitch = (p.sw > 1) ? outa : rowa;

    for (int64_t item = GetBlockIdx(); item < p.totalItems; item += p.numBlocks) {
        const int64_t plane = item / p.numRowBlocks;
        const int64_t rb = item - plane * p.numRowBlocks;
        int64_t rn = R;
        const int64_t ho0 = rb * R;
        if (ho0 + rn > p.Hout) {
            rn = p.Hout - ho0;
        }
        const int64_t c = plane % p.C;
        const int64_t n = plane / p.C;
        const int64_t xPlane = ((n * p.C + c) * p.H) * p.W;
        const int64_t yPlane = ((n * p.C + c) * p.Hout) * p.Wout;
        const float biasF = ReadScalarF32<T>(bPtr, c);

        LocalTensor<float> cur = accBuf.Get<float>();
        LocalTensor<float> alt = altLt;
        Duplicate(cur, 0.0f, (int32_t)(R * rowa));
        if (compOn) {
            Duplicate(alt, 0.0f, (int32_t)(R * rowa));
            Duplicate(compLt, 0.0f, (int32_t)(R * rowa));
        }

        if (p.slabMode) {
            const int64_t S = p.S;
            const int64_t ih0 = ho0 - p.ph;
            int64_t jLo = 0;
            if (ih0 < 0) {
                jLo = -ih0;
                if (jLo > S) {
                    jLo = S;
                }
            }
            int64_t jHi = S;
            if (ih0 + S > p.H) {
                jHi = p.H - ih0;
                if (jHi < 0) {
                    jHi = 0;
                }
                if (jHi > S) {
                    jHi = S;
                }
            }
            if (jHi < jLo) {
                jHi = jLo;
            }
            const int64_t nRows = jHi - jLo;
            // Soft pipelining: the load of tap kw+1 is issued before tap kw is consumed, so the MTE2
            // latency of the next row block overlaps the vector work of the current one instead of
            // being paid again on every tap.  The extra iteration only drains the pipeline.
            for (int64_t kw = 0; kw <= p.Kw; ++kw) {
                if (kw < p.Kw) {
                    int64_t lp = 0, ss = 0, vc = 0, rp = 0;
                    HGeom(kw, p.dw, p.pw, p.Wz, p.W, lp, ss, vc, rp);
                    LocalTensor<T> raw = inQ.AllocTensor<T>();
                    if (nRows > 0 && vc > 0) {
                        DataCopyExtParams cp;
                        cp.blockCount = (uint16_t)nRows;
                        cp.blockLen = (uint32_t)(vc * elemSize);
                        cp.srcStride = (uint32_t)((p.W - vc) * elemSize);
                        cp.dstStride = (uint32_t)((rowa * elemSize - Align32B(p.Wz * elemSize)) / 32);
                        cp.rsv = 0;
                        DataCopyPadExtParams<T> pp;
                        pp.isPad = true;
                        pp.leftPadding = (uint8_t)lp;
                        pp.rightPadding = (uint8_t)rp;
                        pp.paddingValue = ZeroT<T>();
                        DataCopyPad(raw[(int32_t)(jLo * rowa)],
                                    xGm[xPlane + (ih0 + jLo) * p.W + ss], cp, pp);
                    }
                    inQ.EnQue(raw);
                }
                if (kw > 0) {
                    const int64_t prev = kw - 1;
                    int64_t lp = 0, ss = 0, vc = 0, rp = 0;
                    HGeom(prev, p.dw, p.pw, p.Wz, p.W, lp, ss, vc, rp);
                    LocalTensor<T> raw = inQ.DeQue<T>();
                    LocalTensor<float> opLt;
                    const int64_t zLo = (vc > 0) ? jLo : bufRows;
                    const int64_t zHi = (vc > 0) ? jHi : bufRows;
                    PrepareOp<T>(raw, opBufLt, opLt, bufRows, rowa, zLo, zHi);
                    for (int64_t kh = 0; kh < p.Kh; ++kh) {
                        const float w = ReadScalarF32<T>(wPtr, (c * p.Kh + kh) * p.Kw + prev);
                        if (compOn) {
                            CompTap(cur, alt, compLt, prodLt, opLt[(int32_t)(kh * p.dh * rowa)], 0, w,
                                    (int32_t)(R * rowa));
                        } else {
                            Axpy(cur, opLt[(int32_t)(kh * p.dh * rowa)], w, (int32_t)(R * rowa));
                        }
                    }
                    inQ.FreeTensor(raw);
                }
            }
        } else {
            const int64_t nTaps = p.Kh * p.Kw;
            for (int64_t t = 0; t <= nTaps; ++t) {
                if (t < nTaps) {
                    const int64_t kh = t / p.Kw;
                    const int64_t kw = t - kh * p.Kw;
                    const int64_t ih0 = ho0 * p.sh + kh * p.dh - p.ph;
                    int64_t rLo = 0;
                    if (ih0 < 0) {
                        rLo = (-ih0 + p.sh - 1) / p.sh;
                    }
                    int64_t rHi = R;
                    {
                        const int64_t lim = p.H - 1 - ih0;
                        int64_t cand = (lim < 0) ? 0 : (lim / p.sh + 1);
                        if (cand > R) {
                            cand = R;
                        }
                        if (cand < 0) {
                            cand = 0;
                        }
                        rHi = cand;
                    }
                    if (rHi < rLo) {
                        rHi = rLo;
                    }
                    int64_t lp = 0, ss = 0, vc = 0, rp = 0;
                    HGeom(kw, p.dw, p.pw, p.Wz, p.W, lp, ss, vc, rp);
                    LocalTensor<T> raw = inQ.AllocTensor<T>();
                    const int64_t nRows = rHi - rLo;
                    if (nRows > 0 && vc > 0) {
                        DataCopyExtParams cp;
                        cp.blockCount = (uint16_t)nRows;
                        cp.blockLen = (uint32_t)(vc * elemSize);
                        cp.srcStride = (uint32_t)((p.sh * p.W - vc) * elemSize);
                        cp.dstStride = (uint32_t)((rowa * elemSize - Align32B(p.Wz * elemSize)) / 32);
                        cp.rsv = 0;
                        DataCopyPadExtParams<T> pp;
                        pp.isPad = true;
                        pp.leftPadding = (uint8_t)lp;
                        pp.rightPadding = (uint8_t)rp;
                        pp.paddingValue = ZeroT<T>();
                        DataCopyPad(raw[(int32_t)(rLo * rowa)],
                                    xGm[xPlane + (ih0 + rLo * p.sh) * p.W + ss], cp, pp);
                    }
                    inQ.EnQue(raw);
                }
                if (t > 0) {
                    const int64_t pt = t - 1;
                    const int64_t kh = pt / p.Kw;
                    const int64_t kw = pt - kh * p.Kw;
                    const int64_t ih0 = ho0 * p.sh + kh * p.dh - p.ph;
                    int64_t rLo = 0;
                    if (ih0 < 0) {
                        rLo = (-ih0 + p.sh - 1) / p.sh;
                    }
                    int64_t rHi = R;
                    {
                        const int64_t lim = p.H - 1 - ih0;
                        int64_t cand = (lim < 0) ? 0 : (lim / p.sh + 1);
                        if (cand > R) {
                            cand = R;
                        }
                        if (cand < 0) {
                            cand = 0;
                        }
                        rHi = cand;
                    }
                    if (rHi < rLo) {
                        rHi = rLo;
                    }
                    int64_t lp = 0, ss = 0, vc = 0, rp = 0;
                    HGeom(kw, p.dw, p.pw, p.Wz, p.W, lp, ss, vc, rp);
                    LocalTensor<T> raw = inQ.DeQue<T>();
                    LocalTensor<float> opLt;
                    const int64_t zLo = (vc > 0) ? rLo : bufRows;
                    const int64_t zHi = (vc > 0) ? rHi : bufRows;
                    PrepareOp<T>(raw, opBufLt, opLt, bufRows, rowa, zLo, zHi);
                    const int64_t nRows = rHi - rLo;
                    if (nRows > 0) {
                        const float w = ReadScalarF32<T>(wPtr, (c * p.Kh + kh) * p.Kw + kw);
                        if (compOn) {
                            CompTap(cur, alt, compLt, prodLt, opLt[(int32_t)(rLo * rowa)],
                                    (int32_t)(rLo * rowa), w, (int32_t)(nRows * rowa));
                        } else {
                            Axpy(cur[(int32_t)(rLo * rowa)], opLt[(int32_t)(rLo * rowa)], w,
                                 (int32_t)(nRows * rowa));
                        }
                    }
                    inQ.FreeTensor(raw);
                }
            }
        }

        // ---- fold the compensation, add the bias, decimate (sw>1), cast and store --------------
        if (compOn) {
            // The compensated residual is only meaningful while the running sum stayed finite.
            // As soon as one product (or partial sum) is inf/NaN the residual of that step becomes
            // NaN, and folding it in would replace an IEEE-correct +-inf by a NaN.  Compares builds
            // a bit mask of the residuals that are safely finite (|residual| < bound; the comparison
            // is false for NaN, inf and -inf alike), Select turns the unsafe ones into exactly 0 and
            // the residual is then added without ever meeting a non-finite operand.
            LocalTensor<uint8_t> maskLt = maskBuf.Get<uint8_t>();
            // `cur` may alias accBuf or acc2Buf (CompTap swaps the two), so the scratch is taken
            // from prodBuf, which never takes part in that ping-pong and is free after the taps.
            LocalTensor<float> compSan = prodLt;
            Compares(maskLt, compLt, DW_COMP_BOUND, CMPMODE::LT, (uint32_t)nPad);
            Select(compSan, maskLt, compLt, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, (uint32_t)nPad);
            Add(cur, cur, compSan, (int32_t)nPad);
        }
        LocalTensor<float> yLt;
        if (p.sw > 1) {
            yLt = yBuf.Get<float>();
            for (int64_t r = 0; r < rn; ++r) {
                Gather(yLt[(int32_t)(r * outa)], cur[(int32_t)(r * rowa)], offLt, (uint32_t)0,
                       (uint32_t)p.Wout);
            }
            Adds(yLt, yLt, biasF, (int32_t)(rn * outa));
        } else {
            Adds(cur, cur, biasF, (int32_t)(rn * rowa));
            yLt = cur;
        }

        LocalTensor<T> outLt = outQ.AllocTensor<T>();
        StoreCopy<T>(outLt, yLt, (int32_t)(rn * storePitch));
        outQ.EnQue(outLt);
        outLt = outQ.DeQue<T>();
        DataCopyExtParams cpo;
        cpo.blockCount = (uint16_t)rn;
        cpo.blockLen = (uint32_t)(p.Wout * elemSize);
        cpo.srcStride = (uint32_t)((storePitch * elemSize - Align32B(p.Wout * elemSize)) / 32);
        cpo.dstStride = 0;
        cpo.rsv = 0;
        DataCopyPad(yGm[yPlane + ho0 * p.Wout], outLt, cpo);
        outQ.FreeTensor(outLt);
    }
}

} // namespace dwc

/*! \brief UB size and AIV core count of this SoC, queried once and then reused. */
struct DwPlatform {
    uint64_t ubBytes;
    int64_t coreNum;
};

static DwPlatform DwQueryPlatform()
{
    DwPlatform r;
    r.ubBytes = 192 * 1024;
    r.coreNum = 1;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        uint64_t ub = 0;
        plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        if (ub != 0) {
            r.ubBytes = ub;
        }
        const int64_t cn = plat->GetCoreNumAiv();
        if (cn > 0) {
            r.coreNum = cn;
        }
    }
    return r;
}

DWConvParams calc_depthwise_conv_2d_tiling(int64_t N, int64_t C, int64_t H, int64_t W, int64_t Kh,
                                           int64_t Kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw,
                                           int64_t dh, int64_t dw, int64_t Hout, int64_t Wout,
                                           int64_t elemSize)
{
    using namespace dwc;
    DWConvParams p;
    p.N = N;
    p.C = C;
    p.H = H;
    p.W = W;
    p.Kh = Kh;
    p.Kw = Kw;
    p.sh = sh;
    p.sw = sw;
    p.ph = ph;
    p.pw = pw;
    p.dh = dh;
    p.dw = dw;
    p.Hout = Hout;
    p.Wout = Wout;
    p.elemSize = elemSize;
    p.Wz = (Wout - 1) * sw + 1;
    p.rowa = ((p.Wz + ROW_ALIGN - 1) / ROW_ALIGN) * ROW_ALIGN;
    p.outa = ((Wout + ROW_ALIGN - 1) / ROW_ALIGN) * ROW_ALIGN;

    // The platform query sits on the critical path of every launch, so it is performed once per
    // process instead of once per call.
    static const DwPlatform dwPlat = DwQueryPlatform();
    const uint64_t ubBytes = dwPlat.ubBytes;
    const int64_t coreNum = dwPlat.coreNum;

    const int64_t limit = (int64_t)ubBytes - UB_SLACK;
    const int64_t rowF = p.rowa * 4;
    const int64_t rowT = p.rowa * elemSize;
    const int64_t outT = p.outa * 4;

    auto align32 = [](int64_t v) { return ((v + 31) / 32) * 32; };
    // Exact UB footprint of depthwise_conv_2d_kernel for a given row count (mirrors InitBuffer).
    auto usageOf = [&](int64_t rows, int64_t slab) {
        const int64_t bufRows = slab ? (rows + (Kh - 1) * dh) : rows;
        int64_t u = 2 * align32(bufRows * rowT);          // inQ (depth 2)
        u += align32(rows * rowT);                        // outQ
        u += align32(rows * rowF + 256);                  // acc
        if (elemSize == 4) {
            u += 3 * align32(rows * rowF + 256);          // alt, comp, prod
            const int64_t nEl = rows * p.rowa;
            const int64_t nPd = ((nEl + 63) / 64) * 64;
            u += align32(nPd / 8 + 32);                   // Compares bit mask
            u += DW_SELECT_RSV;                           // Select scratch reserved on A2
        } else {
            u += align32(bufRows * rowF);                 // fp32 operand
        }
        if (sw > 1) {
            u += align32(rows * outT);                    // decimated rows
            u += align32(outT);                           // gather offsets
        }
        return u;
    };
    auto maxRows = [&](int64_t slab) {
        int64_t lo = 1;
        int64_t hi = (Hout < MAX_ROWS) ? Hout : MAX_ROWS;
        if (hi < 1) {
            hi = 1;
        }
        if (usageOf(lo, slab) > limit) {
            return (int64_t)0;
        }
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo + 1) / 2;
            if (usageOf(mid, slab) <= limit) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    };

    const int64_t rSlab = (sh == 1) ? maxRows(1) : 0;
    const int64_t rTap = maxRows(0);

    int64_t R;
    int64_t slabMode;
    if (rSlab >= 1 && rSlab * 4 >= rTap) {
        slabMode = 1;
        R = rSlab;
    } else {
        slabMode = 0;
        R = rTap;
    }
    if (R < 1) {
        R = 1;
    }
    if (R > Hout) {
        R = Hout;
    }
    if (R < 1) {
        R = 1;
    }

    // Load balancing.  Every row block repeats the whole Kh*Kw DMA chain of an item, and the fixed
    // part of that chain (descriptor issue, MTE2 completion, queue hand-off) is far larger than the
    // row-proportional work for the narrow tiles, so an item is only worth splitting when the
    // problem would otherwise leave more than half of the cores idle.  R therefore stays at the
    // UB-limited maximum whenever there is roughly one item per core.
    const int64_t planes = N * C;
    if (planes > 0) {
        while (R > 1 && planes * ((Hout + R - 1) / R) * 2 < coreNum) {
            R = R / 2;
        }
    }
    // Even out the row blocks so no work item carries a short tail block.
    if (R < Hout) {
        const int64_t nb = (Hout + R - 1) / R;
        if (nb > 0) {
            R = (Hout + nb - 1) / nb;
        }
    }

    p.R = R;
    p.slabMode = slabMode;
    p.S = slabMode ? (R + (Kh - 1) * dh) : R;
    p.numRowBlocks = (Hout + R - 1) / R;
    p.totalItems = N * C * p.numRowBlocks;
    p.numBlocks = (p.totalItems < coreNum) ? p.totalItems : coreNum;
    if (p.numBlocks < 1) {
        p.numBlocks = 1;
    }
    return p;
}

extern "C" {

void launch_depthwise_conv_2d_float(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, DWConvParams p,
                                    void* stream)
{
    dwc::depthwise_conv_2d_kernel<float><<<p.numBlocks, nullptr, stream>>>(x, w, b, y, p);
}

void launch_depthwise_conv_2d_half(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, DWConvParams p,
                                   void* stream)
{
    dwc::depthwise_conv_2d_kernel<half><<<p.numBlocks, nullptr, stream>>>(x, w, b, y, p);
}

void launch_depthwise_conv_2d_bfloat16(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, DWConvParams p,
                                       void* stream)
{
    dwc::depthwise_conv_2d_kernel<bfloat16_t><<<p.numBlocks, nullptr, stream>>>(x, w, b, y, p);
}

} // extern "C"
