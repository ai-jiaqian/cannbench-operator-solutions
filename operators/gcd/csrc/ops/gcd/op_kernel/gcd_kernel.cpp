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
 * \file gcd_kernel.cpp
 * \brief Gcd kernel + host tiling + launch wrappers (compiled with bisheng + -xasc)
 *
 *   y = gcd(x1, x2)  with numpy broadcasting, y.dtype == x1.dtype (int16 / int32 / int64)
 *
 * Compute strategy per tile
 * -------------------------
 *  (D) Degenerate magnitude.  When the vector-reduced maximum magnitude of both operands is at
 *      most 1, every pair is drawn from {0, 1} and gcd(a, b) == max(|a|, |b|) elementwise
 *      (the four cases 0/0, 0/1, 1/0, 1/1 all agree).  Two vector ops answer the tile.
 *
 *  (V) Vector Euclid on fp32.  Used when the maximum magnitude is at or below 2^22, where
 *      every integer is exact in fp32.  12 ops per step:
 *
 *          c  = [b > 0]                      (b is a magnitude, so c = min(b, 1))
 *          bs = max(b, 1)
 *          q  = rint(a / bs)                 (magic-constant round to nearest)
 *          r  = |a - q * bs|
 *          a' = a + c * (b - a)
 *          b' = c * r                        (freeze on b == 0)
 *
 *      gcd(a,b) == gcd(b, a - q*b) for every integer q and gcd(b,x) == gcd(b,|x|), so the
 *      recurrence is exact.  With a, bs <= 2^22 the fp32 division carries at most 0.25 of
 *      absolute error, so q is within one of the nearest integer and |r| <= 0.75*bs: every
 *      intermediate stays an integer below 2^23, i.e. bit exact in fp32, and b strictly
 *      decreases (b' <= 0.75b < b whenever b >= 1).  A lane freezes on its gcd once b reaches
 *      0, so 0/0 yields 0 and gcd(0, x) yields |x|.
 *
 *  (X) Exact int32 Euclid.  Used for int32 tiles whose magnitude exceeds 2^22.  The operand
 *      magnitudes a, b live in int32 registers as unsigned values (0 .. 2^31) and only the
 *      quotient is taken from the float unit:
 *
 *          q  = clamp(rint(|a| / max(|b|,1))) to <= 2^31 - 128
 *          q*b and r = a - q*b in int32  (wraps modulo 2^32)
 *
 *      q is an integer, so gcd(b, |a - q*b|) == gcd(a, b) whatever its value; and with
 *      |q - a/b| <= 0.5 + (a/b)*2^-24 the true remainder satisfies |r| <= b/2 + a*2^-24
 *      <= 2^30 + 256 < 2^31, so the modulo-2^32 int32 subtraction reproduces the true signed
 *      remainder bit for bit.  b decreases whenever b > ~512; below that the magnitudes are
 *      bounded by 2^30 + 256 and the iteration converges by the ordinary Euclid argument.
 *      Int32 has no exact vector Abs on this toolchain, so |v| is formed as max(v, 0 - v).
 *
 *      Every fp32 scratch buffer is aliased by an int32 view of the same UB region, so each
 *      instruction below is written with a destination buffer that is still dead, and never
 *      aliases a source that still holds live data.
 *
 *  (S) Scalar Euclid on unsigned magnitudes.  Reached only for int64 with |v| > 2^22, using a
 *      shift/subtract binary gcd so no 64-bit division is needed.  The result is narrowed back
 *      with a plain truncating cast, reproducing the wrap-around the torch reference produces
 *      when the true gcd is exactly the type minimum.
 *
 *      Both vector loops use the generous 1.6*log2(max)+8 step bound but also probe for
 *      convergence whenever the vector-reduced maximum of b over the tile is 0 (every lane
 *      frozen on its gcd), so the cost tracks the data rather than the worst case.
 *
 * Narrowing back to int16
 * -----------------------
 *  The vector Cast(float -> int16) saturates, but the reference narrows the int32 gcd by
 *  truncation: a gcd magnitude of exactly 32768 must come back as -32768.  The wrap is applied
 *  explicitly in fp32 before the cast, and only when the tile maximum can actually reach 32768
 *  (gcd <= max(|x1|,|x2|), so a smaller tile maximum makes it a no-op).
 *
 * x2 operand classes inside the run
 * ---------------------------------
 *  AFFINE : x2 offset advances one element per output element  (plain DataCopyPad)
 *  CONST  : one single x2 element reused for the whole row      (vector Duplicate)
 *  REPEAT : the run is P periods of repR elements and x2 advances by repStep per period.
 *
 * REPEAT mode layout (the x2 broadcast along a merged axis)
 * --------------------------------------------------------
 *  Inside a REPEAT row the x2 value changes only every repR elements, and every vector operand
 *  must be 32B aligned.  The tile is therefore laid out as `M` slots of
 *  S = align32(repR*sizeof(T))/sizeof(T) elements, one slot per period:
 *
 *    x1  : one multi-block DataCopyPad (blockCount=M, blockLen=repR*sizeof(T), strides 0).
 *          DataCopyPad packs each block up to the 32B granularity, so the UB pitch is exactly
 *          S*sizeof(T) bytes and every slot starts on a 32B boundary - which is also the
 *          hardware requirement for every block destination of a multi-block DataCopyPad.
 *    x2  : one vector Duplicate per period, also on a 32B boundary.
 *    y   : one multi-block DataCopyPad back to GM (blockCount=M, blockLen=repR*sizeof(T)).
 *
 *  The gcd then runs over n = M*S lanes, including the (S-repR) padding lanes of every slot,
 *  whose results are simply never written back.
 *
 * Software pipeline
 * -----------------
 *  When both operands are MTE2-fed (AFFINE), the tile stream issues the next tile's copy-in
 *  before computing the current one, so the MTE2 latency overlaps the vector work instead of
 *  being exposed on every iteration.  When an operand is produced by the vector unit (CONST /
 *  REPEAT fill) the fill already overlaps the other operand's DMA wait, so the simple
 *  issue-then-compute loop is kept.
 */

#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "gcd_launch.h"

namespace gcd_op {

constexpr float GCD_LIMIT = 4194304.0f;   // 2^22
constexpr float GCD_MAGIC = 8388608.0f;   // 2^23
constexpr float GCD_QMAX = 2147483520.0f; // largest fp32 below 2^31

__aicore__ inline void Unpack2(int64_t w, int32_t& lo, int32_t& hi)
{
    lo = (int32_t)(w & 0xFFFFFFFFLL);
    hi = (int32_t)((w >> 32) & 0xFFFFFFFFLL);
}

__aicore__ inline void Unpack4(int64_t a, int64_t b, int64_t c, int64_t d, int32_t* out)
{
    Unpack2(a, out[0], out[1]);
    Unpack2(b, out[2], out[3]);
    Unpack2(c, out[4], out[5]);
    Unpack2(d, out[6], out[7]);
}

__aicore__ inline uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0u) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Shift/subtract binary gcd: avoids 64-bit division on the scalar unit.
__aicore__ inline uint64_t GcdU64(uint64_t a, uint64_t b)
{
    if (a == 0ull) {
        return b;
    }
    if (b == 0ull) {
        return a;
    }
    uint32_t s = 0u;
    while (((a | b) & 1ull) == 0ull) {
        a >>= 1;
        b >>= 1;
        ++s;
    }
    while ((a & 1ull) == 0ull) {
        a >>= 1;
    }
    while (b != 0ull) {
        while ((b & 1ull) == 0ull) {
            b >>= 1;
        }
        if (a > b) {
            uint64_t t = a;
            a = b;
            b = t;
        }
        b -= a;
    }
    return a << s;
}

/*!
 * \brief Fill a tile with one broadcast value.  Duplicate does not support int64, so 64-bit
 *        tiles go through the int32 scratch (consistent with the whole int64 promotion path).
 */
template <typename T>
__aicore__ inline void FillConst(AscendC::LocalTensor<T> dst, T v, int32_t n,
                                 AscendC::LocalTensor<int32_t> I)
{
    if constexpr (sizeof(T) == 8) {
        AscendC::Duplicate(I, (int32_t)v, n);
        AscendC::Cast(dst, I, AscendC::RoundMode::CAST_NONE, n);
    } else {
        AscendC::Duplicate(dst, v, n);
    }
}

// Maximum of the first 8 lanes of the reduced scratch buffer (scalar hand-off already synced).
__aicore__ inline float ReduceMax8(AscendC::LocalTensor<float> E)
{
    float m = E.GetValue(0);
    for (int32_t i = 1; i < 8; ++i) {
        float v = E.GetValue(i);
        if (v > m) {
            m = v;
        }
    }
    return m;
}

// Clear the power-of-two reduction extent of E before a producer writes only its first n lanes.
__aicore__ inline void GcdZeroTail(AscendC::LocalTensor<float> E, int32_t P, int32_t n)
{
    if (n < P) {
        AscendC::Duplicate(E, 0.0f, P);
    }
}

// Power-of-two halving max reduction of the already-initialised extent E[0..P).
__aicore__ inline void GcdHalveMax(AscendC::LocalTensor<float> E, int32_t P)
{
    int32_t m = P;
    while (m > 8) {
        int32_t h = m >> 1;
        AscendC::Max(E, E, E[h], h);
        m = h;
    }
}

__aicore__ inline int32_t StepCount(float mx, int32_t cap)
{
    int32_t bits = 0;
    float t = mx;
    while (t >= 1.0f) {
        t *= 0.5f;
        ++bits;
    }
    int32_t steps = (int32_t)(1.6f * (float)bits) + 8;
    if (steps < 8) {
        steps = 8;
    }
    if (steps > cap) {
        steps = cap;
    }
    return steps;
}

/*!
 * \brief Exact int32 Euclid for full-range int32 magnitudes (Ai, Bi hold the unsigned
 *        magnitudes of x1 and x2 in int32 storage).  lo receives the resulting magnitude.
 *
 *        Buffer aliasing: sA/sB/sC/sD/sE are each viewed as float and as int32.  The schedule
 *        is written so every instruction writes a buffer whose previous content is dead.
 */
__aicore__ inline void GcdInt32Exact(AscendC::LocalTensor<int32_t> l1, AscendC::LocalTensor<int32_t> l2,
                                     AscendC::LocalTensor<int32_t> lo, int32_t n, int32_t P, float mx,
                                     AscendC::LocalTensor<float> A, AscendC::LocalTensor<float> B,
                                     AscendC::LocalTensor<float> C, AscendC::LocalTensor<float> D,
                                     AscendC::LocalTensor<float> E, AscendC::LocalTensor<int32_t> Ai,
                                     AscendC::LocalTensor<int32_t> Bi, AscendC::LocalTensor<int32_t> Ci,
                                     AscendC::LocalTensor<int32_t> Di, AscendC::LocalTensor<int32_t> Ei)
{
    constexpr AscendC::RoundMode RNONE = AscendC::RoundMode::CAST_NONE;
    constexpr AscendC::RoundMode RRINT = AscendC::RoundMode::CAST_RINT;

    // Magnitudes: |v| == max(v, 0 - v) (int32 has no vector Abs here).  D is the zero source.
    AscendC::Duplicate(D, 0.0f, P);
    AscendC::Sub(Ci, Di, l1, n);
    AscendC::Max(Ai, l1, Ci, n);
    AscendC::Sub(Ci, Di, l2, n);
    AscendC::Max(Bi, l2, Ci, n);

    const int32_t steps = StepCount(mx, 64);
    int32_t it = 0;
    int32_t cps = 0;
    while (it < steps) {
        int32_t chunk = (cps == 0) ? 1 : ((cps == 1) ? 2 : 4);
        if (chunk > steps - it) {
            chunk = steps - it;
        }
        for (int32_t j = 0; j < chunk; ++j) {
            AscendC::Cast(C, Ai, RNONE, n);     // C = |a| as fp32   (sA still live)
            AscendC::Abs(C, C, n);
            AscendC::Cast(D, Bi, RNONE, n);     // D = |b| as fp32   (sB still live)
            AscendC::Abs(D, D, n);
            AscendC::Mins(E, D, 1.0f, n);       // E = c = [b > 0]
            AscendC::Maxs(D, D, 1.0f, n);       // D = bs = max(b, 1)
            AscendC::Div(C, C, D, n);           // C = |a| / bs
            AscendC::Adds(C, C, GCD_MAGIC, n);
            AscendC::Adds(C, C, -GCD_MAGIC, n); // C = q = rint(...)
            AscendC::Mins(C, C, GCD_QMAX, n);   // keep the cast below 2^31
            AscendC::Cast(Di, C, RRINT, n);     // Di = q as int32   (sD bs/bf dead)
            AscendC::Cast(Ci, E, RRINT, n);     // Ci = c as int32   (sC float q consumed)
            AscendC::Mul(Di, Di, Bi, n);        // q * b   (mod 2^32)
            AscendC::Sub(Ei, Bi, Ai, n);        // b - a
            AscendC::Sub(Di, Ai, Di, n);        // r = a - q*b
            AscendC::Mul(Ei, Ei, Ci, n);        // c * (b - a)
            AscendC::Add(Ai, Ai, Ei, n);        // a' = a + c*(b - a)   (sA's 'a' consumed)
            AscendC::Duplicate(E, 0.0f, n);     // E = 0
            AscendC::Sub(Ei, Ei, Di, n);        // -r
            AscendC::Max(Di, Di, Ei, n);        // |r|
            AscendC::Mul(Bi, Di, Ci, n);        // b' = c * |r|
        }
        it += chunk;
        ++cps;
        if (it >= steps) {
            break;
        }
        // Convergence probe: b == 0 everywhere means every lane has frozen.
        GcdZeroTail(E, P, n);
        AscendC::Cast(E, Bi, RNONE, n);
        AscendC::Abs(E, E, n);
        GcdHalveMax(E, P);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (ReduceMax8(E) <= 0.0f) {
            break;
        }
    }

    // lo is int32 as well: copy the magnitude verbatim (a max with itself is a plain move).
    AscendC::Max(lo, Ai, Ai, n);
}

/*!
 * \brief Elementwise gcd of the n-element tiles l1 / l2 into lo.
 */
template <typename T>
__aicore__ inline void GcdComputeTile(AscendC::LocalTensor<T> l1, AscendC::LocalTensor<T> l2,
                                      AscendC::LocalTensor<T> lo, int32_t n,
                                      AscendC::LocalTensor<float> A, AscendC::LocalTensor<float> B,
                                      AscendC::LocalTensor<float> C, AscendC::LocalTensor<float> D,
                                      AscendC::LocalTensor<float> E, AscendC::LocalTensor<int32_t> I,
                                      AscendC::LocalTensor<int32_t> Ai, AscendC::LocalTensor<int32_t> Bi,
                                      AscendC::LocalTensor<int32_t> Ci, AscendC::LocalTensor<int32_t> Di,
                                      AscendC::LocalTensor<int32_t> Ei)
{
    constexpr int32_t TSZ = (int32_t)sizeof(T);
    constexpr AscendC::RoundMode RNONE = AscendC::RoundMode::CAST_NONE;
    constexpr AscendC::RoundMode RRINT = AscendC::RoundMode::CAST_RINT;

    // Power-of-two extent used for the magnitude reduction (never above tileElems).
    int32_t P = 8;
    while (P < n) {
        P <<= 1;
    }

    // Promote both operands to fp32 magnitudes.  The reduction extent above n is cleared
    // first, so a shrunken tail tile cannot leak stale lanes into the reduction.
    GcdZeroTail(A, P, n);
    GcdZeroTail(B, P, n);
    if constexpr (TSZ == 8) {
        AscendC::Cast(I, l1, RNONE, n);
        AscendC::Cast(A, I, RNONE, n);
        AscendC::Cast(I, l2, RNONE, n);
        AscendC::Cast(B, I, RNONE, n);
    } else {
        AscendC::Cast(A, l1, RNONE, n);
        AscendC::Cast(B, l2, RNONE, n);
    }
    AscendC::Abs(A, A, n);
    AscendC::Abs(B, B, n);

    // Vector-reduced maximum magnitude over the tile.
    GcdZeroTail(E, P, n);
    AscendC::Max(E, A, B, n);
    GcdHalveMax(E, P);
    // Vector -> scalar hand-off needs an explicit barrier before the scalar read.
    AscendC::PipeBarrier<PIPE_ALL>();
    const float mx = ReduceMax8(E);

    if (mx <= 1.0f) {
        // Every pair is (0,0), (0,1), (1,0) or (1,1): gcd == max(|x1|, |x2|).
        AscendC::Max(E, A, B, n);
        if constexpr (TSZ == 8) {
            AscendC::Cast(I, E, RRINT, n);
            AscendC::Cast(lo, I, RNONE, n);
        } else {
            AscendC::Cast(lo, E, RRINT, n);
        }
        return;
    }

    if (mx > GCD_LIMIT) {
        if constexpr (TSZ == 4) {
            GcdInt32Exact(l1, l2, lo, n, P, mx, A, B, C, D, E, Ai, Bi, Ci, Di, Ei);
            return;
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int32_t i = 0; i < n; ++i) {
                T v1 = l1.GetValue(i);
                T v2 = l2.GetValue(i);
                if constexpr (TSZ == 8) {
                    uint64_t m1 = ((int64_t)v1 < 0) ? ((uint64_t)0 - (uint64_t)v1) : (uint64_t)v1;
                    uint64_t m2 = ((int64_t)v2 < 0) ? ((uint64_t)0 - (uint64_t)v2) : (uint64_t)v2;
                    lo.SetValue(i, (T)GcdU64(m1, m2));
                } else {
                    uint32_t m1 = ((int32_t)v1 < 0) ? ((uint32_t)0 - (uint32_t)v1) : (uint32_t)v1;
                    uint32_t m2 = ((int32_t)v2 < 0) ? ((uint32_t)0 - (uint32_t)v2) : (uint32_t)v2;
                    lo.SetValue(i, (T)GcdU32(m1, m2));
                }
            }
            // Scalar -> MTE3 hand-off needs an explicit barrier before the copy-out.
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
    }

    const int32_t steps = StepCount(mx, 48);
    int32_t it = 0;
    int32_t cps = 0;
    while (it < steps) {
        int32_t chunk = (cps == 0) ? 1 : ((cps == 1) ? 2 : 4);
        if (chunk > steps - it) {
            chunk = steps - it;
        }
        for (int32_t j = 0; j < chunk; ++j) {
            AscendC::Mins(E, B, 1.0f, n);       // c  = [b > 0]
            AscendC::Maxs(C, B, 1.0f, n);       // bs = max(b, 1)
            AscendC::Div(D, A, C, n);           // f  = a / bs
            AscendC::Adds(D, D, GCD_MAGIC, n);
            AscendC::Adds(D, D, -GCD_MAGIC, n); // q  = rint(f)
            AscendC::Mul(D, D, C, n);           // q * bs
            AscendC::Sub(D, A, D, n);           // r  = a - q*bs
            AscendC::Sub(C, B, A, n);           // b - a
            AscendC::Abs(D, D, n);              // |r|
            AscendC::Mul(C, C, E, n);
            AscendC::Add(A, A, C, n);           // a' = a + c*(b - a)
            AscendC::Mul(B, D, E, n);           // b' = c * |r|
        }
        it += chunk;
        ++cps;
        if (it >= steps) {
            break;
        }
        // Convergence probe: max(b) == 0 over the tile means every lane has frozen.
        GcdZeroTail(E, P, n);
        AscendC::Max(E, B, B, n);
        GcdHalveMax(E, P);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (ReduceMax8(E) <= 0.0f) {
            break;
        }
    }

    if constexpr (TSZ == 2) {
        if (mx >= 32767.5f) {
            // Wrap-around narrowing: Cast(float -> int16) saturates at 32767, but the reference
            // narrows by truncation, so a gcd magnitude of exactly 32768 must come back as
            // -32768.  gcd <= mx, so this is skipped entirely when the tile cannot reach it.
            AscendC::Adds(D, A, -32767.0f, n);
            AscendC::Maxs(D, D, 0.0f, n);
            AscendC::Mins(D, D, 1.0f, n);
            AscendC::Muls(D, D, 65536.0f, n);
            AscendC::Sub(A, A, D, n);
        }
        AscendC::Cast(lo, A, RRINT, n);
    } else if constexpr (TSZ == 8) {
        AscendC::Cast(I, A, RRINT, n);
        AscendC::Cast(lo, I, RNONE, n);
    } else {
        AscendC::Cast(lo, A, RRINT, n);
    }
}

/*!
 * \brief Expand a "repeat" row of the x2 operand into the tile (scalar fallback path).
 *
 * The run is P periods of repR elements; inside period p the source element is
 *   x2[ off2 + p * repStep ]
 * and the tile covers output positions [tt, tt + n).  Only one global read per period is
 * needed, so the scalar fill costs ~n/repR reads plus n stores.
 */
template <typename T>
__aicore__ inline void FillRepeat(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, int64_t off2,
                                  int64_t tt, int32_t n, int64_t repR, int64_t repStep)
{
    int32_t i = 0;
    while (i < n) {
        const int64_t gi = tt + (int64_t)i;
        const int64_t p = gi / repR;
        int64_t nb = (p + 1) * repR - tt; // tile index where period p ends
        int32_t iEnd = (nb < (int64_t)n) ? (int32_t)nb : n;
        if (iEnd <= i) {
            iEnd = n;
        }
        const T val = src.GetValue((int32_t)(off2 + p * repStep));
        for (int32_t k = i; k < iEnd; ++k) {
            dst.SetValue(k, val);
        }
        i = iEnd;
    }
}

/*!
 * \brief One tile of the element-based path.
 */
struct GcdTileDesc {
    int64_t s1;
    int64_t s2;
    int64_t yOff;
    int64_t t;
    int32_t n;
};

/*!
 * \brief Tile generator over the core's whole output range (row boundaries included).
 *        Advances the outer odometer when a row is exhausted.  Returns false when done.
 */
__aicore__ inline bool GcdNextTile(int64_t& row, int64_t& tt, int64_t& off1, int64_t& off2,
                                   int64_t rowEnd, int64_t gEnd, int64_t innerLen, int64_t tileElems,
                                   int64_t* idx, const int32_t* oShape, const int32_t* oS1,
                                   const int32_t* oS2, int32_t kOuter, int32_t cls1, int32_t cls2,
                                   GcdTileDesc& d)
{
    for (;;) {
        if (row > rowEnd) {
            return false;
        }
        const int64_t t1 = (row == rowEnd) ? (gEnd - row * innerLen) : innerLen;
        if (tt < t1) {
            const int64_t rem = t1 - tt;
            d.n = (int32_t)((rem < tileElems) ? rem : tileElems);
            d.s1 = (cls1 == GCD_CLS_AFFINE) ? (off1 + tt) : off1;
            d.s2 = (cls2 == GCD_CLS_AFFINE) ? (off2 + tt) : off2;
            d.t = tt;
            d.yOff = row * innerLen + tt;
            tt += (int64_t)d.n;
            return true;
        }
        ++row;
        if (row > rowEnd) {
            return false;
        }
        tt = 0;
        for (int32_t dd = kOuter - 1; dd >= 0; --dd) {
            ++idx[dd];
            off1 += (int64_t)oS1[dd];
            off2 += (int64_t)oS2[dd];
            if (idx[dd] < (int64_t)oShape[dd]) {
                break;
            }
            idx[dd] = 0;
            off1 -= (int64_t)oS1[dd] * (int64_t)oShape[dd];
            off2 -= (int64_t)oS2[dd] * (int64_t)oShape[dd];
        }
    }
}

/*!
 * \brief Copy the two source tiles of a descriptor into l1 / l2.
 */
template <typename T>
__aicore__ inline void GcdIssueIn(const GcdTileDesc& d, AscendC::LocalTensor<T> l1,
                                  AscendC::LocalTensor<T> l2, AscendC::GlobalTensor<T>& x1Gm,
                                  AscendC::GlobalTensor<T>& x2Gm,
                                  AscendC::DataCopyPadExtParams<T>& padParams,
                                  AscendC::LocalTensor<int32_t> I, bool aff1, bool aff2, bool repSlow,
                                  int64_t repR, int64_t repStep)
{
    AscendC::DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = (uint32_t)(d.n * (int32_t)sizeof(T));
    cp.srcStride = 0;
    cp.dstStride = 0;
    cp.rsv = 0;

    if (aff1) {
        AscendC::DataCopyPad(l1, x1Gm[d.s1], cp, padParams);
    } else {
        FillConst<T>(l1, x1Gm.GetValue((int32_t)d.s1), d.n, I);
    }
    if (aff2) {
        AscendC::DataCopyPad(l2, x2Gm[d.s2], cp, padParams);
    } else if (repSlow) {
        FillRepeat<T>(l2, x2Gm, d.s2, d.t, d.n, repR, repStep);
        // Scalar -> vector hand-off needs an explicit barrier.
        AscendC::PipeBarrier<PIPE_ALL>();
    } else {
        FillConst<T>(l2, x2Gm.GetValue((int32_t)d.s2), d.n, I);
    }
}

} // namespace gcd_op

using gcd_op::Unpack4;
using gcd_op::GcdComputeTile;
using gcd_op::FillConst;
using gcd_op::FillRepeat;
using gcd_op::GcdTileDesc;
using gcd_op::GcdNextTile;
using gcd_op::GcdIssueIn;

template <typename T>
__global__ __aicore__ void gcd_kernel(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    int64_t total, int64_t blockLength, int64_t tileElems,
    int64_t innerLen, int64_t kOuter, int64_t cls1, int64_t cls2,
    int64_t repR, int64_t repStep,
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,
    int64_t s10, int64_t s11, int64_t s12, int64_t s13,
    int64_t s20, int64_t s21, int64_t s22, int64_t s23)
{
    if (total <= 0 || blockLength <= 0 || tileElems <= 0 || innerLen <= 0) {
        return;
    }

    const int64_t blk = (int64_t)AscendC::GetBlockIdx();
    int64_t gStart = blk * blockLength;
    if (gStart >= total) {
        return;
    }
    int64_t gEnd = gStart + blockLength;
    if (gEnd > total) {
        gEnd = total;
    }

    const bool rep2 = (cls2 == GCD_CLS_REPEAT) && (repR > 0);
    // Slot layout of a REPEAT tile: S = align32(repR*sizeof(T))/sizeof(T) elements per period.
    int64_t slotElems = 0;
    if (rep2) {
        const int64_t bytes = repR * (int64_t)sizeof(T);
        slotElems = (((bytes + 31) / 32) * 32) / (int64_t)sizeof(T);
    }
    // The vectorised REPEAT layout needs one whole period per tile slot.
    const bool repFast = rep2 && slotElems > 0 && slotElems <= tileElems;
    if (repFast) {
        // Snap the core range out to whole periods (period boundaries are global multiples
        // of repR because innerLen is a multiple of repR).  The few overlapping elements
        // between neighbouring cores are written twice with the same value.
        gStart = (gStart / repR) * repR;
        gEnd = ((gEnd + repR - 1) / repR) * repR;
        if (gEnd > total) {
            gEnd = total;
        }
    }

    int32_t oShape[GCD_MAX_RANK];
    int32_t oS1[GCD_MAX_RANK];
    int32_t oS2[GCD_MAX_RANK];
    Unpack4(os0, os1, os2, os3, oShape);
    Unpack4(s10, s11, s12, s13, oS1);
    Unpack4(s20, s21, s22, s23, oS2);

    // Only the kOuter outer dims are meaningful; keep the remaining slots neutral.
    for (int32_t d = 0; d < GCD_MAX_RANK; ++d) {
        if (d >= (int32_t)kOuter) {
            oShape[d] = 1;
            oS1[d] = 0;
            oS2[d] = 0;
        }
    }

    int64_t tail[GCD_MAX_RANK];
    for (int32_t d = 0; d < GCD_MAX_RANK; ++d) {
        tail[d] = 1;
    }
    for (int32_t d = (int32_t)kOuter - 2; d >= 0; --d) {
        tail[d] = tail[d + 1] * (int64_t)oShape[d + 1];
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> q1;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> q2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qo;
    const uint32_t tb = (uint32_t)(tileElems * (int64_t)sizeof(T));
    pipe.InitBuffer(q1, 2, tb);
    pipe.InitBuffer(q2, 2, tb);
    pipe.InitBuffer(qo, 2, tb);

    const uint32_t fb = (uint32_t)(tileElems * 4);
    AscendC::TBuf<AscendC::TPosition::VECCALC> sA;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sC;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sD;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sE;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sI;
    pipe.InitBuffer(sA, fb);
    pipe.InitBuffer(sB, fb);
    pipe.InitBuffer(sC, fb);
    pipe.InitBuffer(sD, fb);
    pipe.InitBuffer(sE, fb);
    if constexpr (sizeof(T) == 8) {
        pipe.InitBuffer(sI, fb);
    }

    AscendC::LocalTensor<float> A = sA.Get<float>();
    AscendC::LocalTensor<float> B = sB.Get<float>();
    AscendC::LocalTensor<float> C = sC.Get<float>();
    AscendC::LocalTensor<float> D = sD.Get<float>();
    AscendC::LocalTensor<float> E = sE.Get<float>();
    AscendC::LocalTensor<int32_t> I;
    AscendC::LocalTensor<int32_t> Ai;
    AscendC::LocalTensor<int32_t> Bi;
    AscendC::LocalTensor<int32_t> Ci;
    AscendC::LocalTensor<int32_t> Di;
    AscendC::LocalTensor<int32_t> Ei;
    if constexpr (sizeof(T) == 8) {
        I = sI.Get<int32_t>();
    } else {
        I = sA.Get<int32_t>();
    }
    if constexpr (sizeof(T) == 4) {
        Ai = sA.Get<int32_t>();
        Bi = sB.Get<int32_t>();
        Ci = sC.Get<int32_t>();
        Di = sD.Get<int32_t>();
        Ei = sE.Get<int32_t>();
    }

    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<T> yGm;
    x1Gm.SetGlobalBuffer((__gm__ T*)x1, total);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2, total);
    yGm.SetGlobalBuffer((__gm__ T*)y, total);

    const int64_t rowStart = gStart / innerLen;
    const int64_t rowEnd = (gEnd - 1) / innerLen;

    int64_t idx[GCD_MAX_RANK];
    int64_t off1 = 0;
    int64_t off2 = 0;
    {
        int64_t rem = rowStart;
        for (int32_t d = 0; d < (int32_t)kOuter; ++d) {
            int64_t ix = (tail[d] > 0) ? (rem / tail[d]) : 0;
            rem -= ix * tail[d];
            idx[d] = ix;
            off1 += ix * (int64_t)oS1[d];
            off2 += ix * (int64_t)oS2[d];
        }
    }

    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = (T)0;

    const bool aff1 = (cls1 == GCD_CLS_AFFINE);
    const bool aff2 = (cls2 == GCD_CLS_AFFINE);
    const bool repSlow = rep2 && !repFast;
    // The software pipeline only pays for MTE2-fed operands; a vector-produced operand already
    // overlaps the other one's DMA wait (see the file header).
    const bool pipeOn = aff1 && aff2;

    if (repFast) {
        const int64_t perTilePeriods = tileElems / slotElems;
        for (int64_t row = rowStart; row <= rowEnd; ++row) {
            const int64_t rbase = row * innerLen;
            const int64_t t0 = (row == rowStart) ? (gStart - rbase) : 0;
            const int64_t t1 = (row == rowEnd) ? (gEnd - rbase) : innerLen;
            for (int64_t tt = t0; tt < t1;) {
                const int64_t avail = (t1 - tt) / repR;
                int64_t M = (perTilePeriods < avail) ? perTilePeriods : avail;
                if (M < 1) {
                    break;
                }
                const int32_t n = (int32_t)(M * slotElems);

                AscendC::LocalTensor<T> l1 = q1.AllocTensor<T>();
                AscendC::LocalTensor<T> l2 = q2.AllocTensor<T>();
                AscendC::LocalTensor<T> lo = qo.AllocTensor<T>();

                AscendC::DataCopyExtParams c1;
                c1.blockCount = (uint16_t)M;
                c1.blockLen = (uint32_t)(repR * (int64_t)sizeof(T));
                c1.srcStride = 0;
                c1.dstStride = 0;
                c1.rsv = 0;
                AscendC::DataCopyPad(l1, x1Gm[off1 + tt], c1, padParams);

                const int64_t p0 = tt / repR;
                for (int64_t p = 0; p < M; ++p) {
                    const T v = x2Gm.GetValue((int32_t)(off2 + (p0 + p) * repStep));
                    FillConst<T>(l2[(int32_t)(p * slotElems)], v, (int32_t)slotElems, I);
                }

                q1.EnQue(l1);
                q2.EnQue(l2);
                l1 = q1.DeQue<T>();
                l2 = q2.DeQue<T>();

                GcdComputeTile<T>(l1, l2, lo, n, A, B, C, D, E, I, Ai, Bi, Ci, Di, Ei);

                q1.FreeTensor(l1);
                q2.FreeTensor(l2);
                qo.EnQue(lo);
                lo = qo.DeQue<T>();

                AscendC::DataCopyExtParams co;
                co.blockCount = (uint16_t)M;
                co.blockLen = (uint32_t)(repR * (int64_t)sizeof(T));
                co.srcStride = 0;
                co.dstStride = 0;
                co.rsv = 0;
                AscendC::DataCopyPad(yGm[rbase + tt], lo, co);
                qo.FreeTensor(lo);

                tt += M * repR;
            }

            for (int32_t d = (int32_t)kOuter - 1; d >= 0; --d) {
                ++idx[d];
                off1 += (int64_t)oS1[d];
                off2 += (int64_t)oS2[d];
                if (idx[d] < (int64_t)oShape[d]) {
                    break;
                }
                idx[d] = 0;
                off1 -= (int64_t)oS1[d] * (int64_t)oShape[d];
                off2 -= (int64_t)oS2[d] * (int64_t)oShape[d];
            }
        }
        return;
    }

    // -----------------------------------------------------------------------
    // element-based path over the core's whole output range
    // -----------------------------------------------------------------------
    int64_t row = rowStart;
    int64_t tt = gStart - rowStart * innerLen;
    GcdTileDesc cur;
    GcdTileDesc nxt;

    bool have = GcdNextTile(row, tt, off1, off2, rowEnd, gEnd, innerLen, tileElems, idx, oShape, oS1,
                            oS2, (int32_t)kOuter, (int32_t)cls1, (int32_t)cls2, cur);

    if (pipeOn) {
        if (!have) {
            return;
        }
        {
            AscendC::LocalTensor<T> p1 = q1.AllocTensor<T>();
            AscendC::LocalTensor<T> p2 = q2.AllocTensor<T>();
            GcdIssueIn<T>(cur, p1, p2, x1Gm, x2Gm, padParams, I, aff1, aff2, repSlow, repR, repStep);
            q1.EnQue(p1);
            q2.EnQue(p2);
        }
        bool haveNext =
            GcdNextTile(row, tt, off1, off2, rowEnd, gEnd, innerLen, tileElems, idx, oShape, oS1, oS2,
                        (int32_t)kOuter, (int32_t)cls1, (int32_t)cls2, nxt);

        while (have) {
            AscendC::LocalTensor<T> l1 = q1.DeQue<T>();
            AscendC::LocalTensor<T> l2 = q2.DeQue<T>();
            if (haveNext) {
                AscendC::LocalTensor<T> n1 = q1.AllocTensor<T>();
                AscendC::LocalTensor<T> n2 = q2.AllocTensor<T>();
                GcdIssueIn<T>(nxt, n1, n2, x1Gm, x2Gm, padParams, I, aff1, aff2, repSlow, repR,
                              repStep);
                q1.EnQue(n1);
                q2.EnQue(n2);
            }
            AscendC::LocalTensor<T> lo = qo.AllocTensor<T>();
            GcdComputeTile<T>(l1, l2, lo, cur.n, A, B, C, D, E, I, Ai, Bi, Ci, Di, Ei);
            q1.FreeTensor(l1);
            q2.FreeTensor(l2);
            qo.EnQue(lo);
            lo = qo.DeQue<T>();

            AscendC::DataCopyExtParams co;
            co.blockCount = 1;
            co.blockLen = (uint32_t)(cur.n * (int32_t)sizeof(T));
            co.srcStride = 0;
            co.dstStride = 0;
            co.rsv = 0;
            AscendC::DataCopyPad(yGm[cur.yOff], lo, co);
            qo.FreeTensor(lo);

            cur = nxt;
            have = haveNext;
            haveNext = have ? GcdNextTile(row, tt, off1, off2, rowEnd, gEnd, innerLen, tileElems, idx,
                                          oShape, oS1, oS2, (int32_t)kOuter, (int32_t)cls1,
                                          (int32_t)cls2, nxt)
                            : false;
        }
        return;
    }

    while (have) {
        AscendC::LocalTensor<T> l1 = q1.AllocTensor<T>();
        AscendC::LocalTensor<T> l2 = q2.AllocTensor<T>();
        AscendC::LocalTensor<T> lo = qo.AllocTensor<T>();
        GcdIssueIn<T>(cur, l1, l2, x1Gm, x2Gm, padParams, I, aff1, aff2, repSlow, repR, repStep);
        q1.EnQue(l1);
        q2.EnQue(l2);
        l1 = q1.DeQue<T>();
        l2 = q2.DeQue<T>();

        GcdComputeTile<T>(l1, l2, lo, cur.n, A, B, C, D, E, I, Ai, Bi, Ci, Di, Ei);

        q1.FreeTensor(l1);
        q2.FreeTensor(l2);
        qo.EnQue(lo);
        lo = qo.DeQue<T>();

        AscendC::DataCopyExtParams co;
        co.blockCount = 1;
        co.blockLen = (uint32_t)(cur.n * (int32_t)sizeof(T));
        co.srcStride = 0;
        co.dstStride = 0;
        co.rsv = 0;
        AscendC::DataCopyPad(yGm[cur.yOff], lo, co);
        qo.FreeTensor(lo);

        have = GcdNextTile(row, tt, off1, off2, rowEnd, gEnd, innerLen, tileElems, idx, oShape, oS1,
                           oS2, (int32_t)kOuter, (int32_t)cls1, (int32_t)cls2, cur);
    }
}

// ---------------------------------------------------------------------------
// host tiling
// ---------------------------------------------------------------------------
std::tuple<int64_t, int64_t, int64_t> calc_gcd_tiling(int64_t total, int64_t elemSize)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    if (plat != nullptr) {
        plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    if (ubSize == 0) {
        ubSize = 196608u;
    }
    int64_t coreNum = 1;
    if (plat != nullptr) {
        coreNum = (int64_t)plat->GetCoreNumAiv();
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }

    // queues: 3 * 2 buffers; fp32 scratch: 5; int32 scratch: 1 (int64 only)
    int64_t bytesPerElem = 6 * elemSize + 20 + ((elemSize == 8) ? 4 : 0);
    int64_t budget = (int64_t)ubSize;
    if (budget > 8192) {
        budget -= 8192;
    } else {
        budget = budget / 2;
    }
    int64_t cap = budget / bytesPerElem;
    int64_t tile = 1;
    while (tile * 2 <= cap && tile < 4096) {
        tile *= 2;
    }
    if (tile < 8) {
        tile = 8;
    }

    int64_t numBlocks = coreNum;
    int64_t useful = (total + tile - 1) / tile;
    if (useful < 1) {
        useful = 1;
    }
    if (numBlocks > useful) {
        numBlocks = useful;
    }
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t blockLength = (total + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, blockLength, tile);
}

// ---------------------------------------------------------------------------
// launch wrappers
// ---------------------------------------------------------------------------
namespace gcd_op {

inline void Pack4(const int32_t* v, int64_t& a, int64_t& b, int64_t& c, int64_t& d)
{
    a = ((int64_t)v[0] & 0xFFFFFFFFLL) | (((int64_t)v[1] & 0xFFFFFFFFLL) << 32);
    b = ((int64_t)v[2] & 0xFFFFFFFFLL) | (((int64_t)v[3] & 0xFFFFFFFFLL) << 32);
    c = ((int64_t)v[4] & 0xFFFFFFFFLL) | (((int64_t)v[5] & 0xFFFFFFFFLL) << 32);
    d = ((int64_t)v[6] & 0xFFFFFFFFLL) | (((int64_t)v[7] & 0xFFFFFFFFLL) << 32);
}

} // namespace gcd_op

extern "C" {

void launch_gcd_kernel_int16(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdPlan* plan,
                             int64_t total, int64_t numBlocks, int64_t blockLength,
                             int64_t tileElems, void* stream)
{
    int64_t os0, os1, os2, os3, s10, s11, s12, s13, s20, s21, s22, s23;
    gcd_op::Pack4(plan->oShape, os0, os1, os2, os3);
    gcd_op::Pack4(plan->oS1, s10, s11, s12, s13);
    gcd_op::Pack4(plan->oS2, s20, s21, s22, s23);
    gcd_kernel<int16_t><<<numBlocks, nullptr, stream>>>(
        x1, x2, y, total, blockLength, tileElems, (int64_t)plan->innerLen,
        (int64_t)plan->kOuter, (int64_t)plan->cls1, (int64_t)plan->cls2,
        (int64_t)plan->repR, (int64_t)plan->repStep,
        os0, os1, os2, os3, s10, s11, s12, s13, s20, s21, s22, s23);
}

void launch_gcd_kernel_int32(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdPlan* plan,
                             int64_t total, int64_t numBlocks, int64_t blockLength,
                             int64_t tileElems, void* stream)
{
    int64_t os0, os1, os2, os3, s10, s11, s12, s13, s20, s21, s22, s23;
    gcd_op::Pack4(plan->oShape, os0, os1, os2, os3);
    gcd_op::Pack4(plan->oS1, s10, s11, s12, s13);
    gcd_op::Pack4(plan->oS2, s20, s21, s22, s23);
    gcd_kernel<int32_t><<<numBlocks, nullptr, stream>>>(
        x1, x2, y, total, blockLength, tileElems, (int64_t)plan->innerLen,
        (int64_t)plan->kOuter, (int64_t)plan->cls1, (int64_t)plan->cls2,
        (int64_t)plan->repR, (int64_t)plan->repStep,
        os0, os1, os2, os3, s10, s11, s12, s13, s20, s21, s22, s23);
}

void launch_gcd_kernel_int64(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdPlan* plan,
                             int64_t total, int64_t numBlocks, int64_t blockLength,
                             int64_t tileElems, void* stream)
{
    int64_t os0, os1, os2, os3, s10, s11, s12, s13, s20, s21, s22, s23;
    gcd_op::Pack4(plan->oShape, os0, os1, os2, os3);
    gcd_op::Pack4(plan->oS1, s10, s11, s12, s13);
    gcd_op::Pack4(plan->oS2, s20, s21, s22, s23);
    gcd_kernel<int64_t><<<numBlocks, nullptr, stream>>>(
        x1, x2, y, total, blockLength, tileElems, (int64_t)plan->innerLen,
        (int64_t)plan->kOuter, (int64_t)plan->cls1, (int64_t)plan->cls2,
        (int64_t)plan->repR, (int64_t)plan->repStep,
        os0, os1, os2, os3, s10, s11, s12, s13, s20, s21, s22, s23);
}

} // extern "C"
