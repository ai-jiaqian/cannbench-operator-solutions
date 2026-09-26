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
 * \file cummin_kernel.cpp
 * \brief Cummin (inclusive prefix minimum + argmin index) device kernels, tiling and launch wrappers.
 *
 * Semantics, aligned with torch.cummin:
 *   view the input as [outer, axis, inner] with the scan along `axis`; for each independent
 *   (outer, inner) lane produce the inclusive prefix minimum and the position that produced it.
 *
 *   take = isNaN(cur) || (cur <= run)
 *     - equal finite values keep the LAST occurrence (the index advances),
 *     - once a NaN has been seen the running value stays NaN,
 *     - a fully NaN prefix keeps advancing its index.
 *
 * NaN handling uses an ordered key instead of a self-compare (self comparisons get folded by the
 * compiler on this target):
 *     key(x) = (x > -inf) ? x : -inf
 *   key(NaN) = -inf, key(-inf) = -inf, key(+inf) = +inf, key(finite) = x.  Since a running key of
 *   -inf compares <= every other key,
 *     take    = (key(cur) <= runKey)
 *     runKey  = Min(runKey, key(cur))
 *   reproduces the rule above exactly.
 *
 * Vector-path constraints honoured here (measured on this toolchain by prior work):
 *   - EVERY vector op count must be a whole number of vector repeats (64 lanes for 32-bit elements).
 *     A smaller count silently does nothing, which is how an earlier reduction lost its last steps.
 *   - the 16-bit Select intrinsic is unusable and int32 Select is unreliable, so the running value
 *     is carried in the fp32 comparison domain and the running index in fp32 (indices < 2^24 are
 *     exact in fp32).  The index is narrowed to int32 -> int64 on output.
 *   - Select must use SELMODE::VSEL_TENSOR_TENSOR_MODE (the CMPMASK_SPR path does not advance the
 *     mask pointer across repeats).
 *   - Select must not have dst aliasing src0/src1, so the scan state ping-pongs between two buffers
 *     indexed by the parity of the axis position; no copy-back is needed.
 *   - several independent TBuf objects can hand out overlapping UB addresses on this part, so every
 *     scratch buffer comes from a single TBuf arena addressed with GetWithOffset at 32B aligned
 *     byte offsets.  Every vector op's sub-tensor view therefore starts on a 32B boundary, and every
 *     vector op needs a 32B aligned base address (an unaligned one raises a VEC alignment fault).
 *   - EVERY vector -> scalar edge must be crossed with PipeBarrier<PIPE_ALL>().  With
 *     PipeBarrier<PIPE_V>() the scalar readback of a vector result returned stale UB content and the
 *     gate below silently approved dirty windows (11/20 cases wrong).
 *
 * ------------------------------------------------------------------------------------------------
 * Row program (inner == 1).  A per-element scalar scan costs ~24 cycles/element if the UB load is
 * issued one at a time (and ~2.2 cycles/element once the loop is unrolled and the loads are batched),
 * so the scan is driven by "window minimum gating".
 *
 *   A window of W lanes (a power of two) is reduced to its eight residue-class minima:
 *       level A (W = 512):
 *           Min(p1, src,     src[256], 256)     lane i < 256 covers {i, i+256}
 *           Min(p2, p1,      p1[128],  128)     lane i < 128 covers {i, i+128, i+256, i+384}
 *           Min(p1, p2,      p2[64],    64)     lane i <  64 covers {i, i+64, ..., i+448}
 *           Min(p2, p1,      p1[8],     64)     2 classes of 64 per lane
 *           Min(p1, p2,      p2[16],    64)     4 classes of 64 per lane
 *           Min(p2, p1,      p1[32],    64)     8 classes of 64 per lane
 *       level B (W = 64):
 *           Min(p1, src, src[8],  64);  Min(p2, p1, p1[16], 64);  Min(p1, p2, p2[32], 64)
 *   and min(the eight scalar lanes) is the window minimum.  The first three level-A steps are a
 *   halving tree (cheap: 4+2+1 repeats instead of 6 x 8); the last three are the residue-class shift
 *   chain, whose reads run a few lanes past the live window, which can only make the tested minimum
 *   smaller and is therefore conservative.
 *
 *   The ordered key must be recomputed for EVERY window that is tested:  a fill chain can start in a
 *   tail window (w < 512) where no level-A ever materialised a 512-lane key, so reusing a key from
 *   "the last big window" would test stale lanes (that bug made every float row case wrong while the
 *   int32 cases, which reduce the raw values, stayed correct).
 *
 *   A window whose minimum is STRICTLY greater than the running key cannot contain a single element
 *   that takes, so the whole window is a verbatim copy of the current running value and running
 *   index -- two Duplicates instead of W scalar steps.  Otherwise the window is decomposed: re-test
 *   at 64-lane granularity, and only the 64-lane blocks that still carry a new minimum fall back to
 *   the scalar loop.
 *
 *   The gating test is STRICT so a tie (a repeat of the running minimum) is never handled by the
 *   vector path, because on a tie the running index must advance to the tie position.
 *
 *   skipA: the level-A (512) test is redundant at a 64-lane step that follows a level-B FILL.  A
 *   passed level-B test proves min of [sb, sb+64) > runKey, while the level-A failure at the window
 *   start proved some element <= runKey inside that 512-window; that element sits at p >= sb and
 *   p < sb+512 <= sb+576, and the level-A reduction at sb covers exactly the class minima of
 *   [sb, sb+512), which contains p.  Hence rm <= x[p] <= runKey and level A cannot pass -- skipping
 *   it is never a loss, and it removes one barrier + one reduction + 8 scalar reads per step.
 *
 *   All-equal runs (every element equal to the running minimum) can never satisfy the strict test,
 *   so they would degrade to one level-A round + one level-B round + a scalar block per 64 elements.
 *   The scalar fallback therefore arms a cheap probe: a block whose LAST element took while the
 *   running key did not move is a candidate (it needs no per-element bookkeeping -- it compares runI
 *   with the block's last position and runKey with its pre-block value).  eqMode then tests the next
 *   power-of-two window with a Min AND a Max reduction behind a single barrier; when min == max ==
 *   runKey every element of that window equals the running key, so it is filled with
 *   Duplicate(value, runV) plus Adds(ramp, base) for the index and runI becomes the last position.
 *   A wrong probe only costs one failed test and a passing test is exact, so the path is always
 *   sound.  Random data almost never arms it, while the all-zero case (case 15) stops paying a
 *   useless pair of gate rounds per 64 elements (803us -> 355us measured).
 *
 * Column program (inner > 1).  The innermost dimension supplies the independent lanes and the axis
 * is walked in BATCHES of `nr` rows: the nr loads are issued together into one queue tensor (row r
 * at offset r*Tg), the batch is consumed by nr vector passes, the nr stores are issued together, and
 * the next batch's loads are issued before the current batch is consumed so MTE2 latency overlaps.
 * Batching divides the per-row queue hand-off cost by nr; the batch size is picked at run time from
 * the UB budget (measured optimum is 4 rows - 8 rows measured worse).
 * ================================================================================================ */

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "cummin_launch.h"

namespace cbmn {

__aicore__ inline int64_t AlignUpB(int64_t b)
{
    return (b + 31) / 32 * 32;
}

__aicore__ inline int64_t AlignUp64(int64_t v)
{
    return (v + 63) / 64 * 64;
}

__aicore__ inline float FFromBits(uint32_t u)
{
    union {
        uint32_t u;
        float f;
    } c;
    c.u = u;
    return c.f;
}

__aicore__ inline float NegInfF()
{
    return FFromBits(0xff800000u);
}

__aicore__ inline float PosInfF()
{
    return FFromBits(0x7f800000u);
}

constexpr int64_t kRowChunk = 2048;
constexpr int64_t kColTgMax = 1024;
constexpr int64_t kColBatch = 4;

} // namespace cbmn

/* ================================================================================================ *
 * mode 0 : inner == 1, one contiguous chain per row
 * ================================================================================================ */
template <typename T>
__global__ __aicore__ void cummin_row_kernel(GM_ADDR xp, GM_ADDR vp, GM_ADDR ip, int64_t rows,
                                             int64_t len, int64_t rowsPerBlock, int64_t numBlocks,
                                             int64_t chunk)
{
    constexpr bool k16 = (sizeof(T) == 2);
    constexpr bool kInt = std::is_same<T, int32_t>::value;
    using CT = typename std::conditional<k16, float, T>::type;

    int64_t bid = (int64_t)AscendC::GetBlockIdx();
    if (bid >= numBlocks) {
        return;
    }
    int64_t r0 = bid * rowsPerBlock;
    int64_t r1 = r0 + rowsPerBlock;
    if (r1 > rows) {
        r1 = rows;
    }
    if (r0 >= r1) {
        return;
    }

    // Keep 768 elements of slack so a 512-lane level-A window on a short chunk can never address
    // past the arena.
    const int64_t ne = chunk + 768;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arena;

    int64_t off = 0;
    int64_t oIn = off;
    off += cbmn::AlignUpB(ne * (int64_t)sizeof(T));
    int64_t oCur = off;
    off += cbmn::AlignUpB(ne * 4);
    int64_t oKey = off;
    off += cbmn::AlignUpB(ne * 4);
    int64_t oOutF = off;
    off += cbmn::AlignUpB(ne * 4);
    int64_t oOutIdx = off;
    off += cbmn::AlignUpB(ne * 4);
    int64_t oR1 = off;
    off += cbmn::AlignUpB(ne * 4);
    int64_t oR2 = off;
    off += cbmn::AlignUpB(ne * 4);
    int64_t oR3 = off;
    off += cbmn::AlignUpB(ne * 4);
    int64_t oR4 = off;
    off += cbmn::AlignUpB(ne * 4);
    int64_t oOutI = off;
    off += cbmn::AlignUpB(chunk * 4);
    int64_t oOut64 = off;
    off += cbmn::AlignUpB(chunk * 8);
    int64_t oOutV = off;
    off += cbmn::AlignUpB(chunk * (int64_t)sizeof(T));
    int64_t oNeg = off;
    off += cbmn::AlignUpB(512 * 4);
    int64_t oRamp = off;
    off += cbmn::AlignUpB(512 * 4);
    int64_t oMa = off;
    off += 512;
    pipe.InitBuffer(arena, (uint32_t)off);

    AscendC::LocalTensor<T> inRaw = arena.GetWithOffset<T>((int32_t)ne, (uint32_t)oIn);
    AscendC::LocalTensor<CT> curF = arena.GetWithOffset<CT>((int32_t)ne, (uint32_t)oCur);
    AscendC::LocalTensor<CT> keyV = arena.GetWithOffset<CT>((int32_t)ne, (uint32_t)oKey);
    AscendC::LocalTensor<CT> outF = arena.GetWithOffset<CT>((int32_t)ne, (uint32_t)oOutF);
    AscendC::LocalTensor<float> outIdxF = arena.GetWithOffset<float>((int32_t)ne, (uint32_t)oOutIdx);
    AscendC::LocalTensor<CT> rA = arena.GetWithOffset<CT>((int32_t)ne, (uint32_t)oR1);
    AscendC::LocalTensor<CT> rB = arena.GetWithOffset<CT>((int32_t)ne, (uint32_t)oR2);
    AscendC::LocalTensor<CT> rC = arena.GetWithOffset<CT>((int32_t)ne, (uint32_t)oR3);
    AscendC::LocalTensor<CT> rD = arena.GetWithOffset<CT>((int32_t)ne, (uint32_t)oR4);
    AscendC::LocalTensor<int32_t> outI32 = arena.GetWithOffset<int32_t>((int32_t)chunk, (uint32_t)oOutI);
    AscendC::LocalTensor<int64_t> outI64 = arena.GetWithOffset<int64_t>((int32_t)chunk, (uint32_t)oOut64);
    AscendC::LocalTensor<T> outV = arena.GetWithOffset<T>((int32_t)chunk, (uint32_t)oOutV);
    AscendC::LocalTensor<CT> negT = arena.GetWithOffset<CT>((int32_t)512, (uint32_t)oNeg);
    AscendC::LocalTensor<float> ramp = arena.GetWithOffset<float>((int32_t)512, (uint32_t)oRamp);
    AscendC::LocalTensor<uint8_t> ma = arena.GetWithOffset<uint8_t>((int32_t)512, (uint32_t)oMa);

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> vGm;
    AscendC::GlobalTensor<int64_t> iGm;
    xGm.SetGlobalBuffer((__gm__ T*)xp);
    vGm.SetGlobalBuffer((__gm__ T*)vp);
    iGm.SetGlobalBuffer((__gm__ int64_t*)ip);

    const float fNeg = cbmn::NegInfF();
    const float fPos = cbmn::PosInfF();

    // -inf sentinel for the ordered-key transform (non-integer dtypes only).
    if constexpr (!kInt) {
        AscendC::Duplicate(negT, (CT)fNeg, (uint32_t)512);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __ubuf__ CT* outFP = (__ubuf__ CT*)outF.GetPhyAddr();
    __ubuf__ float* outIdxP = (__ubuf__ float*)outIdxF.GetPhyAddr();

    AscendC::LocalTensor<CT> curV;
    if constexpr (k16) {
        curV = curF;
    } else {
        curV = inRaw;
    }
    auto negView = negT;
    bool rampReady = false;

    for (int64_t row = r0; row < r1; ++row) {
        int64_t base = row * len;
        CT runKey;
        CT runV;
        int32_t runI = 0;
        if constexpr (kInt) {
            runKey = (CT)2147483647;
            runV = (CT)0;
        } else {
            runKey = (CT)fPos;
            runV = (CT)0;
        }

        for (int64_t c0 = 0; c0 < len; c0 += chunk) {
            int64_t rem = len - c0;
            if (rem > chunk) {
                rem = chunk;
            }
            uint32_t cntA = (uint32_t)cbmn::AlignUp64(rem);
            if ((int64_t)cntA > chunk) {
                cntA = (uint32_t)chunk;
            }

            AscendC::DataCopyExtParams cpIn{(uint16_t)1, (uint32_t)(rem * (int64_t)sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> ppIn{false, 0, 0, 0};
            AscendC::DataCopyPad(inRaw, xGm[base + c0], cpIn, ppIn);
            AscendC::PipeBarrier<PIPE_ALL>();

            if constexpr (k16) {
                AscendC::Cast(curF, inRaw, AscendC::RoundMode::CAST_NONE, cntA);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            __ubuf__ CT* curPtr;
            if constexpr (k16) {
                curPtr = (__ubuf__ CT*)curF.GetPhyAddr();
            } else {
                curPtr = (__ubuf__ CT*)inRaw.GetPhyAddr();
            }

            bool eqMode = false;
            bool skipA = false;
            int64_t sb = 0;
            while (sb < rem) {
                int64_t w = rem - sb;

                if (eqMode) {
                    // ---- all-equal window: one barrier, one Min chain, one Max chain ----
                    int64_t W = 512;
                    while (W > 64 && W > w) {
                        W >>= 1;
                    }
                    if (W >= 64 && W <= w && (kInt || runKey > (CT)fNeg)) {
                        const uint32_t WW = (uint32_t)W;
                        auto cq = curV[sb];
                        AscendC::LocalTensor<CT> sq;
                        if constexpr (kInt) {
                            sq = cq;
                        } else {
                            AscendC::Compare(ma, cq, negView, AscendC::CMPMODE::GT, WW);
                            AscendC::Select(keyV[sb], ma, cq, negView, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, WW);
                            sq = keyV[sb];
                        }
                        AscendC::LocalTensor<CT> tmn = rA[sb];
                        AscendC::LocalTensor<CT> tmx = rC[sb];
                        AscendC::LocalTensor<CT> omn = rB[sb];
                        AscendC::LocalTensor<CT> omx = rD[sb];
                        AscendC::Min(tmn, sq, sq[8], WW);
                        AscendC::Max(tmx, sq, sq[8], WW);
                        for (int64_t s = 16; s < W; s <<= 1) {
                            AscendC::Min(omn, tmn, tmn[(int32_t)s], WW);
                            AscendC::Max(omx, tmx, tmx[(int32_t)s], WW);
                            auto sw1 = tmn;
                            tmn = omn;
                            omn = sw1;
                            auto sw2 = tmx;
                            tmx = omx;
                            omx = sw2;
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                        __ubuf__ CT* mp = (__ubuf__ CT*)tmn.GetPhyAddr();
                        __ubuf__ CT* xp = (__ubuf__ CT*)tmx.GetPhyAddr();
                        CT rmn = mp[0];
                        CT rmx = xp[0];
                        for (int q = 1; q < 8; ++q) {
                            CT t1 = mp[q];
                            if (t1 < rmn) {
                                rmn = t1;
                            }
                            CT t2 = xp[q];
                            if (t2 > rmx) {
                                rmx = t2;
                            }
                        }
                        if (rmn == rmx && rmn == runKey) {
                            if (!rampReady) {
                                __ubuf__ float* rp2 = (__ubuf__ float*)ramp.GetPhyAddr();
                                for (int i = 0; i < 512; ++i) {
                                    rp2[i] = (float)i;
                                }
                                rampReady = true;
                                AscendC::PipeBarrier<PIPE_ALL>();
                            }
                            AscendC::Duplicate(outF[sb], runV, WW);
                            AscendC::Adds(outIdxF[sb], ramp, (float)(c0 + sb), WW);
                            runI = (int32_t)(c0 + sb + W - 1);
                            sb += W;
                            continue;
                        }
                    }
                    eqMode = false;
                }

                if (!skipA && w >= 512) {
                    // ---- level A: a 512-lane window ----
                    const int64_t WA = 512;
                    auto cw = curV[sb];
                    auto kw = keyV[sb];
                    AscendC::LocalTensor<CT> src = cw;
                    if constexpr (!kInt) {
                        AscendC::Compare(ma, cw, negView, AscendC::CMPMODE::GT, (uint32_t)WA);
                        AscendC::Select(kw, ma, cw, negView, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, (uint32_t)WA);
                        src = kw;
                    }
                    AscendC::LocalTensor<CT> p1 = rA[sb];
                    AscendC::LocalTensor<CT> p2 = rB[sb];
                    AscendC::Min(p1, src, src[(int32_t)256], (uint32_t)256);
                    AscendC::Min(p2, p1, p1[(int32_t)128], (uint32_t)128);
                    AscendC::Min(p1, p2, p2[(int32_t)64], (uint32_t)64);
                    AscendC::Min(p2, p1, p1[8], (uint32_t)64);
                    AscendC::Min(p1, p2, p2[16], (uint32_t)64);
                    AscendC::Min(p2, p1, p1[32], (uint32_t)64);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    __ubuf__ CT* rp = (__ubuf__ CT*)p2.GetPhyAddr();
                    CT rm = rp[0];
                    for (int q = 1; q < 8; ++q) {
                        CT t = rp[q];
                        if (t < rm) {
                            rm = t;
                        }
                    }
                    if (rm > runKey) {
                        AscendC::Duplicate(outF[sb], runV, (uint32_t)WA);
                        AscendC::Duplicate(outIdxF[sb], (float)runI, (uint32_t)WA);
                        sb += WA;
                        skipA = false;
                        continue;
                    }
                }

                if (w >= 64) {
                    // ---- level B: a 64-lane window ----
                    const int64_t WB = 64;
                    auto cw = curV[sb];
                    auto kw = keyV[sb];
                    AscendC::LocalTensor<CT> src = cw;
                    if constexpr (!kInt) {
                        AscendC::Compare(ma, cw, negView, AscendC::CMPMODE::GT, (uint32_t)WB);
                        AscendC::Select(kw, ma, cw, negView, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                        (uint32_t)WB);
                        src = kw;
                    }
                    AscendC::LocalTensor<CT> p1 = rA[sb];
                    AscendC::LocalTensor<CT> p2 = rB[sb];
                    AscendC::Min(p1, src, src[8], (uint32_t)WB);
                    AscendC::Min(p2, p1, p1[16], (uint32_t)WB);
                    AscendC::Min(p1, p2, p2[32], (uint32_t)WB);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    __ubuf__ CT* rp = (__ubuf__ CT*)p1.GetPhyAddr();
                    CT rm = rp[0];
                    for (int q = 1; q < 8; ++q) {
                        CT t = rp[q];
                        if (t < rm) {
                            rm = t;
                        }
                    }
                    if (rm > runKey) {
                        AscendC::Duplicate(outF[sb], runV, (uint32_t)WB);
                        AscendC::Duplicate(outIdxF[sb], (float)runI, (uint32_t)WB);
                        sb += WB;
                        // the new minimum of the enclosing 512-window must lie ahead of sb
                        skipA = true;
                        continue;
                    }
                }

                // ---- scalar fallback for the dirty 64-lane block ----
                {
                    uint32_t n = (uint32_t)((w < 64) ? w : 64);
                    const float negI = fNeg;
                    const CT keyBefore = runKey;
                    uint32_t j = 0;
                    for (; j + 8 <= n; j += 8) {
                        CT v0 = curPtr[sb + j + 0];
                        CT v1 = curPtr[sb + j + 1];
                        CT v2 = curPtr[sb + j + 2];
                        CT v3 = curPtr[sb + j + 3];
                        CT v4 = curPtr[sb + j + 4];
                        CT v5 = curPtr[sb + j + 5];
                        CT v6 = curPtr[sb + j + 6];
                        CT v7 = curPtr[sb + j + 7];
#define CUMROW_STEP(K, VV)                                                                        \
    {                                                                                             \
        CT cc_ = (VV);                                                                            \
        if (!(cc_ > runKey)) {                                                                    \
            if constexpr (kInt) {                                                                 \
                runKey = cc_;                                                                     \
            } else {                                                                              \
                runKey = (cc_ > negI) ? cc_ : negI;                                               \
            }                                                                                     \
            runV = cc_;                                                                           \
            runI = (int32_t)(c0 + sb + (int64_t)j + (int64_t)(K));                                \
        }                                                                                         \
        outFP[sb + j + (K)] = runV;                                                               \
        outIdxP[sb + j + (K)] = (float)runI;                                                      \
    }
                        CUMROW_STEP(0, v0)
                        CUMROW_STEP(1, v1)
                        CUMROW_STEP(2, v2)
                        CUMROW_STEP(3, v3)
                        CUMROW_STEP(4, v4)
                        CUMROW_STEP(5, v5)
                        CUMROW_STEP(6, v6)
                        CUMROW_STEP(7, v7)
#undef CUMROW_STEP
                    }
                    for (; j < n; ++j) {
                        CT cc_ = curPtr[sb + j];
                        if (!(cc_ > runKey)) {
                            if constexpr (kInt) {
                                runKey = cc_;
                            } else {
                                runKey = (cc_ > negI) ? cc_ : negI;
                            }
                            runV = cc_;
                            runI = (int32_t)(c0 + sb + (int64_t)j);
                        }
                        outFP[sb + j] = runV;
                        outIdxP[sb + j] = (float)runI;
                    }
                    sb += (int64_t)n;
                    skipA = false;
                    // Cheap probe: the block's LAST element took and the key did not move, so every
                    // element of the block may be exactly the running key.  Arm eqMode (a wrong
                    // probe merely costs one failed test).
                    if (runKey == keyBefore && runI == (int32_t)(c0 + sb - 1)) {
                        eqMode = true;
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }

            AscendC::PipeBarrier<PIPE_ALL>();
            if constexpr (k16) {
                AscendC::Cast(outV, outF, AscendC::RoundMode::CAST_RINT, cntA);
            }
            AscendC::Cast(outI32, outIdxF, AscendC::RoundMode::CAST_RINT, cntA);
            AscendC::Cast(outI64, outI32, AscendC::RoundMode::CAST_NONE, cntA);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopyExtParams cpV{(uint16_t)1, (uint32_t)(rem * (int64_t)sizeof(T)), 0, 0, 0};
            if constexpr (k16) {
                AscendC::DataCopyPad(vGm[base + c0], outV, cpV);
            } else {
                AscendC::DataCopyPad(vGm[base + c0], outF, cpV);
            }
            AscendC::DataCopyExtParams cpI{(uint16_t)1, (uint32_t)(rem * 8), 0, 0, 0};
            AscendC::DataCopyPad(iGm[base + c0], outI64, cpI);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
}

/* ================================================================================================ *
 * mode 1 : inner > 1, lanes vectorised, axis walked in batches of `nr` rows
 * ================================================================================================ */
template <typename T>
__global__ __aicore__ void cummin_col_kernel(GM_ADDR xp, GM_ADDR vp, GM_ADDR ip, int64_t outer,
                                             int64_t axis, int64_t inner, int64_t nGroups, int64_t Tg,
                                             int64_t numBlocks)
{
    constexpr bool k16 = (sizeof(T) == 2);
    constexpr bool kInt = std::is_same<T, int32_t>::value;
    using CT = typename std::conditional<k16, float, T>::type;

    int64_t bid = (int64_t)AscendC::GetBlockIdx();
    if (bid >= numBlocks) {
        return;
    }
    int64_t totalTasks = outer * nGroups;
    int64_t tpb = (totalTasks + numBlocks - 1) / numBlocks;
    int64_t t0 = bid * tpb;
    int64_t t1 = t0 + tpb;
    if (t1 > totalTasks) {
        t1 = totalTasks;
    }
    if (t0 >= t1) {
        return;
    }

    const uint32_t uc = (uint32_t)Tg;
    const int64_t rowBytesT = Tg * (int64_t)sizeof(T);
    const int64_t rowBytes64 = Tg * 8;
    // Pick the batch size from the UB budget: three queues, each two deep.
    int64_t perRow = 2 * (rowBytesT + rowBytesT + rowBytes64);
    int64_t nr = cbmn::kColBatch;
    while (nr > 1 && perRow * nr > 120000) {
        nr >>= 1;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> arena;

    int64_t off = 0;
    int64_t oV0 = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oV1 = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oK0 = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oK1 = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oI0 = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oI1 = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oKey = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oCur = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oNeg = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oIdxF = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oOI32 = off;
    off += cbmn::AlignUpB(Tg * 4);
    int64_t oMa = off;
    off += 256;
    int64_t oMb = off;
    off += 256;
    pipe.InitBuffer(arena, (uint32_t)off);

    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outVQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outIQ;
    pipe.InitBuffer(inQ, 2, (uint32_t)(nr * rowBytesT));
    pipe.InitBuffer(outVQ, 2, (uint32_t)(nr * rowBytesT));
    pipe.InitBuffer(outIQ, 2, (uint32_t)(nr * rowBytes64));

    AscendC::LocalTensor<CT> v0 = arena.GetWithOffset<CT>((int32_t)Tg, (uint32_t)oV0);
    AscendC::LocalTensor<CT> v1 = arena.GetWithOffset<CT>((int32_t)Tg, (uint32_t)oV1);
    AscendC::LocalTensor<CT> k0 = arena.GetWithOffset<CT>((int32_t)Tg, (uint32_t)oK0);
    AscendC::LocalTensor<CT> k1 = arena.GetWithOffset<CT>((int32_t)Tg, (uint32_t)oK1);
    AscendC::LocalTensor<float> r0 = arena.GetWithOffset<float>((int32_t)Tg, (uint32_t)oI0);
    AscendC::LocalTensor<float> r1 = arena.GetWithOffset<float>((int32_t)Tg, (uint32_t)oI1);
    AscendC::LocalTensor<CT> keyC = arena.GetWithOffset<CT>((int32_t)Tg, (uint32_t)oKey);
    AscendC::LocalTensor<CT> curF = arena.GetWithOffset<CT>((int32_t)Tg, (uint32_t)oCur);
    AscendC::LocalTensor<CT> negT = arena.GetWithOffset<CT>((int32_t)Tg, (uint32_t)oNeg);
    AscendC::LocalTensor<float> idxF = arena.GetWithOffset<float>((int32_t)Tg, (uint32_t)oIdxF);
    AscendC::LocalTensor<int32_t> oi32 = arena.GetWithOffset<int32_t>((int32_t)Tg, (uint32_t)oOI32);
    AscendC::LocalTensor<uint8_t> ma = arena.GetWithOffset<uint8_t>((int32_t)256, (uint32_t)oMa);
    AscendC::LocalTensor<uint8_t> mb = arena.GetWithOffset<uint8_t>((int32_t)256, (uint32_t)oMb);

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> vGm;
    AscendC::GlobalTensor<int64_t> iGm;
    xGm.SetGlobalBuffer((__gm__ T*)xp);
    vGm.SetGlobalBuffer((__gm__ T*)vp);
    iGm.SetGlobalBuffer((__gm__ int64_t*)ip);

    if constexpr (kInt) {
        AscendC::Duplicate(negT, (int32_t)0, uc);
    } else {
        AscendC::Duplicate(negT, (CT)cbmn::NegInfF(), uc);
    }

    for (int64_t task = t0; task < t1; ++task) {
        int64_t o = task / nGroups;
        int64_t g = task - o * nGroups;
        int64_t g0 = g * Tg;
        int64_t left = inner - g0;
        if (left > Tg) {
            left = Tg;
        }
        uint32_t storeCnt = (uint32_t)left;
        int64_t base = o * axis * inner + g0;
        const uint32_t storeBytesV = (uint32_t)(storeCnt * (int64_t)sizeof(T));
        const uint32_t storeBytesI = (uint32_t)(storeCnt * 8);

        AscendC::DataCopyExtParams cpIn{(uint16_t)1, (uint32_t)(left * (int64_t)sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> ppIn{false, 0, 0, 0};
        AscendC::DataCopyExtParams cpV{(uint16_t)1, storeBytesV, 0, 0, 0};
        AscendC::DataCopyExtParams cpI{(uint16_t)1, storeBytesI, 0, 0, 0};

        if constexpr (kInt) {
            AscendC::Duplicate(v0, (int32_t)2147483647, uc);
        } else {
            AscendC::Duplicate(v0, (CT)0, uc);
            AscendC::Duplicate(k0, (CT)cbmn::PosInfF(), uc);
        }
        AscendC::Duplicate(r0, 0.0f, uc);

        // prologue: queue the first batch's loads so MTE2 overlaps the first scan
        {
            int64_t nrc = axis < nr ? axis : nr;
            auto inP = inQ.AllocTensor<T>();
            for (int64_t r = 0; r < nrc; ++r) {
                AscendC::DataCopyPad(inP[(int32_t)(r * Tg)], xGm[base + r * inner], cpIn, ppIn);
            }
            inQ.EnQue(inP);
        }

        for (int64_t ax0 = 0; ax0 < axis; ax0 += nr) {
            int64_t nrc = axis - ax0;
            if (nrc > nr) {
                nrc = nr;
            }
            if (ax0 + nr < axis) {
                int64_t nrn = axis - (ax0 + nr);
                if (nrn > nr) {
                    nrn = nr;
                }
                auto inN = inQ.AllocTensor<T>();
                for (int64_t r = 0; r < nrn; ++r) {
                    AscendC::DataCopyPad(inN[(int32_t)(r * Tg)], xGm[base + (ax0 + nr + r) * inner], cpIn, ppIn);
                }
                inQ.EnQue(inN);
            }

            auto xin = inQ.DeQue<T>();
            auto ov = outVQ.AllocTensor<T>();
            auto oi64 = outIQ.AllocTensor<int64_t>();

            for (int64_t r = 0; r < nrc; ++r) {
                int64_t ax = ax0 + r;
                int64_t p = ax & 1;
                AscendC::LocalTensor<CT> vP = p ? v1 : v0;
                AscendC::LocalTensor<CT> vQ = p ? v0 : v1;
                AscendC::LocalTensor<CT> kP = p ? k1 : k0;
                AscendC::LocalTensor<CT> kQ = p ? k0 : k1;
                AscendC::LocalTensor<float> rP = p ? r1 : r0;
                AscendC::LocalTensor<float> rQ = p ? r0 : r1;

                auto xrow = xin[(int32_t)(r * Tg)];
                AscendC::LocalTensor<CT> cur;
                if constexpr (k16) {
                    AscendC::Cast(curF, xrow, AscendC::RoundMode::CAST_NONE, uc);
                    cur = curF;
                } else {
                    cur = xrow;
                }

                AscendC::Duplicate(idxF, (float)ax, uc);
                if constexpr (kInt) {
                    AscendC::Compare(mb, cur, vP, AscendC::CMPMODE::LE, uc);
                    AscendC::Min(vQ, vP, cur, uc);
                    AscendC::Select(rQ, mb, idxF, rP, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, uc);
                } else {
                    AscendC::Compare(ma, cur, negT, AscendC::CMPMODE::GT, uc);
                    AscendC::Select(keyC, ma, cur, negT, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, uc);
                    AscendC::Compare(mb, keyC, kP, AscendC::CMPMODE::LE, uc);
                    AscendC::Min(kQ, kP, keyC, uc);
                    AscendC::Select(vQ, mb, cur, vP, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, uc);
                    AscendC::Select(rQ, mb, idxF, rP, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, uc);
                }

                auto ovr = ov[(int32_t)(r * Tg)];
                if constexpr (k16) {
                    AscendC::Cast(ovr, vQ, AscendC::RoundMode::CAST_RINT, uc);
                } else if constexpr (kInt) {
                    AscendC::Adds(ovr, vQ, (int32_t)0, uc);
                } else {
                    AscendC::Muls(ovr, vQ, 1.0f, uc);
                }

                AscendC::Cast(oi32, rQ, AscendC::RoundMode::CAST_RINT, uc);
                AscendC::Cast(oi64[(int32_t)(r * Tg)], oi32, AscendC::RoundMode::CAST_NONE, uc);
            }
            inQ.FreeTensor(xin);
            outVQ.EnQue(ov);
            outIQ.EnQue(oi64);

            auto ovd = outVQ.DeQue<T>();
            auto oid = outIQ.DeQue<int64_t>();
            for (int64_t r = 0; r < nrc; ++r) {
                AscendC::DataCopyPad(vGm[base + (ax0 + r) * inner], ovd[(int32_t)(r * Tg)], cpV);
                AscendC::DataCopyPad(iGm[base + (ax0 + r) * inner], oid[(int32_t)(r * Tg)], cpI);
            }
            outVQ.FreeTensor(ovd);
            outIQ.FreeTensor(oid);
        }
    }
}

/* ================================================================================================ *
 * tiling
 * ================================================================================================ */
CumminTiling calc_cummin_tiling(int64_t outer, int64_t axis, int64_t inner, int64_t elemBytes)
{
    (void)elemBytes;
    int64_t coreNum = 1;
    auto* plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        int64_t cn = (int64_t)plat->GetCoreNumAiv();
        if (cn > 0) {
            coreNum = cn;
        }
    }
    CumminTiling t;
    t.mode = 0;
    t.numBlocks = 1;
    t.rowsPerBlock = 0;
    t.chunk = 0;
    t.Tg = 0;
    t.nGroups = 0;

    if (inner == 1) {
        int64_t rows = outer;
        int64_t nb = coreNum;
        if (nb > rows) {
            nb = rows;
        }
        if (nb < 1) {
            nb = 1;
        }
        t.mode = 0;
        t.numBlocks = nb;
        t.rowsPerBlock = (rows + nb - 1) / nb;
        t.chunk = cbmn::kRowChunk;
    } else {
        // Prefer the widest lane group that keeps the whole inner extent in one group (no wasted
        // lanes); otherwise take the largest group the UB budget allows.  A wider group means
        // fewer, larger DMAs and fewer row-sync rounds, but the task count is outer*nGroups, so
        // shrink the group when the grid would otherwise be too small to fill the cores.
        int64_t a = (inner + 63) / 64 * 64;
        int64_t Tg = (a <= cbmn::kColTgMax) ? a : cbmn::kColTgMax;
        while (Tg > 256 && outer * ((inner + Tg - 1) / Tg) < coreNum) {
            Tg >>= 1;
        }
        if (Tg < 64) {
            Tg = 64;
        }
        t.Tg = Tg;
        t.nGroups = (inner + Tg - 1) / Tg;
        int64_t total = outer * t.nGroups;
        int64_t nb = coreNum;
        if (nb > total) {
            nb = total;
        }
        if (nb < 1) {
            nb = 1;
        }
        t.mode = 1;
        t.numBlocks = nb;
    }
    return t;
}

/* ================================================================================================ *
 * launch wrappers (extern "C", callable from g++)
 * ================================================================================================ */
extern "C" {

void launch_cummin_row_float(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t len,
                             int64_t rowsPerBlock, int64_t numBlocks, int64_t chunk, void* stream)
{
    cummin_row_kernel<float><<<numBlocks, nullptr, stream>>>(x, v, i, rows, len, rowsPerBlock, numBlocks, chunk);
}
void launch_cummin_row_half(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t len,
                            int64_t rowsPerBlock, int64_t numBlocks, int64_t chunk, void* stream)
{
    cummin_row_kernel<half><<<numBlocks, nullptr, stream>>>(x, v, i, rows, len, rowsPerBlock, numBlocks, chunk);
}
void launch_cummin_row_bfloat16(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t len,
                                int64_t rowsPerBlock, int64_t numBlocks, int64_t chunk, void* stream)
{
    cummin_row_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, v, i, rows, len, rowsPerBlock, numBlocks, chunk);
}
void launch_cummin_row_int32(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t len,
                             int64_t rowsPerBlock, int64_t numBlocks, int64_t chunk, void* stream)
{
    cummin_row_kernel<int32_t><<<numBlocks, nullptr, stream>>>(x, v, i, rows, len, rowsPerBlock, numBlocks, chunk);
}

void launch_cummin_col_float(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t outer, int64_t axis, int64_t inner,
                             int64_t nGroups, int64_t Tg, int64_t numBlocks, void* stream)
{
    cummin_col_kernel<float><<<numBlocks, nullptr, stream>>>(x, v, i, outer, axis, inner, nGroups, Tg, numBlocks);
}
void launch_cummin_col_half(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t outer, int64_t axis, int64_t inner,
                            int64_t nGroups, int64_t Tg, int64_t numBlocks, void* stream)
{
    cummin_col_kernel<half><<<numBlocks, nullptr, stream>>>(x, v, i, outer, axis, inner, nGroups, Tg, numBlocks);
}
void launch_cummin_col_bfloat16(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t outer, int64_t axis, int64_t inner,
                                int64_t nGroups, int64_t Tg, int64_t numBlocks, void* stream)
{
    cummin_col_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, v, i, outer, axis, inner, nGroups, Tg, numBlocks);
}
void launch_cummin_col_int32(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t outer, int64_t axis, int64_t inner,
                             int64_t nGroups, int64_t Tg, int64_t numBlocks, void* stream)
{
    cummin_col_kernel<int32_t><<<numBlocks, nullptr, stream>>>(x, v, i, outer, axis, inner, nGroups, Tg, numBlocks);
}

} // extern "C"
