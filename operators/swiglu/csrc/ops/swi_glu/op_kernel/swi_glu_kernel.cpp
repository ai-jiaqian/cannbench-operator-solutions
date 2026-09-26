/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file swi_glu_kernel.cpp
 * \brief SwiGLU kernel + host tiling + launch (compiled with bisheng, --npu-arch=dav-2201)
 *
 * Semantics (torch golden):
 *   x = input.to(float32); x0, x1 = x.chunk(2, dim); output = F.silu(x0) * x1; cast back.
 * Decomposition: d = ((dim%rank)+rank)%rank, D = size[d] (even), H = D/2,
 *   A = prod(sizes[0..d)), B = prod(sizes[d+1..)), L = H*B, N = A*L.
 *   Unit a in [0,A): x0 = in[2aL : 2aL+L], x1 = in[2aL+L : 2aL+2L],
 *   out[aL : (a+1)L] = silu(x0) * x1.
 *
 * silu computed in the fp32 domain (2-step exp chain, no Compare/Select, no tanh):
 *   negH = -x0/2; h = Exp(negH); t = h*h (IEEE-guaranteed overflow to +inf for
 *   x0 < -88.7); t = t + 1; y = x0/t; y *= x1. Special values match torch:
 *   +inf -> +inf, -inf -> NaN, NaN -> NaN, +-0 -> +-0.
 *   The 2-step exp is MANDATORY: direct Exp(-x0) hardware overflow behavior is
 *   unverified; Mul overflow to +inf is IEEE-guaranteed.
 * fp16/bf16 inputs are Cast-up to fp32 (CAST_NONE, exact) and Cast back (CAST_RINT).
 *
 * All GM<->UB transfers use DataCopyPad (plain DataCopy forbidden on this platform).
 *
 * Processing modes:
 *   Mode A (default/large L): per-unit tiled walk. Mode B (small L>1) batches W
 *     full units per 2D DataCopyPad group. Both run in one double-buffered
 *     pipeline: loads for item i+1 are issued (EnQue, queue depth 2) before item
 *     i is consumed, overlapping MTE transfers with vector compute; outQ keeps
 *     at most one outstanding tensor; single-item ranges degenerate to serial.
 *     Per tile/group: DataCopyPad loads, fp32-domain compute chain, DataCopyPad
 *     store of y.
 *   Mode B (small L > 1, batched): W consecutive FULL units per 2D DataCopyPad
 *     group. Load x0: blockCount=W, blockLen=L*esize, srcStride=L*esize (blocks
 *     spaced 2*L*esize apart in GM), dstStride=0 -> UB pitch = AlignUp(L*esize,32)
 *     (Normal mode pads each block to 32B in UB; with stride 0 the next block
 *     starts right after the padded end). The holes between valid lanes are
 *     dummy-filled by hardware, harmless for pure elementwise compute, and never
 *     stored: the UB->GM store (blockCount=W, blockLen=L*esize, srcStride=0,
 *     dstStride=0) emits only blockLen bytes per block, discarding the dummy.
 *     Compute runs over lanes = W*(pitch/esize) from base 0 with explicit count
 *     overloads. INVARIANT: W*pitch <= tileSize, so lanes <= tileElems and every
 *     queue buffer/temp fits the Mode A budget. Gates: Wmax = tileSize/pitch >= 2,
 *     W <= 4095 (blockCount limit), L*esize <= 2097151 (blockLen limit). Partial
 *     (non-unit-aligned) head/tail and stretches without 2+ full units fall back
 *     to Mode A tiles.
 *   Gather mode (L == 1): each output j pairs input[2j] (x0) and input[2j+1] (x1).
 *     Dense interleaved DataCopyPad of 2m contiguous elements into TWO VECIN
 *     buffers, then builtin-pattern GatherMask in place, Normal mode
 *     (reduceMode=false, mask=0, dst==src0 - verified doc semantics):
 *     pattern 1 collects even indices -> x0, pattern 2 collects odd -> x1.
 *     Vector ops run over EXACTLY m lanes (GatherMask dst beyond rsvdCnt is
 *     undefined). repeatTimes = ceil(2m / (256/esize)); each repeat consumes
 *     256B of src, src0RepeatStride = 8 DataBlocks (= 256B, consecutive chunks).
 *     User-defined GatherMask patterns are FORBIDDEN (measured accuracy failure).
 *     The dense buffer holds 2*tileSize bytes which covers the padded last
 *     repeat (repeats*256B <= 2*tileSize for tileElems = tileSize/esize).
 *
 * UB buffer ledger (per core). An fp32 temp holding tileElems = tileSize/esize
 * lanes occupies tileElems*4 = tileSize*4/esize bytes (= tileSize for fp32 input,
 * 2*tileSize for half/bf16 input). L != 1 (Mode A):
 *   inQ0/inQ1: TQue<VECIN,2>  x 2 buffers x tileSize = 6*tileSize
 *   outQ:      TQue<VECOUT,2> x 2 buffers x tileSize = 2*tileSize
 *   fp32 input: temps negH,h,t  = 3 x 1*tileSize            -> total  9*tileSize
 *   half/bf16 : temps x0f,x1f,negH,h,t,yf = 6 x 2*tileSize  -> total 18*tileSize
 *   (unused per-branch TBufs are NOT allocated - if constexpr on T)
 * L == 1 (Gather): inQ buffers hold the dense 2*tileSize load:
 *   inQ0/inQ1: 2 queues x depth 2 x 2*tileSize = 8*tileSize
 *   outQ:      2 buffers x tileSize             = 2*tileSize
 *   fp32: temps 3 x 1*tileSize = 3*tileSize    -> total 13*tileSize <= 40*tileSize - 16KB margin
 *   half : temps 6 x 2*tileSize = 12*tileSize  -> total 22*tileSize <= 40*tileSize - 16KB margin
 *   (the builtin GatherMask pattern may consume up to 8KB UB temp; covered by the
 *   extra 8KB in the 16KB margin)
 *   tileSize = floor((ubSize - ubMargin) / divisor / 128) * 128, divisor 9/18
 *   (Mode A fp32/narrow) or 40 (Gather); ubMargin 8192 (Mode A) / 16384 (Gather);
 *   ubSize from GetCoreMemSize (no hardcoded UB size).
 */

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "swi_glu_launch.h"

constexpr static int64_t SWI_GLU_QUE_DEPTH = 2;

// Tiling: {numBlocks, workPerCore, tileSize(bytes)}
std::tuple<int64_t, int64_t, int64_t> calc_swi_glu_tiling(int64_t A, int64_t L, int64_t esize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    const int64_t N = A * L;
    const int64_t minPerCore = 8192; // 1024 measured NEGATIVE in prior episode
    int64_t want = (N + minPerCore - 1) / minPerCore;
    if (want < 1) want = 1;
    int64_t numBlocks = coreNum < want ? coreNum : want;
    if (numBlocks < 1) numBlocks = 1;
    int64_t workPerCore = (N + numBlocks - 1) / numBlocks;

    const int64_t ubMargin = (L == 1) ? 16384 : 8192; // gather builtin pattern headroom
    int64_t budget = (int64_t)ubSize - ubMargin;
    if (budget < 128) budget = 128;
    // L == 1 keeps a conservative reserve (builtin-pattern headroom); the builtin
    // fixed GatherMask pattern needs no UB tensor. inQ buffers double (2*tileSize).
    const int64_t divisor = (L == 1) ? 40 : ((esize >= 4) ? 9 : 18);
    int64_t tileSize = budget / divisor / 128 * 128;
    if (tileSize < 128) tileSize = 128;
    return std::make_tuple(numBlocks, workPerCore, tileSize);
}

// fp32-domain SwiGLU math on `lanes` lanes; result written to y (VECOUT tensor).
template <typename T>
__aicore__ inline void SwiGluComputeChain(
    const AscendC::LocalTensor<T>& x0, const AscendC::LocalTensor<T>& x1,
    const AscendC::LocalTensor<T>& y,
    AscendC::TBuf<AscendC::QuePosition::VECCALC>& bNegH,
    AscendC::TBuf<AscendC::QuePosition::VECCALC>& bH,
    AscendC::TBuf<AscendC::QuePosition::VECCALC>& bT,
    AscendC::TBuf<AscendC::QuePosition::VECCALC>& bX0f,
    AscendC::TBuf<AscendC::QuePosition::VECCALC>& bX1f,
    AscendC::TBuf<AscendC::QuePosition::VECCALC>& bYf,
    uint32_t lanes)
{
    constexpr bool isFp32 = std::is_same<T, float>::value;
    auto negH = bNegH.Get<float>();
    auto h = bH.Get<float>();
    auto tv = bT.Get<float>();
    if constexpr (isFp32) {
        AscendC::Muls(negH, x0, -0.5f, lanes);
        AscendC::Exp(h, negH, lanes);
        AscendC::Mul(tv, h, h, lanes);
        AscendC::Adds(tv, tv, 1.0f, lanes);
        AscendC::Div(y, x0, tv, lanes);
        AscendC::Mul(y, y, x1, lanes);
    } else {
        auto x0f = bX0f.Get<float>();
        auto x1f = bX1f.Get<float>();
        auto yf = bYf.Get<float>();
        AscendC::Cast(x0f, x0, AscendC::RoundMode::CAST_NONE, lanes);
        AscendC::Cast(x1f, x1, AscendC::RoundMode::CAST_NONE, lanes);
        AscendC::Muls(negH, x0f, -0.5f, lanes);
        AscendC::Exp(h, negH, lanes);
        AscendC::Mul(tv, h, h, lanes);
        AscendC::Adds(tv, tv, 1.0f, lanes);
        AscendC::Div(yf, x0f, tv, lanes);
        AscendC::Mul(yf, yf, x1f, lanes);
        AscendC::Cast(y, yf, AscendC::RoundMode::CAST_RINT, lanes);
    }
}

// Geometry of the work item at cursor (a, off): a Mode B group of n full units
// (isB) or one Mode A tile of n elements. Pure - does not mutate the cursor.
__aicore__ inline void SwiGluItemGeom(int64_t a, int64_t off, int64_t e, int64_t L,
    int64_t tileElems, bool bGate, int64_t Wmax, bool& isB, int64_t& n)
{
    isB = false;
    n = 0;
    if (off == 0 && bGate) {
        const int64_t remainingFull = e / L - a; // full units [a, e/L)
        if (remainingFull >= 2) {
            n = remainingFull;
            if (n > Wmax) n = Wmax;
            if (n > 4095) n = 4095;
            if (n >= 2) {
                isB = true;
                return;
            }
            n = 0;
        }
    }
    n = L - off;
    const int64_t rem = e - (a * L + off);
    if (rem < n) n = rem;
    if (tileElems < n) n = tileElems;
}

// Issue (Alloc + DataCopyPad + EnQue) the loads for one work item.
template <typename T>
__aicore__ inline void SwiGluIssueLoad(
    AscendC::TQue<AscendC::QuePosition::VECIN, SWI_GLU_QUE_DEPTH>& inQ0,
    AscendC::TQue<AscendC::QuePosition::VECIN, SWI_GLU_QUE_DEPTH>& inQ1,
    const AscendC::GlobalTensor<T>& xGm, int64_t a, int64_t off, int64_t L,
    int64_t esize, bool isB, int64_t n)
{
    AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    auto x0l = inQ0.AllocTensor<T>();
    auto x1l = inQ1.AllocTensor<T>();
    if (isB) {
        // W = n full units: blocks spaced 2*L*esize bytes apart in GM
        AscendC::DataCopyExtParams cpIn{(uint16_t)n, (uint32_t)(L * esize),
                                        (uint32_t)(L * esize), 0, 0};
        AscendC::DataCopyPad(x0l, xGm[2 * a * L], cpIn, pp);
        AscendC::DataCopyPad(x1l, xGm[2 * a * L + L], cpIn, pp); // element index
    } else {
        AscendC::DataCopyExtParams cp{(uint32_t)1, (uint32_t)(n * esize), 0, 0, 0};
        AscendC::DataCopyPad(x0l, xGm[2 * a * L + off], cp, pp);
        AscendC::DataCopyPad(x1l, xGm[2 * a * L + off + L], cp, pp);
    }
    inQ0.EnQue(x0l);
    inQ1.EnQue(x1l);
}

template <typename T>
__global__ __aicore__ void swi_glu_kernel(GM_ADDR x, GM_ADDR y,
    int64_t A, int64_t L, int64_t workPerCore, uint32_t tileSize)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, yGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, SWI_GLU_QUE_DEPTH> inQ0;
    AscendC::TQue<AscendC::QuePosition::VECIN, SWI_GLU_QUE_DEPTH> inQ1;
    AscendC::TQue<AscendC::QuePosition::VECOUT, SWI_GLU_QUE_DEPTH> outQ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bNegH, bH, bT, bX0f, bX1f, bYf;

    const bool gather = (L == 1);

    pipe.InitBuffer(inQ0, SWI_GLU_QUE_DEPTH, gather ? tileSize * 2 : tileSize);
    pipe.InitBuffer(inQ1, SWI_GLU_QUE_DEPTH, gather ? tileSize * 2 : tileSize);
    pipe.InitBuffer(outQ, SWI_GLU_QUE_DEPTH, tileSize);
    if constexpr (std::is_same<T, float>::value) {
        pipe.InitBuffer(bNegH, tileSize);
        pipe.InitBuffer(bH, tileSize);
        pipe.InitBuffer(bT, tileSize);
    } else {
        // fp32 temps for tileElems lanes need 4*tileElems = 2*tileSize bytes each
        pipe.InitBuffer(bNegH, tileSize * 2);
        pipe.InitBuffer(bH, tileSize * 2);
        pipe.InitBuffer(bT, tileSize * 2);
        pipe.InitBuffer(bX0f, tileSize * 2);
        pipe.InitBuffer(bX1f, tileSize * 2);
        pipe.InitBuffer(bYf, tileSize * 2);
    }

    xGm.SetGlobalBuffer((__gm__ T *)x);
    yGm.SetGlobalBuffer((__gm__ T *)y);

    const int64_t N = A * L;
    const int64_t s = (int64_t)AscendC::GetBlockIdx() * workPerCore;
    if (s >= N) {
        return;
    }
    const int64_t e = (s + workPerCore > N) ? N : (s + workPerCore);
    const int64_t esize = (int64_t)sizeof(T);
    const int64_t tileElems = (int64_t)tileSize / esize;

    if (gather) {
        // Gather mode (L == 1): output j pairs in[2j], in[2j+1]. Dense interleaved
        // load of 2m elements into two VECIN buffers, builtin GatherMask even/odd
        // de-interleave in place (Normal mode), compute over exactly m lanes.
        const uint32_t elemsPerRepeat = 256 / (uint32_t)esize; // 64 fp32 / 128 half
        const int64_t n = e - s;
        int64_t done = 0;
        while (done < n) {
            int64_t m = tileElems;
            if (n - done < m) m = n - done;
            const uint32_t lanes = (uint32_t)m;
            const int64_t base = s + done;   // output index; input offset = 2*base
            const uint32_t repeats = (uint32_t)((2 * m + elemsPerRepeat - 1) / elemsPerRepeat);

            AscendC::DataCopyExtParams cp{(uint32_t)1, (uint32_t)(2 * m * esize), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};

            auto x0d = inQ0.AllocTensor<T>();
            auto x1d = inQ1.AllocTensor<T>();
            AscendC::DataCopyPad(x0d, xGm[2 * base], cp, pp);
            AscendC::DataCopyPad(x1d, xGm[2 * base], cp, pp);
            inQ0.EnQue(x0d);
            inQ1.EnQue(x1d);
            auto x0q = inQ0.DeQue<T>();
            auto x1q = inQ1.DeQue<T>();

            // builtin fixed patterns: 1 -> even indices (x0), 2 -> odd indices (x1).
            // Normal mode: reduceMode=false, mask=0; src0RepeatStride=8 DataBlocks
            // (=256B) so consecutive repeats consume consecutive 256B src chunks.
            AscendC::GatherMaskParams gp;
            gp.src0BlockStride = 1;
            gp.repeatTimes = repeats;
            gp.src0RepeatStride = (uint16_t)8; // field is uint16_t: narrow explicitly
            gp.src1RepeatStride = 0;
            uint64_t rsvdCnt = 0;
            AscendC::GatherMask(x0q, x0q, (uint8_t)1, false, 0u, gp, rsvdCnt);
            AscendC::GatherMask(x1q, x1q, (uint8_t)2, false, 0u, gp, rsvdCnt);

            auto yl = outQ.AllocTensor<T>();
            SwiGluComputeChain<T>(x0q, x1q, yl, bNegH, bH, bT, bX0f, bX1f, bYf, lanes);
            inQ0.FreeTensor(x0q);
            inQ1.FreeTensor(x1q);
            outQ.EnQue(yl);
            auto yd = outQ.DeQue<T>();
            AscendC::DataCopyExtParams cpOut{(uint32_t)1, (uint32_t)(m * esize), 0, 0, 0};
            AscendC::DataCopyPad(yGm[base], yd, cpOut);
            outQ.FreeTensor(yd);
            done += m;
        }
        return;
    }

    // Mode A/B pipelined walk: issue loads for item i+1 (queue depth 2) before
    // consuming item i, so MTE transfers overlap vector compute. outQ keeps at
    // most one outstanding tensor (Alloc out only after the previous store's
    // Free). Invariant: at most cur + next input tensors outstanding per queue.
    const int64_t pitch = ((L * esize + 31) / 32) * 32;   // UB block pitch, bytes
    const int64_t pitchElems = pitch / esize;
    const int64_t Wmax = (pitch > 0) ? tileSize / pitch : 0;
    const bool bGate = (L > 1) && (Wmax >= 2) && (L * esize <= 2097151);

    int64_t a = s / L;
    int64_t off = s - a * L;
    bool isB = false;
    int64_t n = 0;
    SwiGluItemGeom(a, off, e, L, tileElems, bGate, Wmax, isB, n);
    // issue the first item's loads; inside the loop each item's loads are issued
    // exactly once (as the "next" prefetch of the previous iteration).
    SwiGluIssueLoad<T>(inQ0, inQ1, xGm, a, off, L, esize, isB, n);
    while (true) {
        // 1) advance the cursor (pure) and peek the next item's geometry.
        int64_t a2 = a;
        int64_t off2 = off;
        if (isB) {
            a2 += n;
        } else {
            off2 += n;
            if (off2 == L) {
                ++a2;
                off2 = 0;
            }
        }
        const bool hasNext = (a2 * L + off2 < e);
        bool nextIsB = false;
        int64_t nextN = 0;
        if (hasNext) {
            SwiGluItemGeom(a2, off2, e, L, tileElems, bGate, Wmax, nextIsB, nextN);
            // prefetch: issue loads for the next item now. Invariant: at this
            // point only the current item's tensors are outstanding in each
            // input queue (depth 2), so the Alloc below cannot deadlock.
            SwiGluIssueLoad<T>(inQ0, inQ1, xGm, a2, off2, L, esize, nextIsB, nextN);
        }

        // 2) consume the current item.
        auto x0d = inQ0.DeQue<T>();
        auto x1d = inQ1.DeQue<T>();
        const uint32_t lanes = isB ? (uint32_t)(n * pitchElems) : (uint32_t)n;
        auto yl = outQ.AllocTensor<T>();
        SwiGluComputeChain<T>(x0d, x1d, yl, bNegH, bH, bT, bX0f, bX1f, bYf, lanes);
        inQ0.FreeTensor(x0d);
        inQ1.FreeTensor(x1d);
        outQ.EnQue(yl);
        auto yd = outQ.DeQue<T>();
        if (isB) {
            AscendC::DataCopyExtParams cpOut{(uint16_t)n, (uint32_t)(L * esize), 0, 0, 0};
            AscendC::DataCopyPad(yGm[a * L], yd, cpOut);
        } else {
            AscendC::DataCopyExtParams cpOut{(uint32_t)1, (uint32_t)(n * esize), 0, 0, 0};
            AscendC::DataCopyPad(yGm[a * L + off], yd, cpOut);
        }
        outQ.FreeTensor(yd);

        if (!hasNext) {
            break;
        }
        a = a2;
        off = off2;
        isB = nextIsB;
        n = nextN;
    }
}

// Launch wrappers - regular C functions callable from g++
extern "C" {

void launch_swi_glu_float(GM_ADDR x, GM_ADDR y, int64_t A, int64_t L, int64_t numBlocks, int64_t workPerCore, uint32_t tileSize, void* stream)
{
    swi_glu_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, A, L, workPerCore, tileSize);
}

void launch_swi_glu_half(GM_ADDR x, GM_ADDR y, int64_t A, int64_t L, int64_t numBlocks, int64_t workPerCore, uint32_t tileSize, void* stream)
{
    swi_glu_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, A, L, workPerCore, tileSize);
}

void launch_swi_glu_bf16(GM_ADDR x, GM_ADDR y, int64_t A, int64_t L, int64_t numBlocks, int64_t workPerCore, uint32_t tileSize, void* stream)
{
    swi_glu_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, A, L, workPerCore, tileSize);
}

}
