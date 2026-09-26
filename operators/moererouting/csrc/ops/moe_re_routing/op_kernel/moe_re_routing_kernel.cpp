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
 * \file moe_re_routing_kernel.cpp
 * \brief MoeReRouting kernel + host tiling + launch (bisheng -xasc, dav-2201).
 *
 * Semantics (identical to task/reference.py):
 *   rowStart(i) = sum_{i'<i} sum_j cnt[i'][j]
 *   colStart(j) = sum_{j'<j} sum_i cnt[i'][j']
 *   srcStart(i,j) = rowStart(i) + sum_{j'<j} cnt[i][j']
 *   dstStart(i,j) = colStart(j) + sum_{i'<i} cnt[i'][j]
 *   for k in [0, cnt[i][j]):
 *       permute_tokens[dstStart+k]           = tokens[srcStart+k]
 *       permute_per_token_scales[dstStart+k] = per_token_scales[srcStart+k]
 *       permute_token_idx[dstStart+k]        = srcStart+k
 *   expert_token_num[j] = sum_i cnt[i][j]                (count mode)
 *
 * Structure
 * ---------
 *  1. Blocks own contiguous *destination* token rows [d0, d1).  One (rank, expert) cell clipped
 *     to the window is a "segment": a contiguous source range copied to a contiguous destination
 *     range.
 *  2. The (N, E) count matrix is DMA'd into UB and its column sums are produced by the VECTOR unit
 *     with a halving fold.  No O(N*E) scalar pass remains: the kernel used to spend ~4 UB scalar
 *     ops per count element (a dtype conversion loop, a row-prefix pass and a column-sum pass)
 *     before the walk even started.  For shapes with many cells that pure scalar prologue was the
 *     dominant cost of the whole kernel, so it is now folded away.
 *  3. The fold runs on a raw 32-bit view of the count bytes.  Because addition is modulo 2^32, the
 *     low word of a sum of int64 low-words equals the low 32 bits of the true sum; all column sums
 *     are <= A < 2^31, so the low word *is* the answer.  Both int32 and int64 counts therefore use
 *     the exact same fold code, with W = E*sizeof(CT)/4 words per row; element (i,j) lives at word
 *     i*W + j*CW, CW = sizeof(CT)/4.
 *  4. rowStart is not precomputed either: the walk iterates every column j in [0, E) so the running
 *     `within` sum at the end of a row IS the row sum, and a register `rowTotal` carries rowStart.
 *  5. dstStart is maintained as a single per-column cursor array (colStart initial value, advanced
 *     by cnt[i][j]); it is only touched for j in [jLo, jHi), the exact column range whose interval
 *     [colStart(j), colStart(j+1)) can intersect [d0, d1).
 *
 * Cost model that drives the design
 * ---------------------------------
 *  - A contiguous copy costs essentially one MTE2 (or MTE3) command regardless of payload below
 *    the burst limit, so on every case that is not pure HBM traffic the operator is bound by the
 *    *number* of DMA commands and by the serialising flag waits between them, not by bytes.  For
 *    the HBM bound cases the opposite holds, so the chunk size is derived from the whole UB budget
 *    rather than from a fixed cap.
 *  - Therefore (a) the token and scale streams run as a kernel-wide two-slot software pipeline so
 *    that job n's load is in flight while job n-1 is stored, and (b) the index ramp is built once
 *    per *segment* rather than once per streaming chunk: idx[d] = d + (srcStart - dstStart) holds
 *    for the whole segment, so a single CreateVecIndex plus one flat DMA reproduces it.
 *  - Every UB buffer that either sources or receives a DMA starts on a 32 B boundary, and the
 *    per-slot pitch of every multi-slot buffer is rounded up to 32 B, because `DataCopyPad`
 *    between UB and GM requires a 32 B aligned UB operand.
 *  - All grid / chunk policy numbers live in moe_re_routing_tiling.h.
 *
 * Only blockCount==1 DataCopyPad calls are issued: the multi-block strided form has field-unit and
 * destination base alignment traps.
 *
 * Synchronisation: AscendC inserts NO synchronisation for raw TBuf DMA.  Event ids are compile time
 * constants, each numeric id belongs to exactly one (source -> target) pipe pair, and every SetFlag
 * is matched by exactly one WaitFlag before kernel exit.  Slot selection uses `if (slot == 0) ID0
 * else ID1`, never a runtime-computed id value.  The MTE2 -> V edge of the fold uses its own pair,
 * and a full PIPE_ALL barrier separates the vector fold from the scalar read of its result.
 */

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "moe_re_routing_tiling.h"

/* Event ids.  Each id belongs to exactly one (source -> target) pipe pair. */
constexpr int32_t EV_L0 = 1;    /* token+scale load done, slot 0 : MTE2 -> MTE3 */
constexpr int32_t EV_L1 = 2;    /* token+scale load done, slot 1 : MTE2 -> MTE3 */
constexpr int32_t EV_S0 = 3;    /* token+scale store done, slot 0: MTE3 -> MTE2 */
constexpr int32_t EV_S1 = 4;    /* token+scale store done, slot 1: MTE3 -> MTE2 */
constexpr int32_t EV_W = 5;     /* index ramp built              : V    -> MTE3 */
constexpr int32_t EV_R = 6;     /* index store done              : MTE3 -> V    */
constexpr int32_t EV_M2S = 7;   /* count load done               : MTE2 -> S    */
constexpr int32_t EV_SM3 = 8;   /* expert_token_num written      : S    -> MTE3 */
constexpr int32_t EV_M3S = 9;   /* final store drain             : MTE3 -> S    */
constexpr int32_t EV_M2V = 10;  /* count load done               : MTE2 -> V    */

__aicore__ inline int64_t RrAlign32(int64_t v) { return (v + 31) / 32 * 32; }
__aicore__ inline int64_t RrMin(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline int64_t RrMax(int64_t a, int64_t b) { return a > b ? a : b; }
__aicore__ inline int64_t RrIdxCap(int64_t A) { return A < MOE_RR_IDX_CAP ? A : MOE_RR_IDX_CAP; }

template <typename T, typename CT>
__global__ __aicore__ void MoeRrKernel(GM_ADDR tokGm_, GM_ADDR cntGm_, GM_ADDR sclGm_,
                                       GM_ADDR oTokGm_, GM_ADDR oSclGm_, GM_ADDR oIdxGm_,
                                       GM_ADDR oExpGm_, int64_t A, int64_t H, int64_t N, int64_t E,
                                       int64_t dstPerBlk, int64_t chunkTokens, int64_t hasScales)
{
    constexpr int64_t TSZ = static_cast<int64_t>(sizeof(T));
    constexpr int64_t CSZ = static_cast<int64_t>(sizeof(CT));

    const int64_t NE = N * E;
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());

    AscendC::GlobalTensor<T> tokGm;
    AscendC::GlobalTensor<CT> cntGm;
    AscendC::GlobalTensor<int32_t> cntI32Gm;
    AscendC::GlobalTensor<float> sclGm;
    AscendC::GlobalTensor<T> oTokGm;
    AscendC::GlobalTensor<float> oSclGm;
    AscendC::GlobalTensor<int32_t> oIdxGm;
    AscendC::GlobalTensor<CT> oExpGm;
    tokGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(tokGm_));
    cntGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(cntGm_));
    cntI32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cntGm_));
    if (hasScales != 0) {
        sclGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sclGm_));
    }
    oTokGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(oTokGm_));
    oSclGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(oSclGm_));
    oIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(oIdxGm_));
    oExpGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(oExpGm_));

    /* ---------------------------------------------------------------- UB budget ---------- */
    const int64_t idxCap = RrIdxCap(A);
    const int64_t cntBytes = RrAlign32(NE * CSZ) + 64;
    const int64_t auxBytes = RrAlign32(E * 8) + 64;
    const int64_t expBytes = RrAlign32(E * CSZ) + 64;
    /* per-slot pitches, rounded up to 32 B so every slot start is a legal DMA operand */
    const int64_t tokPitchBytes = RrAlign32(chunkTokens * H * TSZ);
    const int64_t scPitchBytes = RrAlign32(chunkTokens * 4);
    const int64_t tokBufBytes = 2 * tokPitchBytes + 64;
    const int64_t scBufBytes = 2 * scPitchBytes + 64;
    const int64_t idxBufBytes = RrAlign32(idxCap * 4) + 64;
    const int64_t tokSlotElems = tokPitchBytes / TSZ;
    const int64_t scSlotElems = scPitchBytes / 4;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cntRawBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cntFoldBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> auxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;
    pipe.InitBuffer(cntRawBuf, static_cast<uint32_t>(cntBytes));
    pipe.InitBuffer(cntFoldBuf, static_cast<uint32_t>(cntBytes));
    pipe.InitBuffer(auxBuf, static_cast<uint32_t>(auxBytes));
    pipe.InitBuffer(expBuf, static_cast<uint32_t>(expBytes));
    pipe.InitBuffer(tokBuf, static_cast<uint32_t>(tokBufBytes));
    pipe.InitBuffer(scBuf, static_cast<uint32_t>(scBufBytes));
    pipe.InitBuffer(idxBuf, static_cast<uint32_t>(idxBufBytes));

    AscendC::LocalTensor<CT> cntRaw = cntRawBuf.Get<CT>();
    AscendC::LocalTensor<int32_t> cntFold = cntFoldBuf.Get<int32_t>();
    AscendC::LocalTensor<int64_t> aux = auxBuf.Get<int64_t>();
    AscendC::LocalTensor<CT> expUb = expBuf.Get<CT>();
    AscendC::LocalTensor<T> tokUb = tokBuf.Get<T>();
    AscendC::LocalTensor<float> scUb = scBuf.Get<float>();
    AscendC::LocalTensor<int32_t> idxUb = idxBuf.Get<int32_t>();
    /* aux layout: [0, E) = running destination cursor dstStart(i, j) for column j */

    /* ------------------------------------------------- 1. count matrix -> UB -------------- */
    {
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(NE * CSZ), 0, 0, 0};
        AscendC::DataCopyPadExtParams<CT> padC{false, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
        AscendC::DataCopyPad(cntRaw, cntGm[0], cp, padC);
        AscendC::DataCopyPad(cntFold, cntI32Gm[0], cp, padI);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EV_M2V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EV_M2V);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EV_M2S);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EV_M2S);

    /* ------------------------------- 2. column sums by a halving vector fold --------------- */
    /* W = int32 words per count row, CW = int32 words per count element.  Element (i, j) sits at
     * word index i*W + j*CW.  The fold is exact for int64 as well because only the low 32 bits of
     * every element are ever read out of the result (all sums are <= A < 2^31). */
    const int64_t CW = CSZ / 4;
    const int64_t W = E * CW;
    const bool foldOk = ((E * CSZ) % 32) == 0;
    if (foldOk) {
        int64_t n = N;
        while (n > 1) {
            const int64_t h = n / 2;
            if (h > 0) {
                AscendC::Add(cntFold, cntFold, cntFold[static_cast<uint32_t>(h * W)],
                             static_cast<int32_t>(h * W));
            }
            if ((n & 1) != 0) {
                AscendC::Add(cntFold, cntFold, cntFold[static_cast<uint32_t>((n - 1) * W)],
                             static_cast<int32_t>(W));
            }
            n = h;
        }
        /* vector result must be visible to the scalar pipe before any GetValue of it */
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    const int64_t d0 = blk * dstPerBlk;
    const int64_t d1 = RrMin(A, d0 + dstPerBlk);

    /* -------- 3. colSum -> colStart -> dstStart cursor, plus the jLo/jHi column bounds ----- */
    int64_t jLo = E;
    int64_t jHi = E;
    {
        int64_t total = 0;
        for (int64_t j = 0; j < E; ++j) {
            int64_t s;
            if (foldOk) {
                s = static_cast<int64_t>(cntFold.GetValue(static_cast<uint32_t>(j * CW)));
            } else {
                int64_t acc = 0;
                for (int64_t i = 0; i < N; ++i) {
                    acc += static_cast<int64_t>(cntRaw.GetValue(static_cast<uint32_t>(i * E + j)));
                }
                s = acc;
            }
            aux.SetValue(static_cast<uint32_t>(j), total); /* dstStart(0, j) = colStart(j) */
            expUb.SetValue(static_cast<uint32_t>(j), static_cast<CT>(s));
            if (jLo == E && total + s > d0) {
                jLo = j;
            }
            if (jHi == E && total >= d1) {
                jHi = j;
            }
            total += s;
        }
    }
    if (jLo >= jHi) {
        jLo = jHi;
    }

    /* --------------------------------------- 4. expert_token_num via one DMA -------------- */
    if (blk == 0) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EV_SM3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EV_SM3);
        AscendC::DataCopyExtParams cpE{1, static_cast<uint32_t>(E * CSZ), 0, 0, 0};
        AscendC::DataCopyPad(oExpGm, expUb, cpE);
    }

    if (d0 < A) {
        if (hasScales == 0) {
            /* the reference emits exact zeros when the caller carries no scales; both slots are
             * zeroed once so the per-job scale store below needs no special case. */
            AscendC::Duplicate(scUb, 0.0f, static_cast<int32_t>(2 * scSlotElems));
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        AscendC::DataCopyPadExtParams<T> padT{false, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padF{false, 0, 0, 0};

        int64_t jobIdx = 0;
        int32_t prevSlot = 0;
        int64_t prevDstTok = 0;
        int64_t prevP = 0;
        int64_t rowTotal = 0;

        for (int64_t i = 0; i < N; ++i) {
            const int64_t base = i * E;
            int64_t within = 0;
            for (int64_t j = 0; j < E; ++j) {
                const int64_t c =
                    static_cast<int64_t>(cntRaw.GetValue(static_cast<uint32_t>(base + j)));
                if (j >= jLo && j < jHi) {
                    const int64_t dstStart = aux.GetValue(static_cast<uint32_t>(j));
                    const int64_t lo = RrMax(dstStart, d0);
                    const int64_t hi = RrMin(dstStart + c, d1);
                    if (hi > lo) {
                        const int64_t srcStart = rowTotal + within;
                        const int64_t srcBase = srcStart + (lo - dstStart);
                        const int64_t dstBase = lo;
                        const int64_t len = hi - lo;

                        /* ---- index ramp for the whole segment: idx[d] = d + (srcBase - dstBase) */
                        for (int64_t io = 0; io < len; io += idxCap) {
                            const int64_t ip = RrMin(idxCap, len - io);
                            AscendC::CreateVecIndex(idxUb, static_cast<int32_t>(srcBase + io),
                                                    static_cast<int32_t>(ip));
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EV_W);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EV_W);
                            AscendC::DataCopyExtParams cpI{1, static_cast<uint32_t>(ip * 4), 0, 0,
                                                           0};
                            AscendC::DataCopyPad(oIdxGm[static_cast<uint32_t>(dstBase + io)], idxUb,
                                                 cpI);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EV_R);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EV_R);
                        }

                        /* ---- token + scale chunks, two-slot software pipeline ---- */
                        for (int64_t off = 0; off < len; off += chunkTokens) {
                            const int64_t p = RrMin(chunkTokens, len - off);
                            const int64_t srcTok = srcBase + off;
                            const int64_t dstTok = dstBase + off;
                            const int32_t slot = static_cast<int32_t>(jobIdx & 1);

                            /* slot (jobIdx&1) was last written by job jobIdx-2; its store must be
                             * complete before the new load overwrites it. */
                            if (jobIdx >= 2) {
                                if (slot == 0) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S0);
                                } else {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S1);
                                }
                            }
                            AscendC::DataCopyExtParams cpIn{1, static_cast<uint32_t>(p * H * TSZ), 0,
                                                            0, 0};
                            AscendC::DataCopyPad(
                                tokUb[static_cast<uint32_t>(slot * tokSlotElems)],
                                tokGm[static_cast<uint32_t>(srcTok * H)], cpIn, padT);
                            if (hasScales != 0) {
                                AscendC::DataCopyExtParams cpS{1, static_cast<uint32_t>(p * 4), 0,
                                                               0, 0};
                                AscendC::DataCopyPad(
                                    scUb[static_cast<uint32_t>(slot * scSlotElems)],
                                    sclGm[static_cast<uint32_t>(srcTok)], cpS, padF);
                            }
                            if (slot == 0) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EV_L0);
                            } else {
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EV_L1);
                            }

                            /* retire the previous job from the other slot */
                            if (jobIdx >= 1) {
                                if (prevSlot == 0) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EV_L0);
                                } else {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EV_L1);
                                }
                                AscendC::DataCopyExtParams cpOut{
                                    1, static_cast<uint32_t>(prevP * H * TSZ), 0, 0, 0};
                                AscendC::DataCopyPad(
                                    oTokGm[static_cast<uint32_t>(prevDstTok * H)],
                                    tokUb[static_cast<uint32_t>(prevSlot * tokSlotElems)], cpOut);
                                /* the scale output is written unconditionally: with no input
                                 * scales the reference still emits a zero row. */
                                AscendC::DataCopyExtParams cpSO{
                                    1, static_cast<uint32_t>(prevP * 4), 0, 0, 0};
                                AscendC::DataCopyPad(
                                    oSclGm[static_cast<uint32_t>(prevDstTok)],
                                    scUb[static_cast<uint32_t>(prevSlot * scSlotElems)], cpSO);
                                if (prevSlot == 0) {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S0);
                                } else {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S1);
                                }
                            }

                            prevSlot = slot;
                            prevDstTok = dstTok;
                            prevP = p;
                            ++jobIdx;
                        }
                    }
                    aux.SetValue(static_cast<uint32_t>(j), dstStart + c);
                }
                within += c;
            }
            rowTotal += within;
        }

        /* retire the last job */
        if (jobIdx >= 1) {
            if (prevSlot == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EV_L0);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EV_L1);
            }
            AscendC::DataCopyExtParams cpOut{1, static_cast<uint32_t>(prevP * H * TSZ), 0, 0, 0};
            AscendC::DataCopyPad(oTokGm[static_cast<uint32_t>(prevDstTok * H)],
                                 tokUb[static_cast<uint32_t>(prevSlot * tokSlotElems)], cpOut);
            AscendC::DataCopyExtParams cpSO{1, static_cast<uint32_t>(prevP * 4), 0, 0, 0};
            AscendC::DataCopyPad(oSclGm[static_cast<uint32_t>(prevDstTok)],
                                 scUb[static_cast<uint32_t>(prevSlot * scSlotElems)], cpSO);
            if (prevSlot == 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S0);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S1);
            }
        }
        /* consume the (at most two) pending store flags so none is left Set at kernel exit */
        if (jobIdx >= 1) {
            const int32_t sp = static_cast<int32_t>((jobIdx - 1) & 1);
            if (sp == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S0);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S1);
            }
        }
        if (jobIdx >= 2) {
            const int32_t sp = static_cast<int32_t>((jobIdx - 2) & 1);
            if (sp == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S0);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EV_S1);
            }
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EV_M3S);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EV_M3S);
}

/* ------------------------------------------------------------------ host tiling ----------- */

static inline int64_t RrHAlign32(int64_t v) { return (v + 31) / 32 * 32; }
static inline int64_t RrHMin(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t RrHMax(int64_t a, int64_t b) { return a > b ? a : b; }

MoeRRTiling calc_moe_re_routing_tiling(int64_t A, int64_t H, int64_t N, int64_t E, int64_t tokElem,
                                       int64_t cntElem)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    if (coreNum <= 0) {
        coreNum = 1;
    }

    /* Grid policy.  Two independent lower bounds on the useful number of blocks:
     *   - token traffic: give a block at least MOE_RR_BYTES_PER_BLOCK bytes (every block
     *     redundantly walks the (N, E) count matrix and pays a fixed dispatch);
     *   - parallelism floor: never drop below min(NE, MOE_RR_CELL_FLOOR) blocks, because a block
     *     owns one contiguous destination window and a window narrower than the average
     *     (rank, expert) cell splits that cell into several individually synchronised segments.
     * At most min(coreNum, A) blocks can ever be useful. */
    const int64_t NE = N * E;
    const int64_t tokRowBytes = H * tokElem;
    int64_t rowsPerBlk = 1;
    if (tokRowBytes > 0) {
        rowsPerBlk = MOE_RR_BYTES_PER_BLOCK / tokRowBytes;
    }
    if (rowsPerBlk < 1) {
        rowsPerBlk = 1;
    }
    int64_t numBlocks = (A + rowsPerBlk - 1) / rowsPerBlk;
    const int64_t floorBlocks = RrHMin(NE, MOE_RR_CELL_FLOOR);
    if (numBlocks < floorBlocks) {
        numBlocks = floorBlocks;
    }
    if (numBlocks > coreNum) {
        numBlocks = coreNum;
    }
    if (numBlocks > A) {
        numBlocks = A;
    }
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t dstPerBlk = (A + numBlocks - 1) / numBlocks;
    numBlocks = (A + dstPerBlk - 1) / dstPerBlk;

    const int64_t idxCap = (A < MOE_RR_IDX_CAP) ? A : MOE_RR_IDX_CAP;
    const int64_t cntBytes = RrHAlign32(NE * cntElem) + 64;
    const int64_t auxBytes = RrHAlign32(E * 8) + 64;
    const int64_t expBytes = RrHAlign32(E * cntElem) + 64;
    const int64_t idxBytes = RrHAlign32(idxCap * 4) + 64;
    int64_t avail = static_cast<int64_t>(ubSize) - 2 * cntBytes - auxBytes - expBytes - idxBytes -
                    MOE_RR_UB_SAFETY;
    if (avail < 1024) {
        avail = 1024;
    }
    /* per token, both slots together: H token elements + 4 bytes of scale, plus 32 B pitch slack
     * per slot and the per-buffer padding.  Everything the two slots need must fit in `avail`, so
     * the chunk size follows directly from the UB size -- there is no separate hard cap. */
    const int64_t perTokStream = 2 * (H * tokElem + 4) + 128;
    int64_t chunkTokens = avail / perTokStream;
    if (chunkTokens < 1) {
        chunkTokens = 1;
    }
    if (A > 0 && chunkTokens > A) {
        chunkTokens = A;
    }
    MoeRRTiling out;
    out.numBlocks = numBlocks;
    out.dstPerBlk = dstPerBlk;
    out.chunkTokens = chunkTokens;
    out.idxCap = idxCap;
    return out;
}

/* ------------------------------------------------------------------ launches -------------- */

#define MOE_RR_LAUNCH(NAME, TOKT, CNTT)                                                           \
    extern "C" void NAME(GM_ADDR tok, GM_ADDR cnt, GM_ADDR scl, GM_ADDR oTok, GM_ADDR oScl,       \
                         GM_ADDR oIdx, GM_ADDR oExp, int64_t A, int64_t H, int64_t N, int64_t E,  \
                         int64_t numBlocks, int64_t dstPerBlk, int64_t chunkTokens,               \
                         int64_t hasScales, void *stream)                                         \
    {                                                                                             \
        MoeRrKernel<TOKT, CNTT><<<numBlocks, nullptr, stream>>>(tok, cnt, scl, oTok, oScl, oIdx,  \
                                                                oExp, A, H, N, E, dstPerBlk,       \
                                                                chunkTokens, hasScales);           \
    }

MOE_RR_LAUNCH(launch_moe_rr_h_i32, half, int32_t)
MOE_RR_LAUNCH(launch_moe_rr_h_i64, half, int64_t)
MOE_RR_LAUNCH(launch_moe_rr_bf_i32, bfloat16_t, int32_t)
MOE_RR_LAUNCH(launch_moe_rr_bf_i64, bfloat16_t, int64_t)
MOE_RR_LAUNCH(launch_moe_rr_i8_i32, int8_t, int32_t)
MOE_RR_LAUNCH(launch_moe_rr_i8_i64, int8_t, int64_t)
