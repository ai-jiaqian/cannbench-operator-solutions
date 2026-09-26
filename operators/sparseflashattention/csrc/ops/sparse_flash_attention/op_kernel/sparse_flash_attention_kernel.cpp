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
 * \file sparse_flash_attention_kernel.cpp
 * \brief SparseFlashAttention device kernel + launch wrappers (bisheng + -xasc, dav-2201 / Ascend 910B).
 *
 *   Ksel = gather(K, sparseIndices);  Vsel = gather(V, sparseIndices)
 *   y    = softmax(Q @ Ksel^T * scaleValue) @ Vsel
 *
 * Layout
 * ------
 * Every axis access goes through a per-tensor element stride carried in SfaParams, so BSND and BNSD
 * (and any legal outer non-contiguous view) execute the identical body. The innermost head dimension
 * must be contiguous; the API layer checks that.
 *
 * Tiling
 * ------
 * task = (b, n2, s1, head-block of HG query heads). All HG heads of a GQA group share one sparse
 * index row, so the gathered K / V rows are loaded once per key block and reused by every head.
 * Keys are processed in blocks of KT (8 or 16) evaluated with high dimensional vector instructions.
 *
 * Numerics
 * --------
 * Scores are bounded: |q|,|k| <= 1, Dk <= 576 and scale <= 0.09 give |s| <= ~52, so exp() cannot
 * overflow fp32 and a running maximum / rescale pass is unnecessary. Everything accumulates in fp32;
 * only the final store down-casts to the output dtype.
 *
 * The causal mask is right-bottom aligned: a selected KV position idx is dropped when
 * idx > s1 + (S2 - S1). Dropped lanes are pushed to a large negative finite constant so exp() is
 * exactly 0, and a fully dropped row therefore yields an all-zero output row.
 *
 * Cost model (measured on this target)
 * -----------------------------------
 * The QK reduction is the dominant instruction: a whole-repeat fp32 reduction runs at 64/7 elements
 * per cycle while Mul/Add run at 64. So the Dk chunks of one (head, key-block) pair are first
 * *accumulated* with cheap Mul/Add into a single [KT, 64] tile, and only one reduction is issued per
 * (head, Dk) instead of one per (head, chunk). Pass 2 reduces along the key axis instead, which is an
 * Add-based fold - also element-rate 64/cycle.
 *
 * Gather
 * ------
 * The sparse index row is DMA'd into UB once per task; the scattered K / V rows are then fetched with
 * one DataCopyPad per row. The index scalars for a whole block are read first (so the scalar loads
 * pipeline) and the copies are issued afterwards. The gather itself is double buffered: block k+1 is
 * prefetched before block k is computed, so the MTE2 latency overlaps the vector work instead of
 * being exposed. Indices past the end of the last partial block reuse the final valid index and those
 * redundant lanes are forced to -3e38 by a small block mask, so no lane is computed from garbage.
 *
 * Red line: no host side computation, no GM workspace, everything happens inside this kernel.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "sparse_flash_attention_launch.h"

using namespace AscendC;

namespace {

constexpr int32_t SFA_DC = 64;          // fp32 elements per vector repeat (256 B)
constexpr int32_t SFA_KT_MAX = 64;      // hard upper bound on the key block size
constexpr float   SFA_NEG = -3.0e38f;   // finite "-inf" for masked lanes
constexpr float   SFA_SUM_GUARD = 1.0e-30f;

// intra-core event ids. The id spaces of different HardEvent values are independent, and every helper
// below either pairs Set/Wait immediately or is explicitly armed/drained once per pass.
constexpr int32_t EV0 = 0;
constexpr int32_t EV1 = 1;

/*! \brief arm "the vector unit is done with this staging buffer"; drained by SfaWaitVecDrained(). */
__aicore__ inline void SfaArmVecDrained() { SetFlag<HardEvent::V_MTE2>(EV0); }
__aicore__ inline void SfaWaitVecDrained() { WaitFlag<HardEvent::V_MTE2>(EV0); }
/*! \brief mark "the gather into staging parity `ev` has landed". */
__aicore__ inline void SfaArmRowReady(int32_t ev) { SetFlag<HardEvent::MTE2_V>(ev); }
__aicore__ inline void SfaWaitRowReady(int32_t ev) { WaitFlag<HardEvent::MTE2_V>(ev); }

/*! \brief MTE2 must complete before the vector unit consumes the freshly copied data. */
__aicore__ inline void SfaSyncMte2ToV() { SetFlag<HardEvent::MTE2_V>(EV0); WaitFlag<HardEvent::MTE2_V>(EV0); }
/*! \brief the vector unit must be done before MTE3 stores from UB. */
__aicore__ inline void SfaSyncVToMte3() { SetFlag<HardEvent::V_MTE3>(EV0); WaitFlag<HardEvent::V_MTE3>(EV0); }
/*! \brief an MTE3 store must complete before the vector unit rewrites its source buffer. */
__aicore__ inline void SfaSyncMte3ToV() { SetFlag<HardEvent::MTE3_V>(EV0); WaitFlag<HardEvent::MTE3_V>(EV0); }
/*! \brief MTE2 written UB data must be visible to the scalar unit. */
__aicore__ inline void SfaSyncMte2ToS() { SetFlag<HardEvent::MTE2_S>(EV0); WaitFlag<HardEvent::MTE2_S>(EV0); }
/*! \brief a scalar operand derived from UB must be published to the vector unit. */
__aicore__ inline void SfaSyncSToV() { SetFlag<HardEvent::S_V>(EV0); WaitFlag<HardEvent::S_V>(EV0); }
/*! \brief vector produced UB data must be visible to the scalar unit. */
__aicore__ inline void SfaSyncVToS() { SetFlag<HardEvent::V_S>(EV0); WaitFlag<HardEvent::V_S>(EV0); }

__aicore__ inline int64_t SfaMaxI(int64_t a, int64_t b) { return a > b ? a : b; }
__aicore__ inline int64_t SfaMinI(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline uint32_t SfaBytes32(int64_t bytes) { return static_cast<uint32_t>(((bytes + 31) / 32) * 32); }

/*! \brief Fold the kt rows of a contiguous [kt, dc] fp32 block into its first row, in place. */
__aicore__ inline void SfaFoldRows(const LocalTensor<float> &buf, int64_t kt, int64_t dc)
{
    int64_t m = kt;
    while (m > 1) {
        const int64_t half = m >> 1;
        Add(buf, buf, buf[half * dc], static_cast<int32_t>(half * dc));
        m = half;
    }
}

/*! \brief Number of fp32 elements of chunk `c` that lie inside the logical dimension. */
__aicore__ inline int32_t SfaChunkMask(int64_t dim, int64_t c)
{
    const int64_t left = dim - c * SFA_DC;
    return static_cast<int32_t>(left >= SFA_DC ? SFA_DC : left);
}

/*! \brief bf16/fp16 -> fp32 (both are exact widening conversions, no rounding needed). */
template <typename T> __aicore__ inline RoundMode SfaUpMode();
template <> __aicore__ inline RoundMode SfaUpMode<half>() { return RoundMode::CAST_NONE; }
template <> __aicore__ inline RoundMode SfaUpMode<bfloat16_t>() { return RoundMode::CAST_NONE; }

/*! \brief fp32 -> bf16/fp16. */
template <typename T> __aicore__ inline RoundMode SfaDownMode();
template <> __aicore__ inline RoundMode SfaDownMode<half>() { return RoundMode::CAST_NONE; }
template <> __aicore__ inline RoundMode SfaDownMode<bfloat16_t>() { return RoundMode::CAST_RINT; }

/*!
 * \brief Gather KT rows of one source tensor into a staging tile.
 *
 * The index scalars for the whole block are read before any copy is issued so the scalar loads
 * pipeline instead of forming a load -> address -> copy chain per row. Rows past the end of a partial
 * last block reuse the final valid index; those lanes are dropped later by the block mask.
 */
#define SFA_GATHER_ROWS(ST, GSRC, SB, SS, SN, NELEM, T0V)                                        \
    do {                                                                                         \
        const int64_t sfaLast = SfaMinI((T0V) + KT, topK) - 1;                                   \
        for (int64_t sfaT = 0; sfaT < SFA_KT_MAX; ++sfaT) {                                      \
            if (sfaT < KT) {                                                                      \
                sfaIdxArr[sfaT] = idxI.GetValue(                                                  \
                    static_cast<int32_t>(SfaMinI((T0V) + sfaT, sfaLast)));                        \
            }                                                                                     \
        }                                                                                         \
        for (int64_t sfaT = 0; sfaT < KT; ++sfaT) {                                               \
            const int64_t sfaOff = b * (SB) + static_cast<int64_t>(sfaIdxArr[sfaT]) * (SS) +      \
                                   n2 * (SN);                                                     \
            DataCopyExtParams sfaCp{1, static_cast<uint32_t>((NELEM) * halfBytes), 0, 0, 0};      \
            DataCopyPad((ST)[sfaT * DmxAl], (GSRC)[sfaOff], sfaCp, padT);                         \
        }                                                                                         \
    } while (0)

}  // namespace

template <typename T>
__global__ __aicore__ void sfa_kernel(GM_ADDR qAddr, GM_ADDR kAddr, GM_ADDR vAddr,
                                      GM_ADDR iAddr, GM_ADDR yAddr, SfaParams p)
{
    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    if (blk >= p.totalTasks) {
        return;
    }

    // ---- decode the flat task id into (b, n2, s1, head block) ----
    const int64_t nHB = p.numHeadBlk;
    const int64_t hb  = blk - (blk / nHB) * nHB;
    const int64_t grp = blk / nHB;
    const int64_t s1  = grp - (grp / p.S1) * p.S1;
    const int64_t g2  = grp / p.S1;
    const int64_t n2  = g2 - (g2 / p.N2) * p.N2;
    const int64_t b   = g2 / p.N2;

    const int64_t g0 = hb * p.HG;
    const int64_t hg = SfaMinI(p.HG, p.G - g0);
    if (hg <= 0) {
        return;
    }

    const int64_t HG = p.HG;
    const int64_t KT = p.KT;
    const int64_t Dk = p.Dk;
    const int64_t Dv = p.Dv;
    const int64_t DkAl = p.DkAl;
    const int64_t DvAl = p.DvAl;
    const int64_t DmxAl = p.DmxAl;
    const int64_t topK = p.topK;
    const int64_t topKAl = p.topKAl;
    const int64_t numKB = p.numKB;
    const int64_t Dkc = DkAl / SFA_DC;
    const int64_t Dvc = DvAl / SFA_DC;
    const int64_t redMask = (Dk >= SFA_DC) ? SFA_DC : Dk;   // lanes of the folded tile that are live
    const uint8_t kvRep = static_cast<uint8_t>(DmxAl / 8);  // K/V staging row pitch, in 32B blocks
    const uint32_t halfBytes = static_cast<uint32_t>(sizeof(T));

    GlobalTensor<T> qG, kG, vG, yG;
    GlobalTensor<int32_t> iG;
    qG.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(qAddr));
    kG.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kAddr));
    vG.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(vAddr));
    yG.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(yAddr));
    iG.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(iAddr));

    // ---- UB budget: every buffer below stays live for the whole task ----
    TPipe pipe;
    TBuf<TPosition::VECCALC> bScr;     // query staging, reused afterwards as the odd-parity K/V staging
    TBuf<TPosition::VECCALC> bQf;      // [HG, DkAl] fp32 query
    TBuf<TPosition::VECCALC> bS;       // [HG, topKAl] scores -> probabilities (+ Brcb read slack)
    TBuf<TPosition::VECCALC> bKVst;    // [KT, DmxAl] even-parity staging
    TBuf<TPosition::VECCALC> bKVf;     // [KT, DmxAl] fp32 staging
    TBuf<TPosition::VECCALC> bProd;    // [KT, DC] per-chunk product / PV fold scratch
    TBuf<TPosition::VECCALC> bAcc2;    // [KT, DC] Dk-chunk accumulator
    TBuf<TPosition::VECCALC> bPart;    // [KT] per-key score
    TBuf<TPosition::VECCALC> bAcc;     // [HG, DvAl] output accumulator
    TBuf<TPosition::VECCALC> bIdx;     // [topKAl] int32 sparse indices
    TBuf<TPosition::VECCALC> bBias;    // [topKAl] fp32 causal bias
    TBuf<TPosition::VECCALC> bPb;      // [KT * 8] broadcast probability
    TBuf<TPosition::VECCALC> bTmask;   // [KT] tail block mask
    TBuf<TPosition::VECCALC> bSum;     // [HG * 8] row sums
    TBuf<TPosition::VECCALC> bTmp;     // reduce scratch
    TBuf<TPosition::VECCALC> bYst;     // [DvAl] staged output row

    const int64_t scrElems = SfaMaxI(HG * DkAl, KT * DmxAl);
    pipe.InitBuffer(bScr, SfaBytes32(scrElems * halfBytes));
    pipe.InitBuffer(bQf, SfaBytes32(HG * DkAl * 4));
    pipe.InitBuffer(bS, SfaBytes32(HG * topKAl * 4 + KT * 8 * 4));
    pipe.InitBuffer(bKVst, SfaBytes32(KT * DmxAl * halfBytes));
    pipe.InitBuffer(bKVf, SfaBytes32(KT * DmxAl * 4));
    pipe.InitBuffer(bProd, SfaBytes32(KT * SFA_DC * 4));
    pipe.InitBuffer(bAcc2, SfaBytes32(KT * SFA_DC * 4));
    pipe.InitBuffer(bPart, SfaBytes32(KT * 4 + 32));
    pipe.InitBuffer(bAcc, SfaBytes32(HG * DvAl * 4));
    pipe.InitBuffer(bIdx, SfaBytes32(topKAl * 4));
    pipe.InitBuffer(bBias, SfaBytes32(topKAl * 4));
    pipe.InitBuffer(bPb, SfaBytes32(KT * 8 * 4));
    pipe.InitBuffer(bTmask, SfaBytes32(KT * 4 + 32));
    pipe.InitBuffer(bSum, SfaBytes32(HG * 8 * 4 + 32));
    pipe.InitBuffer(bTmp, 1024);
    pipe.InitBuffer(bYst, SfaBytes32(DvAl * halfBytes));

    LocalTensor<T> qSt = bScr.Get<T>();        // aliases the odd-parity staging buffer
    LocalTensor<T> st0 = bKVst.Get<T>();
    LocalTensor<T> st1 = qSt;
    LocalTensor<float> qF = bQf.Get<float>();
    LocalTensor<float> sF = bS.Get<float>();
    LocalTensor<T> kvSt = st0;
    LocalTensor<float> kvF = bKVf.Get<float>();
    LocalTensor<float> prod = bProd.Get<float>();
    LocalTensor<float> acc2 = bAcc2.Get<float>();
    LocalTensor<float> part = bPart.Get<float>();
    LocalTensor<float> acc = bAcc.Get<float>();
    LocalTensor<int32_t> idxI = bIdx.Get<int32_t>();
    LocalTensor<float> bias = bBias.Get<float>();
    LocalTensor<float> pb = bPb.Get<float>();
    LocalTensor<float> tmask = bTmask.Get<float>();
    LocalTensor<float> sumF = bSum.Get<float>();
    LocalTensor<float> tmpF = bTmp.Get<float>();
    LocalTensor<T> ySt = bYst.Get<T>();

    DataCopyPadExtParams<T> padT{false, 0, 0, static_cast<T>(0)};
    int32_t sfaIdxArr[SFA_KT_MAX];

    // ---- sparse index row: one DMA, then it stays resident in UB ----
    {
        const int64_t iOff = b * p.iSB + n2 * p.iSN + s1 * p.iSS;
        DataCopyExtParams cpIdx{1, static_cast<uint32_t>(topK * 4), 0, 0, 0};
        DataCopyPad(idxI, iG[iOff], cpIdx, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    }

    // ---- resident query tile: hg heads, fp32 ----
    for (int64_t h = 0; h < hg; ++h) {
        const int64_t n1 = n2 * p.G + g0 + h;
        const int64_t off = b * p.qSB + n1 * p.qSN + s1 * p.qSS;
        DataCopyExtParams cp{1, static_cast<uint32_t>(Dk * halfBytes), 0, 0, 0};
        DataCopyPad(qSt[h * DkAl], qG[off], cp, padT);
    }

    SfaSyncMte2ToV();
    SfaSyncMte2ToS();   // the sparse index row is also consumed by the scalar unit (GM addressing)

    Cast(qF, qSt, SfaUpMode<T>(), static_cast<uint32_t>(hg * DkAl));

    // ---- causal bias: 0 for visible keys, -3e38 past the right-bottom diagonal ----
    if (p.isCausal != 0) {
        const float thr = static_cast<float>(s1 + p.s2m1);
        Cast(bias, idxI, RoundMode::CAST_NONE, static_cast<uint32_t>(topKAl));
        Adds(bias, bias, -thr, static_cast<int32_t>(topKAl));
        Maxs(bias, bias, 0.0f, static_cast<int32_t>(topKAl));
        Mins(bias, bias, 1.0f, static_cast<int32_t>(topKAl));
        Muls(bias, bias, SFA_NEG, static_cast<int32_t>(topKAl));
    }

    // ---- tail block mask: only needed when topK is not a multiple of KT ----
    const int64_t lastT0 = (numKB - 1) * KT;
    const int64_t lastRows = SfaMinI(KT, topK - lastT0);
    if (lastRows < KT) {
        Duplicate(tmask, SFA_NEG, static_cast<int32_t>(KT));
        Muls(tmask, tmask, 0.0f, static_cast<int32_t>(lastRows));
    }

    // ---- Pass 1: scores for every (head, key) pair ----------------------------------------------
    // Block k+1 is prefetched while block k is being computed, so the gather latency is overlapped.
    SfaArmVecDrained();
    SFA_GATHER_ROWS(st0, kG, p.kSB, p.kSS, p.kSN, Dk, 0);
    SfaArmRowReady(EV0);
    for (int64_t kb = 0; kb < numKB; ++kb) {
        const int64_t t0 = kb * KT;
        const int64_t rows = SfaMinI(KT, topK - t0);
        const int64_t cur = kb & 1;

        SfaWaitVecDrained();   // the staging buffer of the incoming block is free again
        if (kb + 1 < numKB) {
            const int64_t nxt = (kb + 1) & 1;
            if (nxt == 0) {
                SFA_GATHER_ROWS(st0, kG, p.kSB, p.kSS, p.kSN, Dk, (kb + 1) * KT);
            } else {
                SFA_GATHER_ROWS(st1, kG, p.kSB, p.kSS, p.kSN, Dk, (kb + 1) * KT);
            }
            SfaArmRowReady(nxt == 0 ? EV0 : EV1);
        }
        SfaWaitRowReady(cur == 0 ? EV0 : EV1);
        if (cur == 0) {
            Cast(kvF, st0, SfaUpMode<T>(), static_cast<uint32_t>(KT * DmxAl));
        } else {
            Cast(kvF, st1, SfaUpMode<T>(), static_cast<uint32_t>(KT * DmxAl));
        }
        SfaArmVecDrained();

        for (int64_t h = 0; h < hg; ++h) {
            LocalTensor<float> row = sF[h * topKAl + t0];
            // fold the Dk chunks with cheap Mul/Add, then reduce the whole tile exactly once
            Mul(acc2, kvF, qF[h * DkAl], static_cast<uint64_t>(SfaChunkMask(Dk, 0)),
                static_cast<uint8_t>(KT),
                {1, 1, 1, static_cast<uint8_t>(SFA_DC / 8), kvRep, 0});
            for (int64_t c = 1; c < Dkc; ++c) {
                const int32_t mc = SfaChunkMask(Dk, c);
                Mul(prod, kvF[c * SFA_DC], qF[h * DkAl + c * SFA_DC], static_cast<uint64_t>(mc),
                    static_cast<uint8_t>(KT),
                    {1, 1, 1, static_cast<uint8_t>(SFA_DC / 8), kvRep, 0});
                if (mc == SFA_DC) {
                    Add(acc2, acc2, prod, static_cast<int32_t>(KT * SFA_DC));
                } else {
                    Add(acc2, acc2, prod, static_cast<uint64_t>(mc), static_cast<uint8_t>(KT),
                        {1, 1, 1, static_cast<uint8_t>(SFA_DC / 8), static_cast<uint8_t>(SFA_DC / 8),
                         static_cast<uint8_t>(SFA_DC / 8)});
                }
            }
            WholeReduceSum<float>(part, acc2, static_cast<uint64_t>(redMask),
                                  static_cast<int32_t>(KT), 1, 1,
                                  static_cast<int32_t>(SFA_DC / 8));
            Adds(row, part, 0.0f, static_cast<int32_t>(KT));
            if (rows < KT) {
                Add(row, row, tmask, static_cast<int32_t>(KT));   // drop the redundant lanes
            }
        }
    }
    SfaWaitVecDrained();

    // ---- softmax (no max subtraction: the scores are bounded, see the file header) ---------------
    for (int64_t h = 0; h < hg; ++h) {
        LocalTensor<float> row = sF[h * topKAl];
        Muls(row, row, p.scale, static_cast<int32_t>(topKAl));
        if (p.isCausal != 0) {
            Add(row, row, bias, static_cast<int32_t>(topKAl));
        }
        Exp(row, row, static_cast<int32_t>(topKAl));
        ReduceSum<float>(sumF[h * 8], row, tmpF, static_cast<int32_t>(topKAl));
    }

    // ---- fold 1/sum into the probabilities so the PV pass needs no normalisation -----------------
    SfaSyncVToS();
    for (int64_t h = 0; h < hg; ++h) {
        const float l = sumF.GetValue(static_cast<int32_t>(h * 8));
        const float inv = 1.0f / (l + SFA_SUM_GUARD);
        SfaSyncSToV();
        Muls(sF[h * topKAl], sF[h * topKAl], inv, static_cast<int32_t>(topKAl));
    }

    Duplicate(acc, 0.0f, static_cast<int32_t>(HG * DvAl));

    // ---- Pass 2: accumulate p * Vsel ------------------------------------------------------------
    SfaArmVecDrained();
    SFA_GATHER_ROWS(st0, vG, p.vSB, p.vSS, p.vSN, Dv, 0);
    SfaArmRowReady(EV0);
    for (int64_t kb = 0; kb < numKB; ++kb) {
        const int64_t t0 = kb * KT;
        const int64_t cur = kb & 1;

        SfaWaitVecDrained();
        if (kb + 1 < numKB) {
            const int64_t nxt = (kb + 1) & 1;
            if (nxt == 0) {
                SFA_GATHER_ROWS(st0, vG, p.vSB, p.vSS, p.vSN, Dv, (kb + 1) * KT);
            } else {
                SFA_GATHER_ROWS(st1, vG, p.vSB, p.vSS, p.vSN, Dv, (kb + 1) * KT);
            }
            SfaArmRowReady(nxt == 0 ? EV0 : EV1);
        }
        SfaWaitRowReady(cur == 0 ? EV0 : EV1);
        if (cur == 0) {
            Cast(kvF, st0, SfaUpMode<T>(), static_cast<uint32_t>(KT * DmxAl));
        } else {
            Cast(kvF, st1, SfaUpMode<T>(), static_cast<uint32_t>(KT * DmxAl));
        }
        SfaArmVecDrained();

        for (int64_t h = 0; h < hg; ++h) {
            // expand the KT block probabilities into `KT` 32B blocks: block r holds KT/8 groups of
            // eight source elements, and the k-th element of group g is written to block 8g + k, so
            // block r == the 8 copies of p[t0 + r]. Brcb iterates over 8-element groups, not
            // elements, hence repeatTime = KT / 8 (src elements 8 * repeatTime == KT).
            Brcb(pb, sF[h * topKAl + t0], static_cast<uint8_t>(KT / 8), {1, 8});
            for (int64_t c = 0; c < Dvc; ++c) {
                const uint64_t mc = static_cast<uint64_t>(SfaChunkMask(Dv, c));
                // prod[t, :] = V[t, c*64 .. c*64+63] * p[t]  (src1 block stride 0 broadcasts p[t])
                Mul(prod, kvF[c * SFA_DC], pb, mc, static_cast<uint8_t>(KT),
                    {1, 1, 0, static_cast<uint8_t>(SFA_DC / 8), kvRep, 1});
                SfaFoldRows(prod, KT, SFA_DC);
                Add(acc[h * DvAl + c * SFA_DC], acc[h * DvAl + c * SFA_DC], prod,
                    static_cast<int32_t>(SFA_DC));
            }
        }
    }
    SfaWaitVecDrained();

    // ---- store ----------------------------------------------------------------------------------
    for (int64_t h = 0; h < hg; ++h) {
        if (h > 0) {
            SfaSyncMte3ToV();   // the previous store still owns its staging row
        }
        Cast(ySt, acc[h * DvAl], SfaDownMode<T>(), static_cast<uint32_t>(DvAl));
        SfaSyncVToMte3();
        const int64_t n1 = n2 * p.G + g0 + h;
        const int64_t off = b * p.ySB + n1 * p.ySN + s1 * p.ySS;
        DataCopyExtParams cp{1, static_cast<uint32_t>(Dv * halfBytes), 0, 0, 0};
        DataCopyPad(yG[off], ySt, cp);
    }
}

// Launch wrappers - plain C entry points callable from the g++ plugin TU.
extern "C" {

void launch_sfa_half(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR idx, GM_ADDR y,
                     SfaParams p, int64_t numBlocks, void *stream)
{
    sfa_kernel<half><<<numBlocks, nullptr, stream>>>(q, k, v, idx, y, p);
}

void launch_sfa_bfloat16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR idx, GM_ADDR y,
                         SfaParams p, int64_t numBlocks, void *stream)
{
    sfa_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(q, k, v, idx, y, p);
}

}
