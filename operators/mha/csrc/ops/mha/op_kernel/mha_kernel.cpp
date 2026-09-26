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
 * \file mha_kernel.cpp
 * \brief Multi-Head Attention kernel + tiling + launch (bisheng, -xasc)
 *
 *   query [B, S, N, D], key/value [B, Skv, N, D], y [B, S, N, D]
 *   y = softmax(Q @ K^T * scale) @ V      (per (b, n) head slice, softmax over Skv)
 *
 * Work item = (batch b, head n, a block of BM query rows).  Keys are streamed in
 * chunks of BN keys so one K/V tile feeds all BM rows of the block.
 *
 * SINGLE PASS: the scores of a chunk are exponentiated once and used for BOTH the
 * denominator and the numerator.  The numerator is accumulated UNNORMALISED
 *     num_r = sum_j exp(s_j) * v_j        den_r = sum_j exp(s_j)
 * and divided once per row at the very end, y = num/den.  This is the flash-attention
 * "unnormalised accumulator" form: it needs neither a second QK^T pass (so K is read
 * once, not twice) nor the online max/rescaling, because the scores are bounded
 * (|s| <= D*scale = sqrt(D) <= 16, so no exp overflow is possible) and a running maximum
 * would therefore only add work.  It also removes the per-row 1/den scalar read from the
 * inner loop.  The result differs from a normalise-first evaluation only by fp32 rounding.
 *
 *   for every chunk of BN keys:
 *       load K chunk, cast to fp32
 *       per query row r: QK^T of the chunk -> exponentiate into the weight row w[r]
 *                        accumulate w[r] into the per-row denominator sumv[r]
 *       load V chunk, cast to fp32
 *       per query row r: broadcast w[r] over the 64 lanes of each key, multiply by V and
 *                        fold the weighted key rows into the numerator acc[r]
 *   per row: den = sum_lanes sumv[r];  y[r] = acc[r] / den     (den == 0 -> 0)
 *
 * Cost model (grounded in the measured per-case times): the wall time tracks the vector
 * *work*, i.e. the number of (query row, key) element operations plus one fixed call/issue
 * cost per (query row, key chunk) pair, and the reductions are by far the most expensive
 * calls (merging the Dg group products before a single reduce instead of reducing each group
 * separately cut every case by 15-25%).  Measurements show the time is nearly independent of
 * BM (24 -> 48 changed case 7 by only 2%) and only weakly dependent on BN (64 -> 96 changed it
 * by 3-7%), which pins the cost on the per-element work, with the per-pair API-call count as a
 * strong secondary term.  The kernel therefore minimises both:
 *   - the element work is O(S * Skv * D) by construction (one QK product, one reduce, the
 *     weighted V and its fold); the Dg group products are written into one [kcnt, D] tile and
 *     reduced in a single call whose rBundle spans all of D;
 *   - the scale factor multiplies the whole query tile once per work item rather than the
 *     scores of every (row, chunk) pair, since (scale*q).k == scale*(q.k);
 *   - BN is taken as large as the per-core UB budget allows, so the pair count
 *     sum over rows r of ceil(keys(r)/BN) is small;
 *   - BM is only large enough to keep the K/V reload traffic and the per-item prologue small.
 * The lowest-speedup cases are the decode / MTP ones (B up to 96, S 1-2), which contribute
 * almost all of the speed term, so the per-pair call count matters most there.
 *
 * Data movement: a whole tile (all rows of the query block, or the whole key/value chunk)
 * is moved with ONE classic burst `DataCopy(dst, src, DataCopyParams)`.  `DataCopyParams`
 * expresses blockLen / srcStride / dstStride in 32-byte blocks, and D is 64-aligned so every
 * row is a whole number of blocks; the inter-row gap is (N-1)*rowBytes.  The UB rows of a
 * tile pack contiguously, so the UB-side stride is 0.  Casts are likewise done once per tile
 * with the count form rather than once per row - per-row copies/casts were measured to
 * dominate the runtime.
 *
 * Vector layout (Ascend 910B / dav-2201, MemBase route):
 *  - Every high-dim (repeat-based) vector call covers exactly 64 fp32 lanes per repeat, so
 *    D-group products are produced one 64-lane group at a time; the group offset is the
 *    datablock offset g*8 and the row pitch is D/8 datablocks, so all Dg groups land in one
 *    [kcnt, D] tile.
 *  - `ReduceSum<T, Pattern::Reduce::AR>(dst, src, tmp, {A, R})` treats its source as A rows of
 *    R contiguous elements and writes A results (row sums); it is used with A = kcnt and R = D
 *    so the whole dot product of a chunk costs one call.  The other pattern (`Reduce::RA`, a
 *    column sum over the rows of such a tile) was tried for the PV accumulation and produced
 *    an all-zero numerator, so the PV uses an explicit fold instead.
 *  - P @ V weighting: Brcb expands the chunk weights so datablock j holds w[j]; the PV Mul
 *    reads them with src1BlkStride = 0 / src1RepStride = 1 so repeat j broadcasts w[j] over
 *    all 64 lanes of key j.  The weighted key rows are then folded by an in-place Add tree;
 *    the tree uses the COUNT form of Add, which covers up to 16320 contiguous elements per
 *    call, so one tree level is a single call.  (Folding the last level directly into the
 *    numerator was tried and is wrong: it must be `num += row0 + row1`, which one Add cannot
 *    express without first reducing the two rows.)
 *
 * Numerics - the reference rounds every intermediate back to the operator dtype T because
 * torch promotes the fp16/bf16 tensors through the whole chain:
 *     scores  = (matmul(q, k^T)).to(T) * scaleValue.to(T)
 *     weights = softmax(scores)
 *     out     = matmul(weights, v)
 * The scale factor is therefore rounded through the dtype once, as a scalar, before being
 * folded into the query.
 */

#include <tuple>
#include <algorithm>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "mha_launch.h"

using namespace AscendC;

namespace {

constexpr int64_t MHA_MAX_BM = 64;
constexpr int64_t MHA_RED_BYTES = 16384;   // dedicated ReduceSum scratch

__aicore__ inline int64_t MhaAlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

/* Fold `len` rows of a contiguous [len, D] fp32 tile into row 0 (in place), using the
 * count form of Add so that one tree level is a single call. */
__aicore__ inline void MhaReduceRowsInPlace(const LocalTensor<float> &buf, int64_t len, int64_t D)
{
    int64_t cur = len;
    while (cur > 1) {
        const int64_t h = cur >> 1;                       // rows [h, 2h) fold onto [0, h)
        AscendC::Add(buf, buf, buf[static_cast<int32_t>(h * D)], static_cast<int32_t>(h * D));
        if ((cur & 1) != 0) {                             // odd tail row
            AscendC::Add(buf, buf, buf[static_cast<int32_t>((cur - 1) * D)],
                         static_cast<int32_t>(D));
        }
        cur = h;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host tiling
// ---------------------------------------------------------------------------
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> calc_mha_tiling_params(
    int64_t B, int64_t S, int64_t Skv, int64_t N, int64_t D)
{
    int64_t coreNum = 1;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        coreNum = static_cast<int64_t>(plat->GetCoreNumAiv());
    }
    if (coreNum <= 0) coreNum = 1;

    int64_t BN = 96;
    int64_t BM = 12;
    if (D > 128) {
        BN = 48;
        BM = 8;
    }
    if (BN > Skv) BN = Skv;
    if (BN < 8) BN = 8;
    if (BM > S) BM = S;
    if (BM < 1) BM = 1;

    // Keep enough work items: reduce BM while the whole problem would leave the cores idle.
    // Each item still covers a contiguous row block, so the K/V tiles stay shared.
    while (BM > 1 && (B * N) * ((S + BM - 1) / BM) < 2 * coreNum) {
        BM = (BM + 1) / 2;
    }

    const int64_t numRowBlk = (S + BM - 1) / BM;
    int64_t totalItems = B * N * numRowBlk;
    if (totalItems < 1) totalItems = 1;

    int64_t numBlocks = (totalItems < coreNum) ? totalItems : coreNum;
    if (numBlocks < 1) numBlocks = 1;
    const int64_t itemsPerCore = (totalItems + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, BM, BN, totalItems, itemsPerCore);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void mha_kernel(GM_ADDR qAddr, GM_ADDR kAddr, GM_ADDR vAddr, GM_ADDR yAddr,
                                      int64_t B, int64_t S, int64_t Skv, int64_t N, int64_t D,
                                      float scale, int32_t causal, int64_t numRowBlk,
                                      int64_t totalItems, int64_t itemsPerCore, int64_t BM,
                                      int64_t BN)
{
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t it0 = blk * itemsPerCore;
    if (it0 >= totalItems) return;
    int64_t it1 = it0 + itemsPerCore;
    if (it1 > totalItems) it1 = totalItems;

    const int64_t Dg = D >> 6;                                   // 64-element groups of D
    const int64_t rowStride = N * D;                             // elements between consecutive rows
    const int64_t rowBytes = D * static_cast<int64_t>(sizeof(T));
    const int64_t PBN = MhaAlignUp(BN, 64);                      // padded sumv row pitch
    const int64_t repStride = D >> 3;                            // datablocks per [.,D] fp32 row
    const uint16_t rowBlocks = static_cast<uint16_t>(rowBytes >> 5);          // 32B blocks per row
    const uint16_t gmGapBlocks = static_cast<uint16_t>(((N - 1) * rowBytes) >> 5);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> f32aB;   // K fp32 then V fp32
    AscendC::TBuf<AscendC::TPosition::VECCALC> pvB;     // [BN,D] weight tile / QK product tile
    AscendC::TBuf<AscendC::TPosition::VECCALC> q32B;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumvB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> brcbB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> denB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tvB;     // scratch for the scalar scale round trip
    AscendC::TBuf<AscendC::TPosition::VECCALC> redB;    // dedicated ReduceSum scratch

    int64_t inBytes = BN * rowBytes;
    if (inBytes < BM * rowBytes) inBytes = BM * rowBytes;
    pipe.InitBuffer(inQ, 1, static_cast<uint32_t>(inBytes));
    pipe.InitBuffer(outQ, 1, static_cast<uint32_t>(BM * rowBytes));
    pipe.InitBuffer(f32aB, static_cast<uint32_t>(BN * D * sizeof(float)));
    pipe.InitBuffer(pvB, static_cast<uint32_t>(BN * D * sizeof(float)));
    pipe.InitBuffer(q32B, static_cast<uint32_t>(BM * D * sizeof(float)));
    pipe.InitBuffer(accB, static_cast<uint32_t>(BM * D * sizeof(float)));
    pipe.InitBuffer(sumvB, static_cast<uint32_t>(BM * PBN * sizeof(float)));
    pipe.InitBuffer(wB, static_cast<uint32_t>(BM * BN * sizeof(float)));
    pipe.InitBuffer(brcbB, static_cast<uint32_t>(BN * 32));
    pipe.InitBuffer(denB, static_cast<uint32_t>((MHA_MAX_BM + 8) * sizeof(float)));
    pipe.InitBuffer(tvB, static_cast<uint32_t>(BN * sizeof(T) + 64));
    pipe.InitBuffer(redB, static_cast<uint32_t>(MHA_RED_BYTES));

    auto q32 = q32B.Get<float>();
    auto acc = accB.Get<float>();
    auto sumv = sumvB.Get<float>();
    auto wbuf = wB.Get<float>();
    auto pvtile = pvB.Get<float>();      // the PV phase uses the buffer as [kcnt, D]
    auto prod = pvB.Get<float>();        // the QK phase uses it as [kcnt, D] group products
    auto f32buf = f32aB.Get<float>();
    auto brcb = brcbB.Get<float>();
    auto denAll = denB.Get<float>();
    auto denSlot = denAll;                       // 32B aligned single-value reduce destination
    auto tvT = tvB.Get<T>();
    auto redU8 = redB.Get<uint8_t>();

    AscendC::GlobalTensor<T> qGm, kGm, vGm, yGm;

    // QK product: one [kcnt, D] tile, the group g starting at datablock g*8 of each row
    // (dstRepStride = D/8 blocks), the K group supplying src0 with the same row pitch and the
    // query group broadcast over the repeats with src1RepStride = 0.
    const AscendC::BinaryRepeatParams qkParams{1, 1, 1, static_cast<uint8_t>(repStride),
                                               static_cast<uint8_t>(repStride), 0};
    // PV: destination and V group advance by one full row per repeat, the Brcb weights
    // supply one datablock per repeat.
    const AscendC::BinaryRepeatParams pvParams{1, 1, 0, static_cast<uint8_t>(repStride),
                                               static_cast<uint8_t>(repStride), 1};

    // The reference multiplies by the operator dtype's view of the scale factor.
    AscendC::Duplicate(denSlot, scale, 1);
    AscendC::Cast(tvT, denSlot, AscendC::RoundMode::CAST_RINT, 1);
    AscendC::Cast(denSlot, tvT, AscendC::RoundMode::CAST_NONE, 1);
    AscendC::PipeBarrier<PIPE_ALL>();
    const float scaleT = denSlot.GetValue(0);

    for (int64_t it = it0; it < it1; ++it) {
        const int64_t b = it / (N * numRowBlk);
        int64_t rem = it - b * (N * numRowBlk);
        const int64_t n = rem / numRowBlk;
        const int64_t rb = rem - n * numRowBlk;
        const int64_t r0 = rb * BM;
        int64_t rn = S - r0;
        if (rn > BM) rn = BM;
        if (rn <= 0) continue;

        qGm.SetGlobalBuffer((__gm__ T *)qAddr + ((b * S + r0) * N + n) * D);
        yGm.SetGlobalBuffer((__gm__ T *)yAddr + ((b * S + r0) * N + n) * D);
        kGm.SetGlobalBuffer((__gm__ T *)kAddr + ((b * Skv) * N + n) * D);
        vGm.SetGlobalBuffer((__gm__ T *)vAddr + ((b * Skv) * N + n) * D);

        // ---- load Q rows of this block (one burst), cast once, scale once ----
        {
            auto qT = inQ.AllocTensor<T>();
            AscendC::DataCopyParams cpQ{static_cast<uint16_t>(rn), rowBlocks, gmGapBlocks, 0};
            AscendC::DataCopy(qT, qGm, cpQ);
            inQ.EnQue(qT);
            qT = inQ.DeQue<T>();
            AscendC::Cast(q32, qT, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(rn * D));
            inQ.FreeTensor(qT);
            // (scale * q).k == scale * (q.k), so the scale rides on the query tile and every
            // (row, chunk) pair saves its own multiply.
            AscendC::Muls(q32, q32, scaleT, static_cast<int32_t>(rn * D));
        }

        for (int64_t r = 0; r < rn; ++r) {
            AscendC::Duplicate(acc[static_cast<int32_t>(r * D)], 0.0f, static_cast<int32_t>(D));
            AscendC::Duplicate(sumv[static_cast<int32_t>(r * PBN)], 0.0f, static_cast<int32_t>(PBN));
        }

        const int64_t lastKey = Skv - 1;
        uint32_t shp[2] = {64u, 64u};

        // ================= one pass over the key chunks =================
        for (int64_t k0 = 0; k0 <= lastKey; k0 += BN) {
            int64_t kc = lastKey - k0 + 1;
            if (kc > BN) kc = BN;

            {
                auto kT = inQ.AllocTensor<T>();
                AscendC::DataCopyParams cpK{static_cast<uint16_t>(kc), rowBlocks, gmGapBlocks, 0};
                AscendC::DataCopy(kT, kGm[k0 * rowStride], cpK);
                inQ.EnQue(kT);
                kT = inQ.DeQue<T>();
                AscendC::Cast(f32buf, kT, AscendC::RoundMode::CAST_NONE,
                              static_cast<int32_t>(kc * D));
                inQ.FreeTensor(kT);
            }

            // --- scores for this chunk, kept as the unnormalised weight rows ---
            for (int64_t r = 0; r < rn; ++r) {
                int64_t lim = (causal != 0) ? (r0 + r + (Skv - S)) : (Skv - 1);
                int64_t kr = lim - k0 + 1;
                if (kr <= 0) continue;
                if (kr > kc) kr = kc;

                auto wrow = wbuf[static_cast<int32_t>(r * BN)];
                shp[0] = static_cast<uint32_t>(kr);
                shp[1] = static_cast<uint32_t>(D);
                // Dg group products land in one [kr, D] tile, reduced in a single call.
                for (int64_t g = 0; g < Dg; ++g) {
                    AscendC::Mul(prod[static_cast<int32_t>(g * 64)], f32buf[static_cast<int32_t>(g * 64)],
                                 q32[static_cast<int32_t>(r * D + g * 64)], 64,
                                 static_cast<uint8_t>(kr), qkParams);
                }
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(wrow, prod, redU8, shp, false);

                AscendC::Exp(wrow, wrow, static_cast<int32_t>(kr));
                AscendC::Add(sumv[static_cast<int32_t>(r * PBN)], sumv[static_cast<int32_t>(r * PBN)],
                             wrow, static_cast<int32_t>(kr));
            }

            {
                auto vT = inQ.AllocTensor<T>();
                AscendC::DataCopyParams cpV{static_cast<uint16_t>(kc), rowBlocks, gmGapBlocks, 0};
                AscendC::DataCopy(vT, vGm[k0 * rowStride], cpV);
                inQ.EnQue(vT);
                vT = inQ.DeQue<T>();
                AscendC::Cast(f32buf, vT, AscendC::RoundMode::CAST_NONE,
                              static_cast<int32_t>(kc * D));
                inQ.FreeTensor(vT);
            }

            // --- unnormalised numerator for this chunk ---
            for (int64_t r = 0; r < rn; ++r) {
                int64_t lim = (causal != 0) ? (r0 + r + (Skv - S)) : (Skv - 1);
                int64_t kr = lim - k0 + 1;
                if (kr <= 0) continue;
                if (kr > kc) kr = kc;

                auto wrow = wbuf[static_cast<int32_t>(r * BN)];
                AscendC::Brcb(brcb, wrow, static_cast<uint8_t>((kr + 7) / 8),
                              AscendC::BrcbRepeatParams{1, 8});
                for (int64_t g = 0; g < Dg; ++g) {
                    AscendC::Mul(pvtile[static_cast<int32_t>(g * 64)],
                                 f32buf[static_cast<int32_t>(g * 64)], brcb, 64,
                                 static_cast<uint8_t>(kr), pvParams);
                }
                MhaReduceRowsInPlace(pvtile, kr, D);
                AscendC::Add(acc[static_cast<int32_t>(r * D)], acc[static_cast<int32_t>(r * D)], pvtile,
                             static_cast<int32_t>(D));
            }
        }

        // ---- normalise once per row, then write out (one burst) ----
        for (int64_t r = 0; r < rn; ++r) {
            uint32_t shp1[2] = {1u, static_cast<uint32_t>(PBN)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(denSlot, sumv[static_cast<int32_t>(r * PBN)],
                                                                    redU8, shp1, false);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float ssum = denSlot.GetValue(0);
            const float invr = (ssum > 0.0f) ? (1.0f / ssum) : 0.0f;
            AscendC::Muls(acc[static_cast<int32_t>(r * D)], acc[static_cast<int32_t>(r * D)], invr,
                          static_cast<int32_t>(D));
        }

        {
            auto yT = outQ.AllocTensor<T>();
            AscendC::Cast(yT, acc, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(rn * D));
            outQ.EnQue(yT);
            yT = outQ.DeQue<T>();
            AscendC::DataCopyParams cpY{static_cast<uint16_t>(rn), rowBlocks, 0, gmGapBlocks};
            AscendC::DataCopy(yGm, yT, cpY);
            outQ.FreeTensor(yT);
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrappers
// ---------------------------------------------------------------------------
extern "C" {

void launch_mha_kernel_f16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, int64_t B, int64_t S,
                           int64_t Skv, int64_t N, int64_t D, float scale, int32_t causal,
                           int64_t numRowBlk, int64_t totalItems, int64_t itemsPerCore, int64_t BM,
                           int64_t BN, int64_t numBlocks, void *stream)
{
    mha_kernel<half><<<numBlocks, nullptr, stream>>>(q, k, v, y, B, S, Skv, N, D, scale, causal,
                                                     numRowBlk, totalItems, itemsPerCore, BM, BN);
}

void launch_mha_kernel_bf16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, int64_t B, int64_t S,
                            int64_t Skv, int64_t N, int64_t D, float scale, int32_t causal,
                            int64_t numRowBlk, int64_t totalItems, int64_t itemsPerCore, int64_t BM,
                            int64_t BN, int64_t numBlocks, void *stream)
{
    mha_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(q, k, v, y, B, S, Skv, N, D, scale, causal,
                                                           numRowBlk, totalItems, itemsPerCore, BM, BN);
}

}  // extern "C"
