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
 * \file transpose_host.h
 * \brief Host-side tiling for the transpose operator (compiled into the kernel TU by bisheng).
 *
 * The operator is a permutation of the input viewed as [numel/Dcol][Dcol]; the tiling normalises
 * the output to [H][Dcol][U] and, for the gather path, picks the tile geometry (Ta, Tb, TC) that
 * maximises a throughput estimate while fitting in UB.
 *
 * Role of the two run extents:
 *   - Tb = 1 : the tile covers Ta <= P consecutive values of the innermost run dim a; chunking
 *              inside dim a is the classic shape and the only option when dim b is absent.
 *   - Ta = P : the tile covers the whole of dim a crossed with Tb values of dim b (the second run
 *              dim, expanded stride P).  This keeps K = P*Tb large, which is what stops a narrow
 *              innermost run dim from forcing a tiny write blockLen.
 *
 * Batch (NB): when the tile already covers the whole run span and every column (the packed case),
 * consecutive output planes are adjacent in GM, so NB consecutive h values inside one page can be
 * handled in a single queue round: one read group per sub-tile, NB gathers sharing the same offset
 * table (only the source base shifts), and one contiguous write.  That amortises the per-round
 * synchronisation and keeps many scattered read requests in flight, which is what a small tile
 * (few hundred threads of useful work) otherwise pays for on every round.
 */

#ifndef TRANSPOSE_HOST_H
#define TRANSPOSE_HOST_H

#include <cstdint>

#include "platform/platform_ascendc.h"

#include "transpose_launch.h"

static constexpr int64_t TP_TBL_MAX = 4096;      // gather offset table entries per tile
static constexpr int64_t TP_RESERVE = 40 * 1024; // UB bytes left to the framework

static inline int64_t TpAlignUpH(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

// UB bytes needed by one tile of the given shape.  The read queue is deepened for small tiles
// (their scattered reads expose DRAM latency, the gather is too short to hide it).
static inline int64_t TpNeedUb(int64_t K, int64_t TC, int64_t Pj, int64_t gusz, int64_t esz,
                               int64_t inPitch)
{
    int64_t depthIn = (K * inPitch <= 8192) ? 4 : 2;
    if (esz == 1) {
        return (depthIn + 2) * K * inPitch + 2 * TC * TpAlignUpH(K, 32) + 10 * TC * Pj + 8192;
    }
    return depthIn * K * inPitch + 2 * TC * Pj * gusz + 8 * TC * Pj + 8192;
}

void calc_transpose_params(int64_t nd, int64_t esz, int64_t numel, const int64_t* sizes,
                           const int64_t* perm, TPParams* out)
{
    int64_t dd[4] = {0, 0, 0, 0};
    for (int64_t i = 0; i < nd; ++i) {
        int64_t w = i / 2;
        dd[w] |= (sizes[i] & 0xFFFFFFFFLL) << (32 * (i % 2));
    }
    int64_t pp = 0;
    for (int64_t i = 0; i < nd; ++i) {
        pp |= ((perm[i] & 0xFLL) << (4 * i));
    }
    out->n = nd;
    out->d0 = dd[0];
    out->d1 = dd[1];
    out->d2 = dd[2];
    out->d3 = dd[3];
    out->pp = pp;
    out->numel = numel;

    int64_t osz[8];
    int64_t Cj[8];
    for (int64_t i = 0; i < 8; ++i) {
        osz[i] = 1;
        Cj[i] = 0;
    }
    int64_t Dcol = sizes[nd - 1];
    int64_t q = 0;
    for (int64_t i = 0; i < nd; ++i) {
        osz[i] = sizes[perm[i]];
        if (perm[i] == nd - 1) q = i;
    }
    int64_t H = 1;
    int64_t U = 1;
    for (int64_t j = 0; j < q; ++j) H *= osz[j];
    for (int64_t j = q + 1; j < nd; ++j) U *= osz[j];
    for (int64_t j = 0; j < nd; ++j) {
        if (j == q) continue;
        int64_t c = 1;
        for (int64_t k = perm[j] + 1; k <= nd - 2; ++k) c *= sizes[k];
        Cj[j] = c;
    }
    // page: the innermost non-degenerate h dim; consecutive h inside a page are RowStep rows apart
    int64_t PageSize = 1;
    for (int64_t j = q - 1; j >= 0; --j) {
        if (osz[j] > 1) {
            PageSize = osz[j];
            break;
        }
    }

    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = plat->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t ub = (int64_t)ubSize - TP_RESERVE;
    if (ub < 32 * 1024) ub = 32 * 1024;

    out->RA = 1;
    out->K = 1;
    out->Pk = 1;
    out->Pj = 1;
    out->nch = 1;
    out->nlc = 1;
    out->ncc = 1;
    out->nTiles = 1;
    out->nBlocks = 1;
    out->Ta = 1;
    out->Tb = 1;
    out->packed = 0;
    out->overA = 0;
    out->NB = 1;

    if (U == 1) {
        // ---- pure row permutation ----
        out->mode = 0;
        int64_t TC = Dcol;
        int64_t cap = TP_TBL_MAX / (esz > 0 ? esz : 1);
        if (cap < 1) cap = 1;
        if (TC > cap) TC = cap;
        if (TC < 1) TC = 1;
        int64_t rowBytes = TpAlignUpH(TC * esz, 32);
        int64_t RA = (ub / 4) / (rowBytes > 0 ? rowBytes : 1);
        if (RA > 4096) RA = 4096;
        if (RA > PageSize) RA = PageSize;
        if (RA > H) RA = H;
        if (RA < 1) RA = 1;
        out->TC = TC;
        out->RA = RA;
        int64_t ncc = (Dcol + TC - 1) / TC;
        if (ncc < 1) ncc = 1;
        out->ncc = ncc;
        int64_t nWB = (PageSize + RA - 1) / RA;
        if (nWB < 1) nWB = 1;
        int64_t nPages = H / PageSize;
        if (nPages < 1) nPages = 1;
        out->nTiles = nPages * nWB * ncc;
        if (out->nTiles < 1) out->nTiles = 1;
        out->nBlocks = coreNum;
        if (out->nBlocks > out->nTiles) out->nBlocks = out->nTiles;
        return;
    }

    // ---- gather transpose ----
    out->mode = 1;
    int64_t P = osz[nd - 1];
    int64_t gusz = (esz == 8) ? 4 : esz;
    int64_t upe = (esz == 8) ? 2 : 1;

    // the second run dim: the largest j with q < j < n-1 and osz[j] > 1 (its expanded stride
    // is then exactly P, because every dim between it and n-1 has extent 1)
    int64_t bIdx = -1;
    for (int64_t j = nd - 2; j > q; --j) {
        if (osz[j] > 1) {
            bIdx = j;
            break;
        }
    }
    bool hasB = (bIdx >= 0);
    int64_t bExt = hasB ? osz[bIdx] : 1;
    int64_t Ca = Cj[nd - 1];
    int64_t Cb = hasB ? Cj[bIdx] : 0;
    // option A costs P sub-DMAs (one per dim-a value) but strides over dim b; option B costs
    // Tb sub-DMAs and strides over dim a.  Prefer the better strided locality, but only pay
    // many sub-DMAs when dim a is narrow.
    bool overA = hasB && (Cb <= Ca) && (P <= 32);

    int64_t tbc[14] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 16, 24, 32};
    int64_t kcand[8] = {32, 64, 96, 128, 192, 256, 384, 512};
    int64_t candTA[24];
    int64_t candTB[24];
    int64_t nCand = 0;
    {
        int64_t prevA = 0;
        for (int64_t i = 0; i < 8 && nCand < 22; ++i) {
            int64_t Ta = kcand[i];
            if (Ta > P) Ta = P;
            if (Ta == prevA) continue;
            prevA = Ta;
            candTA[nCand] = Ta;
            candTB[nCand] = 1;
            ++nCand;
            if (Ta == P) break;
        }
        if (hasB) {
            int64_t prevB = 1;
            for (int64_t i = 0; i < 14 && nCand < 22; ++i) {
                int64_t Tb = tbc[i];
                if (Tb > bExt) Tb = bExt;
                if (Tb == prevB) continue;
                prevB = Tb;
                candTA[nCand] = P;
                candTB[nCand] = Tb;
                ++nCand;
                if (Tb == bExt) break;
            }
        }
    }

    int64_t bestScore = -1;
    int64_t bestTa = 1;
    int64_t bestTb = 1;
    int64_t bestTC = 1;
    int64_t bestPk = 1;
    int64_t bestPj = 1;
    int64_t bestPacked = 0;
    int64_t bestInPitch = 32;
    int64_t bestNch = 1;
    for (int64_t ci = 0; ci < nCand; ++ci) {
        int64_t Ta = candTA[ci];
        int64_t Tb = candTB[ci];
        if (Ta < 1 || Tb < 1) continue;
        int64_t K = Ta * Tb;
        if (K < 1 || K > U) continue;
        int64_t nchA = (P + Ta - 1) / Ta;
        if (nchA < 1) nchA = 1;
        int64_t nchB = hasB ? ((bExt + Tb - 1) / Tb) : 1;
        if (nchB < 1) nchB = 1;
        int64_t nch = nchA * nchB;
        int64_t nDma = ((overA && hasB && Tb > 1) ? Ta : Tb);
        // sliding-window overlap: the chunks repeat (nch*nchA*nchB*K - Gsz) units of work
        int64_t covNum = nch * K;
        int64_t covDen = P * (hasB ? bExt : 1);
        if (covDen < 1) covDen = 1;
        for (int64_t pv = 0; pv < 2; ++pv) {
            bool packed = (pv == 1);
            if (packed && K != U) continue;
            int64_t Pj;
            if (packed) {
                Pj = K * upe;
            } else {
                Pj = (esz == 1) ? TpAlignUpH(K, 32) : (TpAlignUpH(K * esz, 32) / gusz);
            }
            if (Pj < 1) Pj = 1;
            int64_t TCmax = TP_TBL_MAX / Pj;
            if (TCmax > Dcol) TCmax = Dcol;
            if (packed) {
                if (TCmax < Dcol) continue;
                TCmax = Dcol;
            }
            int64_t TC = TCmax;
            for (; TC >= 1; --TC) {
                int64_t inPitch = TpAlignUpH(TC * esz, 32);
                if (TpNeedUb(K, TC, Pj, gusz, esz, inPitch) <= ub) break;
            }
            if (TC < 1) continue;
            int64_t Pk = (esz == 1) ? TpAlignUpH(TC, 32) : (TpAlignUpH(TC * esz, 32) / gusz);
            int64_t mn = (K < TC) ? K : TC;
            int64_t score = mn * 4096 + K * TC - nDma * 2048 + (packed ? 100000 : 0);
            score = score * covDen / (covNum > 0 ? covNum : 1);
            if (score > bestScore) {
                bestScore = score;
                bestTa = Ta;
                bestTb = Tb;
                bestTC = TC;
                bestPk = Pk;
                bestPj = Pj;
                bestPacked = packed ? 1 : 0;
                bestInPitch = TpAlignUpH(TC * esz, 32);
                bestNch = nch;
            }
        }
    }
    if (bestScore < 0) {
        bestTa = 1;
        bestTb = 1;
        bestTC = 1;
        bestPj = (esz == 1) ? 32 : (32 / gusz);
        bestPk = (esz == 1) ? 32 : (32 / gusz);
        bestPacked = 0;
        bestInPitch = 32;
        bestNch = 1;
    }
    out->Ta = bestTa;
    out->Tb = bestTb;
    out->TC = bestTC;
    out->K = bestTa * bestTb;
    out->Pk = bestPk;
    out->Pj = bestPj;
    out->packed = bestPacked;
    out->overA = overA ? 1 : 0;
    out->nch = bestNch;

    // ---- batch selection (packed tiles only: consecutive output planes are adjacent) ----
    int64_t NB = 1;
    if (bestPacked != 0 && esz != 1) {
        int64_t cands[6] = {8, 6, 4, 3, 2, 1};
        for (int64_t i = 0; i < 6; ++i) {
            int64_t nb = cands[i];
            if (nb > PageSize) continue;
            int64_t depthIn = (nb * out->K * bestInPitch <= 8192) ? 4 : 2;
            int64_t need = depthIn * nb * out->K * bestInPitch + 2 * nb * bestTC * bestPj * gusz +
                           8 * bestTC * bestPj + 8192;
            if (need <= ub) {
                NB = nb;
                break;
            }
        }
    }
    out->NB = NB;

    int64_t Gsz = P * (hasB ? bExt : 1);
    int64_t OG = U / Gsz;
    if (OG < 1) OG = 1;
    out->nlc = OG * bestNch;
    int64_t ncc = (Dcol + out->TC - 1) / out->TC;
    if (ncc < 1) ncc = 1;
    out->ncc = ncc;
    int64_t nPages = H / PageSize;
    if (nPages < 1) nPages = 1;
    int64_t nbPerPage = (PageSize + NB - 1) / NB;
    if (nbPerPage < 1) nbPerPage = 1;
    int64_t nHB = nPages * nbPerPage;
    out->nTiles = nHB * out->nlc * ncc;
    if (out->nTiles < 1) out->nTiles = 1;
    out->nBlocks = coreNum;
    if (out->nBlocks > out->nTiles) out->nBlocks = out->nTiles;
}

#endif // TRANSPOSE_HOST_H
