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
 * \file transpose_kernel.cpp
 * \brief Transpose (arbitrary dimension permutation) kernel.
 *
 *   y[i_0, ..., i_{n-1}] = x[i_{p[0]}, ..., x[p[n-1]]]      (output dim j <- input dim p[j])
 *
 * Normal form
 * -----------
 * With s[k] the input extents, osz[j] = s[p[j]] the output extents and Dcol = s[n-1] the
 * input's innermost (contiguous) extent, let q be the output dim that receives the input's
 * innermost index (p[q] == n-1).  The output is then exactly a 3-D array [H][Dcol][U] with
 *     H = prod_{j<q} osz[j],   U = prod_{j>q} osz[j]
 * and
 *     out3D[h][c][l] = in2D[row(h, l)][c],
 *     in2D = the input viewed as [numel/Dcol][Dcol],
 *     row(h, l) = sum_{j<q} C_j * o_j(h) + sum_{j>q} C_j * o_j(l),
 *     C_j = prod_{p[j]+1 <= k <= n-2} s[k]      (input row stride of output dim j).
 *
 * mode 0 (U == 1): the input's innermost run stays the output's innermost run, so the op is
 *   a pure row permutation.  Every tile is one strided multi-block DMA in and one strided
 *   multi-block DMA out; no UB reorder is needed.
 *
 * mode 1 (U > 1): a tile is TC "columns" (values of the input's innermost index c) crossed
 *   with K "run" units (values of l).  The input is read row by row, which leaves UB as
 *   [K][TC]; the output needs [TC][K] because the U consecutive l values of one column are
 *   contiguous in GM.  The reorder is a UB Gather whose offsets are the byte offsets of the
 *   [K][TC] tile, so the tile's l range must be a contiguous span [l0, l0 + K).
 *
 *   Two run dims are used.  Let a = n-1 (expanded stride 1, input row stride Ca) and let b be
 *   the largest run dim strictly between q and a with extent > 1 (row stride Cb, expanded
 *   stride P = osz[a]).  A tile fixes all other run digits and covers
 *     dim a: o_a in [a0, a0 + Ta),  dim b: o_b in [ob0, ob0 + Tb),
 *   whose l range is contiguous exactly when Ta == P (then l = a + b*P).  Ta == P and Tb > 1
 *   is what keeps K = P*Tb large when the innermost run dim is narrow; that is what stops a
 *   narrow innermost run dim from forcing a tiny write blockLen.
 *
 *   Read layouts (both keep the UB row of an element equal to its l offset inside the tile):
 *     overA (Ta sub-DMAs of Tb blocks, chosen when Ta <= Tb): every sub-DMA walks dim b at a
 *           Cb row stride and lands at UB rows [d*Tb, (d+1)*Tb), so the UB row of the element
 *           with l offset m is (m % Ta)*Tb + m/Ta.
 *     overB (Tb sub-DMAs of Ta blocks): every sub-DMA walks dim a at a Ca row stride and
 *           lands at UB rows [tb*Ta, (tb+1)*Ta), so the UB row is the l offset itself.
 *
 * UB layout rules used here (dav-2201):
 *   - DataCopyPad lays multi-block destinations out at AlignUp(blockLen, 32) byte steps when
 *     the UB-side stride field is 0, so every multi-block transfer keeps the UB stride at 0
 *     and relies on the aligned pitch; every multi-block UB base must be 32B aligned.
 *   - An explicit S->V HardEvent pair must NOT be used as a scalar-store fence here: with no
 *     scalar work between SetFlag and WaitFlag it deadlocks the vector core.
 *   - 8-bit elements are not gathered in the byte domain: the tile is widened to half through
 *     an UNSIGNED byte view, gathered in the 16-bit domain and narrowed back into an
 *     unsigned-byte view of the int8 output buffer (the vector Cast set on this target has
 *     no int8 destination).
 *   - 64-bit elements are gathered through a 32-bit view of both operands.
 *   - MTE2 -> MTE3 reuse of the same UB buffer needs explicit HardEvent fences here, so the
 *     row-permutation path uses them instead of relying on queue events alone.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "transpose_launch.h"
#include "transpose_host.h"

namespace {

using namespace AscendC;

constexpr int64_t TP_MAXN = 8;
constexpr int64_t TP_TBL_MAX = 4096;          // gather offset table entries per tile
constexpr int64_t TP_UB_BUDGET = 140 * 1024;  // UB bytes the tiles may use (platform UB is 192K)
constexpr int64_t TP_KMAX = 128;              // max run units per tile
constexpr int64_t TP_DEPTH = 2;               // queue depth

__aicore__ inline int64_t TpAlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

__aicore__ inline int64_t TpDim(int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t i)
{
    int64_t w = (i < 2) ? d0 : ((i < 4) ? d1 : ((i < 6) ? d2 : d3));
    return (w >> (32 * (i % 2))) & 0xFFFFFFFFLL;
}

struct TpShape {
    int64_t n, Dcol, q, H, U, P, Cl;
    int64_t osz[TP_MAXN];
    int64_t C[TP_MAXN];
    int64_t st[TP_MAXN];
    int64_t PageSize, RowStep;
};

__aicore__ inline void TpDerive(int64_t n, int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                                int64_t pp, TpShape& S)
{
    int64_t s[TP_MAXN];
    int64_t p[TP_MAXN];
    for (int64_t i = 0; i < TP_MAXN; ++i) {
        s[i] = TpDim(d0, d1, d2, d3, i);
        p[i] = (pp >> (4 * i)) & 0xF;
        S.osz[i] = 1;
        S.C[i] = 0;
        S.st[i] = 1;
    }
    S.n = n;
    S.Dcol = s[n - 1];
    int64_t q = 0;
    for (int64_t i = 0; i < n; ++i) {
        S.osz[i] = s[p[i]];
        if (p[i] == n - 1) q = i;
    }
    S.q = q;
    int64_t hExt = 1;
    int64_t uExt = 1;
    for (int64_t j = 0; j < q; ++j) hExt *= S.osz[j];
    for (int64_t j = q + 1; j < n; ++j) uExt *= S.osz[j];
    S.H = hExt;
    S.U = uExt;
    for (int64_t j = 0; j < n; ++j) {
        if (j == q) continue;
        int64_t c = 1;
        for (int64_t k = p[j] + 1; k <= n - 2; ++k) c *= s[k];
        S.C[j] = c;
    }
    for (int64_t j = q - 1; j >= 0; --j) {
        S.st[j] = (j == q - 1) ? 1 : S.st[j + 1] * S.osz[j + 1];
    }
    for (int64_t j = n - 1; j > q; --j) {
        S.st[j] = (j == n - 1) ? 1 : S.st[j + 1] * S.osz[j + 1];
    }
    S.P = S.osz[n - 1];
    S.Cl = (q == n - 1) ? 0 : S.C[n - 1];
    S.PageSize = 1;
    S.RowStep = 0;
    for (int64_t j = q - 1; j >= 0; --j) {
        if (S.osz[j] > 1) {
            S.PageSize = S.osz[j];
            S.RowStep = S.C[j];
            break;
        }
    }
}

// the second run dim: largest j with q < j < n-1 and osz[j] > 1
__aicore__ inline int64_t TpSecondDim(const TpShape& S)
{
    for (int64_t j = S.n - 2; j > S.q; --j) {
        if (S.osz[j] > 1) return j;
    }
    return -1;
}

__aicore__ inline int64_t TpRowH(const TpShape& S, int64_t h)
{
    int64_t r = 0;
    for (int64_t j = 0; j < S.q; ++j) {
        int64_t o = (h / S.st[j]) % S.osz[j];
        r += S.C[j] * o;
    }
    return r;
}

__aicore__ inline int64_t TpRowL(const TpShape& S, int64_t l)
{
    int64_t r = 0;
    for (int64_t j = S.q + 1; j < S.n; ++j) {
        int64_t o = (l / S.st[j]) % S.osz[j];
        r += S.C[j] * o;
    }
    return r;
}

// ---------------------------------------------------------------------------------------
// Row-permutation kernel (U == 1): strided DMA in, strided DMA out.
// ---------------------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void tp_perm_kernel(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1,
                                          int64_t d2, int64_t d3, int64_t pp, int64_t TC,
                                          int64_t RA, int64_t ncc, int64_t nTiles, int64_t nBlocks)
{
    constexpr int64_t esz = (int64_t)sizeof(T);
    TpShape S;
    TpDerive(n, d0, d1, d2, d3, pp, S);

    AscendC::GlobalTensor<T> xG;
    AscendC::GlobalTensor<T> yG;
    xG.SetGlobalBuffer((__gm__ T *)x);
    yG.SetGlobalBuffer((__gm__ T *)y);

    AscendC::TPipe pipe;
    int64_t rowBytes = TpAlignUp(TC * esz, 32);
    // one single-buffer queue per parity: the tile -> UB buffer mapping is fixed and the
    // cross-pipe dependencies are added explicitly below.
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qa;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qb;
    pipe.InitBuffer(qa, 1, (uint32_t)(RA * rowBytes));
    pipe.InitBuffer(qb, 1, (uint32_t)(RA * rowBytes));

    AscendC::TEventID e23 = pipe.AllocEventID<AscendC::HardEvent::MTE2_MTE3>();
    AscendC::TEventID e32a = pipe.AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
    AscendC::TEventID e32b = pipe.AllocEventID<AscendC::HardEvent::MTE3_MTE2>();

    int64_t per = (nTiles + nBlocks - 1) / nBlocks;
    int64_t t0 = per * (int64_t)AscendC::GetBlockIdx();
    int64_t t1 = t0 + per;
    if (t1 > nTiles) t1 = nTiles;
    int64_t nWB = (S.PageSize + RA - 1) / RA;
    int64_t hLast = -1;
    int64_t rh = 0;
    bool used[2] = {false, false};

    for (int64_t t = t0; t < t1; ++t) {
        int64_t rem = t;
        int64_t pg = rem / (nWB * ncc);
        rem -= pg * (nWB * ncc);
        int64_t jb = rem / ncc;
        int64_t cc = rem - jb * ncc;
        int64_t h0 = pg * S.PageSize + jb * RA;
        int64_t rblk = S.PageSize - jb * RA;
        if (rblk > RA) rblk = RA;
        if (h0 + rblk > S.H) rblk = S.H - h0;
        if (rblk <= 0) continue;
        int64_t c0 = cc * TC;
        if (c0 > S.Dcol - TC) c0 = S.Dcol - TC;
        if (h0 != hLast) {
            hLast = h0;
            rh = TpRowH(S, h0);
        }
        int64_t par = (int64_t)((t - t0) & 1);
        // WAR: this parity's UB buffer must have been drained by the previous MTE3
        if (used[par]) {
            if (par == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(e32a);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(e32b);
            }
        }
        auto inT = (par == 0) ? qa.AllocTensor<T>() : qb.AllocTensor<T>();
        AscendC::DataCopyExtParams cpin{(uint16_t)rblk, (uint32_t)(TC * esz),
                                        (uint32_t)((S.RowStep * S.Dcol - TC) * esz), 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(inT, xG[rh * S.Dcol + c0], cpin, pad);
        if (par == 0) {
            qa.EnQue(inT);
            inT = qa.DeQue<T>();
        } else {
            qb.EnQue(inT);
            inT = qb.DeQue<T>();
        }
        // RAW: the UB->GM read must wait for the GM->UB write to land
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(e23);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(e23);
        AscendC::DataCopyExtParams cpout{(uint16_t)rblk, (uint32_t)(TC * esz), 0,
                                         (uint32_t)((S.Dcol - TC) * esz), 0};
        AscendC::DataCopyPad(yG[h0 * S.Dcol + c0], inT, cpout);
        if (par == 0) {
            qa.FreeTensor(inT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(e32a);
        } else {
            qb.FreeTensor(inT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(e32b);
        }
        used[par] = true;
    }
    if (used[0]) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(e32a);
    }
    if (used[1]) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(e32b);
    }
    pipe.ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(e23);
    pipe.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(e32a);
    pipe.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(e32b);
}

// ---------------------------------------------------------------------------------------
// Gather kernel (U > 1)
// ---------------------------------------------------------------------------------------
__aicore__ inline int64_t TpNeedUb(int64_t K, int64_t TC, int64_t Pj, int64_t gusz, int64_t esz,
                                   int64_t inPitch)
{
    int64_t need = TP_DEPTH * K * inPitch;              // read queue
    if (esz == 1) {
        need += 2 * K * inPitch;                        // widened tile (half per byte)
        need += 2 * TC * TpAlignUp(K, 32);              // gather dst staging (half domain)
        need += TP_DEPTH * TC * TpAlignUp(K, 32);       // byte output queue
    } else {
        need += TP_DEPTH * TC * Pj * gusz;              // gather dst queue
    }
    need += 8 * TC * Pj;                                // float + int32 offset tables
    need += 8192;                                       // framework slack
    return need;
}

template <typename T, typename GU, int KIND>
__global__ __aicore__ void tp_gather_kernel(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1,
                                            int64_t d2, int64_t d3, int64_t pp, int64_t numBlocks)
{
    constexpr int64_t esz = (int64_t)sizeof(T);
    constexpr int64_t gusz = (int64_t)sizeof(GU);
    // gather units per element (KIND 2 widens one byte to one half)
    constexpr int64_t upe = (KIND == 3) ? 2 : 1;

    TpShape S;
    TpDerive(n, d0, d1, d2, d3, pp, S);
    if (S.U <= 1) {
        return;  // the host routes U == 1 to the row-permutation kernel
    }

    int64_t bIdx = TpSecondDim(S);
    int64_t Ca = S.C[S.n - 1];
    int64_t Cb = (bIdx >= 0) ? S.C[bIdx] : 0;
    int64_t bExt = (bIdx >= 0) ? S.osz[bIdx] : 1;
    int64_t Gsz = S.P * ((bIdx >= 0) ? bExt : 1);
    int64_t OG = (Gsz > 0) ? (S.U / Gsz) : 1;
    if (OG < 1) OG = 1;

    // ---- tile geometry (derived here: one source of truth) ----
    int64_t Ta = (S.P < TP_KMAX) ? S.P : TP_KMAX;
    if (Ta < 1) Ta = 1;
    int64_t Tb = 1;
    if (bIdx >= 0 && Ta == S.P) {
        // Ta == P keeps the tile's l span contiguous: l = a + b*P
        int64_t tbMax = TP_KMAX / Ta;
        if (tbMax < 1) tbMax = 1;
        Tb = (bExt < tbMax) ? bExt : tbMax;
        if (Tb < 1) Tb = 1;
    }
    int64_t K = Ta * Tb;
    if (K > S.U) {
        K = S.U;
        Tb = (Ta > 0) ? (K / Ta) : 1;
        if (Tb < 1) Tb = 1;
        K = Ta * Tb;
    }
    // overA costs Ta sub-DMAs but walks dim b, overB costs Tb sub-DMAs and walks dim a
    bool overARead = (bIdx >= 0) && (Tb > 1) && (Ta <= Tb);
    int64_t Pj = (esz == 1) ? TpAlignUp(K, 32) : (TpAlignUp(K * esz, 32) / gusz);
    if (Pj < 1) Pj = 1;
    int64_t hi = TP_TBL_MAX / Pj;
    if (hi > S.Dcol) hi = S.Dcol;
    if (hi < 1) hi = 1;
    int64_t lo = 1;
    while (lo < hi) {
        int64_t mid = (lo + hi + 1) / 2;
        if (TpNeedUb(K, mid, Pj, gusz, esz, TpAlignUp(mid * esz, 32)) <= TP_UB_BUDGET) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    int64_t TC = lo;
    if (TC > S.Dcol) TC = S.Dcol;
    if (TC < 1) TC = 1;
    Pj = (esz == 1) ? TpAlignUp(K, 32) : (TpAlignUp(K * esz, 32) / gusz);
    if (Pj < 1) Pj = 1;
    int64_t Pk = (esz == 1) ? TpAlignUp(TC, 32) : (TpAlignUp(TC * esz, 32) / gusz);
    if (Pk < 1) Pk = 1;

    int64_t nchA = (S.P + Ta - 1) / Ta;
    if (nchA < 1) nchA = 1;
    int64_t nchB = (bIdx >= 0) ? ((bExt + Tb - 1) / Tb) : 1;
    if (nchB < 1) nchB = 1;
    int64_t nlc = OG * nchA * nchB;
    int64_t ncc = (S.Dcol + TC - 1) / TC;
    if (ncc < 1) ncc = 1;
    int64_t perH = nlc * ncc;
    int64_t nTiles = S.H * perH;
    if (nTiles < 1) nTiles = 1;
    int64_t nBlocks = numBlocks;
    if (nBlocks > nTiles) nBlocks = nTiles;
    if (nBlocks < 1) nBlocks = 1;

    AscendC::GlobalTensor<T> xG;
    AscendC::GlobalTensor<T> yG;
    xG.SetGlobalBuffer((__gm__ T *)x);
    yG.SetGlobalBuffer((__gm__ T *)y);

    AscendC::TPipe pipe;
    int64_t inPitchBytes = TpAlignUp(TC * esz, 32);
    int64_t inPitchEl = inPitchBytes / esz;
    int64_t outPitchBytes = TpAlignUp(K * esz, 32);

    AscendC::TQue<AscendC::QuePosition::VECIN, TP_DEPTH> inQ;
    pipe.InitBuffer(inQ, TP_DEPTH, (uint32_t)(K * inPitchBytes));

    AscendC::TQue<AscendC::QuePosition::VECOUT, TP_DEPTH> outQ;
    if (KIND == 2) {
        pipe.InitBuffer(outQ, TP_DEPTH, (uint32_t)(TC * outPitchBytes));
    } else {
        pipe.InitBuffer(outQ, TP_DEPTH, (uint32_t)(TC * Pj * gusz));
    }

    // ---- gather offset table (built once: the geometry is fixed for the whole launch) ----
    AscendC::TBuf<AscendC::TPosition::VECCALC> tblFBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tblBuf;
    pipe.InitBuffer(tblFBuf, (uint32_t)(TC * Pj * (int64_t)sizeof(float)));
    pipe.InitBuffer(tblBuf, (uint32_t)(TC * Pj * (int64_t)sizeof(int32_t)));
    auto tblF = tblFBuf.Get<float>();
    auto tblI = tblBuf.Get<int32_t>();
    auto tblU = tblI.ReinterpretCast<uint32_t>();
    {
        int64_t rw = K * upe;  // valid entries in one table row
        for (int64_t m = 0; m < Pj; ++m) {
            int64_t kk = 0;
            int64_t jj = 0;
            if (m < rw) {
                jj = m % upe;
                int64_t lo2 = m / upe;  // l offset inside the tile
                if (overARead && Tb > 1) {
                    kk = (lo2 % Ta) * Tb + (lo2 / Ta);
                } else {
                    kk = lo2;
                }
            }
            tblF.SetValue(m, (float)((kk * Pk + jj) * gusz));
        }
        int64_t lvl = 1;
        while (2 * lvl <= TC) {
            AscendC::Adds(tblF[lvl * Pj], tblF[0], (float)(lvl * upe * gusz), (int32_t)(lvl * Pj));
            lvl *= 2;
        }
        for (int64_t c = lvl; c < TC; ++c) {
            AscendC::Adds(tblF[c * Pj], tblF[0], (float)(c * upe * gusz), (int32_t)Pj);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(tblI, tblF, AscendC::RoundMode::CAST_RINT, (int32_t)(TC * Pj));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---- auxiliary buffers for the 8-bit widening path ----
    AscendC::TBuf<AscendC::TPosition::VECCALC> wideBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dstBuf;
    if (KIND == 2) {
        pipe.InitBuffer(wideBuf, (uint32_t)(K * inPitchBytes * 2));
        pipe.InitBuffer(dstBuf, (uint32_t)(TC * Pj * 2));
    }

    int64_t per = (nTiles + nBlocks - 1) / nBlocks;
    int64_t t0 = per * (int64_t)AscendC::GetBlockIdx();
    int64_t t1 = t0 + per;
    if (t1 > nTiles) t1 = nTiles;

    for (int64_t t = t0; t < t1; ++t) {
        int64_t rem = t;
        int64_t h0 = rem / perH;
        rem -= h0 * perH;
        int64_t lc = rem / ncc;
        int64_t cc = rem - lc * ncc;
        int64_t ja = lc % nchA;
        int64_t rest = lc / nchA;
        int64_t jb = rest % nchB;
        int64_t grp = rest / nchB;
        int64_t a0 = ja * Ta;
        if (a0 > S.P - Ta) a0 = S.P - Ta;
        if (a0 < 0) a0 = 0;
        int64_t ob0 = jb * Tb;
        if (ob0 > bExt - Tb) ob0 = bExt - Tb;
        if (ob0 < 0) ob0 = 0;
        int64_t l0 = grp * Gsz + ob0 * S.P + a0;
        int64_t c0 = cc * TC;
        if (c0 > S.Dcol - TC) c0 = S.Dcol - TC;
        if (c0 < 0) c0 = 0;
        int64_t rh = TpRowH(S, h0);
        int64_t rl = TpRowL(S, l0);
        int64_t rowBase = rh + rl;

        auto inT = inQ.AllocTensor<T>();
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if (overARead) {
            for (int64_t d = 0; d < Ta; ++d) {
                AscendC::DataCopyExtParams cpin{(uint16_t)Tb, (uint32_t)(TC * esz),
                                                (uint32_t)((Cb * S.Dcol - TC) * esz), 0, 0};
                AscendC::DataCopyPad(inT[(d * Tb) * inPitchEl],
                                     xG[(rowBase + d * Ca) * S.Dcol + c0], cpin, pad);
            }
        } else {
            for (int64_t tb = 0; tb < Tb; ++tb) {
                AscendC::DataCopyExtParams cpin{(uint16_t)Ta, (uint32_t)(TC * esz),
                                                (uint32_t)((Ca * S.Dcol - TC) * esz), 0, 0};
                AscendC::DataCopyPad(inT[(tb * Ta) * inPitchEl],
                                     xG[(rowBase + tb * Cb) * S.Dcol + c0], cpin, pad);
            }
        }
        inQ.EnQue(inT);
        inT = inQ.DeQue<T>();

        if (KIND == 2) {
            auto wide = wideBuf.Get<half>();
            auto gdst = dstBuf.Get<half>();
            AscendC::Cast(wide, inT.template ReinterpretCast<uint8_t>(),
                          AscendC::RoundMode::CAST_NONE, (int32_t)(K * Pk));
            inQ.FreeTensor(inT);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(gdst, wide, tblU, (uint32_t)0, (uint32_t)(TC * Pj));
            AscendC::PipeBarrier<PIPE_V>();
            auto outT = outQ.AllocTensor<T>();
            AscendC::Cast(outT.template ReinterpretCast<uint8_t>(), gdst,
                          AscendC::RoundMode::CAST_RINT, (int32_t)(TC * Pj));
            outQ.EnQue(outT);
            outT = outQ.DeQue<T>();
            AscendC::DataCopyExtParams cpout{(uint16_t)TC, (uint32_t)(K * esz), 0,
                                             (uint32_t)((S.U - K) * esz), 0};
            AscendC::DataCopyPad(yG[(h0 * S.Dcol + c0) * S.U + l0], outT, cpout);
            outQ.FreeTensor(outT);
        } else {
            auto outT = outQ.AllocTensor<GU>();
            AscendC::Gather(outT, inT.template ReinterpretCast<GU>(), tblU, (uint32_t)0,
                            (uint32_t)(TC * Pj));
            outQ.EnQue(outT);
            inQ.FreeTensor(inT);
            outT = outQ.DeQue<GU>();
            auto outN = outT.template ReinterpretCast<T>();
            AscendC::DataCopyExtParams cpout{(uint16_t)TC, (uint32_t)(K * esz), 0,
                                             (uint32_t)((S.U - K) * esz), 0};
            AscendC::DataCopyPad(yG[(h0 * S.Dcol + c0) * S.U + l0], outN, cpout);
            outQ.FreeTensor(outT);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------
// Launch wrappers (regular C functions callable from g++)
// ---------------------------------------------------------------------------------------
#define TP_LAUNCH_BODY(TYPE, GUT, KINDC)                                                       \
    if (mode == 1) {                                                                           \
        tp_gather_kernel<TYPE, GUT, KINDC><<<numBlocks, nullptr, stream>>>(x, y, n, d0, d1, d2, \
                                                                          d3, pp, numBlocks);  \
    } else {                                                                                   \
        tp_perm_kernel<TYPE><<<numBlocks, nullptr, stream>>>(x, y, n, d0, d1, d2, d3, pp, TC,  \
                                                             RA, ncc, nTiles, numBlocks);      \
    }

extern "C" {

void launch_transpose_f32(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1, int64_t d2,
                          int64_t d3, int64_t pp, int64_t mode, int64_t TC, int64_t K, int64_t Pk,
                          int64_t Pj, int64_t nch, int64_t ncc, int64_t nlc, int64_t nTiles,
                          int64_t numBlocks, int64_t RA, int64_t numel, int64_t Ta, int64_t Tb,
                          int64_t packed, int64_t overA, int64_t NB, void* stream)
{
    TP_LAUNCH_BODY(float, float, 0)
}

void launch_transpose_f16(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1, int64_t d2,
                          int64_t d3, int64_t pp, int64_t mode, int64_t TC, int64_t K, int64_t Pk,
                          int64_t Pj, int64_t nch, int64_t ncc, int64_t nlc, int64_t nTiles,
                          int64_t numBlocks, int64_t RA, int64_t numel, int64_t Ta, int64_t Tb,
                          int64_t packed, int64_t overA, int64_t NB, void* stream)
{
    TP_LAUNCH_BODY(half, half, 1)
}

void launch_transpose_bf16(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1, int64_t d2,
                           int64_t d3, int64_t pp, int64_t mode, int64_t TC, int64_t K, int64_t Pk,
                           int64_t Pj, int64_t nch, int64_t ncc, int64_t nlc, int64_t nTiles,
                           int64_t numBlocks, int64_t RA, int64_t numel, int64_t Ta, int64_t Tb,
                           int64_t packed, int64_t overA, int64_t NB, void* stream)
{
    TP_LAUNCH_BODY(bfloat16_t, bfloat16_t, 1)
}

void launch_transpose_i8(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1, int64_t d2,
                         int64_t d3, int64_t pp, int64_t mode, int64_t TC, int64_t K, int64_t Pk,
                         int64_t Pj, int64_t nch, int64_t ncc, int64_t nlc, int64_t nTiles,
                         int64_t numBlocks, int64_t RA, int64_t numel, int64_t Ta, int64_t Tb,
                         int64_t packed, int64_t overA, int64_t NB, void* stream)
{
    TP_LAUNCH_BODY(int8_t, half, 2)
}

void launch_transpose_i16(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1, int64_t d2,
                          int64_t d3, int64_t pp, int64_t mode, int64_t TC, int64_t K, int64_t Pk,
                          int64_t Pj, int64_t nch, int64_t ncc, int64_t nlc, int64_t nTiles,
                          int64_t numBlocks, int64_t RA, int64_t numel, int64_t Ta, int64_t Tb,
                          int64_t packed, int64_t overA, int64_t NB, void* stream)
{
    TP_LAUNCH_BODY(int16_t, int16_t, 1)
}

void launch_transpose_i32(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1, int64_t d2,
                          int64_t d3, int64_t pp, int64_t mode, int64_t TC, int64_t K, int64_t Pk,
                          int64_t Pj, int64_t nch, int64_t ncc, int64_t nlc, int64_t nTiles,
                          int64_t numBlocks, int64_t RA, int64_t numel, int64_t Ta, int64_t Tb,
                          int64_t packed, int64_t overA, int64_t NB, void* stream)
{
    TP_LAUNCH_BODY(int32_t, int32_t, 0)
}

void launch_transpose_i64(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1, int64_t d2,
                          int64_t d3, int64_t pp, int64_t mode, int64_t TC, int64_t K, int64_t Pk,
                          int64_t Pj, int64_t nch, int64_t ncc, int64_t nlc, int64_t nTiles,
                          int64_t numBlocks, int64_t RA, int64_t numel, int64_t Ta, int64_t Tb,
                          int64_t packed, int64_t overA, int64_t NB, void* stream)
{
    TP_LAUNCH_BODY(int64_t, int32_t, 3)
}

}  // extern "C"
