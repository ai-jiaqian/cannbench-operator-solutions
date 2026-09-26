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
 * \file softmax_kernel.cpp
 * \brief Softmax kernels + host tiling (compiled with bisheng, --npu-arch=dav-2201)
 *
 * Three device paths, all math in the fp32 domain:
 *  mode 0 K1-std    : inner == 1 and R*esize >= 32B. Rows of length R are contiguous
 *                     (m = outer*R rows). Per tile of G rows: 2D DataCopyPad load with
 *                     rightPadding=-inf up to Kpad=AlignUp(R, 32/esize), SoftMax<float>
 *                     high-order API ({srcM=G, srcK=Kpad, oriSrcK=R}), cast back, store.
 *  mode 1 K1-packed : R*esize < 32B (e.g. [1000003, 2] fp16). Rows are packed contiguously
 *                     in GM; a 1D load fills a [32B -inf header | packed rows] UB buffer,
 *                     a byte-offset Gather de-interleaves into a [G x Kpad] fp16 view
 *                     (pad lanes read the -inf header), SoftMax, cast back, inverse
 *                     Gather re-packs, 1D store.
 *  mode 2 K2        : dim < rank-1. Units = (slice o, column block b). Two strategies
 *                     selected by k2Flags (host tp0, from the lower solve
 *                     3cb1340a6c5168b408df4312f0d21be163278f6de3f5e4aa6c2678f2eae5ded2):
 *                     bit1 two_pass_online (selected): ONE fused pass over row-chunks of
 *                     rT rows computes the running lane max M and the rescaled running sum
 *                     S = S*exp(min(M-cm,0)) + sum(exp(x-Mnew)); then pass C divides.
 *                     M init -FLT_MAX so all-(-inf) columns give 0/0 = NaN and +inf columns
 *                     give inf-inf = NaN -> whole column NaN, matching torch (argued in
 *                     models/implementation_space_v1.yaml, premise P-ONLINE-SEM).
 *                     bit0 twod_pad (selected): ragged column blocks (Wb not a multiple of
 *                     32/esize) use one 2D DataCopyPad per row-chunk with isPad=true,
 *                     rightPadding=(Wl-Wb) -inf instead of rT per-row 1D loads, whenever
 *                     (inner-Wb)*esize is 32B aligned (always true for nB==1). Premise
 *                     P-2DPAD-RAGGED; the per-row path is retained and used when the
 *                     alignment test fails. bit2 twod_pad ragged stores (selected by the
 *                     iteration-3 lower solves, models/solve_record_iteration3*.yaml):
 *                     ragged blocks use ONE 2D DataCopyPad store per row-chunk with
 *                     srcStride=0 - the store-side per-row advance is AlignUp(Wb*esize,32),
 *                     which matches the UB tile pitch Wl*esize exactly (mirror of the probed
 *                     load path; premise P-K2-2DSTORE) - instead of rTt per-row 1D stores,
 *                     whenever (inner-Wb)*esize is 32B aligned. The per-row store path is
 *                     retained and used when the alignment test fails or bit2 is clear.
 *                     Full blocks (Wb % per32 == 0) always use the 2D store as in
 *                     evaluation-0003.
 *                     bit3 whole-lane resident (z2 k2_variant_policy == resident_expand_nB):
 *                     when the host finds the derived rT_res >= 1 the R x WlMax fp32 lane
 *                     block is loaded once, staged into one resident fp32 buffer (pitch
 *                     WlMax) and the per-lane max, exp, sum and divide are computed in
 *                     place on that single copy (x read once, Exp once per element). The
 *                     host derives rT_res_chunk (bc-reserving) and rT_res_perrow (no bc);
 *                     if rT_res_chunk >= CHUNK_MIN (4) the chunk-level phase-1/2/3
 *                     realization over the fp32 scratch buffer bc ([rT_res, WlMax]) is
 *                     selected (bit3 only) with the modeled cost-minimizing chunk size
 *                     over [CHUNK_MIN, rT_res_chunk]; else if rT_res_perrow >= 1 the
 *                     per-row realization is selected (bit3 + bit4) and the resident
 *                     padding lanes are not staged. The host passes rT_res as the chunk
 *                     size; otherwise the incumbent streaming schedule selected by bits
 *                     0-2 runs unchanged.
 *                     bit5 chunk-level two-pass streaming (retained, never selected under
 *                     the bound theta resident_expand_nB): when no resident realization fits, the
 *                     non-resident unit uses the chunk-level two-pass streaming schedule
 *                     (pass A chunk-level Max into an fp32 accumulator acc + per-row
 *                     reduction into M; pass B chunk-level Sub/Exp/Add into acc + per-row
 *                     reduction into S; pass C chunk-level Sub/Exp/Div) instead of the
 *                     per-row streaming schedule. It requires Wl == WlMax; otherwise the
 *                     streaming-online per-row realization runs for that block.
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "softmax_launch.h"

using namespace AscendC;

// Deduce the shape parameter type used by the SoftMax host tiling API without naming it
// (the AscendC::TensorShape alias is not visible in every toolchain variant).
template <typename S> S SmShapeDeduceHelper(uint32_t (*)(const S &srcShape, uint32_t, bool));
using SmShapeT = decltype(SmShapeDeduceHelper(&AscendC::GetSoftMaxMinTmpSize));

namespace {

constexpr int64_t SM_MARGIN = 8192;   // UB safety margin
constexpr int64_t SM_GATHER_CHUNK = 8192;

__aicore__ inline int64_t SminS(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline int64_t SmaxS(int64_t a, int64_t b) { return a > b ? a : b; }
__aicore__ inline int64_t AlignUpS(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

inline int64_t Hmin(int64_t a, int64_t b) { return a < b ? a : b; }
inline int64_t Hmax(int64_t a, int64_t b) { return a > b ? a : b; }
inline int64_t HAlignUp(int64_t v, int64_t a) { return (v + a - 1) / a * a; }
inline float HNegInf() { return -__builtin_huge_valf(); }

template <typename T> __aicore__ inline T SmNegInf()
{
    if constexpr (std::is_same<T, float>::value) {
        return -__builtin_huge_valf();
    } else if constexpr (std::is_same<T, half>::value) {
        return half(-__builtin_huge_valf());
    } else {
        return bfloat16_t(-__builtin_huge_valf());
    }
}

// ---- host helpers for tmp sizing (SoftMaxTiling itself is computed kernel-side) ----

// ---------------- mode 0: K1-std ----------------
template <typename T>
__global__ __aicore__ void softmax_k1_std_kernel(GM_ADDR x, GM_ADDR z,
    int64_t m, int64_t R, int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize)
{
    const int64_t blk = (int64_t)GetBlockIdx();
    const int64_t rowStart = blk * rowsPerCore;
    if (rowStart >= m) {
        return;
    }
    const int64_t rowEnd = SminS(rowStart + rowsPerCore, m);
    const int64_t esize = (int64_t)sizeof(T);
    const int64_t pitchB = Kpad * esize;
    const int64_t tileE = G * Kpad;
    constexpr bool isFp32 = std::is_same<T, float>::value;

    TPipe pipe;
    GlobalTensor<T> xGm, zGm;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    zGm.SetGlobalBuffer((__gm__ T*)z);
    const T negInf = SmNegInf<T>();

    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
    pipe.InitBuffer(inQ, 2, (uint64_t)(G * pitchB));
    pipe.InitBuffer(outQ, 2, (uint64_t)(G * pitchB));

    TBuf<QuePosition::VECCALC> x32Buf, sumBuf, maxBuf, tmpBuf;
    if constexpr (!isFp32) {
        pipe.InitBuffer(x32Buf, (uint64_t)(tileE * 4));
    }
    pipe.InitBuffer(sumBuf, (uint64_t)(G * 32));
    pipe.InitBuffer(maxBuf, (uint64_t)(G * 32));
    pipe.InitBuffer(tmpBuf, (uint64_t)tmpSize);

    auto sumT = sumBuf.Get<float>();
    auto maxT = maxBuf.Get<float>();
    auto tmpT = tmpBuf.Get<uint8_t>();
    // Kernel-side tiling (official pattern; the tmp size is decided by the host).
    tiling::SoftMaxTiling st;
    SoftMaxShapeInfo siTile;
    siTile.srcM = (uint32_t)G;
    siTile.srcK = (uint32_t)Kpad;
    siTile.oriSrcM = (uint32_t)G;
    siTile.oriSrcK = (uint32_t)R;
    SoftMaxTilingFunc((uint32_t)tmpSize, siTile, st, (uint32_t)sizeof(T), (uint32_t)sizeof(float), false);
    SoftMaxShapeInfo si;
    si.srcM = (uint32_t)G;
    si.srcK = (uint32_t)Kpad;
    si.oriSrcM = (uint32_t)G;
    si.oriSrcK = (uint32_t)R;

    for (int64_t row0 = rowStart; row0 < rowEnd; row0 += G) {
        const int64_t Gt = SminS(G, rowEnd - row0);
        auto raw = inQ.AllocTensor<T>();
        DataCopyExtParams lp{(uint16_t)Gt, (uint32_t)(R * esize), 0, 0, 0};
        if (Kpad == R) {
            DataCopyPadExtParams<T> pp{false, 0, 0, negInf};
            DataCopyPad(raw, xGm[row0 * R], lp, pp);
        } else {
            DataCopyPadExtParams<T> pp{true, 0, (uint8_t)(Kpad - R), negInf};
            DataCopyPad(raw, xGm[row0 * R], lp, pp);
        }
        inQ.EnQue(raw);
        raw = inQ.DeQue<T>();

        auto outT = outQ.AllocTensor<T>();
        if constexpr (isFp32) {
            SoftMax<float>(outT, sumT, maxT, raw, tmpT, st, si);
        } else {
            auto x32 = x32Buf.Get<float>();
            Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)tileE);
            SoftMax<float>(x32, sumT, maxT, x32, tmpT, st, si);
            Cast(outT, x32, RoundMode::CAST_RINT, (int32_t)tileE);
        }
        outQ.EnQue(outT);
        auto outR = outQ.DeQue<T>();

        // K1-std store (derived rule k1_store): ONE 2D DataCopyPad per tile. For
        // Kpad == R this is the incumbent store; for Kpad != R it replaces the
        // incumbent per-row loop. The UB source outR has pitch Kpad*esize; with
        // srcStride = 0 the MTE source advance per row is AlignUp(R*esize, 32) =
        // Kpad*esize, and blockLen = R*esize bytes are written per row, so the pad
        // lanes are never transferred and the tensor-final row stays in-bounds.
        {
            DataCopyExtParams sp{(uint16_t)Gt, (uint32_t)(R * esize), 0, 0, 0};
            DataCopyPad(zGm[row0 * R], outR, sp);
        }
        outQ.FreeTensor(outR);
        inQ.FreeTensor(raw);
    }
}

// ---------------- mode 1: K1-packed ----------------
template <typename T>
__global__ __aicore__ void softmax_k1_packed_kernel(GM_ADDR x, GM_ADDR z,
    int64_t m, int64_t R, int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize)
{
    const int64_t blk = (int64_t)GetBlockIdx();
    const int64_t rowStart = blk * rowsPerCore;
    if (rowStart >= m) {
        return;
    }
    const int64_t rowEnd = SminS(rowStart + rowsPerCore, m);
    const int64_t esize = (int64_t)sizeof(T);
    const int64_t hdrE = 32 / esize;          // 32B -inf header, in elements
    const int64_t pkB = G * R * esize;        // packed bytes per tile
    const int64_t tileE = G * Kpad;

    TPipe pipe;
    GlobalTensor<T> xGm, zGm;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    zGm.SetGlobalBuffer((__gm__ T*)z);
    const T negInf = SmNegInf<T>();

    TQue<QuePosition::VECIN, 2> inQ;      // [32B header | packed rows]
    TQue<QuePosition::VECOUT, 2> outQ;    // packed result
    pipe.InitBuffer(inQ, 2, (uint64_t)(32 + pkB));
    pipe.InitBuffer(outQ, 2, (uint64_t)AlignUpS(pkB, 32));

    TBuf<QuePosition::VECCALC> gDstBuf, x32Buf, fwdBuf, invBuf, sumBuf, maxBuf, tmpBuf;
    pipe.InitBuffer(gDstBuf, (uint64_t)(tileE * esize));
    // derived.k1_packed_realization (swap-pairwise) places A_f32/B_f32/S at
    // x32[0], x32[2*Gpad], x32[4*Gpad] (2G fp32 elements each), so the fp32 scratch
    // must hold 4*Gpad + 2G floats. tileE*4 is the incumbent size and is >= the layout
    // for the large G used by the packed path; the max keeps the layout in-bounds for
    // small G as well without changing the host budget's G/tmpSize selection.
    pipe.InitBuffer(x32Buf, (uint64_t)(SmaxS(tileE, 4 * AlignUpS(G, 16) + 2 * G) * 4));
    pipe.InitBuffer(fwdBuf, (uint64_t)(tileE * 4));
    pipe.InitBuffer(invBuf, (uint64_t)(G * R * 4));
    pipe.InitBuffer(sumBuf, (uint64_t)(G * 32));
    pipe.InitBuffer(maxBuf, (uint64_t)(G * 32));
    pipe.InitBuffer(tmpBuf, (uint64_t)tmpSize);

    // Byte-offset ramps (built once per core). For R == 2 the derived
    // k1_packed_realization uses the swap-pairwise formulation: one swap ramp swIdx of
    // 2G uint32 whose entry 2i points at the odd element and entry 2i+1 at the even
    // element of packed row i (offsets into the [32B -inf header | packed rows] buffer),
    // so Gather(raw, swIdx) is the adjacent-swapped packed array. For R != 2 the
    // incumbent forward/inverse Gather ramps are built.
    // derived.k1_packed_realization: the sub-buffer offsets are padded to 32-byte
    // alignment. Gpad = AlignUp(G, 16) makes every sub-buffer base 32B aligned for
    // both T (16 elements = 32B) and fp32/uint32 (16 elements = 64B); without it an
    // odd tile row count G leaves oddT/oddF/odIdx at a 2-byte-aligned UB address and
    // the VEC Gather/arithmetic faults ("UB address accessed by the VEC instruction
    // is not aligned", preflight-0004 case 13).
    // AlignUpS is __aicore__; HAlignUp is a host-only helper and cannot be called
    // from this __global__ function (preflight-0005 compile error).
    const int64_t Gpad = AlignUpS(G, 16);
    auto fwd = fwdBuf.Get<uint32_t>();
    auto inv = invBuf.Get<uint32_t>();
    if (R == 2) {
        // derived.k1_packed_realization: the byte-offset swap ramp fwd[2i] =
        // 32 + (2i+1)*esize, fwd[2i+1] = 32 + (2i)*esize is built once per core for
        // every element size (round-5 swap Gather); Gather(raw, fwd) yields the
        // adjacent-swapped packed array.
        for (int64_t i = 0; i < G; ++i) {
            fwd.SetValue(2 * i, (uint32_t)(32 + (2 * i + 1) * esize));
            fwd.SetValue(2 * i + 1, (uint32_t)(32 + (2 * i) * esize));
        }
    } else {
        for (int64_t g = 0; g < G; ++g) {
            for (int64_t c = 0; c < Kpad; ++c) {
                // pad lanes (c >= R) point at the -inf header (byte 0)
                fwd.SetValue(g * Kpad + c,
                             (c < R) ? (uint32_t)(32 + g * R * esize + c * esize) : 0u);
            }
        }
        for (int64_t j = 0; j < G * R; ++j) {
            inv.SetValue(j, (uint32_t)((j / R) * Kpad * esize + (j % R) * esize));
        }
    }

    auto sumT = sumBuf.Get<float>();
    auto maxT = maxBuf.Get<float>();
    auto tmpT = tmpBuf.Get<uint8_t>();
    auto x32 = x32Buf.Get<float>();
    tiling::SoftMaxTiling st;
    SoftMaxShapeInfo siTile;
    siTile.srcM = (uint32_t)G;
    siTile.srcK = (uint32_t)Kpad;
    siTile.oriSrcM = (uint32_t)G;
    siTile.oriSrcK = (uint32_t)R;
    SoftMaxTilingFunc((uint32_t)tmpSize, siTile, st, (uint32_t)sizeof(T), (uint32_t)sizeof(float), false);
    SoftMaxShapeInfo si;
    si.srcM = (uint32_t)G;
    si.srcK = (uint32_t)Kpad;
    si.oriSrcM = (uint32_t)G;
    si.oriSrcK = (uint32_t)R;

    for (int64_t row0 = rowStart; row0 < rowEnd; row0 += G) {
        const int64_t Gt = SminS(G, rowEnd - row0);
        const int64_t pkBt = Gt * R * esize;
        auto raw = inQ.AllocTensor<T>();
        DataCopyExtParams lp{1, (uint32_t)pkBt, 0, 0, 0};
        DataCopyPadExtParams<T> pp{false, 0, 0, negInf};
        DataCopyPad(raw[hdrE], xGm[row0 * R], lp, pp);
        inQ.EnQue(raw);
        raw = inQ.DeQue<T>();
        // refresh the -inf header of this queue buffer (clobbered never, but cheap)
        Duplicate(raw[0], negInf, (int32_t)hdrE);

        auto outT = outQ.AllocTensor<T>();
        if (R == 2) {
            // derived.k1_packed_realization: the pairwise stable softmax is evaluated in
            // fp32 with A = the packed rows in original order (read at the packed-rows
            // offset raw[hdrE]) and B = the adjacent-swapped packed array produced by the
            // round-5 swap Gather over the ramp built above. A_f32 after the divide is
            // already in packed output order [y0_0, y1_0, y0_1, y1_1, ...], so no
            // interleave Gather is needed. The load/header/store realization is unchanged.
            auto B = gDstBuf.Get<T>();
            auto A_f32 = x32;
            auto B_f32 = x32[2 * Gpad];
            auto S_f32 = x32[4 * Gpad];
            Gather<T>(B, raw, fwd, 0, (uint32_t)(2 * Gt));
            Cast(A_f32, raw[hdrE], RoundMode::CAST_NONE, (int32_t)(2 * Gt));
            Cast(B_f32, B, RoundMode::CAST_NONE, (int32_t)(2 * Gt));
            Max(S_f32, A_f32, B_f32, (int32_t)(2 * Gt));   // M = max(A, B)
            Sub(A_f32, A_f32, S_f32, (int32_t)(2 * Gt));
            Sub(B_f32, B_f32, S_f32, (int32_t)(2 * Gt));
            Exp(A_f32, A_f32, (int32_t)(2 * Gt));
            Exp(B_f32, B_f32, (int32_t)(2 * Gt));
            Add(S_f32, A_f32, B_f32, (int32_t)(2 * Gt));   // S = sum
            Div(A_f32, A_f32, S_f32, (int32_t)(2 * Gt));
            Cast(outT, A_f32, RoundMode::CAST_RINT, (int32_t)(2 * Gt));
        } else {
            auto gDst = gDstBuf.Get<T>();
            for (int64_t j0 = 0; j0 < tileE; j0 += SM_GATHER_CHUNK) {
                const int64_t cnt = SminS(SM_GATHER_CHUNK, tileE - j0);
                Gather<T>(gDst[j0], raw, fwd[j0], 0, (uint32_t)cnt);
            }
            Cast(x32, gDst, RoundMode::CAST_NONE, (int32_t)tileE);
            SoftMax<float>(x32, sumT, maxT, x32, tmpT, st, si);
            Cast(gDst, x32, RoundMode::CAST_RINT, (int32_t)tileE);

            for (int64_t j0 = 0; j0 < G * R; j0 += SM_GATHER_CHUNK) {
                const int64_t cnt = SminS(SM_GATHER_CHUNK, G * R - j0);
                Gather<T>(outT[j0], gDst, inv[j0], 0, (uint32_t)cnt);
            }
        }
        outQ.EnQue(outT);
        auto outR = outQ.DeQue<T>();
        DataCopyExtParams sp{1, (uint32_t)pkBt, 0, 0, 0};
        DataCopyPad(zGm[row0 * R], outR, sp);
        outQ.FreeTensor(outR);
        inQ.FreeTensor(raw);
    }
}

// ---------------- mode 2: K2-colBlock ----------------
template <typename T>
__aicore__ inline void SmK2LoadChunk(const LocalTensor<T>& raw, const GlobalTensor<T>& xGm,
    int64_t rowBase, int64_t row0, int64_t rTt, int64_t inner, int64_t colBase, int64_t Wb,
    int64_t Wl, bool fullBlock, bool ragged2d, T negInf)
{
    const int64_t esize = (int64_t)sizeof(T);
    const bool use2d = fullBlock || ragged2d;
    if (use2d) {
        // Wb*esize is a 32B multiple (full) or arbitrary (ragged+pad): one 2D DataCopyPad
        // per chunk. Rows land at UB pitch Wl*esize = AlignUp(Wb*esize, 32); pad lanes are
        // filled with -inf. blockLen bytes are read exactly per row (in-bounds even for
        // the tensor-final row).
        DataCopyExtParams lp{(uint16_t)rTt, (uint32_t)(Wb * esize),
                             (uint32_t)((inner - Wb) * esize), 0, 0};
        DataCopyPadExtParams<T> pp{fullBlock ? false : true, 0,
                                   (uint8_t)(fullBlock ? 0 : (Wl - Wb)), negInf};
        DataCopyPad(raw, xGm[(rowBase + row0) * inner + colBase], lp, pp);
    } else {
        // ragged block: per-row 1D loads (any byte length proven); rightPadding pads the
        // row tail with -inf up to Wl lanes
        DataCopyPadExtParams<T> pp{true, 0, (uint8_t)(Wl - Wb), negInf};
        for (int64_t r = 0; r < rTt; ++r) {
            DataCopyExtParams lp{1, (uint32_t)(Wb * esize), 0, 0, 0};
            DataCopyPad(raw[r * Wl], xGm[(rowBase + row0 + r) * inner + colBase], lp, pp);
        }
    }
}

template <typename T>
__aicore__ inline void SmK2StoreChunk(const GlobalTensor<T>& zGm, const LocalTensor<T>& outT,
    int64_t rowBase, int64_t row0, int64_t rTt, int64_t inner, int64_t colBase, int64_t Wb,
    int64_t Wl, bool fullBlock, bool ragged2dStore)
{
    const int64_t esize = (int64_t)sizeof(T);
    if (fullBlock || ragged2dStore) {
        // Full blocks: Wb*esize is a 32B multiple. Ragged blocks (bit2, alignment test
        // passed): blockLen = Wb*esize arbitrary, srcStride = 0 - the MTE store advance
        // per source row is AlignUp(blockLen,32), i.e. exactly the UB tile pitch
        // Wl*esize this buffer was written with (mirror of the probed 2D-pad load path,
        // premise P-K2-2DSTORE). blockLen bytes are written exactly per row, so the pad
        // lanes are never transferred and the tensor-final row stays in-bounds.
        DataCopyExtParams sp{(uint16_t)rTt, (uint32_t)(Wb * esize), 0,
                             (uint32_t)((inner - Wb) * esize), 0};
        DataCopyPad(zGm[(rowBase + row0) * inner + colBase], outT, sp);
    } else {
        for (int64_t r = 0; r < rTt; ++r) {
            DataCopyExtParams sp{1, (uint32_t)(Wb * esize), 0, 0, 0};
            DataCopyPad(zGm[(rowBase + row0 + r) * inner + colBase], outT[r * Wl], sp);
        }
    }
}

template <typename T>
__global__ __aicore__ void softmax_k2_kernel(GM_ADDR x, GM_ADDR z, GM_ADDR ws,
    int64_t outer, int64_t R, int64_t inner, int64_t rT, int64_t W, int64_t lastW,
    int64_t nB, int64_t unitsPerCore, int64_t k2Flags, int64_t splitG, int64_t splitGroups)
{
    const int64_t esize = (int64_t)sizeof(T);
    const int64_t per32 = 32 / esize;
    constexpr bool isFp32 = std::is_same<T, float>::value;

    const int64_t units = outer * nB;
    const int64_t blk = (int64_t)GetBlockIdx();
    // bit6 selects the split-R realization: one (unit, member) pair per launched core,
    // unit = blk/g, member = blk%g (k2_partition). Otherwise the incumbent per-core
    // contiguous unit range is used.
    const bool split = (k2Flags & 64) != 0;
    // bit7 selects the cooperative whole-lane resident realization (k2_coop_partition):
    // one (group, member) pair per launched core, member = blk%g, groupIdx = blk/g.
    const bool coop = (k2Flags & 128) != 0;
    int64_t u0 = 0;
    int64_t u1 = 0;
    if (coop) {
        // The cooperative branch below derives its own (member, groupIdx, round) loop;
        // every launched core must reach AscendC::SyncAll() each round, so there is no
        // early return on an out-of-range unit here.
    } else if (split) {
        u0 = blk / splitG;
        u1 = u0 + 1;
    } else {
        u0 = blk * unitsPerCore;
        if (u0 >= units) {
            return;
        }
        u1 = SminS(u0 + unitsPerCore, units);
    }
    const int64_t WlMax = AlignUpS(SmaxS(W, lastW), per32);

    TPipe pipe;
    GlobalTensor<T> xGm, zGm;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    zGm.SetGlobalBuffer((__gm__ T*)z);
    const T negInf = SmNegInf<T>();

    // bit3 selects the whole-lane resident schedule (host-derived rT_res is passed as rT);
    // bit4 selects the per-row realization (host sets it when rT_res_chunk < CHUNK_MIN).
    const bool resident = (k2Flags & 8) != 0;
    const bool perRowRes = (k2Flags & 16) != 0;
    // bit5 selects the chunk-level two-pass streaming schedule for a non-resident unit
    // (the resident_chunk fallback). It requires Wl == WlMax; otherwise the preserved
    // streaming-online per-row realization runs for that block with the host-passed rT.
    const bool chunkStream = (k2Flags & 32) != 0;

    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
    TBuf<QuePosition::VECCALC> resBuf, bcBuf, accBuf, x32Buf, mBuf, sBuf, tBuf, cmBuf, partialBuf, pBuf;

    pipe.InitBuffer(inQ, 2, (uint64_t)(rT * WlMax * esize));
    pipe.InitBuffer(mBuf, (uint64_t)(WlMax * 4));
    pipe.InitBuffer(sBuf, (uint64_t)(WlMax * 4));
    if (coop) {
        // cooperative whole-lane resident budget (k2_coop_budget): one resident fp32
        // sub-block of ceil(R/g) x WlMax, the M/S/tV/pM-pS scratch (4 x WlMax*4) and the
        // inQ/outQ staging. The resident bcBuf, the streaming accBuf/x32Buf/cmBuf and the
        // split partialBuf are NOT allocated in this branch.
        const int64_t coopRowsM = (R + splitG - 1) / splitG;
        pipe.InitBuffer(resBuf, (uint64_t)(coopRowsM * WlMax * 4));
        // chunk-level member scratch (k2_coop_budget): bc (partial-max accumulator /
        // M_p and scale broadcast) and acc (partial-sum accumulator), each [rT, WlMax]
        // fp32, matching the host's fixed + 2*rt*WlMax*4 reservation.
        pipe.InitBuffer(bcBuf, (uint64_t)(rT * WlMax * 4));
        pipe.InitBuffer(accBuf, (uint64_t)(rT * WlMax * 4));
        pipe.InitBuffer(tBuf, (uint64_t)(WlMax * 4));
        pipe.InitBuffer(pBuf, (uint64_t)(WlMax * 4));
        if constexpr (!isFp32) {
            pipe.InitBuffer(outQ, 2, (uint64_t)(rT * WlMax * esize));
        }
    } else if (resident) {
        // one resident fp32 copy of the R x WlMax lane block; the fp32 store uses the
        // per-row path (see phase 3), so no VECOUT queue is needed for fp32.
        pipe.InitBuffer(resBuf, (uint64_t)(R * WlMax * 4));
        // fp32 scratch buffer bc of [rT, WlMax] used by the chunk-level phase-1/2/3
        // realization; allocated only when the chunk-level realization is selected
        // (bit4 clear), matching the bound resident budget (the per-row budget omits the
        // rT_res*WlMax*4 bc term).
        if (!perRowRes) {
            pipe.InitBuffer(bcBuf, (uint64_t)(rT * WlMax * 4));
        }
        if constexpr (!isFp32) {
            pipe.InitBuffer(outQ, 2, (uint64_t)(rT * WlMax * esize));
        }
    } else {
        pipe.InitBuffer(outQ, 2, (uint64_t)(rT * WlMax * esize));
        if (chunkStream) {
            // chunk-level two-pass streaming realization (k2Flags bit5): fp32 staging x32,
            // fp32 accumulator acc and fp32 broadcast bc, each [rT, WlMax]; the budget
            // reserved by the host is inQ + outQ + x32 + acc + bc + M/S + margin.
            pipe.InitBuffer(x32Buf, (uint64_t)(rT * WlMax * 4));
            pipe.InitBuffer(accBuf, (uint64_t)(rT * WlMax * 4));
            pipe.InitBuffer(bcBuf, (uint64_t)(rT * WlMax * 4));
        } else {
            if constexpr (!isFp32) {
                pipe.InitBuffer(x32Buf, (uint64_t)(rT * WlMax * 4));
            }
            pipe.InitBuffer(tBuf, (uint64_t)(WlMax * 4));
            pipe.InitBuffer(cmBuf, (uint64_t)(WlMax * 4));   // online chunk-max (k2Flags bit1)
        }
    }
    if (split) {
        // split-R partial staging: g vectors of WlMax floats each for M_p and S_p read
        // back from the GM workspace (k2_split_ub).
        pipe.InitBuffer(partialBuf, (uint64_t)(2 * splitG * WlMax * 4));
    }

    auto M = mBuf.Get<float>();
    auto S = sBuf.Get<float>();
    const float negFltMax = -3.402823466e+38f;
    const bool online = (k2Flags & 2) != 0;

    if (coop) {
        // ---- cooperative whole-lane resident realization (k2Flags bit7) ----
        // Each launched core is one (group, member) pair of a g-core cooperating group.
        // Every core executes exactly `rounds` rounds and reaches AscendC::SyncAll()
        // each round; a core whose unit is out of range (dummy round) skips the
        // loads/stores/combine but still hits the barrier, so the cross-core barrier is
        // well defined. The member holds an R/g x WlMax fp32 resident sub-block loaded
        // once, computes its per-lane partial max M_p and sum S_p = sum(exp(x - M_p)) in
        // place keeping the exponential values resident using the chunk-level member
        // form (bc/acc scratch), publishes M_p/S_p to the GM workspace, and after the
        // barrier combines the g partials per lane before the phase-3 scale and store
        // (k2_coop_schedule / k2_coop_combine). All arithmetic is fp32; the
        // max-subtraction, the -inf identity pad and the per-lane independent reduction
        // are preserved.
        GlobalTensor<float> wsGm;
        wsGm.SetGlobalBuffer((__gm__ float*)ws);
        auto res = resBuf.Get<float>();
        auto bc = bcBuf.Get<float>();
        auto acc = accBuf.Get<float>();
        auto tV = tBuf.Get<float>();
        auto pScratch = pBuf.Get<float>();

        const int64_t member = blk % splitG;
        const int64_t groupIdx = blk / splitG;
        const int64_t groups = splitGroups;
        const int64_t rounds = unitsPerCore;
        const int64_t rBase = R / splitG;
        const int64_t rRem = R % splitG;
        const int64_t r0 = member * rBase + SminS(member, rRem);
        const int64_t r1 = r0 + rBase + (member < rRem ? 1 : 0);
        const int64_t rowsM = r1 - r0;
        const int64_t Wb = inner;
        const int64_t Wl = WlMax;
        const int64_t colBase = 0;
        const bool fullBlock = (Wb % per32 == 0);
        const bool rag2d = ((k2Flags & 1) != 0) && (((inner - Wb) * esize) % 32 == 0);
        const bool ragStore2d = ((k2Flags & 4) != 0) && (((inner - Wb) * esize) % 32 == 0);

        for (int64_t round = 0; round < rounds; ++round) {
            const int64_t unit = round * groups + groupIdx;
            const bool active = unit < units;
            const int64_t wsBase = ((round * groups + groupIdx) * splitG + member) * 2 * WlMax;
            if (active) {
                // phase 1: load the member's [r0, r1) sub-range once into the resident
                // fp32 block with the chunk-level form (k2_coop_schedule): one
                // chunk-level Max into the fp32 accumulator bc per chunk, then a
                // per-row reduction of bc into the per-lane partial max M_p.
                Duplicate(bc, negFltMax, (int32_t)(rT * WlMax));
                for (int64_t row0 = 0; row0 < rowsM; row0 += rT) {
                    const int64_t rTt = SminS(rT, rowsM - row0);
                    auto raw = inQ.AllocTensor<T>();
                    SmK2LoadChunk<T>(raw, xGm, unit * R, r0 + row0, rTt, inner, colBase, Wb, Wl,
                                     fullBlock, rag2d, negInf);
                    inQ.EnQue(raw);
                    raw = inQ.DeQue<T>();
                    if constexpr (isFp32) {
                        Adds(res[row0 * WlMax], raw, 0.0f, (int32_t)(rTt * Wl));
                    } else {
                        Cast(res[row0 * WlMax], raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                    }
                    Max(bc, bc, res[row0 * WlMax], (int32_t)(rTt * WlMax));
                    inQ.FreeTensor(raw);
                }
                Duplicate(M, negFltMax, (int32_t)Wl);
                for (int64_t r = 0; r < rT; ++r) {
                    Max(M, M, bc[r * WlMax], (int32_t)Wl);
                }
                // phase 2: broadcast M_p into bc, in-place exp(x - M_p) on the resident
                // copy, accumulate the chunk-level partial sums in acc, then reduce
                // acc's rows into the per-lane partial sum S_p.
                for (int64_t r = 0; r < rT; ++r) {
                    Adds(bc[r * WlMax], M, 0.0f, (int32_t)Wl);
                }
                Duplicate(acc, 0.0f, (int32_t)(rT * WlMax));
                for (int64_t row0 = 0; row0 < rowsM; row0 += rT) {
                    const int64_t rTt = SminS(rT, rowsM - row0);
                    Sub(res[row0 * WlMax], res[row0 * WlMax], bc, (int32_t)(rTt * WlMax));
                    Exp(res[row0 * WlMax], res[row0 * WlMax], (int32_t)(rTt * WlMax));
                    Add(acc, acc, res[row0 * WlMax], (int32_t)(rTt * WlMax));
                }
                Duplicate(S, 0.0f, (int32_t)Wl);
                for (int64_t r = 0; r < rT; ++r) {
                    Add(S, S, acc[r * WlMax], (int32_t)Wl);
                }
                // publish M_p/S_p to the per-round GM slot; the barriers order the vector
                // writes against the MTE3 store and the store against SyncAll.
                PipeBarrier<PIPE_ALL>();
                {
                    DataCopyExtParams wp{1, (uint32_t)(Wl * 4), 0, 0, 0};
                    DataCopyPad(wsGm[wsBase], M, wp);
                    DataCopyPad(wsGm[wsBase + WlMax], S, wp);
                }
                PipeBarrier<PIPE_ALL>();
            }
            AscendC::SyncAll();
            if (active) {
                PipeBarrier<PIPE_ALL>();
                // phase 2: per-lane combine M = max_p M_p and S = sum_p S_p*exp(M_p - M).
                // The g partials are streamed through the single WlMax pBuf scratch, so
                // each GM read is ordered against its consuming vector op (and the
                // overwriting read against the prior vector read) by PipeBarrier; this is
                // correctness synchronization of the bound streaming combine.
                Duplicate(M, negFltMax, (int32_t)Wl);
                for (int64_t p = 0; p < splitG; ++p) {
                    DataCopyExtParams rp{1, (uint32_t)(Wl * 4), 0, 0, 0};
                    DataCopyPadExtParams<float> rpp{false, 0, 0, 0.0f};
                    DataCopyPad(pScratch, wsGm[((round * groups + groupIdx) * splitG + p) * 2 * WlMax],
                                rp, rpp);
                    PipeBarrier<PIPE_ALL>();
                    Max(M, M, pScratch, (int32_t)Wl);
                }
                Duplicate(S, 0.0f, (int32_t)Wl);
                for (int64_t p = 0; p < splitG; ++p) {
                    const int64_t baseP = ((round * groups + groupIdx) * splitG + p) * 2 * WlMax;
                    DataCopyExtParams rp{1, (uint32_t)(Wl * 4), 0, 0, 0};
                    DataCopyPadExtParams<float> rpp{false, 0, 0, 0.0f};
                    DataCopyPad(pScratch, wsGm[baseP], rp, rpp);
                    PipeBarrier<PIPE_ALL>();
                    Sub(tV, pScratch, M, (int32_t)Wl);
                    Exp(tV, tV, (int32_t)Wl);
                    PipeBarrier<PIPE_ALL>();
                    DataCopyPad(pScratch, wsGm[baseP + WlMax], rp, rpp);
                    PipeBarrier<PIPE_ALL>();
                    Mul(tV, tV, pScratch, (int32_t)Wl);
                    Add(S, S, tV, (int32_t)Wl);
                }
                // phase 3: scale the resident exponential values by exp(M_p - M)/S and
                // store. The member's own M_p is read back from its workspace slot.
                {
                    DataCopyExtParams rp{1, (uint32_t)(Wl * 4), 0, 0, 0};
                    DataCopyPadExtParams<float> rpp{false, 0, 0, 0.0f};
                    DataCopyPad(pScratch, wsGm[wsBase], rp, rpp);
                }
                PipeBarrier<PIPE_ALL>();
                Sub(tV, pScratch, M, (int32_t)Wl);
                Exp(tV, tV, (int32_t)Wl);
                Div(tV, tV, S, (int32_t)Wl);
                // broadcast the per-lane scale exp(M_p - M)/S into bc, then one
                // chunk-level Mul per chunk on the resident exponential values.
                for (int64_t r = 0; r < rT; ++r) {
                    Adds(bc[r * WlMax], tV, 0.0f, (int32_t)Wl);
                }
                for (int64_t row0 = 0; row0 < rowsM; row0 += rT) {
                    const int64_t rTt = SminS(rT, rowsM - row0);
                    Mul(res[row0 * WlMax], res[row0 * WlMax], bc, (int32_t)(rTt * WlMax));
                    if constexpr (isFp32) {
                        // order the vector Mul against the MTE3 store from res
                        PipeBarrier<PIPE_ALL>();
                        SmK2StoreChunk<T>(zGm, res[row0 * WlMax], unit * R, r0 + row0, rTt,
                                          inner, colBase, Wb, Wl, fullBlock, ragStore2d);
                    } else {
                        auto outT = outQ.AllocTensor<T>();
                        Cast(outT, res[row0 * WlMax], RoundMode::CAST_RINT, (int32_t)(rTt * Wl));
                        outQ.EnQue(outT);
                        auto outR = outQ.DeQue<T>();
                        SmK2StoreChunk<T>(zGm, outR, unit * R, r0 + row0, rTt,
                                          inner, colBase, Wb, Wl, fullBlock, ragStore2d);
                        outQ.FreeTensor(outR);
                    }
                }
            }
            // order this round's MTE3 stores against the next round's phase-1 res writes
            PipeBarrier<PIPE_ALL>();
        }
        return;
    }

    for (int64_t unit = u0; unit < u1; ++unit) {
        const int64_t o = unit / nB;
        const int64_t b = unit - o * nB;
        const int64_t Wb = (b == nB - 1) ? lastW : W;
        const int64_t Wl = AlignUpS(Wb, per32);
        const int64_t colBase = b * W;
        const bool fullBlock = (Wb % per32 == 0);
        // ragged blocks may use the 2D+pad load iff the GM row gap is 32B aligned
        // (always true for nB == 1, where the gap is 0). Falls back to per-row loads.
        const bool rag2d = ((k2Flags & 1) != 0) &&
                           (((inner - Wb) * esize) % 32 == 0);
        // bit2: ragged blocks may use the 2D-pad store (srcStride=0) under the same
        // GM-gap alignment test; otherwise per-row stores (premise P-K2-2DSTORE).
        const bool ragStore2d = ((k2Flags & 4) != 0) &&
                                (((inner - Wb) * esize) % 32 == 0);
        const int64_t rowBase = o * R;  // first GM row of this slice

        if (resident) {
            // ---- whole-lane resident schedule (k2Flags bit3) ----
            // The R x WlMax fp32 lane block is materialized once; max, exp, sum and
            // divide are computed in place on that single copy (x read once, Exp once).
            // When Wl == WlMax the whole row is staged, so the phase-1/2/3 reductions use
            // the chunk-level realization with the fp32 scratch buffer bc; when
            // Wl != WlMax the resident padding lanes are not staged and the round-2
            // per-row realization is used instead.
            auto res = resBuf.Get<float>();
            // chunk-level realization only when the host selected it (bit4 clear) AND the
            // whole row is staged (Wl == WlMax); otherwise the per-row realization is used
            // and the resident padding lanes are not staged.
            const bool chunkLevel = (!perRowRes) && (Wl == WlMax);
            if (chunkLevel) {
                auto bc = bcBuf.Get<float>();
                // phase 1: per-lane max accumulator bc over whole chunks, then reduce
                // bc's rows into M with rT per-row Max calls.
                Duplicate(bc, negFltMax, (int32_t)(rT * WlMax));
                for (int64_t row0 = 0; row0 < R; row0 += rT) {
                    const int64_t rTt = SminS(rT, R - row0);
                    auto raw = inQ.AllocTensor<T>();
                    SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                     fullBlock, rag2d, negInf);
                    inQ.EnQue(raw);
                    raw = inQ.DeQue<T>();
                    if constexpr (isFp32) {
                        Adds(res[row0 * WlMax], raw, 0.0f, (int32_t)(rTt * Wl));
                    } else {
                        Cast(res[row0 * WlMax], raw, RoundMode::CAST_NONE,
                             (int32_t)(rTt * Wl));
                    }
                    Max(bc, bc, res[row0 * WlMax], (int32_t)(rTt * WlMax));
                    inQ.FreeTensor(raw);
                }
                Duplicate(M, negFltMax, (int32_t)Wl);
                for (int64_t r = 0; r < rT; ++r) {
                    Max(M, M, bc[r * WlMax], (int32_t)Wl);
                }
                // phase 2: rebuild bc as a broadcast of M, in-place exp(x - M) on the
                // resident copy, accumulate the per-lane partial sums in bc, then reduce
                // bc's rows into S.
                for (int64_t r = 0; r < rT; ++r) {
                    Adds(bc[r * WlMax], M, 0.0f, (int32_t)Wl);
                }
                for (int64_t row0 = 0; row0 < R; row0 += rT) {
                    const int64_t rTt = SminS(rT, R - row0);
                    Sub(res[row0 * WlMax], res[row0 * WlMax], bc, (int32_t)(rTt * WlMax));
                    Exp(res[row0 * WlMax], res[row0 * WlMax], (int32_t)(rTt * WlMax));
                }
                Duplicate(bc, 0.0f, (int32_t)(rT * WlMax));
                for (int64_t row0 = 0; row0 < R; row0 += rT) {
                    const int64_t rTt = SminS(rT, R - row0);
                    Add(bc, bc, res[row0 * WlMax], (int32_t)(rTt * WlMax));
                }
                Duplicate(S, 0.0f, (int32_t)Wl);
                for (int64_t r = 0; r < rT; ++r) {
                    Add(S, S, bc[r * WlMax], (int32_t)Wl);
                }
                // phase 3: rebuild bc as a broadcast of S, in-place divide, then store.
                for (int64_t r = 0; r < rT; ++r) {
                    Adds(bc[r * WlMax], S, 0.0f, (int32_t)Wl);
                }
                if constexpr (isFp32) {
                    // derived.k2_resident_schedule: chunkLevel implies Wl == WlMax, so
                    // the resident pitch WlMax*4 equals the 2D store's source advance
                    // AlignUp(Wb*esize,32) = Wl*esize. All chunk-level Divs run first,
                    // then ONE PipeBarrier<PIPE_ALL>() orders the vector writes against
                    // the MTE3 stores, then all chunk-level stores run (ONE
                    // SmK2StoreChunk 2D DataCopyPad from res[row0*WlMax] per chunk), then
                    // ONE PipeBarrier<PIPE_ALL>() at the end of the unit so the async
                    // MTE3 stores are ordered against the next unit's phase-1 writes into
                    // the shared res buffer. The barrier count per unit is 2 instead of
                    // one per row-chunk.
                    for (int64_t row0 = 0; row0 < R; row0 += rT) {
                        const int64_t rTt = SminS(rT, R - row0);
                        Div(res[row0 * WlMax], res[row0 * WlMax], bc, (int32_t)(rTt * WlMax));
                    }
                    PipeBarrier<PIPE_ALL>();
                    for (int64_t row0 = 0; row0 < R; row0 += rT) {
                        const int64_t rTt = SminS(rT, R - row0);
                        SmK2StoreChunk<T>(zGm, res[row0 * WlMax], rowBase, row0, rTt,
                                          inner, colBase, Wb, Wl, fullBlock, ragStore2d);
                    }
                    PipeBarrier<PIPE_ALL>();
                } else {
                    for (int64_t row0 = 0; row0 < R; row0 += rT) {
                        const int64_t rTt = SminS(rT, R - row0);
                        Div(res[row0 * WlMax], res[row0 * WlMax], bc, (int32_t)(rTt * WlMax));
                        auto outT = outQ.AllocTensor<T>();
                        Cast(outT, res[row0 * WlMax], RoundMode::CAST_RINT,
                             (int32_t)(rTt * Wl));
                        outQ.EnQue(outT);
                        auto outR = outQ.DeQue<T>();
                        SmK2StoreChunk<T>(zGm, outR, rowBase, row0, rTt, inner, colBase, Wb,
                                          Wl, fullBlock, ragStore2d);
                        outQ.FreeTensor(outR);
                    }
                }
            } else {
                // ---- round-2 per-row realization (Wl != WlMax: padding lanes not staged) ----
                // phase 1: load each chunk once, stage it into the resident fp32 block and
                // fold its rows into the running per-lane max M.
                Duplicate(M, negFltMax, (int32_t)Wl);
                for (int64_t row0 = 0; row0 < R; row0 += rT) {
                    const int64_t rTt = SminS(rT, R - row0);
                    auto raw = inQ.AllocTensor<T>();
                    SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                     fullBlock, rag2d, negInf);
                    inQ.EnQue(raw);
                    raw = inQ.DeQue<T>();
                    if constexpr (isFp32) {
                        for (int64_t r = 0; r < rTt; ++r) {
                            Adds(res[(row0 + r) * WlMax], raw[r * Wl], 0.0f, (int32_t)Wl);
                        }
                    } else {
                        for (int64_t r = 0; r < rTt; ++r) {
                            Cast(res[(row0 + r) * WlMax], raw[r * Wl], RoundMode::CAST_NONE,
                                 (int32_t)Wl);
                        }
                    }
                    for (int64_t r = 0; r < rTt; ++r) {
                        Max(M, M, res[(row0 + r) * WlMax], (int32_t)Wl);
                    }
                    inQ.FreeTensor(raw);
                }
                // phase 2: in-place exp(x - M) on the resident copy, accumulating S.
                Duplicate(S, 0.0f, (int32_t)Wl);
                for (int64_t r = 0; r < R; ++r) {
                    Sub(res[r * WlMax], res[r * WlMax], M, (int32_t)Wl);
                    Exp(res[r * WlMax], res[r * WlMax], (int32_t)Wl);
                    Add(S, S, res[r * WlMax], (int32_t)Wl);
                }
                // phase 3: in-place divide and store from the resident copy.
                for (int64_t row0 = 0; row0 < R; row0 += rT) {
                    const int64_t rTt = SminS(rT, R - row0);
                    for (int64_t r = 0; r < rTt; ++r) {
                        Div(res[(row0 + r) * WlMax], res[(row0 + r) * WlMax], S, (int32_t)Wl);
                    }
                    if constexpr (isFp32) {
                        PipeBarrier<PIPE_ALL>();
                        for (int64_t r = 0; r < rTt; ++r) {
                            DataCopyExtParams sp{1, (uint32_t)(Wb * esize), 0, 0, 0};
                            DataCopyPad(zGm[(rowBase + row0 + r) * inner + colBase],
                                        res[(row0 + r) * WlMax], sp);
                        }
                    } else {
                        auto outT = outQ.AllocTensor<T>();
                        for (int64_t r = 0; r < rTt; ++r) {
                            Cast(outT[r * Wl], res[(row0 + r) * WlMax], RoundMode::CAST_RINT,
                                 (int32_t)Wl);
                        }
                        outQ.EnQue(outT);
                        auto outR = outQ.DeQue<T>();
                        SmK2StoreChunk<T>(zGm, outR, rowBase, row0, rTt, inner, colBase, Wb,
                                          Wl, fullBlock, ragStore2d);
                        outQ.FreeTensor(outR);
                    }
                }
            }
            continue;
        }

        if (split) {
            // ---- split-R streaming-online realization (k2Flags bit6) ----
            // Each launched core owns one (unit, member) pair: unit = blk/g, member =
            // blk%g. The member reduces the disjoint R sub-range [r0, r1) with the same
            // streaming-online schedule the single-core path uses, writes its per-lane
            // partial max M_p and rescaled sum S_p to the GM workspace, a cross-core
            // barrier makes the writes visible, and every member of the unit combines
            // the g partials per lane before the divide. All arithmetic is fp32.
            const int64_t member = blk % splitG;
            const int64_t o = unit / nB;
            const int64_t b = unit - o * nB;
            const int64_t Wb = (b == nB - 1) ? lastW : W;
            const int64_t Wl = AlignUpS(Wb, per32);
            const int64_t colBase = b * W;
            const bool fullBlock = (Wb % per32 == 0);
            const bool rag2d = ((k2Flags & 1) != 0) &&
                               (((inner - Wb) * esize) % 32 == 0);
            const bool ragStore2d = ((k2Flags & 4) != 0) &&
                                    (((inner - Wb) * esize) % 32 == 0);
            const int64_t rowBase = o * R;
            // member's disjoint R sub-range
            const int64_t rBase = R / splitG;
            const int64_t rRem = R % splitG;
            const int64_t r0 = member * rBase + SminS(member, rRem);
            const int64_t r1 = r0 + rBase + (member < rRem ? 1 : 0);

            GlobalTensor<float> wsGm;
            wsGm.SetGlobalBuffer((__gm__ float*)ws);

            auto tV = tBuf.Get<float>();
            auto cmT = cmBuf.Get<float>();

            // phase 1: per-lane online partial max/sum over [r0, r1)
            Duplicate(M, negFltMax, (int32_t)Wl);
            Duplicate(S, 0.0f, (int32_t)Wl);
            if (online) {
                for (int64_t row0 = r0; row0 < r1; row0 += rT) {
                    const int64_t rTt = SminS(rT, r1 - row0);
                    auto raw = inQ.AllocTensor<T>();
                    SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                     fullBlock, rag2d, negInf);
                    inQ.EnQue(raw);
                    raw = inQ.DeQue<T>();
                    if constexpr (isFp32) {
                        Duplicate(cmT, negFltMax, (int32_t)Wl);
                        for (int64_t r = 0; r < rTt; ++r) {
                            Max(cmT, cmT, raw[r * Wl], (int32_t)Wl);
                        }
                        Sub(tV, M, cmT, (int32_t)Wl);
                        Mins(tV, tV, 0.0f, (int32_t)Wl);
                        Exp(tV, tV, (int32_t)Wl);
                        Mul(S, S, tV, (int32_t)Wl);
                        Max(M, M, cmT, (int32_t)Wl);
                        for (int64_t r = 0; r < rTt; ++r) {
                            Sub(tV, raw[r * Wl], M, (int32_t)Wl);
                            Exp(tV, tV, (int32_t)Wl);
                            Add(S, S, tV, (int32_t)Wl);
                        }
                    } else {
                        auto x32 = x32Buf.Get<float>();
                        Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                        Duplicate(cmT, negFltMax, (int32_t)Wl);
                        for (int64_t r = 0; r < rTt; ++r) {
                            Max(cmT, cmT, x32[r * Wl], (int32_t)Wl);
                        }
                        Sub(tV, M, cmT, (int32_t)Wl);
                        Mins(tV, tV, 0.0f, (int32_t)Wl);
                        Exp(tV, tV, (int32_t)Wl);
                        Mul(S, S, tV, (int32_t)Wl);
                        Max(M, M, cmT, (int32_t)Wl);
                        for (int64_t r = 0; r < rTt; ++r) {
                            Sub(tV, x32[r * Wl], M, (int32_t)Wl);
                            Exp(tV, tV, (int32_t)Wl);
                            Add(S, S, tV, (int32_t)Wl);
                        }
                    }
                    inQ.FreeTensor(raw);
                }
            }
            // write this member's partials to the GM workspace (Wl floats each). The
            // barrier before the store orders the phase-1 vector writes to M/S against
            // the MTE3 store; the barrier after it orders the store against SyncAll.
            PipeBarrier<PIPE_ALL>();
            {
                DataCopyExtParams wp{1, (uint32_t)(Wl * 4), 0, 0, 0};
                DataCopyPad(wsGm[(unit * splitG + member) * 2 * WlMax], M, wp);
                DataCopyPad(wsGm[(unit * splitG + member) * 2 * WlMax + WlMax], S, wp);
            }
            PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll();
            // phase 2: combine the unit's g partials per lane
            auto partial = partialBuf.Get<float>();
            auto pM = partial;
            auto pS = partial[splitG * WlMax];
            for (int64_t p = 0; p < splitG; ++p) {
                DataCopyExtParams rp{1, (uint32_t)(Wl * 4), 0, 0, 0};
                DataCopyPadExtParams<float> rpp{false, 0, 0, 0.0f};
                DataCopyPad(pM[p * WlMax], wsGm[(unit * splitG + p) * 2 * WlMax], rp, rpp);
                DataCopyPad(pS[p * WlMax], wsGm[(unit * splitG + p) * 2 * WlMax + WlMax],
                            rp, rpp);
            }
            PipeBarrier<PIPE_ALL>();
            Adds(M, pM, 0.0f, (int32_t)Wl);            // M = M_0
            for (int64_t p = 1; p < splitG; ++p) {
                Max(M, M, pM[p * WlMax], (int32_t)Wl);
            }
            Duplicate(S, 0.0f, (int32_t)Wl);
            for (int64_t p = 0; p < splitG; ++p) {
                Sub(tV, pM[p * WlMax], M, (int32_t)Wl);
                Exp(tV, tV, (int32_t)Wl);
                Mul(tV, tV, pS[p * WlMax], (int32_t)Wl);
                Add(S, S, tV, (int32_t)Wl);
            }
            // phase 3: out = exp(x - M) / S over [r0, r1)
            for (int64_t row0 = r0; row0 < r1; row0 += rT) {
                const int64_t rTt = SminS(rT, r1 - row0);
                auto raw = inQ.AllocTensor<T>();
                SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                 fullBlock, rag2d, negInf);
                inQ.EnQue(raw);
                raw = inQ.DeQue<T>();
                auto outT = outQ.AllocTensor<T>();
                if constexpr (isFp32) {
                    for (int64_t r = 0; r < rTt; ++r) {
                        Sub(tV, raw[r * Wl], M, (int32_t)Wl);
                        Exp(tV, tV, (int32_t)Wl);
                        Div(outT[r * Wl], tV, S, (int32_t)Wl);
                    }
                } else {
                    auto x32 = x32Buf.Get<float>();
                    Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                    for (int64_t r = 0; r < rTt; ++r) {
                        Sub(x32[r * Wl], x32[r * Wl], M, (int32_t)Wl);
                        Exp(x32[r * Wl], x32[r * Wl], (int32_t)Wl);
                        Div(x32[r * Wl], x32[r * Wl], S, (int32_t)Wl);
                    }
                    Cast(outT, x32, RoundMode::CAST_RINT, (int32_t)(rTt * Wl));
                }
                outQ.EnQue(outT);
                auto outR = outQ.DeQue<T>();
                SmK2StoreChunk<T>(zGm, outR, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                  fullBlock, ragStore2d);
                outQ.FreeTensor(outR);
                inQ.FreeTensor(raw);
            }
            continue;
        }

        if (chunkStream && Wl == WlMax) {
            // ---- chunk-level two-pass streaming schedule (k2Flags bit5) ----
            // Pass A: per-lane max over all chunks with one chunk-level Max into the fp32
            // accumulator acc per chunk plus a per-row reduction of acc into M. Pass B:
            // S = sum(exp(x - M)) with a chunk-level Sub/Exp and a chunk-level Add into acc
            // per chunk plus a per-row reduction of acc into S. Pass C: the divide with a
            // chunk-level Sub/Exp/Div per chunk. x is read three times from GM and the
            // number of high-level vector API calls is much smaller than the per-row
            // schedule. The max-subtraction, the fp32 compute domain, the -inf identity pad
            // and the per-lane independent reduction are preserved.
            auto x32 = x32Buf.Get<float>();
            auto acc = accBuf.Get<float>();
            auto bc = bcBuf.Get<float>();
            // pass A: per-lane max
            Duplicate(acc, negFltMax, (int32_t)(rT * WlMax));
            for (int64_t row0 = 0; row0 < R; row0 += rT) {
                const int64_t rTt = SminS(rT, R - row0);
                auto raw = inQ.AllocTensor<T>();
                SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                 fullBlock, rag2d, negInf);
                inQ.EnQue(raw);
                raw = inQ.DeQue<T>();
                if constexpr (isFp32) {
                    Adds(x32, raw, 0.0f, (int32_t)(rTt * Wl));
                } else {
                    Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                }
                Max(acc, acc, x32, (int32_t)(rTt * Wl));
                inQ.FreeTensor(raw);
            }
            Duplicate(M, negFltMax, (int32_t)Wl);
            for (int64_t r = 0; r < rT; ++r) {
                Max(M, M, acc[r * WlMax], (int32_t)Wl);
            }
            // pass B: S = sum(exp(x - M))
            for (int64_t r = 0; r < rT; ++r) {
                Adds(bc[r * WlMax], M, 0.0f, (int32_t)Wl);
            }
            Duplicate(acc, 0.0f, (int32_t)(rT * WlMax));
            for (int64_t row0 = 0; row0 < R; row0 += rT) {
                const int64_t rTt = SminS(rT, R - row0);
                auto raw = inQ.AllocTensor<T>();
                SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                 fullBlock, rag2d, negInf);
                inQ.EnQue(raw);
                raw = inQ.DeQue<T>();
                if constexpr (isFp32) {
                    Adds(x32, raw, 0.0f, (int32_t)(rTt * Wl));
                } else {
                    Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                }
                Sub(x32, x32, bc, (int32_t)(rTt * Wl));
                Exp(x32, x32, (int32_t)(rTt * Wl));
                Add(acc, acc, x32, (int32_t)(rTt * Wl));
                inQ.FreeTensor(raw);
            }
            Duplicate(S, 0.0f, (int32_t)Wl);
            for (int64_t r = 0; r < rT; ++r) {
                Add(S, S, acc[r * WlMax], (int32_t)Wl);
            }
            // pass C: out = exp(x - M) / S
            for (int64_t r = 0; r < rT; ++r) {
                Adds(bc[r * WlMax], S, 0.0f, (int32_t)Wl);
            }
            for (int64_t row0 = 0; row0 < R; row0 += rT) {
                const int64_t rTt = SminS(rT, R - row0);
                auto raw = inQ.AllocTensor<T>();
                SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                 fullBlock, rag2d, negInf);
                inQ.EnQue(raw);
                raw = inQ.DeQue<T>();
                if constexpr (isFp32) {
                    Adds(x32, raw, 0.0f, (int32_t)(rTt * Wl));
                } else {
                    Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                }
                Sub(x32, x32, bc, (int32_t)(rTt * Wl));
                Exp(x32, x32, (int32_t)(rTt * Wl));
                Div(x32, x32, bc, (int32_t)(rTt * Wl));
                if constexpr (isFp32) {
                    // x32 pitch is WlMax*4 = Wl*4 here, matching SmK2StoreChunk's UB pitch
                    SmK2StoreChunk<T>(zGm, x32, rowBase, row0, rTt, inner, colBase, Wb,
                                      Wl, fullBlock, ragStore2d);
                } else {
                    auto outT = outQ.AllocTensor<T>();
                    Cast(outT, x32, RoundMode::CAST_RINT, (int32_t)(rTt * Wl));
                    outQ.EnQue(outT);
                    auto outR = outQ.DeQue<T>();
                    SmK2StoreChunk<T>(zGm, outR, rowBase, row0, rTt, inner, colBase, Wb,
                                      Wl, fullBlock, ragStore2d);
                    outQ.FreeTensor(outR);
                }
                inQ.FreeTensor(raw);
            }
            continue;
        }

        // streaming-online per-row realization (incumbent). It is also the Wl != WlMax
        // fallback of the chunk-level streaming schedule; in that case acc/bc (>= WlMax*4)
        // serve as the tV/cmT scratch so the bound chunk-level budget is not exceeded.
        LocalTensor<float> tV;
        LocalTensor<float> cmT;
        if (chunkStream) {
            tV = accBuf.Get<float>();
            cmT = bcBuf.Get<float>();
        } else {
            tV = tBuf.Get<float>();
            cmT = cmBuf.Get<float>();
        }
        if (online) {
            // ---- fused pass A+B: online lane max + rescaled running sum ----
            // M stays -FLT_MAX until a finite chunk max appears; every exp argument is
            // min(M-cm, 0) or x-Mnew <= 0, so no overflow, and degenerate columns follow
            // torch semantics (+inf -> NaN via inf-inf, all -inf -> 0/0 = NaN).
            Duplicate(M, negFltMax, (int32_t)Wl);
            Duplicate(S, 0.0f, (int32_t)Wl);
            for (int64_t row0 = 0; row0 < R; row0 += rT) {
                const int64_t rTt = SminS(rT, R - row0);
                auto raw = inQ.AllocTensor<T>();
                SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                                 fullBlock, rag2d, negInf);
                inQ.EnQue(raw);
                raw = inQ.DeQue<T>();
                if constexpr (isFp32) {
                    Duplicate(cmT, negFltMax, (int32_t)Wl);
                    for (int64_t r = 0; r < rTt; ++r) {
                        Max(cmT, cmT, raw[r * Wl], (int32_t)Wl);
                    }
                    Sub(tV, M, cmT, (int32_t)Wl);      // M - cm (M not yet updated)
                    Mins(tV, tV, 0.0f, (int32_t)Wl);   // min(M-cm,0) = M - Mnew
                    Exp(tV, tV, (int32_t)Wl);
                    Mul(S, S, tV, (int32_t)Wl);
                    Max(M, M, cmT, (int32_t)Wl);       // M <- Mnew (dst aliases src1)
                    for (int64_t r = 0; r < rTt; ++r) {
                        Sub(tV, raw[r * Wl], M, (int32_t)Wl);
                        Exp(tV, tV, (int32_t)Wl);
                        Add(S, S, tV, (int32_t)Wl);
                    }
                } else {
                    auto x32 = x32Buf.Get<float>();
                    Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                    Duplicate(cmT, negFltMax, (int32_t)Wl);
                    for (int64_t r = 0; r < rTt; ++r) {
                        Max(cmT, cmT, x32[r * Wl], (int32_t)Wl);
                    }
                    Sub(tV, M, cmT, (int32_t)Wl);
                    Mins(tV, tV, 0.0f, (int32_t)Wl);
                    Exp(tV, tV, (int32_t)Wl);
                    Mul(S, S, tV, (int32_t)Wl);
                    Max(M, M, cmT, (int32_t)Wl);
                    for (int64_t r = 0; r < rTt; ++r) {
                        Sub(tV, x32[r * Wl], M, (int32_t)Wl);
                        Exp(tV, tV, (int32_t)Wl);
                        Add(S, S, tV, (int32_t)Wl);
                    }
                }
                inQ.FreeTensor(raw);
            }
        } else {
        // ---- pass A: per-lane running max ----
        Duplicate(M, negFltMax, (int32_t)Wl);
        for (int64_t row0 = 0; row0 < R; row0 += rT) {
            const int64_t rTt = SminS(rT, R - row0);
            auto raw = inQ.AllocTensor<T>();
            SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                             fullBlock, rag2d, negInf);
            inQ.EnQue(raw);
            raw = inQ.DeQue<T>();
            if constexpr (isFp32) {
                for (int64_t r = 0; r < rTt; ++r) {
                    Max(M, M, raw[r * Wl], (int32_t)Wl);
                }
            } else {
                auto x32 = x32Buf.Get<float>();
                Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                for (int64_t r = 0; r < rTt; ++r) {
                    Max(M, M, x32[r * Wl], (int32_t)Wl);
                }
            }
            inQ.FreeTensor(raw);
        }

        // ---- pass B: S = sum(exp(x - M)) per lane ----
        Duplicate(S, 0.0f, (int32_t)Wl);
        for (int64_t row0 = 0; row0 < R; row0 += rT) {
            const int64_t rTt = SminS(rT, R - row0);
            auto raw = inQ.AllocTensor<T>();
            SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                             fullBlock, rag2d, negInf);
            inQ.EnQue(raw);
            raw = inQ.DeQue<T>();
            if constexpr (isFp32) {
                for (int64_t r = 0; r < rTt; ++r) {
                    Sub(tV, raw[r * Wl], M, (int32_t)Wl);
                    Exp(tV, tV, (int32_t)Wl);
                    Add(S, S, tV, (int32_t)Wl);
                }
            } else {
                auto x32 = x32Buf.Get<float>();
                Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                for (int64_t r = 0; r < rTt; ++r) {
                    Sub(tV, x32[r * Wl], M, (int32_t)Wl);
                    Exp(tV, tV, (int32_t)Wl);
                    Add(S, S, tV, (int32_t)Wl);
                }
            }
            inQ.FreeTensor(raw);
        }
        }

        // ---- pass C: out = exp(x - M) / S, cast, store ----
        for (int64_t row0 = 0; row0 < R; row0 += rT) {
            const int64_t rTt = SminS(rT, R - row0);
            auto raw = inQ.AllocTensor<T>();
            SmK2LoadChunk<T>(raw, xGm, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                             fullBlock, rag2d, negInf);
            inQ.EnQue(raw);
            raw = inQ.DeQue<T>();
            auto outT = outQ.AllocTensor<T>();
            if constexpr (isFp32) {
                for (int64_t r = 0; r < rTt; ++r) {
                    Sub(tV, raw[r * Wl], M, (int32_t)Wl);
                    Exp(tV, tV, (int32_t)Wl);
                    Div(outT[r * Wl], tV, S, (int32_t)Wl);
                }
            } else {
                auto x32 = x32Buf.Get<float>();
                Cast(x32, raw, RoundMode::CAST_NONE, (int32_t)(rTt * Wl));
                for (int64_t r = 0; r < rTt; ++r) {
                    Sub(x32[r * Wl], x32[r * Wl], M, (int32_t)Wl);
                    Exp(x32[r * Wl], x32[r * Wl], (int32_t)Wl);
                    Div(x32[r * Wl], x32[r * Wl], S, (int32_t)Wl);
                }
                Cast(outT, x32, RoundMode::CAST_RINT, (int32_t)(rTt * Wl));
            }
            outQ.EnQue(outT);
            auto outR = outQ.DeQue<T>();
            SmK2StoreChunk<T>(zGm, outR, rowBase, row0, rTt, inner, colBase, Wb, Wl,
                              fullBlock, ragStore2d);
            outQ.FreeTensor(outR);
            inQ.FreeTensor(raw);
        }
    }
}

} // namespace

// ---------------- host tiling ----------------
void calc_softmax_tiling(int64_t rank, int64_t dim, const int64_t* shape, int64_t esize,
                         SoftmaxTilingOut* out)
{
    auto pm = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizeU64 = 0;
    pm->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeU64);
    int64_t ub = (int64_t)ubSizeU64;
    int64_t coreNum = (int64_t)pm->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    const int64_t R = shape[dim];
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer *= shape[i];
    }
    for (int64_t i = dim + 1; i < rank; ++i) {
        inner *= shape[i];
    }
    const int64_t per32 = 32 / esize;
    const bool isFp32 = (esize == 4);

    out->mode = 0;
    out->numBlocks = 1;
    out->rowsPerCore = 1;
    out->G = 1;
    out->Kpad = 1;
    out->tmpSize = 0;
    out->tp0 = 0;
    out->tp1 = 0;
    out->tp2 = 0;
    out->tp3 = 0;
    out->tp4 = 0;
    out->tp5 = 0;
    out->tp6 = 0;
    out->tp7 = 0;
    out->outer = outer;
    out->R = R;
    out->inner = inner;
    out->W = inner;
    out->lastW = inner;
    out->nB = 1;
    out->splitG = 0;
    out->splitGroups = 0;
    out->wsBytes = 0;

    if (inner == 1) {
        // ---- K1 ----
        // Rows of length R: slice o occupies elements [o*R, o*R+R)  =>  m = outer rows.
        const int64_t m = outer;
        const int64_t Kpad = HAlignUp(R, per32);
        const int64_t pitchB = Kpad * esize;
        int64_t G = 1;
        int64_t tmpSize = 0;
        if (R * esize >= 32) {
            // mode 0 budget: inQ 2 + outQ 2 (+ x32 4 for sub-fp32) per row of pitchB
            const int64_t bytesPerRow = (isFp32 ? 4 : 8) * pitchB;
            const int64_t Gmax = std::min<int64_t>(4095, m);
            for (int64_t g = Gmax; g >= 1; --g) {
                const int64_t fixed = bytesPerRow * g + 64 * g + SM_MARGIN;
                const int64_t avail = ub - fixed;
                if (avail < 1024) {
                    continue;
                }
                SmShapeT sh(std::vector<int64_t>{g, Kpad});
                const uint32_t minT = GetSoftMaxMinTmpSize(sh, 4, false);
                const uint32_t maxT = GetSoftMaxMaxTmpSize(sh, 4, false);
                if ((int64_t)minT > avail) {
                    continue;
                }
                G = g;
                tmpSize = (int64_t)std::min<uint64_t>((uint64_t)maxT, (uint64_t)avail);
                break;
            }
            if (tmpSize == 0) {
                SmShapeT sh1(std::vector<int64_t>{1, Kpad});
                tmpSize = (int64_t)GetSoftMaxMinTmpSize(sh1, 4, false);
                G = 1;
            }
            out->mode = 0;
        } else {
            // mode 1 budget
            int64_t Gmax = std::min<int64_t>((int64_t)2048, m);
            for (int64_t g = Gmax; g >= 1; --g) {
                if (g * Kpad > 16384) {
                    continue;
                }
                const int64_t pkB = g * R * esize;
                const int64_t fixed = 2 * (32 + pkB) + 2 * HAlignUp(pkB, 32) +
                                     g * Kpad * esize + 4 * g * Kpad + 4 * g * Kpad +
                                     4 * g * R + 64 * g + SM_MARGIN;
                const int64_t avail = ub - fixed;
                if (avail < 1024) {
                    continue;
                }
                SmShapeT sh(std::vector<int64_t>{g, Kpad});
                const uint32_t minT = GetSoftMaxMinTmpSize(sh, 4, false);
                const uint32_t maxT = GetSoftMaxMaxTmpSize(sh, 4, false);
                if ((int64_t)minT > avail) {
                    continue;
                }
                G = g;
                tmpSize = (int64_t)std::min<uint64_t>((uint64_t)maxT, (uint64_t)avail);
                break;
            }
            if (tmpSize == 0) {
                SmShapeT sh1(std::vector<int64_t>{1, Kpad});
                tmpSize = (int64_t)GetSoftMaxMinTmpSize(sh1, 4, false);
                G = 1;
            }
            out->mode = 1;
        }
        out->Kpad = Kpad;
        out->G = G;
        out->tmpSize = tmpSize;
        out->numBlocks = std::min<int64_t>(coreNum, m);
        out->rowsPerCore = (m + out->numBlocks - 1) / out->numBlocks;
    } else {
        // ---- K2 ----
        const int64_t nBmax = std::max<int64_t>((int64_t)1, inner / per32);

        // Incumbent balanced-split geometry for a given nB target (Lower derived
        // k2_geometry): nB = clamp(target, 1, nBmax); W = AlignUp(ceil(inner/nB),
        // per32) (at least per32); nB = ceil(inner/W); lastW = inner-(nB-1)*W;
        // WlMax = AlignUp(max(W,lastW), per32). Lower solve 90bd087e7c3a06f0877f9ed17d4f803070be36e5dc64c967788e25e3cbd1b321
        // selected k2_column_balance = balanced (premise P-K2-BALANCE): split inner into
        // equal per32-aligned blocks so lastW <= W. This shrinks WlMax, which raises the
        // row-chunk rT and cuts the per-unit MTE op count on the critical core
        // (case 4: 1536 -> 378 2D ops/unit; all blocks full both before and after).
        auto k2Geom = [&](int64_t nBtarget, int64_t& nB, int64_t& W, int64_t& lastW,
                          int64_t& WlMax) {
            nB = std::max<int64_t>((int64_t)1, std::min<int64_t>(nBmax, nBtarget));
            if (nB <= 1) {
                nB = 1;
                W = inner;
                lastW = inner;
            } else {
                W = HAlignUp((inner + nB - 1) / nB, per32);
                if (W < per32) {
                    W = per32;
                }
                nB = (inner - 1) / W + 1;
                if (nB <= 1) {
                    nB = 1;
                    W = inner;
                    lastW = inner;
                } else {
                    lastW = inner - (nB - 1) * W;   // guaranteed in [1, W]
                }
            }
            WlMax = HAlignUp(Hmax(W, lastW), per32);
        };

        // bc-free resident chunk size rT_res_perrow for a given WlMax (Lower derived
        // k2_resident_budget): the largest rt in [1, R] with
        // R*WlMax*4 + 4*WlMax*4 + SM_MARGIN + 2*rt*WlMax*esize
        //   + (isFp32 ? 0 : 2*rt*WlMax*esize) <= UB; 0 if even rt=1 does not fit.
        auto residentPerRowRt = [&](int64_t WlMax) -> int64_t {
            const int64_t base = R * WlMax * 4 + 4 * WlMax * 4 + SM_MARGIN;
            for (int64_t rt = R; rt >= 1; --rt) {
                const int64_t per = 2 * rt * WlMax * esize
                                  + (isFp32 ? 0 : 2 * rt * WlMax * esize);
                if (base + per <= ub) {
                    return rt;
                }
            }
            return 0;
        };

        // Incumbent geometry (nB0).
        int64_t nBtarget0 = 1;
        if (outer < coreNum) {
            nBtarget0 = std::min<int64_t>(nBmax, (coreNum + outer - 1) / outer);
        }
        int64_t nB;
        int64_t W;
        int64_t lastW;
        int64_t WlMax;
        k2Geom(nBtarget0, nB, W, lastW, WlMax);
        // Base (incumbent) geometry nB0, captured before resident_expand_nB may replace
        // it. The split-R candidate geometry is derived from nB0 (k2_split_geometry),
        // not from the possibly-expanded incumbent geometry.
        const int64_t nB0 = nB;

        // resident_expand_nB (bound theta, models/decision_plan.json
        // k2_expand_nB_geometry): if the incumbent geometry's resident block already
        // fits (bc-free rT_res_perrow >= 1) the incumbent geometry is used with the
        // resident schedule and no expansion is attempted; otherwise every nB candidate
        // in [nB0, nBmax] whose resident block fits is evaluated, the candidate with the
        // lowest per-core modeled resident cost is selected, and it is used with the
        // resident schedule only when that cost is below the per-core modeled streaming
        // cost of the incumbent geometry; otherwise the incumbent geometry and the
        // streaming-online schedule are kept. Ties keep the first (lowest-nB) candidate,
        // matching the executable model's strict `<` comparison. The constants and
        // formulas are exactly those of the derived entry k2_expand_nB_geometry.
        const int64_t vw = isFp32 ? 64 : 128;
        const double A_COST = 0.002335288485221164;
        const double B_COST = 0.11719289889315515;
        const double C_CALL = 0.008;
        const double C_CHUNK = 0.05;
        const double ALPHA = 0.124;
        const double C_V = 0.00601;
        const double C_PR = 0.0105;
        constexpr int64_t CHUNK_MIN = 4;

        auto geomUpc = [&](int64_t gnB) -> int64_t {
            const int64_t gunits = outer * gnB;
            const int64_t gnb = std::min<int64_t>(coreNum, gunits);
            return (gunits + gnb - 1) / gnb;
        };
        auto streamingCost = [&](int64_t gWlMax, int64_t gupc) -> double {
            const double X = (double)gupc * (double)R * (double)((gWlMax + vw - 1) / vw) * 7.0;
            const double N = (double)gupc * (double)R;
            return A_COST * X + B_COST * N;
        };
        // Per-core modeled resident cost of one candidate geometry. Returns false when no
        // realization fits. rT_res and the realization follow k2_resident_budget.
        auto residentCost = [&](int64_t gnB, int64_t gW, int64_t gWlMax, int64_t gupc,
                                double& cost, int64_t& rTresOut, bool& perRowOut) -> bool {
            const int64_t base = R * gWlMax * 4 + 4 * gWlMax * 4 + SM_MARGIN;
            int64_t rtc = 0;
            for (int64_t rt = R; rt >= 1; --rt) {
                const int64_t per = 2 * rt * gWlMax * esize
                                  + (isFp32 ? 0 : 2 * rt * gWlMax * esize);
                if (base + per + rt * gWlMax * 4 <= ub) { rtc = rt; break; }
            }
            const int64_t rtp = residentPerRowRt(gWlMax);
            if (rtp < 1) {
                return false;
            }
            int64_t rTres = 0;
            bool perRow = false;
            int64_t VWops = 0;
            int64_t Calls = 0;
            int64_t chunks = 0;
            if (rtc >= CHUNK_MIN) {
                int64_t bestRt = CHUNK_MIN;
                double bestC = 0.0;
                bool have = false;
                for (int64_t rt = CHUNK_MIN; rt <= rtc; ++rt) {
                    const int64_t ch = (R + rt - 1) / rt;
                    const int64_t vwo = 5 * ch * ((rt * gWlMax + vw - 1) / vw)
                                      + 4 * rt * ((gW + vw - 1) / vw);
                    const int64_t cl = 5 * ch + 4 * rt;
                    const double c = A_COST * (double)vwo + C_CALL * (double)cl
                                   + C_CHUNK * (double)ch;
                    if (!have || c <= bestC + 1e-12) { bestC = c; bestRt = rt; have = true; }
                }
                rTres = bestRt;
                perRow = false;
                chunks = (R + rTres - 1) / rTres;
                VWops = 5 * chunks * ((rTres * gWlMax + vw - 1) / vw)
                      + 4 * rTres * ((gW + vw - 1) / vw);
                Calls = 5 * chunks + 4 * rTres;
            } else {
                rTres = rtp;
                perRow = true;
                chunks = (R + rTres - 1) / rTres;
                VWops = 7 * R * ((gW + vw - 1) / vw)
                      + chunks * ((rTres * gWlMax + vw - 1) / vw);
                Calls = 7 * R + chunks;
            }
            const double X = (double)gupc * (double)R * (double)((gWlMax + vw - 1) / vw) * 7.0;
            const double N = (double)gupc * (double)R;
            const int64_t pr = (gnB > 1 && (((inner - gW) * esize) % 32 != 0)) ? 1 : 0;
            const double P = (double)gupc * (double)R * (double)pr;
            cost = (double)gupc * (A_COST * (double)VWops + C_CALL * (double)Calls
                                   + C_CHUNK * (double)chunks)
                 + (B_COST - 7.0 * C_CALL) * N
                 - (ALPHA * C_V * X + C_PR * P);
            rTresOut = rTres;
            perRowOut = perRow;
            return true;
        };

        const double tStream = streamingCost(WlMax, geomUpc(nB));
        double bestResCost = 0.0;
        bool haveBest = false;
        bool useResident = false;
        int64_t bestRT = 0;
        bool bestPerRow = false;
        int64_t bestNB = nB, bestW = W, bestLastW = lastW, bestWlMax = WlMax;
        // resident_expand_nB (k2_expand_nB_geometry): if the incumbent geometry's
        // resident block already fits (bc-free rT_res_perrow >= 1), the incumbent
        // geometry is used with the resident schedule and no expansion is attempted.
        // Otherwise the nB candidates in [nB0, nBmax] whose resident block fits are
        // searched and the cheapest fitting one is used only when its per-core modeled
        // resident cost is below the incumbent streaming cost.
        if (residentPerRowRt(WlMax) >= 1) {
            double c;
            int64_t rt;
            bool pr;
            if (residentCost(nB, W, WlMax, geomUpc(nB), c, rt, pr)) {
                useResident = true;
                bestRT = rt;
                bestPerRow = pr;
                bestResCost = c;   // selected incumbent resident prediction (T0 omitted)
            }
        } else {
            for (int64_t cand = nB; cand <= nBmax; ++cand) {
                int64_t cnB, cW, clastW, cWlMax;
                k2Geom(cand, cnB, cW, clastW, cWlMax);
                double c;
                int64_t rt;
                bool pr;
                if (!residentCost(cnB, cW, cWlMax, geomUpc(cnB), c, rt, pr)) {
                    continue;
                }
                if (!haveBest || c < bestResCost) {
                    bestResCost = c;
                    haveBest = true;
                    bestNB = cnB; bestW = cW; bestLastW = clastW; bestWlMax = cWlMax;
                    bestRT = rt; bestPerRow = pr;
                }
            }
            if (haveBest && bestResCost < tStream) {
                useResident = true;
                nB = bestNB; W = bestW; lastW = bestLastW; WlMax = bestWlMax;
            }
        }

        // split_r_selective (bound theta): after the parent resident_expand_nB
        // selection, evaluate split-R candidates g in [2, coreNum] over the incumbent
        // base geometry nB0 (k2_split_geometry), keep only those with
        // units_s*g <= coreNum and rT_split >= 4 (k2_split_rt), and use the cheapest
        // one when its modeled per-core cost is below the selected incumbent
        // prediction t_inc (k2_split_cost / k2_split_selection). splitCost includes the
        // calibrated split fixed cost C_SPLIT_FIXED = 18.0 us. T0 is a common intercept
        // that cancels in both the g tie-break and the incumbent comparison.
        const double tInc = useResident ? bestResCost : tStream;
        // k2_split_selection: a chunk_level resident incumbent is the most efficient
        // single-core realization and is kept unchanged; split is not evaluated for it
        // (round-2 measurement refuted split against the chunk_level resident on case 16).
        const bool incumbentChunkLevel = useResident && !bestPerRow;
        bool useSplit = false;
        int64_t splitG = 0;
        int64_t splitRT = 0;
        int64_t splitWlMax = 0;
        int64_t splitUnits = 0;
        int64_t splitNB = 0;
        int64_t splitW = 0;
        int64_t splitLastW = 0;
        // Selected split prediction t_split (k2_split_selection); hoisted so the
        // cooperative comparison below can use it as t_inc when the split realization
        // is selected (k2_coop_selection: t_inc is the split prediction t_split).
        double bestSplitCost = 0.0;
        if (!incumbentChunkLevel) {
            constexpr double C_SYNC = 3.0;
            constexpr double C_SPLIT_FIXED = 18.0;
            constexpr double C_GM = 1e-6;   // us/byte, GM workspace traffic term
            constexpr int64_t SPLIT_RT_MIN = 4;
            bool haveSplit = false;
            for (int64_t g = 2; g <= coreNum; ++g) {
                const int64_t nBs = std::max<int64_t>((int64_t)1, (nB0 + g - 1) / g);
                int64_t snB, sW, slastW, sWlMax;
                k2Geom(nBs, snB, sW, slastW, sWlMax);
                const int64_t sunits = outer * snB;
                if (sunits * g > coreNum) {
                    continue;
                }
                int64_t srt = 0;
                for (int64_t rt = std::min<int64_t>(R, (int64_t)4095); rt >= 1; --rt) {
                    const int64_t by = (isFp32 ? 4 : 8) * rt * sWlMax * esize
                                     + 16 * sWlMax + 2 * g * sWlMax * 4 + SM_MARGIN;
                    if (by <= ub) { srt = rt; break; }
                }
                if (srt < SPLIT_RT_MIN) {
                    continue;
                }
                const int64_t rows = (R + g - 1) / g;
                const double X7 = (double)rows * (double)((sWlMax + vw - 1) / vw) * 7.0;
                const double N = (double)rows;
                // derived.k2_split_cost: includes the GM workspace traffic term
                // C_GM*(2+2g)*WlMax_s*4 (matches models/k2_split_solver.py _split_pred).
                const double splitCost = A_COST * X7 + B_COST * N + C_SYNC + C_SPLIT_FIXED
                                       + C_GM * (double)(2 + 2 * g) * (double)sWlMax * 4.0;
                if (!haveSplit || splitCost < bestSplitCost) {
                    haveSplit = true;
                    bestSplitCost = splitCost;
                    splitG = g;
                    splitRT = srt;
                    splitWlMax = sWlMax;
                    splitUnits = sunits;
                    splitNB = snB;
                    splitW = sW;
                    splitLastW = slastW;
                }
            }
            if (haveSplit && bestSplitCost < tInc) {
                useSplit = true;
            }
        }

        // coop_resident_selective (bound theta, k2_coop_selection): after the
        // split_r_selective selection, t_inc is the selected incumbent prediction
        // (bestResCost for a resident incumbent, tStream for a streaming incumbent,
        // bestSplitCost for a split incumbent). Evaluate cooperative whole-lane resident
        // candidates restricted to nB = 1 (k2_coop_geometry) and use the cheapest one
        // when its modeled per-core cost is below t_inc (ties -> smallest g). T0 is a
        // common intercept and cancels in both the g tie-break and the comparison.
        const double tIncSel = useSplit ? bestSplitCost : tInc;
        bool useCoop = false;
        int64_t coopG = 0;
        int64_t coopGroups = 0;
        int64_t coopRounds = 0;
        int64_t coopRT = 0;
        int64_t coopWlMax = 0;
        {
            constexpr double C_SYNC = 3.0;
            constexpr double C_GM = 1e-6;   // us/byte, GM workspace traffic term
            // round-2 calibration of the combine GM read + per-partial barrier cost
            // (derived.k2_coop_selection, matches models/coop_solver.py _coop_pred)
            constexpr double C_COMBINE = 3.4e-4;   // us per lane
            const int64_t coopWl = HAlignUp(inner, per32);
            const int64_t coopUnits = outer;
            double bestCoopCost = 0.0;
            bool haveCoop = false;
            for (int64_t g = 2; g <= coreNum; ++g) {
                const int64_t groups = coreNum / g;
                if (groups < 1) {
                    continue;
                }
                const int64_t rounds = (coopUnits + groups - 1) / groups;
                const int64_t rowsM = (R + g - 1) / g;
                const int64_t fixed = rowsM * coopWl * 4 + 4 * coopWl * 4 + SM_MARGIN;
                if (fixed > ub) {
                    continue;
                }
                int64_t rt = 0;
                for (int64_t t = rowsM; t >= 1; --t) {
                    // k2_coop_budget: the member's chunk-level scratch bc+acc
                    // (2*rt*WlMax*4) plus the inQ/outQ staging
                    // ((isFp32?2:4)*rt*WlMax*esize) must fit with the resident
                    // sub-block and the M/S/tV/pBuf scratch.
                    const int64_t per = 2 * t * coopWl * 4
                                      + (isFp32 ? 2 : 4) * t * coopWl * esize;
                    if (fixed + per <= ub) { rt = t; break; }
                }
                if (rt < 1) {
                    continue;
                }
                const int64_t chunks = (rowsM + rt - 1) / rt;
                // chunk-level member features (k2_coop_selection): the same form as
                // the parent's single-core chunk_level resident
                // (_resident_chunk_features with R -> rowsM and Wl -> WlMax).
                const int64_t VWops = 5 * chunks * ((rt * coopWl + vw - 1) / vw)
                                    + 4 * rt * ((coopWl + vw - 1) / vw);
                const int64_t Calls = 5 * chunks + 4 * rt;
                const int64_t combineVWops = 5 * g * ((coopWl + vw - 1) / vw);
                const int64_t combineCalls = 7 * g;
                const double cost = (double)rounds
                    * (A_COST * (double)VWops + (B_COST - 7.0 * C_CALL) * (double)rowsM
                       + C_CALL * (double)Calls + C_CHUNK * (double)chunks)
                    + (double)rounds * (A_COST * (double)combineVWops
                                        + C_CALL * (double)combineCalls)
                    + C_SYNC * (double)rounds
                    + C_GM * (double)rounds * 4.0 * (double)g * (double)coopWl * 4.0
                    + C_COMBINE * (double)rounds * (double)g * (double)coopWl;
                if (!haveCoop || cost < bestCoopCost) {
                    haveCoop = true;
                    bestCoopCost = cost;
                    coopG = g;
                    coopGroups = groups;
                    coopRounds = rounds;
                    coopRT = rt;
                    coopWlMax = coopWl;
                }
            }
            if (haveCoop && bestCoopCost < tIncSel) {
                useCoop = true;
            }
        }

        // Resident whole-lane variant (k2Flags bit3): the geometry, realization and
        // rT_res were selected by the cost-based resident_expand_nB search above
        // (models/decision_plan.json k2_expand_nB_geometry / k2_resident_budget). The
        // chunk-level realization is used by the kernel only when bit4 is clear AND
        // Wl == WlMax; otherwise the round-2 per-row realization runs (k2_resident_schedule).
        const bool resident = useResident;
        const bool perRowRes = useResident && bestPerRow;
        const int64_t rTres = bestRT;
        out->mode = 2;
        if (useCoop) {
            // cooperative whole-lane resident realization (k2_flags bit7): full-width
            // column block (nB = 1), one (group, member) pair per launched core.
            out->tp0 = 7 | 128;
            out->W = inner;
            out->lastW = inner;
            out->nB = 1;
            out->G = coopRT;
            out->splitG = coopG;
            out->splitGroups = coopGroups;
            out->numBlocks = coopGroups * coopG;
            out->rowsPerCore = coopRounds;
            out->wsBytes = coopRounds * out->numBlocks * 2 * coopWlMax * 4;
        } else if (useSplit) {
            // split-R realization: one (unit, member) pair per launched core.
            out->tp0 = 7 | 64;
            out->W = splitW;
            out->lastW = splitLastW;
            out->nB = splitNB;
            out->G = splitRT;
            out->splitG = splitG;
            out->wsBytes = splitUnits * splitG * 2 * splitWlMax * 4;
            out->numBlocks = splitUnits * splitG;
            out->rowsPerCore = 1;
        } else {
            const int64_t pitchBmax = WlMax * esize;
            int64_t rT = 1;
            for (int64_t rt = std::min<int64_t>(R, (int64_t)4095); rt >= 1; --rt) {
                const int64_t per = (isFp32 ? 4 : 8) * rt * pitchBmax;
                // 16 B/lane: M,S,tV (3x4B) + online chunk-max cmT (4B)
                const int64_t fixed = per + 16 * WlMax + SM_MARGIN;
                if (fixed <= ub) {
                    rT = rt;
                    break;
                }
            }
            // Lower solves: flags fb73735d.../3cb1340a... selected bit0 twod_pad ragged
            // loads, bit1 online two-pass; solve 90bd087e... re-justified both and
            // selected the balanced column split above (k2_column_balance). The
            // iteration-3 lower solves (models/solve_record_iteration3*.yaml) selected
            // bit2 twod_pad ragged stores (k2_ragged_store), guarded in-kernel by the
            // same GM-gap alignment test as the load path; per-row stores remain the
            // fallback when the test fails. bit3 is the resident-variant flag; bit4 marks
            // the per-row resident realization (rT_res_chunk < CHUNK_MIN). Under the
            // bound theta the non-resident fallback is the incumbent streaming-online
            // schedule, so bit5 (chunk-level two-pass streaming) is NEVER set by the host;
            // the kernel bit5 branch is retained but unreachable (models/decision_plan.json
            // k2_flags / k2_streaming_chunk_budget / k2_streaming_chunk_schedule).
            out->tp0 = 7 | (resident ? 8 : 0) | (perRowRes ? 16 : 0);
            out->W = W;
            out->lastW = lastW;
            out->nB = nB;
            out->G = resident ? rTres : rT;
            const int64_t totalUnits = outer * nB;
            out->numBlocks = std::min<int64_t>(coreNum, totalUnits);
            out->rowsPerCore = (totalUnits + out->numBlocks - 1) / out->numBlocks;
        }
    }
}

// ---------------- launch wrappers ----------------
extern "C" {

void launch_softmax_k1_std_float(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream)
{
    softmax_k1_std_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, m, R, G, Kpad,
        rowsPerCore, tmpSize);
}

void launch_softmax_k1_std_half(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream)
{
    softmax_k1_std_kernel<half><<<numBlocks, nullptr, stream>>>(x, z, m, R, G, Kpad,
        rowsPerCore, tmpSize);
}

void launch_softmax_k1_std_bf16(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream)
{
    softmax_k1_std_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, z, m, R, G, Kpad,
        rowsPerCore, tmpSize);
}

void launch_softmax_k1_packed_float(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream)
{
    softmax_k1_packed_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, m, R, G, Kpad,
        rowsPerCore, tmpSize);
}

void launch_softmax_k1_packed_half(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream)
{
    softmax_k1_packed_kernel<half><<<numBlocks, nullptr, stream>>>(x, z, m, R, G, Kpad,
        rowsPerCore, tmpSize);
}

void launch_softmax_k1_packed_bf16(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream)
{
    softmax_k1_packed_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, z, m, R, G, Kpad,
        rowsPerCore, tmpSize);
}

void launch_softmax_k2_float(GM_ADDR x, GM_ADDR z, GM_ADDR ws, int64_t numBlocks, int64_t outer,
    int64_t R, int64_t inner, int64_t rT, int64_t W, int64_t lastW, int64_t nB, int64_t unitsPerCore,
    int64_t k2Flags, int64_t splitG, int64_t splitGroups, void* stream)
{
    softmax_k2_kernel<float><<<numBlocks, nullptr, stream>>>(x, z, ws, outer, R, inner,
        rT, W, lastW, nB, unitsPerCore, k2Flags, splitG, splitGroups);
}

void launch_softmax_k2_half(GM_ADDR x, GM_ADDR z, GM_ADDR ws, int64_t numBlocks, int64_t outer,
    int64_t R, int64_t inner, int64_t rT, int64_t W, int64_t lastW, int64_t nB, int64_t unitsPerCore,
    int64_t k2Flags, int64_t splitG, int64_t splitGroups, void* stream)
{
    softmax_k2_kernel<half><<<numBlocks, nullptr, stream>>>(x, z, ws, outer, R, inner,
        rT, W, lastW, nB, unitsPerCore, k2Flags, splitG, splitGroups);
}

void launch_softmax_k2_bf16(GM_ADDR x, GM_ADDR z, GM_ADDR ws, int64_t numBlocks, int64_t outer,
    int64_t R, int64_t inner, int64_t rT, int64_t W, int64_t lastW, int64_t nB, int64_t unitsPerCore,
    int64_t k2Flags, int64_t splitG, int64_t splitGroups, void* stream)
{
    softmax_k2_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, z, ws, outer, R, inner,
        rT, W, lastW, nB, unitsPerCore, k2Flags, splitG, splitGroups);
}

} // extern "C"
