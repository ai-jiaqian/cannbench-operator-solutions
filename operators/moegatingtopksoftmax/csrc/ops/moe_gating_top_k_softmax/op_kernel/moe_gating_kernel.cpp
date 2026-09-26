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
 * \file moe_gating_kernel.cpp
 * \brief MoeGatingTopKSoftmax (per-row softmax + topk + index outputs) kernel, tiling and launch.
 *        Compiled with bisheng + -xasc for Ascend 910B (dav-2201).
 *
 *  p        = softmax(x, dim=-1)                       (fp32 on device)
 *  y        = topk(p, k) values                        (dtype of x)
 *  expert   = column index of each selected value      (int32)
 *  row_idx  = j * totalRows + r                        (int32, flattened global position)
 *
 *  finished[r] == true  =>  expert_idx[.., j] = inner (== E) for that row; y and row_idx unchanged.
 *
 * Layout: the (..., E) tensor is flattened to (totalRows, inner).  Each core owns a contiguous band of
 * rows and walks it in tiles of R rows.  A row lives in UB with a padded pitch L = align32(inner)
 * (fp32 view) so every data block boundary stays aligned; the tail lanes of a row hold a very negative
 * sentinel so that padded lanes always sort last.
 *
 * All arithmetic is fp32.  The topk uses the ISASI sort proposal: Sort32 per 32-element chunk followed
 * by a MrgSort merge tree whose width shrinks exactly to 1 (a "split" step is a free relabelling of the
 * already sorted contiguous runs).  Extract finally splits the (score, index) structures into value and
 * index planes.
 *
 * Note: exp(x - max) is sorted instead of the normalised softmax.  Division by the row sum is a
 * positive per-row scale, so it cannot change the ordering or the tie behaviour of the sort; dividing
 * only the first align(k, block) columns after the sort therefore reproduces the reference output
 * while removing a whole strip loop from the hot path.
 */

#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "moe_gating_launch.h"

using namespace AscendC;

namespace mig {
constexpr float kNEG = -3.0e38f;
constexpr int64_t kMaxSortRepeat = 255;

__aicore__ inline int64_t alignUp(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

__aicore__ inline int64_t imin(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
} // namespace mig

/*! \brief compile time dispatch for the (un)supporting dtype conversion pairs */
template <typename T, bool IS_FLOAT>
struct MoeCast;

template <typename T>
struct MoeCast<T, true> {
    static __aicore__ inline void Up(const LocalTensor<float>& dst, const LocalTensor<T>& src, int32_t n)
    {
        (void)dst;
        (void)src;
        (void)n;
    }
    static __aicore__ inline void Down(const LocalTensor<T>& dst, const LocalTensor<float>& src, int32_t n)
    {
        (void)dst;
        (void)src;
        (void)n;
    }
};

template <typename T>
struct MoeCast<T, false> {
    static __aicore__ inline void Up(const LocalTensor<float>& dst, const LocalTensor<T>& src, int32_t n)
    {
        Cast(dst, src, RoundMode::CAST_NONE, n);
    }
    static __aicore__ inline void Down(const LocalTensor<T>& dst, const LocalTensor<float>& src, int32_t n)
    {
        Cast(dst, src, RoundMode::CAST_RINT, n);
    }
};

template <typename T>
__global__ __aicore__ void moe_gating_kernel(GM_ADDR xPtr, GM_ADDR finPtr, GM_ADDR yPtr, GM_ADDR eiPtr,
                                             GM_ADDR riPtr, int64_t totalRows, int64_t inner, int64_t k,
                                             int64_t rowsPerCore, int64_t rowsPerTile, int64_t hasFin)
{
    const int64_t coreIdx = GetBlockIdx();
    const int64_t rowStart0 = coreIdx * rowsPerCore;
    if (rowStart0 >= totalRows) {
        return;
    }
    const int64_t rowEndAll = mig::imin(rowStart0 + rowsPerCore, totalRows);
    const int64_t nRowsAll = rowEndAll - rowStart0;
    if (nRowsAll <= 0) {
        return;
    }

    const int64_t L = mig::alignUp(inner, 32);
    const int64_t Lb = L / 8;
    const int64_t nb = L / 32;
    const int64_t kz8 = mig::alignUp(k, 8);
    const int64_t nstrip = (L + 63) / 64;
    const int64_t nstripPad = mig::alignUp(nstrip, 8);
    const int64_t R = rowsPerTile;
    const int64_t m8cnt = mig::alignUp(R, 8);
    const int64_t ts = (int64_t)sizeof(T);
    const int64_t inPadBlk = (L * ts - mig::alignUp(inner * ts, 32)) / 32;
    const int64_t yPadBlk = (L * ts - mig::alignUp(k * ts, 32)) / 32;
    const int64_t eiPadBlk = (L - kz8) / 8;
    const int64_t riPadBlk = (kz8 - 8) / 8;
    const int64_t kstrip = (kz8 + 63) / 64;
    const int64_t kLb = kz8 / 8;
    // Elements the copy-in must right-pad so that a row block reaches a 32B boundary.
    const int64_t rpBytes = (32 - (inner * ts) % 32) % 32;
    const int64_t rpElems = rpBytes / ts;
    // When inner is already a multiple of 32 the GM pitch and the UB row pitch coincide, so the whole
    // tile can be moved with one contiguous transfer instead of `rc` row-granular blocks.
    const bool contiguousLoad = (L == inner);
    const bool isFloat = std::is_same<T, float>::value;

    AscendC::TPipe pipe;
    TBuf<TPosition::VECCALC> tbA, tbB, tbXf, tbIdx, tbI, tbRi;
    TBuf<TPosition::VECCALC> tbMp, tbSp, tbMt, tbSt, tbM8, tbS8, tbAr, tbArI, tbCol, tbRow, tbFu, tbFh, tbFf;

    pipe.InitBuffer(tbA, R * L * 8 + 32);
    pipe.InitBuffer(tbB, R * L * 8 + 32);
    pipe.InitBuffer(tbXf, R * L * 4 + 32);
    pipe.InitBuffer(tbIdx, R * L * 4 + 32);
    pipe.InitBuffer(tbI, R * L * 4 + 32);
    pipe.InitBuffer(tbRi, R * kz8 * 4 + 32);
    pipe.InitBuffer(tbMp, R * nstripPad * 4 + 32);
    pipe.InitBuffer(tbSp, R * nstripPad * 4 + 32);
    pipe.InitBuffer(tbMt, m8cnt * 4 + 32);
    pipe.InitBuffer(tbSt, m8cnt * 4 + 32);
    pipe.InitBuffer(tbM8, m8cnt * 32 + 32);
    pipe.InitBuffer(tbS8, m8cnt * 32 + 32);
    pipe.InitBuffer(tbAr, L * 4 + 32);
    pipe.InitBuffer(tbArI, L * 4 + 32);
    pipe.InitBuffer(tbCol, kz8 * 4 + 32);
    pipe.InitBuffer(tbRow, m8cnt * 4 + 32);
    pipe.InitBuffer(tbFu, 256);
    pipe.InitBuffer(tbFh, m8cnt * 2 + 64);
    pipe.InitBuffer(tbFf, m8cnt * 4 + 64);

    LocalTensor<float> xf = tbXf.Get<float>();
    LocalTensor<float> mp = tbMp.Get<float>();
    LocalTensor<float> sp = tbSp.Get<float>();
    LocalTensor<float> mt = tbMt.Get<float>();
    LocalTensor<float> st = tbSt.Get<float>();
    LocalTensor<float> m8l = tbM8.Get<float>();
    LocalTensor<float> s8l = tbS8.Get<float>();
    LocalTensor<float> arng = tbAr.Get<float>();
    LocalTensor<int32_t> arngI = tbArI.Get<int32_t>();
    LocalTensor<float> colv = tbCol.Get<float>();
    LocalTensor<float> rwf = tbRow.Get<float>();
    LocalTensor<float> iF = tbI.Get<float>();
    LocalTensor<int32_t> idxI = tbIdx.Get<int32_t>();
    LocalTensor<int32_t> riI = tbRi.Get<int32_t>();
    LocalTensor<int32_t> iout = tbI.Get<int32_t>();
    LocalTensor<uint32_t> idxU32 = tbIdx.Get<uint32_t>();
    LocalTensor<uint32_t> ioutU32 = tbI.Get<uint32_t>();
    LocalTensor<uint8_t> fu = tbFu.Get<uint8_t>();
    LocalTensor<half> fh = tbFh.Get<half>();
    LocalTensor<float> ff = tbFf.Get<float>();
    LocalTensor<float> bufA = tbA.Get<float>();
    LocalTensor<float> bufB = tbB.Get<float>();
    LocalTensor<T> yv = tbB.Get<T>();

    LocalTensor<T> rawT = isFloat ? tbXf.Get<T>() : tbA.Get<T>();
    LocalTensor<T> yOutSrc = isFloat ? tbXf.Get<T>() : yv;

    GlobalTensor<T> xG;
    GlobalTensor<T> yG;
    GlobalTensor<int32_t> eiG;
    GlobalTensor<int32_t> riG;
    GlobalTensor<uint8_t> finG;
    xG.SetGlobalBuffer((__gm__ T*)xPtr);
    yG.SetGlobalBuffer((__gm__ T*)yPtr);
    eiG.SetGlobalBuffer((__gm__ int32_t*)eiPtr);
    riG.SetGlobalBuffer((__gm__ int32_t*)riPtr);
    if (hasFin != 0) {
        finG.SetGlobalBuffer((__gm__ uint8_t*)finPtr);
    }

    // ---------- one time initialisation ----------
    Duplicate(rawT, static_cast<T>(mig::kNEG), (int32_t)(R * L));
    for (int64_t i = 0; i < L; ++i) {
        arng.SetValue(i, (float)i);
        arngI.SetValue(i, (int32_t)i);
    }
    for (int64_t j = 0; j < kz8; ++j) {
        colv.SetValue(j, (float)(j * totalRows));
    }
    if (nstrip > 1) {
        Duplicate(mp, mig::kNEG, (int32_t)(R * nstripPad));
        Duplicate(sp, 0.0f, (int32_t)(R * nstripPad));
    }
    PipeBarrier<PIPE_ALL>();

    // The index plane fed to Sort32 is identical for every tile unless `finished` rewrites it, so it is
    // materialised once for the whole launch.
    if (hasFin == 0) {
        for (int64_t s = 0; s < nstrip; ++s) {
            const int64_t cnt = mig::imin(64, L - 64 * s);
            UnaryRepeatParams up;
            up.dstBlkStride = 1;
            up.srcBlkStride = 1;
            up.dstRepStride = (uint8_t)Lb;
            up.srcRepStride = 0;
            Adds(idxI[64 * s], arngI[64 * s], 0, (int32_t)cnt, (int32_t)R, up);
        }
        PipeBarrier<PIPE_ALL>();
    }

    DataCopyPadExtParams<T> padT;
    padT.isPad = true;
    padT.leftPadding = 0;
    padT.rightPadding = (uint8_t)rpElems;
    padT.paddingValue = static_cast<T>(mig::kNEG);

    for (int64_t base = 0; base < nRowsAll; base += R) {
        const int64_t rc = mig::imin(R, nRowsAll - base);
        const int32_t rc32 = (int32_t)rc;
        const int64_t gRow0 = rowStart0 + base;

        // ---------- 1. load the tile ----------
        {
            DataCopyExtParams cp;
            if (contiguousLoad) {
                cp.blockCount = 1;
                cp.blockLen = (uint32_t)(rc * inner * ts);
                cp.srcStride = 0;
                cp.dstStride = 0;
                cp.rsv = 0;
            } else {
                cp.blockCount = (uint16_t)rc;
                cp.blockLen = (uint32_t)(inner * ts);
                cp.srcStride = 0;
                cp.dstStride = (uint32_t)inPadBlk;
                cp.rsv = 0;
            }
            DataCopyPad(rawT, xG[gRow0 * inner], cp, padT);
        }
        PipeBarrier<PIPE_ALL>();
        MoeCast<T, std::is_same<T, float>::value>::Up(xf, rawT, (int32_t)(rc * L));

        // ---------- 2. index source tile (only when `finished` rewrites it) ----------
        if (hasFin != 0) {
            for (int64_t s = 0; s < nstrip; ++s) {
                const int64_t cnt = mig::imin(64, L - 64 * s);
                UnaryRepeatParams up;
                up.dstBlkStride = 1;
                up.srcBlkStride = 1;
                up.dstRepStride = (uint8_t)Lb;
                up.srcRepStride = 0;
                Adds(iF[64 * s], arng[64 * s], 0.0f, (int32_t)cnt, rc32, up);
            }
            {
                DataCopyExtParams cp;
                cp.blockCount = 1;
                cp.blockLen = (uint32_t)rc;
                cp.srcStride = 0;
                cp.dstStride = 0;
                cp.rsv = 0;
                DataCopyPadExtParams<uint8_t> pp{false, 0, 0, 0};
                DataCopyPad(fu, finG[gRow0], cp, pp);
            }
            PipeBarrier<PIPE_ALL>();
            Cast(fh, fu, RoundMode::CAST_NONE, rc32);
            Cast(ff, fh, RoundMode::CAST_NONE, rc32);
            Brcb(m8l, ff, (uint8_t)((rc + 7) / 8), {1, 8});
            // idx' = idx - fin * (idx - inner)   ==  inner when fin else idx
            Muls(bufA, iF, 1.0f, (int32_t)(rc * L));
            Adds(iF, iF, (float)(-inner), (int32_t)(rc * L));
            for (int64_t s = 0; s < nstrip; ++s) {
                const int64_t cnt = mig::imin(64, L - 64 * s);
                BinaryRepeatParams bp;
                bp.dstBlkStride = 1;
                bp.src0BlkStride = 1;
                bp.src1BlkStride = 0;
                bp.dstRepStride = (uint8_t)Lb;
                bp.src0RepStride = (uint8_t)Lb;
                bp.src1RepStride = 1;
                Mul(iF[64 * s], iF[64 * s], m8l, (int32_t)cnt, rc32, bp);
            }
            Sub(iF, bufA, iF, (int32_t)(rc * L));
            Cast(idxI, iF, RoundMode::CAST_RINT, (int32_t)(rc * L));
        }

        // ---------- 3. row max ----------
        if (nstrip == 1) {
            WholeReduceMax(mt, xf, (int32_t)L, rc32, 1, 1, (int32_t)Lb, ReduceOrder::ORDER_ONLY_VALUE);
        } else {
            for (int64_t s = 0; s < nstrip; ++s) {
                const int64_t cnt = mig::imin(64, L - 64 * s);
                WholeReduceMax(mp[s], xf[64 * s], (int32_t)cnt, rc32, (int32_t)nstripPad, 1, (int32_t)Lb,
                               ReduceOrder::ORDER_ONLY_VALUE);
            }
            WholeReduceMax(mt, mp, (int32_t)nstripPad, rc32, 1, 1, (int32_t)(nstripPad / 8),
                           ReduceOrder::ORDER_ONLY_VALUE);
        }

        // ---------- 4. exp(x - rowmax) ----------
        Brcb(m8l, mt, (uint8_t)((rc + 7) / 8), {1, 8});
        for (int64_t s = 0; s < nstrip; ++s) {
            const int64_t cnt = mig::imin(64, L - 64 * s);
            BinaryRepeatParams bp;
            bp.dstBlkStride = 1;
            bp.src0BlkStride = 1;
            bp.src1BlkStride = 0;
            bp.dstRepStride = (uint8_t)Lb;
            bp.src0RepStride = (uint8_t)Lb;
            bp.src1RepStride = 1;
            Sub(xf[64 * s], xf[64 * s], m8l, (int32_t)cnt, rc32, bp);
        }
        Exp(xf, xf, (int32_t)(rc * L));

        // ---------- 5. row sum ----------
        if (nstrip == 1) {
            WholeReduceSum(st, xf, (int32_t)L, rc32, 1, 1, (int32_t)Lb);
        } else {
            for (int64_t s = 0; s < nstrip; ++s) {
                const int64_t cnt = mig::imin(64, L - 64 * s);
                WholeReduceSum(sp[s], xf[64 * s], (int32_t)cnt, rc32, (int32_t)nstripPad, 1, (int32_t)Lb);
            }
            WholeReduceSum(st, sp, (int32_t)nstripPad, rc32, 1, 1, (int32_t)(nstripPad / 8));
        }
        Brcb(s8l, st, (uint8_t)((rc + 7) / 8), {1, 8});

        // ---------- 6. sort exp(x - max) per row ----------
        Sort32(bufA, xf, idxU32, (int32_t)(rc * nb));
        int64_t chunks = nb;
        int64_t run = 32;
        LocalTensor<float> cur = bufA;
        LocalTensor<float> other = bufB;
        while (chunks > 1) {
            if ((chunks % 4) != 0) {
                chunks *= 2;
                run /= 2;
                continue;
            }
            MrgSortSrcList<float> srcList;
            srcList.src1 = cur[0];
            srcList.src2 = cur[run * 2];
            srcList.src3 = cur[run * 4];
            srcList.src4 = cur[run * 6];
            MrgSort4Info params;
            params.elementLengths[0] = (uint16_t)run;
            params.elementLengths[1] = (uint16_t)run;
            params.elementLengths[2] = (uint16_t)run;
            params.elementLengths[3] = (uint16_t)run;
            params.ifExhaustedSuspension = false;
            params.validBit = 15;
            params.repeatTimes = (int32_t)(rc * chunks / 4);
            MrgSort<float>(other, srcList, params);
            chunks /= 4;
            run *= 4;
            LocalTensor<float> tmp = cur;
            cur = other;
            other = tmp;
        }

        // ---------- 7. split the (score, index) structures ----------
        Extract<float>(xf, ioutU32, cur, (int32_t)(rc * L / 32));

        // Normalise only the leading columns that can reach the output; the sort order is unaffected by
        // the positive per-row scale, so dividing after the sort is equivalent and cheaper.
        for (int64_t s = 0; s < kstrip; ++s) {
            const int64_t cnt = mig::imin(64, kz8 - 64 * s);
            BinaryRepeatParams bp;
            bp.dstBlkStride = 1;
            bp.src0BlkStride = 1;
            bp.src1BlkStride = 0;
            bp.dstRepStride = (uint8_t)Lb;
            bp.src0RepStride = (uint8_t)Lb;
            bp.src1RepStride = 1;
            Div(xf[64 * s], xf[64 * s], s8l, (int32_t)cnt, rc32, bp);
        }
        MoeCast<T, std::is_same<T, float>::value>::Down(yv, xf, (int32_t)(rc * L));

        // ---------- 8. row index tile ----------
        for (int64_t r = 0; r < rc; ++r) {
            rwf.SetValue(r, (float)(gRow0 + r));
        }
        Brcb(m8l, rwf, (uint8_t)((rc + 7) / 8), {1, 8});
        for (int64_t s = 0; s < kstrip; ++s) {
            const int64_t cnt = mig::imin(64, kz8 - 64 * s);
            UnaryRepeatParams up;
            up.dstBlkStride = 1;
            up.srcBlkStride = 0;
            up.dstRepStride = (uint8_t)kLb;
            up.srcRepStride = 1;
            Adds(bufA[64 * s], m8l, 0.0f, (int32_t)cnt, rc32, up);
        }
        for (int64_t s = 0; s < kstrip; ++s) {
            const int64_t cnt = mig::imin(64, kz8 - 64 * s);
            BinaryRepeatParams bp;
            bp.dstBlkStride = 1;
            bp.src0BlkStride = 1;
            bp.src1BlkStride = 1;
            bp.dstRepStride = (uint8_t)kLb;
            bp.src0RepStride = (uint8_t)kLb;
            bp.src1RepStride = 0;
            Add(bufA[64 * s], bufA[64 * s], colv[64 * s], (int32_t)cnt, rc32, bp);
        }
        Cast(riI, bufA, RoundMode::CAST_RINT, (int32_t)(rc * kz8));

        // ---------- 9. write outputs ----------
        PipeBarrier<PIPE_ALL>();
        {
            DataCopyExtParams cp;
            cp.blockCount = (uint16_t)rc;
            cp.blockLen = (uint32_t)(k * ts);
            cp.srcStride = (uint32_t)yPadBlk;
            cp.dstStride = 0;
            cp.rsv = 0;
            DataCopyPad(yG[gRow0 * k], yOutSrc, cp);
        }
        {
            DataCopyExtParams cp;
            cp.blockCount = (uint16_t)rc;
            cp.blockLen = (uint32_t)(k * 4);
            cp.srcStride = (uint32_t)eiPadBlk;
            cp.dstStride = 0;
            cp.rsv = 0;
            DataCopyPad(eiG[gRow0 * k], iout, cp);
        }
        {
            DataCopyExtParams cp;
            cp.blockCount = (uint16_t)rc;
            cp.blockLen = (uint32_t)(k * 4);
            cp.srcStride = (uint32_t)riPadBlk;
            cp.dstStride = 0;
            cp.rsv = 0;
            DataCopyPad(riG[gRow0 * k], riI, cp);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

/*!
 * \brief host side tiling: (numBlocks, rowsPerCore, rowsPerTile)
 */
std::tuple<int64_t, int64_t, int64_t> calc_moe_gating_tiling_params(int64_t totalRows, int64_t inner, int64_t k,
                                                                    int64_t typeSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    const int64_t L = (inner + 31) / 32 * 32;
    const int64_t kz8 = (k + 7) / 8 * 8;
    // Reserve: framework reserved UB, the small side buffers (~25KB) and a safety margin.
    int64_t avail = (int64_t)ubSize - 40960;
    if (avail < 4096) {
        avail = 4096;
    }
    const int64_t perRow = L * 28 + kz8 * 4 + 256;
    const int64_t rUb = avail / perRow;
    const int64_t rSort = mig::kMaxSortRepeat * 32 / L;
    int64_t R = std::min<int64_t>(128, std::min(rUb, rSort));
    if (R < 1) {
        R = 1;
    }
    int64_t numBlocks = std::min<int64_t>(coreNum, totalRows);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t rowsPerCore = (totalRows + numBlocks - 1) / numBlocks;
    if (rowsPerCore < 1) {
        rowsPerCore = 1;
    }
    return std::make_tuple(numBlocks, rowsPerCore, R);
}

extern "C" {

void launch_moe_gating_kernel_float(GM_ADDR x, GM_ADDR finished, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx,
                                    int64_t totalRows, int64_t inner, int64_t k, int64_t numBlocks,
                                    int64_t rowsPerCore, int64_t rowsPerTile, int64_t hasFinished, void* stream)
{
    moe_gating_kernel<float><<<numBlocks, nullptr, stream>>>(x, finished, y, expertIdx, rowIdx, totalRows, inner, k,
                                                             rowsPerCore, rowsPerTile, hasFinished);
}

void launch_moe_gating_kernel_half(GM_ADDR x, GM_ADDR finished, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx,
                                   int64_t totalRows, int64_t inner, int64_t k, int64_t numBlocks,
                                   int64_t rowsPerCore, int64_t rowsPerTile, int64_t hasFinished, void* stream)
{
    moe_gating_kernel<half><<<numBlocks, nullptr, stream>>>(x, finished, y, expertIdx, rowIdx, totalRows, inner, k,
                                                            rowsPerCore, rowsPerTile, hasFinished);
}

void launch_moe_gating_kernel_bfloat16(GM_ADDR x, GM_ADDR finished, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx,
                                       int64_t totalRows, int64_t inner, int64_t k, int64_t numBlocks,
                                       int64_t rowsPerCore, int64_t rowsPerTile, int64_t hasFinished, void* stream)
{
    moe_gating_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, finished, y, expertIdx, rowIdx, totalRows, inner,
                                                                  k, rowsPerCore, rowsPerTile, hasFinished);
}
}
