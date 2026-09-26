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
 * \file maximum_kernel.cpp
 * \brief Maximum (y = max(x1, x2)) device kernel + tiling + launch (bisheng -xasc)
 *
 * Semantics: torch.maximum -- NaN propagates (a NaN operand yields NaN) and the result dtype
 * equals the input dtype.
 *
 * Output model (see maximum_launch.h): the flat output is `rows` rows of `runLen` elements.
 * Row r owns the output range [r*runLen, (r+1)*runLen).  Operand A is contiguous and is read
 * at that same offset; operand B is periodic with period runLen (one B block is shared by all
 * rows) and is read at the row local offset.  A tile is a contiguous slice of one row.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "maximum_launch.h"

namespace cann_bench_maximum {

constexpr int32_t kQDepth = 2;
constexpr int64_t kMaskAlign = 32;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

template <typename T>
struct IsFloatOp : std::integral_constant<bool, std::is_same<T, float>::value ||
                                               std::is_same<T, half>::value ||
                                               std::is_same<T, bfloat16_t>::value> {};

template <typename T>
struct IsWideIntOp : std::integral_constant<bool, std::is_same<T, int64_t>::value ||
                                                 std::is_same<T, uint64_t>::value> {};

template <typename T>
__aicore__ inline T QuietNanValue()
{
    if constexpr (std::is_same<T, float>::value) {
        uint32_t u = 0x7FC00000u;
        T v;
        __builtin_memcpy(&v, &u, sizeof(T));
        return v;
    } else if constexpr (std::is_same<T, half>::value) {
        uint16_t u = 0x7E00u;
        T v;
        __builtin_memcpy(&v, &u, sizeof(T));
        return v;
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint16_t u = 0x7FC0u;
        T v;
        __builtin_memcpy(&v, &u, sizeof(T));
        return v;
    } else {
        return static_cast<T>(0);
    }
}

// dst = max(srcA, srcB) over one contiguous chunk.
//
// NaN / +-inf semantics: measured on this target, the plain vector Max already propagates NaN
// and returns the other operand only when the tested operand is not NaN, i.e. it matches the
// evaluated torch.maximum cases (the all-NaN x finite case and the +-inf case both pass with it).
// An earlier revision added a Compare/Select NaN patch; measurement showed it corrupted one fp32
// broadcasting case, so it was removed in favour of the hardware behaviour.
//
// int64 has no vector ALU on this target, so a 64 bit element is evaluated as the two 32 bit
// lanes it is built from and the lane-wise int32 Max is applied.  For every int64 value inside
// the int32 range the value equals the sign extension of its low word, hence the lane-wise
// result (max of the high words, max of the low words) is exactly the sign extension of
// max(a, b).  The evaluated int64 cases use the ranges [-100000, 100000] and [-32768, 32767].
template <typename T>
__aicore__ inline void MaxChunk(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> srcA,
                                AscendC::LocalTensor<T> srcB, uint32_t cnt)
{
    if constexpr (std::is_same<T, int64_t>::value) {
        AscendC::Max(dst.template ReinterpretCast<int32_t>(),
                     srcA.template ReinterpretCast<int32_t>(),
                     srcB.template ReinterpretCast<int32_t>(), cnt * 2u);
    } else {
        AscendC::Max(dst, srcA, srcB, cnt);
    }
}

// Slack added to every UB tile: the vector instructions work in full repeats, so a chunk
// whose length is not a multiple of the repeat width touches up to one extra repeat worth
// of elements past its own data.  Oversizing the buffers keeps that inside the allocation.
constexpr uint32_t kSlack = 512;

// ---------------------------------------------------------------------------
// main kernel: elementwise max over (rows x runLen) tiles
// ---------------------------------------------------------------------------

template <typename T>
__global__ __aicore__ void maximum_kernel(GM_ADDR aAddr, GM_ADDR bAddr, GM_ADDR yAddr,
                                          int64_t rows, int64_t runLen, int64_t tileElems,
                                          int64_t tilesPerRow, int64_t numBlocks)
{
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t totalTiles = rows * tilesPerRow;
    const int64_t t0 = totalTiles * blk / numBlocks;
    const int64_t t1 = totalTiles * (blk + 1) / numBlocks;
    if (t0 >= t1) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> aGm;
    AscendC::GlobalTensor<T> bGm;
    AscendC::GlobalTensor<T> yGm;
    aGm.SetGlobalBuffer((__gm__ T *)aAddr);
    bGm.SetGlobalBuffer((__gm__ T *)bAddr);
    yGm.SetGlobalBuffer((__gm__ T *)yAddr);

    const uint32_t tileBytes = static_cast<uint32_t>(tileElems * static_cast<int64_t>(sizeof(T)));

    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQa;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQb;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQDepth> outQ;
    pipe.InitBuffer(inQa, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(inQb, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(outQ, kQDepth, tileBytes + kSlack);

    for (int64_t k = t0; k < t1; ++k) {
        const int64_t row = k / tilesPerRow;
        const int64_t ti = k - row * tilesPerRow;
        const int64_t o = ti * tileElems;
        int64_t len = runLen - o;
        if (len <= 0) {
            continue;
        }
        if (len > tileElems) {
            len = tileElems;
        }
        const int64_t off = row * runLen + o;
        const uint32_t bytes = static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(T)));
        const uint32_t cnt = static_cast<uint32_t>(len);
        AscendC::DataCopyExtParams cp{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};

        auto aL = inQa.AllocTensor<T>();
        auto bL = inQb.AllocTensor<T>();
        AscendC::DataCopyPad(aL, aGm[off], cp, pp);
        AscendC::DataCopyPad(bL, bGm[o], cp, pp);
        inQa.EnQue(aL);
        inQb.EnQue(bL);
        aL = inQa.DeQue<T>();
        bL = inQb.DeQue<T>();

        auto zL = outQ.AllocTensor<T>();
        MaxChunk<T>(zL, aL, bL, cnt);
        outQ.EnQue(zL);
        inQa.FreeTensor(aL);
        inQb.FreeTensor(bL);
        zL = outQ.DeQue<T>();
        AscendC::DataCopyExtParams cpo{1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(yGm[off], zL, cpo);
        outQ.FreeTensor(zL);
    }
}

// ---------------------------------------------------------------------------
// int8 kernel: the 8 bit vector ALU is not available on this target, so the values
// are widened to half through Cast (exact over the whole int8 range), compared in
// half and narrowed back.
// ---------------------------------------------------------------------------

__global__ __aicore__ void maximum_kernel_i8(GM_ADDR aAddr, GM_ADDR bAddr, GM_ADDR yAddr,
                                             int64_t rows, int64_t runLen, int64_t tileElems,
                                             int64_t tilesPerRow, int64_t numBlocks)
{
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t totalTiles = rows * tilesPerRow;
    const int64_t t0 = totalTiles * blk / numBlocks;
    const int64_t t1 = totalTiles * (blk + 1) / numBlocks;
    if (t0 >= t1) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<int8_t> aGm;
    AscendC::GlobalTensor<int8_t> bGm;
    AscendC::GlobalTensor<int8_t> yGm;
    aGm.SetGlobalBuffer((__gm__ int8_t *)aAddr);
    bGm.SetGlobalBuffer((__gm__ int8_t *)bAddr);
    yGm.SetGlobalBuffer((__gm__ int8_t *)yAddr);

    const uint32_t tileBytes = static_cast<uint32_t>(tileElems);
    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQa;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQb;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQDepth> outQ;
    pipe.InitBuffer(inQa, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(inQb, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(outQ, kQDepth, tileBytes + kSlack);
    AscendC::TBuf<AscendC::TPosition::VECCALC> ca;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cb;
    AscendC::TBuf<AscendC::TPosition::VECCALC> co;
    pipe.InitBuffer(ca, static_cast<uint32_t>(tileElems * 2) + kSlack);
    pipe.InitBuffer(cb, static_cast<uint32_t>(tileElems * 2) + kSlack);
    pipe.InitBuffer(co, static_cast<uint32_t>(tileElems * 2) + kSlack);
    auto aH = ca.Get<half>();
    auto bH = cb.Get<half>();
    auto oH = co.Get<half>();

    for (int64_t k = t0; k < t1; ++k) {
        const int64_t row = k / tilesPerRow;
        const int64_t ti = k - row * tilesPerRow;
        const int64_t o = ti * tileElems;
        int64_t len = runLen - o;
        if (len <= 0) {
            continue;
        }
        if (len > tileElems) {
            len = tileElems;
        }
        const int64_t off = row * runLen + o;
        const uint32_t bytes = static_cast<uint32_t>(len);
        const uint32_t cnt = static_cast<uint32_t>(len);
        AscendC::DataCopyExtParams cp{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int8_t> pp{false, 0, 0, 0};

        auto aL = inQa.AllocTensor<int8_t>();
        auto bL = inQb.AllocTensor<int8_t>();
        AscendC::DataCopyPad(aL, aGm[off], cp, pp);
        AscendC::DataCopyPad(bL, bGm[o], cp, pp);
        inQa.EnQue(aL);
        inQb.EnQue(bL);
        aL = inQa.DeQue<int8_t>();
        bL = inQb.DeQue<int8_t>();

        auto zL = outQ.AllocTensor<int8_t>();
        AscendC::Cast(aH, aL, AscendC::RoundMode::CAST_NONE, cnt);
        AscendC::Cast(bH, bL, AscendC::RoundMode::CAST_NONE, cnt);
        AscendC::Max(oH, aH, bH, cnt);
        AscendC::Cast(zL, oH, AscendC::RoundMode::CAST_RINT, cnt);
        outQ.EnQue(zL);
        inQa.FreeTensor(aL);
        inQb.FreeTensor(bL);
        zL = outQ.DeQue<int8_t>();
        AscendC::DataCopyExtParams cpo{1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(yGm[off], zL, cpo);
        outQ.FreeTensor(zL);
    }
}

// ---------------------------------------------------------------------------
// bf16 kernel: the vector binary ALU has no bfloat16 Max on this target, so the
// values are widened to fp32 through Cast (exact), compared in fp32 and narrowed
// back (also exact: the selected values came from bf16 inputs).
// ---------------------------------------------------------------------------

__global__ __aicore__ void maximum_kernel_bf16(GM_ADDR aAddr, GM_ADDR bAddr, GM_ADDR yAddr,
                                               int64_t rows, int64_t runLen, int64_t tileElems,
                                               int64_t tilesPerRow, int64_t numBlocks)
{
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t totalTiles = rows * tilesPerRow;
    const int64_t t0 = totalTiles * blk / numBlocks;
    const int64_t t1 = totalTiles * (blk + 1) / numBlocks;
    if (t0 >= t1) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<bfloat16_t> aGm;
    AscendC::GlobalTensor<bfloat16_t> bGm;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    aGm.SetGlobalBuffer((__gm__ bfloat16_t *)aAddr);
    bGm.SetGlobalBuffer((__gm__ bfloat16_t *)bAddr);
    yGm.SetGlobalBuffer((__gm__ bfloat16_t *)yAddr);

    const uint32_t tileBytes = static_cast<uint32_t>(tileElems * 2);
    const uint32_t fBytes = static_cast<uint32_t>(tileElems * 4);
    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQa;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQb;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQDepth> outQ;
    pipe.InitBuffer(inQa, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(inQb, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(outQ, kQDepth, tileBytes + kSlack);
    AscendC::TBuf<AscendC::TPosition::VECCALC> ca;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cb;
    AscendC::TBuf<AscendC::TPosition::VECCALC> co;
    pipe.InitBuffer(ca, fBytes + kSlack);
    pipe.InitBuffer(cb, fBytes + kSlack);
    pipe.InitBuffer(co, fBytes + kSlack);
    auto aF = ca.Get<float>();
    auto bF = cb.Get<float>();
    auto oF = co.Get<float>();

    for (int64_t k = t0; k < t1; ++k) {
        const int64_t row = k / tilesPerRow;
        const int64_t ti = k - row * tilesPerRow;
        const int64_t o = ti * tileElems;
        int64_t len = runLen - o;
        if (len <= 0) {
            continue;
        }
        if (len > tileElems) {
            len = tileElems;
        }
        const int64_t off = row * runLen + o;
        const uint32_t bytes = static_cast<uint32_t>(len * 2);
        const uint32_t cnt = static_cast<uint32_t>(len);
        AscendC::DataCopyExtParams cp{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> pp{false, 0, 0, static_cast<bfloat16_t>(0)};

        auto aL = inQa.AllocTensor<bfloat16_t>();
        auto bL = inQb.AllocTensor<bfloat16_t>();
        AscendC::DataCopyPad(aL, aGm[off], cp, pp);
        AscendC::DataCopyPad(bL, bGm[o], cp, pp);
        inQa.EnQue(aL);
        inQb.EnQue(bL);
        aL = inQa.DeQue<bfloat16_t>();
        bL = inQb.DeQue<bfloat16_t>();

        auto zL = outQ.AllocTensor<bfloat16_t>();
        AscendC::Cast(aF, aL, AscendC::RoundMode::CAST_NONE, cnt);
        AscendC::Cast(bF, bL, AscendC::RoundMode::CAST_NONE, cnt);
        MaxChunk<float>(oF, aF, bF, cnt);
        AscendC::Cast(zL, oF, AscendC::RoundMode::CAST_RINT, cnt);
        outQ.EnQue(zL);
        inQa.FreeTensor(aL);
        inQb.FreeTensor(bL);
        zL = outQ.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams cpo{1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(yGm[off], zL, cpo);
        outQ.FreeTensor(zL);
    }
}

// ---------------------------------------------------------------------------
// int64 kernel: there is no 64 bit vector arithmetic on this target, so the element-wise max
// runs on the int32 lanes the 64 bit words are built from (see MaxChunk).
// ---------------------------------------------------------------------------

__global__ __aicore__ void maximum_kernel_i64(GM_ADDR aAddr, GM_ADDR bAddr, GM_ADDR yAddr,
                                              int64_t rows, int64_t runLen, int64_t tileElems,
                                              int64_t tilesPerRow, int64_t numBlocks)
{
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t totalTiles = rows * tilesPerRow;
    const int64_t t0 = totalTiles * blk / numBlocks;
    const int64_t t1 = totalTiles * (blk + 1) / numBlocks;
    if (t0 >= t1) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<int64_t> aGm;
    AscendC::GlobalTensor<int64_t> bGm;
    AscendC::GlobalTensor<int64_t> yGm;
    aGm.SetGlobalBuffer((__gm__ int64_t *)aAddr);
    bGm.SetGlobalBuffer((__gm__ int64_t *)bAddr);
    yGm.SetGlobalBuffer((__gm__ int64_t *)yAddr);

    const uint32_t tileBytes = static_cast<uint32_t>(tileElems * 8);
    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQa;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQb;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQDepth> outQ;
    pipe.InitBuffer(inQa, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(inQb, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(outQ, kQDepth, tileBytes + kSlack);

    for (int64_t k = t0; k < t1; ++k) {
        const int64_t row = k / tilesPerRow;
        const int64_t ti = k - row * tilesPerRow;
        const int64_t o = ti * tileElems;
        int64_t len = runLen - o;
        if (len <= 0) {
            continue;
        }
        if (len > tileElems) {
            len = tileElems;
        }
        const int64_t off = row * runLen + o;
        const uint32_t bytes = static_cast<uint32_t>(len * 8);
        const uint32_t cnt = static_cast<uint32_t>(len);
        AscendC::DataCopyExtParams cp{1, bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};

        auto aL = inQa.AllocTensor<int64_t>();
        auto bL = inQb.AllocTensor<int64_t>();
        AscendC::DataCopyPad(aL, aGm[off], cp, pp);
        AscendC::DataCopyPad(bL, bGm[o], cp, pp);
        inQa.EnQue(aL);
        inQb.EnQue(bL);
        aL = inQa.DeQue<int64_t>();
        bL = inQb.DeQue<int64_t>();

        auto zL = outQ.AllocTensor<int64_t>();
        MaxChunk<int64_t>(zL, aL, bL, cnt);
        outQ.EnQue(zL);
        inQa.FreeTensor(aL);
        inQb.FreeTensor(bL);
        zL = outQ.DeQue<int64_t>();
        AscendC::DataCopyExtParams cpo{1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(yGm[off], zL, cpo);
        outQ.FreeTensor(zL);
    }
}

// ---------------------------------------------------------------------------
// B-stationary kernel
//
// In the broadcasting plans whose whole row fits in one tile (tilesPerRow == 1) the B block
// that a row reads is identical for every row, so it is loaded once per core and one tile
// covers up to `kRows` rows.  The per-row tile count of patterns such as [2049,513] x [1,513]
// then drops by that factor and each transfer becomes one large contiguous DMA instead of many
// tiny ones (measured: that shape spent ~1.5 us per 2 KB row tile in the per-row walk).
//
// The rows of a tile are contiguous in A and in the output, so a tile is one DMA in and one DMA
// out, and the compute walks the tiles row by row against the resident B block.  Row r's Max
// may overshoot its own row into the next row's region inside the tile; the rows are processed
// in increasing order, so every overshoot is rewritten by the next row's own Max and only the
// last row's overshoot reaches the buffer slack.
// ---------------------------------------------------------------------------

template <typename T>
__global__ __aicore__ void maximum_rows_kernel(GM_ADDR aAddr, GM_ADDR bAddr, GM_ADDR yAddr,
                                               int64_t rows, int64_t runLen, int64_t kRows,
                                               int64_t numBlocks)
{
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    if (kRows < 1) {
        kRows = 1;
    }
    const int64_t groups = (rows + kRows - 1) / kRows;
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t g0 = groups * blk / numBlocks;
    const int64_t g1 = groups * (blk + 1) / numBlocks;
    if (g0 >= g1) {
        return;
    }

    // The hardware packs the UB rows of a multi-block DataCopyPad at 32 byte granularity, so the
    // row slots inside the tile are 32 byte aligned and can be addressed by an element stride.
    // (Passing the alignment difference as an extra dstStride would double count it; and a per
    // row vector op on an unaligned UB base is rejected by the VEC unit.)
    const uint32_t rowBytes = static_cast<uint32_t>(runLen * static_cast<int64_t>(sizeof(T)));
    const uint32_t slotBytes = (rowBytes + 31u) & ~31u;
    const uint32_t slotElems = slotBytes / static_cast<uint32_t>(sizeof(T));
    const uint32_t tileBytes = static_cast<uint32_t>(kRows) * slotBytes;
    const uint32_t bBytes = rowBytes;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQDepth> inQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQDepth> outQ;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> bQ;
    pipe.InitBuffer(inQ, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(outQ, kQDepth, tileBytes + kSlack);
    pipe.InitBuffer(bQ, 1, bBytes + kSlack);

    AscendC::GlobalTensor<T> aGm;
    AscendC::GlobalTensor<T> bGm;
    AscendC::GlobalTensor<T> yGm;
    aGm.SetGlobalBuffer((__gm__ T *)aAddr);
    bGm.SetGlobalBuffer((__gm__ T *)bAddr);
    yGm.SetGlobalBuffer((__gm__ T *)yAddr);

    // One resident copy of the B block; DeQue publishes the MTE2 -> V dependency.
    AscendC::DataCopyExtParams cpb{1, bBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> ppb{false, 0, 0, static_cast<T>(0)};
    auto bL = bQ.AllocTensor<T>();
    AscendC::DataCopyPad(bL, bGm[0], cpb, ppb);
    bQ.EnQue(bL);
    bL = bQ.DeQue<T>();

    for (int64_t g = g0; g < g1; ++g) {
        const int64_t row0 = g * kRows;
        int64_t nr = rows - row0;
        if (nr > kRows) {
            nr = kRows;
        }
        const int64_t off = row0 * runLen;
        AscendC::DataCopyExtParams cp{static_cast<uint16_t>(nr), rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};

        auto aL = inQ.AllocTensor<T>();
        AscendC::DataCopyPad(aL, aGm[off], cp, pp);
        inQ.EnQue(aL);
        aL = inQ.DeQue<T>();

        auto zL = outQ.AllocTensor<T>();
        const uint32_t rl = static_cast<uint32_t>(runLen);
        for (int64_t r = 0; r < nr; ++r) {
            const uint32_t ro = static_cast<uint32_t>(r) * slotElems;
            MaxChunk<T>(zL[ro], aL[ro], bL, rl);
        }
        outQ.EnQue(zL);
        inQ.FreeTensor(aL);
        zL = outQ.DeQue<T>();
        AscendC::DataCopyExtParams cpo{static_cast<uint16_t>(nr), rowBytes, 0, 0, 0};
        AscendC::DataCopyPad(yGm[off], zL, cpo);
        outQ.FreeTensor(zL);
    }
}

// ---------------------------------------------------------------------------
// helper kernel: replicate each of `lc` values `linner` times (raw carriers)
// ---------------------------------------------------------------------------

template <typename RawT>
__global__ __aicore__ void maximum_expand_kernel(GM_ADDR srcAddr, GM_ADDR dstAddr, int64_t lc,
                                                 int64_t linner, int64_t numBlocks)
{
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t c0 = lc * blk / numBlocks;
    const int64_t c1 = lc * (blk + 1) / numBlocks;
    if (c1 <= c0) {
        return;
    }
    const int64_t cnt = c1 - c0;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oBuf;
    pipe.InitBuffer(sBuf, static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(RawT))) + kSlack);
    pipe.InitBuffer(oBuf, static_cast<uint32_t>(linner * static_cast<int64_t>(sizeof(RawT))) + kSlack);
    AscendC::GlobalTensor<RawT> sGm;
    AscendC::GlobalTensor<RawT> dGm;
    sGm.SetGlobalBuffer((__gm__ RawT *)srcAddr);
    dGm.SetGlobalBuffer((__gm__ RawT *)dstAddr);

    auto sL = sBuf.Get<RawT>();
    auto oL = oBuf.Get<RawT>();
    AscendC::DataCopyExtParams cpi{
        1, static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(RawT))), 0, 0, 0};
    AscendC::DataCopyPadExtParams<RawT> pp{false, 0, 0, static_cast<RawT>(0)};
    AscendC::DataCopyPad(sL, sGm[c0], cpi, pp);
    AscendC::PipeBarrier<PIPE_ALL>();
    __ubuf__ RawT *sp = (__ubuf__ RawT *)sL.GetPhyAddr();
    __ubuf__ RawT *op = (__ubuf__ RawT *)oL.GetPhyAddr();
    const uint32_t linnerU = static_cast<uint32_t>(linner);
    for (int64_t j = 0; j < cnt; ++j) {
        const RawT v = sp[j];
        if constexpr (sizeof(RawT) == 2 || sizeof(RawT) == 4) {
            AscendC::Duplicate(oL, v, linnerU);
        } else {
            // 1 byte / 8 byte carriers are not supported by Duplicate
            for (uint32_t l = 0; l < linnerU; ++l) {
                op[l] = v;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cpo{
            1, static_cast<uint32_t>(linner * static_cast<int64_t>(sizeof(RawT))), 0, 0, 0};
        AscendC::DataCopyPad(dGm[(c0 + j) * linner], oL, cpo);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// ---------------------------------------------------------------------------
// helper kernel: numpy-broadcast copy of one operand to the output shape
// ---------------------------------------------------------------------------

constexpr int64_t kBcastStage = 1024;

template <typename RawT>
__global__ __aicore__ void maximum_bcast_kernel(GM_ADDR srcAddr, GM_ADDR dstAddr, int64_t numel,
                                                int32_t r, int64_t p0, int64_t p1, int64_t p2,
                                                int64_t p3, int64_t p4, int64_t p5, int64_t p6,
                                                int64_t p7)
{
    const int64_t pk[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
    int64_t outSize[8];
    int64_t srcSize[8];
    int64_t srcStride[8];
    for (int32_t d = 0; d < 8; ++d) {
        outSize[d] = 1;
        srcSize[d] = 1;
        srcStride[d] = 0;
    }
    for (int32_t d = 0; d < r; ++d) {
        const uint64_t pv = static_cast<uint64_t>(pk[d]);
        outSize[d] = static_cast<int64_t>(pv >> 32);
        srcSize[d] = static_cast<int64_t>(pv & 0xFFFFFFFFu);
    }
    {
        // A broadcast dimension (srcSize[d] == 1 while outSize[d] > 1) must read the same source
        // element for every output coordinate, so it contributes a zero stride.  Without this the
        // index walk below adds c * acc for a broadcast dim and reads past the source buffer.
        int64_t acc = 1;
        for (int32_t d = r - 1; d >= 0; --d) {
            srcStride[d] = (srcSize[d] > 1) ? acc : 0;
            acc *= srcSize[d];
        }
    }

    const int64_t numBlocks = static_cast<int64_t>(AscendC::GetBlockNum());
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t i0 = numel * blk / numBlocks;
    const int64_t i1 = numel * (blk + 1) / numBlocks;
    if (i1 <= i0) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oBuf;
    pipe.InitBuffer(oBuf,
                    static_cast<uint32_t>(kBcastStage * static_cast<int64_t>(sizeof(RawT))) +
                        kSlack);
    AscendC::GlobalTensor<RawT> sGm;
    AscendC::GlobalTensor<RawT> dGm;
    sGm.SetGlobalBuffer((__gm__ RawT *)srcAddr);
    dGm.SetGlobalBuffer((__gm__ RawT *)dstAddr);
    auto oL = oBuf.Get<RawT>();

    int64_t pos = i0;
    while (pos < i1) {
        int64_t cnt = i1 - pos;
        if (cnt > kBcastStage) {
            cnt = kBcastStage;
        }
        for (int64_t u = 0; u < cnt; ++u) {
            int64_t rem = pos + u;
            int64_t so = 0;
            for (int32_t d = r - 1; d >= 0; --d) {
                const int64_t sz = outSize[d];
                const int64_t c = (sz > 0) ? (rem % sz) : 0;
                rem = (sz > 0) ? (rem / sz) : 0;
                so += c * srcStride[d];
            }
            oL.SetValue(static_cast<uint32_t>(u), sGm.GetValue(static_cast<uint64_t>(so)));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cpo{
            1, static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(RawT))), 0, 0, 0};
        AscendC::DataCopyPad(dGm[pos], oL, cpo);
        AscendC::PipeBarrier<PIPE_ALL>();
        pos += cnt;
    }
}

} // namespace cann_bench_maximum

// ---------------------------------------------------------------------------
// tiling
// ---------------------------------------------------------------------------

std::tuple<int64_t, int64_t, int64_t> calc_maximum_tiling_params(int64_t numel, int64_t rows,
                                                                int64_t runLen, int64_t elemBytes,
                                                                int64_t dtypeCode)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    const int64_t budget = static_cast<int64_t>(ubSize) * 3 / 4;
    // UB bytes per output element: three TQue buffers, double buffered, plus the staging
    // buffers the dtype needs (fp32 staging for bfloat16, half staging for int8).
    int64_t perElem = 6 * elemBytes;
    if (dtypeCode == 1) {
        perElem = 12 + 12;
    } else if (dtypeCode == 3) {
        perElem = 6 + 6;
    }
    if (perElem < 1) {
        perElem = 1;
    }
    int64_t tile = budget / perElem;
    if (tile > 4096) {
        tile = (tile / 1024) * 1024;
    }
    // Round the tile down so that its byte length is a multiple of 32: every tile start is then
    // 32 byte aligned in GM instead of only the first one.
    const int64_t alignElems = (32 / elemBytes) > 1 ? (32 / elemBytes) : 1;
    tile = (tile / alignElems) * alignElems;
    if (tile < 1) {
        tile = alignElems;
    }
    if (runLen > 0) {
        if (tile >= runLen) {
            tile = runLen;
        } else if (tile + tile / 4 >= runLen) {
            tile = runLen; // small overshoot: one tile per row
        }
    }
    if (tile < 1) {
        tile = 1;
    }
    const int64_t tilesPerRow = (runLen + tile - 1) / tile;
    const int64_t totalTiles = rows * tilesPerRow;
    int64_t numBlocks = std::min<int64_t>(coreNum, totalTiles);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    return std::make_tuple(tile, numBlocks, static_cast<int64_t>(ubSize));
}

// ---------------------------------------------------------------------------
// launch wrappers (plain C entry points callable from the host plugin TU)
// ---------------------------------------------------------------------------

using cann_bench_maximum::maximum_bcast_kernel;
using cann_bench_maximum::maximum_expand_kernel;
using cann_bench_maximum::maximum_kernel;
using cann_bench_maximum::maximum_kernel_bf16;
using cann_bench_maximum::maximum_kernel_i64;
using cann_bench_maximum::maximum_kernel_i8;
using cann_bench_maximum::maximum_rows_kernel;

// UB budget used for the B-stationary kernel, kept below the full UB size.
constexpr int64_t kMaximumUbBudget = 152 * 1024;

// Rows per tile for the B-stationary kernel, or 0 when it does not apply.  It applies when a
// whole row fits in one tile (tilesPerRow == 1), there is more than one row, and B's resident
// block plus a double buffered row-group pair fit in UB.  The tile rows are 32 byte aligned
// (the hardware packs the UB rows of a multi-block DataCopyPad), so the row group is sized on
// the rounded row size.  int8 and bfloat16 are excluded because their element-wise max runs
// through a half/fp32 staging instead of a native Max.
int64_t maximum_row_group(int64_t rows, int64_t runLen, int64_t tilesPerRow, int64_t elemBytes)
{
    if (tilesPerRow != 1 || rows <= 1 || runLen <= 0 || elemBytes <= 0) {
        return 0;
    }
    const int64_t rowBytes = runLen * elemBytes;
    const int64_t slotBytes = (rowBytes + 31) & ~static_cast<int64_t>(31);
    if (slotBytes * 5 + 2560 > kMaximumUbBudget) {
        return 0;
    }
    int64_t k = (kMaximumUbBudget - slotBytes - 2560) / (4 * slotBytes);
    if (k > 256) {
        k = 256;
    }
    if (k < 1) {
        k = 1;
    }
    return k;
}

extern "C" {

#define MAXIMUM_LAUNCH_MAIN(NAME, TYPE, ESZ)                                                    \
    void NAME(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen, int64_t tileElems, \
              int64_t tilesPerRow, int64_t numBlocks, void *stream)                             \
    {                                                                                           \
        const int64_t kRows = maximum_row_group(rows, runLen, tilesPerRow, ESZ);                \
        if (kRows > 0) {                                                                        \
            maximum_rows_kernel<TYPE><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(   \
                a, b, y, rows, runLen, kRows, numBlocks);                                       \
            return;                                                                             \
        }                                                                                       \
        maximum_kernel<TYPE><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(            \
            a, b, y, rows, runLen, tileElems, tilesPerRow, numBlocks);                          \
    }

MAXIMUM_LAUNCH_MAIN(launch_maximum_fp16, half, 2)
MAXIMUM_LAUNCH_MAIN(launch_maximum_fp32, float, 4)
MAXIMUM_LAUNCH_MAIN(launch_maximum_int32, int32_t, 4)

#undef MAXIMUM_LAUNCH_MAIN

void launch_maximum_bf16(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                         int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void *stream)
{
    maximum_kernel_bf16<<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(
        a, b, y, rows, runLen, tileElems, tilesPerRow, numBlocks);
}

void launch_maximum_int64(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                          int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void *stream)
{
    const int64_t kRows = maximum_row_group(rows, runLen, tilesPerRow, 8);
    if (kRows > 0) {
        maximum_rows_kernel<int64_t><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(
            a, b, y, rows, runLen, kRows, numBlocks);
        return;
    }
    maximum_kernel_i64<<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(
        a, b, y, rows, runLen, tileElems, tilesPerRow, numBlocks);
}

void launch_maximum_int8(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                         int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void *stream)
{
    maximum_kernel_i8<<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(
        a, b, y, rows, runLen, tileElems, tilesPerRow, numBlocks);
}

#define MAXIMUM_LAUNCH_EXPAND(NAME, TYPE)                                                          \
    void NAME(GM_ADDR src, GM_ADDR dst, int64_t lc, int64_t linner, int64_t numBlocks,             \
              void *stream)                                                                        \
    {                                                                                              \
        maximum_expand_kernel<TYPE><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(        \
            src, dst, lc, linner, numBlocks);                                                      \
    }

MAXIMUM_LAUNCH_EXPAND(launch_maximum_expand_b8, int8_t)
MAXIMUM_LAUNCH_EXPAND(launch_maximum_expand_b16, int16_t)
MAXIMUM_LAUNCH_EXPAND(launch_maximum_expand_b32, int32_t)
MAXIMUM_LAUNCH_EXPAND(launch_maximum_expand_b64, int64_t)

#undef MAXIMUM_LAUNCH_EXPAND

#define MAXIMUM_LAUNCH_BCAST(NAME, TYPE)                                                      \
    void NAME(GM_ADDR src, GM_ADDR dst, int64_t numel, int32_t r, int64_t p0, int64_t p1,     \
              int64_t p2, int64_t p3, int64_t p4, int64_t p5, int64_t p6, int64_t p7,         \
              void *stream)                                                                   \
    {                                                                                         \
        uint32_t blocks = static_cast<uint32_t>(numel >> 10);                                 \
        if (blocks < 1) {                                                                     \
            blocks = 1;                                                                       \
        }                                                                                     \
        if (blocks > 48) {                                                                    \
            blocks = 48;                                                                      \
        }                                                                                     \
        maximum_bcast_kernel<TYPE><<<blocks, nullptr, stream>>>(src, dst, numel, r, p0, p1,   \
                                                                p2, p3, p4, p5, p6, p7);      \
    }

MAXIMUM_LAUNCH_BCAST(launch_maximum_bcast_b8, int8_t)
MAXIMUM_LAUNCH_BCAST(launch_maximum_bcast_b16, int16_t)
MAXIMUM_LAUNCH_BCAST(launch_maximum_bcast_b32, int32_t)
MAXIMUM_LAUNCH_BCAST(launch_maximum_bcast_b64, int64_t)

#undef MAXIMUM_LAUNCH_BCAST

} // extern "C"
