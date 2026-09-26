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
 * \file unique_kernel.cpp
 * \brief Unique (y = sorted distinct values of x, optional inverse) - kernel + tiling + launch.
 *        Compiled with bisheng (-xasc), target Ascend 910B / dav-2201 (generic SIMD / MemBase route).
 *
 * Route (bounded key space): every supported dtype is mapped to a *monotone* unsigned key ("bin")
 *   uint8  : bin = v
 *   int8   : bin = v + 128
 *   fp16/bf16 : bin = monotone16(bits)  (order preserving, -0.0 -> 0x7FFF, +0.0 -> 0x8000)
 *   int32  : bin = v - min(v)
 *   int64  : bin = v - min(v)
 * Deduplication and rank computation are then done with a *presence bitmap* over [0, BIN):
 *   - every block sets the bits of its own chunk in a private UB bitmap,
 *   - the per block bitmaps are OR-merged into one global bitmap in GM,
 *   - rank(b) = number of set bits strictly below b  (prefix popcount of the bitmap).
 * y[rank(b)] = value(b) for every set b, and inverse[i] = rank(bin(x[i])).  This needs no sorting
 * and no per element scatter, which keeps the wide uint8 (268M element) case cheap.
 *
 * +/-0.0 handling follows torch.unique: the two bit patterns share one unique value only when both
 * are actually present; when only one sign occurs it is preserved.  The merge is applied as
 * "bin 0x7FFF (=-0.0) is treated as absent whenever 0x8000 (=+0.0) is also present".
 *
 * All host/device data movement is DMA based (DataCopyPad) except for the fp32 route: its presence
 * bitmap spans the whole 32 bit key space, so it cannot live in UB.  There every block owns one
 * disjoint key shard in GM and updates only its own words through scalar GetValue/SetValue (the
 * compiler's automatic DCCI makes those writes visible to the DMA reads of the later kernels); the
 * uint8/int8/fp16/bf16/int32/int64 routes touch UB only and raise no Scalar<->DataCache question.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "unique_launch.h"

namespace unique_impl {

constexpr int32_t MAX_BMP_WORDS = 8192;      // up to 262144 bins
constexpr int32_t TILE_ELEMS = 4096;         // elements per DMA tile
constexpr int32_t EMIT_CHUNK = 4096;         // elements per y output flush
constexpr int64_t MIN_ELEMS_PER_CORE = 4096;

__aicore__ inline uint32_t Popc32(uint32_t v)
{
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0F0F0F0Fu;
    return (v * 0x01010101u) >> 24;
}

// Monotone unsigned key of one element.
template <typename T>
__aicore__ inline uint32_t BinOf(T v, int64_t minVal)
{
    if constexpr (std::is_same<T, uint8_t>::value) {
        return static_cast<uint32_t>(v);
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return static_cast<uint32_t>(static_cast<int32_t>(v) + 128);
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        uint32_t u = static_cast<uint32_t>(v);
        return ((u & 0x8000u) != 0u) ? ((~u) & 0xFFFFu) : (u | 0x8000u);
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return static_cast<uint32_t>(static_cast<int64_t>(v) - minVal);
    } else {
        return static_cast<uint32_t>(v - minVal);
    }
}

// Inverse of BinOf.
template <typename T>
__aicore__ inline T UnmapOf(uint32_t b, int64_t minVal)
{
    if constexpr (std::is_same<T, uint8_t>::value) {
        return static_cast<uint8_t>(b);
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return static_cast<int8_t>(static_cast<int32_t>(b) - 128);
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        return static_cast<uint16_t>(((b & 0x8000u) != 0u) ? (b & 0x7FFFu) : ((~b) & 0xFFFFu));
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return static_cast<int32_t>(static_cast<int64_t>(b) + minVal);
    } else {
        return static_cast<int64_t>(static_cast<int64_t>(b) + minVal);
    }
}

// Key space size.
template <typename T>
__aicore__ inline int64_t BinCountOf(int64_t minVal, int64_t maxVal)
{
    if constexpr (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) {
        return 256;
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        return 65536;
    } else {
        int64_t n = maxVal - minVal + 1;
        if (n < 1) n = 1;
        if (n > (int64_t)MAX_BMP_WORDS * 32) n = (int64_t)MAX_BMP_WORDS * 32;
        return n;
    }
}

// -0.0 / +0.0 bins (uint16 float reinterpreting dtypes only).
template <typename T>
__aicore__ inline void ZeroMergeBins(int32_t &neg0, int32_t &pos0)
{
    if constexpr (std::is_same<T, uint16_t>::value) {
        neg0 = 0x7FFF;
        pos0 = 0x8000;
    } else {
        neg0 = -1;
        pos0 = -1;
    }
}

}  // namespace unique_impl

using namespace unique_impl;

// ---------------------------------------------------------------------------
// min / max of x (int32 / int64 only, the only dtypes whose bin origin is data dependent)
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void unique_minmax_partial(GM_ADDR x, GM_ADDR out, int64_t numel, int64_t blockLen,
                                                 uint32_t tileElems)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int64_t> outGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    outGm.SetGlobalBuffer((__gm__ int64_t *)out);

    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resBuf;
    pipe.InitBuffer(inBuf, tileElems * sizeof(T));
    pipe.InitBuffer(resBuf, 64);

    const int64_t start = blockLen * static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t end = start + blockLen;
    if (end > numel) end = numel;

    int64_t mn = 9223372036854775807LL;
    int64_t mx = (-9223372036854775807LL - 1);

    auto inL = inBuf.Get<T>();
    for (int64_t off = start; off < end; off += static_cast<int64_t>(tileElems)) {
        int64_t n = end - off;
        if (n > static_cast<int64_t>(tileElems)) n = static_cast<int64_t>(tileElems);
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, xGm[off], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ T *xp = (__ubuf__ T *)inL.GetPhyAddr();
        for (int64_t i = 0; i < n; ++i) {
            int64_t v = static_cast<int64_t>(xp[i]);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    auto resL = resBuf.Get<int64_t>();
    __ubuf__ int64_t *rp = (__ubuf__ int64_t *)resL.GetPhyAddr();
    rp[0] = mn;
    rp[1] = mx;
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t c = static_cast<int64_t>(AscendC::GetBlockIdx());
    AscendC::DataCopyExtParams cp2{1, 16, 0, 0, 0};
    AscendC::DataCopyPad(outGm[c * 2], resL, cp2);
}

__global__ __aicore__ void unique_minmax_final(GM_ADDR in, GM_ADDR out, int32_t nC)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<int64_t> inGm;
    AscendC::GlobalTensor<int64_t> outGm;
    inGm.SetGlobalBuffer((__gm__ int64_t *)in);
    outGm.SetGlobalBuffer((__gm__ int64_t *)out);

    AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 8192);
    auto l = buf.Get<int64_t>();

    AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(nC * 16), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
    AscendC::DataCopyPad(l, inGm, cp, pp);
    AscendC::PipeBarrier<PIPE_ALL>();

    __ubuf__ int64_t *p = (__ubuf__ int64_t *)l.GetPhyAddr();
    int64_t mn = 9223372036854775807LL;
    int64_t mx = (-9223372036854775807LL - 1);
    for (int32_t i = 0; i < nC; ++i) {
        if (p[2 * i] < mn) mn = p[2 * i];
        if (p[2 * i + 1] > mx) mx = p[2 * i + 1];
    }
    p[0] = mn;
    p[1] = mx;
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cp2{1, 16, 0, 0, 0};
    AscendC::DataCopyPad(outGm, l, cp2);
}

// ---------------------------------------------------------------------------
// presence bitmap (one private region per block)
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void unique_bitmap(GM_ADDR x, GM_ADDR bmpOut, GM_ADDR minmax, int64_t numel,
                                         int64_t blockLen, uint32_t tileElems, uint32_t bmpWords)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    AscendC::GlobalTensor<int64_t> mmGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmpOut);
    mmGm.SetGlobalBuffer((__gm__ int64_t *)minmax);

    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmBuf;
    pipe.InitBuffer(inBuf, tileElems * sizeof(T));
    pipe.InitBuffer(bmpBuf, (uint32_t)MAX_BMP_WORDS * 4u);
    pipe.InitBuffer(mmBuf, 64);

    auto mmL = mmBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(mmL, mmGm, cp, pp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t minVal = ((__ubuf__ int64_t *)mmL.GetPhyAddr())[0];

    auto bmpL = bmpBuf.Get<uint32_t>();
    __ubuf__ uint32_t *bp = (__ubuf__ uint32_t *)bmpL.GetPhyAddr();
    for (uint32_t w = 0; w < bmpWords; ++w) {
        bp[w] = 0u;
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    const int64_t start = blockLen * static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t end = start + blockLen;
    if (end > numel) end = numel;

    auto inL = inBuf.Get<T>();
    for (int64_t off = start; off < end; off += static_cast<int64_t>(tileElems)) {
        int64_t n = end - off;
        if (n > static_cast<int64_t>(tileElems)) n = static_cast<int64_t>(tileElems);
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, xGm[off], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ T *xp = (__ubuf__ T *)inL.GetPhyAddr();
        for (int64_t i = 0; i < n; ++i) {
            uint32_t b = BinOf<T>(xp[i], minVal);
            bp[b >> 5] |= (1u << (b & 31u));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    const int64_t c = static_cast<int64_t>(AscendC::GetBlockIdx());
    AscendC::DataCopyExtParams cp2{1, bmpWords * 4u, 0, 0, 0};
    AscendC::DataCopyPad(bmpGm[c * static_cast<int64_t>(bmpWords)], bmpL, cp2);
}

// ---------------------------------------------------------------------------
// OR merge of the per block bitmaps into one global bitmap
// ---------------------------------------------------------------------------
__global__ __aicore__ void unique_merge(GM_ADDR in, GM_ADDR out, int32_t nC, uint32_t bmpWords)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<uint32_t> inGm;
    AscendC::GlobalTensor<uint32_t> outGm;
    inGm.SetGlobalBuffer((__gm__ uint32_t *)in);
    outGm.SetGlobalBuffer((__gm__ uint32_t *)out);

    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    pipe.InitBuffer(accBuf, (uint32_t)MAX_BMP_WORDS * 4u);
    pipe.InitBuffer(tmpBuf, (uint32_t)MAX_BMP_WORDS * 4u);

    auto accL = accBuf.Get<uint32_t>();
    auto tmpL = tmpBuf.Get<uint32_t>();
    __ubuf__ uint32_t *ap = (__ubuf__ uint32_t *)accL.GetPhyAddr();

    for (uint32_t w = 0; w < bmpWords; ++w) {
        ap[w] = 0u;
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    for (int32_t c = 0; c < nC; ++c) {
        AscendC::DataCopyExtParams cp{1, bmpWords * 4u, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(tmpL, inGm[static_cast<int64_t>(c) * static_cast<int64_t>(bmpWords)], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ uint32_t *tp = (__ubuf__ uint32_t *)tmpL.GetPhyAddr();
        for (uint32_t w = 0; w < bmpWords; ++w) {
            ap[w] |= tp[w];
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::DataCopyExtParams cp2{1, bmpWords * 4u, 0, 0, 0};
    AscendC::DataCopyPad(outGm, accL, cp2);
}

// ---------------------------------------------------------------------------
// number of distinct values (+ whether -0.0 had to be merged into +0.0)
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void unique_count(GM_ADDR bmp, GM_ADDR wsK, uint32_t bmpWords)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    AscendC::GlobalTensor<int64_t> kGm;
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmp);
    kGm.SetGlobalBuffer((__gm__ int64_t *)wsK);

    AscendC::TBuf<AscendC::TPosition::VECCALC> bmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    pipe.InitBuffer(bmpBuf, (uint32_t)MAX_BMP_WORDS * 4u);
    pipe.InitBuffer(outBuf, 64);

    auto bmpL = bmpBuf.Get<uint32_t>();
    AscendC::DataCopyExtParams cp{1, bmpWords * 4u, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
    AscendC::DataCopyPad(bmpL, bmpGm, cp, pp);
    AscendC::PipeBarrier<PIPE_ALL>();
    __ubuf__ uint32_t *bp = (__ubuf__ uint32_t *)bmpL.GetPhyAddr();

    int32_t neg0, pos0;
    ZeroMergeBins<T>(neg0, pos0);
    int64_t merged = 0;
    if (neg0 >= 0) {
        uint32_t nb = (bp[(uint32_t)neg0 >> 5] >> ((uint32_t)neg0 & 31u)) & 1u;
        uint32_t pb = (bp[(uint32_t)pos0 >> 5] >> ((uint32_t)pos0 & 31u)) & 1u;
        if (nb != 0u && pb != 0u) merged = 1;
    }

    int64_t k = 0;
    for (uint32_t w = 0; w < bmpWords; ++w) {
        k += (int64_t)Popc32(bp[w]);
    }
    k -= merged;

    auto outL = outBuf.Get<int64_t>();
    __ubuf__ int64_t *op = (__ubuf__ int64_t *)outL.GetPhyAddr();
    op[0] = k;
    op[1] = merged;
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cp2{1, 16, 0, 0, 0};
    AscendC::DataCopyPad(kGm, outL, cp2);
}

// ---------------------------------------------------------------------------
// inverse index: inverse[i] = rank(bin(x[i]))
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void unique_inverse(GM_ADDR x, GM_ADDR inverse, GM_ADDR bmp, GM_ADDR wsK,
                                          GM_ADDR minmax, int64_t numel, int64_t blockLen,
                                          uint32_t tileElems, uint32_t bmpWords)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int64_t> invGm;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    AscendC::GlobalTensor<int64_t> kGm;
    AscendC::GlobalTensor<int64_t> mmGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    invGm.SetGlobalBuffer((__gm__ int64_t *)inverse);
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmp);
    kGm.SetGlobalBuffer((__gm__ int64_t *)wsK);
    mmGm.SetGlobalBuffer((__gm__ int64_t *)minmax);

    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pwBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kBug;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmBuf;
    pipe.InitBuffer(inBuf, tileElems * sizeof(T));
    pipe.InitBuffer(outBuf, tileElems * sizeof(int64_t));
    pipe.InitBuffer(bmpBuf, (uint32_t)MAX_BMP_WORDS * 4u);
    pipe.InitBuffer(pwBuf, (uint32_t)MAX_BMP_WORDS * 4u);
    pipe.InitBuffer(kBug, 64);
    pipe.InitBuffer(mmBuf, 64);

    auto bmpL = bmpBuf.Get<uint32_t>();
    {
        AscendC::DataCopyExtParams cp{1, bmpWords * 4u, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(bmpL, bmpGm, cp, pp);
    }
    auto kL = kBug.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(kL, kGm, cp, pp);
    }
    auto mmL = mmBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(mmL, mmGm, cp, pp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    __ubuf__ uint32_t *bp = (__ubuf__ uint32_t *)bmpL.GetPhyAddr();
    const int64_t merged = ((__ubuf__ int64_t *)kL.GetPhyAddr())[1];
    const int64_t minVal = ((__ubuf__ int64_t *)mmL.GetPhyAddr())[0];

    int32_t neg0, pos0;
    ZeroMergeBins<T>(neg0, pos0);

    // prefix popcount of the bitmap
    auto pwL = pwBuf.Get<uint32_t>();
    __ubuf__ uint32_t *pw = (__ubuf__ uint32_t *)pwL.GetPhyAddr();
    uint32_t run = 0;
    for (uint32_t w = 0; w < bmpWords; ++w) {
        pw[w] = run;
        uint32_t word = bp[w];
        if (merged != 0 && neg0 >= 0 && ((uint32_t)neg0 >> 5) == w) {
            word &= ~(1u << ((uint32_t)neg0 & 31u));
        }
        run += Popc32(word);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    const int64_t start = blockLen * static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t end = start + blockLen;
    if (end > numel) end = numel;

    auto inL = inBuf.Get<T>();
    auto outL = outBuf.Get<int64_t>();
    for (int64_t off = start; off < end; off += static_cast<int64_t>(tileElems)) {
        int64_t n = end - off;
        if (n > static_cast<int64_t>(tileElems)) n = static_cast<int64_t>(tileElems);
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(n * (int64_t)sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, xGm[off], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();

        __ubuf__ T *xp = (__ubuf__ T *)inL.GetPhyAddr();
        __ubuf__ int64_t *op = (__ubuf__ int64_t *)outL.GetPhyAddr();
        for (int64_t i = 0; i < n; ++i) {
            uint32_t b = BinOf<T>(xp[i], minVal);
            if (merged != 0 && neg0 >= 0 && b == (uint32_t)neg0) {
                b = (uint32_t)pos0;
            }
            uint32_t w = b >> 5;
            uint32_t s = b & 31u;
            uint32_t mask = (s == 0u) ? 0u : ((1u << s) - 1u);
            uint32_t r = pw[w] + Popc32(bp[w] & mask);
            op[i] = (int64_t)r;
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopyExtParams cp2{1, static_cast<uint32_t>(n * (int64_t)sizeof(int64_t)), 0, 0, 0};
        AscendC::DataCopyPad(invGm[off], outL, cp2);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// ---------------------------------------------------------------------------
// y output: y[rank(b)] = value(b) for every present b
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void unique_emit(GM_ADDR y, GM_ADDR minmax, GM_ADDR bmp, GM_ADDR wsK,
                                       uint32_t bmpWords)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    AscendC::GlobalTensor<int64_t> mmGm;
    AscendC::GlobalTensor<int64_t> kGm;
    yGm.SetGlobalBuffer((__gm__ T *)y);
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmp);
    mmGm.SetGlobalBuffer((__gm__ int64_t *)minmax);
    kGm.SetGlobalBuffer((__gm__ int64_t *)wsK);

    AscendC::TBuf<AscendC::TPosition::VECCALC> bmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kBug;
    pipe.InitBuffer(bmpBuf, (uint32_t)MAX_BMP_WORDS * 4u);
    pipe.InitBuffer(outBuf, (uint32_t)EMIT_CHUNK * sizeof(T));
    pipe.InitBuffer(mmBuf, 64);
    pipe.InitBuffer(kBug, 64);

    auto bmpL = bmpBuf.Get<uint32_t>();
    {
        AscendC::DataCopyExtParams cp{1, bmpWords * 4u, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(bmpL, bmpGm, cp, pp);
    }
    auto mmL = mmBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(mmL, mmGm, cp, pp);
    }
    auto kL = kBug.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(kL, kGm, cp, pp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    __ubuf__ uint32_t *bp = (__ubuf__ uint32_t *)bmpL.GetPhyAddr();
    const int64_t minVal = ((__ubuf__ int64_t *)mmL.GetPhyAddr())[0];
    const int64_t maxVal = ((__ubuf__ int64_t *)mmL.GetPhyAddr())[1];
    const int64_t merged = ((__ubuf__ int64_t *)kL.GetPhyAddr())[1];

    int32_t neg0, pos0;
    ZeroMergeBins<T>(neg0, pos0);
    const int64_t binCount = BinCountOf<T>(minVal, maxVal);

    auto outL = outBuf.Get<T>();
    __ubuf__ T *op = (__ubuf__ T *)outL.GetPhyAddr();
    int32_t fill = 0;
    int64_t emitted = 0;
    for (int64_t b = 0; b < binCount; ++b) {
        uint32_t ub = (uint32_t)b;
        uint32_t bit = (bp[ub >> 5] >> (ub & 31u)) & 1u;
        if (bit == 0u) {
            continue;
        }
        if (merged != 0 && neg0 >= 0 && ub == (uint32_t)neg0) {
            continue;
        }
        op[fill] = UnmapOf<T>(ub, minVal);
        ++fill;
        ++emitted;
        if (fill == EMIT_CHUNK) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams cp{1, (uint32_t)EMIT_CHUNK * (uint32_t)sizeof(T), 0, 0, 0};
            AscendC::DataCopyPad(yGm[emitted - (int64_t)EMIT_CHUNK], outL, cp);
            AscendC::PipeBarrier<PIPE_ALL>();
            fill = 0;
        }
    }
    if (fill > 0) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)fill * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(yGm[emitted - (int64_t)fill], outL, cp);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------
template <typename T>
void LaunchStage1(GM_ADDR x, GM_ADDR inverse, GM_ADDR wsPart, GM_ADDR wsMM, GM_ADDR wsBmpCore,
                  GM_ADDR wsBmpGlobal, GM_ADDR wsK, int64_t numel, int64_t numBlocks, int64_t blockLen,
                  uint32_t tileElems, uint32_t bmpWords, int64_t needInverse, void *stream)
{
    if constexpr (std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value) {
        unique_minmax_partial<T><<<numBlocks, nullptr, stream>>>(x, wsPart, numel, blockLen, tileElems);
        unique_minmax_final<<<1, nullptr, stream>>>(wsPart, wsMM, (int32_t)numBlocks);
    }
    unique_bitmap<T><<<numBlocks, nullptr, stream>>>(x, wsBmpCore, wsMM, numel, blockLen, tileElems, bmpWords);
    unique_merge<<<1, nullptr, stream>>>(wsBmpCore, wsBmpGlobal, (int32_t)numBlocks, bmpWords);
    unique_count<T><<<1, nullptr, stream>>>(wsBmpGlobal, wsK, bmpWords);
    if (needInverse != 0) {
        unique_inverse<T><<<numBlocks, nullptr, stream>>>(x, inverse, wsBmpGlobal, wsK, wsMM, numel,
                                                          blockLen, tileElems, bmpWords);
    }
}

template <typename T>
void LaunchEmit(GM_ADDR y, GM_ADDR wsMM, GM_ADDR wsBmpGlobal, GM_ADDR wsK, uint32_t bmpWords, void *stream)
{
    unique_emit<T><<<1, nullptr, stream>>>(y, wsMM, wsBmpGlobal, wsK, bmpWords);
}

// ---------------------------------------------------------------------------
// float32 route: sharded dense presence bitmap over the monotone 32 bit key
// ---------------------------------------------------------------------------
namespace f32impl {

constexpr int64_t KEY_BLK = 16;        // bitmap words per prefix block (512 keys)
constexpr int32_t WTILE = 4096;        // bitmap words per DMA tile in the scan kernels
constexpr int32_t ETILE = 2048;        // values buffered per y flush
constexpr uint32_t NEG0_KEY = 0x7FFFFFFFu;
constexpr uint32_t POS0_KEY = 0x80000000u;

__aicore__ inline uint32_t KeyOfBits(uint32_t b)
{
    uint32_t mask = (b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return (b | mask) - (b & mask);
}

__aicore__ inline uint32_t BitsOfKey(uint32_t k)
{
    return (k & 0x80000000u) ? (k - 0x80000000u) : (k ^ 0xFFFFFFFFu);
}

__aicore__ inline float ValOfKey(uint32_t k)
{
    uint32_t b = BitsOfKey(k);
    return *reinterpret_cast<float *>(&b);
}

}  // namespace f32impl

using namespace f32impl;

// per block key min/max
__global__ __aicore__ void unique_f32_keymin(GM_ADDR x, GM_ADDR out, int64_t numel, int64_t blockLen,
                                             uint32_t tileElems)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<uint32_t> xGm;
    AscendC::GlobalTensor<int64_t> outGm;
    xGm.SetGlobalBuffer((__gm__ uint32_t *)x);
    outGm.SetGlobalBuffer((__gm__ int64_t *)out);

    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resBuf;
    pipe.InitBuffer(inBuf, tileElems * 4u);
    pipe.InitBuffer(resBuf, 64);

    const int64_t c = (int64_t)AscendC::GetBlockIdx();
    int64_t start = blockLen * c;
    int64_t end = start + blockLen;
    if (end > numel) end = numel;
    uint32_t kmin = 0xFFFFFFFFu;
    uint32_t kmax = 0u;
    auto inL = inBuf.Get<uint32_t>();
    for (int64_t off = start; off < end; off += (int64_t)tileElems) {
        int64_t n = end - off;
        if (n > (int64_t)tileElems) n = (int64_t)tileElems;
        AscendC::DataCopyExtParams cp{1, (uint32_t)(n * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, xGm[off], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ uint32_t *p = (__ubuf__ uint32_t *)inL.GetPhyAddr();
        for (int64_t i = 0; i < n; ++i) {
            uint32_t k = KeyOfBits(p[i]);
            if (k < kmin) kmin = k;
            if (k > kmax) kmax = k;
        }
    }
    auto resL = resBuf.Get<int64_t>();
    __ubuf__ int64_t *op = (__ubuf__ int64_t *)resL.GetPhyAddr();
    op[0] = (int64_t)kmin;
    op[1] = (int64_t)kmax;
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cp2{1, 16, 0, 0, 0};
    AscendC::DataCopyPad(outGm[c * 2], resL, cp2);
}

// core c owns the key shard [keyMin + c*shardWords*32, +shardWords*32) and scans the whole input,
// setting the presence bits of the elements that fall into its shard (disjoint regions -> no races).
// It also reports the number of distinct keys it found in the shard.
__global__ __aicore__ void unique_f32_shard_bitset(GM_ADDR x, GM_ADDR bmp, GM_ADDR mm, GM_ADDR cnt,
                                                   int64_t numel, uint32_t tileElems, int64_t shardWords)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<uint32_t> xGm;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    AscendC::GlobalTensor<int64_t> mmGm;
    AscendC::GlobalTensor<int64_t> cntGm;
    xGm.SetGlobalBuffer((__gm__ uint32_t *)x);
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmp);
    mmGm.SetGlobalBuffer((__gm__ int64_t *)mm);
    cntGm.SetGlobalBuffer((__gm__ int64_t *)cnt);

    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zBuf;
    pipe.InitBuffer(inBuf, tileElems * 4u);
    pipe.InitBuffer(mmBuf, 64);
    pipe.InitBuffer(resBuf, 64);
    pipe.InitBuffer(zBuf, 4096 * 4u);

    auto mmL = mmBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(mmL, mmGm, cp, pp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t keyMin = ((__ubuf__ int64_t *)mmL.GetPhyAddr())[0];

    const int64_t c = (int64_t)AscendC::GetBlockIdx();
    const int64_t shardLo = keyMin + c * shardWords * 32;
    const int64_t shardHi = shardLo + shardWords * 32;
    const int64_t bmpBase = c * shardWords;
    int64_t hits = 0;

    // This block owns the whole shard region, so it can zero it before setting any bit: the bitmap is
    // allocated with empty() and must not rely on the content of fresh device memory.
    {
        auto zL = zBuf.Get<uint32_t>();
        AscendC::Duplicate(zL, (uint32_t)0, 4096);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t off = 0; off < shardWords; off += 4096) {
            int64_t nw = shardWords - off;
            if (nw > 4096) nw = 4096;
            AscendC::DataCopyExtParams cz{1, (uint32_t)(nw * 4), 0, 0, 0};
            AscendC::DataCopyPad(bmpGm[bmpBase + off], zL, cz);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    auto inL = inBuf.Get<uint32_t>();
    for (int64_t off = 0; off < numel; off += (int64_t)tileElems) {
        int64_t n = numel - off;
        if (n > (int64_t)tileElems) n = (int64_t)tileElems;
        AscendC::DataCopyExtParams cp{1, (uint32_t)(n * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, xGm[off], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ uint32_t *p = (__ubuf__ uint32_t *)inL.GetPhyAddr();
        for (int64_t i = 0; i < n; ++i) {
            uint32_t k = KeyOfBits(p[i]);
            if ((int64_t)k >= shardLo && (int64_t)k < shardHi) {
                int64_t delta = (int64_t)k - shardLo;
                uint32_t w = (uint32_t)(bmpBase + (delta >> 5));
                uint32_t bit = 1u << (uint32_t)(delta & 31);
                uint32_t cur = bmpGm.GetValue(w);
                if ((cur & bit) == 0u) {
                    bmpGm.SetValue(w, cur | bit);
                    ++hits;
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    auto resL = resBuf.Get<int64_t>();
    ((__ubuf__ int64_t *)resL.GetPhyAddr())[0] = hits;
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cp2{1, 8, 0, 0, 0};
    AscendC::DataCopyPad(cntGm[c], resL, cp2);
}

// single core: exclusive prefix of the per shard distinct counts -> bases, total distinct -> K and
// the -0.0 / +0.0 merge flag (the two keys are adjacent, so they share at most two bitmap words).
__global__ __aicore__ void unique_f32_shard_scan(GM_ADDR cnt, GM_ADDR base, GM_ADDR wsK, GM_ADDR mm,
                                                 GM_ADDR bmp, int64_t nShards, int64_t keyMin,
                                                 int64_t words)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<int64_t> cntGm;
    AscendC::GlobalTensor<int64_t> baseGm;
    AscendC::GlobalTensor<int64_t> kGm;
    AscendC::GlobalTensor<int64_t> mmGm;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    cntGm.SetGlobalBuffer((__gm__ int64_t *)cnt);
    baseGm.SetGlobalBuffer((__gm__ int64_t *)base);
    kGm.SetGlobalBuffer((__gm__ int64_t *)wsK);
    mmGm.SetGlobalBuffer((__gm__ int64_t *)mm);
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmp);

    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    pipe.InitBuffer(bBuf, 4096);
    pipe.InitBuffer(mmBuf, 64);
    pipe.InitBuffer(wBuf, 64);
    pipe.InitBuffer(outBuf, 64);

    auto bL = bBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, (uint32_t)(nShards * 8), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(bL, cntGm, cp, pp);
    }
    auto mmL = mmBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(mmL, mmGm, cp, pp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    __ubuf__ int64_t *cp32 = (__ubuf__ int64_t *)bL.GetPhyAddr();
    const int64_t kmin = ((__ubuf__ int64_t *)mmL.GetPhyAddr())[0];
    const int64_t span = words * 32;

    int64_t cum = 0;
    for (int64_t i = 0; i < nShards; ++i) {
        int64_t v = cp32[i];
        cp32[i] = cum;
        cum += v;
    }
    auto outL = outBuf.Get<int64_t>();
    auto wL = wBuf.Get<uint32_t>();

    // merge detection: -0.0 key (0x7FFFFFFF) and +0.0 key (0x80000000)
    const int64_t pNeg = (int64_t)NEG0_KEY - kmin;
    const int64_t pPos = (int64_t)POS0_KEY - kmin;
    int64_t merged = 0;
    if (pNeg >= 0 && pNeg + 1 < span && pPos >= 0 && pPos < span) {
        const int64_t wN = pNeg >> 5;
        const int64_t wP = pPos >> 5;
        const int64_t w0 = (wN < wP) ? wN : wP;
        const int64_t nw = (wP - w0) + 1;
        AscendC::DataCopyExtParams cp{1, (uint32_t)(nw * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(wL, bmpGm[w0], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ uint32_t *wp = (__ubuf__ uint32_t *)wL.GetPhyAddr();
        uint32_t nb = (wp[(int64_t)(wN - w0)] >> (uint32_t)(pNeg & 31)) & 1u;
        uint32_t pb = (wp[(int64_t)(wP - w0)] >> (uint32_t)(pPos & 31)) & 1u;
        if (nb != 0u && pb != 0u) {
            merged = 1;
        }
    }

    __ubuf__ int64_t *op = (__ubuf__ int64_t *)outL.GetPhyAddr();
    op[0] = cum - merged;
    op[1] = merged;
    AscendC::PipeBarrier<PIPE_ALL>();
    {
        AscendC::DataCopyExtParams cp{1, (uint32_t)(nShards * 8), 0, 0, 0};
        AscendC::DataCopyPad(baseGm, bL, cp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPad(kGm, outL, cp);
    }
}

// core c walks its own shard bitmap and writes the exclusive prefix count at every KEY_BLK boundary.
__global__ __aicore__ void unique_f32_prefix(GM_ADDR bmp, GM_ADDR base, GM_ADDR prefix,
                                             int64_t shardWords)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    AscendC::GlobalTensor<int64_t> baseGm;
    AscendC::GlobalTensor<int32_t> preGm;
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmp);
    baseGm.SetGlobalBuffer((__gm__ int64_t *)base);
    preGm.SetGlobalBuffer((__gm__ int32_t *)prefix);

    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf;
    pipe.InitBuffer(wBuf, WTILE * 4u);
    pipe.InitBuffer(oBuf, (WTILE / 16) * 4u + 64);
    pipe.InitBuffer(bBuf, 64);

    auto bL = bBuf.Get<int64_t>();
    const int64_t c = (int64_t)AscendC::GetBlockIdx();
    {
        AscendC::DataCopyExtParams cp{1, 8, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(bL, baseGm[c], cp, pp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    int64_t run = ((__ubuf__ int64_t *)bL.GetPhyAddr())[0];

    const int64_t wStart = c * shardWords;
    const int64_t nBlocks = shardWords / 16;
    auto wL = wBuf.Get<uint32_t>();
    auto oL = oBuf.Get<int32_t>();
    for (int64_t b = 0; b < nBlocks; b += (int64_t)(WTILE / 16)) {
        int64_t nb = nBlocks - b;
        if (nb > (int64_t)(WTILE / 16)) nb = (int64_t)(WTILE / 16);
        int64_t nw = nb * 16;
        AscendC::DataCopyExtParams cp{1, (uint32_t)(nw * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(wL, bmpGm[wStart + b * 16], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ uint32_t *wp = (__ubuf__ uint32_t *)wL.GetPhyAddr();
        __ubuf__ int32_t *op = (__ubuf__ int32_t *)oL.GetPhyAddr();
        for (int64_t j = 0; j < nb; ++j) {
            op[j] = (int32_t)run;
            int64_t s = 0;
            for (int64_t t = 0; t < 16; ++t) s += (int64_t)Popc32(wp[j * 16 + t]);
            run += s;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp2{1, (uint32_t)(nb * 4), 0, 0, 0};
        AscendC::DataCopyPad(preGm[(int64_t)(c * shardWords / 16) + b], oL, cp2);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// inverse[i] = rank(bin(x[i])) = the exclusive prefix count at its block + the popcount below the bit
__global__ __aicore__ void unique_f32_inverse(GM_ADDR x, GM_ADDR inverse, GM_ADDR bmp, GM_ADDR prefix,
                                              GM_ADDR mm, GM_ADDR wsK, int64_t numel, int64_t blockLen,
                                              uint32_t tileElems, int64_t keyMin)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<uint32_t> xGm;
    AscendC::GlobalTensor<int64_t> invGm;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    AscendC::GlobalTensor<int32_t> preGm;
    AscendC::GlobalTensor<int64_t> kGm;
    xGm.SetGlobalBuffer((__gm__ uint32_t *)x);
    invGm.SetGlobalBuffer((__gm__ int64_t *)inverse);
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmp);
    preGm.SetGlobalBuffer((__gm__ int32_t *)prefix);
    kGm.SetGlobalBuffer((__gm__ int64_t *)wsK);

    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kBuf;
    pipe.InitBuffer(inBuf, tileElems * 4u);
    pipe.InitBuffer(outBuf, tileElems * 8u);
    pipe.InitBuffer(kBuf, 64);

    auto kL = kBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(kL, kGm, cp, pp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t merged = ((__ubuf__ int64_t *)kL.GetPhyAddr())[1];

    const int64_t c = (int64_t)AscendC::GetBlockIdx();
    int64_t start = blockLen * c;
    int64_t end = start + blockLen;
    if (end > numel) end = numel;

    auto inL = inBuf.Get<uint32_t>();
    auto outL = outBuf.Get<int64_t>();
    for (int64_t off = start; off < end; off += (int64_t)tileElems) {
        int64_t n = end - off;
        if (n > (int64_t)tileElems) n = (int64_t)tileElems;
        AscendC::DataCopyExtParams cp{1, (uint32_t)(n * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, xGm[off], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ uint32_t *xp = (__ubuf__ uint32_t *)inL.GetPhyAddr();
        __ubuf__ int64_t *op = (__ubuf__ int64_t *)outL.GetPhyAddr();
        for (int64_t i = 0; i < n; ++i) {
            uint32_t k = KeyOfBits(xp[i]);
            int64_t delta = (int64_t)k - keyMin;
            int64_t w = delta >> 5;
            int64_t blk = w >> 4;
            int64_t r = (int64_t)preGm.GetValue((uint32_t)blk);
            for (int64_t j = blk * 16; j < w; ++j) {
                r += (int64_t)Popc32(bmpGm.GetValue((uint32_t)j));
            }
            uint32_t s = (uint32_t)(delta & 31);
            uint32_t mask = (s == 0u) ? 0u : ((1u << s) - 1u);
            r += (int64_t)Popc32(bmpGm.GetValue((uint32_t)w) & mask);
            if (merged != 0 && k >= POS0_KEY) {
                r -= 1;
            }
            op[i] = r;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams cp2{1, (uint32_t)(n * 8), 0, 0, 0};
        AscendC::DataCopyPad(invGm[off], outL, cp2);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// y: walk the shard bitmap in key order, rank each present bit, write the value at that rank
__global__ __aicore__ void unique_f32_emit(GM_ADDR y, GM_ADDR bmp, GM_ADDR base, GM_ADDR mm, GM_ADDR wsK,
                                           int64_t shardWords)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<uint32_t> bmpGm;
    AscendC::GlobalTensor<int64_t> baseGm;
    AscendC::GlobalTensor<int64_t> mmGm;
    AscendC::GlobalTensor<int64_t> kGm;
    yGm.SetGlobalBuffer((__gm__ float *)y);
    bmpGm.SetGlobalBuffer((__gm__ uint32_t *)bmp);
    baseGm.SetGlobalBuffer((__gm__ int64_t *)base);
    mmGm.SetGlobalBuffer((__gm__ int64_t *)mm);
    kGm.SetGlobalBuffer((__gm__ int64_t *)wsK);

    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mmBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kBuf;
    pipe.InitBuffer(wBuf, WTILE * 4u);
    pipe.InitBuffer(oBuf, ETILE * 4u + 64);
    pipe.InitBuffer(bBuf, 64);
    pipe.InitBuffer(mmBuf, 64);
    pipe.InitBuffer(kBuf, 64);

    const int64_t c = (int64_t)AscendC::GetBlockIdx();
    auto bL = bBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 8, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(bL, baseGm[c], cp, pp);
    }
    auto mmL = mmBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(mmL, mmGm, cp, pp);
    }
    auto kkL = kBuf.Get<int64_t>();
    {
        AscendC::DataCopyExtParams cp{1, 16, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(kkL, kGm, cp, pp);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    const int64_t keyMin = ((__ubuf__ int64_t *)mmL.GetPhyAddr())[0];
    const int64_t merged = ((__ubuf__ int64_t *)kkL.GetPhyAddr())[1];
    int64_t nat = ((__ubuf__ int64_t *)bL.GetPhyAddr())[0];

    const int64_t wStart = c * shardWords;
    const int64_t shardLo = keyMin + c * shardWords * 32;
    auto wL = wBuf.Get<uint32_t>();
    auto oL = oBuf.Get<float>();
    __ubuf__ float *op = (__ubuf__ float *)oL.GetPhyAddr();
    int32_t fill = 0;
    int64_t fillPos = 0;

    for (int64_t b = 0; b < shardWords; b += (int64_t)WTILE) {
        int64_t nw = shardWords - b;
        if (nw > (int64_t)WTILE) nw = (int64_t)WTILE;
        AscendC::DataCopyExtParams cp{1, (uint32_t)(nw * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(wL, bmpGm[wStart + b], cp, pp);
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ uint32_t *wp = (__ubuf__ uint32_t *)wL.GetPhyAddr();
        for (int64_t j = 0; j < nw; ++j) {
            uint32_t word = wp[j];
            if (word == 0u) {
                continue;
            }
            const int64_t wkey0 = shardLo + (b + j) * 32;
            for (int32_t t = 0; t < 32; ++t) {
                if (((word >> (uint32_t)t) & 1u) == 0u) {
                    continue;
                }
                uint32_t k = (uint32_t)(wkey0 + t);
                // nat is the number of distinct keys strictly below k (natural rank)
                if (!(merged != 0 && k == NEG0_KEY)) {
                    int64_t idx = nat;
                    if (merged != 0 && k >= POS0_KEY) {
                        idx -= 1;
                    }
                    if (fill == 0) {
                        fillPos = idx;
                    } else if (idx != fillPos + fill) {
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopyExtParams co{1, (uint32_t)fill * 4u, 0, 0, 0};
                        AscendC::DataCopyPad(yGm[fillPos], oL, co);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        fill = 0;
                        fillPos = idx;
                    }
                    op[fill] = ValOfKey(k);
                    ++fill;
                    if (fill == ETILE) {
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopyExtParams co{1, (uint32_t)fill * 4u, 0, 0, 0};
                        AscendC::DataCopyPad(yGm[fillPos], oL, co);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        fill = 0;
                    }
                }
                ++nat;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    if (fill > 0) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams co{1, (uint32_t)fill * 4u, 0, 0, 0};
        AscendC::DataCopyPad(yGm[fillPos], oL, co);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// ---------------------------------------------------------------------------
// tiling
// ---------------------------------------------------------------------------
std::tuple<int64_t, int64_t, int64_t> calc_unique_tiling(int64_t numel)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    if (coreNum > 512) coreNum = 512;
    int64_t need = (numel + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
    if (need < 1) need = 1;
    int64_t numBlocks = need < coreNum ? need : coreNum;
    int64_t blockLen = (numel + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, blockLen, (int64_t)TILE_ELEMS);
}

// ---------------------------------------------------------------------------
// extern "C" per dtype entry points
// ---------------------------------------------------------------------------
extern "C" {

#define UNIQUE_STAGE1_IMPL(NAME, TYPE)                                                                   \
    void unique_stage1_##NAME(GM_ADDR x, GM_ADDR inverse, GM_ADDR wsPart, GM_ADDR wsMM,                  \
                              GM_ADDR wsBmpCore, GM_ADDR wsBmpGlobal, GM_ADDR wsK, int64_t numel,        \
                              int64_t numBlocks, int64_t blockLen, uint32_t tileElems, uint32_t bmpWords, \
                              int64_t needInverse, void *stream)                                          \
    {                                                                                                    \
        LaunchStage1<TYPE>(x, inverse, wsPart, wsMM, wsBmpCore, wsBmpGlobal, wsK, numel, numBlocks,       \
                           blockLen, tileElems, bmpWords, needInverse, stream);                          \
    }

#define UNIQUE_EMIT_IMPL(NAME, TYPE)                                                                     \
    void unique_emit_##NAME(GM_ADDR y, GM_ADDR wsMM, GM_ADDR wsBmpGlobal, GM_ADDR wsK, uint32_t bmpWords, \
                            void *stream)                                                                 \
    {                                                                                                    \
        LaunchEmit<TYPE>(y, wsMM, wsBmpGlobal, wsK, bmpWords, stream);                                    \
    }

UNIQUE_STAGE1_IMPL(u8, uint8_t)
UNIQUE_STAGE1_IMPL(i8, int8_t)
UNIQUE_STAGE1_IMPL(u16, uint16_t)
UNIQUE_STAGE1_IMPL(i32, int32_t)
UNIQUE_STAGE1_IMPL(i64, int64_t)

UNIQUE_EMIT_IMPL(u8, uint8_t)
UNIQUE_EMIT_IMPL(i8, int8_t)
UNIQUE_EMIT_IMPL(u16, uint16_t)
UNIQUE_EMIT_IMPL(i32, int32_t)
UNIQUE_EMIT_IMPL(i64, int64_t)

void unique_f32_pre(GM_ADDR x, GM_ADDR wsPart, GM_ADDR wsMM, int64_t numel, int64_t numBlocks,
                    int64_t blockLen, uint32_t tileElems, void *stream)
{
    unique_f32_keymin<<<numBlocks, nullptr, stream>>>(x, wsPart, numel, blockLen, tileElems);
    unique_minmax_final<<<1, nullptr, stream>>>(wsPart, wsMM, (int32_t)numBlocks);
}

void unique_f32_bitset(GM_ADDR x, GM_ADDR bmp, GM_ADDR wsMM, GM_ADDR cnt, int64_t numel,
                       int64_t numBlocks, uint32_t tileElems, int64_t shardWords, void *stream)
{
    unique_f32_shard_bitset<<<numBlocks, nullptr, stream>>>(x, bmp, wsMM, cnt, numel, tileElems,
                                                            shardWords);
}

void unique_f32_scan(GM_ADDR cnt, GM_ADDR base, GM_ADDR wsK, GM_ADDR wsMM, GM_ADDR bmp,
                     int64_t numBlocks, int64_t keyMin, int64_t words, void *stream)
{
    unique_f32_shard_scan<<<1, nullptr, stream>>>(cnt, base, wsK, wsMM, bmp, numBlocks, keyMin, words);
}

void unique_f32_pref(GM_ADDR bmp, GM_ADDR base, GM_ADDR prefix, int64_t numBlocks, int64_t shardWords,
                     void *stream)
{
    unique_f32_prefix<<<numBlocks, nullptr, stream>>>(bmp, base, prefix, shardWords);
}

void unique_f32_inv(GM_ADDR x, GM_ADDR inverse, GM_ADDR bmp, GM_ADDR prefix, GM_ADDR wsMM,
                    GM_ADDR wsK, int64_t numel, int64_t numBlocks, int64_t blockLen,
                    uint32_t tileElems, int64_t keyMin, void *stream)
{
    unique_f32_inverse<<<numBlocks, nullptr, stream>>>(x, inverse, bmp, prefix, wsMM, wsK, numel,
                                                       blockLen, tileElems, keyMin);
}

void unique_f32_out(GM_ADDR y, GM_ADDR bmp, GM_ADDR base, GM_ADDR wsMM, GM_ADDR wsK,
                    int64_t numBlocks, int64_t shardWords, void *stream)
{
    unique_f32_emit<<<numBlocks, nullptr, stream>>>(y, bmp, base, wsMM, wsK, shardWords);
}
}
