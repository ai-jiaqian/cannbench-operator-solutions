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
 * \brief Unique device kernels + launch wrappers (bisheng + -xasc, dav-2201).
 *
 *   y       = distinct values of flatten(x), ascending numeric order
 *   inverse = int64 rank array with flatten(x)[i] == y[inverse[i]]
 *
 * Every input element is mapped by an order preserving transform to an unsigned code so
 * that numeric order equals code order:
 *   uint8  : code = v                    int8  : code = bits ^ 0x80
 *   int32  : code = bits ^ 0x80000000    int64 : code = bits ^ 0x8000..
 *   fp16/bf16 : sign ? ~bits : bits | 0x8000   (16 bit, -0.0 folded onto +0.0)
 *   fp32      : sign ? ~bits : bits | 0x80000000
 *
 * Two strategies are available and the choice is made on device from the reduced code range
 * (maxKey - minKey + 1):
 *
 *   dense (cap <= 262144):
 *     uk_dpres builds a per-core bitmap over the reduced code range, uk_dscan ORs the
 *     per-core bitmaps and builds the exclusive word prefix (+ D), uk_dy decodes the distinct
 *     codes into y and uk_dinv derives each element rank as
 *     wordPrefix[code>>5] + popcount(bitmapWord & ((1 << (code & 31)) - 1)).
 *
 *   radix (cap > 262144):
 *     LSD radix sort (8 bit digits, 256 bins) of the (reduced code, original index) pairs.
 *     uk_rhist/uk_rscan/uk_rscatter implement the stable counting sort pass; the scatter
 *     stages the locally digit sorted tile in UB and writes each digit block with a
 *     DataCopyPad so that no scalar GM store is ever needed.  uk_rflag/uk_rbase/uk_remit
 *     turn runs into y and into a per sorted-position rank array, and a second radix sort of
 *     (original index, rank) ordered by original index yields inverse in linear order.
 *
 * All per element work outside DMA is UB scalar access or vector primitives; the only GM
 * stores are DataCopy/DataCopyPad transfers from UB.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "unique_launch.h"

using namespace AscendC;

namespace {

constexpr int64_t UK_TILE = 2048;  // elements staged per input tile
// step used by the bitmap OR-merge in uk_dscan_kernel; the builtin is asked for 2x the step
// so the whole range is covered regardless of how many elements the call actually processes.
constexpr int64_t UK_ORSTEP = 64;
// number of independent bitmap lanes used to break the dependent read-modify-write chain in
// uk_dpres_kernel (uk_dpres_kernel is the dominant cost of the dense path)
constexpr int64_t UK_LANES = 4;
// upper bound on the payload of a single DataCopyPad burst (elements of uint32_t => 4 KB)
constexpr int64_t UK_BURST = 1024;
constexpr int64_t UK_STILE = 2048; // elements staged per scatter/remit tile
constexpr int64_t UK_YOUT = 2048;  // y emit buffer elements

constexpr int64_t UK_EV0 = 0;
constexpr int64_t UK_EV1 = 1;
constexpr int64_t UK_EV2 = 2;
constexpr int64_t UK_EV3 = 3;
constexpr int64_t UK_EV4 = 4;
constexpr int64_t UK_EV5 = 5;
constexpr int64_t UK_EV6 = 6;
constexpr int64_t UK_EV7 = 7;

// ---------------------------------------------------------------------------
// raw element types per dtype code
// ---------------------------------------------------------------------------
template <int64_t DT>
struct UkRaw;

template <>
struct UkRaw<UK_DT_U8> {
    using T = uint8_t;
};
template <>
struct UkRaw<UK_DT_I8> {
    using T = uint8_t;
};
template <>
struct UkRaw<UK_DT_F16> {
    using T = uint16_t;
};
template <>
struct UkRaw<UK_DT_BF16> {
    using T = uint16_t;
};
template <>
struct UkRaw<UK_DT_I32> {
    using T = uint32_t;
};
template <>
struct UkRaw<UK_DT_I64> {
    using T = uint64_t;
};
template <>
struct UkRaw<UK_DT_F32> {
    using T = uint32_t;
};

// order preserving transform: element bits -> reduced monotonic code
template <int64_t DT>
__aicore__ inline uint32_t UkCode(uint64_t r, uint64_t minKey)
{
    if constexpr (DT == UK_DT_U8) {
        return (uint32_t)(r & 0xFFu);
    } else if constexpr (DT == UK_DT_I8) {
        return ((uint32_t)(r & 0xFFu)) ^ 0x80u;
    } else if constexpr (DT == UK_DT_F16 || DT == UK_DT_BF16) {
        uint32_t b = (uint32_t)(r & 0xFFFFu);
        uint32_t k = (b & 0x8000u) ? ((~b) & 0xFFFFu) : (b | 0x8000u);
        if (k == 0x7FFFu) {
            k = 0x8000u;
        }
        return k;
    } else if constexpr (DT == UK_DT_I32) {
        uint32_t k = ((uint32_t)r) ^ 0x80000000u;
        return k - (uint32_t)minKey;
    } else if constexpr (DT == UK_DT_I64) {
        uint64_t k = r ^ 0x8000000000000000ull;
        return (uint32_t)(k - minKey);
    } else {
        uint32_t b = (uint32_t)(r & 0xFFFFFFFFu);
        uint32_t k = (b & 0x80000000u) ? (~b) : (b | 0x80000000u);
        if (k == 0x7FFFFFFFu) {
            k = 0x80000000u;
        }
        return k - (uint32_t)minKey;
    }
}

// inverse transform: reduced code -> element bits
template <int64_t DT>
__aicore__ inline uint64_t UkDecode(uint32_t code, uint64_t minKey)
{
    if constexpr (DT == UK_DT_U8) {
        return (uint64_t)code;
    } else if constexpr (DT == UK_DT_I8) {
        return (uint64_t)(code ^ 0x80u);
    } else if constexpr (DT == UK_DT_F16 || DT == UK_DT_BF16) {
        uint32_t k = code;
        return (uint64_t)((k >= 0x8000u) ? (k - 0x8000u) : ((~k) & 0xFFFFu));
    } else if constexpr (DT == UK_DT_I32) {
        uint32_t k = code + (uint32_t)minKey;
        return (uint64_t)(k ^ 0x80000000u);
    } else if constexpr (DT == UK_DT_I64) {
        uint64_t k = (uint64_t)code + minKey;
        return k ^ 0x8000000000000000ull;
    } else {
        uint32_t k = code + (uint32_t)minKey;
        uint32_t b = (k >= 0x80000000u) ? (k - 0x80000000u) : (~k);
        return (uint64_t)b;
    }
}

__aicore__ inline uint32_t UkPopc(uint32_t x)
{
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (uint32_t)((x * 0x01010101u) >> 24);
}

__aicore__ inline int64_t UkCtz(uint32_t x)
{
    int64_t n = 0;
    while ((x & 1u) == 0u) {
        x >>= 1;
        ++n;
    }
    return n;
}

struct UkPar {
    int64_t minKey;
    int64_t cap;
    int64_t mode;
    int64_t npass;
    int64_t d;
};

__aicore__ inline void UkReadPar(AscendC::TPipe &pipe, GM_ADDR params, UkPar &p)
{
    AscendC::TBuf<AscendC::TPosition::VECCALC> b;
    pipe.InitBuffer(b, 128);
    AscendC::LocalTensor<int64_t> l = b.Get<int64_t>();
    AscendC::GlobalTensor<int64_t> g;
    g.SetGlobalBuffer((__gm__ int64_t *)params);
    AscendC::DataCopyExtParams cp{1, 64, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
    AscendC::DataCopyPad(l, g, cp, pp);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV7);
    p.minKey = l.GetValue(UK_P_MINKEY);
    p.cap = l.GetValue(UK_P_CAP);
    p.mode = l.GetValue(UK_P_MODE);
    p.npass = l.GetValue(UK_P_NPASS);
    p.d = l.GetValue(UK_P_D);
}

__aicore__ inline int64_t UkMinI64(int64_t a, int64_t b) { return (a < b) ? a : b; }

} // namespace

// ===========================================================================
// setup kernels
// ===========================================================================

__global__ __aicore__ void uk_setparams_kernel(GM_ADDR params, int64_t minKey, int64_t cap, int64_t mode,
                                               int64_t npass, int64_t numBlocks, int64_t blockLength)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b;
    pipe.InitBuffer(b, 128);
    AscendC::LocalTensor<int64_t> l = b.Get<int64_t>();
    l.SetValue(UK_P_MINKEY, minKey);
    l.SetValue(UK_P_CAP, cap);
    l.SetValue(UK_P_MODE, mode);
    l.SetValue(UK_P_NPASS, npass);
    l.SetValue(UK_P_D, (int64_t)0);
    l.SetValue(UK_P_NUMBLK, numBlocks);
    l.SetValue(UK_P_BLOCKLEN, blockLength);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV0);
    AscendC::GlobalTensor<int64_t> g;
    g.SetGlobalBuffer((__gm__ int64_t *)params);
    AscendC::DataCopyExtParams cp{1, (uint32_t)(8 * 8), 0, 0, 0};
    AscendC::DataCopyPad(g, l, cp);
}

// per core min / max of the raw integer values; the tile is staged as 32-bit words so no
// wide UB element type is ever needed
template <typename IT>
__global__ __aicore__ void uk_mm_kernel(GM_ADDR x, GM_ADDR mm, int64_t n, int64_t blockLength)
{
    constexpr int64_t WPE = (int64_t)sizeof(IT) / 4; // 32-bit words per element (1 or 2)
    int64_t blk = AscendC::GetBlockIdx();
    int64_t start = blk * blockLength;
    int64_t cnt = n - start;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b;
    pipe.InitBuffer(b, UK_TILE * 8 + 256);
    AscendC::LocalTensor<uint32_t> tl = b.Get<uint32_t>();
    AscendC::GlobalTensor<uint32_t> xg;
    xg.SetGlobalBuffer((__gm__ uint32_t *)x);

    int64_t mn = 0;
    int64_t mx = 0;
    bool first = true;
    for (int64_t off = 0; off < cnt; off += UK_TILE) {
        int64_t len = UkMinI64(cnt - off, UK_TILE);
        int64_t words = len * WPE;
        AscendC::DataCopyExtParams cp{1, (uint32_t)(words * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(tl, xg[(start + off) * WPE], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        for (int64_t i = 0; i < len; ++i) {
            int64_t v;
            if constexpr (WPE == 2) {
                uint64_t lo = (uint64_t)tl.GetValue(2 * i);
                uint64_t hi = (uint64_t)tl.GetValue(2 * i + 1);
                v = (int64_t)(lo | (hi << 32));
            } else {
                v = (int64_t)(int32_t)tl.GetValue(i);
            }
            if (first) {
                mn = v;
                mx = v;
                first = false;
            } else {
                if (v < mn) {
                    mn = v;
                }
                if (v > mx) {
                    mx = v;
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
    }

    AscendC::LocalTensor<int64_t> ol = b.Get<int64_t>();
    ol.SetValue(0, mn);
    ol.SetValue(1, mx);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
    AscendC::GlobalTensor<int64_t> mg;
    mg.SetGlobalBuffer((__gm__ int64_t *)mm);
    AscendC::DataCopyExtParams cp2{1, 16, 0, 0, 0};
    AscendC::DataCopyPad(mg[blk * 2], ol, cp2);
}

// cross core reduce of the integer min/max and reduced-range decision
__global__ __aicore__ void uk_prep_kernel(GM_ADDR mm, GM_ADDR params, int64_t numBlocks, int64_t dt)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> b;
    pipe.InitBuffer(b, 8192);
    AscendC::LocalTensor<int64_t> l = b.Get<int64_t>();
    AscendC::GlobalTensor<int64_t> g;
    g.SetGlobalBuffer((__gm__ int64_t *)mm);
    uint32_t nv = (uint32_t)(numBlocks * 2);
    AscendC::DataCopyExtParams cp{1, (uint32_t)(nv * 8), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int64_t> pp{false, 0, 0, 0};
    AscendC::DataCopyPad(l, g, cp, pp);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);

    int64_t mnVal = l.GetValue(0);
    int64_t mxVal = l.GetValue(1);
    for (int64_t i = 1; i < numBlocks; ++i) {
        int64_t a = l.GetValue(i * 2);
        int64_t c = l.GetValue(i * 2 + 1);
        if (a < mnVal) {
            mnVal = a;
        }
        if (c > mxVal) {
            mxVal = c;
        }
    }

    uint64_t minKey;
    uint64_t maxKey;
    if (dt == UK_DT_I32) {
        minKey = (uint64_t)(((uint32_t)mnVal) ^ 0x80000000u);
        maxKey = (uint64_t)(((uint32_t)mxVal) ^ 0x80000000u);
    } else {
        minKey = ((uint64_t)mnVal) ^ 0x8000000000000000ull;
        maxKey = ((uint64_t)mxVal) ^ 0x8000000000000000ull;
    }
    uint64_t cap = maxKey - minKey + 1ull;
    if (cap == 0ull) {
        cap = ~0ull;
    }
    int64_t mode = (cap <= (uint64_t)UK_MAXCAP) ? 0 : 1;
    int64_t npass = 0;
    if (mode == 1) {
        uint64_t r = cap - 1ull;
        int64_t bits = 0;
        while (r != 0ull) {
            ++bits;
            r >>= 1;
        }
        npass = (bits + 7) / 8;
        if (npass < 1) {
            npass = 1;
        }
        if (npass > 4) {
            npass = 4;
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV1);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV1);
    l.SetValue(UK_P_MINKEY, (int64_t)minKey);
    l.SetValue(UK_P_CAP, (int64_t)cap);
    l.SetValue(UK_P_MODE, mode);
    l.SetValue(UK_P_NPASS, npass);
    l.SetValue(UK_P_D, (int64_t)0);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
    AscendC::GlobalTensor<int64_t> pg;
    pg.SetGlobalBuffer((__gm__ int64_t *)params);
    AscendC::DataCopyExtParams cp5{1, (uint32_t)(5 * 8), 0, 0, 0};
    AscendC::DataCopyPad(pg, l, cp5);
}

// ===========================================================================
// dense path
// ===========================================================================

template <int64_t DT>
__global__ __aicore__ void uk_dpres_kernel(GM_ADDR x, GM_ADDR bitmaps, GM_ADDR params, int64_t n,
                                           int64_t blockLength, int64_t minKey, int64_t cap)
{
    int64_t blk = AscendC::GetBlockIdx();
    int64_t start = blk * blockLength;
    int64_t cnt = n - start;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }

    AscendC::TPipe pipe;
    // minKey is produced on device by uk_prep_kernel for the integer dtypes, so it is read from the
    // shared params block.  cap stays an argument so that every dense kernel agrees on the bitmap
    // stride; for the integer path it is passed as UK_MAXCAP and the real range is only a guard.
    UkPar pv;
    UkReadPar(pipe, params, pv);
    if (pv.mode != 0 || pv.cap <= 0 || pv.cap > cap) {
        return;
    }
    uint64_t mk = (uint64_t)pv.minKey;
    int64_t capWords = (pv.cap + 31) / 32;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bb;
    // The per-lane base must stay 32 byte aligned for the vector OR merge below, and it also has
    // to be at least capWords + 128 so the 2*count merge overrun stays inside the buffer.
    int64_t laneStride = ((capWords + 128) + 7) / 8 * 8;
    // Four independent lanes cost 4x the bitmap in UB; keep the total inside a safe budget and fall
    // back to a single lane when a very wide reduced range would not fit (still correct, just less
    // latency overlap).
    int64_t lanes = UK_LANES;
    if (laneStride * UK_LANES * 4 + UK_TILE * 8 + 1024 > 120 * 1024) {
        lanes = 1;
    }
    pipe.InitBuffer(bb, laneStride * lanes * 4 + 256);
    AscendC::LocalTensor<uint32_t> bmA = bb.Get<uint32_t>();
    AscendC::TBuf<AscendC::TPosition::VECCALC> bt;
    pipe.InitBuffer(bt, UK_TILE * 8 + 256);
    using RT = typename UkRaw<DT>::T;
    AscendC::LocalTensor<RT> tl = bt.Get<RT>();

    for (int64_t w = 0; w < laneStride * lanes; ++w) {
        bmA.SetValue(w, (uint32_t)0);
    }

    AscendC::GlobalTensor<RT> xg;
    xg.SetGlobalBuffer((__gm__ RT *)x);
    for (int64_t off = 0; off < cnt; off += UK_TILE) {
        int64_t len = UkMinI64(cnt - off, UK_TILE);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(len * (int64_t)sizeof(RT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<RT> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(tl, xg[start + off], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        // The bitmap update is a scatter read-modify-write, so a single chain is latency bound.
        // UK_LANES independent bitmaps are updated in parallel, one per unrolled element, so the
        // UB round trips overlap; the lanes are OR-merged into lane 0 afterwards.
        int64_t i = 0;
        for (; lanes >= UK_LANES && i + UK_LANES <= len; i += UK_LANES) {
            uint32_t c0 = UkCode<DT>((uint64_t)tl.GetValue(i + 0), mk);
            uint32_t c1 = UkCode<DT>((uint64_t)tl.GetValue(i + 1), mk);
            uint32_t c2 = UkCode<DT>((uint64_t)tl.GetValue(i + 2), mk);
            uint32_t c3 = UkCode<DT>((uint64_t)tl.GetValue(i + 3), mk);
            uint32_t w0 = c0 >> 5;
            uint32_t w1 = c1 >> 5;
            uint32_t w2 = c2 >> 5;
            uint32_t w3 = c3 >> 5;
            bmA.SetValue(w0, bmA.GetValue(w0) | (1u << (c0 & 31u)));
            bmA.SetValue(laneStride + w1, bmA.GetValue(laneStride + w1) | (1u << (c1 & 31u)));
            bmA.SetValue(laneStride * 2 + w2, bmA.GetValue(laneStride * 2 + w2) | (1u << (c2 & 31u)));
            bmA.SetValue(laneStride * 3 + w3, bmA.GetValue(laneStride * 3 + w3) | (1u << (c3 & 31u)));
        }
        for (; i < len; ++i) {
            uint32_t c = UkCode<DT>((uint64_t)tl.GetValue(i), mk);
            uint32_t w = c >> 5;
            bmA.SetValue(w, bmA.GetValue(w) | (1u << (c & 31u)));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
    }

    AscendC::SetFlag<AscendC::HardEvent::S_V>(UK_EV1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(UK_EV1);
    for (int64_t l = 1; l < lanes; ++l) {
        for (int64_t base = 0; base < capWords; base += UK_ORSTEP) {
            int64_t nn = capWords - base;
            if (nn > (int64_t)UK_ORSTEP) {
                nn = (int64_t)UK_ORSTEP;
            }
            AscendC::Or(bmA[base], bmA[base], bmA[l * laneStride + base], (int32_t)(2 * nn));
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(UK_EV2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(UK_EV2);
    AscendC::GlobalTensor<uint32_t> bg;
    bg.SetGlobalBuffer((__gm__ uint32_t *)bitmaps);
    AscendC::DataCopyExtParams cp2{1, (uint32_t)(capWords * 4), 0, 0, 0};
    AscendC::DataCopyPad(bg[blk * capWords], bmA, cp2);
}

__global__ __aicore__ void uk_dscan_kernel(GM_ADDR bitmaps, GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR params,
                                           int64_t numBlocks, int64_t cap)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    UkPar pv;
    UkReadPar(pipe, params, pv);
    if (pv.mode != 0 || pv.cap <= 0 || pv.cap > cap) {
        return;
    }
    int64_t capWords = (pv.cap + 31) / 32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ba;
    pipe.InitBuffer(ba, capWords * 4 + 2048);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bt;
    pipe.InitBuffer(bt, capWords * 4 + 2048);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bp;
    pipe.InitBuffer(bp, 256);
    AscendC::LocalTensor<uint32_t> acc = ba.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> tmp = bt.Get<uint32_t>();
    AscendC::LocalTensor<int64_t> pl = bp.Get<int64_t>();

    AscendC::GlobalTensor<uint32_t> bg;
    bg.SetGlobalBuffer((__gm__ uint32_t *)bitmaps);

    AscendC::Duplicate(acc, (uint32_t)0, (int32_t)capWords);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UK_EV0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UK_EV0);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(UK_EV3);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(UK_EV3);
    for (int64_t w = 0; w < capWords; ++w) {
        acc.SetValue(w, 0u);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(UK_EV3);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(UK_EV3);

    // The OR-merge is issued in fixed 64 element steps requesting twice the step, so the whole
    // [0, capWords) range is covered no matter how the vector builtin counts its elements.
    for (int64_t c = 0; c < numBlocks; ++c) {
        AscendC::DataCopyExtParams cp{1, (uint32_t)(capWords * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(tmp, bg[c * capWords], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(UK_EV1);
        for (int64_t base = 0; base < capWords; base += UK_ORSTEP) {
            int64_t n = capWords - base;
            if (n > (int64_t)UK_ORSTEP) {
                n = (int64_t)UK_ORSTEP;
            }
            AscendC::Or(acc[base], acc[base], tmp[base], (int32_t)(2 * n));
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UK_EV0);
    }

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(UK_EV2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(UK_EV2);
    AscendC::GlobalTensor<uint32_t> cg;
    cg.SetGlobalBuffer((__gm__ uint32_t *)comb);
    AscendC::DataCopyExtParams cpw{1, (uint32_t)(capWords * 4), 0, 0, 0};
    AscendC::DataCopyPad(cg, acc, cpw);

    AscendC::SetFlag<AscendC::HardEvent::V_S>(UK_EV4);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(UK_EV4);
    uint32_t run = 0;
    for (int64_t w = 0; w < capWords; ++w) {
        uint32_t v = acc.GetValue(w);
        tmp.SetValue(w, run);
        run += UkPopc(v);
    }
    AscendC::GlobalTensor<uint32_t> wg;
    wg.SetGlobalBuffer((__gm__ uint32_t *)wordPrefix);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV5);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV5);
    AscendC::DataCopyPad(wg, tmp, cpw);

    pl.SetValue(0, (int64_t)run);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV6);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV6);
    AscendC::GlobalTensor<int64_t> pg;
    pg.SetGlobalBuffer((__gm__ int64_t *)params);
    AscendC::DataCopyExtParams cp8{1, 8, 0, 0, 0};
    AscendC::DataCopyPad(pg[UK_P_D], pl, cp8);
}

template <int64_t DT>
__global__ __aicore__ void uk_dy_kernel(GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR yBig, GM_ADDR params,
                                        int64_t cap, int64_t numBlocks, int64_t minKey)
{
    AscendC::TPipe pipe;
    UkPar pv;
    UkReadPar(pipe, params, pv);
    if (pv.mode != 0 || pv.cap <= 0 || pv.cap > cap) {
        return;
    }
    uint64_t mk = (uint64_t)pv.minKey;
    int64_t capWords = (pv.cap + 31) / 32;
    int64_t blk = AscendC::GetBlockIdx();
    int64_t w0 = blk * capWords / numBlocks;
    int64_t w1 = (blk + 1) * capWords / numBlocks;
    int64_t myw = w1 - w0;
    if (myw < 0) {
        myw = 0;
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> bc;
    pipe.InitBuffer(bc, myw * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bw;
    pipe.InitBuffer(bw, myw * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bo;
    pipe.InitBuffer(bo, UK_YOUT * 8 + 256);
    AscendC::LocalTensor<uint32_t> cw = bc.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> ww = bw.Get<uint32_t>();
    using RT = typename UkRaw<DT>::T;
    AscendC::LocalTensor<RT> ol = bo.Get<RT>();

    AscendC::GlobalTensor<uint32_t> cg;
    cg.SetGlobalBuffer((__gm__ uint32_t *)comb);
    AscendC::GlobalTensor<uint32_t> wg;
    wg.SetGlobalBuffer((__gm__ uint32_t *)wordPrefix);
    AscendC::GlobalTensor<RT> yg;
    yg.SetGlobalBuffer((__gm__ RT *)yBig);

    if (myw > 0) {
        AscendC::DataCopyExtParams cp{1, (uint32_t)(myw * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(cw, cg[w0], cp, pp);
        AscendC::DataCopyPad(ww, wg[w0], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
    }

    int64_t outCnt = 0;
    int64_t outStart = 0;
    for (int64_t j = 0; j < myw; ++j) {
        uint32_t bits = cw.GetValue(j);
        uint32_t cur = ww.GetValue(j);
        while (bits != 0u) {
            int64_t pb = UkCtz(bits);
            uint32_t code = (uint32_t)((uint32_t)(w0 + j) * 32u + (uint32_t)pb);
            if (outCnt == 0) {
                outStart = (int64_t)cur;
            }
            if (outCnt >= UK_YOUT) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
                AscendC::DataCopyExtParams ocp{1, (uint32_t)(outCnt * (int64_t)sizeof(RT)), 0, 0, 0};
                AscendC::DataCopyPad(yg[outStart], ol, ocp);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(UK_EV2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(UK_EV2);
                outCnt = 0;
                outStart = (int64_t)cur;
            }
            ol.SetValue(outCnt, (RT)UkDecode<DT>(code, mk));
            ++outCnt;
            ++cur;
            bits &= (bits - 1u);
        }
    }
    if (outCnt > 0) {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::DataCopyExtParams ocp{1, (uint32_t)(outCnt * (int64_t)sizeof(RT)), 0, 0, 0};
        AscendC::DataCopyPad(yg[outStart], ol, ocp);
    }
}

template <int64_t DT>
__global__ __aicore__ void uk_dinv_kernel(GM_ADDR x, GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR inv,
                                          GM_ADDR params, int64_t n, int64_t blockLength, int64_t minKey,
                                          int64_t cap)
{
    int64_t blk = AscendC::GetBlockIdx();
    int64_t start = blk * blockLength;
    int64_t cnt = n - start;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }

    AscendC::TPipe pipe;
    UkPar pv;
    UkReadPar(pipe, params, pv);
    if (pv.mode != 0 || pv.cap <= 0 || pv.cap > cap) {
        return;
    }
    uint64_t mk = (uint64_t)pv.minKey;
    int64_t capWords = (pv.cap + 31) / 32;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bc;
    pipe.InitBuffer(bc, capWords * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bw;
    pipe.InitBuffer(bw, capWords * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bt;
    pipe.InitBuffer(bt, UK_TILE * 8 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bo;
    pipe.InitBuffer(bo, UK_TILE * 8 + 256);
    AscendC::LocalTensor<uint32_t> cw = bc.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> ww = bw.Get<uint32_t>();
    using RT = typename UkRaw<DT>::T;
    AscendC::LocalTensor<RT> tl = bt.Get<RT>();
    AscendC::LocalTensor<int64_t> ol = bo.Get<int64_t>();

    AscendC::GlobalTensor<uint32_t> cg;
    cg.SetGlobalBuffer((__gm__ uint32_t *)comb);
    AscendC::GlobalTensor<uint32_t> wg;
    wg.SetGlobalBuffer((__gm__ uint32_t *)wordPrefix);
    AscendC::GlobalTensor<RT> xg;
    xg.SetGlobalBuffer((__gm__ RT *)x);
    AscendC::GlobalTensor<int64_t> ig;
    ig.SetGlobalBuffer((__gm__ int64_t *)inv);

    AscendC::DataCopyExtParams cpb{1, (uint32_t)(capWords * 4), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
    AscendC::DataCopyPad(cw, cg, cpb, pp);
    AscendC::DataCopyPad(ww, wg, cpb, pp);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);

    for (int64_t off = 0; off < cnt; off += UK_TILE) {
        int64_t len = UkMinI64(cnt - off, UK_TILE);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(len * (int64_t)sizeof(RT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<RT> pp2{false, 0, 0, 0};
        AscendC::DataCopyPad(tl, xg[start + off], cp, pp2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV1);
        for (int64_t i = 0; i < len; ++i) {
            uint32_t c = UkCode<DT>((uint64_t)tl.GetValue(i), mk);
            uint32_t w = c >> 5;
            uint32_t bit = c & 31u;
            uint32_t mask = (bit == 0u) ? 0u : ((1u << bit) - 1u);
            uint32_t rank = ww.GetValue(w) + UkPopc(cw.GetValue(w) & mask);
            ol.SetValue(i, (int64_t)rank);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
        AscendC::DataCopyExtParams cpo{1, (uint32_t)(len * 8), 0, 0, 0};
        AscendC::DataCopyPad(ig[start + off], ol, cpo);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(UK_EV3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(UK_EV3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV1);
    }
}

// ===========================================================================
// radix path
// ===========================================================================

template <int64_t DT>
__global__ __aicore__ void uk_rpack_kernel(GM_ADDR x, GM_ADDR keys, GM_ADDR pay, int64_t n, int64_t blockLength,
                                           int64_t minKey)
{
    int64_t blk = AscendC::GetBlockIdx();
    int64_t start = blk * blockLength;
    int64_t cnt = n - start;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }

    AscendC::TPipe pipe;
    uint64_t mk = (uint64_t)minKey;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bt;
    pipe.InitBuffer(bt, UK_STILE * 8 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bk;
    pipe.InitBuffer(bk, UK_STILE * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bp;
    pipe.InitBuffer(bp, UK_STILE * 4 + 256);
    using RT = typename UkRaw<DT>::T;
    AscendC::LocalTensor<RT> tl = bt.Get<RT>();
    AscendC::LocalTensor<uint32_t> kl = bk.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> pl = bp.Get<uint32_t>();

    AscendC::GlobalTensor<RT> xg;
    xg.SetGlobalBuffer((__gm__ RT *)x);
    AscendC::GlobalTensor<uint32_t> kg;
    kg.SetGlobalBuffer((__gm__ uint32_t *)keys);
    AscendC::GlobalTensor<uint32_t> pg;
    pg.SetGlobalBuffer((__gm__ uint32_t *)pay);

    for (int64_t off = 0; off < cnt; off += UK_STILE) {
        int64_t len = UkMinI64(cnt - off, UK_STILE);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(len * (int64_t)sizeof(RT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<RT> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(tl, xg[start + off], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV1);
        for (int64_t i = 0; i < len; ++i) {
            kl.SetValue(i, UkCode<DT>((uint64_t)tl.GetValue(i), mk));
            pl.SetValue(i, (uint32_t)(start + off + i));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
        AscendC::DataCopyExtParams cpk{1, (uint32_t)(len * 4), 0, 0, 0};
        AscendC::DataCopyPad(kg[start + off], kl, cpk);
        AscendC::DataCopyPad(pg[start + off], pl, cpk);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(UK_EV3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(UK_EV3);
    }
}

__global__ __aicore__ void uk_rhist_kernel(GM_ADDR keys, GM_ADDR hist, int64_t n, int64_t blockLength, int64_t pass)
{
    int64_t blk = AscendC::GetBlockIdx();
    int64_t start = blk * blockLength;
    int64_t cnt = n - start;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bh;
    pipe.InitBuffer(bh, UK_BINS * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bt;
    pipe.InitBuffer(bt, UK_TILE * 4 + 256);
    AscendC::LocalTensor<uint32_t> hl = bh.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> kl = bt.Get<uint32_t>();

    // scalar zeroing only: a vector builtin is not guaranteed to process the whole count
    for (int64_t d = 0; d < UK_BINS; ++d) {
        hl.SetValue(d, 0u);
    }

    AscendC::GlobalTensor<uint32_t> kg;
    kg.SetGlobalBuffer((__gm__ uint32_t *)keys);
    uint32_t sh = (uint32_t)(pass * 8);
    for (int64_t off = 0; off < cnt; off += UK_TILE) {
        int64_t len = UkMinI64(cnt - off, UK_TILE);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(len * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(kl, kg[start + off], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        for (int64_t i = 0; i < len; ++i) {
            uint32_t d = (kl.GetValue(i) >> sh) & 0xFFu;
            hl.SetValue(d, hl.GetValue(d) + 1u);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
    }

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
    AscendC::GlobalTensor<uint32_t> hg;
    hg.SetGlobalBuffer((__gm__ uint32_t *)hist);
    AscendC::DataCopyExtParams cph{1, (uint32_t)(UK_BINS * 4), 0, 0, 0};
    AscendC::DataCopyPad(hg[blk * UK_BINS], hl, cph);
}

__global__ __aicore__ void uk_rscan_kernel(GM_ADDR hist, GM_ADDR start, int64_t numBlocks)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    int64_t total = numBlocks * UK_BINS;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bh;
    pipe.InitBuffer(bh, total * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bs;
    pipe.InitBuffer(bs, total * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bd;
    pipe.InitBuffer(bd, UK_BINS * 8 + 256);
    AscendC::LocalTensor<uint32_t> hl = bh.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> sl = bs.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> dl = bd.Get<uint32_t>();

    AscendC::GlobalTensor<uint32_t> hg;
    hg.SetGlobalBuffer((__gm__ uint32_t *)hist);
    AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
    // the whole numBlocks*256 entry control array is moved in fixed 4 KB bursts rather than one
    // very long DataCopyPad, because a single burst of tens of KB is not reliably supported.
    for (int64_t c0 = 0; c0 < total; c0 += UK_BURST) {
        int64_t cn = total - c0;
        if (cn > UK_BURST) {
            cn = UK_BURST;
        }
        AscendC::DataCopyExtParams cp{1, (uint32_t)(cn * 4), 0, 0, 0};
        AscendC::DataCopyPad(hl[c0], hg[c0], cp, pp);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);

    // The destination layout must be digit-major, block-minor: every element of digit d, over all
    // blocks, has to live in ONE contiguous range that is ordered by block.  A flat block-major
    // prefix would place each block's digit groups inside that block's own segment, which only
    // sorts every block separately and never performs a global stable radix pass.
    AscendC::LocalTensor<uint32_t> gpre = dl[0];
    AscendC::LocalTensor<uint32_t> rn = dl[UK_BINS];
    for (int64_t d = 0; d < UK_BINS; ++d) {
        gpre.SetValue(d, 0u);
        rn.SetValue(d, 0u);
    }
    for (int64_t i = 0; i < total; ++i) {
        uint32_t d = (uint32_t)((int64_t)i & (int64_t)(UK_BINS - 1));
        gpre.SetValue(d, gpre.GetValue(d) + hl.GetValue(i));
    }
    uint32_t run = 0;
    for (int64_t d = 0; d < UK_BINS; ++d) {
        uint32_t t = gpre.GetValue(d);
        gpre.SetValue(d, run);
        run += t;
    }
    for (int64_t b = 0; b < numBlocks; ++b) {
        for (int64_t d = 0; d < UK_BINS; ++d) {
            uint32_t v = hl.GetValue(b * UK_BINS + d);
            sl.SetValue(b * UK_BINS + d, gpre.GetValue(d) + rn.GetValue(d));
            rn.SetValue(d, rn.GetValue(d) + v);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
    AscendC::GlobalTensor<uint32_t> sg;
    sg.SetGlobalBuffer((__gm__ uint32_t *)start);
    for (int64_t c0 = 0; c0 < total; c0 += UK_BURST) {
        int64_t cn = total - c0;
        if (cn > UK_BURST) {
            cn = UK_BURST;
        }
        AscendC::DataCopyExtParams cp{1, (uint32_t)(cn * 4), 0, 0, 0};
        AscendC::DataCopyPad(sg[c0], sl[c0], cp);
    }
}

__global__ __aicore__ void uk_rscatter_kernel(GM_ADDR kIn, GM_ADDR pIn, GM_ADDR kOut, GM_ADDR pOut, GM_ADDR start,
                                              int64_t n, int64_t blockLength, int64_t pass)
{
    int64_t blk = AscendC::GetBlockIdx();
    int64_t s0 = blk * blockLength;
    int64_t cnt = n - s0;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bc;
    pipe.InitBuffer(bc, UK_BINS * 4 * 5 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bk;
    pipe.InitBuffer(bk, UK_STILE * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bp;
    pipe.InitBuffer(bp, UK_STILE * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bok;
    pipe.InitBuffer(bok, (UK_STILE + 2048) * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bop;
    pipe.InitBuffer(bop, (UK_STILE + 2048) * 4 + 256);
    AscendC::LocalTensor<uint32_t> ctl = bc.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> ik = bk.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> ip = bp.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> ok = bok.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> op = bop.Get<uint32_t>();

    AscendC::GlobalTensor<uint32_t> sg;
    sg.SetGlobalBuffer((__gm__ uint32_t *)start);
    AscendC::GlobalTensor<uint32_t> kig;
    kig.SetGlobalBuffer((__gm__ uint32_t *)kIn);
    AscendC::GlobalTensor<uint32_t> pig;
    pig.SetGlobalBuffer((__gm__ uint32_t *)pIn);
    AscendC::GlobalTensor<uint32_t> kog;
    kog.SetGlobalBuffer((__gm__ uint32_t *)kOut);
    AscendC::GlobalTensor<uint32_t> pog;
    pog.SetGlobalBuffer((__gm__ uint32_t *)pOut);

    AscendC::DataCopyExtParams cps{1, (uint32_t)(UK_BINS * 4), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
    AscendC::DataCopyPad(ctl, sg[blk * UK_BINS], cps, pp);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);

    // ctl layout: [0,256) = mystart, [256,512) = lcnt, [512,768) = tcnt,
    //             [768,1024) = tslot, [1024,1280) = tpos/gstart
    AscendC::LocalTensor<uint32_t> mystartL = ctl[0];
    AscendC::LocalTensor<uint32_t> lcntL = ctl[256];
    AscendC::LocalTensor<uint32_t> tcntL = ctl[512];
    AscendC::LocalTensor<uint32_t> tslotL = ctl[768];
    AscendC::LocalTensor<uint32_t> tposL = ctl[1024];

    for (int64_t d = 0; d < UK_BINS; ++d) {
        lcntL.SetValue(d, (uint32_t)0);
    }

    uint32_t sh = (uint32_t)(pass * 8);
    for (int64_t off = 0; off < cnt; off += UK_STILE) {
        int64_t len = UkMinI64(cnt - off, UK_STILE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(UK_EV3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(UK_EV3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV4);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(len * 4), 0, 0, 0};
        AscendC::DataCopyPad(ik, kig[s0 + off], cp, pp);
        AscendC::DataCopyPad(ip, pig[s0 + off], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);

        for (int64_t d = 0; d < UK_BINS; ++d) {
            tcntL.SetValue(d, (uint32_t)0);
        }
        for (int64_t i = 0; i < len; ++i) {
            uint32_t d = (ik.GetValue(i) >> sh) & 0xFFu;
            tcntL.SetValue(d, tcntL.GetValue(d) + 1u);
        }
        uint32_t acc = 0;
        for (int64_t d = 0; d < UK_BINS; ++d) {
            uint32_t t = tcntL.GetValue(d);
            uint32_t slot = (t + 7u) & ~7u;
            tslotL.SetValue(d, acc);
            tposL.SetValue(d, acc);
            acc += slot;
        }
        for (int64_t i = 0; i < len; ++i) {
            uint32_t k = ik.GetValue(i);
            uint32_t d = (k >> sh) & 0xFFu;
            uint32_t pos = tposL.GetValue(d);
            tposL.SetValue(d, pos + 1u);
            ok.SetValue(pos, k);
            op.SetValue(pos, ip.GetValue(i));
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        for (int64_t d = 0; d < UK_BINS; ++d) {
            uint32_t t = tcntL.GetValue(d);
            if (t == 0u) {
                continue;
            }
            uint32_t gpos = mystartL.GetValue(d) + lcntL.GetValue(d);
            AscendC::DataCopyExtParams cpo{1, (uint32_t)(t * 4), 0, 0, 0};
            AscendC::DataCopyPad(kog[gpos], ok[tslotL.GetValue(d)], cpo);
            AscendC::DataCopyPad(pog[gpos], op[tslotL.GetValue(d)], cpo);
        }
        for (int64_t d = 0; d < UK_BINS; ++d) {
            lcntL.SetValue(d, lcntL.GetValue(d) + tcntL.GetValue(d));
        }
    }
}

__global__ __aicore__ void uk_rflag_kernel(GM_ADDR keys, GM_ADDR flags, GM_ADDR cnts, int64_t n, int64_t blockLength)
{
    int64_t blk = AscendC::GetBlockIdx();
    int64_t s0 = blk * blockLength;
    int64_t cnt = n - s0;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bt;
    pipe.InitBuffer(bt, UK_STILE * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf;
    pipe.InitBuffer(bf, UK_STILE + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> b1;
    pipe.InitBuffer(b1, 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bc;
    pipe.InitBuffer(bc, 256);
    AscendC::LocalTensor<uint32_t> kl = bt.Get<uint32_t>();
    AscendC::LocalTensor<uint8_t> fl = bf.Get<uint8_t>();
    AscendC::LocalTensor<uint32_t> pv = b1.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> cc = bc.Get<uint32_t>();

    AscendC::GlobalTensor<uint32_t> kg;
    kg.SetGlobalBuffer((__gm__ uint32_t *)keys);
    AscendC::GlobalTensor<uint8_t> fg;
    fg.SetGlobalBuffer((__gm__ uint8_t *)flags);
    AscendC::GlobalTensor<uint32_t> cg;
    cg.SetGlobalBuffer((__gm__ uint32_t *)cnts);

    uint32_t prev = 0;
    if (s0 > 0) {
        AscendC::DataCopyExtParams cp1{1, 4, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp1{false, 0, 0, 0};
        AscendC::DataCopyPad(pv, kg[s0 - 1], cp1, pp1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV5);
        prev = pv.GetValue(0);
    }

    uint32_t local = 0;
    for (int64_t off = 0; off < cnt; off += UK_STILE) {
        int64_t len = UkMinI64(cnt - off, UK_STILE);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(len * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(kl, kg[s0 + off], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        for (int64_t i = 0; i < len; ++i) {
            uint32_t k = kl.GetValue(i);
            uint8_t f;
            if (s0 + off + i == 0) {
                f = 1;
            } else {
                f = (k != prev) ? 1 : 0;
            }
            fl.SetValue(i, f);
            local += (uint32_t)f;
            prev = k;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::DataCopyExtParams cpf{1, (uint32_t)len, 0, 0, 0};
        AscendC::DataCopyPad(fg[s0 + off], fl, cpf);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(UK_EV2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(UK_EV2);
    }

    cc.SetValue(0, local);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV3);
    AscendC::DataCopyExtParams cpc{1, 4, 0, 0, 0};
    AscendC::DataCopyPad(cg[blk], cc, cpc);
}

__global__ __aicore__ void uk_rbase_kernel(GM_ADDR cnts, GM_ADDR base, GM_ADDR params, int64_t numBlocks)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bc;
    pipe.InitBuffer(bc, numBlocks * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bb;
    pipe.InitBuffer(bb, numBlocks * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bp;
    pipe.InitBuffer(bp, 256);
    AscendC::LocalTensor<uint32_t> cl = bc.Get<uint32_t>();
    AscendC::LocalTensor<uint32_t> bl = bb.Get<uint32_t>();
    AscendC::LocalTensor<int64_t> pl = bp.Get<int64_t>();

    AscendC::GlobalTensor<uint32_t> cg;
    cg.SetGlobalBuffer((__gm__ uint32_t *)cnts);
    AscendC::DataCopyExtParams cp{1, (uint32_t)(numBlocks * 4), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
    AscendC::DataCopyPad(cl, cg, cp, pp);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);

    uint32_t run = 0;
    for (int64_t i = 0; i < numBlocks; ++i) {
        uint32_t v = cl.GetValue(i);
        bl.SetValue(i, run);
        run += v;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
    AscendC::GlobalTensor<uint32_t> bg;
    bg.SetGlobalBuffer((__gm__ uint32_t *)base);
    AscendC::DataCopyPad(bg, bl, cp);

    pl.SetValue(0, (int64_t)run);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV2);
    AscendC::GlobalTensor<int64_t> pg;
    pg.SetGlobalBuffer((__gm__ int64_t *)params);
    AscendC::DataCopyExtParams cp8{1, 8, 0, 0, 0};
    AscendC::DataCopyPad(pg[UK_P_D], pl, cp8);
}

template <int64_t DT>
__global__ __aicore__ void uk_remit_kernel(GM_ADDR keys, GM_ADDR flags, GM_ADDR base, GM_ADDR yBig, GM_ADDR rank,
                                           int64_t n, int64_t blockLength, int64_t minKey)
{
    int64_t blk = AscendC::GetBlockIdx();
    int64_t s0 = blk * blockLength;
    int64_t cnt = n - s0;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }

    AscendC::TPipe pipe;
    uint64_t mk = (uint64_t)minKey;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bt;
    pipe.InitBuffer(bt, UK_STILE * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bf;
    pipe.InitBuffer(bf, UK_STILE + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> br;
    pipe.InitBuffer(br, UK_STILE * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> by;
    pipe.InitBuffer(by, UK_STILE * 8 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> b1;
    pipe.InitBuffer(b1, 512);
    AscendC::LocalTensor<uint32_t> kl = bt.Get<uint32_t>();
    AscendC::LocalTensor<uint8_t> fl = bf.Get<uint8_t>();
    AscendC::LocalTensor<uint32_t> rl = br.Get<uint32_t>();
    using RT = typename UkRaw<DT>::T;
    AscendC::LocalTensor<RT> yl = by.Get<RT>();
    AscendC::LocalTensor<uint32_t> one = b1.Get<uint32_t>();

    AscendC::GlobalTensor<uint32_t> kg;
    kg.SetGlobalBuffer((__gm__ uint32_t *)keys);
    AscendC::GlobalTensor<uint8_t> fg;
    fg.SetGlobalBuffer((__gm__ uint8_t *)flags);
    AscendC::GlobalTensor<uint32_t> bg;
    bg.SetGlobalBuffer((__gm__ uint32_t *)base);
    AscendC::GlobalTensor<uint32_t> rg;
    rg.SetGlobalBuffer((__gm__ uint32_t *)rank);
    AscendC::GlobalTensor<RT> yg;
    yg.SetGlobalBuffer((__gm__ RT *)yBig);

    AscendC::DataCopyExtParams cp1{1, 4, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> pp1{false, 0, 0, 0};
    AscendC::DataCopyPad(one, bg[blk], cp1, pp1);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV5);
    uint32_t mybase = one.GetValue(0);

    int64_t cur = 0;
    for (int64_t off = 0; off < cnt; off += UK_STILE) {
        int64_t len = UkMinI64(cnt - off, UK_STILE);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(len * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(kl, kg[s0 + off], cp, pp);
        AscendC::DataCopyExtParams cpf{1, (uint32_t)len, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> ppf{false, 0, 0, 0};
        AscendC::DataCopyPad(fl, fg[s0 + off], cpf, ppf);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);

        int64_t yc = 0;
        int64_t ystart = 0;
        for (int64_t i = 0; i < len; ++i) {
            uint8_t f = fl.GetValue(i);
            if (off == 0 && i == 0) {
                cur = (f != 0) ? (int64_t)mybase : ((int64_t)mybase - 1);
            } else if (f != 0) {
                cur += 1;
            }
            rl.SetValue(i, (uint32_t)cur);
            if (f != 0) {
                if (yc == 0) {
                    ystart = cur;
                }
                yl.SetValue(yc, (RT)UkDecode<DT>(kl.GetValue(i), mk));
                ++yc;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::DataCopyExtParams cpr{1, (uint32_t)(len * 4), 0, 0, 0};
        AscendC::DataCopyPad(rg[s0 + off], rl, cpr);
        if (yc > 0) {
            AscendC::DataCopyExtParams cpy{1, (uint32_t)(yc * (int64_t)sizeof(RT)), 0, 0, 0};
            AscendC::DataCopyPad(yg[ystart], yl, cpy);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(UK_EV2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(UK_EV2);
    }
}

__global__ __aicore__ void uk_rwiden_kernel(GM_ADDR rankSorted, GM_ADDR inv, int64_t n, int64_t blockLength)
{
    int64_t blk = AscendC::GetBlockIdx();
    int64_t s0 = blk * blockLength;
    int64_t cnt = n - s0;
    if (cnt > blockLength) {
        cnt = blockLength;
    }
    if (cnt < 0) {
        cnt = 0;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> br;
    pipe.InitBuffer(br, UK_STILE * 4 + 256);
    AscendC::TBuf<AscendC::TPosition::VECCALC> bo;
    pipe.InitBuffer(bo, UK_STILE * 8 + 256);
    AscendC::LocalTensor<uint32_t> rl = br.Get<uint32_t>();
    AscendC::LocalTensor<int64_t> ol = bo.Get<int64_t>();

    AscendC::GlobalTensor<uint32_t> rg;
    rg.SetGlobalBuffer((__gm__ uint32_t *)rankSorted);
    AscendC::GlobalTensor<int64_t> ig;
    ig.SetGlobalBuffer((__gm__ int64_t *)inv);

    for (int64_t off = 0; off < cnt; off += UK_STILE) {
        int64_t len = UkMinI64(cnt - off, UK_STILE);
        AscendC::DataCopyExtParams cp{1, (uint32_t)(len * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(rl, rg[s0 + off], cp, pp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(UK_EV0);
        for (int64_t i = 0; i < len; ++i) {
            ol.SetValue(i, (int64_t)rl.GetValue(i));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(UK_EV1);
        AscendC::DataCopyExtParams cpo{1, (uint32_t)(len * 8), 0, 0, 0};
        AscendC::DataCopyPad(ig[s0 + off], ol, cpo);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(UK_EV2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(UK_EV2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(UK_EV0);
    }
}

// ===========================================================================
// host side tiling + launch wrappers
// ===========================================================================

int64_t calc_unique_blocks(int64_t n)
{
    constexpr int64_t MIN_ELEMS_PER_CORE = 8192;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = (plat != nullptr) ? (int64_t)plat->GetCoreNumAiv() : 1;
    if (coreNum <= 0) {
        coreNum = 1;
    }
    int64_t nb = (n + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
    if (nb < 1) {
        nb = 1;
    }
    if (nb > coreNum) {
        nb = coreNum;
    }
    return nb;
}

static inline int64_t UkGrid(int64_t n, int64_t blockLength)
{
    if (blockLength <= 0) {
        return 1;
    }
    int64_t nb = (n + blockLength - 1) / blockLength;
    return (nb < 1) ? 1 : nb;
}

extern "C" {

void ukL_setparams(GM_ADDR params, int64_t minKey, int64_t cap, int64_t mode, int64_t npass, int64_t numBlocks,
                   int64_t blockLength, void* stream)
{
    uk_setparams_kernel<<<1, nullptr, stream>>>(params, minKey, cap, mode, npass, numBlocks, blockLength);
}

void ukL_mm32(GM_ADDR x, GM_ADDR mm, int64_t n, int64_t blockLength, void* stream)
{
    uk_mm_kernel<int32_t><<<UkGrid(n, blockLength), nullptr, stream>>>(x, mm, n, blockLength);
}

void ukL_mm64(GM_ADDR x, GM_ADDR mm, int64_t n, int64_t blockLength, void* stream)
{
    uk_mm_kernel<int64_t><<<UkGrid(n, blockLength), nullptr, stream>>>(x, mm, n, blockLength);
}

void ukL_prep(GM_ADDR mm, GM_ADDR params, int64_t numBlocks, int64_t keyBase, int64_t nv64, void* stream)
{
    uk_prep_kernel<<<1, nullptr, stream>>>(mm, params, numBlocks, nv64);
}

void ukL_dpres(GM_ADDR x, GM_ADDR bitmaps, GM_ADDR params, int64_t n, int64_t blockLength, int64_t dt,
               int64_t minKey, int64_t cap, void* stream)
{
    int64_t nb = UkGrid(n, blockLength);
    switch (dt) {
        case UK_DT_U8:
            uk_dpres_kernel<UK_DT_U8><<<nb, nullptr, stream>>>(x, bitmaps, params, n, blockLength, minKey, cap);
            break;
        case UK_DT_I8:
            uk_dpres_kernel<UK_DT_I8><<<nb, nullptr, stream>>>(x, bitmaps, params, n, blockLength, minKey, cap);
            break;
        case UK_DT_F16:
            uk_dpres_kernel<UK_DT_F16><<<nb, nullptr, stream>>>(x, bitmaps, params, n, blockLength, minKey, cap);
            break;
        case UK_DT_BF16:
            uk_dpres_kernel<UK_DT_BF16><<<nb, nullptr, stream>>>(x, bitmaps, params, n, blockLength, minKey, cap);
            break;
        case UK_DT_I32:
            uk_dpres_kernel<UK_DT_I32><<<nb, nullptr, stream>>>(x, bitmaps, params, n, blockLength, minKey, cap);
            break;
        case UK_DT_I64:
            uk_dpres_kernel<UK_DT_I64><<<nb, nullptr, stream>>>(x, bitmaps, params, n, blockLength, minKey, cap);
            break;
        default:
            uk_dpres_kernel<UK_DT_F32><<<nb, nullptr, stream>>>(x, bitmaps, params, n, blockLength, minKey, cap);
            break;
    }
}

void ukL_dscan(GM_ADDR bitmaps, GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR params, int64_t numBlocks, int64_t cap,
               void* stream)
{
    uk_dscan_kernel<<<1, nullptr, stream>>>(bitmaps, comb, wordPrefix, params, numBlocks, cap);
}

void ukL_dy(GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR yBig, GM_ADDR params, int64_t cap, int64_t numBlocks,
            int64_t dt, int64_t minKey, void* stream)
{
    switch (dt) {
        case UK_DT_U8:
            uk_dy_kernel<UK_DT_U8><<<numBlocks, nullptr, stream>>>(comb, wordPrefix, yBig, params, cap, numBlocks,
                                                                   minKey);
            break;
        case UK_DT_I8:
            uk_dy_kernel<UK_DT_I8><<<numBlocks, nullptr, stream>>>(comb, wordPrefix, yBig, params, cap, numBlocks,
                                                                   minKey);
            break;
        case UK_DT_F16:
            uk_dy_kernel<UK_DT_F16><<<numBlocks, nullptr, stream>>>(comb, wordPrefix, yBig, params, cap, numBlocks,
                                                                    minKey);
            break;
        case UK_DT_BF16:
            uk_dy_kernel<UK_DT_BF16><<<numBlocks, nullptr, stream>>>(comb, wordPrefix, yBig, params, cap, numBlocks,
                                                                     minKey);
            break;
        case UK_DT_I32:
            uk_dy_kernel<UK_DT_I32><<<numBlocks, nullptr, stream>>>(comb, wordPrefix, yBig, params, cap, numBlocks,
                                                                    minKey);
            break;
        case UK_DT_I64:
            uk_dy_kernel<UK_DT_I64><<<numBlocks, nullptr, stream>>>(comb, wordPrefix, yBig, params, cap, numBlocks,
                                                                    minKey);
            break;
        default:
            uk_dy_kernel<UK_DT_F32><<<numBlocks, nullptr, stream>>>(comb, wordPrefix, yBig, params, cap, numBlocks,
                                                                    minKey);
            break;
    }
}

void ukL_dinv(GM_ADDR x, GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR inv, GM_ADDR params, int64_t n,
              int64_t blockLength, int64_t dt, int64_t minKey, int64_t cap, void* stream)
{
    int64_t nb = UkGrid(n, blockLength);
    switch (dt) {
        case UK_DT_U8:
            uk_dinv_kernel<UK_DT_U8><<<nb, nullptr, stream>>>(x, comb, wordPrefix, inv, params, n, blockLength,
                                                              minKey, cap);
            break;
        case UK_DT_I8:
            uk_dinv_kernel<UK_DT_I8><<<nb, nullptr, stream>>>(x, comb, wordPrefix, inv, params, n, blockLength,
                                                              minKey, cap);
            break;
        case UK_DT_F16:
            uk_dinv_kernel<UK_DT_F16><<<nb, nullptr, stream>>>(x, comb, wordPrefix, inv, params, n, blockLength,
                                                               minKey, cap);
            break;
        case UK_DT_BF16:
            uk_dinv_kernel<UK_DT_BF16><<<nb, nullptr, stream>>>(x, comb, wordPrefix, inv, params, n, blockLength,
                                                                minKey, cap);
            break;
        case UK_DT_I32:
            uk_dinv_kernel<UK_DT_I32><<<nb, nullptr, stream>>>(x, comb, wordPrefix, inv, params, n, blockLength,
                                                               minKey, cap);
            break;
        case UK_DT_I64:
            uk_dinv_kernel<UK_DT_I64><<<nb, nullptr, stream>>>(x, comb, wordPrefix, inv, params, n, blockLength,
                                                               minKey, cap);
            break;
        default:
            uk_dinv_kernel<UK_DT_F32><<<nb, nullptr, stream>>>(x, comb, wordPrefix, inv, params, n, blockLength,
                                                               minKey, cap);
            break;
    }
}

void ukL_rpack(GM_ADDR x, GM_ADDR keys, GM_ADDR pay, int64_t n, int64_t blockLength, int64_t dt, int64_t minKey,
               void* stream)
{
    int64_t nb = UkGrid(n, blockLength);
    switch (dt) {
        case UK_DT_I32:
            uk_rpack_kernel<UK_DT_I32><<<nb, nullptr, stream>>>(x, keys, pay, n, blockLength, minKey);
            break;
        case UK_DT_I64:
            uk_rpack_kernel<UK_DT_I64><<<nb, nullptr, stream>>>(x, keys, pay, n, blockLength, minKey);
            break;
        default:
            uk_rpack_kernel<UK_DT_F32><<<nb, nullptr, stream>>>(x, keys, pay, n, blockLength, minKey);
            break;
    }
}

void ukL_rhist(GM_ADDR keys, GM_ADDR hist, int64_t n, int64_t blockLength, int64_t pass, void* stream)
{
    uk_rhist_kernel<<<UkGrid(n, blockLength), nullptr, stream>>>(keys, hist, n, blockLength, pass);
}

void ukL_rscan(GM_ADDR hist, GM_ADDR start, int64_t numBlocks, void* stream)
{
    uk_rscan_kernel<<<1, nullptr, stream>>>(hist, start, numBlocks);
}

void ukL_rscatter(GM_ADDR kIn, GM_ADDR pIn, GM_ADDR kOut, GM_ADDR pOut, GM_ADDR start, int64_t n,
                  int64_t blockLength, int64_t pass, void* stream)
{
    uk_rscatter_kernel<<<UkGrid(n, blockLength), nullptr, stream>>>(kIn, pIn, kOut, pOut, start, n, blockLength,
                                                                    pass);
}

void ukL_rflag(GM_ADDR keys, GM_ADDR flags, GM_ADDR cnts, int64_t n, int64_t blockLength, void* stream)
{
    uk_rflag_kernel<<<UkGrid(n, blockLength), nullptr, stream>>>(keys, flags, cnts, n, blockLength);
}

void ukL_rbase(GM_ADDR cnts, GM_ADDR base, GM_ADDR params, int64_t numBlocks, void* stream)
{
    uk_rbase_kernel<<<1, nullptr, stream>>>(cnts, base, params, numBlocks);
}

void ukL_remit(GM_ADDR keys, GM_ADDR flags, GM_ADDR base, GM_ADDR yBig, GM_ADDR rank, int64_t n,
               int64_t blockLength, int64_t dt, int64_t minKey, void* stream)
{
    int64_t nb = UkGrid(n, blockLength);
    switch (dt) {
        case UK_DT_I32:
            uk_remit_kernel<UK_DT_I32><<<nb, nullptr, stream>>>(keys, flags, base, yBig, rank, n, blockLength,
                                                                minKey);
            break;
        case UK_DT_I64:
            uk_remit_kernel<UK_DT_I64><<<nb, nullptr, stream>>>(keys, flags, base, yBig, rank, n, blockLength,
                                                                minKey);
            break;
        default:
            uk_remit_kernel<UK_DT_F32><<<nb, nullptr, stream>>>(keys, flags, base, yBig, rank, n, blockLength,
                                                                minKey);
            break;
    }
}

void ukL_rwiden(GM_ADDR rankSorted, GM_ADDR inv, int64_t n, int64_t blockLength, void* stream)
{
    uk_rwiden_kernel<<<UkGrid(n, blockLength), nullptr, stream>>>(rankSorted, inv, n, blockLength);
}

} // extern "C"
