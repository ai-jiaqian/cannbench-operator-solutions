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
 * \file dynamic_quant_kernel.cpp
 * \brief DynamicQuant device kernel + host tiling/launch (compiled with bisheng + -xasc)
 *
 * Math (per token = along the last dim):
 *     abs_max  = max_j |x[..., j]|
 *     scaleOut = clamp(abs_max, min = 1e-12) / 127
 *     y        = round(x / scaleOut)                 (int8)
 *
 * Everything is evaluated in float32 (fp16/bf16 -> fp32 is exact) and the
 * rounding is fp32 -> fp16 CAST_RINT followed by fp16 -> int8 CAST_RINT.  The
 * invariant inv = 127 / clamp(abs_max, 1e-12) guarantees |x * inv| <= 127 for
 * every finite x (including the degenerate abs_max < 1e-12 case), so the golden's
 * [-128,127] clamp is redundant and the int8 cast saturates the special values.
 * NaN survives the outer clamps and narrows to 0 in the int8 cast; inf*0 = NaN
 * follows the reference (x/inf -> 0, inf/inf -> NaN -> 0).
 *
 * Layout
 * ------
 * x is viewed as [M, N].  Each core owns a contiguous band of rows and walks it
 * in tiles of R rows.  The UB element pitch P0 = NCp*64 is shared by every dtype
 * so that a flat Cast / Mul keeps every row 32B aligned:
 *     fp16/bf16 in : P0*es bytes   fp32 work : P0 elements
 *     fp16 mid     : P0 elements   int8 out  : P0 bytes
 * NCp is rounded up to a power of two so every fold level halves an even number
 * of chunks.  The DataCopyPad load pads each row block to 32B, so the written
 * prefix of a row is NA = align(N*es,32)/es elements; the 32B dummy fill uses the
 * row's first element (harmless: it is already a member of the row) and the
 * remaining [NA, P0) region is zero, which is the identity for a max over
 * absolute values.  Those buffers are zeroed once before the tile loop.
 *
 * Row reduction (the dominant cost) - folding instead of a full-length reduce:
 * WholeReduceMax/ReduceMax run at 64/7 elements/cycle on DAV_2201, while an
 * elementwise Max runs at 64 elements/cycle.  So each 64-element chunk of a row
 * is first folded against its neighbour with a plain elementwise Max (a stride-2
 * chunk walk that consumes exactly the row pitch), which costs ~1/64 cycle per
 * element instead of ~7/64.  Levels keep halving (in place while the walk stride
 * still fits the 8-bit stride field, then through two small packed scratch
 * regions) until one 64-lane chunk per row remains; a single high-dim
 * WholeReduceMax over those lanes then produces the per-token abs-max.
 *
 * Scaling is vector only (no scalar readback): Brcb expands the per-row
 * abs-maxima into 32B slots of 8 identical lanes, which lets the reciprocal be
 * applied with one high-dim Mul per chunk column or per row, and the same slot
 * layout feeds the 4-byte-per-row scale store.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "dynamic_quant_launch.h"

using namespace AscendC;

namespace {
constexpr int64_t DQ_MAX_R = 64;
constexpr int64_t DQ_SMALL_FLOATS = 2048;
constexpr int64_t DQ_TMP_BYTES = 4096;
constexpr float DQ_EPS = 1e-12f;
constexpr float DQ_INV127 = 1.0f / 127.0f;
constexpr int64_t DQ_INPLACE_MAX_STEP = 4;   // 2*step*8 <= 64 datablocks
} // namespace

__aicore__ inline int64_t DqAlignUp(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

inline int64_t DqAlignUpHost(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

inline int64_t DqNextPow2Host(int64_t v)
{
    int64_t p = 1;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

__aicore__ inline int64_t DqMinI64(int64_t a, int64_t b)
{
    return (a < b) ? a : b;
}

template <typename T, int IND>
__global__ __aicore__ void dynamic_quant_kernel(GM_ADDR xPtr, GM_ADDR yPtr, GM_ADDR sPtr,
                                                int64_t M, int64_t N, int64_t rowsPerCore,
                                                int64_t R, int64_t P0, int64_t NCp, int64_t NA)
{
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t rowStart = blk * rowsPerCore;
    if (rowStart >= M) {
        return;
    }
    int64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > M) {
        rowEnd = M;
    }
    if (N <= 0) {
        return;
    }

    const int64_t es = static_cast<int64_t>(sizeof(T));
    const int64_t tileElems = R * P0;
    const uint32_t inDstStride = static_cast<uint32_t>((P0 * es - DqAlignUp(N * es, 32)) / 32);
    const uint32_t outSrcStride = static_cast<uint32_t>((P0 - DqAlignUp(N, 32)) / 32);
    const bool chunkMul = ((P0 / 8) <= 255);

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int8_t> yGm;
    AscendC::GlobalTensor<float> sGm;
    xGm.SetGlobalBuffer((__gm__ T *)xPtr);
    yGm.SetGlobalBuffer((__gm__ int8_t *)yPtr);
    sGm.SetGlobalBuffer((__gm__ float *)sPtr);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, IND> qIn;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qOut;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qScale;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bF32;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bHalf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bRegA;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bRegB;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bSmall;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bTmp;

    const int64_t regAChunks = (NCp >= 16) ? (NCp / 16) : 1;
    const int64_t regBChunks = (NCp >= 32) ? (NCp / 32) : 1;

    pipe.InitBuffer(qIn, IND, static_cast<uint32_t>(tileElems * es));
    pipe.InitBuffer(qOut, 1, static_cast<uint32_t>(tileElems));
    pipe.InitBuffer(qScale, 2, static_cast<uint32_t>(DQ_MAX_R * 32));
    pipe.InitBuffer(bF32, static_cast<uint32_t>(tileElems * 4));
    pipe.InitBuffer(bHalf, static_cast<uint32_t>(tileElems * 2));
    pipe.InitBuffer(bRegA, static_cast<uint32_t>(R * regAChunks * 64 * 4));
    pipe.InitBuffer(bRegB, static_cast<uint32_t>(R * regBChunks * 64 * 4));
    pipe.InitBuffer(bSmall, static_cast<uint32_t>(DQ_SMALL_FLOATS * 4));
    pipe.InitBuffer(bTmp, static_cast<uint32_t>(DQ_TMP_BYTES));

    AscendC::LocalTensor<float> f32 = bF32.Get<float>();
    AscendC::LocalTensor<half> hf = bHalf.Get<half>();
    AscendC::LocalTensor<float> regA = bRegA.Get<float>();
    AscendC::LocalTensor<float> regB = bRegB.Get<float>();
    AscendC::LocalTensor<float> small = bSmall.Get<float>();
    AscendC::LocalTensor<float> amax = small;
    AscendC::LocalTensor<float> slots = small[128];
    AscendC::LocalTensor<float> invS = small[768];
    AscendC::LocalTensor<float> tmp = bTmp.Get<float>();
    (void)tmp;

    // ---- zero every input buffer once: [NA, P0) of each row must stay 0 ----
    {
        AscendC::LocalTensor<T> bufs[IND];
        for (int i = 0; i < IND; ++i) {
            bufs[i] = qIn.template AllocTensor<T>();
            AscendC::Duplicate(bufs[i], static_cast<T>(0), static_cast<int32_t>(tileElems));
        }
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ev);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ev);
        for (int i = 0; i < IND; ++i) {
            qIn.FreeTensor(bufs[i]);
        }
    }

    for (int64_t rowBase = rowStart; rowBase < rowEnd; rowBase += R) {
        int64_t rn = rowEnd - rowBase;
        if (rn > R) {
            rn = R;
        }
        const int32_t nElems = static_cast<int32_t>(rn * P0);
        const int32_t nRows = static_cast<int32_t>(rn);
        const int64_t rcb = (rn + 7) / 8;
        const int32_t nSlots = static_cast<int32_t>(rcb * 64);

        // ---- load one tile of rn rows (row pitch P0), cast to fp32 ----
        auto inLocal = qIn.template AllocTensor<T>();
        AscendC::DataCopyExtParams cpIn{static_cast<uint16_t>(rn), static_cast<uint32_t>(N * es), 0,
                                        inDstStride, 0};
        AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
        AscendC::DataCopyPad(inLocal, xGm[rowBase * N], cpIn, padIn);
        qIn.EnQue(inLocal);
        inLocal = qIn.template DeQue<T>();

        AscendC::Cast(f32, inLocal, AscendC::RoundMode::CAST_NONE, nElems);

        // ---- per row abs-max by folding ----
        AscendC::Abs(f32, f32, nElems);

        int64_t cnt = NCp;
        int64_t step = 1;
        AscendC::LocalTensor<float> cur = f32;
        bool packed = false;
        while (cnt >= 2 && step <= DQ_INPLACE_MAX_STEP) {
            const int64_t pairs = cnt / 2;
            const uint8_t reps = static_cast<uint8_t>(rn * pairs);
            const uint8_t st = static_cast<uint8_t>(2 * step * 8);
            AscendC::Max(f32, f32, f32[step * 64], 64, reps, {1, 1, 1, st, st, st});
            cnt = pairs;
            step <<= 1;
        }
        if (cnt >= 2) {
            // switch to the packed region: the source walk still uses the tile pitch
            const int64_t pairs = cnt / 2;
            const uint8_t reps = static_cast<uint8_t>(rn * pairs);
            const uint8_t st = static_cast<uint8_t>(2 * step * 8);
            AscendC::Max(regA, f32, f32[step * 64], 64, reps, {1, 1, 1, 8, st, st});
            cur = regA;
            cnt = pairs;
            int64_t which = 1;
            while (cnt >= 2) {
                const int64_t p2 = cnt / 2;
                const uint8_t r2 = static_cast<uint8_t>(rn * p2);
                AscendC::LocalTensor<float> dstR = (which == 1) ? regB : regA;
                AscendC::Max(dstR, cur, cur[64], 64, r2, {1, 1, 1, 8, 16, 16});
                cur = dstR;
                cnt = p2;
                which = 1 - which;
            }
            packed = true;
        }
        const int32_t laneStride = static_cast<int32_t>((packed ? 1 : NCp) * 8);
        AscendC::WholeReduceMax(amax, cur, 64, nRows, 1, 1, laneStride,
                                AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::Maxs(amax, amax, DQ_EPS, nRows);

        // ---- restore the signed values (the fold destroyed the tile) ----
        AscendC::Cast(f32, inLocal, AscendC::RoundMode::CAST_NONE, nElems);
        qIn.FreeTensor(inLocal);

        // ---- per-token scale: one 32B slot per row (8 equal lanes) ----
        AscendC::Brcb(slots, amax, static_cast<uint8_t>(rcb), {1, 8});
        auto scLocal = qScale.AllocTensor<float>();
        AscendC::Muls(scLocal, slots, DQ_INV127, nSlots);
        qScale.EnQue(scLocal);
        scLocal = qScale.DeQue<float>();
        AscendC::DataCopyExtParams cpS{static_cast<uint16_t>(rn), 4u, 0, 0, 0};
        AscendC::DataCopyPad(sGm[rowBase], scLocal, cpS);
        qScale.FreeTensor(scLocal);

        AscendC::Reciprocal(invS, slots, nSlots);
        AscendC::Muls(invS, invS, 127.0f, nSlots);

        // ---- scale every row by its own reciprocal (vector only) ----
        if (chunkMul) {
            const uint8_t rs = static_cast<uint8_t>(P0 / 8);
            for (int64_t c = 0; c < NCp; ++c) {
                AscendC::Mul(f32[c * 64], f32[c * 64], invS, 64, static_cast<uint8_t>(rn),
                             {1, 1, 0, rs, rs, 1});
            }
        } else {
            for (int64_t r = 0; r < rn; ++r) {
                int64_t c0 = 0;
                while (c0 < NCp) {
                    const int64_t cc = DqMinI64(NCp - c0, 200);
                    AscendC::Mul(f32[r * P0 + c0 * 64], f32[r * P0 + c0 * 64], invS[r * 8], 64,
                                 static_cast<uint8_t>(cc), {1, 1, 0, 8, 8, 0});
                    c0 += cc;
                }
            }
        }

        // ---- round (RINT) and narrow to int8 ----
        AscendC::Cast(hf, f32, AscendC::RoundMode::CAST_RINT, nElems);

        auto yLocal = qOut.AllocTensor<int8_t>();
        AscendC::Cast(yLocal, hf, AscendC::RoundMode::CAST_RINT, nElems);
        qOut.EnQue(yLocal);
        yLocal = qOut.DeQue<int8_t>();

        AscendC::DataCopyExtParams cpOut{static_cast<uint16_t>(rn), static_cast<uint32_t>(N),
                                         outSrcStride, 0, 0};
        AscendC::DataCopyPad(yGm[rowBase * N], yLocal, cpOut);
        qOut.FreeTensor(yLocal);
    }
}

// ---------------------------------------------------------------------------
// Host tiling: (numBlocks, rowsPerCore, rowsPerTile, inDepth)
// ---------------------------------------------------------------------------
std::tuple<int64_t, int64_t, int64_t, int64_t> calc_dynamic_quant_tiling(int64_t M, int64_t N)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    const int64_t NC = DqAlignUpHost(N, 64) / 64;
    const int64_t NCp = DqNextPow2Host(NC > 0 ? NC : 1);
    const int64_t P0 = NCp * 64;
    const int64_t regChunks = ((NCp >= 16) ? (NCp / 16) : 1) + ((NCp >= 32) ? (NCp / 32) : 1);
    const int64_t avail = static_cast<int64_t>(ubSize) - 32768;

    // per row: input(2*IND) + fp32(4) + fp16 mid(2) + int8 out(1) + scratch
    int64_t R = 0;
    int64_t inDepth = 2;
    for (int64_t ind = 2; ind >= 1; --ind) {
        const int64_t perRow = P0 * (2 * ind + 4 + 2 + 1) + regChunks * 64 * 4;
        int64_t rr = (avail > 0) ? (avail / perRow) : 1;
        if (rr >= 1) {
            R = std::min<int64_t>(rr, DQ_MAX_R);
            inDepth = ind;
            break;
        }
    }
    if (R < 1) {
        R = 1;
        inDepth = 1;
    }

    int64_t numBlocks = std::min(coreNum, M);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    const int64_t rowsPerCore = (M + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, rowsPerCore, R, inDepth);
}

// ---------------------------------------------------------------------------
// Launch wrappers (regular C functions callable from g++)
// ---------------------------------------------------------------------------
extern "C" {

void launch_dynamic_quant_half(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                               int64_t M, int64_t N, int64_t numBlocks,
                               int64_t rowsPerCore, int64_t rowsPerTile, int64_t inDepth, void *stream)
{
    const int64_t NCp = DqNextPow2Host(DqAlignUpHost(N, 64) / 64);
    const int64_t P0 = NCp * 64;
    const int64_t NA = DqAlignUpHost(N * 2, 32) / 2;
    if (inDepth == 2) {
        dynamic_quant_kernel<half, 2><<<numBlocks, nullptr, stream>>>(x, y, scale, M, N, rowsPerCore,
                                                                     rowsPerTile, P0, NCp, NA);
    } else {
        dynamic_quant_kernel<half, 1><<<numBlocks, nullptr, stream>>>(x, y, scale, M, N, rowsPerCore,
                                                                     rowsPerTile, P0, NCp, NA);
    }
}

void launch_dynamic_quant_bfloat16(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                   int64_t M, int64_t N, int64_t numBlocks,
                                   int64_t rowsPerCore, int64_t rowsPerTile, int64_t inDepth,
                                   void *stream)
{
    const int64_t NCp = DqNextPow2Host(DqAlignUpHost(N, 64) / 64);
    const int64_t P0 = NCp * 64;
    const int64_t NA = DqAlignUpHost(N * 2, 32) / 2;
    if (inDepth == 2) {
        dynamic_quant_kernel<bfloat16_t, 2><<<numBlocks, nullptr, stream>>>(x, y, scale, M, N,
                                                                          rowsPerCore, rowsPerTile,
                                                                          P0, NCp, NA);
    } else {
        dynamic_quant_kernel<bfloat16_t, 1><<<numBlocks, nullptr, stream>>>(x, y, scale, M, N,
                                                                          rowsPerCore, rowsPerTile,
                                                                          P0, NCp, NA);
    }
}

} // extern "C"
