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
 * \file sigmoid_kernel.cpp
 * \brief Sigmoid host code - tiling + kernel launch (compiled with bisheng + -xasc)
 *
 *   y = 1 / (1 + e^(-x))
 *
 * Elementwise over the flat 1D view of the (contiguous) ND tensor.  Every AIV core owns a
 * contiguous element range and walks it in double buffered GM<->UB tiles.
 *
 * Compute precision per dtype:
 *   float32   : native fp32 chain.
 *   float16   : native HALF chain (Muls/Exp/Adds/Div all in half, no upcast).  The op is
 *               vector-pipe limited and a half instruction covers 128 elements per repeat vs
 *               64 for fp32.
 *   bfloat16  : there is no native bf16 vector math on this target, so the input is widened
 *               exactly to fp32, the fp32 chain runs, and the result is narrowed with CAST_RINT.
 *
 * The reciprocal step uses Div against an all-ones vector for float32/float16 on purpose:
 * AscendC::Reciprocal is a low precision table based approximation and would break those accuracy
 * thresholds. bfloat16 (threshold 2^-7) has enough tolerance for it and uses Reciprocal, which also
 * frees the all-ones operand so its DMA tile can be 25% larger.
 *
 * Special values (identical to torch.sigmoid):
 *   x = +inf  ->  e^(-x) = 0      ->  y = 1
 *   x = -inf  ->  e^(-x) = +inf   ->  y = 0
 *   x = 0     ->  e^(0)   = 1     ->  y = 0.5
 *   x = NaN   ->  NaN propagates
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "sigmoid_launch.h"

using namespace AscendC;

constexpr static int64_t SIGMOID_PIPELINE_DEPTH = 2;
// Smallest workload that justifies an extra core (keeps tiny tensors off the launch path).
constexpr static int64_t SIGMOID_MIN_ELEMS_PER_CORE = 8192;

/*!
 * \brief per dtype compute selection.
 *   CalcT   : type the vector chain runs in (16-bit types cover 128 elements / repeat)
 *   kStage  : input has no 16-bit vector math -> widen to fp32, compute, narrow back
 *   kCalcSize : sizeof(CalcT) needed by the host side UB budget
 */
template <typename T>
struct SigmoidTraits {
    using CalcT = float;
    static constexpr bool kStage = false;
    static constexpr bool kReciprocal = false;
    static constexpr int64_t kCalcSize = 4;
};

template <>
struct SigmoidTraits<half> {
    using CalcT = half;
    static constexpr bool kStage = false;
    static constexpr bool kReciprocal = false;
    static constexpr int64_t kCalcSize = 2;
};

template <>
struct SigmoidTraits<bfloat16_t> {
    using CalcT = float;
    static constexpr bool kStage = true;
    // bf16 tolerance (2^-7) comfortably absorbs the ~1e-3 relative error of the hardware
    // reciprocal, and dropping the resident all-ones operand frees enough UB for a 25% larger
    // DMA tile. fp16/fp32 thresholds are far too tight for that trade and keep the exact Div.
    static constexpr bool kReciprocal = true;
    static constexpr int64_t kCalcSize = 4;
};

/*!
 * \brief process one tile: GM->UB, sigmoid, UB->GM.
 *
 * Every vector op writes a destination distinct from its operands (c0, c1 and the queue
 * tensors are never read and written in place), so c0 and c1 alternate roles instead of
 * aliasing each other.
 * \param aligned  true when this is a whole tile and every participating address is 32B (in fact
 *                512B) aligned, which lets the plain burst DataCopy fast path run; the sub-tile
 *                 tail (and a short core) takes the DataCopyPad path instead.
 */
template <typename T>
__aicore__ inline void SigmoidTile(
    TQue<QuePosition::VECIN, SIGMOID_PIPELINE_DEPTH>& inQueue,
    TQue<QuePosition::VECOUT, SIGMOID_PIPELINE_DEPTH>& outQueue,
    const LocalTensor<typename SigmoidTraits<T>::CalcT>& c0,
    const LocalTensor<typename SigmoidTraits<T>::CalcT>& c1,
    const LocalTensor<typename SigmoidTraits<T>::CalcT>& ones,
    const GlobalTensor<T>& xGm,
    const GlobalTensor<T>& yGm,
    int64_t offset,
    int64_t count,
    bool aligned)
{
    using CalcT = typename SigmoidTraits<T>::CalcT;
    if (count <= 0) {
        return;
    }
    const int32_t n = static_cast<int32_t>(count);
    const uint32_t byteLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
    DataCopyExtParams copyParams{1, byteLen, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // ---- copy in ----
    LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
    if (aligned) {
        DataCopy(xLocal, xGm[offset], static_cast<uint32_t>(count));
    } else {
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
    }
    inQueue.EnQue(xLocal);
    xLocal = inQueue.DeQue<T>();

    LocalTensor<T> yLocal = outQueue.AllocTensor<T>();

    // ---- c1 = -x (widening exactly to fp32 first for 16-bit inputs without vector math) ----
    if constexpr (SigmoidTraits<T>::kStage) {
        Cast<float, T>(c0, xLocal, RoundMode::CAST_NONE, n);
        Muls(c1, c0, static_cast<float>(-1.0), n);
    } else {
        Muls(c1, xLocal, static_cast<CalcT>(-1.0), n);
    }

    // ---- c0 = e^(-x); c1 = 1 + e^(-x); y = 1 / c1 ----
    Exp(c0, c1, n);
    Adds(c1, c0, static_cast<CalcT>(1.0), n);
    if constexpr (SigmoidTraits<T>::kStage) {
        if constexpr (SigmoidTraits<T>::kReciprocal) {
            Reciprocal(c0, c1, n);
        } else {
            Div(c0, ones, c1, n);
        }
        Cast<T, float>(yLocal, c0, RoundMode::CAST_RINT, n);
    } else {
        if constexpr (SigmoidTraits<T>::kReciprocal) {
            Reciprocal(yLocal, c1, n);
        } else {
            Div(yLocal, ones, c1, n);
        }
    }

    inQueue.FreeTensor(xLocal);
    outQueue.EnQue(yLocal);
    yLocal = outQueue.DeQue<T>();
    if (aligned) {
        DataCopy(yGm[offset], yLocal, static_cast<uint32_t>(count));
    } else {
        DataCopyPad(yGm[offset], yLocal, copyParams);
    }
    outQueue.FreeTensor(yLocal);
}

template <typename T>
__global__ __aicore__ void sigmoid_kernel(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t blockLength,
                                          uint32_t tileElems)
{
    using CalcT = typename SigmoidTraits<T>::CalcT;
    TPipe pipe;
    TQue<QuePosition::VECIN, SIGMOID_PIPELINE_DEPTH> inQueue;
    TQue<QuePosition::VECOUT, SIGMOID_PIPELINE_DEPTH> outQueue;
    TBuf<TPosition::VECCALC> calcBuf;

    pipe.InitBuffer(inQueue, SIGMOID_PIPELINE_DEPTH, tileElems * sizeof(T));
    pipe.InitBuffer(outQueue, SIGMOID_PIPELINE_DEPTH, tileElems * sizeof(T));
    constexpr int64_t kCalcBuffers = SigmoidTraits<T>::kReciprocal ? 2 : 3;
    pipe.InitBuffer(calcBuf, kCalcBuffers * static_cast<int64_t>(tileElems) * static_cast<int64_t>(sizeof(CalcT)));

    LocalTensor<CalcT> calc = calcBuf.Get<CalcT>();
    LocalTensor<CalcT> c0 = calc;
    LocalTensor<CalcT> c1 = calc[tileElems];
    LocalTensor<CalcT> ones = c0;  // placeholder: the reciprocal path has no all-ones operand
    if constexpr (!SigmoidTraits<T>::kReciprocal) {
        ones = calc[2 * static_cast<int64_t>(tileElems)];
        Duplicate(ones, static_cast<CalcT>(1.0), static_cast<int32_t>(tileElems));
    }

    int64_t blockStart = blockLength * static_cast<int64_t>(GetBlockIdx());
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockStart);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockStart);

    int64_t currentLength = totalLength - blockStart;
    if (currentLength > blockLength) {
        currentLength = blockLength;
    }
    if (currentLength <= 0) {
        return;
    }

    const int64_t tileElemsI = static_cast<int64_t>(tileElems);
    const int64_t fullTiles = currentLength / tileElemsI;
    const int64_t tailLength = currentLength - fullTiles * tileElemsI;

    for (int64_t i = 0; i < fullTiles; ++i) {
        SigmoidTile<T>(inQueue, outQueue, c0, c1, ones, xGm, yGm, i * tileElemsI, tileElemsI, true);
    }
    if (tailLength > 0) {
        SigmoidTile<T>(inQueue, outQueue, c0, c1, ones, xGm, yGm, fullTiles * tileElemsI, tailLength, false);
    }
}

/*!
 * \brief host side tiling: returns (numBlocks, blockLength, tileElems)
 *
 * Per-element UB footprint: 2 queues * sizeof(T) [double buffered] * 2 (in+out)
 *                          + calcBuffers * calcSize (c0, c1, and an all-ones operand for the dtypes
 *                            that use the exact Div instead of the hardware Reciprocal).
 * Keep a 15% reserve so queue/pipe bookkeeping can never push us past the UB limit.
 *
 * The tile is additionally floored at 16 KB worth of elements so every MTE2 burst stays
 * at the size where GM<->UB bandwidth is fully efficient (below ~16 KB the per-transfer
 * overhead eats a measurable fraction of the bandwidth).
 */
std::tuple<int64_t, int64_t, int64_t> calc_sigmoid_tiling_params(int64_t totalLength, int64_t mode)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    int64_t numBlocks = std::min(coreNum, (totalLength + SIGMOID_MIN_ELEMS_PER_CORE - 1) / SIGMOID_MIN_ELEMS_PER_CORE);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    // Every core owns a whole number of 256-element chunks, so each core's base element offset (and
    // therefore every full-tile burst inside it) lands on a 512B/1024B boundary. 512B-aligned GM
    // addresses transfer at a markedly higher rate than merely 32B-aligned ones.
    constexpr int64_t kBlockAlignElems = 256;
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    blockLength = (blockLength + kBlockAlignElems - 1) / kBlockAlignElems * kBlockAlignElems;
    numBlocks = (totalLength + blockLength - 1) / blockLength;
    if (numBlocks < 1) {
        numBlocks = 1;
    }

    const int64_t typeSize = (mode == 0) ? 4 : 2;
    const int64_t calcSize = (mode == 1) ? 2 : 4;
    // bfloat16 (mode 2) uses the hardware Reciprocal, so it keeps only c0/c1; float/half keep the
    // resident all-ones operand for the exact Div.
    const int64_t calcBuffers = (mode == 2) ? 2 : 3;
    const int64_t bytesPerElem = 4 * typeSize + calcBuffers * calcSize;

    int64_t tileElems = static_cast<int64_t>(ubSize) * 17 / 20 / bytesPerElem;
    tileElems = tileElems / 256 * 256;
    // Floor the burst at 16 KB worth of elements (32B aligned for both 16-bit and fp32).
    const int64_t minTileElems = (16384 + typeSize - 1) / typeSize;
    if (tileElems < minTileElems) {
        tileElems = minTileElems;
    }
    if (tileElems < 256) {
        tileElems = 256;
    }
    if (tileElems > blockLength) {
        tileElems = (blockLength + 255) / 256 * 256;
    }
    if (tileElems < 256) {
        tileElems = 256;
    }
    return std::make_tuple(numBlocks, blockLength, tileElems);
}

// Launch wrappers - regular C functions callable from g++
extern "C" {

void launch_sigmoid_kernel_float(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                                 int64_t blockLength, uint32_t tileElems, void* stream)
{
    sigmoid_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems);
}

void launch_sigmoid_kernel_half(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                                int64_t blockLength, uint32_t tileElems, void* stream)
{
    sigmoid_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems);
}

void launch_sigmoid_kernel_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                                    int64_t blockLength, uint32_t tileElems, void* stream)
{
    sigmoid_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems);
}

} // extern "C"
