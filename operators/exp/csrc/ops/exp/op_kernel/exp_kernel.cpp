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
 * \file exp_kernel.cpp
 * \brief Exp device kernel + host tiling/launch (compiled with bisheng + -xasc).
 *
 * Reference semantics (task/reference.py):
 *     compute_dtype = float32 for float16/bfloat16 inputs, otherwise the input dtype
 *     t = scale * x + shift
 *     if (base > 0) t = t * ln(base)
 *     y = exp(t)
 *     y = y cast back to the input dtype
 *
 * The scalar ln(base) factor is folded on the host into the affine constants
 *     lnb  = (base > 0) ? ln(base) : 1.0
 *     coefA = scale * lnb,  coefB = shift * lnb
 * so the device only issues two vector ops before the exponential:
 *     t = coefA * x + coefB  (fp32)
 *     y = exp(t)
 * For base == 1, lnb == 0 exactly, so coefA == coefB == 0 and the result is
 * exactly exp(0) == 1, matching the golden bit for bit.
 *
 * Layout: flat 1D view of the contiguous ND tensor (elementwise op, so shape
 * only affects the element count). The tile axis is split across cores at
 * *whole tile* granularity, so every core's base offset is tile-aligned and
 * the bulk of the data moves with plain (aligned) DataCopy. Only the trailing
 * sub-tile uses DataCopyPad.
 *
 * AscendC::Exp supports half/float only, so bfloat16 always goes through the
 * fp32 intermediate. For float16/bfloat16 the whole computation runs in fp32
 * and the single final rounding back to the input dtype uses CAST_RINT
 * (round-half-to-even), which is what torch's `.to(dtype)` does.
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

using namespace AscendC;

// Double buffering of the GM <-> UB copies (1 tile in flight while 1 computes).
constexpr static int64_t PIPELINE_DEPTH = 2;
// Kept free so TQue/TBuf bookkeeping never overflows the Unified Buffer.
constexpr static int64_t UB_SAFETY_MARGIN = 16 * 1024;
// Every tile holds a whole number of 256-element units, so every tile byte
// length (>= 512 B) is a multiple of the 32 B DataCopy granularity.
constexpr static int64_t ELEM_ALIGN = 256;

/*!
 * \brief One tile: x -> coefA*x + coefB -> exp -> y.
 * \note  Exp forbids src/dst overlap, so the fp32 intermediates always live in
 *        the dedicated VECCALC scratch buffers f1 (affine) and f2 (exp result
 *        for the 16-bit dtypes).
 *
 *        The two identity cases of the affine map are elided because they are
 *        EXACT for every input, including the special values:
 *          - coefA == 1.0f: 1*x is bit-identical to x (1*inf == inf, 1*nan == nan).
 *          - coefB == 0.0f: x + 0.0f differs from x only for x == -0.0f, and
 *            exp(-0.0f) == exp(+0.0f) == 1.0f.
 *        coefA == 0.0f (base == 1) is deliberately NOT elided: 0*x is what turns
 *        an inf/nan input into nan, exactly like the reference, so the multiply
 *        has to stay.
 */
template <typename T>
__aicore__ inline void ExpTileCompute(const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal,
                                      LocalTensor<float> &f1, LocalTensor<float> &f2,
                                      int32_t count, float coefA, float coefB)
{
    if constexpr (sizeof(T) == 2) {
        // fp16/bf16 -> fp32 is exact; compute in fp32; round back half-to-even.
        Cast(f1, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        if (coefA != 1.0f) {
            Muls(f1, f1, coefA, count);
        }
        if (coefB != 0.0f) {
            Adds(f1, f1, coefB, count);
        }
        Exp(f2, f1, count);
        Cast(yLocal, f2, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
    } else {
        if (coefA == 1.0f && coefB == 0.0f) {
            // Pure Exp path: the affine map is the identity.
            Exp(yLocal, xLocal, count);
        } else {
            if (coefA == 1.0f) {
                Adds(f1, xLocal, coefB, count);
            } else if (coefB == 0.0f) {
                Muls(f1, xLocal, coefA, count);
            } else {
                Muls(f1, xLocal, coefA, count);
                Adds(f1, f1, coefB, count);
            }
            Exp(yLocal, f1, count);
        }
    }
}

template <typename T>
__global__ __aicore__ void exp_kernel(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                                      int64_t tilesPerCore, uint32_t tileElems, float coefA, float coefB)
{
    const int64_t blockIdx = AscendC::GetBlockIdx();
    const int64_t tileElemsI = static_cast<int64_t>(tileElems);
    const int64_t start = blockIdx * tilesPerCore * tileElemsI;
    if (start >= totalLength) {
        return;
    }
    int64_t remaining = totalLength - start;
    int64_t coreLength = tilesPerCore * tileElemsI;
    if (coreLength > remaining) {
        coreLength = remaining;
    }
    // The last launched core also absorbs the (< 1 tile) tail of the tensor.
    if (blockIdx == numBlocks - 1) {
        coreLength = remaining;
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueue;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueue;
    TBuf<QuePosition::VECCALC> calcBuf;

    const uint32_t tileBytes = tileElems * static_cast<uint32_t>(sizeof(T));
    pipe.InitBuffer(inQueue, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(outQueue, PIPELINE_DEPTH, tileBytes);
    // Deliberately unconditional (2 * tileElems floats) for every dtype, so the
    // UB layout does not depend on T.
    pipe.InitBuffer(calcBuf, 2u * tileElems * static_cast<uint32_t>(sizeof(float)));

    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> f1 = calc;
    LocalTensor<float> f2 = calc[static_cast<int32_t>(tileElems)];

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) + start);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + start);

    const int64_t fullTiles = coreLength / tileElemsI;
    const int64_t tailElems = coreLength - fullTiles * tileElemsI;

    // ---- bulk: aligned, contiguous, plain DataCopy (fast path) ----
    for (int64_t i = 0; i < fullTiles; ++i) {
        const int64_t offset = i * tileElemsI;

        auto xLocal = inQueue.AllocTensor<T>();
        DataCopy(xLocal, xGm[offset], tileElems);
        inQueue.EnQue(xLocal);
        xLocal = inQueue.DeQue<T>();

        auto yLocal = outQueue.AllocTensor<T>();
        ExpTileCompute<T>(yLocal, xLocal, f1, f2, static_cast<int32_t>(tileElems), coefA, coefB);
        inQueue.FreeTensor(xLocal);

        outQueue.EnQue(yLocal);
        yLocal = outQueue.DeQue<T>();
        DataCopy(yGm[offset], yLocal, tileElems);
        outQueue.FreeTensor(yLocal);
    }

    // ---- tail: < 1 tile, possibly unaligned -> DataCopyPad ----
    if (tailElems > 0) {
        const int64_t offset = fullTiles * tileElemsI;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailElems * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        auto xLocal = inQueue.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
        xLocal = inQueue.DeQue<T>();

        auto yLocal = outQueue.AllocTensor<T>();
        ExpTileCompute<T>(yLocal, xLocal, f1, f2, static_cast<int32_t>(tailElems), coefA, coefB);
        inQueue.FreeTensor(xLocal);

        outQueue.EnQue(yLocal);
        yLocal = outQueue.DeQue<T>();
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }
}

/*!
 * \brief Host tiling. Returns (numBlocks, tilesPerCore, tileElems).
 *
 * UB live bytes per element are 2*elemSize (VECIN) + 2*elemSize (VECOUT) +
 * 8 bytes (two fp32 scratch buffers), all multiplied by the tile length.
 */
std::tuple<int64_t, int64_t, uint32_t> calc_exp_tiling_params(int64_t totalLength, int64_t elemSize)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 4096;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    const int64_t bytesPerElem = 2 * elemSize + 2 * elemSize + 2 * 4;
    int64_t tileElems = (static_cast<int64_t>(ubSize) - UB_SAFETY_MARGIN) / bytesPerElem;
    tileElems = tileElems / ELEM_ALIGN * ELEM_ALIGN;
    if (tileElems < ELEM_ALIGN) {
        tileElems = ELEM_ALIGN;
    }

    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    if (numBlocks <= 0) {
        numBlocks = 1;
    }
    const int64_t totalTiles = (totalLength + tileElems - 1) / tileElems;
    int64_t tilesPerCore = (totalTiles + numBlocks - 1) / numBlocks;
    if (tilesPerCore <= 0) {
        tilesPerCore = 1;
    }
    return std::make_tuple(numBlocks, tilesPerCore, static_cast<uint32_t>(tileElems));
}

// Launch wrappers - plain C functions callable from the g++ plugin TU.
extern "C" {

void launch_exp_kernel_float(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                             int64_t tilesPerCore, uint32_t tileElems, float coefA, float coefB, void* stream)
{
    exp_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, totalLength, numBlocks, tilesPerCore, tileElems,
                                                      coefA, coefB);
}

void launch_exp_kernel_half(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                            int64_t tilesPerCore, uint32_t tileElems, float coefA, float coefB, void* stream)
{
    exp_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, totalLength, numBlocks, tilesPerCore, tileElems,
                                                     coefA, coefB);
}

void launch_exp_kernel_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                                int64_t tilesPerCore, uint32_t tileElems, float coefA, float coefB, void* stream)
{
    exp_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, totalLength, numBlocks, tilesPerCore, tileElems,
                                                           coefA, coefB);
}

} // extern "C"
