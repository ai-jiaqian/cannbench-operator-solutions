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
 * \file mish_kernel.cpp
 * \brief Mish elementwise kernel + tiling + launch (compiled with bisheng + -xasc)
 *
 * Math (exact, unified, no sign branch):
 *   y = x * tanh(softplus(x)) = x * tanh(log(1 + e^x))
 *   Let s = e^x and E = e^{2 softplus(x)} = (1 + s)^2, then
 *     tanh(softplus(x)) = (E - 1)/(E + 1) = (s^2 + 2s)/(s^2 + 2s + 2)
 *   The exponent is clamped from above: s = exp(min(x, 40)) so that s^2 never
 *   overflows fp32 (s^2 <= 5.54e34).  For x >= ~20 the exact tanh(softplus(x))
 *   rounds to 1.0 in fp32 anyway, so the clamp is lossless for the result.
 *
 *   Special values (matching torch.nn.functional.mish):
 *     x -> -inf : s -> 0, A -> 0, y = (-inf) * 0 = NaN
 *     x -> +inf : A -> 1,        y = +inf
 *     x = NaN   : propagates through the final multiply
 *
 * All arithmetic is performed in fp32.  fp16/bf16 inputs are widened with
 * CAST_NONE and the result is narrowed with CAST_RINT (round half to even).
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 3;
constexpr static int64_t TILE_ALIGN = 64;
// Unified Buffer bytes per element:
//   fp32      : in 3*4 + out 3*4 + 3 fp32 tmp buffers            = 36
//   fp16/bf16 : in 3*2 + out 3*2 + fp32 bufX + 3 fp32 tmp bufs   = 28
constexpr static int64_t BYTES_PER_ELEM_F32 = 36;
constexpr static int64_t BYTES_PER_ELEM_CAST = 28;
constexpr static int64_t UB_RESERVE = 8192;
constexpr static float CLAMP_MAX = 40.0f;
constexpr static int64_t MIN_ELEMS_PER_CORE = 8192;

template <typename T>
struct MishIsFloat {
    static constexpr bool value = false;
};
template <>
struct MishIsFloat<float> {
    static constexpr bool value = true;
};

template <typename T>
__global__ __aicore__ void mish_kernel(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                                       int64_t blockLength, uint32_t tileElems)
{
    constexpr bool kCast = !MishIsFloat<T>::value;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf3;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufX;

    uint32_t tileBytesT = tileElems * static_cast<uint32_t>(sizeof(T));
    uint32_t tileBytesF = tileElems * static_cast<uint32_t>(sizeof(float));
    pipe.InitBuffer(inQueue, PIPELINE_DEPTH, tileBytesT);
    pipe.InitBuffer(outQueue, PIPELINE_DEPTH, tileBytesT);
    pipe.InitBuffer(tmpBuf1, tileBytesF);
    pipe.InitBuffer(tmpBuf2, tileBytesF);
    pipe.InitBuffer(tmpBuf3, tileBytesF);
    if constexpr (kCast) {
        pipe.InitBuffer(bufX, tileBytesF);
    }

    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t start = blockIdx * blockLength;
    if (start >= totalLength) {
        return;
    }
    int64_t curLen = totalLength - start;
    if (curLen > blockLength) {
        curLen = blockLength;
    }

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer((__gm__ T*)x + start, curLen);
    yGm.SetGlobalBuffer((__gm__ T*)y + start, curLen);

    int64_t offset = 0;
    while (offset < curLen) {
        int64_t n = curLen - offset;
        if (n > static_cast<int64_t>(tileElems)) {
            n = static_cast<int64_t>(tileElems);
        }
        uint32_t copyBytes = static_cast<uint32_t>(n * static_cast<int64_t>(sizeof(T)));

        // ---- CopyIn ----
        AscendC::LocalTensor<T> xLocal = inQueue.template AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
        xLocal = inQueue.template DeQue<T>();

        // ---- Compute ----
        AscendC::LocalTensor<T> yLocal = outQueue.template AllocTensor<T>();
        AscendC::LocalTensor<float> t1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> t2 = tmpBuf2.Get<float>();
        AscendC::LocalTensor<float> t3 = tmpBuf3.Get<float>();

        AscendC::LocalTensor<float> xf;
        if constexpr (kCast) {
            xf = bufX.Get<float>();
            AscendC::Cast(xf, xLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
        } else {
            xf = xLocal.template ReinterpretCast<float>();
        }

        int32_t c = static_cast<int32_t>(n);
        // A = tanh(softplus(x)) = (s^2 + 2s) / (s^2 + 2s + 2),  s = exp(min(x, 40))
        AscendC::Mins(t1, xf, CLAMP_MAX, c);   // t1 = min(x, 40)
        AscendC::Exp(t2, t1, c);               // t2 = s
        AscendC::Mul(t1, t2, t2, c);           // t1 = s^2
        AscendC::Axpy(t1, t2, 2.0f, c);        // t1 = num = s^2 + 2s      (fused mul+add)
        AscendC::Adds(t2, t1, 2.0f, c);        // t2 = den = num + 2
        AscendC::Div(t3, t1, t2, c);           // t3 = A

        if constexpr (kCast) {
            // dst for the result must be a VECOUT buffer; reuse t1 (free after Div).
            AscendC::Mul(t1, xf, t3, c);       // t1 = y (fp32)
            AscendC::Cast(yLocal, t1, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(n));
        } else {
            AscendC::LocalTensor<float> yf = yLocal.template ReinterpretCast<float>();
            AscendC::Mul(yf, xf, t3, c);       // y = x * A
        }

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);

        // ---- CopyOut ----
        AscendC::LocalTensor<T> yOut = outQueue.template DeQue<T>();
        AscendC::DataCopyPad(yGm[offset], yOut, copyParams);
        outQueue.FreeTensor(yOut);

        offset += n;
    }
}

// Tiling function (host side, called from the plugin TU).
std::tuple<int64_t, int64_t, uint32_t> calc_mish_tiling_params(int64_t totalLength, bool isCast)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    int64_t numBlocks = 1;
    if (totalLength > 0) {
        numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    }
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    if (blockLength < 1) {
        blockLength = 1;
    }
    // Align the per-block span up to TILE_ALIGN elements so that every block
    // (except possibly the trailing empty ones) starts at a 32-byte aligned GM
    // address and every full tile transfer is aligned.  The kernel early-returns
    // blocks whose start is >= totalLength and clamps each block's length, so
    // rounding up is safe.
    blockLength = (blockLength + TILE_ALIGN - 1) / TILE_ALIGN * TILE_ALIGN;

    int64_t bpe = isCast ? BYTES_PER_ELEM_CAST : BYTES_PER_ELEM_F32;
    int64_t tileElems = (static_cast<int64_t>(ubSize) - UB_RESERVE) / bpe;
    tileElems = tileElems / TILE_ALIGN * TILE_ALIGN;
    if (tileElems < TILE_ALIGN) {
        tileElems = TILE_ALIGN;
    }
    return std::make_tuple(numBlocks, blockLength, static_cast<uint32_t>(tileElems));
}

// Launch wrappers - regular C functions callable from g++
extern "C" {

void launch_mish_kernel_float(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                              int64_t blockLength, uint32_t tileElems, void* stream)
{
    mish_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems);
}

void launch_mish_kernel_half(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                             int64_t blockLength, uint32_t tileElems, void* stream)
{
    mish_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems);
}

void launch_mish_kernel_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                                 int64_t blockLength, uint32_t tileElems, void* stream)
{
    mish_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems);
}

} // extern "C"
