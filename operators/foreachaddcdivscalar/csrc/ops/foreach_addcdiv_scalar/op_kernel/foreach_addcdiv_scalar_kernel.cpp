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
 * \file foreach_addcdiv_scalar_kernel.cpp
 * \brief ForeachAddcdivScalar kernel + tiling + launch (bisheng, -xasc)
 *
 * Per TensorList element i:
 *     y_i = x1_i + (x2_i / x3_i) * scalar
 *
 * Numeric policy (matches task/reference.py):
 *   - float32 : all three steps are carried out in fp32 directly, in the same
 *               order as the golden expression (Div, then Muls, then Add).
 *   - fp16/bf16: the operands are widened to fp32 with Cast(CAST_NONE), the
 *               arithmetic runs in fp32 and the final result is narrowed back to
 *               the input dtype with Cast(CAST_RINT) (round-to-nearest-even),
 *               which is what torch's .to(dtype) does.  This is also mandatory
 *               for bf16 because the vector arithmetic units on dav-2201 do not
 *               accept bf16 operands for Muls/Adds.
 *
 * Everything (including every Cast) happens on device; no host round trip.
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

namespace foreach_addcdiv_scalar_ns {

constexpr int64_t PIPELINE_DEPTH = 2;
constexpr int64_t TILE_ELEM_ALIGN = 64;    // elements: keeps every tile 32B aligned
constexpr int64_t MIN_ELEMS_PER_CORE = 4096;
constexpr int64_t UB_RESERVE_BYTES = 8192;

__aicore__ inline int64_t MinI64(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

}  // namespace foreach_addcdiv_scalar_ns

using namespace foreach_addcdiv_scalar_ns;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void foreach_addcdiv_scalar_kernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                                         int64_t totalLength, int64_t blockLength,
                                                         uint32_t tileElems, float scalar)
{
    constexpr bool IS_F32 = (sizeof(T) == sizeof(float));

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> q1;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> q2;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> q3;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> qy;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch;

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t coreLength = totalLength - blockIdx * blockLength;
    if (coreLength > blockLength) {
        coreLength = blockLength;
    }
    if (coreLength <= 0) {
        return;
    }

    const uint32_t tileBytes = tileElems * static_cast<uint32_t>(sizeof(T));
    pipe.InitBuffer(q1, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(q2, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(q3, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(qy, PIPELINE_DEPTH, tileBytes);
    if constexpr (!IS_F32) {
        // Three fp32 scratch tiles carved out of ONE TBuf: independent TBuf
        // allocations are not reliable on dav-2201, GetWithOffset is.
        pipe.InitBuffer(scratch, 3 * tileElems * static_cast<uint32_t>(sizeof(float)));
    }

    const int64_t base = blockIdx * blockLength;
    AscendC::GlobalTensor<T> g1;
    AscendC::GlobalTensor<T> g2;
    AscendC::GlobalTensor<T> g3;
    AscendC::GlobalTensor<T> gy;
    g1.SetGlobalBuffer((__gm__ T *)x1 + base, coreLength);
    g2.SetGlobalBuffer((__gm__ T *)x2 + base, coreLength);
    g3.SetGlobalBuffer((__gm__ T *)x3 + base, coreLength);
    gy.SetGlobalBuffer((__gm__ T *)y + base, coreLength);

    const uint32_t bytesPerElem = static_cast<uint32_t>(sizeof(T));

    int64_t off = 0;
    while (off < coreLength) {
        const uint32_t n = static_cast<uint32_t>(MinI64(static_cast<int64_t>(tileElems), coreLength - off));
        const uint32_t copyBytes = n * bytesPerElem;

        AscendC::DataCopyExtParams copyInParams{1, copyBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padInParams{false, 0, 0, 0};

        AscendC::LocalTensor<T> t1 = q1.AllocTensor<T>();
        AscendC::LocalTensor<T> t2 = q2.AllocTensor<T>();
        AscendC::LocalTensor<T> t3 = q3.AllocTensor<T>();
        AscendC::DataCopyPad(t1, g1[static_cast<uint32_t>(off)], copyInParams, padInParams);
        AscendC::DataCopyPad(t2, g2[static_cast<uint32_t>(off)], copyInParams, padInParams);
        AscendC::DataCopyPad(t3, g3[static_cast<uint32_t>(off)], copyInParams, padInParams);
        q1.EnQue(t1);
        q2.EnQue(t2);
        q3.EnQue(t3);
        t1 = q1.DeQue<T>();
        t2 = q2.DeQue<T>();
        t3 = q3.DeQue<T>();

        AscendC::LocalTensor<T> ty = qy.AllocTensor<T>();
        if constexpr (IS_F32) {
            // y = x1 + (x2 / x3) * scalar, same operation order as the golden.
            AscendC::Div(ty, t2, t3, static_cast<int32_t>(n));
            AscendC::Muls(ty, ty, scalar, static_cast<int32_t>(n));
            AscendC::Add(ty, t1, ty, static_cast<int32_t>(n));
            q1.FreeTensor(t1);
            q2.FreeTensor(t2);
            q3.FreeTensor(t3);
        } else {
            const int32_t n32 = static_cast<int32_t>(n);
            const int32_t tileCount = static_cast<int32_t>(tileElems);
            const int32_t tileF32Bytes = static_cast<int32_t>(tileElems * static_cast<uint32_t>(sizeof(float)));
            AscendC::LocalTensor<float> f1 = scratch.GetWithOffset<float>(tileCount, 0);
            AscendC::LocalTensor<float> f2 = scratch.GetWithOffset<float>(tileCount, tileF32Bytes);
            AscendC::LocalTensor<float> f3 = scratch.GetWithOffset<float>(tileCount, 2 * tileF32Bytes);

            AscendC::Cast(f2, t2, AscendC::RoundMode::CAST_NONE, n32);
            AscendC::Cast(f3, t3, AscendC::RoundMode::CAST_NONE, n32);
            AscendC::Cast(f1, t1, AscendC::RoundMode::CAST_NONE, n32);
            q1.FreeTensor(t1);
            q2.FreeTensor(t2);
            q3.FreeTensor(t3);

            AscendC::Div(f2, f2, f3, n32);
            AscendC::Muls(f2, f2, scalar, n32);
            AscendC::Add(f1, f1, f2, n32);
            AscendC::Cast(ty, f1, AscendC::RoundMode::CAST_RINT, n32);
        }
        qy.EnQue(ty);

        ty = qy.DeQue<T>();
        AscendC::DataCopyExtParams copyOutParams{1, copyBytes, 0, 0, 0};
        AscendC::DataCopyPad(gy[static_cast<uint32_t>(off)], ty, copyOutParams);
        qy.FreeTensor(ty);

        off += static_cast<int64_t>(n);
    }
}

// ---------------------------------------------------------------------------
// Tiling
// ---------------------------------------------------------------------------
std::tuple<int64_t, int64_t, int64_t> calc_foreach_addcdiv_scalar_tiling_params(int64_t totalLength,
                                                                                int64_t dtypeSize)
{
    if (totalLength <= 0) {
        return std::make_tuple(static_cast<int64_t>(1), static_cast<int64_t>(0), TILE_ELEM_ALIGN);
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    // UB budget per tile, in bytes: 4 queues x PIPELINE_DEPTH x dtypeSize, plus
    // (for 2-byte dtypes) 3 fp32 scratch tiles.
    const bool lowPrecision = (dtypeSize != 4);
    int64_t bytesPerElemPerTile = PIPELINE_DEPTH * 4 * dtypeSize;
    if (lowPrecision) {
        bytesPerElemPerTile += 3 * static_cast<int64_t>(sizeof(float));
    }
    int64_t ubBytes = static_cast<int64_t>(ubSize);
    if (ubBytes <= UB_RESERVE_BYTES) {
        ubBytes = UB_RESERVE_BYTES;
    }
    int64_t tileElems = (ubBytes - UB_RESERVE_BYTES) / bytesPerElemPerTile;
    tileElems = (tileElems / TILE_ELEM_ALIGN) * TILE_ELEM_ALIGN;
    if (tileElems < TILE_ELEM_ALIGN) {
        tileElems = TILE_ELEM_ALIGN;
    }

    // Core split: cap the number of cores by a minimum amount of work per core,
    // then re-derive the grid so the (aligned) block length covers the tensor.
    int64_t numBlocks = (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
    numBlocks = std::min(numBlocks, coreNum);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    blockLength = ((blockLength + TILE_ELEM_ALIGN - 1) / TILE_ELEM_ALIGN) * TILE_ELEM_ALIGN;
    numBlocks = (totalLength + blockLength - 1) / blockLength;
    if (numBlocks < 1) {
        numBlocks = 1;
    }

    return std::make_tuple(numBlocks, blockLength, tileElems);
}

// ---------------------------------------------------------------------------
// Launch wrappers (plain C symbols, called from the g++ plugin)
// ---------------------------------------------------------------------------
extern "C" {

void launch_foreach_addcdiv_scalar_float(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                         int64_t totalLength, int64_t numBlocks, int64_t blockLength,
                                         uint32_t tileElems, float scalar, void *stream)
{
    foreach_addcdiv_scalar_kernel<float><<<numBlocks, nullptr, stream>>>(x1, x2, x3, y, totalLength,
                                                                        blockLength, tileElems, scalar);
}

void launch_foreach_addcdiv_scalar_half(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                        int64_t totalLength, int64_t numBlocks, int64_t blockLength,
                                        uint32_t tileElems, float scalar, void *stream)
{
    foreach_addcdiv_scalar_kernel<half><<<numBlocks, nullptr, stream>>>(x1, x2, x3, y, totalLength,
                                                                       blockLength, tileElems, scalar);
}

void launch_foreach_addcdiv_scalar_bfloat16(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                            int64_t totalLength, int64_t numBlocks, int64_t blockLength,
                                            uint32_t tileElems, float scalar, void *stream)
{
    foreach_addcdiv_scalar_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x1, x2, x3, y, totalLength,
                                                                             blockLength, tileElems, scalar);
}

}  // extern "C"
