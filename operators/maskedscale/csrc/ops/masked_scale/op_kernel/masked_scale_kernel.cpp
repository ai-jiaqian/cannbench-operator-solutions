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
 * \file masked_scale_kernel.cpp
 * \brief MaskedScale device kernel + host tiling/launch (bisheng / -xasc).
 *
 * Math: y = x * mask * scale, output dtype == x dtype.
 *
 * Strategy
 * --------
 * Every (x, mask) dtype combination is evaluated in float32 (the widest supported
 * type) and the result is rounded to the x dtype exactly once at the end:
 *   - fp16 / bf16 products are computed in fp32 and rounded once, which stays far
 *     inside the precision budget of the reference (x * mask * scale).to(x.dtype).
 *   - int8 / uint8 masks are exactly representable, so casting them to fp32 is
 *     lossless.
 *
 * int8_t / uint8_t have no direct -> float Cast on DAV_2201, therefore they are
 * routed through an fp16 scratch (exact for the whole 8-bit range) before float.
 *
 * The tensor is flattened and each core owns one contiguous block
 * (blockLength = ceil(totalLength / numBlocks) rounded up to MS_TILE_ALIGN); the
 * tail is simply the last loop iteration and is handled by DataCopyPad, which
 * also covers every unaligned or non-power-of-two shape in the case catalog.
 *
 * Bandwidth notes (Ascend 910B / A2):
 *   - Both the per-core block length and the per-tile length are multiples of
 *     MS_TILE_ALIGN = 512 elements. Because the torch base pointers are >=512B
 *     aligned, this makes every x / mask / y per-tile GM address 512B-aligned.
 *     A2 transfers gain a large amount of bandwidth from 512B-aligned GM
 *     addresses versus 32B-aligned ones, and observed per-case time improved on
 *     every case when this alignment was introduced.
 *   - All TQue/TBuf allocations are unconditional, but the scratch buffers that
 *     a given (Tx, Tm) instantiation does not need are sized to a harmless
 *     minimum instead of a full tile, so every dtype pair fits the same
 *     per-element UB budget and therefore the same (large) tile.
 */

#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

using namespace AscendC;

constexpr static int64_t MS_PIPELINE_DEPTH = 2;
// Headroom for queue bookkeeping; the scratch-buffer accounting is exact.
constexpr static int64_t MS_UB_SAFETY_MARGIN = 16 * 1024;
// Element granularity of both the per-core block and the per-tile length: keeps
// every per-tile GM offset a multiple of 512 * sizeof(element) >= 512 bytes.
constexpr static int64_t MS_TILE_ALIGN = 512;
// Minimum byte size used for a scratch buffer that an instantiation never touches.
constexpr static uint32_t MS_MIN_SCRATCH_BYTES = 32;

/*!
 * \brief Convert a mask tile to float32.
 * int8_t / uint8_t are staged through half (exact for the full 8-bit range) because
 * DAV_2201 has no direct 8-bit integer -> float Cast.
 */
template <typename Tm>
__aicore__ inline void MsMaskToF32(const LocalTensor<float>& dst,
                                   const LocalTensor<Tm>& src,
                                   const LocalTensor<half>& scratch,
                                   int32_t count)
{
    if constexpr (std::is_same<Tm, int8_t>::value || std::is_same<Tm, uint8_t>::value) {
        Cast(scratch, src, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        Cast(dst, scratch, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
    }
}

/*!
 * \brief MaskedScale kernel - flattened 1D elementwise over one contiguous block per core.
 */
template <typename Tx, typename Tm>
__global__ __aicore__ void masked_scale_kernel(GM_ADDR x, GM_ADDR mask, GM_ADDR y,
                                               int64_t totalLength, int64_t blockLength,
                                               uint32_t tileElems, float scale)
{
    constexpr bool X_IS_F32 = std::is_same<Tx, float>::value;
    constexpr bool M_IS_F32 = std::is_same<Tm, float>::value;
    constexpr bool M_IS_8BIT = std::is_same<Tm, int8_t>::value || std::is_same<Tm, uint8_t>::value;

    TPipe pipe;
    GlobalTensor<Tx> xGm;
    GlobalTensor<Tm> mGm;
    GlobalTensor<Tx> yGm;
    TQue<QuePosition::VECIN, MS_PIPELINE_DEPTH> qX;
    TQue<QuePosition::VECIN, MS_PIPELINE_DEPTH> qM;
    TQue<QuePosition::VECOUT, MS_PIPELINE_DEPTH> qY;
    TBuf<TPosition::VECCALC> bX;
    TBuf<TPosition::VECCALC> bM;
    TBuf<TPosition::VECCALC> bS;
    TBuf<TPosition::VECCALC> bY;

    const uint32_t tileBytesX = static_cast<uint32_t>(tileElems * sizeof(Tx));
    const uint32_t tileBytesM = static_cast<uint32_t>(tileElems * sizeof(Tm));
    const uint32_t tileBytesF = static_cast<uint32_t>(tileElems * sizeof(float));
    const uint32_t tileBytesH = static_cast<uint32_t>(tileElems * sizeof(half));
    // Unused scratch buffers are allocated min-size, keeping the InitBuffer calls
    // themselves unconditional for every dtype pair.
    const uint32_t bufXBytes = X_IS_F32 ? MS_MIN_SCRATCH_BYTES : tileBytesF;
    const uint32_t bufYBytes = X_IS_F32 ? MS_MIN_SCRATCH_BYTES : tileBytesF;
    const uint32_t bufMBytes = M_IS_F32 ? MS_MIN_SCRATCH_BYTES : tileBytesF;
    const uint32_t bufSBytes = M_IS_8BIT ? tileBytesH : MS_MIN_SCRATCH_BYTES;

    pipe.InitBuffer(qX, MS_PIPELINE_DEPTH, tileBytesX);
    pipe.InitBuffer(qM, MS_PIPELINE_DEPTH, tileBytesM);
    pipe.InitBuffer(qY, MS_PIPELINE_DEPTH, tileBytesX);
    pipe.InitBuffer(bX, bufXBytes);
    pipe.InitBuffer(bM, bufMBytes);
    pipe.InitBuffer(bS, bufSBytes);
    pipe.InitBuffer(bY, bufYBytes);

    const int64_t start = static_cast<int64_t>(GetBlockIdx()) * blockLength;
    int64_t currentLen = totalLength - start;
    if (currentLen > blockLength) currentLen = blockLength;
    if (currentLen <= 0) return;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ Tx*>(x) + start);
    mGm.SetGlobalBuffer(reinterpret_cast<__gm__ Tm*>(mask) + start);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ Tx*>(y) + start);

    const int64_t tile = static_cast<int64_t>(tileElems);
    const int64_t numIters = (currentLen + tile - 1) / tile;

    for (int64_t i = 0; i < numIters; ++i) {
        const int64_t off = i * tile;
        int64_t cnt = currentLen - off;
        if (cnt > tile) cnt = tile;
        const int32_t n = static_cast<int32_t>(cnt);

        DataCopyExtParams cpx{1, static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(Tx))), 0, 0, 0};
        DataCopyExtParams cpm{1, static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(Tm))), 0, 0, 0};
        DataCopyPadExtParams<Tx> padx{false, 0, 0, 0};
        DataCopyPadExtParams<Tm> padm{false, 0, 0, 0};

        auto xL = qX.AllocTensor<Tx>();
        auto mL = qM.AllocTensor<Tm>();
        DataCopyPad(xL, xGm[off], cpx, padx);
        DataCopyPad(mL, mGm[off], cpm, padm);
        qX.EnQue(xL);
        qM.EnQue(mL);
        xL = qX.DeQue<Tx>();
        mL = qM.DeQue<Tm>();

        auto yL = qY.AllocTensor<Tx>();

        LocalTensor<float> maskF;
        if constexpr (M_IS_F32) {
            maskF = mL;
        } else {
            maskF = bM.Get<float>();
            MsMaskToF32<Tm>(maskF, mL, bS.Get<half>(), n);
        }

        if constexpr (X_IS_F32) {
            Mul(yL, xL, maskF, n);
            Muls(yL, yL, scale, n);
        } else {
            LocalTensor<float> xF = bX.Get<float>();
            LocalTensor<float> yF = bY.Get<float>();
            Cast(xF, xL, RoundMode::CAST_NONE, static_cast<uint32_t>(n));
            Mul(yF, xF, maskF, n);
            Muls(yF, yF, scale, n);
            Cast(yL, yF, RoundMode::CAST_RINT, static_cast<uint32_t>(n));
        }

        qX.FreeTensor(xL);
        qM.FreeTensor(mL);
        qY.EnQue(yL);
        yL = qY.DeQue<Tx>();
        DataCopyPad(yGm[off], yL, cpx);
        qY.FreeTensor(yL);
    }
}

/*!
 * \brief Host tiling: (numBlocks, blockLength, tileElems).
 * The per-element UB budget is
 *   4*xBytes + 2*maskBytes + scratch
 * with scratch = (xBytes != 4 ? 8 : 0) + (maskBytes != 4 ? 4 : 0) + (maskBytes == 1 ? 2 : 0)
 * (fp32 accumulate + fp32 mask + fp16 8-bit staging), which is exactly 24 B/elem
 * for every supported dtype pair.
 */
std::tuple<int64_t, int64_t, int64_t> calc_masked_scale_tiling_params(int64_t totalLength,
                                                                     int64_t xBytes,
                                                                     int64_t maskBytes)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 4096;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    if (numBlocks <= 0) numBlocks = 1;
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    blockLength = ((blockLength + MS_TILE_ALIGN - 1) / MS_TILE_ALIGN) * MS_TILE_ALIGN;
    if (blockLength <= 0) blockLength = MS_TILE_ALIGN;

    int64_t scratch = (xBytes != 4 ? 8 : 0) + (maskBytes != 4 ? 4 : 0) + (maskBytes == 1 ? 2 : 0);
    int64_t perElem = 4 * xBytes + 2 * maskBytes + scratch;
    int64_t usable = static_cast<int64_t>(ubSize) - MS_UB_SAFETY_MARGIN;
    if (usable < 0) usable = static_cast<int64_t>(ubSize) / 2;
    int64_t tileElems = usable / perElem;
    tileElems = (tileElems / MS_TILE_ALIGN) * MS_TILE_ALIGN;
    if (tileElems < MS_TILE_ALIGN) {
        tileElems = MS_TILE_ALIGN;
    }
    if (tileElems > blockLength) {
        tileElems = blockLength;
    }
    return std::make_tuple(numBlocks, blockLength, tileElems);
}

// Launch wrappers - regular C functions callable from g++.
extern "C" {

#define MS_LAUNCH_DEF(NAME, TX, TM)                                                                        \
void launch_masked_scale_##NAME(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, \
                                int64_t blockLength, uint32_t tileElems, float scale, void* stream)         \
{                                                                                                           \
    masked_scale_kernel<TX, TM><<<numBlocks, nullptr, stream>>>(x, mask, y, totalLength, blockLength,       \
                                                                tileElems, scale);                          \
}

MS_LAUNCH_DEF(half_int8, half, int8_t)
MS_LAUNCH_DEF(half_uint8, half, uint8_t)
MS_LAUNCH_DEF(half_half, half, half)
MS_LAUNCH_DEF(half_bfloat16, half, bfloat16_t)
MS_LAUNCH_DEF(half_float, half, float)

MS_LAUNCH_DEF(bfloat16_int8, bfloat16_t, int8_t)
MS_LAUNCH_DEF(bfloat16_uint8, bfloat16_t, uint8_t)
MS_LAUNCH_DEF(bfloat16_half, bfloat16_t, half)
MS_LAUNCH_DEF(bfloat16_bfloat16, bfloat16_t, bfloat16_t)
MS_LAUNCH_DEF(bfloat16_float, bfloat16_t, float)

MS_LAUNCH_DEF(float_int8, float, int8_t)
MS_LAUNCH_DEF(float_uint8, float, uint8_t)
MS_LAUNCH_DEF(float_half, float, half)
MS_LAUNCH_DEF(float_bfloat16, float, bfloat16_t)
MS_LAUNCH_DEF(float_float, float, float)

#undef MS_LAUNCH_DEF

} // extern "C"
