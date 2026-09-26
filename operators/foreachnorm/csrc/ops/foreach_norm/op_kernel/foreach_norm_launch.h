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
 * \file foreach_norm_launch.h
 * \brief Launch / tiling declarations for g++ plugin side (ForeachNorm)
 */

#ifndef FOREACH_NORM_LAUNCH_H
#define FOREACH_NORM_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// p-norm computation modes shared between host dispatch and device kernels.
enum ForeachNormMode : int32_t {
    FN_MODE_GENERAL = 0,  // (sum |x|^p)^(1/p) via exp/ln for a generic p
    FN_MODE_L1 = 1,       // sum |x|
    FN_MODE_L2 = 2,       // sqrt(sum x^2)
    FN_MODE_MAX = 3,      // max |x|            (p = +inf)
    FN_MODE_MIN = 4,      // min |x|            (p = -inf)
    FN_MODE_L0 = 5        // count of non-zero elements (p = 0)
};

// Host-side tiling parameters.
struct ForeachNormTiling {
    int64_t numBlocks;
    int64_t blockLength;
    uint32_t tileElems;
};

ForeachNormTiling calc_foreach_norm_tiling(int64_t totalLength);

extern "C" {

void launch_foreach_norm_reduce_float(GM_ADDR x, GM_ADDR ws, int64_t totalLength, int64_t blockLength,
                                      uint32_t tileElems, float p, int32_t mode, int64_t numBlocks, void* stream);
void launch_foreach_norm_reduce_half(GM_ADDR x, GM_ADDR ws, int64_t totalLength, int64_t blockLength,
                                     uint32_t tileElems, float p, int32_t mode, int64_t numBlocks, void* stream);
void launch_foreach_norm_reduce_bf16(GM_ADDR x, GM_ADDR ws, int64_t totalLength, int64_t blockLength,
                                     uint32_t tileElems, float p, int32_t mode, int64_t numBlocks, void* stream);

// One finalize kernel handles the whole tensor list. `ws` points at listLen * numBlocks partials
// laid out contiguously (tensor i owns ws[i*numBlocks .. (i+1)*numBlocks)); `y` points at a
// contiguous listLen-element output buffer (element i is y[i]).
void launch_foreach_norm_finalize_float(GM_ADDR ws, GM_ADDR y, int64_t listLen, int64_t numBlocks, float p,
                                        int32_t mode, void* stream);
void launch_foreach_norm_finalize_half(GM_ADDR ws, GM_ADDR y, int64_t listLen, int64_t numBlocks, float p,
                                       int32_t mode, void* stream);
void launch_foreach_norm_finalize_bf16(GM_ADDR ws, GM_ADDR y, int64_t listLen, int64_t numBlocks, float p,
                                       int32_t mode, void* stream);
}

#endif // FOREACH_NORM_LAUNCH_H
