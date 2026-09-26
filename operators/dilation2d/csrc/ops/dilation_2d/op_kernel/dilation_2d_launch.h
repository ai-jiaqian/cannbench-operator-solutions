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
 * \file dilation_2d_launch.h
 * \brief Launch / tiling declarations for g++ (Dilation2D: y = max(x + filter))
 */

#ifndef DILATION_2D_LAUNCH_H
#define DILATION_2D_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host tiling result.  Shared POD between the bisheng (host) tiling function and the
// g++ plugin.
struct Dilation2dTiling {
    int64_t rowsTotal;  // N * outH (one "row" == one output plane row)
    int64_t blockRows;  // rows handled by one core
    int64_t tw;         // output columns per tile
    int64_t pad;        // padded channel pitch (elements), multiple of 16
    int64_t cacheR;     // input-row ring cache depth (>= 2)
};

// Tiling entry point (defined in the kernel translation unit, called from the plugin).
Dilation2dTiling calc_dilation_2d_tiling(int64_t N, int64_t outH, int64_t outW, int64_t C,
                                         int64_t fh, int64_t fw, int64_t sH, int64_t sW,
                                         int64_t rH, int64_t rW);

extern "C" {

void launch_dilation_2d_kernel_half(
    GM_ADDR x, GM_ADDR filter, GM_ADDR y,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t outH, int64_t outW, int64_t fh, int64_t fw,
    int64_t sH, int64_t sW, int64_t rH, int64_t rW,
    int64_t padTop, int64_t padLeft,
    int64_t rowsTotal, int64_t blockRows, int64_t tw, int64_t pad,
    int64_t cacheR, int64_t numBlocks, void* stream);
}

#endif  // DILATION_2D_LAUNCH_H
