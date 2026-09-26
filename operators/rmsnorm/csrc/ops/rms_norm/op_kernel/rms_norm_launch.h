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
 * \file rms_norm_launch.h
 * \brief Launch / tiling declarations for the g++ compiled plugin layer.
 *
 * Frozen z6 structure: ONE launch per call. The per-row statistic is computed and
 * consumed on the resident on-chip tile and is never materialized in global memory.
 */

#ifndef RMS_NORM_LAUNCH_H
#define RMS_NORM_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host side tiling: returns (numBlocks, blockRows, tileRows)
std::tuple<int64_t, int64_t, int64_t> calc_rms_norm_tiling_params(int64_t numRow, int64_t numCol,
                                                                  int64_t elemSize);

extern "C" {

void launch_rms_norm_kernel_float(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t numRow, int64_t numCol, float epsilon, float invD,
    int64_t numBlocks, int64_t blockRows, int64_t tileRows, void* stream);

void launch_rms_norm_kernel_half(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t numRow, int64_t numCol, float epsilon, float invD,
    int64_t numBlocks, int64_t blockRows, int64_t tileRows, void* stream);

void launch_rms_norm_kernel_bfloat16(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t numRow, int64_t numCol, float epsilon, float invD,
    int64_t numBlocks, int64_t blockRows, int64_t tileRows, void* stream);

}

#define RMS_NORM_LAUNCH_FLOAT(x, g, y, nr, nc, eps, invd, nb, br, tr, stream) \
    launch_rms_norm_kernel_float(x, g, y, nr, nc, eps, invd, nb, br, tr, stream)
#define RMS_NORM_LAUNCH_HALF(x, g, y, nr, nc, eps, invd, nb, br, tr, stream) \
    launch_rms_norm_kernel_half(x, g, y, nr, nc, eps, invd, nb, br, tr, stream)
#define RMS_NORM_LAUNCH_BF16(x, g, y, nr, nc, eps, invd, nb, br, tr, stream) \
    launch_rms_norm_kernel_bfloat16(x, g, y, nr, nc, eps, invd, nb, br, tr, stream)

#endif // RMS_NORM_LAUNCH_H
