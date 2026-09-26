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
 * \file rope_launch.h
 * \brief Launch / tiling declarations shared between the bisheng kernel TU and the g++ plugin TU.
 */

#ifndef ROPE_LAUNCH_H
#define ROPE_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

// Host tiling: returns (numBlocks, rowsPerChunk, ubBudgetBytes).
std::tuple<int64_t, int64_t, int64_t> calc_rope_tiling_params(int64_t B, int64_t S, int64_t N, int64_t D,
                                                              int64_t layout, int64_t mode, int64_t esz);

extern "C" {

void launch_rope_kernel_float(GM_ADDR q, GM_ADDR k, GM_ADDR cs, GM_ADDR sn, GM_ADDR qo, GM_ADDR ko,
                              int64_t B, int64_t S, int64_t N, int64_t D, int64_t layout, int64_t mode,
                              int64_t cos3d, int64_t numBlocks, int64_t rmax, int64_t budget,
                              void *stream);

void launch_rope_kernel_half(GM_ADDR q, GM_ADDR k, GM_ADDR cs, GM_ADDR sn, GM_ADDR qo, GM_ADDR ko,
                             int64_t B, int64_t S, int64_t N, int64_t D, int64_t layout, int64_t mode,
                             int64_t cos3d, int64_t numBlocks, int64_t rmax, int64_t budget,
                             void *stream);

void launch_rope_kernel_bf16(GM_ADDR q, GM_ADDR k, GM_ADDR cs, GM_ADDR sn, GM_ADDR qo, GM_ADDR ko,
                             int64_t B, int64_t S, int64_t N, int64_t D, int64_t layout, int64_t mode,
                             int64_t cos3d, int64_t numBlocks, int64_t rmax, int64_t budget,
                             void *stream);
}

#endif // ROPE_LAUNCH_H
