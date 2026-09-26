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
 * \file dynamic_quant_launch.h
 * \brief Launch function declarations for g++ (DynamicQuant: per-token symmetric dynamic quantization)
 */

#ifndef DYNAMIC_QUANT_LAUNCH_H
#define DYNAMIC_QUANT_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host tiling function: returns (numBlocks, rowsPerCore, rowsPerTile, inDepth)
std::tuple<int64_t, int64_t, int64_t, int64_t> calc_dynamic_quant_tiling(int64_t M, int64_t N);

extern "C" {

void launch_dynamic_quant_half(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                               int64_t M, int64_t N, int64_t numBlocks,
                               int64_t rowsPerCore, int64_t rowsPerTile, int64_t inDepth,
                               void *stream);

void launch_dynamic_quant_bfloat16(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                   int64_t M, int64_t N, int64_t numBlocks,
                                   int64_t rowsPerCore, int64_t rowsPerTile, int64_t inDepth,
                                   void *stream);

} // extern "C"

#endif // DYNAMIC_QUANT_LAUNCH_H
