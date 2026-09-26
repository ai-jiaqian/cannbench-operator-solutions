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
 * \file add_rms_norm_dynamic_quant_launch.h
 * \brief Launch function declarations for g++
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_LAUNCH_H
#define ADD_RMS_NORM_DYNAMIC_QUANT_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling function declaration: returns (numBlocks, rowsPerCore, tileElems, rowBlock)
//   rowBlock > 0 -> batched block path, rowBlock rows per iteration
//   rowBlock == 0 -> chunk path, tileElems lanes per chunk
std::tuple<int64_t, int64_t, int64_t, int64_t> calc_ardq_tiling(int64_t numRows, int64_t rowLen);

// Launch function declarations
extern "C" {
void launch_ardq_half(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR xOut,
                      GM_ADDR scaleOut, int64_t numRows, int64_t rowLen, int64_t numBlocks,
                      int64_t rowsPerCore, uint32_t tileElems, int64_t rowBlock, float epsilon,
                      void* stream);
void launch_ardq_bf16(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR xOut,
                      GM_ADDR scaleOut, int64_t numRows, int64_t rowLen, int64_t numBlocks,
                      int64_t rowsPerCore, uint32_t tileElems, int64_t rowBlock, float epsilon,
                      void* stream);
}

#endif  // ADD_RMS_NORM_DYNAMIC_QUANT_LAUNCH_H
