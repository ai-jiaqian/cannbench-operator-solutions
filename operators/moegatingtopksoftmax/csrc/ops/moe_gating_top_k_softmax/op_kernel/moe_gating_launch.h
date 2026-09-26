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
 * \file moe_gating_launch.h
 * \brief Tiling / launch declarations shared between the bisheng kernel TU and the g++ plugin TU.
 */

#ifndef MOE_GATING_LAUNCH_H
#define MOE_GATING_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host side tiling: returns (numBlocks, rowsPerCore, rowsPerTile)
std::tuple<int64_t, int64_t, int64_t> calc_moe_gating_tiling_params(int64_t totalRows, int64_t inner, int64_t k,
                                                                    int64_t typeSize);

extern "C" {
void launch_moe_gating_kernel_float(GM_ADDR x, GM_ADDR finished, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx,
                                    int64_t totalRows, int64_t inner, int64_t k, int64_t numBlocks,
                                    int64_t rowsPerCore, int64_t rowsPerTile, int64_t hasFinished, void* stream);
void launch_moe_gating_kernel_half(GM_ADDR x, GM_ADDR finished, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx,
                                   int64_t totalRows, int64_t inner, int64_t k, int64_t numBlocks,
                                   int64_t rowsPerCore, int64_t rowsPerTile, int64_t hasFinished, void* stream);
void launch_moe_gating_kernel_bfloat16(GM_ADDR x, GM_ADDR finished, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR rowIdx,
                                       int64_t totalRows, int64_t inner, int64_t k, int64_t numBlocks,
                                       int64_t rowsPerCore, int64_t rowsPerTile, int64_t hasFinished, void* stream);
}

#endif // MOE_GATING_LAUNCH_H
