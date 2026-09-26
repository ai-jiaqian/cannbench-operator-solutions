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
 * \file exp_launch.h
 * \brief Tiling / launch declarations for the Exp operator. This header is
 *        shared by the bisheng kernel translation unit and the g++ plugin
 *        translation unit, so it must stay free of AscendC types.
 */

#ifndef EXP_LAUNCH_H
#define EXP_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host side tiling: returns (numBlocks, tilesPerCore, tileElems)
std::tuple<int64_t, int64_t, uint32_t> calc_exp_tiling_params(int64_t totalLength, int64_t elemSize);

extern "C" {

void launch_exp_kernel_float(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                             int64_t tilesPerCore, uint32_t tileElems, float coefA, float coefB,
                             void* stream);

void launch_exp_kernel_half(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                            int64_t tilesPerCore, uint32_t tileElems, float coefA, float coefB,
                            void* stream);

void launch_exp_kernel_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
                                int64_t tilesPerCore, uint32_t tileElems, float coefA, float coefB,
                                void* stream);

} // extern "C"

#endif // EXP_LAUNCH_H
