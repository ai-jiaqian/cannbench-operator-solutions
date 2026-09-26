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
 * \file dqsq_launch.h
 * \brief Shared declarations between the bisheng kernel TU and the g++ plugin TU.
 */

#ifndef DQSQ_LAUNCH_H
#define DQSQ_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/*!
 * \brief Host tiling for DequantSwigluQuant.
 *
 * \return (numBlocks, rowsPerCore, rowsPerTile)
 */
std::tuple<int64_t, int64_t, int64_t> calc_dqsq_tiling(int64_t M, int64_t N2, int64_t elemSize,
                                                       int64_t isInt32, int64_t hasQScale);

extern "C" {

void launch_dqsq_half(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR wscale, GM_ADDR ascale,
                      GM_ADDR qscale, int64_t M, int64_t N2, int64_t numBlocks, int64_t rowsPerCore,
                      int64_t tr, int64_t activateLeft, int64_t hasQScale, void *stream);

void launch_dqsq_bfloat16(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR wscale, GM_ADDR ascale,
                          GM_ADDR qscale, int64_t M, int64_t N2, int64_t numBlocks, int64_t rowsPerCore,
                          int64_t tr, int64_t activateLeft, int64_t hasQScale, void *stream);

void launch_dqsq_int32(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR wscale, GM_ADDR ascale,
                       GM_ADDR qscale, int64_t M, int64_t N2, int64_t numBlocks, int64_t rowsPerCore,
                       int64_t tr, int64_t activateLeft, int64_t hasQScale, void *stream);

}  // extern "C"

#endif  // DQSQ_LAUNCH_H
