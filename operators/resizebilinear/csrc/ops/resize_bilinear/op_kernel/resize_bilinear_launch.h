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
 * \file resize_bilinear_launch.h
 * \brief Host-side declarations for ResizeBilinear (visible to g++).
 */

#ifndef RESIZE_BILINEAR_LAUNCH_H
#define RESIZE_BILINEAR_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/*!
 * \brief Host tiling helper (defined in the bisheng-compiled kernel translation unit).
 *
 * \param totalRows  N * C * H_out (one output image row per work item)
 * \return (numBlocks, rowsPerCore)
 */
std::tuple<int64_t, int64_t> calc_resize_bilinear_tiling(int64_t totalRows);

extern "C" {

void launch_resize_bilinear_float(GM_ADDR x, GM_ADDR y, int64_t totalRows, int64_t numBlocks,
                                  int64_t rowsPerCore, int64_t hIn, int64_t wIn, int64_t hOut,
                                  int64_t wOut, float scaleH, float scaleW, int32_t alignCorners,
                                  void* stream);

void launch_resize_bilinear_half(GM_ADDR x, GM_ADDR y, int64_t totalRows, int64_t numBlocks,
                                 int64_t rowsPerCore, int64_t hIn, int64_t wIn, int64_t hOut,
                                 int64_t wOut, float scaleH, float scaleW, int32_t alignCorners,
                                 void* stream);

void launch_resize_bilinear_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalRows, int64_t numBlocks,
                                     int64_t rowsPerCore, int64_t hIn, int64_t wIn, int64_t hOut,
                                     int64_t wOut, float scaleH, float scaleW, int32_t alignCorners,
                                     void* stream);

}  // extern "C"

#endif  // RESIZE_BILINEAR_LAUNCH_H
