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
 * \file roi_align_launch.h
 * \brief Host visible declarations for ROIAlign (consumed by the g++ compiled plugin TU).
 */

#ifndef ROI_ALIGN_LAUNCH_H
#define ROI_ALIGN_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

/*! Host tiling result (POD, shared by the bisheng and g++ translation units). */
struct RoiAlignTiling {
    int64_t CbP;          /* channels per channel-block (multiple of 8)     */
    int64_t nCb;          /* number of channel blocks                        */
    int64_t OWp;          /* power-of-two lane pitch for output columns      */
    int64_t OP;           /* padded store pitch (32 byte aligned, power of 2) */
    int64_t perm;         /* 1 when the accumulator must be repacked for store */
    int64_t pitchMax;     /* max aligned stage pitch (elements)              */
    int64_t gwCap;        /* number of ix slots whose x geometry is cached   */
    int64_t numUnits;     /* total work units (N * nCb)                      */
    int64_t numBlocks;    /* grid size                                       */
    int64_t unitsPerCore; /* units handled by one core                       */
};

/*! Host tiling helper (defined in the bisheng compiled kernel translation unit). */
RoiAlignTiling calc_roi_align_tiling(int64_t B, int64_t C, int64_t H, int64_t W, int64_t N,
                                     int64_t oh, int64_t ow, int64_t esz, int64_t sr);

extern "C" {

void launch_roi_align_float(GM_ADDR x, GM_ADDR boxes, GM_ADDR y,
                            int64_t B, int64_t C, int64_t H, int64_t W, int64_t N,
                            int64_t oh, int64_t ow, float ss, int64_t sr, int64_t aligned,
                            int64_t CbP, int64_t nCb, int64_t OWp, int64_t OP, int64_t perm,
                            int64_t pitchMax, int64_t gwCap,
                            int64_t numUnits, int64_t unitsPerCore,
                            int64_t numBlocks, void *stream);

void launch_roi_align_half(GM_ADDR x, GM_ADDR boxes, GM_ADDR y,
                           int64_t B, int64_t C, int64_t H, int64_t W, int64_t N,
                           int64_t oh, int64_t ow, float ss, int64_t sr, int64_t aligned,
                           int64_t CbP, int64_t nCb, int64_t OWp, int64_t OP, int64_t perm,
                           int64_t pitchMax, int64_t gwCap,
                           int64_t numUnits, int64_t unitsPerCore,
                           int64_t numBlocks, void *stream);
}

#endif  // ROI_ALIGN_LAUNCH_H
