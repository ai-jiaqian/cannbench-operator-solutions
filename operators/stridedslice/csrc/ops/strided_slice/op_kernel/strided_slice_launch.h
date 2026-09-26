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
 * \file strided_slice_launch.h
 * \brief Shared declarations between the g++ plugin and the bisheng kernel for StridedSlice.
 */

#ifndef STRIDED_SLICE_LAUNCH_H
#define STRIDED_SLICE_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/* Tiling / plan bundle computed by the host side helper and consumed by the kernel. */
struct StridedSliceTiling {
    int64_t outNumel = 0;      /* number of output elements */
    int64_t baseOffset = 0;    /* element offset of output[0] inside x */
    int64_t innerLen = 1;      /* length of the innermost effective output dim (path 2/3) */
    int64_t innerStep = 1;     /* element step of the innermost effective output dim */
    int64_t rowCount = 1;      /* product of the effective sizes except the innermost */
    int64_t path = 0;          /* 0=flat copy, 1=row-group multi-block, 2=span+gather, 3=scalar */
    int64_t numUnits = 1;      /* total work units */
    int64_t unitLen = 1;       /* path0: elems/chunk, path1: rows/unit, path2: elems/chunk */
    int64_t rowGap = 0;        /* path1: element distance between consecutive rows */
    int64_t ubGap = 0;         /* path1: explicit UB row gap (bytes) = align32(rowBytes) - rowBytes */
    int64_t nRowsPerGroup = 1; /* path1: rows inside one uniformly strided group */
    int64_t chunksPerGroup = 1;/* path1: row chunks per group */
    int64_t chunksPerRow = 1;  /* path2: inner chunks per row */
    int64_t nd = 0;            /* number of odometer dims */
    int64_t sz[8] = {0};       /* odometer dim sizes */
    int64_t st[8] = {0};       /* odometer dim element steps */
    int64_t numBlocks = 1;     /* blocks to launch */
};

/* Plan/tiling entry point (defined in the kernel translation unit, called from the plugin).
 * `size` / `step` describe the effective output dims (output length > 1) in output order. */
StridedSliceTiling calc_strided_slice_tiling(int64_t baseOffset, int64_t outNumel, int64_t rank,
                                             const int64_t *size, const int64_t *step,
                                             int64_t elemBytes);

extern "C" {

void launch_strided_slice_i8(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream);
void launch_strided_slice_u8(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream);
void launch_strided_slice_i32(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream);
void launch_strided_slice_i64(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream);
void launch_strided_slice_f16(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream);
void launch_strided_slice_bf16(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream);
void launch_strided_slice_f32(GM_ADDR x, GM_ADDR y, StridedSliceTiling t, void *stream);
}

#endif  // STRIDED_SLICE_LAUNCH_H
