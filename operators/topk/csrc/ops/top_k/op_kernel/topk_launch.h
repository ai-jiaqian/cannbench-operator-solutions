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
 * \file topk_launch.h
 * \brief Tiling / launch declarations shared between the bisheng kernel TU and the g++ plugin TU.
 */

#ifndef TOPK_LAUNCH_H
#define TOPK_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Input dtype codes shared by host and device.
enum TopkTypeCode {
    TK_T_F32 = 0,
    TK_T_F16 = 1,
    TK_T_BF16 = 2,
    TK_T_I8 = 3,
    TK_T_U8 = 4,
    TK_T_I32 = 5,
    TK_T_I64 = 6,
};

// Host tiling result.
struct TopkPlan {
    int64_t outer;
    int64_t reduce;
    int64_t inner;
    int64_t k;
    int64_t largest;   // 1 for largest, 0 for smallest
    int64_t mode;      // 0: reduce axis is innermost (contiguous row), 1: middle axis
    int64_t Cs;        // columns handled by one unit (1 for mode 0)
    int64_t R;         // reduce-axis tile length (power-of-two multiple of 32)
    int64_t groups;    // column groups per outer slice (1 for mode 0)
    int64_t numUnits;  // outer * groups
    int64_t numBlocks;
    int64_t ch;        // output chunk length along k
    int64_t accPitch;  // bytes per column accumulator slot
    int64_t coreNum;   // number of AIV cores on this platform
    int64_t ubNeeded;  // estimated UB bytes for the chosen tiling
};

TopkPlan calc_topk_plan(int64_t outer, int64_t reduce, int64_t inner, int64_t k, int64_t largest,
                        int64_t typeSize);

// Execution modes.
//   0: one unit == one contiguous reduce-axis row (inner == 1)
//   1: one unit == a group of Cs adjacent inner indices of one outer slice
//   2: row split - unit u owns one unequal slice of a single row; the top-k of the slice is
//      written to the value workspace (dtype T) and the index workspace (int64)
//   3: merge - a single unit over a contiguous value array (dtype T) whose indices are read
//      from the int64 index workspace
#define TK_MODE_ROW 0
#define TK_MODE_COL 1
#define TK_MODE_CHUNK 2
#define TK_MODE_MERGE 3

extern "C" {
void launch_topk(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR ix, int64_t typeCode, int64_t outer,
                 int64_t reduce, int64_t inner, int64_t k, int64_t largest, int64_t Cs, int64_t R,
                 int64_t groups, int64_t numUnits, int64_t numBlocks, int64_t ch, int64_t mode,
                 int64_t accPitch, int64_t chunkRem, void *stream);
}

#endif // TOPK_LAUNCH_H
