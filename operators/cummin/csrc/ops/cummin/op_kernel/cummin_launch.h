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
 * \file cummin_launch.h
 * \brief Tiling description + launch entry points for the Cummin direct-launch kernel.
 *        Shared by the bisheng-compiled kernel translation unit and the g++ compiled plugin.
 */

#ifndef CUMMIN_LAUNCH_H
#define CUMMIN_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/* Cummin tiling description.
 *   mode 0 : row program (inner == 1)  -- each row is one contiguous chain.
 *   mode 1 : column program (inner > 1) -- lanes are independent, axis walked row by row.
 */
struct CumminTiling {
    int64_t mode;         /* 0 = row program, 1 = column program */
    int64_t numBlocks;    /* grid size (AIV blocks) */
    int64_t rowsPerBlock; /* mode 0: rows handled by one block */
    int64_t chunk;        /* mode 0: elements staged per contiguous chunk */
    int64_t Tg;           /* mode 1: lanes per group */
    int64_t nGroups;      /* mode 1: number of lane groups per outer index */
};

CumminTiling calc_cummin_tiling(int64_t outer, int64_t axis, int64_t inner, int64_t elemBytes);

extern "C" {

void launch_cummin_row_float(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t len,
                             int64_t rowsPerBlock, int64_t numBlocks, int64_t chunk, void* stream);
void launch_cummin_row_half(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t len,
                            int64_t rowsPerBlock, int64_t numBlocks, int64_t chunk, void* stream);
void launch_cummin_row_bfloat16(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t len,
                                int64_t rowsPerBlock, int64_t numBlocks, int64_t chunk, void* stream);
void launch_cummin_row_int32(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t rows, int64_t len,
                             int64_t rowsPerBlock, int64_t numBlocks, int64_t chunk, void* stream);

void launch_cummin_col_float(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t outer, int64_t axis, int64_t inner,
                             int64_t nGroups, int64_t Tg, int64_t numBlocks, void* stream);
void launch_cummin_col_half(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t outer, int64_t axis, int64_t inner,
                            int64_t nGroups, int64_t Tg, int64_t numBlocks, void* stream);
void launch_cummin_col_bfloat16(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t outer, int64_t axis, int64_t inner,
                                int64_t nGroups, int64_t Tg, int64_t numBlocks, void* stream);
void launch_cummin_col_int32(GM_ADDR x, GM_ADDR v, GM_ADDR i, int64_t outer, int64_t axis, int64_t inner,
                             int64_t nGroups, int64_t Tg, int64_t numBlocks, void* stream);
}

#endif // CUMMIN_LAUNCH_H
