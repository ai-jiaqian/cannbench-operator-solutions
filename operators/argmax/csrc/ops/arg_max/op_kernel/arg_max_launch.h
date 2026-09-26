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
 * \file arg_max_launch.h
 * \brief Tiling / launch declarations shared between the bisheng kernel TU and the g++ plugin TU.
 *
 *   indices = argmax(input, dim=dim)          (torch.argmax semantics, int64 output)
 *
 * The ND tensor is viewed as (outer, reduceLen, inner) with `dim` the reduce axis:
 *   outer     = prod(shape[:dim])
 *   reduceLen = shape[dim]
 *   inner     = prod(shape[dim+1:])
 * The output (indices) is contiguous as (outer, inner).
 */

#ifndef ARG_MAX_LAUNCH_H
#define ARG_MAX_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// dtype codes (shared with the plugin)
enum ArgMaxDtypeCode {
    AM_HALF = 0,
    AM_FLOAT = 1,
    AM_BF16 = 2,
    AM_INT32 = 3,
    AM_INT64 = 4
};

// Host side tiling result, computed once per call in the kernel TU.
struct ArgMaxPlan {
    int64_t mode;      // 0 = reduce axis innermost (row kernel), 1 = middle axis (column kernel)
    int64_t outer;
    int64_t reduceLen;
    int64_t inner;

    // ---- row mode ----
    int64_t segRow;     // chunk width along the reduce axis
    int64_t nChunk;     // ceil(reduceLen / segRow)
    int64_t unitsRow;   // nChunk == 1 : outer ; else outer * nChunk
    int64_t blocksRow;
    int64_t upbRow;     // units per block
    int64_t unitsRow2;  // combine stage units (= outer)
    int64_t blocksRow2;
    int64_t upbRow2;
    int64_t batchRows;  // int64 output rows gathered per DMA (row merge stage)
    int64_t pitchRow;   // row mode: UB row pitch in elements (32B aligned)
    int64_t rbRow;      // row mode: rows per batched DMA

    // ---- column mode ----
    int64_t wTile;      // column tile width (multiple of 64)
    int64_t nColTile;   // ceil(inner / wTile)
    int64_t nseg;       // split of the reduce axis
    int64_t segD;       // ceil(reduceLen / nseg)
    int64_t unitsCol;   // outer * nColTile * nseg
    int64_t blocksCol;
    int64_t upbCol;
    int64_t unitsCol2;  // outer * nColTile
    int64_t blocksCol2;
    int64_t upbCol2;
    int64_t rb;         // reduce rows per DMA in the column kernel
    int64_t directCol;  // 1 when the reduce axis needs no split (write int64 straight out)

    int64_t wsFloats;   // device workspace size in floats (0 when unused)
};

ArgMaxPlan calc_argmax_plan(int64_t outer, int64_t reduceLen, int64_t inner, int64_t dtypeCode);

extern "C" {

void launch_argmax_row_direct_f16(GM_ADDR x, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_direct_f32(GM_ADDR x, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_direct_bf16(GM_ADDR x, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_direct_i32(GM_ADDR x, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_direct_i64(GM_ADDR x, GM_ADDR y, const ArgMaxPlan* p, void* stream);

void launch_argmax_row_partial_f16(GM_ADDR x, GM_ADDR ws, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_partial_f32(GM_ADDR x, GM_ADDR ws, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_partial_bf16(GM_ADDR x, GM_ADDR ws, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_partial_i32(GM_ADDR x, GM_ADDR ws, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_partial_i64(GM_ADDR x, GM_ADDR ws, const ArgMaxPlan* p, void* stream);
void launch_argmax_row_combine(GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream);

void launch_argmax_col_partial_f16(GM_ADDR x, GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_col_partial_f32(GM_ADDR x, GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_col_partial_bf16(GM_ADDR x, GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_col_partial_i32(GM_ADDR x, GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_col_partial_i64(GM_ADDR x, GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream);
void launch_argmax_col_combine(GM_ADDR ws, GM_ADDR y, const ArgMaxPlan* p, void* stream);
}

#endif // ARG_MAX_LAUNCH_H
