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
 * \file ce_launch.h
 * \brief CrossEntropyLoss tiling query + launch declarations shared between the bisheng kernel TU and
 *        the g++ plugin TU.
 */

#ifndef CE_LAUNCH_H
#define CE_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// plan[] layout produced on the host:
//   plan[0] = mode         0: row kernel (P==1)     1: column kernel (P>1)
//   plan[1] = numBlocks    grid size of the main kernel
//   plan[2] = bs / W       rows per tile (mode 0) or columns per tile (mode 1)
//   plan[3] = coreNum
//   plan[4] = cb           class rows held in UB per column-kernel chunk (mode 1)
//   plan[5] = nsplit       class-axis split factor (1 = no split; >1 -> lane partials)
//   plan[6] = mergeBlocks  grid size of the merge kernel / fold count of the finalize
//   plan[7] = lanes        groups * W, padded lane count of the column kernel
//   plan[8] = partialElems 4 * nsplit * lanes, elements of the f32 lane-partial buffer
extern "C" void ce_query_plan(int64_t N, int64_t C, int64_t P, int64_t esz, int64_t eszT, int64_t redMode,
                              int64_t* plan);

extern "C" {

/* ---------------- P == 1 : row kernel ---------------- */
void launch_ce_row_float_i32(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, int64_t R, int64_t C, int64_t ignoreIdx,
                             int64_t redMode, int64_t numBlocks, int64_t bs, void* stream);
void launch_ce_row_float_i64(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, int64_t R, int64_t C, int64_t ignoreIdx,
                             int64_t redMode, int64_t numBlocks, int64_t bs, void* stream);
void launch_ce_row_half_i32(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, int64_t R, int64_t C, int64_t ignoreIdx,
                            int64_t redMode, int64_t numBlocks, int64_t bs, void* stream);
void launch_ce_row_half_i64(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, int64_t R, int64_t C, int64_t ignoreIdx,
                            int64_t redMode, int64_t numBlocks, int64_t bs, void* stream);
void launch_ce_row_bf16_i32(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, int64_t R, int64_t C, int64_t ignoreIdx,
                            int64_t redMode, int64_t numBlocks, int64_t bs, void* stream);
void launch_ce_row_bf16_i64(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, int64_t R, int64_t C, int64_t ignoreIdx,
                            int64_t redMode, int64_t numBlocks, int64_t bs, void* stream);

/* ---------------- P > 1 : column kernel ---------------- */
void launch_ce_col_float_i32(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, GM_ADDR lp, int64_t N, int64_t C,
                             int64_t P, int64_t ignoreIdx, int64_t redMode, int64_t numBlocks, int64_t W,
                             int64_t cb, int64_t nsplit, void* stream);
void launch_ce_col_float_i64(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, GM_ADDR lp, int64_t N, int64_t C,
                             int64_t P, int64_t ignoreIdx, int64_t redMode, int64_t numBlocks, int64_t W,
                             int64_t cb, int64_t nsplit, void* stream);
void launch_ce_col_half_i32(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, GM_ADDR lp, int64_t N, int64_t C,
                            int64_t P, int64_t ignoreIdx, int64_t redMode, int64_t numBlocks, int64_t W,
                            int64_t cb, int64_t nsplit, void* stream);
void launch_ce_col_half_i64(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, GM_ADDR lp, int64_t N, int64_t C,
                            int64_t P, int64_t ignoreIdx, int64_t redMode, int64_t numBlocks, int64_t W,
                            int64_t cb, int64_t nsplit, void* stream);
void launch_ce_col_bf16_i32(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, GM_ADDR lp, int64_t N, int64_t C,
                            int64_t P, int64_t ignoreIdx, int64_t redMode, int64_t numBlocks, int64_t W,
                            int64_t cb, int64_t nsplit, void* stream);
void launch_ce_col_bf16_i64(GM_ADDR x, GM_ADDR t, GM_ADDR o, GM_ADDR p, GM_ADDR lp, int64_t N, int64_t C,
                            int64_t P, int64_t ignoreIdx, int64_t redMode, int64_t numBlocks, int64_t W,
                            int64_t cb, int64_t nsplit, void* stream);

/* ---------------- merge of the class-axis partials ---------------- */
void launch_ce_merge_float(GM_ADDR lp, GM_ADDR o, GM_ADDR p, int64_t N, int64_t C, int64_t P, int64_t redMode,
                           int64_t numBlocks, int64_t W, int64_t nsplit, void* stream);
void launch_ce_merge_half(GM_ADDR lp, GM_ADDR o, GM_ADDR p, int64_t N, int64_t C, int64_t P, int64_t redMode,
                          int64_t numBlocks, int64_t W, int64_t nsplit, void* stream);
void launch_ce_merge_bf16(GM_ADDR lp, GM_ADDR o, GM_ADDR p, int64_t N, int64_t C, int64_t P, int64_t redMode,
                          int64_t numBlocks, int64_t W, int64_t nsplit, void* stream);

/* ---------------- finalize (scalar output) ---------------- */
void launch_ce_final_float(GM_ADDR part, GM_ADDR o, int64_t numBlocks, int64_t redMode, void* stream);
void launch_ce_final_half(GM_ADDR part, GM_ADDR o, int64_t numBlocks, int64_t redMode, void* stream);
void launch_ce_final_bf16(GM_ADDR part, GM_ADDR o, int64_t numBlocks, int64_t redMode, void* stream);
}

#endif // CE_LAUNCH_H
