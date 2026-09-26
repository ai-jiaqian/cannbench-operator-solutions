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
 * \file moe_re_routing_launch.h
 * \brief Host visible tiling / launch declarations for MoeReRouting.
 *
 *   srcStart(i, j) = sum_{i'<i} rowSum(i') + sum_{j'<j} cnt[i][j']
 *   dstStart(i, j) = sum_{j'<j} colSum(j') + sum_{i'<i} cnt[i'][j]
 *   for k in [0, cnt[i][j]):
 *       permute_tokens        [dstStart + k] = tokens            [srcStart + k]
 *       permute_per_token_scales[dstStart+k] = per_token_scales  [srcStart + k]
 *       permute_token_idx     [dstStart + k] = srcStart + k
 *   expert_token_num[j] = colSum(j)
 *
 * Both the source slice and the destination slice of one (rank, expert) cell are contiguous, so
 * the operator is a concatenation of N*E contiguous block copies plus one index ramp per cell.
 *
 * This header must stay plain C++ (no AscendC types): it is included by the g++ plugin TU.
 */

#ifndef MOE_RE_ROUTING_LAUNCH_H
#define MOE_RE_ROUTING_LAUNCH_H

#include <cstdint>

#include "moe_re_routing_tiling.h"

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/* Pure metadata arithmetic; no tensor data is touched on the host. */
MoeRRTiling calc_moe_re_routing_tiling(int64_t A, int64_t H, int64_t N, int64_t E,
                                       int64_t tokElem, int64_t cntElem);

extern "C" {

/* token dtype [h = half, bf = bfloat16, i8 = int8] x count dtype [i32 / i64] */
void launch_moe_rr_h_i32(GM_ADDR tok, GM_ADDR cnt, GM_ADDR scl, GM_ADDR oTok, GM_ADDR oScl,
                         GM_ADDR oIdx, GM_ADDR oExp, int64_t A, int64_t H, int64_t N, int64_t E,
                         int64_t numBlocks, int64_t dstPerBlk, int64_t chunkTokens,
                         int64_t hasScales, void* stream);
void launch_moe_rr_h_i64(GM_ADDR tok, GM_ADDR cnt, GM_ADDR scl, GM_ADDR oTok, GM_ADDR oScl,
                         GM_ADDR oIdx, GM_ADDR oExp, int64_t A, int64_t H, int64_t N, int64_t E,
                         int64_t numBlocks, int64_t dstPerBlk, int64_t chunkTokens,
                         int64_t hasScales, void* stream);
void launch_moe_rr_bf_i32(GM_ADDR tok, GM_ADDR cnt, GM_ADDR scl, GM_ADDR oTok, GM_ADDR oScl,
                          GM_ADDR oIdx, GM_ADDR oExp, int64_t A, int64_t H, int64_t N, int64_t E,
                          int64_t numBlocks, int64_t dstPerBlk, int64_t chunkTokens,
                          int64_t hasScales, void* stream);
void launch_moe_rr_bf_i64(GM_ADDR tok, GM_ADDR cnt, GM_ADDR scl, GM_ADDR oTok, GM_ADDR oScl,
                          GM_ADDR oIdx, GM_ADDR oExp, int64_t A, int64_t H, int64_t N, int64_t E,
                          int64_t numBlocks, int64_t dstPerBlk, int64_t chunkTokens,
                          int64_t hasScales, void* stream);
void launch_moe_rr_i8_i32(GM_ADDR tok, GM_ADDR cnt, GM_ADDR scl, GM_ADDR oTok, GM_ADDR oScl,
                          GM_ADDR oIdx, GM_ADDR oExp, int64_t A, int64_t H, int64_t N, int64_t E,
                          int64_t numBlocks, int64_t dstPerBlk, int64_t chunkTokens,
                          int64_t hasScales, void* stream);
void launch_moe_rr_i8_i64(GM_ADDR tok, GM_ADDR cnt, GM_ADDR scl, GM_ADDR oTok, GM_ADDR oScl,
                          GM_ADDR oIdx, GM_ADDR oExp, int64_t A, int64_t H, int64_t N, int64_t E,
                          int64_t numBlocks, int64_t dstPerBlk, int64_t chunkTokens,
                          int64_t hasScales, void* stream);

}  // extern "C"

#endif  // MOE_RE_ROUTING_LAUNCH_H
