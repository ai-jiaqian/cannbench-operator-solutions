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
 * \file mha_launch.h
 * \brief Host visible tiling / launch declarations for MHA (compiled with g++).
 *
 *   query [B, S, N, D], key/value [B, Skv, N, D], y [B, S, N, D]
 *   y = softmax(Q @ K^T * scale) @ V
 */

#ifndef MHA_LAUNCH_H
#define MHA_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

/* Returns (numBlocks, BM, BN, totalItems, itemsPerCore). */
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> calc_mha_tiling_params(
    int64_t B, int64_t S, int64_t Skv, int64_t N, int64_t D);

extern "C" {

void launch_mha_kernel_f16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, int64_t B, int64_t S,
                           int64_t Skv, int64_t N, int64_t D, float scale, int32_t causal,
                           int64_t numRowBlk, int64_t totalItems, int64_t itemsPerCore, int64_t BM,
                           int64_t BN, int64_t numBlocks, void *stream);

void launch_mha_kernel_bf16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, int64_t B, int64_t S,
                            int64_t Skv, int64_t N, int64_t D, float scale, int32_t causal,
                            int64_t numRowBlk, int64_t totalItems, int64_t itemsPerCore, int64_t BM,
                            int64_t BN, int64_t numBlocks, void *stream);
}

#endif  // MHA_LAUNCH_H
