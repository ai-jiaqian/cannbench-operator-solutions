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
 * \file gather_launch.h
 * \brief Launch function declarations for g++ (Gather operator)
 */

#ifndef GATHER_LAUNCH_H
#define GATHER_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

#define G_ARGS_DECL                                                                          \
    GM_ADDR x, GM_ADDR ix, GM_ADDR y, int64_t n, int64_t dim,                                \
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,                                      \
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,                                      \
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,                                      \
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,                                      \
    int64_t xMid, int64_t xRowStride, int64_t outerOut, int64_t mid, int64_t innerOut,       \
    int64_t Cc, int64_t RB, int64_t unitsPerA, int64_t numTasks, int64_t numBlocks

#define G_ARGS_CALL                                                                          \
    x, ix, y, n, dim,                                                                        \
    xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7,                                                  \
    os0, os1, os2, os3, os4, os5, os6, os7,                                                  \
    xMid, xRowStride, outerOut, mid, innerOut,                                               \
    Cc, RB, unitsPerA, numTasks, numBlocks

#define G_SLOW_ARGS_DECL                                                                     \
    GM_ADDR x, GM_ADDR ix, GM_ADDR y, int64_t n, int64_t dim,                                \
    int64_t xs0, int64_t xs1, int64_t xs2, int64_t xs3,                                      \
    int64_t xs4, int64_t xs5, int64_t xs6, int64_t xs7,                                      \
    int64_t os0, int64_t os1, int64_t os2, int64_t os3,                                      \
    int64_t os4, int64_t os5, int64_t os6, int64_t os7,                                      \
    int64_t xMid, int64_t xNumel, int64_t totalOut, int64_t blockLength,                     \
    int64_t tileElems, int64_t numBlocks

#define G_SLOW_ARGS_CALL                                                                     \
    x, ix, y, n, dim,                                                                        \
    xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7,                                                  \
    os0, os1, os2, os3, os4, os5, os6, os7,                                                  \
    xMid, xNumel, totalOut, blockLength, tileElems, numBlocks

// clang-format off
extern "C" {
void gather_launch_fast_f32_i32(G_ARGS_DECL, void* stream);
void gather_launch_fast_f32_i64(G_ARGS_DECL, void* stream);
void gather_launch_fast_f16_i32(G_ARGS_DECL, void* stream);
void gather_launch_fast_f16_i64(G_ARGS_DECL, void* stream);
void gather_launch_fast_bf16_i32(G_ARGS_DECL, void* stream);
void gather_launch_fast_bf16_i64(G_ARGS_DECL, void* stream);
void gather_launch_fast_i8_i32(G_ARGS_DECL, void* stream);
void gather_launch_fast_i8_i64(G_ARGS_DECL, void* stream);
void gather_launch_fast_i32_i32(G_ARGS_DECL, void* stream);
void gather_launch_fast_i32_i64(G_ARGS_DECL, void* stream);
void gather_launch_fast64_i32(G_ARGS_DECL, void* stream);
void gather_launch_fast64_i64(G_ARGS_DECL, void* stream);

void gather_launch_slow2_f32_i8(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_f32_i32(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_f32_i64(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_f16_i8(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_f16_i32(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_f16_i64(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_bf16_i8(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_bf16_i32(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_bf16_i64(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i8_i8(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i8_i32(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i8_i64(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i32_i8(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i32_i32(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i32_i64(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i64_i8(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i64_i32(G_SLOW_ARGS_DECL, void* stream);
void gather_launch_slow2_i64_i64(G_SLOW_ARGS_DECL, void* stream);
}
// clang-format on

#endif // GATHER_LAUNCH_H
