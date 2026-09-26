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
 * \file unique_launch.h
 * \brief Host-side (g++) declarations for the Unique operator.
 *
 * The device side implements a "presence bitmap + rank" dense route for every dtype whose monotone
 * key space is bounded (uint8 / int8 / fp16 / bf16 / int32 / int64 with a small value span) and an
 * LSD radix route for float32 (unbounded key space).
 *
 * Host responsibilities are limited to:
 *   - tiling parameters (block count / per block element count / tile size),
 *   - device workspace allocation,
 *   - reading back the single scalar "number of unique values" (the output length of y is data
 *     dependent, so the size of y cannot be known without it),
 *   - launching the kernels.
 * No tensor element is ever computed on the host.
 */

#ifndef UNIQUE_LAUNCH_H
#define UNIQUE_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling: returns (numBlocks, blockLength, tileElems)
std::tuple<int64_t, int64_t, int64_t> calc_unique_tiling(int64_t numel);

// Bits per core bitmap region (words of 32 bits). One region per block.
constexpr int64_t UNIQUE_MAX_BMP_WORDS = 8192;

extern "C" {

#define UNIQUE_STAGE1_DECL(NAME)                                                                        \
    void unique_stage1_##NAME(GM_ADDR x, GM_ADDR inverse, GM_ADDR wsPart, GM_ADDR wsMM,                 \
                              GM_ADDR wsBmpCore, GM_ADDR wsBmpGlobal, GM_ADDR wsK, int64_t numel,       \
                              int64_t numBlocks, int64_t blockLen, uint32_t tileElems, uint32_t bmpWords, \
                              int64_t needInverse, void *stream);

#define UNIQUE_EMIT_DECL(NAME)                                                                          \
    void unique_emit_##NAME(GM_ADDR y, GM_ADDR wsMM, GM_ADDR wsBmpGlobal, GM_ADDR wsK, uint32_t bmpWords, \
                            void *stream);

UNIQUE_STAGE1_DECL(u8)
UNIQUE_STAGE1_DECL(i8)
UNIQUE_STAGE1_DECL(u16)
UNIQUE_STAGE1_DECL(i32)
UNIQUE_STAGE1_DECL(i64)

UNIQUE_EMIT_DECL(u8)
UNIQUE_EMIT_DECL(i8)
UNIQUE_EMIT_DECL(u16)
UNIQUE_EMIT_DECL(i32)
UNIQUE_EMIT_DECL(i64)

// float32 route: sharded dense presence bitmap over the monotone 32 bit key.
// stage pre   : per block key min/max (+ final reduce)
// stage bitset: each block owns a key shard and sets its presence bits, reporting the shard's
//               distinct count
// stage scan  : single block - per shard rank bases, total distinct count K and the -0.0/+0.0 flag
// stage pref  : per block exclusive prefix count at every 16-word block of its shard bitmap
// stage inv   : inverse index
// stage out   : y
void unique_f32_pre(GM_ADDR x, GM_ADDR wsPart, GM_ADDR wsMM, int64_t numel, int64_t numBlocks,
                    int64_t blockLen, uint32_t tileElems, void *stream);
void unique_f32_bitset(GM_ADDR x, GM_ADDR bmp, GM_ADDR wsMM, GM_ADDR cnt, int64_t numel,
                       int64_t numBlocks, uint32_t tileElems, int64_t shardWords, void *stream);
void unique_f32_scan(GM_ADDR cnt, GM_ADDR base, GM_ADDR wsK, GM_ADDR wsMM, GM_ADDR bmp,
                     int64_t numBlocks, int64_t keyMin, int64_t words, void *stream);
void unique_f32_pref(GM_ADDR bmp, GM_ADDR base, GM_ADDR prefix, int64_t numBlocks, int64_t shardWords,
                     void *stream);
void unique_f32_inv(GM_ADDR x, GM_ADDR inverse, GM_ADDR bmp, GM_ADDR prefix, GM_ADDR wsMM,
                    GM_ADDR wsK, int64_t numel, int64_t numBlocks, int64_t blockLen,
                    uint32_t tileElems, int64_t keyMin, void *stream);
void unique_f32_out(GM_ADDR y, GM_ADDR bmp, GM_ADDR base, GM_ADDR wsMM, GM_ADDR wsK,
                    int64_t numBlocks, int64_t shardWords, void *stream);
}

#endif // UNIQUE_LAUNCH_H
