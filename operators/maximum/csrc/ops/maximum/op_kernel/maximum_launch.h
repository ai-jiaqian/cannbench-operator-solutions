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
 * \file maximum_launch.h
 * \brief Launch / tiling declarations shared by the bisheng kernel TU and the g++ plugin TU.
 *
 * Maximum: y = max(x1, x2) elementwise with numpy-style broadcasting.
 *
 * Output model used by the device kernel: the flat output is viewed as
 *   rows x runLen  (row-major),  row r covering output elements [r*runLen, (r+1)*runLen).
 * For every row the "A" operand is the contiguous flat segment [r*runLen, (r+1)*runLen)
 * of the (already broadcast-expanded) first operand, and the "B" operand for a tile at
 * column offset `o` is the segment [o, o+len) of the periodic B block bSrc, which is
 * shared by every row.  When bSrc is not directly available in that form the host first
 * materialises it into a workspace with one of the helper kernels below.
 *
 * dtypeCode: 0 = float16, 1 = bfloat16, 2 = float32, 3 = int8, 4 = int32, 5 = int64
 */

#ifndef CANN_BENCH_MAXIMUM_LAUNCH_H
#define CANN_BENCH_MAXIMUM_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Returns (tileElems, numBlocks, ubSize).
std::tuple<int64_t, int64_t, int64_t> calc_maximum_tiling_params(
    int64_t numel, int64_t rows, int64_t runLen, int64_t elemBytes, int64_t dtypeCode);

extern "C" {

// ---- main elementwise kernel ----
void launch_maximum_fp16(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                         int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void* stream);
void launch_maximum_bf16(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                         int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void* stream);
void launch_maximum_fp32(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                         int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void* stream);
void launch_maximum_int8(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                         int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void* stream);
void launch_maximum_int32(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                          int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void* stream);
void launch_maximum_int64(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows, int64_t runLen,
                          int64_t tileElems, int64_t tilesPerRow, int64_t numBlocks, void* stream);

// ---- helpers (raw element-width carriers; pure data movement) ----
// bDst[c * linner + l] = bSrc[c] for c in [0, lc), l in [0, linner)
void launch_maximum_expand_b8(GM_ADDR src, GM_ADDR dst, int64_t lc, int64_t linner,
                              int64_t numBlocks, void* stream);
void launch_maximum_expand_b16(GM_ADDR src, GM_ADDR dst, int64_t lc, int64_t linner,
                               int64_t numBlocks, void* stream);
void launch_maximum_expand_b32(GM_ADDR src, GM_ADDR dst, int64_t lc, int64_t linner,
                               int64_t numBlocks, void* stream);
void launch_maximum_expand_b64(GM_ADDR src, GM_ADDR dst, int64_t lc, int64_t linner,
                               int64_t numBlocks, void* stream);

// dst[i] = src[broadcast_index(i)] with numpy right-aligned broadcast of src (rank r) to
// the output shape.  packed[d] = (outSize[d] << 32) | srcSize[d], d in [0, r).
void launch_maximum_bcast_b8(GM_ADDR src, GM_ADDR dst, int64_t numel, int32_t r,
                             int64_t p0, int64_t p1, int64_t p2, int64_t p3,
                             int64_t p4, int64_t p5, int64_t p6, int64_t p7, void* stream);
void launch_maximum_bcast_b16(GM_ADDR src, GM_ADDR dst, int64_t numel, int32_t r,
                              int64_t p0, int64_t p1, int64_t p2, int64_t p3,
                              int64_t p4, int64_t p5, int64_t p6, int64_t p7, void* stream);
void launch_maximum_bcast_b32(GM_ADDR src, GM_ADDR dst, int64_t numel, int32_t r,
                              int64_t p0, int64_t p1, int64_t p2, int64_t p3,
                              int64_t p4, int64_t p5, int64_t p6, int64_t p7, void* stream);
void launch_maximum_bcast_b64(GM_ADDR src, GM_ADDR dst, int64_t numel, int32_t r,
                              int64_t p0, int64_t p1, int64_t p2, int64_t p3,
                              int64_t p4, int64_t p5, int64_t p6, int64_t p7, void* stream);

} // extern "C"

#endif // CANN_BENCH_MAXIMUM_LAUNCH_H
