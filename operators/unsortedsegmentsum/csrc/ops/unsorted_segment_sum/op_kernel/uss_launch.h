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
 * \file uss_launch.h
 * \brief UnsortedSegmentSum launch / tiling declarations shared between the bisheng kernel translation
 *        unit and the g++ plugin translation unit.  Plain C++ only (no device types).
 */

#ifndef USS_LAUNCH_H
#define USS_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

// mode 0 "owner"     : the output space is split into (segment, inner-chunk) units.  Every core owns a
//                      contiguous unit range, stably buckets its rows per segment (counting sort, so the
//                      in-segment row order is the reference's ascending order) and writes the final
//                      (zero filled for empty segments) results straight to y.  No workspace.
// mode 1 "partition" : every core owns a contiguous row range and accumulates into a private UB
//                      accumulator of numSeg elements (used for inner == 1); the per-block partials are
//                      published to a workspace that a small second kernel folds into y.
struct UssTiling {
    int64_t mode;
    int64_t N;
    int64_t inner;
    int64_t numSeg;
    int64_t numBlocks;
    // mode 0
    int64_t chunkElems;
    int64_t nChunks;
    int64_t unitsPerCore;
    int64_t segArrMax;
    int64_t idTile;
    int64_t cap;
    // mode 1
    int64_t rowsPerCore;
    int64_t rowTile;
    int64_t wsElems;
    int64_t redBlocks;
};

// Host tiling for one UnsortedSegmentSum invocation.  szT / szId / szA are element sizes.
UssTiling calc_uss_tiling(int64_t N, int64_t inner, int64_t numSeg, int64_t szT, int64_t szId, int64_t szA);

extern "C" {

// ------------------------------------------------------------------------------------------------
// mode 0 owner kernel, one entry point per (data dtype, ids dtype) pair.
// ------------------------------------------------------------------------------------------------
#define USS_OWNER_DECL(TAG)                                                                                 \
    void launch_uss_owner_##TAG(GM_ADDR x, GM_ADDR ids, GM_ADDR y, int64_t N, int64_t inner, int64_t numSeg, \
                                int64_t chunkElems, int64_t nChunks, int64_t unitsPerCore, int64_t segArrMax, \
                                int64_t idTile, int64_t cap, void *stream);

USS_OWNER_DECL(f16_i32)
USS_OWNER_DECL(f16_i64)
USS_OWNER_DECL(bf16_i32)
USS_OWNER_DECL(bf16_i64)
USS_OWNER_DECL(f32_i32)
USS_OWNER_DECL(f32_i64)
USS_OWNER_DECL(i32_i32)
USS_OWNER_DECL(i32_i64)
USS_OWNER_DECL(i64_i32)
USS_OWNER_DECL(i64_i64)

// ------------------------------------------------------------------------------------------------
// mode 1 partition kernel (inner == 1)
// ------------------------------------------------------------------------------------------------
#define USS_PART_DECL(TAG)                                                                       \
    void launch_uss_part_##TAG(GM_ADDR x, GM_ADDR ids, GM_ADDR ws, int64_t N, int64_t numSeg,     \
                               int64_t numBlocks, int64_t rowsPerCore, int64_t rowTile, void *stream);

USS_PART_DECL(f16_i32)
USS_PART_DECL(f16_i64)
USS_PART_DECL(bf16_i32)
USS_PART_DECL(bf16_i64)
USS_PART_DECL(f32_i32)
USS_PART_DECL(f32_i64)
USS_PART_DECL(i32_i32)
USS_PART_DECL(i32_i64)
USS_PART_DECL(i64_i32)
USS_PART_DECL(i64_i64)

// ------------------------------------------------------------------------------------------------
// mode 1 fold kernel (A = accumulator dtype, T = output dtype)
// ------------------------------------------------------------------------------------------------
#define USS_RED_DECL(TAG)                                                                     \
    void launch_uss_red_##TAG(GM_ADDR ws, GM_ADDR y, int64_t numSeg, int64_t parts,           \
                              int64_t redBlocks, void *stream);

USS_RED_DECL(f32_f16)
USS_RED_DECL(f32_bf16)
USS_RED_DECL(f32_f32)
USS_RED_DECL(i32_i32)
USS_RED_DECL(i32_i64)

} // extern "C"

#endif // USS_LAUNCH_H
