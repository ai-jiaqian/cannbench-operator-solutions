/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file softmax_launch.h
 * \brief Launch function declarations for g++ (Softmax)
 */

#ifndef SOFTMAX_LAUNCH_H
#define SOFTMAX_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host tiling result (filled by calc_softmax_tiling, defined in the bisheng TU)
struct SoftmaxTilingOut {
    int64_t mode;         // 0 = K1-std, 1 = K1-packed, 2 = K2-colBlock (incl. per-row fallback)
    int64_t numBlocks;    // cores to launch
    int64_t rowsPerCore;  // K1: rows per core; K2: units per core
    int64_t G;            // K1: rows per tile; K2: rT rows per chunk
    int64_t Kpad;         // K1: padded row width (elements); K2 unused
    int64_t tmpSize;      // K1: SoftMax shared tmp buffer bytes; K2 unused
    int64_t tp0;          // K1: packed SoftMaxTiling (unused); K2: strategy flags
                          // (bit0 twod_pad ragged loads, bit1 online two-pass,
                          //  bit2 twod_pad ragged stores, bit3 whole-lane resident,
                          //  bit4 per-row resident realization,
                          //  bit5 chunk-level two-pass streaming)
    int64_t tp1;
    int64_t tp2;
    int64_t tp3;
    int64_t tp4;
    int64_t tp5;
    int64_t tp6;
    int64_t tp7;
    // K2 fields
    int64_t outer;
    int64_t R;
    int64_t inner;
    int64_t W;            // full column-block width (elements)
    int64_t lastW;        // last column-block width (elements)
    int64_t nB;           // number of column blocks
    int64_t splitG;       // K2 split-R / cooperative group size g (0 when neither is used)
    int64_t splitGroups;  // K2 cooperative-resident group count (0 when the cooperative
                          // realization is not used)
    int64_t wsBytes;      // K2 split-R / cooperative GM workspace bytes (0 when neither is used)
};

// Tiling calculation (defined in the bisheng-compiled kernel TU)
void calc_softmax_tiling(int64_t rank, int64_t dim, const int64_t* shape, int64_t esize,
                         SoftmaxTilingOut* out);

// Launch function declarations
extern "C" {
// mode 0: K1-std
void launch_softmax_k1_std_float(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream);
void launch_softmax_k1_std_half(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream);
void launch_softmax_k1_std_bf16(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream);

// mode 1: K1-packed
void launch_softmax_k1_packed_float(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream);
void launch_softmax_k1_packed_half(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream);
void launch_softmax_k1_packed_bf16(GM_ADDR x, GM_ADDR z, int64_t numBlocks, int64_t m, int64_t R,
    int64_t G, int64_t Kpad, int64_t rowsPerCore, int64_t tmpSize, void* stream);

// mode 2: K2-colBlock (k2Flags: bit0 = twod_pad ragged loads, bit1 = online two-pass,
//          bit2 = twod_pad ragged stores, bit3 = whole-lane resident reduction,
//          bit4 = per-row resident realization, bit6 = split-R cooperating group,
//          bit7 = cooperative whole-lane resident group)
//          ws is the split-R / cooperative GM workspace (may be null when neither is
//          selected); splitG is the split-R / cooperative group size g (ignored when
//          neither is selected); splitGroups is the cooperative group count (ignored
//          when bit7 is clear).
void launch_softmax_k2_float(GM_ADDR x, GM_ADDR z, GM_ADDR ws, int64_t numBlocks, int64_t outer,
    int64_t R, int64_t inner, int64_t rT, int64_t W, int64_t lastW, int64_t nB, int64_t unitsPerCore,
    int64_t k2Flags, int64_t splitG, int64_t splitGroups, void* stream);
void launch_softmax_k2_half(GM_ADDR x, GM_ADDR z, GM_ADDR ws, int64_t numBlocks, int64_t outer,
    int64_t R, int64_t inner, int64_t rT, int64_t W, int64_t lastW, int64_t nB, int64_t unitsPerCore,
    int64_t k2Flags, int64_t splitG, int64_t splitGroups, void* stream);
void launch_softmax_k2_bf16(GM_ADDR x, GM_ADDR z, GM_ADDR ws, int64_t numBlocks, int64_t outer,
    int64_t R, int64_t inner, int64_t rT, int64_t W, int64_t lastW, int64_t nB, int64_t unitsPerCore,
    int64_t k2Flags, int64_t splitG, int64_t splitGroups, void* stream);
}

#endif // SOFTMAX_LAUNCH_H
