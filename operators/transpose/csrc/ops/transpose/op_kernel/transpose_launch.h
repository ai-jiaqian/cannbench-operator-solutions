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
 * \file transpose_launch.h
 * \brief Host tiling struct + launch declarations shared by the kernel TU (bisheng) and the plugin TU (g++).
 */

#ifndef TRANSPOSE_LAUNCH_H
#define TRANSPOSE_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Plain aggregate of int64_t so the two translation units agree on the ABI.
struct TPParams {
    int64_t n;       // rank (2..8)
    int64_t d0;      // sizes[0] | sizes[1] << 32
    int64_t d1;      // sizes[2] | sizes[3] << 32
    int64_t d2;      // sizes[4] | sizes[5] << 32
    int64_t d3;      // sizes[6] | sizes[7] << 32
    int64_t pp;      // perm packed 4 bits per entry
    int64_t mode;    // 0 = pure row permutation (U == 1), 1 = gather transpose
    int64_t TC;      // column tile width (elements)
    int64_t K;       // run tile length (elements, mode 1)
    int64_t Pk;      // UB pitch of a read row, in gather units
    int64_t Pj;      // UB pitch of a gather-dst row, in gather units
    int64_t nch;     // mode 1: run chunks per group
    int64_t ncc;     // column chunks
    int64_t nlc;     // mode 1: total run chunks
    int64_t nTiles;  // total tiles
    int64_t nBlocks; // grid size
    int64_t RA;      // mode 0: rows per tile
    int64_t numel;
    int64_t Ta;      // mode 1: values of the innermost run dim covered by one tile
    int64_t Tb;      // mode 1: values of the second run dim covered by one tile
    int64_t packed;  // mode 1: 1 = the tile's output region is one contiguous block
    int64_t overA;   // mode 1: 1 = the read DMA strides over the second run dim
    int64_t NB;      // mode 1 packed: number of consecutive h values handled per queue round
};

// Host side tiling computation (compiled by bisheng, called from g++).
void calc_transpose_params(int64_t nd, int64_t esz, int64_t numel, const int64_t* sizes,
                           const int64_t* perm, TPParams* out);

extern "C" {

#define TP_LAUNCH_DECL(NAME)                                                                        \
    void NAME(GM_ADDR x, GM_ADDR y, int64_t n, int64_t d0, int64_t d1, int64_t d2, int64_t d3,      \
              int64_t pp, int64_t mode, int64_t TC, int64_t K, int64_t Pk, int64_t Pj, int64_t nch,  \
              int64_t ncc, int64_t nlc, int64_t nTiles, int64_t nBlocks, int64_t RA, int64_t numel,  \
              int64_t Ta, int64_t Tb, int64_t packed, int64_t overA, int64_t NB, void* stream);

TP_LAUNCH_DECL(launch_transpose_f32)
TP_LAUNCH_DECL(launch_transpose_f16)
TP_LAUNCH_DECL(launch_transpose_bf16)
TP_LAUNCH_DECL(launch_transpose_i8)
TP_LAUNCH_DECL(launch_transpose_i16)
TP_LAUNCH_DECL(launch_transpose_i32)
TP_LAUNCH_DECL(launch_transpose_i64)

#undef TP_LAUNCH_DECL
}

#endif // TRANSPOSE_LAUNCH_H
