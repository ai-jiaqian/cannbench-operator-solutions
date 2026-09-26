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
 * \file quant_matmul_launch.h
 * \brief Host/kernel shared declarations for QuantMatmul
 *        y = cast( ((x1 @ x2) + i32bias?) * scale + offset? , out dtype ) * pertoken? + floatbias?
 *
 * Data flow: AIC Pure-Cube kernel computes the int8 x int8 -> int32 product into a GM workspace;
 * the AIV epilogue kernel applies the whole (pre-scale bias / scale / offset / pertoken /
 * post-scale bias) chain in float32 and performs the single narrowing cast to the output dtype.
 */

#ifndef QUANT_MATMUL_LAUNCH_H
#define QUANT_MATMUL_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

namespace qm {

// TCubeTiling is a plain POD of 50 int32 (200 bytes) in the kernel-side kernel_tiling.h.
// We carry it as a 64-slot int32 blob so the launch signature never depends on the
// host-side framework class layout.
constexpr int32_t QM_TILING_INTS = 64;
constexpr int32_t QM_TILING_BYTES = 200;

// epilogue feature flags
constexpr int64_t QM_F_SCALE_PERCHAN = 1LL << 0;
constexpr int64_t QM_F_SCALE_BF16 = 1LL << 1;
constexpr int64_t QM_F_HAS_OFFSET = 1LL << 2;
constexpr int64_t QM_F_OFFSET_PERCHAN = 1LL << 3;
constexpr int64_t QM_F_HAS_PT = 1LL << 4;
constexpr int64_t QM_F_BIAS_SHIFT = 5;
constexpr int64_t QM_F_BIAS_MASK = 7LL << QM_F_BIAS_SHIFT;
constexpr int64_t QM_F_BIAS_PERCHAN = 1LL << 8;
constexpr int64_t QM_F_PT_MOD_M = 1LL << 9;

constexpr int64_t QM_BIAS_NONE = 0;
constexpr int64_t QM_BIAS_I32 = 1;
constexpr int64_t QM_BIAS_F16 = 2;
constexpr int64_t QM_BIAS_BF16 = 3;
constexpr int64_t QM_BIAS_F32 = 4;

// by-value kernel argument block for the cube kernel
struct QmCubeArgs {
    int32_t tiling[QM_TILING_INTS];
    int64_t M;
    int64_t N;
    int64_t K;
    int64_t batch;      // leading batch count (A/B/C carry a batch stride)
    int64_t numBlocks;  // total blocks to launch = batch * nr * nc
    int64_t nr;   // row blocks per batch
    int64_t nc;   // column blocks per batch
    int64_t scM;  // rows per block
    int64_t scN;  // columns per block
};

// by-value kernel argument block for the epilogue kernel
struct QmEpiArgs {
    int64_t rows;       // number of output rows processed by this launch
    int64_t M;          // true M (for the pertoken index when rows covers several batches)
    int64_t N;
    int64_t flags;
    int64_t numBlocks;
};

}  // namespace qm

extern "C" {

// Host-side tiling helper (compiled into the bisheng kernel translation unit).
// Fills tilingOut with the serialized single-core TCubeTiling and returns the number of AIC
// blocks the manual grid wants for this shape.
// tilingInfo (int64): [0]=libWs bytes [1]=coreNumAic [2]=coreNumAiv [3]=numBlocks
//                     [4]=nr [5]=nc [6]=scM [7]=scN
int64_t qm_calc_cube_tiling(int64_t M, int64_t N, int64_t K, int64_t nAicHint,
                            int32_t* tilingOut, int64_t* tilingInfo);

// Kernel 1: Pure-Cube int8 x int8 -> int32 into cOut.
void qm_launch_cube_i32(GM_ADDR x1, GM_ADDR x2, GM_ADDR cOut, GM_ADDR sysWs,
                        const qm::QmCubeArgs* args, void* stream);

// Kernel 2: generic dequant epilogue writing float16 / bfloat16 output.
void qm_launch_epilogue_f16(GM_ADDR cIn, GM_ADDR out, GM_ADDR scale, GM_ADDR offset,
                            GM_ADDR pt, GM_ADDR bias, const qm::QmEpiArgs* args, void* stream);
void qm_launch_epilogue_bf16(GM_ADDR cIn, GM_ADDR out, GM_ADDR scale, GM_ADDR offset,
                             GM_ADDR pt, GM_ADDR bias, const qm::QmEpiArgs* args, void* stream);
}

#endif  // QUANT_MATMUL_LAUNCH_H
