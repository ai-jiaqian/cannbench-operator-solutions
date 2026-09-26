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
 * \file gelu_launch.h
 * \brief Launch function declarations for g++
 */

#ifndef GELU_LAUNCH_H
#define GELU_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling function declaration
// returns: numBlocks, blockLength, tileElementCount
// storageDtype: 0 = float32, 1 = float16, 2 = bfloat16 (host-supplied from x.scalar_type())
std::tuple<int64_t, int64_t, int64_t> calc_gelu_tiling_params(int64_t totalLength, int32_t mode,
    int64_t typeSize, int32_t storageDtype);

// Launch function declarations: one per compile-time (storage dtype, mode)
// specialization. erf = mode 0, tanh = mode 1. The [S10] single-loop coverage
// needs no host-computed bulk/tail split boundary, so no bulkLength parameter.
extern "C" {
void launch_gelu_kernel_float_erf(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength,
    uint32_t tileElementCount, void* stream);
void launch_gelu_kernel_float_tanh(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength,
    uint32_t tileElementCount, void* stream);
void launch_gelu_kernel_half_erf(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength,
    uint32_t tileElementCount, void* stream);
void launch_gelu_kernel_half_tanh(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength,
    uint32_t tileElementCount, void* stream);
void launch_gelu_kernel_bfloat16_erf(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength,
    uint32_t tileElementCount, void* stream);
void launch_gelu_kernel_bfloat16_tanh(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength,
    uint32_t tileElementCount, void* stream);
}

// Convenience macros for API layer
#define GELU_KERNEL_LAUNCH_FLOAT_ERF(x, y, len, blocks, blkLen, tileSz, stream) \
    launch_gelu_kernel_float_erf(x, y, len, blocks, blkLen, tileSz, stream)
#define GELU_KERNEL_LAUNCH_FLOAT_TANH(x, y, len, blocks, blkLen, tileSz, stream) \
    launch_gelu_kernel_float_tanh(x, y, len, blocks, blkLen, tileSz, stream)
#define GELU_KERNEL_LAUNCH_HALF_ERF(x, y, len, blocks, blkLen, tileSz, stream) \
    launch_gelu_kernel_half_erf(x, y, len, blocks, blkLen, tileSz, stream)
#define GELU_KERNEL_LAUNCH_HALF_TANH(x, y, len, blocks, blkLen, tileSz, stream) \
    launch_gelu_kernel_half_tanh(x, y, len, blocks, blkLen, tileSz, stream)
#define GELU_KERNEL_LAUNCH_BFLOAT16_ERF(x, y, len, blocks, blkLen, tileSz, stream) \
    launch_gelu_kernel_bfloat16_erf(x, y, len, blocks, blkLen, tileSz, stream)
#define GELU_KERNEL_LAUNCH_BFLOAT16_TANH(x, y, len, blocks, blkLen, tileSz, stream) \
    launch_gelu_kernel_bfloat16_tanh(x, y, len, blocks, blkLen, tileSz, stream)

#endif // GELU_LAUNCH_H
