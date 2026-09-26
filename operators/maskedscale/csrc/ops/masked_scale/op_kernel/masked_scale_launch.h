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
 * \file masked_scale_launch.h
 * \brief Launch / tiling declarations for MaskedScale (y = x * mask * scale), visible to g++.
 */

#ifndef MASKED_SCALE_LAUNCH_H
#define MASKED_SCALE_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host tiling function: returns (numBlocks, blockLength, tileElems)
std::tuple<int64_t, int64_t, int64_t> calc_masked_scale_tiling_params(int64_t totalLength,
                                                                     int64_t xBytes,
                                                                     int64_t maskBytes);

extern "C" {

void launch_masked_scale_half_int8(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_half_uint8(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_half_half(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_half_bfloat16(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_half_float(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);

void launch_masked_scale_bfloat16_int8(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_bfloat16_uint8(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_bfloat16_half(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_bfloat16_bfloat16(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_bfloat16_float(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);

void launch_masked_scale_float_int8(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_float_uint8(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_float_half(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_float_bfloat16(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);
void launch_masked_scale_float_float(GM_ADDR x, GM_ADDR mask, GM_ADDR y, int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems, float scale, void* stream);

} // extern "C"

#endif // MASKED_SCALE_LAUNCH_H
