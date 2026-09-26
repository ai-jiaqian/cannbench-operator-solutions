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
 * \file foreach_addcdiv_scalar_launch.h
 * \brief Launch + tiling declarations for g++ (host side, plain C++ only)
 */

#ifndef FOREACH_ADDCDIV_SCALAR_LAUNCH_H
#define FOREACH_ADDCDIV_SCALAR_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling: returns (numBlocks, blockLength, tileElems)
std::tuple<int64_t, int64_t, int64_t> calc_foreach_addcdiv_scalar_tiling_params(int64_t totalLength,
                                                                                int64_t dtypeSize);

// Launch function declarations (one per supported element dtype)
extern "C" {
void launch_foreach_addcdiv_scalar_float(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                         int64_t totalLength, int64_t numBlocks, int64_t blockLength,
                                         uint32_t tileElems, float scalar, void *stream);
void launch_foreach_addcdiv_scalar_half(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                        int64_t totalLength, int64_t numBlocks, int64_t blockLength,
                                        uint32_t tileElems, float scalar, void *stream);
void launch_foreach_addcdiv_scalar_bfloat16(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                            int64_t totalLength, int64_t numBlocks, int64_t blockLength,
                                            uint32_t tileElems, float scalar, void *stream);
}

// Convenience macros for the API layer
#define FOREACH_ADDCDIV_SCALAR_LAUNCH_FLOAT(x1, x2, x3, y, len, blocks, blkLen, tile, sc, stream) \
    launch_foreach_addcdiv_scalar_float(x1, x2, x3, y, len, blocks, blkLen, tile, sc, stream)

#define FOREACH_ADDCDIV_SCALAR_LAUNCH_HALF(x1, x2, x3, y, len, blocks, blkLen, tile, sc, stream) \
    launch_foreach_addcdiv_scalar_half(x1, x2, x3, y, len, blocks, blkLen, tile, sc, stream)

#define FOREACH_ADDCDIV_SCALAR_LAUNCH_BFLOAT16(x1, x2, x3, y, len, blocks, blkLen, tile, sc, stream) \
    launch_foreach_addcdiv_scalar_bfloat16(x1, x2, x3, y, len, blocks, blkLen, tile, sc, stream)

#endif // FOREACH_ADDCDIV_SCALAR_LAUNCH_H
