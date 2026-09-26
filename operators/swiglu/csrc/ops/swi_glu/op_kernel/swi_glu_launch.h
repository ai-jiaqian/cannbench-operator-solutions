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
 * \file swi_glu_launch.h
 * \brief Launch function declarations for g++ (SwiGlu)
 */

#ifndef SWI_GLU_LAUNCH_H
#define SWI_GLU_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling function declaration (defined in the bisheng-compiled kernel TU)
std::tuple<int64_t, int64_t, int64_t> calc_swi_glu_tiling(int64_t A, int64_t L, int64_t esize);

// Launch function declarations
extern "C" {
void launch_swi_glu_float(GM_ADDR x, GM_ADDR y, int64_t A, int64_t L, int64_t numBlocks, int64_t workPerCore, uint32_t tileSize, void* stream);
void launch_swi_glu_half(GM_ADDR x, GM_ADDR y, int64_t A, int64_t L, int64_t numBlocks, int64_t workPerCore, uint32_t tileSize, void* stream);
void launch_swi_glu_bf16(GM_ADDR x, GM_ADDR y, int64_t A, int64_t L, int64_t numBlocks, int64_t workPerCore, uint32_t tileSize, void* stream);
}

// Convenience macros for API layer
#define SWI_GLU_KERNEL_LAUNCH_FLOAT(x, y, a, l, blocks, wpc, tileSz, stream) \
    launch_swi_glu_float(x, y, a, l, blocks, wpc, tileSz, stream)

#define SWI_GLU_KERNEL_LAUNCH_HALF(x, y, a, l, blocks, wpc, tileSz, stream) \
    launch_swi_glu_half(x, y, a, l, blocks, wpc, tileSz, stream)

#define SWI_GLU_KERNEL_LAUNCH_BF16(x, y, a, l, blocks, wpc, tileSz, stream) \
    launch_swi_glu_bf16(x, y, a, l, blocks, wpc, tileSz, stream)

#endif // SWI_GLU_LAUNCH_H
