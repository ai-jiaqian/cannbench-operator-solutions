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
 * \file apply_adam_w_launch.h
 * \brief ApplyAdamW launch declarations (plain C++, shared by the bisheng kernel and the g++ plugin).
 *
 * All step-dependent scalar coefficients are folded on the caller side so the kernel body is a
 * pure elementwise chain:
 *   m_hat  = a1 * m + c1 * grad                 (a1/c1 already carry the sign and scale of lr)
 *   v_hat  = a2 * v + c2 * grad * grad
 *   update = m_hat / (sqrt(v_hat) + epsilon)    (epsilon is OUTSIDE the sqrt)
 *   y      = var * scaleX + update              (scaleX == 1 when weight_decay == 0)
 */

#ifndef APPLY_ADAM_W_LAUNCH_H
#define APPLY_ADAM_W_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host tiling: returns (numBlocks, blockLength, tileElems).
std::tuple<int64_t, int64_t, int64_t> calc_apply_adam_w_tiling_params(int64_t totalLength, int64_t elemBytes);

extern "C" {

void launch_apply_adam_w_kernel_float(
    GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems,
    float a1, float c1, float a2, float c2, float epsilon, float scaleX, int64_t hasWd,
    void* stream);

void launch_apply_adam_w_kernel_half(
    GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems,
    float a1, float c1, float a2, float c2, float epsilon, float scaleX, int64_t hasWd,
    void* stream);

void launch_apply_adam_w_kernel_bfloat16(
    GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems,
    float a1, float c1, float a2, float c2, float epsilon, float scaleX, int64_t hasWd,
    void* stream);

} // extern "C"

// Convenience macros for the API layer
#define APPLY_ADAM_W_LAUNCH_FLOAT(var, grad, m, v, y, total, nb, bl, tile, a1, c1, a2, c2, eps, sc, wd, stream) \
    launch_apply_adam_w_kernel_float(var, grad, m, v, y, total, nb, bl, tile, a1, c1, a2, c2, eps, sc, wd, stream)

#define APPLY_ADAM_W_LAUNCH_HALF(var, grad, m, v, y, total, nb, bl, tile, a1, c1, a2, c2, eps, sc, wd, stream) \
    launch_apply_adam_w_kernel_half(var, grad, m, v, y, total, nb, bl, tile, a1, c1, a2, c2, eps, sc, wd, stream)

#define APPLY_ADAM_W_LAUNCH_BF16(var, grad, m, v, y, total, nb, bl, tile, a1, c1, a2, c2, eps, sc, wd, stream) \
    launch_apply_adam_w_kernel_bfloat16(var, grad, m, v, y, total, nb, bl, tile, a1, c1, a2, c2, eps, sc, wd, stream)

#endif // APPLY_ADAM_W_LAUNCH_H
