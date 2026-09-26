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
 * \file gmsq_launch.h
 * \brief Shared declarations for GroupedMatmulSwigluQuant (op_kernel <-> op_plugin).
 *
 * The fused operator is implemented by two independent device kernels that run back to back on the
 * caller's stream:
 *
 *   1. gmsq_cube_kernel : grouped matmul   x[rows_g] @ weight[g]      int8 x int8 -> int32 workspace
 *   2. gmsq_vec_kernel  : dequant + SwiGLU + per-token requant
 *
 * `group_list` arrives as an operator attribute (a python list), so the cumsum boundaries travel to the
 * device as a small by-value POD: the direct-launch entry point has no GM tiling blob to load them from.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_LAUNCH_H
#define GROUPED_MATMUL_SWIGLU_QUANT_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

//! Maximum number of experts carried by value to the device kernels.
#define GMSQ_MAX_E 64

/*!
 * \brief By-value cumsum boundaries: ends[g] is the exclusive end row of expert g.
 *
 * start(g) = (g == 0) ? 0 : ends[g-1]   and   start(0) = 0, ends[E-1] == M.
 */
struct GmsqGroupArg {
    int32_t count;
    int32_t pad;
    int32_t ends[GMSQ_MAX_E];
};

extern "C" {

/*! Number of cube (AIC) cores of the current device, never 0. */
int64_t gmsq_calc_aic_num();

/*! Number of vector (AIV) cores of the current device, never 0. */
int64_t gmsq_calc_aiv_num();

/*! Size in bytes of the GM workspace the Matmul high level API needs. */
int64_t gmsq_calc_workspace_size();

/*!
 * \brief Grouped matmul stage: for every expert g, c[rows_g, :] = x[rows_g, :] @ w[g].
 *
 * \param x     int8   [M, K]
 * \param w     int8   [E, K, N]
 * \param c     int32  [M, N] result workspace (every element is written)
 * \param ws    system workspace for the Matmul high level API
 */
void launch_gmsq_cube(GM_ADDR x, GM_ADDR w, GM_ADDR c, GM_ADDR ws, int64_t M, int64_t K, int64_t N,
                      int64_t E, GmsqGroupArg gl, void* stream);

/*!
 * \brief Dequant / SwiGLU / requant stage.
 *
 * \param mm      int32 [M, N]  raw grouped matmul result
 * \param xScale  f32   [M]     per-token dequant factor
 * \param wScale  f32   [E, N]  per-channel dequant factor
 * \param y       int8  [M, N/2] output
 * \param yScale  f32   [M]     per-token requant factor
 */
void launch_gmsq_vec(GM_ADDR mm, GM_ADDR xScale, GM_ADDR wScale, GM_ADDR y, GM_ADDR yScale, int64_t M,
                     int64_t N, int64_t E, GmsqGroupArg gl, void* stream);

}  // extern "C"

#endif  // GROUPED_MATMUL_SWIGLU_QUANT_LAUNCH_H
