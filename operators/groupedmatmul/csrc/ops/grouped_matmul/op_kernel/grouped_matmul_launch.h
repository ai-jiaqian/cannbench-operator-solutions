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
 * \file grouped_matmul_launch.h
 * \brief Shared declarations for the GroupedMatmul operator (op_kernel <-> op_plugin).
 *
 * Math (per expert g, rows_g = [gl[g-1], gl[g]) with gl[-1] = 0):
 *     y[rows_g, :] = x[rows_g, :] @ weight[g] (+ bias[g])
 *
 * This header is included from both the bisheng translation unit (kernel + host launch code) and the g++
 * translation unit (torch plugin), so it must stay plain C++ and must not pull in any Ascend C header.
 *
 * The cumsum boundaries of group_list travel to the device by value: group_list is an operator attribute
 * (a Python list) and kernel-direct-invoke has no GM tiling blob, so there is nothing to load them from.
 * The plan (row/column chunking, task count) and the Matmul tiling are both derived inside the bisheng
 * translation unit from these boundaries plus N / K, so no host side tensor read-back is ever needed.
 */

#ifndef GROUPED_MATMUL_LAUNCH_H
#define GROUPED_MATMUL_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

//! Maximum number of experts carried by value to the device kernel.
#define GMM_MAX_E 64

//! dtype codes used across the C ABI.
#define GMM_DT_HALF 0
#define GMM_DT_BF16 1
#define GMM_DT_FLOAT 2

/*! \brief By-value cumsum boundaries: ends[g] is the last token row (exclusive) of expert g, ends[-1] = 0. */
struct GmmGroupArg {
    int32_t count;
    int32_t ends[GMM_MAX_E];
};

extern "C" {

/*! System workspace size (bytes) required by the Matmul high level API. */
int64_t calc_gmm_workspace_size();

/*!
 * \brief GroupedMatmul: for every expert g, y[rows_g] = x[rows_g] @ weight[g] (+ bias[g]).
 *
 * \param x        [M, K]
 * \param w        [E, K, N] when transB == 0, [E, N, K] when transB == 1
 * \param b        [E, N] bias or nullptr
 * \param y        [M, N]
 * \param ws       system workspace for the Matmul high level API
 * \param xDtype   GMM_DT_* code of x / weight / y
 * \param biasDtype GMM_DT_* code of bias (ignored when hasBias == 0)
 * \param hasBias  0 / 1
 * \param transB   0 / 1
 */
void launch_gmm(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR ws, int64_t M, int64_t N, int64_t K,
                GmmGroupArg gl, int32_t xDtype, int32_t biasDtype, int32_t hasBias, int32_t transB,
                void* stream);
}

#endif  // GROUPED_MATMUL_LAUNCH_H
