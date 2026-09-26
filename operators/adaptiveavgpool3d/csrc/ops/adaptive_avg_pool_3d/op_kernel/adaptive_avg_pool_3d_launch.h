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
 * \file adaptive_avg_pool_3d_launch.h
 * \brief Shared declarations between the g++ plugin and the bisheng kernel TU.
 *
 * The host tiling helper is declared `extern "C"` so the g++ plugin and the
 * bisheng-compiled host code share one unmangled symbol (a mangled mismatch
 * would still link -- undefined symbols are allowed in a shared object -- and
 * then abort the process on the first call).
 */

#ifndef ADAPTIVE_AVG_POOL_3D_LAUNCH_H
#define ADAPTIVE_AVG_POOL_3D_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

extern "C" {

/*! \brief Host tiling: fills core count, per-core task count and the UB byte size. */
void calc_aap3d_tiling(int64_t totalTasks, int64_t* outBlocks, int64_t* outTasksPerCore,
                       int64_t* outUbBytes);

void launch_aap3d_kernel_float(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D, int64_t H,
                               int64_t W, int64_t OD, int64_t OH, int64_t OW, int64_t numBlocks,
                               int64_t tasksPerCore, int64_t ubBytes, void* stream);

void launch_aap3d_kernel_half(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D, int64_t H,
                              int64_t W, int64_t OD, int64_t OH, int64_t OW, int64_t numBlocks,
                              int64_t tasksPerCore, int64_t ubBytes, void* stream);

void launch_aap3d_kernel_bf16(GM_ADDR x, GM_ADDR y, int64_t N, int64_t C, int64_t D, int64_t H,
                              int64_t W, int64_t OD, int64_t OH, int64_t OW, int64_t numBlocks,
                              int64_t tasksPerCore, int64_t ubBytes, void* stream);
}

#endif  // ADAPTIVE_AVG_POOL_3D_LAUNCH_H
