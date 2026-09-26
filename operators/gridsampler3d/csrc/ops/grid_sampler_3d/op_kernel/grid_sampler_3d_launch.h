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
 * \file grid_sampler_3d_launch.h
 * \brief Launch declarations for GridSampler3D (visible to the g++ plugin TU).
 */

#ifndef GRID_SAMPLER_3D_LAUNCH_H
#define GRID_SAMPLER_3D_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

// tiling: (Cp, padFront, nb1, tpc1, nb2, ppc2, nb3, tpc3)
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
calc_grid_sampler_3d_tiling(int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, int64_t OD,
                            int64_t OH, int64_t OW);

// mode: 0 = bilinear (trilinear), 1 = nearest
// padMode: 0 = zeros, 1 = border, 2 = reflection
extern "C" {

void launch_grid_sampler_3d_float(GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR xT, GM_ADDR yT,
                                  int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, int64_t OD,
                                  int64_t OH, int64_t OW, int64_t Cp, int64_t padFront, int64_t mode,
                                  int64_t padMode, int64_t alignC, int64_t nb1, int64_t tpc1, int64_t nb2,
                                  int64_t ppc2, int64_t nb3, int64_t tpc3, int64_t xTpitch, void *stream);

void launch_grid_sampler_3d_half(GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR xT, GM_ADDR yT,
                                 int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, int64_t OD,
                                 int64_t OH, int64_t OW, int64_t Cp, int64_t padFront, int64_t mode,
                                 int64_t padMode, int64_t alignC, int64_t nb1, int64_t tpc1, int64_t nb2,
                                 int64_t ppc2, int64_t nb3, int64_t tpc3, int64_t xTpitch, void *stream);
}

#endif  // GRID_SAMPLER_3D_LAUNCH_H
