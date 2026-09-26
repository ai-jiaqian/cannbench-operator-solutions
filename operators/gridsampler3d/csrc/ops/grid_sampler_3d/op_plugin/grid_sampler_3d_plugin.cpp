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
 * \file grid_sampler_3d_plugin.cpp
 * \brief GridSampler3D API layer - torch bindings (compiled with g++)
 */

#include <string>
#include <tuple>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/grid_sampler_3d_launch.h"

namespace cann_bench {

namespace {

int64_t GsPadMode(const std::string &m)
{
    if (m == "zeros") {
        return 0;
    }
    if (m == "border") {
        return 1;
    }
    if (m == "reflection") {
        return 2;
    }
    TORCH_CHECK(false, "grid_sampler_3d: unsupported padding_mode '", m, "'.");
    return 0;
}

int64_t GsInterpMode(const std::string &m)
{
    if (m == "bilinear") {
        return 0;
    }
    if (m == "nearest") {
        return 1;
    }
    TORCH_CHECK(false, "grid_sampler_3d: unsupported interpolation_mode '", m,
                "' (bicubic is not defined for 5-D input).");
    return 0;
}

void GsValidate(const torch::Tensor &x, const torch::Tensor &grid)
{
    TORCH_CHECK(x.dim() == 5, "grid_sampler_3d: x must be 5-D (N, C, D, H, W).");
    TORCH_CHECK(grid.dim() == 5, "grid_sampler_3d: grid must be 5-D (N, Do, Ho, Wo, 3).");
    TORCH_CHECK(grid.size(4) == 3, "grid_sampler_3d: grid last dim must be 3.");
    TORCH_CHECK(x.size(0) == grid.size(0), "grid_sampler_3d: batch size mismatch.");
    TORCH_CHECK(x.scalar_type() == grid.scalar_type(),
                "grid_sampler_3d: x and grid must have the same dtype.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kFloat32,
                "grid_sampler_3d: only float16 / float32 are supported.");
    TORCH_CHECK(x.device() == grid.device(), "grid_sampler_3d: x and grid must share one device.");
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("grid_sampler_3d(Tensor x, Tensor grid, str interpolation_mode=\"bilinear\", "
          "str padding_mode=\"zeros\", bool align_corners=False) -> Tensor");
}

torch::Tensor grid_sampler_3d_meta(const torch::Tensor &x, const torch::Tensor &grid,
                                   std::string interpolation_mode, std::string padding_mode,
                                   bool align_corners)
{
    (void)interpolation_mode;
    (void)padding_mode;
    (void)align_corners;
    GsValidate(x, grid);
    return torch::empty({x.size(0), x.size(1), grid.size(1), grid.size(2), grid.size(3)},
                        x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("grid_sampler_3d", grid_sampler_3d_meta);
}

torch::Tensor grid_sampler_3d_npu(const torch::Tensor &x, const torch::Tensor &grid,
                                  std::string interpolation_mode, std::string padding_mode,
                                  bool align_corners)
{
    GsValidate(x, grid);
    const c10::OptionalDeviceGuard guard(x.device());

    auto xc = x.is_contiguous() ? x : x.contiguous();
    auto gc = grid.is_contiguous() ? grid : grid.contiguous();

    const int64_t N = xc.size(0);
    const int64_t C = xc.size(1);
    const int64_t D = xc.size(2);
    const int64_t H = xc.size(3);
    const int64_t W = xc.size(4);
    const int64_t OD = gc.size(1);
    const int64_t OH = gc.size(2);
    const int64_t OW = gc.size(3);

    TORCH_CHECK(N > 0 && C > 0 && D > 0 && H > 0 && W > 0, "grid_sampler_3d: empty input.");
    TORCH_CHECK(OD > 0 && OH > 0 && OW > 0, "grid_sampler_3d: empty grid.");

    const int64_t DHW = D * H * W;
    const int64_t OS = OD * OH * OW;
    TORCH_CHECK(DHW >= 16 && OS >= 16,
                "grid_sampler_3d: the flattened spatial axes must hold at least 16 elements.");

    int64_t Cp = 0, padFront = 0, nb1 = 0, tpc1 = 0, nb2 = 0, ppc2 = 0, nb3 = 0, tpc3 = 0;
    std::tie(Cp, padFront, nb1, tpc1, nb2, ppc2, nb3, tpc3) =
        calc_grid_sampler_3d_tiling(N, C, D, H, W, OD, OH, OW);

    auto y = torch::empty({N, C, OD, OH, OW}, xc.options());
    // channel-last staging of the input and of the sampled output (padded channel axis,
    // plus head/tail room so that x0 == -1 and the last row stay inside the buffer)
    auto xT = torch::empty({padFront + N * DHW * Cp + 16 * Cp + 64}, xc.options());
    auto yT = torch::empty({N * OS * Cp}, xc.options());

    const int64_t padMode = GsPadMode(padding_mode);
    const int64_t mode = GsInterpMode(interpolation_mode);
    const int64_t alignC = align_corners ? 1 : 0;

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x_ptr = (GM_ADDR)xc.data_ptr();
    auto g_ptr = (GM_ADDR)gc.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto xt_ptr = (GM_ADDR)xT.data_ptr();
    auto yt_ptr = (GM_ADDR)yT.data_ptr();
    const auto dtype = xc.scalar_type();
    const int64_t xTpitch = Cp;

    auto acl_call = [=]() -> int {
        if (dtype == torch::kFloat32) {
            launch_grid_sampler_3d_float(x_ptr, g_ptr, y_ptr, xt_ptr, yt_ptr, N, C, D, H, W, OD, OH,
                                         OW, Cp, padFront, mode, padMode, alignC, nb1, tpc1, nb2, ppc2,
                                         nb3, tpc3, xTpitch, stream);
        } else {
            launch_grid_sampler_3d_half(x_ptr, g_ptr, y_ptr, xt_ptr, yt_ptr, N, C, D, H, W, OD, OH,
                                        OW, Cp, padFront, mode, padMode, alignC, nb1, tpc1, nb2, ppc2,
                                        nb3, tpc3, xTpitch, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GridSampler3D", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("grid_sampler_3d", grid_sampler_3d_npu);
}

}  // namespace cann_bench
