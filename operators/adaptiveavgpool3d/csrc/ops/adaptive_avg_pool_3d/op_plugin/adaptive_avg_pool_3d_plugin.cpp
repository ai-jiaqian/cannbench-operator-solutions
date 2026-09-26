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
 * \file adaptive_avg_pool_3d_plugin.cpp
 * \brief AdaptiveAvgPool3D API layer - torch bindings (g++)
 *
 * The host only does shape algebra and one kernel launch.  No tensor data is
 * ever read back, no dtype conversion runs through host memory, and no cache is
 * kept between calls.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/adaptive_avg_pool_3d_launch.h"

namespace cann_bench {

namespace {

std::vector<int64_t> Aap3dOutputShape(const torch::Tensor& x,
                                      const std::vector<int64_t>& outputSize)
{
    TORCH_CHECK(x.dim() == 5, "adaptive_avg_pool_3d: x must be 5-D (N, C, D, H, W).");
    TORCH_CHECK(outputSize.size() == 3, "adaptive_avg_pool_3d: output_size must have 3 elements.");
    for (size_t i = 0; i < outputSize.size(); ++i) {
        TORCH_CHECK(outputSize[i] > 0, "adaptive_avg_pool_3d: output_size elements must be > 0.");
    }
    return {x.size(0), x.size(1), outputSize[0], outputSize[1], outputSize[2]};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("adaptive_avg_pool_3d(Tensor x, int[] output_size) -> Tensor");
}

torch::Tensor aap3d_meta(const torch::Tensor& x, const std::vector<int64_t>& outputSize)
{
    return at::empty(Aap3dOutputShape(x, outputSize), x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("adaptive_avg_pool_3d", aap3d_meta);
}

torch::Tensor aap3d_npu(const torch::Tensor& xIn, const std::vector<int64_t>& outputSize)
{
    const c10::OptionalDeviceGuard guard(xIn.device());
    auto outShape = Aap3dOutputShape(xIn, outputSize);

    TORCH_CHECK(xIn.scalar_type() == at::kFloat || xIn.scalar_type() == at::kHalf ||
                    xIn.scalar_type() == at::kBFloat16,
                "adaptive_avg_pool_3d: only float32, float16 and bfloat16 are supported.");

    auto x = xIn.is_contiguous() ? xIn : xIn.contiguous();
    auto y = at::empty(outShape, x.options());

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);
    const int64_t OD = outShape[2];
    const int64_t OH = outShape[3];
    const int64_t OW = outShape[4];
    if (N == 0 || C == 0 || D == 0 || H == 0 || W == 0) {
        return y;
    }

    const int64_t totalTasks = N * C * OD;
    int64_t numBlocks = 1;
    int64_t tasksPerCore = 1;
    int64_t ubBytes = 0;
    calc_aap3d_tiling(totalTasks, &numBlocks, &tasksPerCore, &ubBytes);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = reinterpret_cast<GM_ADDR>(const_cast<void*>(x.data_ptr()));
    auto yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());
    const auto dtype = x.scalar_type();

    auto aclCall = [=]() -> int {
        if (dtype == at::kFloat) {
            launch_aap3d_kernel_float(xPtr, yPtr, N, C, D, H, W, OD, OH, OW, numBlocks,
                                      tasksPerCore, ubBytes, stream);
        } else if (dtype == at::kHalf) {
            launch_aap3d_kernel_half(xPtr, yPtr, N, C, D, H, W, OD, OH, OW, numBlocks,
                                     tasksPerCore, ubBytes, stream);
        } else {
            launch_aap3d_kernel_bf16(xPtr, yPtr, N, C, D, H, W, OD, OH, OW, numBlocks,
                                     tasksPerCore, ubBytes, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("AdaptiveAvgPool3D", aclCall);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("adaptive_avg_pool_3d", aap3d_npu);
}

}  // namespace cann_bench
