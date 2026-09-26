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
 * \file sigmoid_plugin.cpp
 * \brief Sigmoid API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/sigmoid_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("sigmoid(Tensor x) -> Tensor");
}

torch::Tensor sigmoid_meta(const torch::Tensor &x)
{
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                    x.scalar_type() == torch::kBFloat16,
                "cann_bench.sigmoid only supports float32, float16 and bfloat16 inputs.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("sigmoid", sigmoid_meta);
}

torch::Tensor sigmoid_npu(const torch::Tensor &x)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto y = sigmoid_meta(xc);

    const int64_t totalLength = xc.numel();
    int64_t mode = 0;
    if (xc.scalar_type() == torch::kFloat16) {
        mode = 1;
    } else if (xc.scalar_type() == torch::kBFloat16) {
        mode = 2;
    }

    int64_t numBlocks = 1;
    int64_t blockLength = 0;
    int64_t tileElems = 0;
    std::tie(numBlocks, blockLength, tileElems) = calc_sigmoid_tiling_params(totalLength, mode);
    const uint32_t tileElemsU32 = static_cast<uint32_t>(tileElems);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x_ptr = (GM_ADDR)xc.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        if (xc.scalar_type() == torch::kFloat32) {
            launch_sigmoid_kernel_float(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElemsU32, stream);
        } else if (xc.scalar_type() == torch::kFloat16) {
            launch_sigmoid_kernel_half(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElemsU32, stream);
        } else {
            launch_sigmoid_kernel_bfloat16(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElemsU32, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Sigmoid", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("sigmoid", sigmoid_npu);
}

} // namespace cann_bench
