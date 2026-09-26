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
 * \file dynamic_quant_plugin.cpp
 * \brief DynamicQuant API layer - torch bindings (compiled with g++)
 */

#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/dynamic_quant_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("dynamic_quant(Tensor x) -> (Tensor y, Tensor scale)");
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_quant_meta(const torch::Tensor &x)
{
    TORCH_CHECK(x.dim() >= 2, "dynamic_quant requires a tensor with rank >= 2.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16,
                "dynamic_quant only supports float16 / bfloat16 inputs.");
    std::vector<int64_t> scaleShape(x.sizes().begin(), x.sizes().end() - 1);
    auto y = torch::empty_like(x, x.options().dtype(torch::kChar));
    auto scale = torch::empty(scaleShape, x.options().dtype(torch::kFloat));
    return std::make_tuple(y, scale);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("dynamic_quant", dynamic_quant_meta);
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_quant_npu(const torch::Tensor &x)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto outputs = dynamic_quant_meta(xc);
    auto y = std::get<0>(outputs);
    auto scale = std::get<1>(outputs);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const int64_t N = xc.size(-1);
    TORCH_CHECK(N > 0, "dynamic_quant requires a non-zero last dimension.");
    const int64_t M = xc.numel() / N;
    if (M <= 0) {
        return outputs;
    }

    int64_t numBlocks = 1;
    int64_t rowsPerCore = 1;
    int64_t rowsPerTile = 1;
    int64_t inDepth = 2;
    std::tie(numBlocks, rowsPerCore, rowsPerTile, inDepth) = calc_dynamic_quant_tiling(M, N);

    auto x_ptr = (GM_ADDR)xc.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto s_ptr = (GM_ADDR)scale.data_ptr();

    auto acl_call = [=]() -> int {
        if (xc.scalar_type() == torch::kFloat16) {
            launch_dynamic_quant_half(x_ptr, y_ptr, s_ptr, M, N, numBlocks, rowsPerCore, rowsPerTile,
                                      inDepth, stream);
        } else {
            launch_dynamic_quant_bfloat16(x_ptr, y_ptr, s_ptr, M, N, numBlocks, rowsPerCore, rowsPerTile,
                                          inDepth, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("DynamicQuant", acl_call);
    return outputs;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("dynamic_quant", dynamic_quant_npu);
}

} // namespace cann_bench
