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
 * \file add_rms_norm_dynamic_quant_plugin.cpp
 * \brief AddRmsNormDynamicQuant API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/add_rms_norm_dynamic_quant_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("add_rms_norm_dynamic_quant(Tensor x1, Tensor x2, Tensor gamma, float epsilon) -> (Tensor y, Tensor xOut, Tensor scaleOut)");
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> add_rms_norm_dynamic_quant_meta(
    const torch::Tensor &x1, const torch::Tensor &x2, const torch::Tensor &gamma, double epsilon)
{
    (void)epsilon;
    TORCH_CHECK(x1.dim() >= 1, "add_rms_norm_dynamic_quant: x1 must have at least one dimension.");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "add_rms_norm_dynamic_quant: x1 and x2 must share a shape.");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == x1.size(-1),
                "add_rms_norm_dynamic_quant: gamma must be 1-D with size == x1.size(-1).");
    TORCH_CHECK(gamma.scalar_type() == x1.scalar_type(),
                "add_rms_norm_dynamic_quant: gamma dtype must match x1 dtype.");

    auto y = torch::empty_like(x1, x1.options().dtype(torch::kChar));
    auto xOut = torch::empty_like(x1);
    std::vector<int64_t> scaleShape(x1.sizes().begin(), x1.sizes().end() - 1);
    auto scaleOut = torch::empty(scaleShape, x1.options().dtype(torch::kFloat));
    return std::make_tuple(y, xOut, scaleOut);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("add_rms_norm_dynamic_quant", add_rms_norm_dynamic_quant_meta);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> add_rms_norm_dynamic_quant_npu(
    const torch::Tensor &x1, const torch::Tensor &x2, const torch::Tensor &gamma, double epsilon)
{
    const c10::OptionalDeviceGuard guard(x1.device());
    auto x1c = x1.is_contiguous() ? x1 : x1.contiguous();
    auto x2c = x2.is_contiguous() ? x2 : x2.contiguous();
    auto gc = gamma.is_contiguous() ? gamma : gamma.contiguous();

    auto outs = add_rms_norm_dynamic_quant_meta(x1c, x2c, gc, epsilon);
    auto y = std::get<0>(outs);
    auto xOut = std::get<1>(outs);
    auto scaleOut = std::get<2>(outs);

    const int64_t rowLen = x1c.size(-1);
    const int64_t total = x1c.numel();
    if (total == 0 || rowLen <= 0) {
        return outs;
    }
    const int64_t numRows = total / rowLen;

    const auto dtype = x1c.scalar_type();
    const bool isBf16 = (dtype == torch::kBFloat16);
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "add_rms_norm_dynamic_quant: only float16/bfloat16 inputs are supported.");
    TORCH_CHECK(rowLen <= 16384,
                "add_rms_norm_dynamic_quant: row length must not exceed 16384.");

    int64_t numBlocks = 1;
    int64_t rowsPerCore = numRows;
    int64_t tileElems = rowLen;
    int64_t rowBlock = 0;
    std::tie(numBlocks, rowsPerCore, tileElems, rowBlock) = calc_ardq_tiling(numRows, rowLen);
    const uint32_t tileU = static_cast<uint32_t>(tileElems);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto x1Ptr = (GM_ADDR)x1c.data_ptr();
    auto x2Ptr = (GM_ADDR)x2c.data_ptr();
    auto gPtr = (GM_ADDR)gc.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();
    auto xoPtr = (GM_ADDR)xOut.data_ptr();
    auto sPtr = (GM_ADDR)scaleOut.data_ptr();
    const float epsF = static_cast<float>(epsilon);

    auto acl_call = [=]() -> int {
        if (isBf16) {
            launch_ardq_bf16(x1Ptr, x2Ptr, gPtr, yPtr, xoPtr, sPtr, numRows, rowLen, numBlocks,
                             rowsPerCore, tileU, rowBlock, epsF, stream);
        } else {
            launch_ardq_half(x1Ptr, x2Ptr, gPtr, yPtr, xoPtr, sPtr, numRows, rowLen, numBlocks,
                             rowsPerCore, tileU, rowBlock, epsF, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("AddRmsNormDynamicQuant", acl_call);
    return outs;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("add_rms_norm_dynamic_quant", add_rms_norm_dynamic_quant_npu);
}

}  // namespace cann_bench
