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
 * \file moe_gating_plugin.cpp
 * \brief MoeGatingTopKSoftmax API layer - torch bindings (compiled with g++)
 */

#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/moe_gating_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("moe_gating_top_k_softmax(Tensor x, Tensor? finished=None, int k=1) -> (Tensor y, Tensor expert_idx, Tensor row_idx)");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_gating_top_k_softmax_meta(
    const torch::Tensor &x, const c10::optional<torch::Tensor> &finished, int64_t k)
{
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                    x.scalar_type() == torch::kBFloat16,
                "cann_bench.moe_gating_top_k_softmax only supports float32, float16 and bfloat16 inputs.");
    TORCH_CHECK(x.dim() >= 2, "cann_bench.moe_gating_top_k_softmax requires a tensor with rank >= 2.");
    const int64_t inner = x.size(-1);
    TORCH_CHECK(inner > 0, "cann_bench.moe_gating_top_k_softmax requires a non-empty last dimension.");
    TORCH_CHECK(k > 0 && k <= inner, "cann_bench.moe_gating_top_k_softmax requires 0 < k <= x.size(-1).");
    if (finished.has_value() && finished->defined()) {
        TORCH_CHECK(finished->scalar_type() == torch::kBool,
                    "cann_bench.moe_gating_top_k_softmax requires a bool 'finished' tensor.");
    }
    std::vector<int64_t> outShape(x.sizes().begin(), x.sizes().end() - 1);
    outShape.push_back(k);
    auto y = torch::empty(outShape, x.options());
    auto expertIdx = torch::empty(outShape, x.options().dtype(torch::kInt32));
    auto rowIdx = torch::empty(outShape, x.options().dtype(torch::kInt32));
    return std::make_tuple(y, expertIdx, rowIdx);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("moe_gating_top_k_softmax", moe_gating_top_k_softmax_meta);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_gating_top_k_softmax_npu(
    const torch::Tensor &x, const c10::optional<torch::Tensor> &finished, int64_t k)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto outputs = moe_gating_top_k_softmax_meta(xc, finished, k);
    auto y = std::get<0>(outputs);
    auto expertIdx = std::get<1>(outputs);
    auto rowIdx = std::get<2>(outputs);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const int64_t inner = xc.size(-1);
    const int64_t totalRows = xc.numel() / inner;
    TORCH_CHECK(totalRows > 0, "cann_bench.moe_gating_top_k_softmax requires at least one row.");
    const int64_t typeSize = (xc.scalar_type() == torch::kFloat32) ? 4 : 2;

    int64_t numBlocks = 1;
    int64_t rowsPerCore = 1;
    int64_t rowsPerTile = 1;
    std::tie(numBlocks, rowsPerCore, rowsPerTile) = calc_moe_gating_tiling_params(totalRows, inner, k, typeSize);

    torch::Tensor finC;
    void *finPtr = nullptr;
    int64_t hasFin = 0;
    if (finished.has_value() && finished->defined()) {
        finC = finished->contiguous();
        TORCH_CHECK(finC.scalar_type() == torch::kBool,
                    "cann_bench.moe_gating_top_k_softmax requires a bool 'finished' tensor.");
        TORCH_CHECK(finC.numel() == totalRows, "cann_bench.moe_gating_top_k_softmax: finished must have numel == rows.");
        finPtr = (void *)finC.data_ptr();
        hasFin = 1;
    }

    auto x_ptr = (void *)xc.data_ptr();
    auto y_ptr = (void *)y.data_ptr();
    auto ei_ptr = (void *)expertIdx.data_ptr();
    auto ri_ptr = (void *)rowIdx.data_ptr();

    auto acl_call = [=]() -> int {
        if (xc.scalar_type() == torch::kFloat32) {
            launch_moe_gating_kernel_float(x_ptr, finPtr, y_ptr, ei_ptr, ri_ptr, totalRows, inner, k, numBlocks,
                                           rowsPerCore, rowsPerTile, hasFin, stream);
        } else if (xc.scalar_type() == torch::kFloat16) {
            launch_moe_gating_kernel_half(x_ptr, finPtr, y_ptr, ei_ptr, ri_ptr, totalRows, inner, k, numBlocks,
                                          rowsPerCore, rowsPerTile, hasFin, stream);
        } else {
            launch_moe_gating_kernel_bfloat16(x_ptr, finPtr, y_ptr, ei_ptr, ri_ptr, totalRows, inner, k, numBlocks,
                                              rowsPerCore, rowsPerTile, hasFin, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MoeGatingTopKSoftmax", acl_call);
    return outputs;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("moe_gating_top_k_softmax", moe_gating_top_k_softmax_npu);
}

} // namespace cann_bench
