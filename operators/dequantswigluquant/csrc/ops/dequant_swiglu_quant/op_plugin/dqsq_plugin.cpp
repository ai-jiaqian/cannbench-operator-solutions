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
 * \file dqsq_plugin.cpp
 * \brief DequantSwigluQuant API layer - torch bindings (compiled with g++)
 *
 *   dequant_swiglu_quant(x, weight_scale, activation_scale, quant_scale, activate_left) -> (y, scale)
 *     y     : [TokensNum, H] int8
 *     scale : [TokensNum] float32
 */

#include <tuple>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/dqsq_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("dequant_swiglu_quant(Tensor x, Tensor? weight_scale=None, Tensor? activation_scale=None, "
          "Tensor? quant_scale=None, bool activate_left=False) -> (Tensor y, Tensor scale)");
}

static std::tuple<torch::Tensor, torch::Tensor> DqsqMeta(
    const torch::Tensor &x, const c10::optional<torch::Tensor> &weight_scale,
    const c10::optional<torch::Tensor> &activation_scale,
    const c10::optional<torch::Tensor> &quant_scale, bool activate_left)
{
    (void)weight_scale;
    (void)activation_scale;
    (void)quant_scale;
    (void)activate_left;
    TORCH_CHECK(x.dim() == 2, "dequant_swiglu_quant requires a 2D input [TokensNum, 2H].");
    const int64_t M = x.size(0);
    const int64_t N2 = x.size(1);
    TORCH_CHECK(N2 % 2 == 0, "dequant_swiglu_quant requires an even last dimension.");
    const int64_t H = N2 / 2;
    auto y = torch::empty({M, H}, x.options().dtype(torch::kChar));
    auto scale = torch::empty({M}, x.options().dtype(torch::kFloat));
    return std::make_tuple(y, scale);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("dequant_swiglu_quant", DqsqMeta);
}

static std::tuple<torch::Tensor, torch::Tensor> DqsqNpu(
    const torch::Tensor &x, const c10::optional<torch::Tensor> &weight_scale,
    const c10::optional<torch::Tensor> &activation_scale,
    const c10::optional<torch::Tensor> &quant_scale, bool activate_left)
{
    const c10::OptionalDeviceGuard guard(x.device());
    TORCH_CHECK(x.dim() == 2, "dequant_swiglu_quant requires a 2D input [TokensNum, 2H].");
    TORCH_CHECK(x.is_contiguous(), "dequant_swiglu_quant requires a contiguous input.");

    auto outputs = DqsqMeta(x, weight_scale, activation_scale, quant_scale, activate_left);
    auto y = std::get<0>(outputs);
    auto scale = std::get<1>(outputs);

    const int64_t M = x.size(0);
    const int64_t N2 = x.size(1);
    if (M <= 0 || N2 <= 0) {
        return outputs;
    }
    const int64_t H = N2 / 2;

    const auto xt = x.scalar_type();
    const bool isInt32 = (xt == torch::kInt32);
    if (isInt32) {
        TORCH_CHECK(weight_scale.has_value() && weight_scale->defined(),
                    "weight_scale is required when x is int32.");
        TORCH_CHECK(activation_scale.has_value() && activation_scale->defined(),
                    "activation_scale is required when x is int32.");
        TORCH_CHECK(weight_scale->scalar_type() == torch::kFloat, "weight_scale must be float32.");
        TORCH_CHECK(activation_scale->scalar_type() == torch::kFloat,
                    "activation_scale must be float32.");
        TORCH_CHECK(weight_scale->is_contiguous() && activation_scale->is_contiguous(),
                    "weight_scale / activation_scale must be contiguous.");
        TORCH_CHECK(weight_scale->numel() == N2, "weight_scale must have 2H elements.");
        TORCH_CHECK(activation_scale->numel() == M,
                    "activation_scale must have TokensNum elements.");
    } else {
        TORCH_CHECK(xt == torch::kFloat16 || xt == torch::kBFloat16,
                    "dequant_swiglu_quant supports int32 / float16 / bfloat16 inputs.");
        TORCH_CHECK(!(weight_scale.has_value() && weight_scale->defined()),
                    "weight_scale must be None for float16 / bfloat16 inputs.");
        TORCH_CHECK(!(activation_scale.has_value() && activation_scale->defined()),
                    "activation_scale must be None for float16 / bfloat16 inputs.");
    }

    const bool hasQ = quant_scale.has_value() && quant_scale->defined();
    if (hasQ) {
        TORCH_CHECK(quant_scale->scalar_type() == torch::kFloat, "quant_scale must be float32.");
        TORCH_CHECK(quant_scale->is_contiguous(), "quant_scale must be contiguous.");
        TORCH_CHECK(quant_scale->numel() == H, "quant_scale must have H elements.");
    }

    int64_t numBlocks = 1;
    int64_t rowsPerCore = 1;
    int64_t tr = 1;
    std::tie(numBlocks, rowsPerCore, tr) = calc_dqsq_tiling(
        M, N2, static_cast<int64_t>(x.element_size()), isInt32 ? 1 : 0, hasQ ? 1 : 0);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    GM_ADDR xPtr = reinterpret_cast<GM_ADDR>(x.data_ptr());
    GM_ADDR yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());
    GM_ADDR sPtr = reinterpret_cast<GM_ADDR>(scale.data_ptr());
    GM_ADDR wsPtr = nullptr;
    GM_ADDR asPtr = nullptr;
    GM_ADDR qsPtr = nullptr;
    if (isInt32) {
        wsPtr = reinterpret_cast<GM_ADDR>(weight_scale->data_ptr());
        asPtr = reinterpret_cast<GM_ADDR>(activation_scale->data_ptr());
    }
    if (hasQ) {
        qsPtr = reinterpret_cast<GM_ADDR>(quant_scale->data_ptr());
    }
    const int64_t activateLeft = activate_left ? 1 : 0;
    const int64_t hasQScale = hasQ ? 1 : 0;

    auto acl_call = [=]() -> int {
        if (xt == torch::kFloat16) {
            launch_dqsq_half(xPtr, yPtr, sPtr, wsPtr, asPtr, qsPtr, M, N2, numBlocks, rowsPerCore, tr,
                             activateLeft, hasQScale, stream);
        } else if (xt == torch::kBFloat16) {
            launch_dqsq_bfloat16(xPtr, yPtr, sPtr, wsPtr, asPtr, qsPtr, M, N2, numBlocks, rowsPerCore,
                                 tr, activateLeft, hasQScale, stream);
        } else {
            launch_dqsq_int32(xPtr, yPtr, sPtr, wsPtr, asPtr, qsPtr, M, N2, numBlocks, rowsPerCore, tr,
                              activateLeft, hasQScale, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("DequantSwigluQuant", acl_call);
    return outputs;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("dequant_swiglu_quant", DqsqNpu);
}

}  // namespace cann_bench
