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
 * \file apply_adam_w_plugin.cpp
 * \brief ApplyAdamW API layer - torch bindings (compiled with g++).
 *
 * apply_adam_w(Tensor var, Tensor grad, Tensor m, Tensor v, float lr, float beta1, float beta2,
 *              float weight_decay, float epsilon=1e-8, int step=1, bool maximize=False) -> Tensor
 *
 * Only metadata (shapes / dtypes / raw device pointers) is inspected on the host; every element of
 * the result is produced by the custom device kernel. The step-dependent normalisation and the sign
 * of lr are folded into scalar coefficients that are handed to the kernel.
 */

#include <cmath>
#include <tuple>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/apply_adam_w_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("apply_adam_w(Tensor var, Tensor grad, Tensor m, Tensor v, float lr, float beta1, float beta2, "
          "float weight_decay, float epsilon=1e-8, int step=1, bool maximize=False) -> Tensor");
}

torch::Tensor apply_adam_w_meta(
    const torch::Tensor &var, const torch::Tensor &grad, const torch::Tensor &m, const torch::Tensor &v,
    double lr, double beta1, double beta2, double weight_decay, double epsilon, int64_t step, bool maximize)
{
    (void)lr; (void)beta1; (void)beta2; (void)weight_decay; (void)epsilon; (void)step; (void)maximize;
    TORCH_CHECK(var.sizes() == grad.sizes(), "apply_adam_w: var and grad must have the same shape.");
    TORCH_CHECK(var.sizes() == m.sizes(), "apply_adam_w: var and m must have the same shape.");
    TORCH_CHECK(var.sizes() == v.sizes(), "apply_adam_w: var and v must have the same shape.");
    return torch::empty_like(var);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("apply_adam_w", apply_adam_w_meta);
}

torch::Tensor apply_adam_w_npu(
    const torch::Tensor &var, const torch::Tensor &grad, const torch::Tensor &m, const torch::Tensor &v,
    double lr, double beta1, double beta2, double weight_decay, double epsilon, int64_t step, bool maximize)
{
    const c10::OptionalDeviceGuard guard(var.device());
    auto y = apply_adam_w_meta(var, grad, m, v, lr, beta1, beta2, weight_decay, epsilon, step, maximize);

    const auto dtype = var.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "apply_adam_w: only float32, float16 and bfloat16 are supported.");
    TORCH_CHECK(grad.scalar_type() == dtype && m.scalar_type() == dtype && v.scalar_type() == dtype,
                "apply_adam_w: all inputs must share one dtype.");

    const int64_t totalLength = var.numel();
    if (totalLength == 0) {
        return y;
    }

    const int64_t elemBytes = (dtype == torch::kFloat32) ? 4 : 2;
    int64_t numBlocks = 1;
    int64_t blockLength = 1;
    int64_t tileElems = 32;
    std::tie(numBlocks, blockLength, tileElems) = calc_apply_adam_w_tiling_params(totalLength, elemBytes);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    GM_ADDR varPtr = reinterpret_cast<GM_ADDR>(var.data_ptr());
    GM_ADDR gradPtr = reinterpret_cast<GM_ADDR>(grad.data_ptr());
    GM_ADDR mPtr = reinterpret_cast<GM_ADDR>(m.data_ptr());
    GM_ADDR vPtr = reinterpret_cast<GM_ADDR>(v.data_ptr());
    GM_ADDR yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());

    // Scalar attribute derivation (host-side scalar work only; no tensor data is touched).
    const double denomA = 1.0 - std::pow(beta1, static_cast<double>(step));
    const double denomB = 1.0 - std::pow(beta2, static_cast<double>(step));
    const double invA = (denomA != 0.0) ? (1.0 / denomA) : 0.0;
    const double invB = (denomB != 0.0) ? (1.0 / denomB) : 0.0;

    const float lrSigned = static_cast<float>(maximize ? lr : -lr);

    // lrSigned is folded into a1/c1 so the kernel's update term already carries the sign/scale of
    // lr; scaleX folds the decoupled weight decay into the final affine.
    const float a1 = static_cast<float>(beta1 * invA * lrSigned);
    const float c1 = static_cast<float>((1.0 - beta1) * invA * lrSigned);
    const float a2 = static_cast<float>(beta2 * invB);
    const float c2 = static_cast<float>((1.0 - beta2) * invB);
    const float epsF = static_cast<float>(epsilon);
    const float scaleX = 1.0f + lrSigned * static_cast<float>(weight_decay);
    const int64_t hasWd = (weight_decay != 0.0) ? 1 : 0;
    const uint32_t tile = static_cast<uint32_t>(tileElems);

    auto acl_call = [=]() -> int {
        if (dtype == torch::kFloat32) {
            APPLY_ADAM_W_LAUNCH_FLOAT(varPtr, gradPtr, mPtr, vPtr, yPtr, totalLength,
                                      numBlocks, blockLength, tile,
                                      a1, c1, a2, c2, epsF, scaleX, hasWd, stream);
        } else if (dtype == torch::kFloat16) {
            APPLY_ADAM_W_LAUNCH_HALF(varPtr, gradPtr, mPtr, vPtr, yPtr, totalLength,
                                     numBlocks, blockLength, tile,
                                     a1, c1, a2, c2, epsF, scaleX, hasWd, stream);
        } else {
            APPLY_ADAM_W_LAUNCH_BF16(varPtr, gradPtr, mPtr, vPtr, yPtr, totalLength,
                                     numBlocks, blockLength, tile,
                                     a1, c1, a2, c2, epsF, scaleX, hasWd, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ApplyAdamW", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("apply_adam_w", apply_adam_w_npu);
}

} // namespace cann_bench
