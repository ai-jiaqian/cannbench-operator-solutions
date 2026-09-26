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
 * \file exp_plugin.cpp
 * \brief Exp API layer - torch bindings (compiled with g++)
 */

#include <cmath>
#include <tuple>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/exp_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("exp(Tensor x, float base=-1.0, float scale=1.0, float shift=0.0) -> Tensor");
}

torch::Tensor exp_meta(const torch::Tensor &x, double base, double scale, double shift)
{
    (void)base;
    (void)scale;
    (void)shift;
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("exp", exp_meta);
}

torch::Tensor exp_npu(const torch::Tensor &x, double base, double scale, double shift)
{
    const c10::OptionalDeviceGuard guard(x.device());
    TORCH_CHECK(x.is_contiguous(), "cann_bench.exp: input must be contiguous.");
    auto y = exp_meta(x, base, scale, shift);

    const int64_t totalLength = x.numel();
    int64_t elemSize = 4;
    if (x.scalar_type() == torch::kFloat32) {
        elemSize = 4;
    } else if (x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kBFloat16) {
        elemSize = 2;
    } else {
        TORCH_CHECK(false, "cann_bench.exp: unsupported dtype, expect float16/float32/bfloat16.");
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t numBlocks = 1;
    int64_t tilesPerCore = 1;
    uint32_t tileElems = 256;
    std::tie(numBlocks, tilesPerCore, tileElems) = calc_exp_tiling_params(totalLength, elemSize);

    // base <= 0 means the natural base e; base > 0 uses the supplied base.
    // Folding ln(base) into the affine constants is numerically equivalent to
    // the reference ordering ((scale*x + shift) * ln(base)) and is exact for
    // base == 1, where ln(1) == 0 collapses the result to exp(0) == 1.
    const double lnb = (base > 0.0) ? std::log(base) : 1.0;
    const float coefA = static_cast<float>(scale * lnb);
    const float coefB = static_cast<float>(shift * lnb);

    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        if (x.scalar_type() == torch::kFloat32) {
            launch_exp_kernel_float(x_ptr, y_ptr, totalLength, numBlocks, tilesPerCore, tileElems, coefA, coefB,
                                    stream);
        } else if (x.scalar_type() == torch::kFloat16) {
            launch_exp_kernel_half(x_ptr, y_ptr, totalLength, numBlocks, tilesPerCore, tileElems, coefA, coefB,
                                   stream);
        } else {
            launch_exp_kernel_bfloat16(x_ptr, y_ptr, totalLength, numBlocks, tilesPerCore, tileElems, coefA, coefB,
                                       stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Exp", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("exp", exp_npu);
}

} // namespace cann_bench
