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
 * \file gelu_plugin.cpp
 * \brief Gelu API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/gelu_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("gelu(Tensor x, str approximate='none') -> Tensor");
}

static int32_t parse_gelu_mode(const std::string &approximate)
{
    TORCH_CHECK(approximate == "none" || approximate == "tanh",
        "approximate must be 'none' or 'tanh'.");
    return (approximate == "tanh") ? 1 : 0;
}

torch::Tensor gelu_meta(const torch::Tensor &x, const std::string &approximate)
{
    parse_gelu_mode(approximate);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16,
        "gelu only supports float32/float16/bfloat16.");
    TORCH_CHECK(x.is_contiguous(), "gelu requires a contiguous input tensor.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("gelu", gelu_meta);
}

torch::Tensor gelu_npu(const torch::Tensor &x, const std::string &approximate)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto y = gelu_meta(x, approximate);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    int32_t mode = parse_gelu_mode(approximate);
    int64_t totalLength = x.numel();
    int64_t typeSize = x.element_size();
    // Storage-dtype code supplied to the tiling function: 0 = float32,
    // 1 = float16, 2 = bfloat16. The dtype set was validated in gelu_meta.
    int32_t storageDtype;
    if (x.scalar_type() == torch::kFloat32) {
        storageDtype = 0;
    } else if (x.scalar_type() == torch::kFloat16) {
        storageDtype = 1;
    } else {
        storageDtype = 2;
    }
    int64_t numBlocks, blockLength, tileElementCount;
    std::tie(numBlocks, blockLength, tileElementCount) =
        calc_gelu_tiling_params(totalLength, mode, typeSize, storageDtype);

    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        // Select the compile-time (storage dtype, mode) specialization. The mode
        // is resolved here on the host and is not passed into the kernel. The
        // [S10] single-loop coverage needs no host-computed bulk/tail split.
        if (x.scalar_type() == torch::kFloat32) {
            if (mode == 0) {
                GELU_KERNEL_LAUNCH_FLOAT_ERF(x_ptr, y_ptr, totalLength, numBlocks, blockLength,
                    (uint32_t)tileElementCount, stream);
            } else {
                GELU_KERNEL_LAUNCH_FLOAT_TANH(x_ptr, y_ptr, totalLength, numBlocks, blockLength,
                    (uint32_t)tileElementCount, stream);
            }
        } else if (x.scalar_type() == torch::kFloat16) {
            if (mode == 0) {
                GELU_KERNEL_LAUNCH_HALF_ERF(x_ptr, y_ptr, totalLength, numBlocks, blockLength,
                    (uint32_t)tileElementCount, stream);
            } else {
                GELU_KERNEL_LAUNCH_HALF_TANH(x_ptr, y_ptr, totalLength, numBlocks, blockLength,
                    (uint32_t)tileElementCount, stream);
            }
        } else {
            if (mode == 0) {
                GELU_KERNEL_LAUNCH_BFLOAT16_ERF(x_ptr, y_ptr, totalLength, numBlocks, blockLength,
                    (uint32_t)tileElementCount, stream);
            } else {
                GELU_KERNEL_LAUNCH_BFLOAT16_TANH(x_ptr, y_ptr, totalLength, numBlocks, blockLength,
                    (uint32_t)tileElementCount, stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Gelu", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("gelu", gelu_npu);
}

} // namespace cann_bench
