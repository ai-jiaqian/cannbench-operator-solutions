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
 * \file mish_plugin.cpp
 * \brief Mish API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/mish_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("mish(Tensor x) -> Tensor");
}

torch::Tensor mish_meta(const torch::Tensor &x)
{
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                    x.scalar_type() == torch::kBFloat16,
                "cann_bench.mish only supports float32, float16 and bfloat16 inputs.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("mish", mish_meta);
}

torch::Tensor mish_npu(const torch::Tensor &x)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto y = mish_meta(xc);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t totalLength = xc.numel();

    bool isCast = (xc.scalar_type() != torch::kFloat32);
    int64_t numBlocks = 1;
    int64_t blockLength = 1;
    uint32_t tileElems = 64;
    std::tie(numBlocks, blockLength, tileElems) = calc_mish_tiling_params(totalLength, isCast);

    auto x_ptr = (GM_ADDR)xc.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        if (xc.scalar_type() == torch::kFloat32) {
            MISH_KERNEL_LAUNCH_FLOAT(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElems, stream);
        } else if (xc.scalar_type() == torch::kFloat16) {
            MISH_KERNEL_LAUNCH_HALF(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElems, stream);
        } else {
            MISH_KERNEL_LAUNCH_BFLOAT16(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElems, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Mish", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("mish", mish_npu);
}

} // namespace cann_bench
