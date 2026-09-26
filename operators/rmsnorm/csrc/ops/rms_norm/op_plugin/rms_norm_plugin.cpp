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
 * \file rms_norm_plugin.cpp
 * \brief RmsNorm API layer - torch bindings (compiled with g++)
 *
 * Frozen z6 structure: the host only inspects tensor metadata (shape / dtype /
 * contiguity), allocates the output, computes the single launch configuration and
 * issues exactly ONE launch on the current NPU stream. No statistic workspace is
 * allocated, no device data is read back to the host, no dtype or layout conversion
 * is routed through host memory, and nothing is cached across calls.
 */

#include <tuple>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/rms_norm_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("rms_norm(Tensor x, Tensor gamma, float epsilon=1e-6) -> Tensor");
}

torch::Tensor rms_norm_meta(const torch::Tensor &x, const torch::Tensor &gamma, double epsilon)
{
    (void)gamma;
    (void)epsilon;
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("rms_norm", rms_norm_meta);
}

torch::Tensor rms_norm_npu(const torch::Tensor &x, const torch::Tensor &gamma, double epsilon)
{
    TORCH_CHECK(x.dim() >= 1, "rms_norm: x must have at least one dimension");
    TORCH_CHECK(gamma.dim() == 1, "rms_norm: gamma must be 1-D");
    TORCH_CHECK(x.scalar_type() == gamma.scalar_type(), "rms_norm: gamma dtype must match x");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                    x.scalar_type() == torch::kBFloat16,
                "rms_norm: unsupported dtype (float32/float16/bfloat16 only)");
    const int64_t numCol = gamma.numel();
    TORCH_CHECK(numCol > 0, "rms_norm: gamma must be non-empty");
    TORCH_CHECK(x.numel() % numCol == 0, "rms_norm: x.numel() must be a multiple of gamma size");

    const c10::OptionalDeviceGuard guard(x.device());

    // The submitted Kernel consumes the input layout directly; no host-side tensor
    // transform participates in the result (SUB-BEH-002).
    const torch::Tensor &in = x;
    const torch::Tensor &g = gamma;
    auto y = torch::empty_like(in);
    const int64_t numRow = in.numel() / numCol;
    if (numRow <= 0) {
        return y;
    }

    int64_t numBlocks = 1;
    int64_t blockRows = 1;
    int64_t tileRows = 1;
    std::tie(numBlocks, blockRows, tileRows) =
        calc_rms_norm_tiling_params(numRow, numCol, in.element_size());

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = (GM_ADDR)in.data_ptr();
    auto gPtr = (GM_ADDR)g.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();
    const float epsF = static_cast<float>(epsilon);
    const float invD = 1.0f / static_cast<float>(numCol);
    const auto dtype = in.scalar_type();

    auto acl_call = [=]() -> int {
        if (dtype == torch::kFloat32) {
            RMS_NORM_LAUNCH_FLOAT(xPtr, gPtr, yPtr, numRow, numCol, epsF, invD,
                                  numBlocks, blockRows, tileRows, stream);
        } else if (dtype == torch::kFloat16) {
            RMS_NORM_LAUNCH_HALF(xPtr, gPtr, yPtr, numRow, numCol, epsF, invD,
                                 numBlocks, blockRows, tileRows, stream);
        } else if (dtype == torch::kBFloat16) {
            RMS_NORM_LAUNCH_BF16(xPtr, gPtr, yPtr, numRow, numCol, epsF, invD,
                                 numBlocks, blockRows, tileRows, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("RmsNorm", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("rms_norm", rms_norm_npu);
}

} // namespace cann_bench
