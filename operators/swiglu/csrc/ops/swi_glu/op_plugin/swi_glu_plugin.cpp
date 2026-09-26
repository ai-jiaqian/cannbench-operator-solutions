/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file swi_glu_plugin.cpp
 * \brief SwiGlu API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/swi_glu_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("swi_glu(Tensor input, int dim=-1) -> Tensor");
}

static torch::Tensor swi_glu_meta(const torch::Tensor &x, int64_t dim)
{
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
                    x.scalar_type() == torch::kBFloat16,
        "swi_glu only supports float32, float16 and bfloat16, got: ", x.scalar_type());
    const int64_t rank = x.dim();
    TORCH_CHECK(rank >= 1, "swi_glu input must have at least 1 dimension.");
    const int64_t d = dim < 0 ? dim + rank : dim;
    TORCH_CHECK(d >= 0 && d < rank, "swi_glu dim out of range.");
    TORCH_CHECK(x.is_contiguous(), "swi_glu input must be contiguous.");
    TORCH_CHECK(x.size(d) % 2 == 0, "swi_glu requires size[dim] to be even, got ", x.size(d),
        " at dim ", d, ".");
    std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
    out_sizes[d] = out_sizes[d] / 2;
    return torch::empty(out_sizes, x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("swi_glu", swi_glu_meta);
}

static torch::Tensor swi_glu_npu(const torch::Tensor &x, int64_t dim)
{
    const c10::OptionalDeviceGuard guard(x.device());
    const int64_t rank = x.dim();
    const int64_t d = dim < 0 ? dim + rank : dim;
    auto y = swi_glu_meta(x, d);
    if (x.numel() == 0 || y.numel() == 0) {
        return y;
    }

    const int64_t D = x.size(d);
    const int64_t H = D / 2;
    int64_t A = 1;
    for (int64_t i = 0; i < d; ++i) {
        A *= x.size(i);
    }
    int64_t B = 1;
    for (int64_t i = d + 1; i < rank; ++i) {
        B *= x.size(i);
    }
    const int64_t L = H * B;
    const int64_t esize = x.element_size();

    int64_t numBlocks, workPerCore, tileSize;
    std::tie(numBlocks, workPerCore, tileSize) = calc_swi_glu_tiling(A, L, esize);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        if (x.scalar_type() == torch::kFloat32) {
            SWI_GLU_KERNEL_LAUNCH_FLOAT(x_ptr, y_ptr, A, L, numBlocks, workPerCore, tileSize, stream);
        } else if (x.scalar_type() == torch::kFloat16) {
            SWI_GLU_KERNEL_LAUNCH_HALF(x_ptr, y_ptr, A, L, numBlocks, workPerCore, tileSize, stream);
        } else if (x.scalar_type() == torch::kBFloat16) {
            SWI_GLU_KERNEL_LAUNCH_BF16(x_ptr, y_ptr, A, L, numBlocks, workPerCore, tileSize, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("SwiGlu", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("swi_glu", swi_glu_npu);
}

} // namespace cann_bench
