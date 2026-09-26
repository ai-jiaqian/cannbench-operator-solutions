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
 * \file softmax_plugin.cpp
 * \brief Softmax API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/softmax_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("softmax(Tensor x, int dim=-1) -> Tensor");
}

static bool softmax_dtype_ok(torch::ScalarType st)
{
    return st == torch::kFloat32 || st == torch::kFloat16 || st == torch::kBFloat16;
}

static torch::Tensor softmax_meta(const torch::Tensor &x, int64_t dim)
{
    TORCH_CHECK(softmax_dtype_ok(x.scalar_type()),
                "softmax: unsupported dtype, expected float32/float16/bfloat16.");
    const int64_t rank = x.dim();
    TORCH_CHECK(rank >= 1 && rank <= 8, "softmax: input rank must be in [1, 8].");
    int64_t d = dim < 0 ? dim + rank : dim;
    TORCH_CHECK(d >= 0 && d < rank, "softmax: dim out of range.");
    TORCH_CHECK(x.is_contiguous(), "softmax: input must be contiguous.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("softmax", softmax_meta);
}

static torch::Tensor softmax_npu(const torch::Tensor &x, int64_t dim)
{
    const c10::OptionalDeviceGuard guard(x.device());
    const int64_t rank = x.dim();
    const int64_t d = dim < 0 ? dim + rank : dim;
    auto y = softmax_meta(x, d);
    if (x.numel() == 0 || y.numel() == 0) {
        return y;  // empty guard: nothing to reduce
    }

    int64_t shape[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int64_t i = 0; i < rank; ++i) {
        shape[i] = x.size(i);
    }
    const int64_t esize = x.element_size();

    SoftmaxTilingOut t;
    calc_softmax_tiling(rank, d, shape, esize, &t);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    const auto st = x.scalar_type();

    // split-R GM workspace (k2_split_workspace): allocated only when the host selected
    // the split-R realization (wsBytes > 0); the uint8 tensor is captured by the launch
    // lambda so it stays alive until the launch completes.
    torch::Tensor ws;
    GM_ADDR ws_ptr = nullptr;
    if (t.wsBytes > 0) {
        ws = torch::empty({t.wsBytes}, x.options().dtype(torch::kUInt8));
        ws_ptr = (GM_ADDR)ws.data_ptr();
    }

    auto acl_call = [=]() -> int {
        if (t.mode == 0) {
            if (st == torch::kFloat32) {
                launch_softmax_k1_std_float(x_ptr, y_ptr, t.numBlocks, t.outer, t.R,
                    t.G, t.Kpad, t.rowsPerCore, t.tmpSize, stream);
            } else if (st == torch::kFloat16) {
                launch_softmax_k1_std_half(x_ptr, y_ptr, t.numBlocks, t.outer, t.R,
                    t.G, t.Kpad, t.rowsPerCore, t.tmpSize, stream);
            } else {
                launch_softmax_k1_std_bf16(x_ptr, y_ptr, t.numBlocks, t.outer, t.R,
                    t.G, t.Kpad, t.rowsPerCore, t.tmpSize, stream);
            }
        } else if (t.mode == 1) {
            if (st == torch::kFloat32) {
                launch_softmax_k1_packed_float(x_ptr, y_ptr, t.numBlocks, t.outer, t.R,
                    t.G, t.Kpad, t.rowsPerCore, t.tmpSize, stream);
            } else if (st == torch::kFloat16) {
                launch_softmax_k1_packed_half(x_ptr, y_ptr, t.numBlocks, t.outer, t.R,
                    t.G, t.Kpad, t.rowsPerCore, t.tmpSize, stream);
            } else {
                launch_softmax_k1_packed_bf16(x_ptr, y_ptr, t.numBlocks, t.outer, t.R,
                    t.G, t.Kpad, t.rowsPerCore, t.tmpSize, stream);
            }
        } else {
            if (st == torch::kFloat32) {
                launch_softmax_k2_float(x_ptr, y_ptr, ws_ptr, t.numBlocks, t.outer, t.R, t.inner,
                    t.G, t.W, t.lastW, t.nB, t.rowsPerCore, t.tp0, t.splitG, t.splitGroups, stream);
            } else if (st == torch::kFloat16) {
                launch_softmax_k2_half(x_ptr, y_ptr, ws_ptr, t.numBlocks, t.outer, t.R, t.inner,
                    t.G, t.W, t.lastW, t.nB, t.rowsPerCore, t.tp0, t.splitG, t.splitGroups, stream);
            } else {
                launch_softmax_k2_bf16(x_ptr, y_ptr, ws_ptr, t.numBlocks, t.outer, t.R, t.inner,
                    t.G, t.W, t.lastW, t.nB, t.rowsPerCore, t.tp0, t.splitG, t.splitGroups, stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Softmax", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("softmax", softmax_npu);
}

} // namespace cann_bench
