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
 * \file cummin_plugin.cpp
 * \brief Cummin API layer - torch bindings (compiled with g++).
 *
 * Everything happens on device: shape decomposition is metadata only, the host never reads tensor
 * values and never converts dtypes or layouts.
 */

#include <tuple>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/cummin_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("cummin(Tensor input, int dim) -> (Tensor, Tensor)");
}

std::tuple<torch::Tensor, torch::Tensor> cummin_meta(const torch::Tensor& input, int64_t dim)
{
    int64_t rank = input.dim();
    int64_t d = dim < 0 ? dim + rank : dim;
    TORCH_CHECK(d >= 0 && d < rank, "cummin: dim out of range");
    auto values = torch::empty_like(input);
    auto indices = torch::empty(input.sizes(), input.options().dtype(torch::kInt64));
    return std::make_tuple(values, indices);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("cummin", cummin_meta);
}

std::tuple<torch::Tensor, torch::Tensor> cummin_npu(const torch::Tensor& input, int64_t dim)
{
    const c10::OptionalDeviceGuard guard(input.device());
    auto x = input.contiguous();

    int64_t rank = x.dim();
    int64_t d = dim < 0 ? dim + rank : dim;
    TORCH_CHECK(d >= 0 && d < rank, "cummin: dim out of range");

    int64_t axis = x.size(d);
    int64_t outer = 1;
    for (int64_t i = 0; i < d; ++i) {
        outer *= x.size(i);
    }
    int64_t inner = 1;
    for (int64_t i = d + 1; i < rank; ++i) {
        inner *= x.size(i);
    }

    auto values = torch::empty_like(x);
    auto indices = torch::empty(x.sizes(), x.options().dtype(torch::kInt64));
    if (x.numel() == 0) {
        return std::make_tuple(values, indices);
    }

    CumminTiling t = calc_cummin_tiling(outer, axis, inner, (int64_t)x.element_size());

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = (GM_ADDR)x.data_ptr();
    auto vPtr = (GM_ADDR)values.data_ptr();
    auto iPtr = (GM_ADDR)indices.data_ptr();
    auto dtype = x.scalar_type();

    const int64_t nb = t.numBlocks;
    const int64_t rpb = t.rowsPerBlock;
    const int64_t chunk = t.chunk;
    const int64_t nGroups = t.nGroups;
    const int64_t Tg = t.Tg;
    const int64_t mode = t.mode;

    auto aclCall = [=]() -> int {
        if (mode == 0) {
            if (dtype == torch::kFloat32) {
                launch_cummin_row_float(xPtr, vPtr, iPtr, outer, axis, rpb, nb, chunk, stream);
            } else if (dtype == torch::kFloat16) {
                launch_cummin_row_half(xPtr, vPtr, iPtr, outer, axis, rpb, nb, chunk, stream);
            } else if (dtype == torch::kBFloat16) {
                launch_cummin_row_bfloat16(xPtr, vPtr, iPtr, outer, axis, rpb, nb, chunk, stream);
            } else if (dtype == torch::kInt32) {
                launch_cummin_row_int32(xPtr, vPtr, iPtr, outer, axis, rpb, nb, chunk, stream);
            } else {
                TORCH_CHECK(false, "cummin: unsupported dtype ", dtype);
            }
        } else {
            if (dtype == torch::kFloat32) {
                launch_cummin_col_float(xPtr, vPtr, iPtr, outer, axis, inner, nGroups, Tg, nb, stream);
            } else if (dtype == torch::kFloat16) {
                launch_cummin_col_half(xPtr, vPtr, iPtr, outer, axis, inner, nGroups, Tg, nb, stream);
            } else if (dtype == torch::kBFloat16) {
                launch_cummin_col_bfloat16(xPtr, vPtr, iPtr, outer, axis, inner, nGroups, Tg, nb, stream);
            } else if (dtype == torch::kInt32) {
                launch_cummin_col_int32(xPtr, vPtr, iPtr, outer, axis, inner, nGroups, Tg, nb, stream);
            } else {
                TORCH_CHECK(false, "cummin: unsupported dtype ", dtype);
            }
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("Cummin", aclCall);
    return std::make_tuple(values, indices);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("cummin", cummin_npu);
}

} // namespace cann_bench
