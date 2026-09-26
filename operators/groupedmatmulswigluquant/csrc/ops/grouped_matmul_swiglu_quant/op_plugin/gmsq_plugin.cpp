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
 * \file gmsq_plugin.cpp
 * \brief GroupedMatmulSwigluQuant API layer - torch bindings (compiled with g++).
 *
 * grouped_matmul_swiglu_quant(x, weight, weight_scale, x_scale, group_list) -> (Tensor y, Tensor y_scale)
 *
 *   y[rows_g]       = requant( SwiGLU( dequant( x[rows_g] @ weight[g] ) ) )
 *   y_scale[rows_g] = max_j |act| / 127
 *
 * group_list is the cumsum attribute (a python list of E token counts).  Only metadata is inspected on
 * the host; the arithmetic, including the workspace contents, lives in the two custom device kernels.
 */

#include <cstring>
#include <tuple>
#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/gmsq_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor weight_scale, Tensor x_scale, "
          "int[] group_list) -> (Tensor y, Tensor y_scale)");
}

static std::tuple<torch::Tensor, torch::Tensor> GmsqMeta(const torch::Tensor &x, const torch::Tensor &weight,
                                                         const torch::Tensor &weight_scale,
                                                         const torch::Tensor &x_scale,
                                                         c10::List<int64_t> group_list)
{
    (void)weight_scale;
    (void)x_scale;
    (void)group_list;
    TORCH_CHECK(x.dim() == 2, "grouped_matmul_swiglu_quant: x must be 2D [M, K]");
    TORCH_CHECK(weight.dim() == 3, "grouped_matmul_swiglu_quant: weight must be 3D [E, K, N]");
    const int64_t M = x.size(0);
    const int64_t N = weight.size(2);
    TORCH_CHECK(N % 2 == 0, "grouped_matmul_swiglu_quant: N must be even");
    auto y = torch::empty({M, N / 2}, x.options().dtype(torch::kChar));
    auto yScale = torch::empty({M}, x.options().dtype(torch::kFloat32));
    return std::make_tuple(y, yScale);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("grouped_matmul_swiglu_quant", GmsqMeta);
}

static std::tuple<torch::Tensor, torch::Tensor> GmsqNpu(const torch::Tensor &x, const torch::Tensor &weight,
                                                        const torch::Tensor &weight_scale,
                                                        const torch::Tensor &x_scale,
                                                        c10::List<int64_t> group_list)
{
    const c10::OptionalDeviceGuard guard(x.device());
    TORCH_CHECK(x.scalar_type() == torch::kChar, "grouped_matmul_swiglu_quant: x must be int8");
    TORCH_CHECK(weight.scalar_type() == torch::kChar, "grouped_matmul_swiglu_quant: weight must be int8");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat,
                "grouped_matmul_swiglu_quant: weight_scale must be float32");
    TORCH_CHECK(x_scale.scalar_type() == torch::kFloat,
                "grouped_matmul_swiglu_quant: x_scale must be float32");

    auto xc = x.contiguous();
    auto wc = weight.contiguous();
    auto wsc = weight_scale.contiguous();
    auto xsc = x_scale.contiguous();

    auto out = GmsqMeta(xc, wc, wsc, xsc, group_list);
    auto y = std::get<0>(out);
    auto yScale = std::get<1>(out);

    const int64_t M = xc.size(0);
    const int64_t K = xc.size(1);
    const int64_t E = wc.size(0);
    const int64_t N = wc.size(2);
    if (M == 0 || N == 0 || E == 0) {
        return out;
    }
    TORCH_CHECK(wc.size(1) == K, "grouped_matmul_swiglu_quant: K mismatch between x and weight");
    TORCH_CHECK(N % 2 == 0, "grouped_matmul_swiglu_quant: N must be even");
    TORCH_CHECK(xsc.numel() == M, "grouped_matmul_swiglu_quant: x_scale must have M elements");
    TORCH_CHECK(wsc.size(0) == E && wsc.size(1) == N,
                "grouped_matmul_swiglu_quant: weight_scale must be [E, N]");

    // group_list is an attribute: copy the cumsum boundaries into a by-value POD for the kernels.
    GmsqGroupArg ga;
    std::memset(&ga, 0, sizeof(ga));
    const int64_t nGroups = static_cast<int64_t>(group_list.size());
    TORCH_CHECK(nGroups == E, "grouped_matmul_swiglu_quant: group_list length must equal the expert count");
    TORCH_CHECK(nGroups >= 1 && nGroups <= GMSQ_MAX_E,
                "grouped_matmul_swiglu_quant: unsupported expert count");
    int64_t prev = 0;
    for (int64_t g = 0; g < nGroups; ++g) {
        const int64_t v = group_list[g];
        TORCH_CHECK(v >= prev && v <= M, "grouped_matmul_swiglu_quant: group_list must be a cumsum <= M");
        ga.ends[g] = static_cast<int32_t>(v);
        prev = v;
    }
    ga.count = static_cast<int32_t>(nGroups);
    TORCH_CHECK(prev == M, "grouped_matmul_swiglu_quant: group_list[-1] must equal M");

    // The raw int32 matmul result is produced and fully written by the cube kernel, so it needs no
    // host side initialisation.
    auto mmBuf = torch::empty({M, N}, xc.options().dtype(torch::kInt32));
    int64_t wsSize = gmsq_calc_workspace_size();
    if (wsSize < 1024) {
        wsSize = 1024;
    }
    auto sysWs = torch::empty({wsSize}, xc.options().dtype(torch::kUInt8));

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    GM_ADDR xPtr = reinterpret_cast<GM_ADDR>(xc.data_ptr());
    GM_ADDR wPtr = reinterpret_cast<GM_ADDR>(wc.data_ptr());
    GM_ADDR wscPtr = reinterpret_cast<GM_ADDR>(wsc.data_ptr());
    GM_ADDR xscPtr = reinterpret_cast<GM_ADDR>(xsc.data_ptr());
    GM_ADDR yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());
    GM_ADDR ysPtr = reinterpret_cast<GM_ADDR>(yScale.data_ptr());
    GM_ADDR mmPtr = reinterpret_cast<GM_ADDR>(mmBuf.data_ptr());
    GM_ADDR wsPtr = reinterpret_cast<GM_ADDR>(sysWs.data_ptr());

    auto acl_call = [=]() -> int {
        launch_gmsq_cube(xPtr, wPtr, mmPtr, wsPtr, M, K, N, E, ga, stream);
        launch_gmsq_vec(mmPtr, xscPtr, wscPtr, yPtr, ysPtr, M, N, E, ga, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GroupedMatmulSwigluQuant", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("grouped_matmul_swiglu_quant", GmsqNpu);
}

}  // namespace cann_bench
