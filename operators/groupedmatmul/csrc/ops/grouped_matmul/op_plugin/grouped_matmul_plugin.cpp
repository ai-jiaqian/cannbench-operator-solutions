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
 * \file grouped_matmul_plugin.cpp
 * \brief GroupedMatmul API layer - torch bindings (compiled with g++).
 *
 * grouped_matmul(x, weight, group_list, bias=None, split_item=0, transpose_weight=False) -> Tensor[]
 *   y[rows_g] = x[rows_g] @ weight[g] (+ bias[g]),  rows_g = [group_list[g-1], group_list[g])
 *   split_item 0/1 -> one tensor per expert (views along M of the joined output)
 *   split_item 2/3 -> a single [M, N] tensor
 *
 * group_list is a cumsum attribute (a Python list), so the boundaries are copied into a by-value POD and
 * travel to the device with the kernel arguments.  No tensor data is ever read back to the host: the whole
 * arithmetic, including the workspace contents, lives inside the custom device kernel.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include <cstring>
#include <vector>

#include "../op_kernel/grouped_matmul_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    // NOTE: `group_list` precedes the optional `bias` so that `bias` can carry a default value, which
    // lets callers omit it entirely (a schema argument without a default must always be supplied).
    m.def("grouped_matmul(Tensor x, Tensor weight, int[] group_list, Tensor? bias=None, int split_item=0, "
          "bool transpose_weight=False) -> Tensor[]");
}

static int32_t GmmDtypeCode(const torch::Tensor& t)
{
    const auto st = t.scalar_type();
    if (st == torch::kFloat16) {
        return GMM_DT_HALF;
    }
    if (st == torch::kBFloat16) {
        return GMM_DT_BF16;
    }
    if (st == torch::kFloat32) {
        return GMM_DT_FLOAT;
    }
    TORCH_CHECK(false, "grouped_matmul: unsupported dtype");
    return GMM_DT_FLOAT;
}

/*!
 * \brief Materialise the operator result list from the joined [M, N] output.
 */
static c10::List<torch::Tensor> GmmMakeOutputs(const torch::Tensor& y, int64_t E,
                                               const std::vector<int64_t>& ends, int64_t split_item)
{
    c10::List<torch::Tensor> out;
    if (split_item == 0 || split_item == 1) {
        int64_t s = 0;
        for (int64_t g = 0; g < E; ++g) {
            const int64_t e = ends[static_cast<size_t>(g)];
            out.push_back(y.narrow(0, s, e - s));
            s = e;
        }
    } else {
        out.push_back(y);
    }
    return out;
}

static std::vector<int64_t> GmmCollectEnds(c10::List<int64_t> group_list, int64_t E, int64_t M)
{
    std::vector<int64_t> ends(static_cast<size_t>(E));
    int64_t prev = 0;
    for (int64_t g = 0; g < E; ++g) {
        const int64_t v = group_list[static_cast<size_t>(g)];
        TORCH_CHECK(v >= prev, "grouped_matmul: group_list must be a non-decreasing cumsum");
        TORCH_CHECK(v <= M, "grouped_matmul: group_list values must not exceed M");
        ends[static_cast<size_t>(g)] = v;
        prev = v;
    }
    TORCH_CHECK(prev == M, "grouped_matmul: group_list[-1] must equal M");
    return ends;
}

static c10::List<torch::Tensor> grouped_matmul_meta(const torch::Tensor& x, const torch::Tensor& weight,
                                                    c10::List<int64_t> group_list,
                                                    const c10::optional<torch::Tensor>& bias, int64_t split_item,
                                                    bool transpose_weight)
{
    (void)bias;
    TORCH_CHECK(x.dim() == 2, "grouped_matmul: x must be 2D [M, K]");
    TORCH_CHECK(weight.dim() == 3, "grouped_matmul: weight must be 3D [E, K, N] or [E, N, K]");
    const int64_t M = x.size(0);
    const int64_t E = weight.size(0);
    const int64_t N = transpose_weight ? weight.size(1) : weight.size(2);
    TORCH_CHECK(static_cast<int64_t>(group_list.size()) == E,
                "grouped_matmul: group_list length must equal the number of experts");
    const std::vector<int64_t> ends = GmmCollectEnds(group_list, E, M);
    auto y = torch::empty({M, N}, x.options());
    return GmmMakeOutputs(y, E, ends, split_item);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("grouped_matmul", grouped_matmul_meta);
}

static c10::List<torch::Tensor> grouped_matmul_npu(const torch::Tensor& x, const torch::Tensor& weight,
                                                   c10::List<int64_t> group_list,
                                                   const c10::optional<torch::Tensor>& bias, int64_t split_item,
                                                   bool transpose_weight)
{
    const c10::OptionalDeviceGuard guard(x.device());

    auto xc = x.contiguous();
    auto wc = weight.contiguous();

    const int64_t M = xc.size(0);
    const int64_t K = xc.size(1);
    const int64_t E = wc.size(0);
    TORCH_CHECK(E >= 1, "grouped_matmul: weight must contain at least one expert");
    TORCH_CHECK(E <= GMM_MAX_E, "grouped_matmul: at most ", GMM_MAX_E, " experts are supported");
    TORCH_CHECK(static_cast<int64_t>(group_list.size()) == E,
                "grouped_matmul: group_list length must equal the number of experts");

    const int64_t N = transpose_weight ? wc.size(1) : wc.size(2);
    const int64_t kStored = transpose_weight ? wc.size(2) : wc.size(1);
    TORCH_CHECK(kStored == K, "grouped_matmul: K dimension mismatch between x and weight");

    const std::vector<int64_t> ends = GmmCollectEnds(group_list, E, M);

    const bool hasBias = bias.has_value();
    torch::Tensor bc;
    if (hasBias) {
        bc = bias->contiguous();
        TORCH_CHECK(bc.numel() == E * N, "grouped_matmul: bias must have shape [E, N]");
    }

    auto y = torch::empty({M, N}, xc.options());
    auto out = GmmMakeOutputs(y, E, ends, split_item);

    const int32_t xDtype = GmmDtypeCode(xc);
    const int32_t biasDtype = hasBias ? GmmDtypeCode(bc) : xDtype;
    const int32_t hasBiasI = hasBias ? 1 : 0;
    const int32_t transBI = transpose_weight ? 1 : 0;

    GmmGroupArg ga;
    std::memset(&ga, 0, sizeof(ga));
    ga.count = static_cast<int32_t>(E);
    for (int64_t g = 0; g < E; ++g) {
        ga.ends[static_cast<size_t>(g)] = static_cast<int32_t>(ends[static_cast<size_t>(g)]);
    }

    int64_t wsSize = calc_gmm_workspace_size();
    if (wsSize < 1024) {
        wsSize = 1024;
    }
    auto sysWs = torch::empty({wsSize}, xc.options().dtype(torch::kUInt8));

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    GM_ADDR xPtr = reinterpret_cast<GM_ADDR>(xc.data_ptr());
    GM_ADDR wPtr = reinterpret_cast<GM_ADDR>(wc.data_ptr());
    GM_ADDR bPtr = hasBias ? reinterpret_cast<GM_ADDR>(bc.data_ptr()) : nullptr;
    GM_ADDR yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());
    GM_ADDR wsPtr = reinterpret_cast<GM_ADDR>(sysWs.data_ptr());

    auto acl_call = [=]() -> int {
        launch_gmm(xPtr, wPtr, bPtr, yPtr, wsPtr, M, N, K, ga, xDtype, biasDtype, hasBiasI, transBI, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GroupedMatmul", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("grouped_matmul", grouped_matmul_npu);
}

}  // namespace cann_bench
