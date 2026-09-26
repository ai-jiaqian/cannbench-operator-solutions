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
 * \file mha_plugin.cpp
 * \brief MHA API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <tuple>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/mha_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("mha(Tensor query, Tensor key, Tensor value, float scaleValue=-1.0, bool is_causal=False) -> Tensor");
}

torch::Tensor mha_meta(const torch::Tensor &query, const torch::Tensor &key, const torch::Tensor &value,
                       double scaleValue, bool is_causal)
{
    (void)key;
    (void)value;
    (void)scaleValue;
    (void)is_causal;
    TORCH_CHECK(query.dim() == 4, "mha: query must be [B, S, N, D].");
    return torch::empty_like(query);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("mha", mha_meta);
}

torch::Tensor mha_npu(const torch::Tensor &query, const torch::Tensor &key, const torch::Tensor &value,
                      double scaleValue, bool is_causal)
{
    const c10::OptionalDeviceGuard guard(query.device());
    auto q = query.is_contiguous() ? query : query.contiguous();
    auto k = key.is_contiguous() ? key : key.contiguous();
    auto v = value.is_contiguous() ? value : value.contiguous();
    auto y = torch::empty_like(q);

    const int64_t B = q.size(0);
    const int64_t S = q.size(1);
    const int64_t N = q.size(2);
    const int64_t D = q.size(3);
    const int64_t Skv = k.size(1);

    float scale = static_cast<float>(scaleValue);
    if (!(scale > 0.0f)) {
        scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    }

    int64_t numBlocks = 1;
    int64_t BM = 1;
    int64_t BN = 32;
    int64_t totalItems = 1;
    int64_t itemsPerCore = 1;
    std::tie(numBlocks, BM, BN, totalItems, itemsPerCore) = calc_mha_tiling_params(B, S, Skv, N, D);
    const int64_t numRowBlk = (S + BM - 1) / BM;

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto qPtr = (GM_ADDR)q.data_ptr();
    auto kPtr = (GM_ADDR)k.data_ptr();
    auto vPtr = (GM_ADDR)v.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();

    const int32_t causal = is_causal ? 1 : 0;
    const bool isBf16 = (q.scalar_type() == torch::kBFloat16);

    auto acl_call = [=]() -> int {
        if (isBf16) {
            launch_mha_kernel_bf16(qPtr, kPtr, vPtr, yPtr, B, S, Skv, N, D, scale, causal, numRowBlk,
                                   totalItems, itemsPerCore, BM, BN, numBlocks, stream);
        } else {
            launch_mha_kernel_f16(qPtr, kPtr, vPtr, yPtr, B, S, Skv, N, D, scale, causal, numRowBlk,
                                  totalItems, itemsPerCore, BM, BN, numBlocks, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Mha", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("mha", mha_npu);
}

}  // namespace cann_bench
