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
 * \file rope_plugin.cpp
 * \brief ApplyRotaryPosEmb API layer - torch bindings (compiled with g++).
 *
 * Only shape / dtype metadata is inspected here; all arithmetic and data movement happens inside the
 * custom device kernel.
 */

#include <tuple>
#include <string>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/rope_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("apply_rotary_pos_emb(Tensor query, Tensor key, Tensor cos, Tensor sin, int layout=0, "
          "str rotaryMode=\"half\") -> (Tensor, Tensor)");
}

static std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb_meta(
    const torch::Tensor &query, const torch::Tensor &key, const torch::Tensor &cos,
    const torch::Tensor &sin, int64_t layout, c10::string_view rotaryMode)
{
    (void)cos;
    (void)sin;
    (void)layout;
    (void)rotaryMode;
    TORCH_CHECK(query.dim() == 4, "query must be 4-D (B,S,N,D) or (B,N,S,D)");
    TORCH_CHECK(key.dim() == 4, "key must be 4-D (B,S,N,D) or (B,N,S,D)");
    TORCH_CHECK(query.sizes() == key.sizes(), "query and key must have the same shape");
    return std::make_tuple(torch::empty_like(query), torch::empty_like(key));
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("apply_rotary_pos_emb", apply_rotary_pos_emb_meta);
}

static std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb_npu(
    const torch::Tensor &query, const torch::Tensor &key, const torch::Tensor &cos,
    const torch::Tensor &sin, int64_t layout, c10::string_view rotaryMode)
{
    TORCH_CHECK(query.dim() == 4, "query must be 4-D (B,S,N,D) or (B,N,S,D)");
    TORCH_CHECK(key.dim() == 4, "key must be 4-D (B,S,N,D) or (B,N,S,D)");
    TORCH_CHECK(query.sizes() == key.sizes(), "query and key must have the same shape");
    TORCH_CHECK(layout == 0 || layout == 1, "layout must be 0 or 1");

    const int64_t B = query.size(0);
    int64_t S = 0;
    int64_t N = 0;
    if (layout == 0) {
        S = query.size(1);
        N = query.size(2);
    } else {
        N = query.size(1);
        S = query.size(2);
    }
    const int64_t D = query.size(3);
    TORCH_CHECK(D >= 2 && (D % 2) == 0, "head_dim must be positive and even");
    const int64_t H = D / 2;

    int64_t cos3d = 0;
    if (cos.dim() == 3) {
        cos3d = 1;
        TORCH_CHECK(cos.size(0) == B && cos.size(1) == S && cos.size(2) == H, "cos/sin shape mismatch");
    } else {
        TORCH_CHECK(cos.dim() == 2 && cos.size(0) == S && cos.size(1) == H, "cos/sin shape mismatch");
    }
    TORCH_CHECK(sin.sizes() == cos.sizes(), "cos and sin must have the same shape");

    int64_t mode = 0;
    if (rotaryMode == "interleaved") {
        mode = 1;
    } else {
        TORCH_CHECK(rotaryMode == "half", "rotaryMode must be \"half\" or \"interleaved\"");
        mode = 0;
    }

    const auto dt = query.scalar_type();
    int64_t esz = 0;
    int32_t which = -1;
    if (dt == torch::kFloat32) {
        esz = 4;
        which = 0;
    } else if (dt == torch::kFloat16) {
        esz = 2;
        which = 1;
    } else if (dt == torch::kBFloat16) {
        esz = 2;
        which = 2;
    } else {
        TORCH_CHECK(false, "unsupported dtype for apply_rotary_pos_emb");
    }
    TORCH_CHECK(key.scalar_type() == dt && cos.scalar_type() == dt && sin.scalar_type() == dt,
                "all inputs must share the same dtype");

    const c10::OptionalDeviceGuard guard(query.device());
    auto q = query.contiguous();
    auto k = key.contiguous();
    auto cc = cos.contiguous();
    auto ss = sin.contiguous();
    auto qo = torch::empty_like(q);
    auto ko = torch::empty_like(k);
    if (q.numel() == 0) {
        return std::make_tuple(qo, ko);
    }

    int64_t numBlocks = 1;
    int64_t rmax = 1;
    int64_t budget = 1 << 20;
    std::tie(numBlocks, rmax, budget) = calc_rope_tiling_params(B, S, N, D, layout, mode, esz);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    GM_ADDR qp = reinterpret_cast<GM_ADDR>(q.data_ptr());
    GM_ADDR kp = reinterpret_cast<GM_ADDR>(k.data_ptr());
    GM_ADDR cp = reinterpret_cast<GM_ADDR>(cc.data_ptr());
    GM_ADDR sp = reinterpret_cast<GM_ADDR>(ss.data_ptr());
    GM_ADDR qop = reinterpret_cast<GM_ADDR>(qo.data_ptr());
    GM_ADDR kop = reinterpret_cast<GM_ADDR>(ko.data_ptr());

    auto acl_call = [=]() -> int {
        if (which == 0) {
            launch_rope_kernel_float(qp, kp, cp, sp, qop, kop, B, S, N, D, layout, mode, cos3d,
                                     numBlocks, rmax, budget, stream);
        } else if (which == 1) {
            launch_rope_kernel_half(qp, kp, cp, sp, qop, kop, B, S, N, D, layout, mode, cos3d,
                                    numBlocks, rmax, budget, stream);
        } else {
            launch_rope_kernel_bf16(qp, kp, cp, sp, qop, kop, B, S, N, D, layout, mode, cos3d,
                                    numBlocks, rmax, budget, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ApplyRotaryPosEmb", acl_call);
    return std::make_tuple(qo, ko);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("apply_rotary_pos_emb", apply_rotary_pos_emb_npu);
}

} // namespace cann_bench
