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
 * \file engram_plugin.cpp
 * \brief EngramGateFusion API layer - torch bindings (compiled with g++).
 *
 * No host side data movement happens here: the plugin only allocates the device workspace and
 * launches the device kernels in stream order.  The [j][d] <-> [d][j] transposes of the
 * conv_state are done by kernels, and every dtype conversion is a device Cast.  The weight prep
 * (cwT = transpose(cw)*wc and w12 = w1*w2) is NOT a kernel: each main block builds it in UB,
 * which removes one launch from every call.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/engram_launch.h"

namespace cann_bench {

namespace {

struct EngramSizes {
    int64_t B;
    int64_t L;
    int64_t HC;
    int64_t D;
    int64_t K;
    int64_t dil;
    int64_t H;
};

EngramSizes EngramCheck(const torch::Tensor &keys, const torch::Tensor &hidden,
                        const torch::Tensor &value, const torch::Tensor &w1,
                        const torch::Tensor &w2, const torch::Tensor &wc,
                        const torch::Tensor &cw, const c10::optional<torch::Tensor> &cs,
                        int64_t hcMult, int64_t kernelSize, int64_t dilation)
{
    TORCH_CHECK(keys.dim() == 4, "engram_gate_fusion: keys must be [B, L, HC, D].");
    TORCH_CHECK(hidden.sizes() == keys.sizes(), "engram_gate_fusion: hidden_states shape mismatch.");
    TORCH_CHECK(value.dim() == 3, "engram_gate_fusion: value must be [B, L, D].");

    EngramSizes s{};
    s.B = keys.size(0);
    s.L = keys.size(1);
    s.HC = keys.size(2);
    s.D = keys.size(3);
    s.K = kernelSize;
    s.dil = dilation;
    s.H = (s.K - 1) * s.dil;
    TORCH_CHECK(s.HC == hcMult, "engram_gate_fusion: hc_mult does not match keys.size(2).");
    TORCH_CHECK(value.size(0) == s.B && value.size(1) == s.L && value.size(2) == s.D,
                "engram_gate_fusion: value shape mismatch.");
    TORCH_CHECK(w1.dim() == 2 && w1.size(0) == s.HC && w1.size(1) == s.D,
                "engram_gate_fusion: norm1_weight must be [HC, D].");
    TORCH_CHECK(w2.dim() == 2 && w2.size(0) == s.HC && w2.size(1) == s.D,
                "engram_gate_fusion: norm2_weight must be [HC, D].");
    TORCH_CHECK(wc.dim() == 2 && wc.size(0) == s.HC && wc.size(1) == s.D,
                "engram_gate_fusion: conv_norm_weight must be [HC, D].");
    TORCH_CHECK(cw.dim() == 3 && cw.size(0) == s.HC * s.D && cw.size(2) == s.K,
                "engram_gate_fusion: conv_weight must be [HC*D, 1, K].");
    TORCH_CHECK(s.K >= 1 && s.dil >= 1, "engram_gate_fusion: invalid kernel/dilation.");
    TORCH_CHECK(keys.scalar_type() == torch::kBFloat16, "engram_gate_fusion: keys must be bf16.");
    TORCH_CHECK(hidden.scalar_type() == torch::kBFloat16, "engram_gate_fusion: hidden must be bf16.");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "engram_gate_fusion: value must be bf16.");
    TORCH_CHECK(w1.scalar_type() == torch::kFloat, "engram_gate_fusion: norm1_weight must be fp32.");
    TORCH_CHECK(w2.scalar_type() == torch::kFloat, "engram_gate_fusion: norm2_weight must be fp32.");
    TORCH_CHECK(wc.scalar_type() == torch::kFloat,
                "engram_gate_fusion: conv_norm_weight must be fp32.");
    TORCH_CHECK(cw.scalar_type() == torch::kFloat, "engram_gate_fusion: conv_weight must be fp32.");
    TORCH_CHECK(keys.is_contiguous() && hidden.is_contiguous() && value.is_contiguous(),
                "engram_gate_fusion: inputs must be contiguous.");
    TORCH_CHECK(w1.is_contiguous() && w2.is_contiguous() && wc.is_contiguous() && cw.is_contiguous(),
                "engram_gate_fusion: weights must be contiguous.");
    TORCH_CHECK(keys.size(0) > 0 && keys.size(1) > 0 && s.D > 0,
                "engram_gate_fusion: empty input is not supported.");
    if (cs.has_value() && cs->defined()) {
        TORCH_CHECK(cs->dim() == 3 && cs->size(0) == s.B && cs->size(1) == s.HC * s.D &&
                        cs->size(2) == s.H,
                    "engram_gate_fusion: conv_state must be [B, HC*D, (K-1)*dil].");
        TORCH_CHECK(cs->scalar_type() == torch::kBFloat16,
                    "engram_gate_fusion: conv_state must be bf16.");
        TORCH_CHECK(cs->is_contiguous(), "engram_gate_fusion: conv_state must be contiguous.");
    }
    return s;
}

std::tuple<torch::Tensor, torch::Tensor> EngramOuts(const torch::Tensor &keys, const EngramSizes &s)
{
    auto out = torch::empty({s.B, s.L, s.HC, s.D}, keys.options());
    auto sout = torch::empty({s.B, s.HC * s.D, s.H}, keys.options());
    return std::make_tuple(out, sout);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def(
        "engram_gate_fusion(Tensor keys, Tensor hidden_states, Tensor value, Tensor norm1_weight, "
        "Tensor norm2_weight, Tensor conv_norm_weight, Tensor conv_weight, Tensor? conv_state=None, "
        "int hc_mult=4, int hidden_size=1024, int kernel_size=4, int dilation=3, "
        "float norm_eps=1e-5) -> (Tensor, Tensor)");
}

std::tuple<torch::Tensor, torch::Tensor> engram_meta(
    const torch::Tensor &keys, const torch::Tensor &hidden, const torch::Tensor &value,
    const torch::Tensor &w1, const torch::Tensor &w2, const torch::Tensor &wc,
    const torch::Tensor &cw, const c10::optional<torch::Tensor> &cs, int64_t hcMult,
    int64_t hiddenSize, int64_t kernelSize, int64_t dilation, double normEps)
{
    (void)hiddenSize;
    (void)normEps;
    EngramSizes s = EngramCheck(keys, hidden, value, w1, w2, wc, cw, cs, hcMult, kernelSize, dilation);
    return EngramOuts(keys, s);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("engram_gate_fusion", engram_meta);
}

std::tuple<torch::Tensor, torch::Tensor> engram_npu(
    const torch::Tensor &keys, const torch::Tensor &hidden, const torch::Tensor &value,
    const torch::Tensor &w1, const torch::Tensor &w2, const torch::Tensor &wc,
    const torch::Tensor &cw, const c10::optional<torch::Tensor> &cs, int64_t hcMult,
    int64_t hiddenSize, int64_t kernelSize, int64_t dilation, double normEps)
{
    (void)hiddenSize;
    const c10::OptionalDeviceGuard guard(keys.device());
    EngramSizes s = EngramCheck(keys, hidden, value, w1, w2, wc, cw, cs, hcMult, kernelSize, dilation);
    auto outs = EngramOuts(keys, s);
    torch::Tensor out = std::get<0>(outs);
    torch::Tensor sout = std::get<1>(outs);

    const bool hasState = cs.has_value() && cs->defined() && s.H > 0;
    const int64_t Dpad = (s.D + 15) / 16 * 16;

    EngramTiling t = calc_engram_tiling(s.B, s.L, s.HC, s.D, s.K, s.dil);

    /* One fp32 device workspace holding just the [B, HC, H, Dpad] carrier of the state rows,
     * plus [wsi] when a conv_state is supplied.  The old cwT / w12 workspaces are gone: the main
     * kernel builds both of them in UB. */
    const int64_t ws_elems = (s.H > 0) ? (s.B * s.HC * s.H * Dpad) : 1;
    auto fopt = keys.options().dtype(torch::kFloat32);
    auto wspace = torch::empty({ws_elems}, fopt);
    torch::Tensor wsi;
    if (hasState) {
        wsi = torch::empty({s.B * s.HC * s.H * s.D}, fopt);
    }

    float* wsBase = wspace.data_ptr<float>();
    auto ws_ptr = (GM_ADDR)wsBase;

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto keys_ptr = (GM_ADDR)keys.data_ptr();
    auto hidden_ptr = (GM_ADDR)hidden.data_ptr();
    auto value_ptr = (GM_ADDR)value.data_ptr();
    auto w1_ptr = (GM_ADDR)w1.data_ptr();
    auto w2_ptr = (GM_ADDR)w2.data_ptr();
    auto wc_ptr = (GM_ADDR)wc.data_ptr();
    auto cw_ptr = (GM_ADDR)cw.data_ptr();
    auto out_ptr = (GM_ADDR)out.data_ptr();
    auto sout_ptr = (GM_ADDR)sout.data_ptr();
    auto wsi_ptr = hasState ? (GM_ADDR)wsi.data_ptr<float>() : (GM_ADDR)nullptr;
    auto cs_ptr = hasState ? (GM_ADDR)cs->data_ptr() : (GM_ADDR)nullptr;

    const int64_t B = s.B;
    const int64_t L = s.L;
    const int64_t HC = s.HC;
    const int64_t D = s.D;
    const int64_t K = s.K;
    const int64_t dil = s.dil;
    const int64_t H = s.H;
    const int64_t hasStateI = hasState ? 1 : 0;
    const float eps = (float)normEps;
    const EngramTiling tt = t;

    auto acl_call = [=]() -> int {
        if (hasStateI != 0) {
            launch_engram_state_in(cs_ptr, wsi_ptr, B, HC, D, H, tt.dtState, tt.nDT, stream);
        }
        launch_engram_main(keys_ptr, hidden_ptr, value_ptr, w1_ptr, w2_ptr, wc_ptr, cw_ptr, wsi_ptr,
                           ws_ptr, out_ptr, B, L, HC, D, K, dil, H, hasStateI, eps, tt.nChunks,
                           tt.chunkLt, stream);
        if (H > 0) {
            launch_engram_state_out(ws_ptr, sout_ptr, B, HC, D, H, tt.dtState, tt.nDT, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("EngramGateFusion", acl_call);
    return std::make_tuple(out, sout);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("engram_gate_fusion", engram_npu);
}

}  // namespace cann_bench
