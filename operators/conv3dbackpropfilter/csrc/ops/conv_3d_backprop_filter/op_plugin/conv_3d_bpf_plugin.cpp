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
 * \file conv_3d_bpf_plugin.cpp
 * \brief Conv3DBackpropFilter torch bindings (compiled with g++)
 */

#include <vector>
#include <tuple>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/conv_3d_bpf_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("conv_3d_backprop_filter(Tensor x, Tensor grad, int[] strides, int[] pads, "
          "int[] dilations, int groups=1, int[] filter_size) -> Tensor");
}

static int64_t c3_pick(const std::vector<int64_t> &v, size_t i, int64_t dflt)
{
    return i < v.size() ? v[i] : dflt;
}

torch::Tensor conv_3d_backprop_filter_meta(
    const torch::Tensor &x, const torch::Tensor &grad,
    const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
    const std::vector<int64_t> &dilations, int64_t groups,
    const std::vector<int64_t> &filter_size)
{
    TORCH_CHECK(filter_size.size() == 5, "filter_size must have 5 elements.");
    TORCH_CHECK(x.dim() == 5 && grad.dim() == 5, "x and grad must be 5D (NCDHW).");
    TORCH_CHECK(groups >= 1, "groups must be positive.");
    TORCH_CHECK(x.scalar_type() == grad.scalar_type(), "x and grad must have the same dtype.");
    TORCH_CHECK(x.size(0) == grad.size(0), "x and grad must have the same batch size.");

    std::vector<int64_t> outSizes(filter_size.begin(), filter_size.end());
    return torch::empty(outSizes, x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("conv_3d_backprop_filter", conv_3d_backprop_filter_meta);
}

torch::Tensor conv_3d_backprop_filter_npu(
    const torch::Tensor &xIn, const torch::Tensor &gradIn,
    const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
    const std::vector<int64_t> &dilations, int64_t groups,
    const std::vector<int64_t> &filter_size)
{
    const c10::OptionalDeviceGuard guard(xIn.device());

    TORCH_CHECK(xIn.dim() == 5 && gradIn.dim() == 5, "x and grad must be 5D (NCDHW).");
    TORCH_CHECK(filter_size.size() == 5, "filter_size must have 5 elements.");
    TORCH_CHECK(strides.size() == 3, "strides must have 3 elements.");
    TORCH_CHECK(dilations.size() == 3, "dilations must have 3 elements.");
    TORCH_CHECK(pads.size() == 6, "pads must have 6 elements.");
    TORCH_CHECK(groups >= 1, "groups must be positive.");
    TORCH_CHECK(xIn.scalar_type() == gradIn.scalar_type(), "x and grad must share a dtype.");
    TORCH_CHECK(xIn.scalar_type() == torch::kFloat16 || xIn.scalar_type() == torch::kBFloat16,
                "conv_3d_backprop_filter supports float16 and bfloat16 only.");

    auto x = xIn.contiguous();
    auto grad = gradIn.contiguous();

    TORCH_CHECK(x.size(0) == grad.size(0), "x and grad must have the same batch size.");
    TORCH_CHECK(x.size(1) % groups == 0 && grad.size(1) % groups == 0,
                "C_in and C_out must be divisible by groups.");

    auto y = conv_3d_backprop_filter_meta(x, grad, strides, pads, dilations, groups, filter_size);

    const int64_t N = x.size(0);
    const int64_t Cin = x.size(1);
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);
    const int64_t Cout = grad.size(1);
    const int64_t Dout = grad.size(2);
    const int64_t Hout = grad.size(3);
    const int64_t Wout = grad.size(4);

    const int64_t Kd = filter_size[2];
    const int64_t Kh = filter_size[3];
    const int64_t Kw = filter_size[4];

    const int64_t sd = c3_pick(strides, 0, 1);
    const int64_t sh = c3_pick(strides, 1, 1);
    const int64_t sw = c3_pick(strides, 2, 1);
    const int64_t pd = c3_pick(pads, 0, 0);
    const int64_t ph = c3_pick(pads, 2, 0);
    const int64_t pw = c3_pick(pads, 4, 0);
    const int64_t ddc = c3_pick(dilations, 0, 1);
    const int64_t dhc = c3_pick(dilations, 1, 1);
    const int64_t dwc = c3_pick(dilations, 2, 1);

    const int64_t Cin_g = Cin / groups;
    const int64_t Cout_g = Cout / groups;
    const int64_t KhKw = Kh * Kw;
    if (KhKw <= 0 || Dout <= 0 || Hout <= 0 || Wout <= 0 || Cin_g <= 0 || Cout_g <= 0) {
        return y;
    }

    C3BpfTiling t = calc_conv_3d_bpf_tiling(N, Cin, D, H, W, Cout, Dout, Hout, Wout,
                                            Kd, Kh, Kw, sd, sh, sw, pd, ph, pw,
                                            ddc, dhc, dwc, groups, (int64_t)torch::elementSize(x.scalar_type()));
    if (t.numItems <= 0) {
        return y;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto dtype = x.scalar_type();
    const int64_t needSplit = (t.subPl > 1) ? 1 : 0;

    torch::Tensor xs;
    if (needSplit) {
        const int64_t numel = N * Cin * t.subPl * D * H * t.jmax;
        xs = torch::empty({numel}, x.options());
    }

    GM_ADDR xPtr = reinterpret_cast<GM_ADDR>(x.data_ptr());
    GM_ADDR gPtr = reinterpret_cast<GM_ADDR>(grad.data_ptr());
    GM_ADDR yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());
    GM_ADDR xsPtr = needSplit ? reinterpret_cast<GM_ADDR>(xs.data_ptr()) : xPtr;

    const int64_t rows = N * Cin * D * H;
    const int64_t DH = D * H;

    auto acl_call = [=]() -> int {
        if (needSplit) {
            if (dtype == torch::kFloat16) {
                launch_conv3d_bpf_split_half(xPtr, xsPtr, rows, W, t.jmax, t.subPl, DH, t.numBlocks, stream);
            } else {
                launch_conv3d_bpf_split_bf16(xPtr, xsPtr, rows, W, t.jmax, t.subPl, DH, t.numBlocks, stream);
            }
        }
        if (dtype == torch::kFloat16) {
            launch_conv3d_bpf_main_half(
                xPtr, gPtr, yPtr, xsPtr, N, Cin, D, H, W, Cout, Dout, Hout, Wout,
                Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, ddc, dhc, dwc,
                Cin_g, Cout_g, groups, t.rowStride, t.subPl, t.RW,
                t.CT, t.NT, t.RT, t.ncpg, t.nColTilesPerCi, t.numItems, t.numBlocks,
                t.b1x1, stream);
        } else {
            launch_conv3d_bpf_main_bf16(
                xPtr, gPtr, yPtr, xsPtr, N, Cin, D, H, W, Cout, Dout, Hout, Wout,
                Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, ddc, dhc, dwc,
                Cin_g, Cout_g, groups, t.rowStride, t.subPl, t.RW,
                t.CT, t.NT, t.RT, t.ncpg, t.nColTilesPerCi, t.numItems, t.numBlocks,
                t.b1x1, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Conv3DBackpropFilter", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("conv_3d_backprop_filter", conv_3d_backprop_filter_npu);
}

} // namespace cann_bench
