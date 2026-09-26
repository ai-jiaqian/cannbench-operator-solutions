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
 * \file depthwise_conv_2d_plugin.cpp
 * \brief DepthwiseConv2D API layer - torch bindings (compiled with g++).
 *
 * Only shapes / strides / dtypes are inspected on the host; every arithmetic and data movement step
 * happens inside the custom device kernel.
 */

#include <algorithm>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/depthwise_conv_2d_launch.h"

namespace cann_bench {

namespace {

int64_t dwPair(const c10::List<int64_t>& v, size_t i, int64_t dflt)
{
    const size_t n = v.size();
    if (n == 0) {
        return dflt;
    }
    if (i < n) {
        return v[i];
    }
    return v[n - 1];
}

std::vector<int64_t> dwVec(const c10::List<int64_t>& v)
{
    std::vector<int64_t> out;
    const size_t n = v.size();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(v[i]);
    }
    return out;
}

struct DWGeom {
    int64_t N, C, H, W;
    int64_t Kh, Kw;
    int64_t sh, sw, ph, pw, dh, dw;
    int64_t Hout, Wout;
};

void dwCheck(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
             const c10::List<int64_t>& kernelSize, int64_t groups)
{
    TORCH_CHECK(x.dim() == 4, "depthwise_conv_2d: x must be a 4-D tensor [N, C, H, W].");
    TORCH_CHECK(weight.dim() == 3, "depthwise_conv_2d: weight must be a 3-D tensor [C, K_h, K_w].");
    TORCH_CHECK(bias.dim() == 1, "depthwise_conv_2d: bias must be a 1-D tensor [C].");
    TORCH_CHECK(weight.size(0) == x.size(1),
                "depthwise_conv_2d: weight channels must match x channels.");
    TORCH_CHECK(bias.size(0) == x.size(1), "depthwise_conv_2d: bias length must match x channels.");
    TORCH_CHECK(groups == x.size(1), "depthwise_conv_2d: groups must equal the number of channels.");
    if (kernelSize.size() >= 2) {
        TORCH_CHECK(kernelSize[0] == weight.size(1) && kernelSize[1] == weight.size(2),
                    "depthwise_conv_2d: kernelSize must match weight.shape[1:].");
    }
    const auto dt = x.scalar_type();
    TORCH_CHECK(dt == torch::kFloat32 || dt == torch::kFloat16 || dt == torch::kBFloat16,
                "depthwise_conv_2d: only float32, float16 and bfloat16 are supported.");
    TORCH_CHECK(weight.scalar_type() == dt && bias.scalar_type() == dt,
                "depthwise_conv_2d: x, weight and bias must share the same dtype.");
}

DWGeom dwGeom(const torch::Tensor& x, const torch::Tensor& weight,
              const c10::List<int64_t>& stride, const c10::List<int64_t>& padding,
              const c10::List<int64_t>& dilation)
{
    DWGeom g;
    g.N = x.size(0);
    g.C = x.size(1);
    g.H = x.size(2);
    g.W = x.size(3);
    g.Kh = weight.size(1);
    g.Kw = weight.size(2);
    g.sh = dwPair(stride, 0, 1);
    g.sw = dwPair(stride, 1, 1);
    g.ph = dwPair(padding, 0, 0);
    g.pw = dwPair(padding, 1, 0);
    g.dh = dwPair(dilation, 0, 1);
    g.dw = dwPair(dilation, 1, 1);
    if (g.sh < 1) g.sh = 1;
    if (g.sw < 1) g.sw = 1;
    if (g.dh < 1) g.dh = 1;
    if (g.dw < 1) g.dw = 1;
    g.Hout = (g.H + 2 * g.ph - g.dh * (g.Kh - 1) - 1) / g.sh + 1;
    g.Wout = (g.W + 2 * g.pw - g.dw * (g.Kw - 1) - 1) / g.sw + 1;
    return g;
}

torch::Tensor dwMakeOutput(const torch::Tensor& x, const torch::Tensor& weight,
                           const torch::Tensor& bias, const c10::List<int64_t>& kernelSize,
                           const c10::List<int64_t>& stride, const c10::List<int64_t>& padding,
                           const c10::List<int64_t>& dilation, int64_t groups)
{
    dwCheck(x, weight, bias, kernelSize, groups);
    const DWGeom g = dwGeom(x, weight, stride, padding, dilation);
    TORCH_CHECK(g.Hout >= 1, "depthwise_conv_2d: computed H_out must be >= 1.");
    TORCH_CHECK(g.Wout >= 1, "depthwise_conv_2d: computed W_out must be >= 1.");
    return torch::empty({g.N, g.C, g.Hout, g.Wout}, x.options());
}

} // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("depthwise_conv_2d(Tensor x, Tensor weight, Tensor bias, int[] kernelSize, int[] stride, "
          "int[] padding, int[] dilation, int groups) -> Tensor y");
}

torch::Tensor depthwise_conv_2d_meta(const torch::Tensor& x, const torch::Tensor& weight,
                                     const torch::Tensor& bias, c10::List<int64_t> kernelSize,
                                     c10::List<int64_t> stride, c10::List<int64_t> padding,
                                     c10::List<int64_t> dilation, int64_t groups)
{
    return dwMakeOutput(x, weight, bias, kernelSize, stride, padding, dilation, groups);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("depthwise_conv_2d", depthwise_conv_2d_meta);
}

torch::Tensor depthwise_conv_2d_npu(const torch::Tensor& x, const torch::Tensor& weight,
                                    const torch::Tensor& bias, c10::List<int64_t> kernelSize,
                                    c10::List<int64_t> stride, c10::List<int64_t> padding,
                                    c10::List<int64_t> dilation, int64_t groups)
{
    const c10::OptionalDeviceGuard guard(x.device());
    dwCheck(x, weight, bias, kernelSize, groups);

    auto xc = x.contiguous();
    auto wc = weight.contiguous();
    auto bc = bias.contiguous();
    auto y = dwMakeOutput(xc, wc, bc, kernelSize, stride, padding, dilation, groups);

    const DWGeom g = dwGeom(xc, wc, stride, padding, dilation);
    const int64_t elemSize = (int64_t)xc.element_size();
    DWConvParams p = calc_depthwise_conv_2d_tiling(g.N, g.C, g.H, g.W, g.Kh, g.Kw, g.sh, g.sw, g.ph,
                                                   g.pw, g.dh, g.dw, g.Hout, g.Wout, elemSize);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = (GM_ADDR)xc.data_ptr();
    auto wPtr = (GM_ADDR)wc.data_ptr();
    auto bPtr = (GM_ADDR)bc.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        const auto dt = xc.scalar_type();
        if (dt == torch::kFloat32) {
            launch_depthwise_conv_2d_float(xPtr, wPtr, bPtr, yPtr, p, stream);
        } else if (dt == torch::kFloat16) {
            launch_depthwise_conv_2d_half(xPtr, wPtr, bPtr, yPtr, p, stream);
        } else {
            launch_depthwise_conv_2d_bfloat16(xPtr, wPtr, bPtr, yPtr, p, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("DepthwiseConv2D", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("depthwise_conv_2d", depthwise_conv_2d_npu);
}

} // namespace cann_bench
