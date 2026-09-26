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
 * \file conv2d_plugin.cpp
 * \brief Conv2D API layer - torch bindings (compiled with g++).
 *
 * Only shapes / strides / dtypes are inspected on the host; every arithmetic and data movement step
 * happens inside the custom device kernel.
 */

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/conv2d_launch.h"

namespace cann_bench {

struct Conv2dGeom {
    int64_t N, Cin, H, W;
    int64_t Cout, Kh, Kw;
    int64_t sh, sw;
    int64_t pt, pb, pl, pr;
    int64_t dh, dw;
    int64_t Hout, Wout;
};

static inline int64_t conv2d_pick(c10::IntArrayRef a, int64_t idx, int64_t fallback)
{
    if (a.size() == 0) {
        return fallback;
    }
    if (idx < (int64_t)a.size()) {
        return a[idx];
    }
    return a[a.size() - 1];
}

// pads follow the prototype order [pad_top, pad_bottom, pad_left, pad_right]; the 2-element and
// 1-element spellings used by some callers are folded onto the same four values.
static void conv2d_pads(c10::IntArrayRef pads, int64_t &pt, int64_t &pb, int64_t &pl, int64_t &pr)
{
    if (pads.size() >= 4) {
        pt = pads[0]; pb = pads[1]; pl = pads[2]; pr = pads[3];
    } else if (pads.size() == 2) {
        pt = pads[0]; pb = pads[0]; pl = pads[1]; pr = pads[1];
    } else {
        pt = conv2d_pick(pads, 0, 0);
        pb = pt; pl = pt; pr = pt;
    }
}

static Conv2dGeom conv2d_geom(const torch::Tensor &x, const torch::Tensor &filter,
                              c10::IntArrayRef strides, c10::IntArrayRef pads,
                              c10::IntArrayRef dilations)
{
    Conv2dGeom g;
    g.N = x.size(0);
    g.Cin = x.size(1);
    g.H = x.size(2);
    g.W = x.size(3);
    g.Cout = filter.size(0);
    g.Kh = filter.size(2);
    g.Kw = filter.size(3);
    g.sh = conv2d_pick(strides, 0, 1);
    g.sw = conv2d_pick(strides, 1, 1);
    conv2d_pads(pads, g.pt, g.pb, g.pl, g.pr);
    g.dh = conv2d_pick(dilations, 0, 1);
    g.dw = conv2d_pick(dilations, 1, 1);
    if (g.sh < 1) g.sh = 1;
    if (g.sw < 1) g.sw = 1;
    if (g.dh < 1) g.dh = 1;
    if (g.dw < 1) g.dw = 1;
    g.Hout = (g.H + g.pt + g.pb - g.dh * (g.Kh - 1) - 1) / g.sh + 1;
    g.Wout = (g.W + g.pl + g.pr - g.dw * (g.Kw - 1) - 1) / g.sw + 1;
    return g;
}

static void conv2d_check(const torch::Tensor &x, const torch::Tensor &filter, const torch::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 4, "conv_2d: x must be a 4-D tensor [N, C_in, H, W].");
    TORCH_CHECK(filter.dim() == 4, "conv_2d: filter must be a 4-D tensor [C_out, C_in, K_h, K_w].");
    TORCH_CHECK(bias.dim() == 1, "conv_2d: bias must be a 1-D tensor [C_out].");
    TORCH_CHECK(filter.size(1) == x.size(1), "conv_2d: filter C_in must match x C_in.");
    TORCH_CHECK(bias.size(0) == filter.size(0), "conv_2d: bias length must match C_out.");
    const auto dt = x.scalar_type();
    TORCH_CHECK(dt == torch::kFloat32 || dt == torch::kFloat16 || dt == torch::kBFloat16,
                "conv_2d: only float32, float16 and bfloat16 are supported.");
    TORCH_CHECK(filter.scalar_type() == dt && bias.scalar_type() == dt,
                "conv_2d: x, filter and bias must share the same dtype.");
}

torch::Tensor conv2d_meta(const torch::Tensor &x, const torch::Tensor &filter, const torch::Tensor &bias,
                          c10::IntArrayRef strides, c10::IntArrayRef pads, c10::IntArrayRef dilations)
{
    conv2d_check(x, filter, bias);
    Conv2dGeom g = conv2d_geom(x, filter, strides, pads, dilations);
    TORCH_CHECK(g.Hout >= 1 && g.Wout >= 1, "conv_2d: computed output spatial size must be >= 1.");
    std::vector<int64_t> outSizes{g.N, g.Cout, g.Hout, g.Wout};
    return torch::empty(outSizes, x.options());
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("conv_2d(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations) -> Tensor y");
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("conv_2d", conv2d_meta);
}

torch::Tensor conv2d_npu(const torch::Tensor &x, const torch::Tensor &filter, const torch::Tensor &bias,
                         c10::IntArrayRef strides, c10::IntArrayRef pads, c10::IntArrayRef dilations)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto fc = filter.contiguous();
    auto bc = bias.contiguous();
    auto y = conv2d_meta(xc, fc, bc, strides, pads, dilations);

    Conv2dGeom g = conv2d_geom(xc, fc, strides, pads, dilations);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t dtypeCode = CONV2D_DTYPE_HALF;
    if (xc.scalar_type() == torch::kFloat32) {
        dtypeCode = CONV2D_DTYPE_FLOAT;
    } else if (xc.scalar_type() == torch::kBFloat16) {
        dtypeCode = CONV2D_DTYPE_BF16;
    }

    Conv2dTiling tl = calc_conv2d_tiling(g.N, g.Cin, g.H, g.W, g.Cout, g.Kh, g.Kw,
                                         g.sh, g.sw, g.dh, g.dw, g.pt, g.pb, g.pl, g.pr,
                                         g.Hout, g.Wout, dtypeCode);

    GM_ADDR xPtr = (GM_ADDR)xc.data_ptr();
    GM_ADDR wPtr = (GM_ADDR)fc.data_ptr();
    GM_ADDR bPtr = (GM_ADDR)bc.data_ptr();
    GM_ADDR yPtr = (GM_ADDR)y.data_ptr();
    const auto dtype = xc.scalar_type();

    auto acl_call = [=]() -> int {
        if (dtype == torch::kFloat32) {
            launch_conv2d_kernel_float(xPtr, wPtr, bPtr, yPtr,
                                       g.N, g.Cin, g.H, g.W, g.Cout, g.Kh, g.Kw,
                                       g.sh, g.sw, g.dh, g.dw, g.pt, g.pl, g.pr, g.Hout, g.Wout,
                                       tl.CT, tl.HBT, tl.Pr, tl.PitW, tl.ciChunk,
                                       tl.numCoBlk, tl.numHoBlk, tl.totalItems,
                                       tl.gsP, tl.gsRep, tl.khGroup, tl.numBlocks, stream);
        } else if (dtype == torch::kFloat16) {
            launch_conv2d_kernel_half(xPtr, wPtr, bPtr, yPtr,
                                      g.N, g.Cin, g.H, g.W, g.Cout, g.Kh, g.Kw,
                                      g.sh, g.sw, g.dh, g.dw, g.pt, g.pl, g.pr, g.Hout, g.Wout,
                                      tl.CT, tl.HBT, tl.Pr, tl.PitW, tl.ciChunk,
                                      tl.numCoBlk, tl.numHoBlk, tl.totalItems,
                                      tl.gsP, tl.gsRep, tl.khGroup, tl.numBlocks, stream);
        } else {
            launch_conv2d_kernel_bfloat16(xPtr, wPtr, bPtr, yPtr,
                                          g.N, g.Cin, g.H, g.W, g.Cout, g.Kh, g.Kw,
                                          g.sh, g.sw, g.dh, g.dw, g.pt, g.pl, g.pr, g.Hout, g.Wout,
                                          tl.CT, tl.HBT, tl.Pr, tl.PitW, tl.ciChunk,
                                          tl.numCoBlk, tl.numHoBlk, tl.totalItems,
                                          tl.gsP, tl.gsRep, tl.khGroup, tl.numBlocks, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Conv2d", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("conv_2d", conv2d_npu);
}

} // namespace cann_bench
