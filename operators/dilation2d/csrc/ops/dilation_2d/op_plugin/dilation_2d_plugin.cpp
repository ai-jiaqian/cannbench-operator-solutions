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
 * \file dilation_2d_plugin.cpp
 * \brief Dilation2D API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <string>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/dilation_2d_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("dilation_2d(Tensor x, Tensor filter, int[] strides, int[] rates, str padding_mode, "
          "int[] pads, bool ceil_mode, str data_format) -> Tensor");
}

namespace {

struct D2Shape {
    int64_t n;
    int64_t h;
    int64_t w;
    int64_t c;
    int64_t fh;
    int64_t fw;
    int64_t sH;
    int64_t sW;
    int64_t rH;
    int64_t rW;
    int64_t outH;
    int64_t outW;
    int64_t padTop;
    int64_t padLeft;
};

int64_t D2ListAt(const c10::List<int64_t> &l, int64_t idx, int64_t defVal)
{
    if (idx < static_cast<int64_t>(l.size())) {
        return l[idx];
    }
    return defVal;
}

// Mirrors the reference implementation exactly (task/reference.py).
D2Shape D2ComputeShape(const torch::Tensor &x, const torch::Tensor &filter,
                       const c10::List<int64_t> &strides, const c10::List<int64_t> &rates,
                       const std::string &padding_mode, const c10::List<int64_t> &pads,
                       bool ceil_mode)
{
    D2Shape s;
    s.n = x.size(0);
    s.h = x.size(1);
    s.w = x.size(2);
    s.c = x.size(3);
    s.fh = filter.size(0);
    s.fw = filter.size(1);
    s.sH = D2ListAt(strides, 1, 1);
    s.sW = D2ListAt(strides, 2, 1);
    s.rH = D2ListAt(rates, 1, 1);
    s.rW = D2ListAt(rates, 2, 1);
    if (s.sH < 1) {
        s.sH = 1;
    }
    if (s.sW < 1) {
        s.sW = 1;
    }
    if (s.rH < 1) {
        s.rH = 1;
    }
    if (s.rW < 1) {
        s.rW = 1;
    }

    const int64_t effH = (s.fh - 1) * s.rH + 1;
    const int64_t effW = (s.fw - 1) * s.rW + 1;
    const int64_t pt = D2ListAt(pads, 0, 0);
    const int64_t pb = D2ListAt(pads, 1, 0);
    const int64_t pl = D2ListAt(pads, 2, 0);
    const int64_t pr = D2ListAt(pads, 3, 0);

    if (padding_mode == "SAME") {
        s.outH = (s.h + s.sH - 1) / s.sH;
        s.outW = (s.w + s.sW - 1) / s.sW;
        const int64_t padH = std::max((s.outH - 1) * s.sH + effH - s.h, static_cast<int64_t>(0));
        const int64_t padW = std::max((s.outW - 1) * s.sW + effW - s.w, static_cast<int64_t>(0));
        s.padTop = padH / 2;
        s.padLeft = padW / 2;
    } else if (padding_mode == "VALID") {
        s.outH = (s.h - effH + s.sH) / s.sH;
        s.outW = (s.w - effW + s.sW) / s.sW;
        s.padTop = pt;
        s.padLeft = pl;
    } else {
        const int64_t hp = s.h + pt + pb;
        const int64_t wp = s.w + pl + pr;
        s.outH = (hp - effH + s.sH) / s.sH;
        s.outW = (wp - effW + s.sW) / s.sW;
        if (ceil_mode) {
            s.outH = (hp - effH + s.sH - 1) / s.sH + 1;
            s.outW = (wp - effW + s.sW - 1) / s.sW + 1;
        }
        s.padTop = pt;
        s.padLeft = pl;
    }
    return s;
}

void D2Validate(const torch::Tensor &x, const torch::Tensor &filter)
{
    TORCH_CHECK(x.dim() == 4, "dilation_2d: x must be 4D (NHWC).");
    TORCH_CHECK(filter.dim() == 3, "dilation_2d: filter must be 3D [fh, fw, C].");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "dilation_2d: only float16 x is supported.");
    TORCH_CHECK(filter.scalar_type() == at::kHalf, "dilation_2d: only float16 filter is supported.");
    TORCH_CHECK(x.size(3) == filter.size(2),
                "dilation_2d: x's last dim must match filter's last dim.");
}

}  // namespace

torch::Tensor dilation_2d_meta(const torch::Tensor &x, const torch::Tensor &filter,
                               c10::List<int64_t> strides, c10::List<int64_t> rates,
                               std::string padding_mode, c10::List<int64_t> pads, bool ceil_mode,
                               std::string data_format)
{
    (void)data_format;
    D2Validate(x, filter);
    D2Shape s = D2ComputeShape(x, filter, strides, rates, padding_mode, pads, ceil_mode);
    TORCH_CHECK(s.outH > 0 && s.outW > 0, "dilation_2d: computed output spatial size is not positive.");
    return at::empty({s.n, s.outH, s.outW, s.c}, x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("dilation_2d", dilation_2d_meta);
}

torch::Tensor dilation_2d_npu(const torch::Tensor &xIn, const torch::Tensor &filterIn,
                              c10::List<int64_t> strides, c10::List<int64_t> rates,
                              std::string padding_mode, c10::List<int64_t> pads, bool ceil_mode,
                              std::string data_format)
{
    (void)data_format;
    const c10::OptionalDeviceGuard guard(xIn.device());
    D2Validate(xIn, filterIn);

    auto x = xIn.contiguous();
    auto filter = filterIn.contiguous();
    D2Shape s = D2ComputeShape(x, filter, strides, rates, padding_mode, pads, ceil_mode);
    TORCH_CHECK(s.outH > 0 && s.outW > 0, "dilation_2d: computed output spatial size is not positive.");

    auto y = at::empty({s.n, s.outH, s.outW, s.c}, x.options());

    Dilation2dTiling tiling = calc_dilation_2d_tiling(s.n, s.outH, s.outW, s.c, s.fh, s.fw,
                                                      s.sH, s.sW, s.rH, s.rW);
    int64_t numBlocks = (tiling.rowsTotal + tiling.blockRows - 1) / tiling.blockRows;
    if (numBlocks < 1) {
        numBlocks = 1;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = (GM_ADDR)x.data_ptr();
    auto fPtr = (GM_ADDR)filter.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        launch_dilation_2d_kernel_half(xPtr, fPtr, yPtr, s.n, s.h, s.w, s.c, s.outH, s.outW,
                                       s.fh, s.fw, s.sH, s.sW, s.rH, s.rW, s.padTop, s.padLeft,
                                       tiling.rowsTotal, tiling.blockRows, tiling.tw, tiling.pad,
                                       tiling.cacheR, numBlocks, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Dilation2D", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("dilation_2d", dilation_2d_npu);
}

}  // namespace cann_bench
