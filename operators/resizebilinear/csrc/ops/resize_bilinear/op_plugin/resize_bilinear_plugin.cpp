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
 * \file resize_bilinear_plugin.cpp
 * \brief ResizeBilinear API layer - torch bindings (compiled with g++).
 *
 * Registration contract: the two optional list attributes are declared as
 * `c10::optional<c10::List<T>>`, which infers to the `T[]?` schema the operator
 * contract requires.  A `c10::OptionalArrayRef` signature has no inferrable
 * schema in this torch build and aborts the host process while the library is
 * being loaded.
 *
 * The host side only derives shapes and the (scalar) coordinate mapping
 * constants.  All tensor arithmetic happens inside the device kernel.
 */

#include <cmath>
#include <cstdint>
#include <tuple>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/resize_bilinear_launch.h"

namespace cann_bench {

namespace {

using RbIntList = c10::optional<c10::List<int64_t>>;
using RbFloatList = c10::optional<c10::List<double>>;

struct RbShape {
    int64_t n = 0;
    int64_t c = 0;
    int64_t hIn = 0;
    int64_t wIn = 0;
    int64_t hOut = 0;
    int64_t wOut = 0;
};

/*! Resolve the output spatial size the same way F.interpolate does:
 *  output_size wins, otherwise floor(in_size * scale_factor), otherwise identity. */
RbShape rb_resolve_shape(const torch::Tensor& x, const RbIntList& output_size,
                         const RbFloatList& scale_factor)
{
    TORCH_CHECK(x.dim() == 4, "resize_bilinear: input must be a 4D tensor (N, C, H, W).");
    const auto dtype = x.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                "resize_bilinear: unsupported dtype, expect float32 / float16 / bfloat16.");

    RbShape s;
    s.n = x.size(0);
    s.c = x.size(1);
    s.hIn = x.size(2);
    s.wIn = x.size(3);

    const bool hasOs = output_size.has_value() && output_size->size() >= 2;
    const bool hasSf = scale_factor.has_value() && scale_factor->size() >= 2;
    if (hasOs) {
        s.hOut = (*output_size)[0];
        s.wOut = (*output_size)[1];
    } else if (hasSf) {
        s.hOut = static_cast<int64_t>(std::floor(static_cast<double>(s.hIn) * (*scale_factor)[0]));
        s.wOut = static_cast<int64_t>(std::floor(static_cast<double>(s.wIn) * (*scale_factor)[1]));
    } else {
        s.hOut = s.hIn;
        s.wOut = s.wIn;
    }
    TORCH_CHECK(s.hOut > 0 && s.wOut > 0,
                "resize_bilinear: output spatial size must be positive (got ", s.hOut, " x ",
                s.wOut, ").");
    return s;
}

/*! Coordinate mapping scale, identical to torch's area_pixel_compute_scale /
 *  compute_scales_value:
 *    align_corners == true : scale = (in - 1) / (out - 1)   [0 when out <= 1]
 *    align_corners == false: scale = 1 / scale_factor when usable, else in / out
 *  The cast order (double -> float for 1/scale_factor, float division otherwise)
 *  mirrors the reference exactly. */
float rb_resolve_scale(int64_t inSize, int64_t outSize, bool alignCorners, bool haveScaleFactor,
                       double scaleFactor)
{
    if (alignCorners) {
        return (outSize > 1) ? (static_cast<float>(inSize - 1) / static_cast<float>(outSize - 1))
                             : 0.0f;
    }
    if (haveScaleFactor && scaleFactor > 0.0) {
        return static_cast<float>(1.0 / scaleFactor);
    }
    return static_cast<float>(inSize) / static_cast<float>(outSize);
}

}  // namespace

torch::Tensor resize_bilinear_meta(const torch::Tensor& x, RbIntList output_size,
                                   bool align_corners, RbFloatList scale_factor)
{
    (void)align_corners;
    const RbShape s = rb_resolve_shape(x, output_size, scale_factor);
    return torch::empty({s.n, s.c, s.hOut, s.wOut}, x.options());
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("resize_bilinear(Tensor x, int[]? output_size=None, bool align_corners=False, "
          "float[]? scale_factor=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("resize_bilinear", resize_bilinear_meta);
}

torch::Tensor resize_bilinear_npu(const torch::Tensor& x, RbIntList output_size,
                                  bool align_corners, RbFloatList scale_factor)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.is_contiguous() ? x : x.contiguous();
    const RbShape s = rb_resolve_shape(xc, output_size, scale_factor);
    auto y = torch::empty({s.n, s.c, s.hOut, s.wOut}, xc.options());

    const int64_t totalRows = s.n * s.c * s.hOut;
    TORCH_CHECK(totalRows > 0, "resize_bilinear: empty output.");

    const bool hasOs = output_size.has_value() && output_size->size() >= 2;
    const bool hasSf = scale_factor.has_value() && scale_factor->size() >= 2;
    const double sfH = hasSf ? (*scale_factor)[0] : 0.0;
    const double sfW = hasSf ? (*scale_factor)[1] : 0.0;
    const bool useSfH = (!hasOs) && sfH > 0.0;
    const bool useSfW = (!hasOs) && sfW > 0.0;

    const float scaleH = rb_resolve_scale(s.hIn, s.hOut, align_corners, useSfH, sfH);
    const float scaleW = rb_resolve_scale(s.wIn, s.wOut, align_corners, useSfW, sfW);
    const int32_t alignI = align_corners ? 1 : 0;

    int64_t numBlocks = 1;
    int64_t rowsPerCore = 1;
    std::tie(numBlocks, rowsPerCore) = calc_resize_bilinear_tiling(totalRows);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x_ptr = (GM_ADDR)xc.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    const int64_t hIn = s.hIn;
    const int64_t wIn = s.wIn;
    const int64_t hOut = s.hOut;
    const int64_t wOut = s.wOut;
    const auto dtype = xc.scalar_type();

    auto acl_call = [=]() -> int {
        if (dtype == torch::kFloat32) {
            launch_resize_bilinear_float(x_ptr, y_ptr, totalRows, numBlocks, rowsPerCore, hIn, wIn,
                                         hOut, wOut, scaleH, scaleW, alignI, stream);
        } else if (dtype == torch::kFloat16) {
            launch_resize_bilinear_half(x_ptr, y_ptr, totalRows, numBlocks, rowsPerCore, hIn, wIn,
                                        hOut, wOut, scaleH, scaleW, alignI, stream);
        } else {
            launch_resize_bilinear_bfloat16(x_ptr, y_ptr, totalRows, numBlocks, rowsPerCore, hIn,
                                            wIn, hOut, wOut, scaleH, scaleW, alignI, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ResizeBilinear", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("resize_bilinear", resize_bilinear_npu);
}

}  // namespace cann_bench
