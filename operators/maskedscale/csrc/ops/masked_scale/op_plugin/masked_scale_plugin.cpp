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
 * \file masked_scale_plugin.cpp
 * \brief MaskedScale API layer - torch bindings (compiled with g++).
 */

#include <tuple>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/masked_scale_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("masked_scale(Tensor x, Tensor mask, float scale=1.0) -> Tensor");
}

torch::Tensor masked_scale_meta(const torch::Tensor &x, const torch::Tensor &mask, double scale)
{
    (void)scale;
    TORCH_CHECK(x.sizes() == mask.sizes(), "The shapes of x and mask must be the same.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("masked_scale", masked_scale_meta);
}

torch::Tensor masked_scale_npu(const torch::Tensor &x, const torch::Tensor &mask, double scale)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto mc = mask.contiguous();
    auto y = masked_scale_meta(xc, mc, scale);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    const int64_t totalLength = xc.numel();
    const auto xst = xc.scalar_type();
    const auto mst = mc.scalar_type();

    TORCH_CHECK(xst == torch::kFloat16 || xst == torch::kBFloat16 || xst == torch::kFloat32,
                "cann_bench.masked_scale: unsupported x dtype.");
    TORCH_CHECK(mst == torch::kChar || mst == torch::kByte || mst == torch::kFloat16 ||
                mst == torch::kBFloat16 || mst == torch::kFloat32,
                "cann_bench.masked_scale: unsupported mask dtype.");

    int64_t xBytes = (xst == torch::kFloat32) ? 4 : 2;
    int64_t maskBytes = 1;
    if (mst == torch::kFloat32) {
        maskBytes = 4;
    } else if (mst == torch::kFloat16 || mst == torch::kBFloat16) {
        maskBytes = 2;
    }

    int64_t numBlocks = 1, blockLength = 1, tileElems = 1;
    std::tie(numBlocks, blockLength, tileElems) =
        calc_masked_scale_tiling_params(totalLength, xBytes, maskBytes);
    const float scaleF = static_cast<float>(scale);
    const uint32_t tile = static_cast<uint32_t>(tileElems);

    auto x_ptr = (GM_ADDR)xc.data_ptr();
    auto m_ptr = (GM_ADDR)mc.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
#define MS_DISPATCH(XTORCH, MTORCH, LAUNCHFN)                                                        \
    if (xst == torch::k##XTORCH && mst == torch::k##MTORCH) {                                        \
        LAUNCHFN(x_ptr, m_ptr, y_ptr, totalLength, numBlocks, blockLength, tile, scaleF, stream);     \
        return 0;                                                                                     \
    }
        MS_DISPATCH(Half, Char, launch_masked_scale_half_int8)
        MS_DISPATCH(Half, Byte, launch_masked_scale_half_uint8)
        MS_DISPATCH(Half, Half, launch_masked_scale_half_half)
        MS_DISPATCH(Half, BFloat16, launch_masked_scale_half_bfloat16)
        MS_DISPATCH(Half, Float, launch_masked_scale_half_float)

        MS_DISPATCH(BFloat16, Char, launch_masked_scale_bfloat16_int8)
        MS_DISPATCH(BFloat16, Byte, launch_masked_scale_bfloat16_uint8)
        MS_DISPATCH(BFloat16, Half, launch_masked_scale_bfloat16_half)
        MS_DISPATCH(BFloat16, BFloat16, launch_masked_scale_bfloat16_bfloat16)
        MS_DISPATCH(BFloat16, Float, launch_masked_scale_bfloat16_float)

        MS_DISPATCH(Float, Char, launch_masked_scale_float_int8)
        MS_DISPATCH(Float, Byte, launch_masked_scale_float_uint8)
        MS_DISPATCH(Float, Half, launch_masked_scale_float_half)
        MS_DISPATCH(Float, BFloat16, launch_masked_scale_float_bfloat16)
        MS_DISPATCH(Float, Float, launch_masked_scale_float_float)
#undef MS_DISPATCH
        TORCH_CHECK(false, "cann_bench.masked_scale: unsupported dtype pair.");
    };
    at_npu::native::OpCommand::RunOpApi("MaskedScale", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("masked_scale", masked_scale_npu);
}

} // namespace cann_bench
