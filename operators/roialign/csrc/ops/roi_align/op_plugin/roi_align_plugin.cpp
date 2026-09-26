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
 * \file roi_align_plugin.cpp
 * \brief ROIAlign API layer - torch bindings (compiled with g++).
 *
 * The host side only derives shapes and scalar tiling parameters; every output element is produced
 * by the custom device kernel.  No host readback and no host tensor arithmetic.
 */

#include <cstdint>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/roi_align_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("roi_align(Tensor x, Tensor boxes, int outputHeight, int outputWidth, "
          "float spatial_scale, int sampling_ratio=-1, bool aligned=False) -> Tensor");
}

torch::Tensor roi_align_meta(const torch::Tensor &x, const torch::Tensor &boxes,
                             int64_t outputHeight, int64_t outputWidth,
                             double spatial_scale, int64_t sampling_ratio, bool aligned)
{
    (void)spatial_scale;
    (void)sampling_ratio;
    (void)aligned;
    TORCH_CHECK(x.dim() == 4, "roi_align: x must be a 4D tensor (B, C, H, W).");
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 5,
                "roi_align: boxes must be a (numBoxes, 5) tensor.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
                "roi_align: only float32 / float16 inputs are supported.");
    TORCH_CHECK(outputHeight > 0 && outputWidth > 0,
                "roi_align: outputHeight / outputWidth must be positive.");
    return torch::empty({boxes.size(0), x.size(1), outputHeight, outputWidth}, x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("roi_align", roi_align_meta);
}

torch::Tensor roi_align_npu(const torch::Tensor &x, const torch::Tensor &boxes,
                            int64_t outputHeight, int64_t outputWidth,
                            double spatial_scale, int64_t sampling_ratio, bool aligned)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto bc = boxes.contiguous();
    auto y = roi_align_meta(xc, bc, outputHeight, outputWidth, spatial_scale, sampling_ratio, aligned);

    const int64_t B = xc.size(0);
    const int64_t C = xc.size(1);
    const int64_t H = xc.size(2);
    const int64_t W = xc.size(3);
    const int64_t N = bc.size(0);
    if (N <= 0 || B <= 0 || C <= 0 || H <= 0 || W <= 0) {
        return y;
    }

    const bool isF32 = (xc.scalar_type() == torch::kFloat32);
    const int64_t esz = isF32 ? 4 : 2;

    const RoiAlignTiling t = calc_roi_align_tiling(B, C, H, W, N, outputHeight, outputWidth, esz,
                                                   sampling_ratio);
    if (t.numUnits <= 0 || t.CbP <= 0 || t.OWp <= 0 || t.numBlocks <= 0) {
        return y;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = (GM_ADDR)xc.data_ptr();
    auto bPtr = (GM_ADDR)bc.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();
    const float ssF = static_cast<float>(spatial_scale);
    const int64_t srI = sampling_ratio;
    const int64_t alI = aligned ? 1 : 0;

    auto acl_call = [=]() -> int {
        if (isF32) {
            launch_roi_align_float(xPtr, bPtr, yPtr, B, C, H, W, N,
                                   outputHeight, outputWidth, ssF, srI, alI,
                                   t.CbP, t.nCb, t.OWp, t.OP, t.perm, t.pitchMax, t.gwCap,
                                   t.numUnits, t.unitsPerCore, t.numBlocks, stream);
        } else {
            launch_roi_align_half(xPtr, bPtr, yPtr, B, C, H, W, N,
                                  outputHeight, outputWidth, ssF, srI, alI,
                                  t.CbP, t.nCb, t.OWp, t.OP, t.perm, t.pitchMax, t.gwCap,
                                  t.numUnits, t.unitsPerCore, t.numBlocks, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("RoiAlign", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("roi_align", roi_align_npu);
}

}  // namespace cann_bench
