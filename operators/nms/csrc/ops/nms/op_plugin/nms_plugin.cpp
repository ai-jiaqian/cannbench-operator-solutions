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
 * \file nms_plugin.cpp
 * \brief NMS API layer - torch bindings (compiled with g++)
 */

#include <algorithm>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/nms_launch.h"

namespace cann_bench {

constexpr int64_t NMS_PAD = 32;

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor");
}

static int64_t nms_pad(int64_t n)
{
    return ((n + NMS_PAD - 1) / NMS_PAD) * NMS_PAD;
}

torch::Tensor nms_meta(const torch::Tensor &boxes, const torch::Tensor &scores,
                       double iouThreshold)
{
    (void)iouThreshold;
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "nms: boxes must be [N, 4].");
    TORCH_CHECK(scores.dim() == 1 && scores.size(0) == boxes.size(0),
                "nms: scores must be [N] with the same N as boxes.");
    /* keep_indices has shape [M] with M <= N, but M is a data-dependent result of the greedy loop.
     * Reading it back to the host would require a device->host transfer, which the on-device
     * contract forbids, so the operator publishes the tightest bound it can derive from the inputs:
     * a length-N tensor whose first M entries are the kept original indices (M is produced entirely
     * on device, the trailing entries are zero-filled). */
    return torch::empty({boxes.size(0)}, boxes.options().dtype(torch::kLong));
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("nms", nms_meta);
}

torch::Tensor nms_npu(const torch::Tensor &boxesIn, const torch::Tensor &scoresIn,
                      double iouThreshold)
{
    const c10::OptionalDeviceGuard guard(boxesIn.device());

    auto boxes = boxesIn.is_contiguous() ? boxesIn : boxesIn.contiguous();
    auto scores = scoresIn.is_contiguous() ? scoresIn : scoresIn.contiguous();

    auto out = nms_meta(boxes, scores, iouThreshold);
    const int64_t n = boxes.size(0);
    if (n <= 0) {
        return out;
    }
    const int64_t nPad = nms_pad(n);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    /* scratch: x1 | y1 | x2 | y2 packed fp32 arrays, each nPad long */
    auto scratch = torch::empty({4 * nPad}, boxes.options().dtype(torch::kFloat32));
    auto base = reinterpret_cast<char *>(scratch.data_ptr());
    const int64_t stride = nPad * static_cast<int64_t>(sizeof(float));
    GM_ADDR x1Ptr = reinterpret_cast<GM_ADDR>(base);
    GM_ADDR y1Ptr = reinterpret_cast<GM_ADDR>(base + stride);
    GM_ADDR x2Ptr = reinterpret_cast<GM_ADDR>(base + 2 * stride);
    GM_ADDR y2Ptr = reinterpret_cast<GM_ADDR>(base + 3 * stride);

    auto boxesPtr = reinterpret_cast<GM_ADDR>(boxes.data_ptr());
    auto scoresPtr = reinterpret_cast<GM_ADDR>(scores.data_ptr());
    auto outPtr = reinterpret_cast<GM_ADDR>(out.data_ptr());
    const float thr = static_cast<float>(iouThreshold);

    const int64_t layoutBlocks = std::min<int64_t>(48, std::max<int64_t>(1, (n + 255) / 256));

    auto acl_call = [=]() -> int {
        launch_nms_layout_kernel(boxesPtr, x1Ptr, y1Ptr, x2Ptr, y2Ptr, n, layoutBlocks, stream);
        launch_nms_apply_kernel(x1Ptr, y1Ptr, x2Ptr, y2Ptr, scoresPtr, outPtr, n, nPad, thr,
                                stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Nms", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("nms", nms_npu);
}

}  // namespace cann_bench
