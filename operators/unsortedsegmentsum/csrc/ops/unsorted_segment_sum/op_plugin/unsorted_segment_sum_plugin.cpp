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
 * \file unsorted_segment_sum_plugin.cpp
 * \brief UnsortedSegmentSum API layer - torch bindings (compiled with g++)
 *
 * The host side only performs integer tiling arithmetic and workspace allocation; every element of
 * arithmetic, the zero fill of empty segments and every conversion happens inside the custom device
 * kernels placed on the current NPU stream.
 */

#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/uss_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("unsorted_segment_sum(Tensor data, Tensor segment_ids, int num_segments) -> Tensor");
}

static bool UssDataSupported(const torch::Tensor &t)
{
    const auto dt = t.scalar_type();
    return dt == torch::kFloat16 || dt == torch::kBFloat16 || dt == torch::kFloat32 ||
           dt == torch::kInt32 || dt == torch::kInt64;
}

torch::Tensor unsorted_segment_sum_meta(const torch::Tensor &data, const torch::Tensor &segment_ids,
                                        int64_t num_segments)
{
    TORCH_CHECK(data.dim() >= 1, "cann_bench.unsorted_segment_sum: data must have rank >= 1.");
    TORCH_CHECK(UssDataSupported(data), "cann_bench.unsorted_segment_sum: unsupported data dtype.");
    TORCH_CHECK(segment_ids.scalar_type() == torch::kInt32 || segment_ids.scalar_type() == torch::kInt64,
                "cann_bench.unsorted_segment_sum: segment_ids must be int32 or int64.");
    TORCH_CHECK(segment_ids.dim() == 1, "cann_bench.unsorted_segment_sum: segment_ids must be 1-D.");
    TORCH_CHECK(segment_ids.size(0) == data.size(0),
                "cann_bench.unsorted_segment_sum: segment_ids length must equal data.shape[0].");
    TORCH_CHECK(num_segments >= 0, "cann_bench.unsorted_segment_sum: num_segments must be non-negative.");

    std::vector<int64_t> outShape;
    outShape.push_back(num_segments);
    for (int64_t i = 1; i < data.dim(); ++i) {
        outShape.push_back(data.size(i));
    }
    return torch::empty(outShape, data.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("unsorted_segment_sum", unsorted_segment_sum_meta);
}

#define USS_OWNER_CALL(TAG)                                                                        \
    launch_uss_owner_##TAG(xPtr, idPtr, yPtr, t.N, t.inner, t.numSeg, t.chunkElems, t.nChunks,      \
                           t.unitsPerCore, t.segArrMax, t.idTile, t.cap, stream)

#define USS_PART_CALL(TAG)                                                                         \
    launch_uss_part_##TAG(xPtr, idPtr, wsPtr, t.N, t.numSeg, t.numBlocks, t.rowsPerCore, t.rowTile, \
                          stream)

#define USS_RED_CALL(TAG) launch_uss_red_##TAG(wsPtr, yPtr, t.numSeg, t.numBlocks, t.redBlocks, stream)

torch::Tensor unsorted_segment_sum_npu(const torch::Tensor &data, const torch::Tensor &segment_ids,
                                       int64_t num_segments)
{
    const c10::OptionalDeviceGuard guard(data.device());
    auto x = data.contiguous();
    auto ids = segment_ids.contiguous();
    auto y = unsorted_segment_sum_meta(x, ids, num_segments);

    const int64_t N = x.size(0);
    int64_t inner = 1;
    for (int64_t i = 1; i < x.dim(); ++i) {
        inner *= x.size(i);
    }
    if (inner <= 0 || num_segments <= 0) {
        return y;
    }

    const auto dType = x.scalar_type();
    const auto iType = ids.scalar_type();
    const bool intAcc = (dType == torch::kInt32 || dType == torch::kInt64);
    const int64_t szT = static_cast<int64_t>(x.element_size());
    const int64_t szId = static_cast<int64_t>(ids.element_size());
    const int64_t szA = 4;

    const UssTiling t = calc_uss_tiling(N, inner, num_segments, szT, szId, szA);

    torch::Tensor ws;
    if (t.mode == 1) {
        ws = torch::empty({t.wsElems}, x.options().dtype(intAcc ? torch::kInt32 : torch::kFloat32));
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    void *xPtr = x.data_ptr();
    void *idPtr = ids.data_ptr();
    void *yPtr = y.data_ptr();
    void *wsPtr = (t.mode == 1) ? ws.data_ptr() : nullptr;

    auto acl_call = [=]() -> int {
        if (t.mode == 1) {
            if (dType == torch::kFloat16) {
                if (iType == torch::kInt32) {
                    USS_PART_CALL(f16_i32);
                } else {
                    USS_PART_CALL(f16_i64);
                }
                USS_RED_CALL(f32_f16);
            } else if (dType == torch::kBFloat16) {
                if (iType == torch::kInt32) {
                    USS_PART_CALL(bf16_i32);
                } else {
                    USS_PART_CALL(bf16_i64);
                }
                USS_RED_CALL(f32_bf16);
            } else if (dType == torch::kFloat32) {
                if (iType == torch::kInt32) {
                    USS_PART_CALL(f32_i32);
                } else {
                    USS_PART_CALL(f32_i64);
                }
                USS_RED_CALL(f32_f32);
            } else if (dType == torch::kInt32) {
                if (iType == torch::kInt32) {
                    USS_PART_CALL(i32_i32);
                } else {
                    USS_PART_CALL(i32_i64);
                }
                USS_RED_CALL(i32_i32);
            } else {
                if (iType == torch::kInt32) {
                    USS_PART_CALL(i64_i32);
                } else {
                    USS_PART_CALL(i64_i64);
                }
                USS_RED_CALL(i32_i64);
            }
        } else {
            if (dType == torch::kFloat16) {
                if (iType == torch::kInt32) {
                    USS_OWNER_CALL(f16_i32);
                } else {
                    USS_OWNER_CALL(f16_i64);
                }
            } else if (dType == torch::kBFloat16) {
                if (iType == torch::kInt32) {
                    USS_OWNER_CALL(bf16_i32);
                } else {
                    USS_OWNER_CALL(bf16_i64);
                }
            } else if (dType == torch::kFloat32) {
                if (iType == torch::kInt32) {
                    USS_OWNER_CALL(f32_i32);
                } else {
                    USS_OWNER_CALL(f32_i64);
                }
            } else if (dType == torch::kInt32) {
                if (iType == torch::kInt32) {
                    USS_OWNER_CALL(i32_i32);
                } else {
                    USS_OWNER_CALL(i32_i64);
                }
            } else {
                if (iType == torch::kInt32) {
                    USS_OWNER_CALL(i64_i32);
                } else {
                    USS_OWNER_CALL(i64_i64);
                }
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("UnsortedSegmentSum", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("unsorted_segment_sum", unsorted_segment_sum_npu);
}

} // namespace cann_bench
