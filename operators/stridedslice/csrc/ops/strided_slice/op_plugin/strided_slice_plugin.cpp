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
 * \file strided_slice_plugin.cpp
 * \brief StridedSlice API layer - torch bindings + request reduction (compiled with g++).
 *
 * Mirrors task/reference.py exactly: it walks begin/end/strides together with the five masks and
 * builds the index specs that the reference feeds to x[tuple(indices)].  From those it derives the
 * output shape, the element offset of output[0] inside x, and the "effective" dimensions (output
 * length > 1) with their element steps, which is all the kernel needs.
 *
 * No host side readback, no tensor method that computes anything, no layout/dtype conversion.
 *
 * NOTE on the registration: the Meta and PrivateUse1 kernels must take the SAME C++ argument types
 * (`std::vector<int64_t>` for the int[] attributes).  Declaring the Meta one with
 * `const std::vector<int64_t>&` made the shared library abort on import (every correctness run then
 * returned exit_code 134 with zero canonical reports), while the same body with matching signature
 * styles loads and runs.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <algorithm>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/strided_slice_launch.h"

namespace cann_bench {

static constexpr int SS_NEW = 0;
static constexpr int SS_SLICE = 1;
static constexpr int SS_SHRINK = 2;

struct SsIdx {
    int kind;
    int64_t axis;
    int64_t start;
    int64_t step;
    int64_t len;
};

static int64_t SsClamp(int64_t v, int64_t lo, int64_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static int64_t SsBit(const std::vector<int64_t> &v, int64_t idx)
{
    return (idx < static_cast<int64_t>(v.size())) ? v[idx] : 0;
}

/* Build the reference's `indices` list. */
static void SsBuildIndices(const std::vector<int64_t> &shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &end, const std::vector<int64_t> &strides,
                           int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                           int64_t shrink_axis_mask, int64_t new_axis_mask, std::vector<SsIdx> &out)
{
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t nparams = static_cast<int64_t>(begin.size());

    int64_t ellipsis_pos = -1;
    for (int64_t i = 0; i < 32; ++i) {
        if (ellipsis_mask & (static_cast<int64_t>(1) << i)) {
            ellipsis_pos = i;
            break;
        }
    }
    int64_t num_new_axis = 0;
    for (int64_t i = 0; i < nparams; ++i) {
        if (new_axis_mask & (static_cast<int64_t>(1) << i)) {
            ++num_new_axis;
        }
    }
    int64_t num_ellipsis_dims = 0;
    if (ellipsis_pos >= 0) {
        num_ellipsis_dims = ndim - (nparams - num_new_axis - 1);
        if (num_ellipsis_dims < 0) {
            num_ellipsis_dims = 0;
        }
    }

    int64_t input_dim_idx = 0;
    int64_t param_idx = 0;
    while (input_dim_idx < ndim || param_idx < nparams) {
        if (param_idx < nparams && (new_axis_mask & (static_cast<int64_t>(1) << param_idx))) {
            out.push_back({SS_NEW, 0, 0, 0, 1});
            ++param_idx;
            continue;
        }
        if (ellipsis_pos >= 0 && param_idx == ellipsis_pos) {
            for (int64_t k = 0; k < num_ellipsis_dims && input_dim_idx < ndim; ++k) {
                out.push_back({SS_SLICE, input_dim_idx, 0, 1, shape[input_dim_idx]});
                ++input_dim_idx;
            }
            ++param_idx;
            continue;
        }
        if (input_dim_idx < ndim && param_idx < nparams) {
            const int64_t dim_size = shape[input_dim_idx];
            int64_t b = SsBit(begin, param_idx);
            int64_t e = (param_idx < static_cast<int64_t>(end.size())) ? end[param_idx] : dim_size;
            int64_t s = (param_idx < static_cast<int64_t>(strides.size())) ? strides[param_idx] : 1;
            if (s == 0) {
                s = 1;
            }
            if (b < 0) {
                b += dim_size;
            }
            if (e < 0) {
                e += dim_size;
            }
            if (begin_mask & (static_cast<int64_t>(1) << param_idx)) {
                b = (s > 0) ? 0 : dim_size - 1;
            }
            if (end_mask & (static_cast<int64_t>(1) << param_idx)) {
                e = (s > 0) ? dim_size : -1;
            }
            if (shrink_axis_mask & (static_cast<int64_t>(1) << param_idx)) {
                int64_t ii = b;
                if (ii < 0) {
                    ii += dim_size;
                }
                ii = SsClamp(ii, 0, dim_size > 0 ? dim_size - 1 : 0);
                out.push_back({SS_SHRINK, input_dim_idx, ii, 0, 1});
            } else {
                int64_t start = 0;
                int64_t len = 0;
                if (s > 0) {
                    const int64_t bb = SsClamp(b, 0, dim_size);
                    const int64_t ee = SsClamp(e, 0, dim_size);
                    start = bb;
                    len = (ee > bb) ? (ee - bb + s - 1) / s : 0;
                } else {
                    const int64_t bb = SsClamp(b, -1, dim_size - 1);
                    const int64_t ee = SsClamp(e, -1, dim_size - 1);
                    start = bb;
                    len = (bb > ee) ? (bb - ee + (-s) - 1) / (-s) : 0;
                }
                out.push_back({SS_SLICE, input_dim_idx, start, s, len});
            }
            ++input_dim_idx;
            ++param_idx;
        } else if (input_dim_idx < ndim) {
            out.push_back({SS_SLICE, input_dim_idx, 0, 1, shape[input_dim_idx]});
            ++input_dim_idx;
        } else {
            if (param_idx < nparams && (new_axis_mask & (static_cast<int64_t>(1) << param_idx))) {
                out.push_back({SS_NEW, 0, 0, 0, 1});
            }
            ++param_idx;
        }
    }
}

struct SsPlan {
    std::vector<int64_t> outShape;
    int64_t outNumel = 0;
    int64_t baseOffset = 0;
    std::vector<int64_t> effSize;
    std::vector<int64_t> effStep;
};

static SsPlan SsBuildPlan(const torch::Tensor &x, const std::vector<int64_t> &begin,
                          const std::vector<int64_t> &end, const std::vector<int64_t> &strides,
                          int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                          int64_t shrink_axis_mask, int64_t new_axis_mask)
{
    const int64_t ndim = x.dim();
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(std::max<int64_t>(ndim, 0)));
    for (int64_t i = 0; i < ndim; ++i) {
        shape.push_back(x.size(i));
    }
    std::vector<int64_t> inStride(static_cast<size_t>(std::max<int64_t>(ndim, 0)), 1);
    for (int64_t i = ndim - 2; i >= 0; --i) {
        inStride[i] = inStride[i + 1] * shape[i + 1];
    }

    std::vector<SsIdx> idx;
    SsBuildIndices(shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, shrink_axis_mask,
                   new_axis_mask, idx);

    SsPlan p;
    p.baseOffset = 0;
    int64_t numel = 1;
    for (size_t i = 0; i < idx.size(); ++i) {
        const SsIdx &ix = idx[i];
        if (ix.kind == SS_SHRINK) {
            p.baseOffset += ix.start * inStride[ix.axis];
        } else if (ix.kind == SS_NEW) {
            p.outShape.push_back(1);
        } else {
            p.outShape.push_back(ix.len);
            numel *= ix.len;
            p.baseOffset += ix.start * inStride[ix.axis];
            if (ix.len > 1) {
                p.effSize.push_back(ix.len);
                p.effStep.push_back(ix.step * inStride[ix.axis]);
            }
        }
    }
    p.outNumel = numel;
    return p;
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("strided_slice(Tensor x, int[] begin, int[] end, int[] strides, int begin_mask, int end_mask, "
          "int ellipsis_mask, int shrink_axis_mask, int new_axis_mask) -> Tensor");
}

static torch::Tensor SsMeta(const torch::Tensor &x, std::vector<int64_t> begin, std::vector<int64_t> end,
                            std::vector<int64_t> strides, int64_t begin_mask, int64_t end_mask,
                            int64_t ellipsis_mask, int64_t shrink_axis_mask, int64_t new_axis_mask)
{
    SsPlan p = SsBuildPlan(x, begin, end, strides, begin_mask, end_mask, ellipsis_mask, shrink_axis_mask,
                           new_axis_mask);
    return torch::empty(p.outShape, x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("strided_slice", SsMeta);
}

static torch::Tensor strided_slice_npu(const torch::Tensor &x, std::vector<int64_t> begin,
                                       std::vector<int64_t> end, std::vector<int64_t> strides,
                                       int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask,
                                       int64_t shrink_axis_mask, int64_t new_axis_mask)
{
    const c10::OptionalDeviceGuard guard(x.device());
    TORCH_CHECK(x.dim() <= 8, "strided_slice: only tensors with at most 8 dims are supported.");
    TORCH_CHECK(x.is_contiguous(), "strided_slice: x must be contiguous.");

    SsPlan p = SsBuildPlan(x, begin, end, strides, begin_mask, end_mask, ellipsis_mask, shrink_axis_mask,
                           new_axis_mask);
    auto y = torch::empty(p.outShape, x.options());
    if (p.outNumel <= 0) {
        return y;
    }

    const int64_t elemBytes = static_cast<int64_t>(x.element_size());
    const int64_t rank = static_cast<int64_t>(p.effSize.size());
    StridedSliceTiling t = calc_strided_slice_tiling(p.baseOffset, p.outNumel, rank, p.effSize.data(),
                                                     p.effStep.data(), elemBytes);
    if (t.outNumel <= 0 || t.numUnits <= 0 || t.numBlocks <= 0) {
        return y;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    auto xPtr = reinterpret_cast<GM_ADDR>(x.data_ptr());
    auto yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());
    const auto dtype = x.scalar_type();

    auto aclCall = [=]() -> int {
        if (dtype == torch::kFloat32) {
            launch_strided_slice_f32(xPtr, yPtr, t, stream);
        } else if (dtype == torch::kFloat16) {
            launch_strided_slice_f16(xPtr, yPtr, t, stream);
        } else if (dtype == torch::kBFloat16) {
            launch_strided_slice_bf16(xPtr, yPtr, t, stream);
        } else if (dtype == torch::kInt32) {
            launch_strided_slice_i32(xPtr, yPtr, t, stream);
        } else if (dtype == torch::kInt64) {
            launch_strided_slice_i64(xPtr, yPtr, t, stream);
        } else if (dtype == torch::kChar) {
            launch_strided_slice_i8(xPtr, yPtr, t, stream);
        } else if (dtype == torch::kByte) {
            launch_strided_slice_u8(xPtr, yPtr, t, stream);
        } else {
            TORCH_CHECK(false, "strided_slice: unsupported dtype");
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("StridedSlice", aclCall);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("strided_slice", strided_slice_npu);
}

}  // namespace cann_bench
