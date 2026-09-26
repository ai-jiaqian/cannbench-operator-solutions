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
 * \file arg_max_plugin.cpp
 * \brief ArgMax API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/arg_max_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("arg_max(Tensor input, int dim, bool keepdim=False) -> Tensor");
}

static int32_t ArgMaxDtypeCode(const torch::Tensor &t)
{
    switch (t.scalar_type()) {
        case torch::kFloat16:
            return AM_HALF;
        case torch::kFloat32:
            return AM_FLOAT;
        case torch::kBFloat16:
            return AM_BF16;
        case torch::kInt32:
            return AM_INT32;
        case torch::kInt64:
            return AM_INT64;
        default:
            return -1;
    }
}

static std::vector<int64_t> ArgMaxOutShape(const torch::Tensor &x, int64_t dim, bool keepdim)
{
    int64_t rank = x.dim();
    std::vector<int64_t> shape;
    if (rank == 0) {
        if (keepdim) {
            shape.push_back(1);
        }
        return shape;
    }
    int64_t d = dim < 0 ? dim + rank : dim;
    for (int64_t i = 0; i < rank; ++i) {
        if (i == d) {
            if (keepdim) {
                shape.push_back(1);
            }
        } else {
            shape.push_back(x.size(i));
        }
    }
    return shape;
}

torch::Tensor arg_max_meta(const torch::Tensor &x, int64_t dim, bool keepdim)
{
    TORCH_CHECK(ArgMaxDtypeCode(x) >= 0, "cann_bench.arg_max: unsupported input dtype ", x.scalar_type());
    if (x.dim() > 0) {
        TORCH_CHECK(dim >= -x.dim() && dim < x.dim(), "cann_bench.arg_max: dim out of range");
    }
    return torch::empty(ArgMaxOutShape(x, dim, keepdim), x.options().dtype(torch::kInt64));
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("arg_max", arg_max_meta);
}

torch::Tensor arg_max_npu(const torch::Tensor &x, int64_t dim, bool keepdim)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto out = arg_max_meta(xc, dim, keepdim);

    int64_t rank = xc.dim();
    int64_t outer = 1;
    int64_t inner = 1;
    int64_t reduceLen = 1;
    if (rank > 0) {
        int64_t d = dim < 0 ? dim + rank : dim;
        for (int64_t i = 0; i < d; ++i) {
            outer *= xc.size(i);
        }
        reduceLen = xc.size(d);
        for (int64_t i = d + 1; i < rank; ++i) {
            inner *= xc.size(i);
        }
    }
    TORCH_CHECK(reduceLen >= 1, "cann_bench.arg_max: reduction dimension must be non-empty");

    int32_t code = ArgMaxDtypeCode(xc);
    ArgMaxPlan plan = calc_argmax_plan(outer, reduceLen, inner, code);

    torch::Tensor ws;
    void *wsPtr = nullptr;
    if (plan.wsFloats > 0) {
        ws = torch::empty({plan.wsFloats}, xc.options().dtype(torch::kFloat));
        wsPtr = ws.data_ptr();
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = (GM_ADDR)xc.data_ptr();
    auto yPtr = (GM_ADDR)out.data_ptr();
    auto wsRaw = (GM_ADDR)wsPtr;

    ArgMaxPlan localPlan = plan;
    auto acl_call = [=]() -> int {
        if (localPlan.mode == 0) {
            if (localPlan.nChunk == 1) {
                switch (code) {
                    case AM_HALF:
                        launch_argmax_row_direct_f16(xPtr, yPtr, &localPlan, stream);
                        break;
                    case AM_FLOAT:
                        launch_argmax_row_direct_f32(xPtr, yPtr, &localPlan, stream);
                        break;
                    case AM_BF16:
                        launch_argmax_row_direct_bf16(xPtr, yPtr, &localPlan, stream);
                        break;
                    case AM_INT32:
                        launch_argmax_row_direct_i32(xPtr, yPtr, &localPlan, stream);
                        break;
                    default:
                        launch_argmax_row_direct_i64(xPtr, yPtr, &localPlan, stream);
                        break;
                }
            } else {
                switch (code) {
                    case AM_HALF:
                        launch_argmax_row_partial_f16(xPtr, wsRaw, &localPlan, stream);
                        break;
                    case AM_FLOAT:
                        launch_argmax_row_partial_f32(xPtr, wsRaw, &localPlan, stream);
                        break;
                    case AM_BF16:
                        launch_argmax_row_partial_bf16(xPtr, wsRaw, &localPlan, stream);
                        break;
                    case AM_INT32:
                        launch_argmax_row_partial_i32(xPtr, wsRaw, &localPlan, stream);
                        break;
                    default:
                        launch_argmax_row_partial_i64(xPtr, wsRaw, &localPlan, stream);
                        break;
                }
                launch_argmax_row_combine(wsRaw, yPtr, &localPlan, stream);
            }
        } else {
            switch (code) {
                case AM_HALF:
                    launch_argmax_col_partial_f16(xPtr, wsRaw, yPtr, &localPlan, stream);
                    break;
                case AM_FLOAT:
                    launch_argmax_col_partial_f32(xPtr, wsRaw, yPtr, &localPlan, stream);
                    break;
                case AM_BF16:
                    launch_argmax_col_partial_bf16(xPtr, wsRaw, yPtr, &localPlan, stream);
                    break;
                case AM_INT32:
                    launch_argmax_col_partial_i32(xPtr, wsRaw, yPtr, &localPlan, stream);
                    break;
                default:
                    launch_argmax_col_partial_i64(xPtr, wsRaw, yPtr, &localPlan, stream);
                    break;
            }
            if (localPlan.directCol == 0) {
                launch_argmax_col_combine(wsRaw, yPtr, &localPlan, stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ArgMax", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("arg_max", arg_max_npu);
}

} // namespace cann_bench
