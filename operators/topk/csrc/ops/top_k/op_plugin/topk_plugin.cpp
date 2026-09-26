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
 * \file topk_plugin.cpp
 * \brief TopK API layer - torch bindings (compiled with g++)
 */

#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/topk_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("top_k(Tensor x, int k, int dim, bool largest=True) -> (Tensor, Tensor)");
}

static std::tuple<torch::Tensor, torch::Tensor> topk_meta(const torch::Tensor &x, int64_t k, int64_t dim,
                                                          bool largest)
{
    (void)largest;
    TORCH_CHECK(x.scalar_type() == torch::kFloat16 || x.scalar_type() == torch::kFloat32 ||
                    x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kChar ||
                    x.scalar_type() == torch::kByte || x.scalar_type() == torch::kInt ||
                    x.scalar_type() == torch::kLong,
                "cann_bench.top_k: unsupported input dtype ", x.scalar_type());
    const int64_t rank = x.dim();
    TORCH_CHECK(rank >= 1 && rank <= 8, "cann_bench.top_k: input must have 1..8 dimensions.");
    int64_t d = dim < 0 ? dim + rank : dim;
    TORCH_CHECK(d >= 0 && d < rank, "cann_bench.top_k: dim out of range.");
    TORCH_CHECK(k >= 1 && k <= x.size(d), "cann_bench.top_k: k out of range.");
    TORCH_CHECK(k <= 2048, "cann_bench.top_k: k must not exceed 2048.");

    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        shape.push_back(i == d ? k : x.size(i));
    }
    auto y = torch::empty(shape, x.options());
    auto idx = torch::empty(shape, x.options().dtype(torch::kLong));
    return std::make_tuple(y, idx);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("top_k", topk_meta);
}

static std::tuple<torch::Tensor, torch::Tensor> topk_npu(const torch::Tensor &x, int64_t k, int64_t dim,
                                                         bool largest)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    auto outs = topk_meta(xc, k, dim, largest);
    auto y = std::get<0>(outs);
    auto idx = std::get<1>(outs);
    if (xc.numel() == 0) {
        return outs;
    }

    const int64_t rank = xc.dim();
    const int64_t d = dim < 0 ? dim + rank : dim;
    int64_t outer = 1;
    int64_t inner = 1;
    const auto sizes = xc.sizes();
    for (int64_t i = 0; i < d; ++i) {
        outer *= sizes[i];
    }
    const int64_t reduce = sizes[d];
    for (int64_t i = d + 1; i < rank; ++i) {
        inner *= sizes[i];
    }

    const int64_t typeSize = static_cast<int64_t>(xc.element_size());
    TopkPlan p = calc_topk_plan(outer, reduce, inner, k, largest ? 1 : 0, typeSize);

    // ---- single-row fast path -----------------------------------------------------------------
    // A lone row (outer * inner == 1) can only feed ONE core with the plain scheme, so a long
    // reduce axis wastes the whole device.  Split it across cores: phase 1 computes the top-k of
    // each (unequal) reduce slice into a workspace, phase 2 merges those partials.  Every slice
    // must be at least k long so that phase 1 always emits exactly k entries per slice.
    bool useSplit = false;
    int64_t split = 0;
    int64_t chunkBase = 0;
    int64_t chunkRem = 0;
    TopkPlan p1 = p;
    TopkPlan p2 = p;
    if (p.mode == TK_MODE_ROW && outer == 1 && inner == 1 && p.coreNum > 1) {
        int64_t s = p.coreNum;
        const int64_t minChunk = (k > 32) ? k : 32;
        while (s > 1 && (reduce / s) < minChunk) {
            s /= 2;
        }
        if (s > 1 && (reduce / s) >= k) {
            split = s;
            chunkBase = reduce / split;
            chunkRem = reduce - chunkBase * split;
            p1 = calc_topk_plan(split, chunkBase, 1, k, p.largest, typeSize);
            p2 = calc_topk_plan(1, split * k, 1, k, p.largest, typeSize);
            useSplit = (p1.numBlocks > 1) && (p1.R > 0) && (p2.R > 0);
        }
    }
    torch::Tensor wsV;
    torch::Tensor wsI;
    if (useSplit) {
        wsV = torch::empty({split * k}, xc.options());
        wsI = torch::empty({split * k}, xc.options().dtype(torch::kLong));
    }

    int64_t tc = TK_T_F32;
    switch (xc.scalar_type()) {
        case torch::kFloat32: tc = TK_T_F32; break;
        case torch::kFloat16: tc = TK_T_F16; break;
        case torch::kBFloat16: tc = TK_T_BF16; break;
        case torch::kChar: tc = TK_T_I8; break;
        case torch::kByte: tc = TK_T_U8; break;
        case torch::kInt: tc = TK_T_I32; break;
        default: tc = TK_T_I64; break;
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = (GM_ADDR)xc.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();
    auto iPtr = (GM_ADDR)idx.data_ptr();

    auto acl_call = [=]() -> int {
        if (useSplit) {
            launch_topk(xPtr, (GM_ADDR)wsV.data_ptr(), (GM_ADDR)wsI.data_ptr(), nullptr, tc, split,
                        chunkBase, 1, k, p.largest, 1, p1.R, 1, split, p1.numBlocks, p1.ch,
                        TK_MODE_CHUNK, p1.accPitch, chunkRem, stream);
            launch_topk((GM_ADDR)wsV.data_ptr(), yPtr, iPtr, (GM_ADDR)wsI.data_ptr(), tc, 1, split * k, 1,
                        k, p.largest, 1, p2.R, 1, 1, 1, p2.ch, TK_MODE_MERGE, p2.accPitch, 0, stream);
        } else {
            launch_topk(xPtr, yPtr, iPtr, nullptr, tc, p.outer, p.reduce, p.inner, p.k, p.largest, p.Cs,
                        p.R, p.groups, p.numUnits, p.numBlocks, p.ch, p.mode, p.accPitch, 0, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("TopK", acl_call);
    return outs;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("top_k", topk_npu);
}

} // namespace cann_bench
