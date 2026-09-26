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
 * \file gcd_plugin.cpp
 * \brief Gcd API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/gcd_launch.h"

namespace cann_bench {

namespace {

// Right-aligned numpy broadcast shape of x1 and x2.
void InferBroadcastShape(const torch::Tensor& x1, const torch::Tensor& x2, std::vector<int64_t>& out)
{
    const int64_t r1 = static_cast<int64_t>(x1.dim());
    const int64_t r2 = static_cast<int64_t>(x2.dim());
    const int64_t r = std::max(r1, r2);
    out.assign(static_cast<size_t>(std::max<int64_t>(r, 1)), 1);
    for (int64_t d = 0; d < r; ++d) {
        const int64_t i1 = d - (r - r1);
        const int64_t i2 = d - (r - r2);
        const int64_t a = (i1 >= 0) ? x1.size(i1) : 1;
        const int64_t b = (i2 >= 0) ? x2.size(i2) : 1;
        TORCH_CHECK(a == b || a == 1 || b == 1, "gcd: input shapes are not broadcastable");
        out[static_cast<size_t>(d)] = std::max(a, b);
    }
}

// Build the trailing-run plan from the two (contiguous) operand shapes.
void BuildGcdPlan(const torch::Tensor& a1, const torch::Tensor& a2, GcdPlan& p)
{
    constexpr int64_t MAXR = GCD_MAX_RANK;
    const int64_t r1 = static_cast<int64_t>(a1.dim());
    const int64_t r2 = static_cast<int64_t>(a2.dim());
    const int64_t r = std::max(r1, r2);

    int64_t sz1[MAXR];
    int64_t sz2[MAXR];
    int64_t outShape[MAXR];
    int64_t s1[MAXR];
    int64_t s2[MAXR];
    int64_t oStride[MAXR];
    for (int64_t d = 0; d < MAXR; ++d) {
        sz1[d] = 1;
        sz2[d] = 1;
        outShape[d] = 1;
        s1[d] = 0;
        s2[d] = 0;
        oStride[d] = 1;
    }
    for (int64_t d = 0; d < r; ++d) {
        const int64_t i1 = d - (r - r1);
        const int64_t i2 = d - (r - r2);
        sz1[d] = (i1 >= 0) ? a1.size(i1) : 1;
        sz2[d] = (i2 >= 0) ? a2.size(i2) : 1;
        outShape[d] = std::max(sz1[d], sz2[d]);
    }
    {
        int64_t acc1 = 1;
        int64_t acc2 = 1;
        for (int64_t d = r - 1; d >= 0; --d) {
            s1[d] = (sz1[d] == 1) ? 0 : acc1;
            s2[d] = (sz2[d] == 1) ? 0 : acc2;
            acc1 *= sz1[d];
            acc2 *= sz2[d];
        }
    }
    {
        int64_t acc = 1;
        for (int64_t d = r - 1; d >= 0; --d) {
            oStride[d] = acc;
            acc *= outShape[d];
        }
    }

    // Maximal trailing run over which both operands are uniformly affine or constant.
    int64_t k = 0;
    int64_t st1 = 0;
    int64_t st2 = 0;
    for (int64_t d = r - 1; d >= 0; --d) {
        int64_t t1 = 0;
        int64_t t2 = 0;
        if (s1[d] == 0) {
            t1 = GCD_CLS_CONST;
        } else if (s1[d] == oStride[d]) {
            t1 = GCD_CLS_AFFINE;
        } else {
            break;
        }
        if (s2[d] == 0) {
            t2 = GCD_CLS_CONST;
        } else if (s2[d] == oStride[d]) {
            t2 = GCD_CLS_AFFINE;
        } else {
            break;
        }
        if (k > 0 && (t1 != st1 || t2 != st2)) {
            break;
        }
        st1 = t1;
        st2 = t2;
        ++k;
    }

    int64_t innerLen = 1;
    for (int64_t d = r - k; d < r; ++d) {
        innerLen *= outShape[d];
    }
    int64_t kOuter = r - k;

    // Merge the dimension just outside the run when x1 stays at the run stride and x2 turns
    // into either a plain constant or a per-period constant ("repeat") operand.
    int64_t repR = 0;
    int64_t repStep = 0;
    while (kOuter >= 1 && innerLen > 1) {
        const int64_t d = kOuter - 1;
        const int64_t P = outShape[d];
        if (P <= 1 || s1[d] != innerLen) {
            break;
        }
        if (repR != 0) {
            break;
        }
        if (st2 == GCD_CLS_CONST && s2[d] == 0) {
            // x2 stays constant across the merged dimension: extend the run, keep looking.
            innerLen *= P;
            kOuter -= 1;
            continue;
        }
        if (st2 == GCD_CLS_CONST) {
            // x2 advances by s2[d] per period of innerLen elements.
            repR = innerLen;
            repStep = s2[d];
            innerLen *= P;
            kOuter -= 1;
            st2 = GCD_CLS_REPEAT;
            break;
        }
        // cls2 == AFFINE with s2[d] != 0 cannot be represented as a single run.
        break;
    }

    int64_t rowsTotal = 1;
    for (int64_t d = 0; d < kOuter; ++d) {
        rowsTotal *= outShape[d];
    }

    for (int64_t d = 0; d < MAXR; ++d) {
        p.oShape[d] = 0;
        p.oS1[d] = 0;
        p.oS2[d] = 0;
    }
    for (int64_t d = 0; d < kOuter; ++d) {
        p.oShape[d] = static_cast<int32_t>(outShape[d]);
        p.oS1[d] = static_cast<int32_t>(s1[d]);
        p.oS2[d] = static_cast<int32_t>(s2[d]);
    }
    p.innerLen = static_cast<int32_t>(innerLen);
    p.rowsTotal = static_cast<int32_t>(rowsTotal);
    p.kOuter = static_cast<int32_t>(kOuter);
    p.cls1 = static_cast<int32_t>(st1);
    p.cls2 = static_cast<int32_t>(st2);
    p.repR = static_cast<int32_t>(repR);
    p.repStep = static_cast<int32_t>(repStep);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("gcd(Tensor x1, Tensor x2) -> Tensor");
}

torch::Tensor gcd_meta(const torch::Tensor& x1, const torch::Tensor& x2)
{
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "gcd: inputs must share one dtype");
    std::vector<int64_t> outShape;
    InferBroadcastShape(x1, x2, outShape);
    return torch::empty(outShape, x1.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("gcd", gcd_meta);
}

torch::Tensor gcd_npu(const torch::Tensor& x1, const torch::Tensor& x2)
{
    const c10::OptionalDeviceGuard guard(x1.device());

    auto a1 = x1.contiguous();
    auto a2 = x2.contiguous();

    std::vector<int64_t> outShape;
    InferBroadcastShape(a1, a2, outShape);
    torch::Tensor out = torch::empty(outShape, a1.options());

    const int64_t total = out.numel();
    if (total == 0) {
        return out;
    }

    GcdPlan plan;
    BuildGcdPlan(a1, a2, plan);

    const int64_t elemSize = static_cast<int64_t>(a1.element_size());
    int64_t numBlocks = 0;
    int64_t blockLength = 0;
    int64_t tileElems = 0;
    std::tie(numBlocks, blockLength, tileElems) = calc_gcd_tiling(total, elemSize);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    GM_ADDR p1 = (GM_ADDR)a1.data_ptr();
    GM_ADDR p2 = (GM_ADDR)a2.data_ptr();
    GM_ADDR po = (GM_ADDR)out.data_ptr();
    const auto dtype = a1.scalar_type();

    auto acl_call = [=]() -> int {
        if (dtype == torch::kInt16) {
            launch_gcd_kernel_int16(p1, p2, po, &plan, total, numBlocks, blockLength, tileElems, stream);
        } else if (dtype == torch::kInt32) {
            launch_gcd_kernel_int32(p1, p2, po, &plan, total, numBlocks, blockLength, tileElems, stream);
        } else if (dtype == torch::kInt64) {
            launch_gcd_kernel_int64(p1, p2, po, &plan, total, numBlocks, blockLength, tileElems, stream);
        } else {
            return -1;
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Gcd", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("gcd", gcd_npu);
}

} // namespace cann_bench
