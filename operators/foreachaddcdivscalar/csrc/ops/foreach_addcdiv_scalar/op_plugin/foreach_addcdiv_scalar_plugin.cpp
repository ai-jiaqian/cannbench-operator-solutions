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
 * \file foreach_addcdiv_scalar_plugin.cpp
 * \brief ForeachAddcdivScalar API layer - torch bindings (compiled with g++)
 *
 * Host code only prepares buffers / parameters and launches the device kernel for
 * every element of the three TensorLists.  No tensor data ever leaves the device:
 * there is no .item() / .cpu() / .numpy() / host-side copy or conversion anywhere.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include <tuple>
#include <vector>

#include "../op_kernel/foreach_addcdiv_scalar_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("foreach_addcdiv_scalar(Tensor[] x1, Tensor[] x2, Tensor[] x3, float scalar) -> Tensor[]");
}

static at::Tensor to_contiguous_device(const at::Tensor &t)
{
    return t.is_contiguous() ? t : t.contiguous();
}

std::vector<at::Tensor> foreach_addcdiv_scalar_meta(const c10::List<at::Tensor> &x1,
                                                    const c10::List<at::Tensor> &x2,
                                                    const c10::List<at::Tensor> &x3,
                                                    double scalar)
{
    (void)scalar;
    const size_t listLen = x1.size();
    TORCH_CHECK(x2.size() == listLen && x3.size() == listLen,
                "foreach_addcdiv_scalar: x1/x2/x3 must have the same list length");
    std::vector<at::Tensor> out;
    out.reserve(listLen);
    for (size_t i = 0; i < listLen; ++i) {
        out.push_back(at::empty_like(x1.get(i)));
    }
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("foreach_addcdiv_scalar", foreach_addcdiv_scalar_meta);
}

std::vector<at::Tensor> foreach_addcdiv_scalar_npu(const c10::List<at::Tensor> &x1,
                                                   const c10::List<at::Tensor> &x2,
                                                   const c10::List<at::Tensor> &x3,
                                                   double scalar)
{
    const size_t listLen = x1.size();
    TORCH_CHECK(listLen > 0, "foreach_addcdiv_scalar: empty tensor list");
    TORCH_CHECK(x2.size() == listLen && x3.size() == listLen,
                "foreach_addcdiv_scalar: x1/x2/x3 must have the same list length");

    // Owning copies of the input tensors so the deferred launch lambda can
    // capture everything by value (nothing may dangle when it returns).
    std::vector<at::Tensor> in1;
    std::vector<at::Tensor> in2;
    std::vector<at::Tensor> in3;
    in1.reserve(listLen);
    in2.reserve(listLen);
    in3.reserve(listLen);
    for (size_t i = 0; i < listLen; ++i) {
        const at::Tensor &a = x1.get(i);
        const at::Tensor &b = x2.get(i);
        const at::Tensor &c = x3.get(i);
        TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(),
                    "foreach_addcdiv_scalar: x1/x2/x3 element dtypes must match");
        TORCH_CHECK(a.numel() == b.numel() && a.numel() == c.numel(),
                    "foreach_addcdiv_scalar: x1/x2/x3 element element counts must match");
        TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kHalf ||
                        a.scalar_type() == at::kBFloat16,
                    "foreach_addcdiv_scalar: unsupported dtype ", a.scalar_type());
        in1.push_back(to_contiguous_device(a));
        in2.push_back(to_contiguous_device(b));
        in3.push_back(to_contiguous_device(c));
    }

    const c10::OptionalDeviceGuard guard(in1[0].device());

    std::vector<at::Tensor> out;
    out.reserve(listLen);
    for (size_t i = 0; i < listLen; ++i) {
        out.push_back(at::empty_like(in1[i]));
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const float scalarF = static_cast<float>(scalar);

    auto acl_call = [=]() -> int {
        for (size_t i = 0; i < in1.size(); ++i) {
            const at::Tensor &a = in1[i];
            const at::Tensor &b = in2[i];
            const at::Tensor &c = in3[i];
            const at::Tensor &o = out[i];

            const int64_t totalLength = a.numel();
            if (totalLength <= 0) {
                continue;
            }

            int64_t numBlocks = 1;
            int64_t blockLength = totalLength;
            int64_t tileElems = 64;
            std::tie(numBlocks, blockLength, tileElems) =
                calc_foreach_addcdiv_scalar_tiling_params(totalLength, static_cast<int64_t>(a.element_size()));
            const uint32_t tileElemsArg = static_cast<uint32_t>(tileElems);

            auto x1Ptr = (GM_ADDR)a.data_ptr();
            auto x2Ptr = (GM_ADDR)b.data_ptr();
            auto x3Ptr = (GM_ADDR)c.data_ptr();
            auto yPtr = (GM_ADDR)o.data_ptr();

            if (a.scalar_type() == at::kFloat) {
                FOREACH_ADDCDIV_SCALAR_LAUNCH_FLOAT(x1Ptr, x2Ptr, x3Ptr, yPtr, totalLength, numBlocks,
                                                    blockLength, tileElemsArg, scalarF, stream);
            } else if (a.scalar_type() == at::kHalf) {
                FOREACH_ADDCDIV_SCALAR_LAUNCH_HALF(x1Ptr, x2Ptr, x3Ptr, yPtr, totalLength, numBlocks,
                                                   blockLength, tileElemsArg, scalarF, stream);
            } else {
                FOREACH_ADDCDIV_SCALAR_LAUNCH_BFLOAT16(x1Ptr, x2Ptr, x3Ptr, yPtr, totalLength, numBlocks,
                                                       blockLength, tileElemsArg, scalarF, stream);
            }
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("ForeachAddcdivScalar", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("foreach_addcdiv_scalar", foreach_addcdiv_scalar_npu);
}

}  // namespace cann_bench
