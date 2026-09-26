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
 * \file foreach_norm_plugin.cpp
 * \brief ForeachNorm torch bindings + host tiling (compiled with g++).
 *
 * ForeachNorm(Tensor[] x, float scalar) -> Tensor[] y
 *   y[i] = (sum_j |x[i][j]|^scalar)^(1/scalar)   (special-cased for 0 / +-inf)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include <cmath>
#include <vector>
#include <algorithm>

#include "../op_kernel/foreach_norm_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("foreach_norm(Tensor[] x, float scalar) -> Tensor[]");
}

c10::List<torch::Tensor> foreach_norm_meta(c10::List<torch::Tensor> x, double scalar)
{
    (void)scalar;
    c10::List<torch::Tensor> out;
    int64_t listLen = (int64_t)x.size();
    for (int64_t i = 0; i < listLen; ++i) {
        torch::Tensor t = x.get(i);
        out.push_back(torch::empty({}, t.options()));
    }
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("foreach_norm", foreach_norm_meta);
}

c10::List<torch::Tensor> foreach_norm_npu(c10::List<torch::Tensor> x, double scalar)
{
    c10::List<torch::Tensor> out;
    int64_t listLen = (int64_t)x.size();
    if (listLen == 0) {
        return out;
    }

    torch::Tensor first = x.get(0);
    const c10::OptionalDeviceGuard guard(first.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    const auto dtype = first.scalar_type();
    for (int64_t i = 0; i < listLen; ++i) {
        TORCH_CHECK(x.get(i).scalar_type() == dtype, "foreach_norm requires a uniform tensor list dtype.");
    }

    float p = static_cast<float>(scalar);
    int32_t mode = FN_MODE_GENERAL;
    if (p == 1.0f) {
        mode = FN_MODE_L1;
    } else if (p == 2.0f) {
        mode = FN_MODE_L2;
    } else if (std::isinf(p)) {
        mode = (p > 0.0f) ? FN_MODE_MAX : FN_MODE_MIN;
    } else if (p == 0.0f) {
        mode = FN_MODE_L0;
    }

    // One common block count for every tensor so the whole-list workspace layout is uniform.
    std::vector<int64_t> lengths(listLen);
    int64_t commonBlocks = 8;
    uint32_t tileElems = calc_foreach_norm_tiling(1).tileElems;
    for (int64_t i = 0; i < listLen; ++i) {
        torch::Tensor t = x.get(i);
        lengths[i] = t.numel();
        ForeachNormTiling tl = calc_foreach_norm_tiling(lengths[i]);
        tileElems = tl.tileElems;
        commonBlocks = std::max(commonBlocks, tl.numBlocks);
    }

    // Outputs are the elements of one contiguous buffer (list dtype is uniform).
    torch::Tensor outFlat = torch::empty({listLen}, first.options());
    for (int64_t i = 0; i < listLen; ++i) {
        torch::Tensor oi = outFlat[i];
        out.push_back(oi);
    }

    torch::Tensor ws = torch::empty({listLen * commonBlocks}, first.options().dtype(torch::kFloat32));
    GM_ADDR wsPtr = (GM_ADDR)ws.data_ptr();
    GM_ADDR yPtr = (GM_ADDR)outFlat.data_ptr();

    std::vector<GM_ADDR> xPtrs(listLen);
    std::vector<int64_t> blockLengths(listLen);
    for (int64_t i = 0; i < listLen; ++i) {
        torch::Tensor t = x.get(i);
        xPtrs[i] = (GM_ADDR)t.data_ptr();
        int64_t bl = (lengths[i] > 0) ? ((lengths[i] + commonBlocks - 1) / commonBlocks) : 1;
        if (bl < 1) {
            bl = 1;
        }
        // Keep every block start 32B aligned (matches calc_foreach_norm_tiling).
        bl = ((bl + 15) / 16) * 16;
        blockLengths[i] = bl;
    }

    const int32_t dt = (int32_t)dtype;
    auto acl_call = [=]() -> int {
        for (int64_t i = 0; i < listLen; ++i) {
            int64_t totalLength = lengths[i];
            if (totalLength <= 0) {
                continue;
            }
            GM_ADDR curWs = (GM_ADDR)((float *)wsPtr + i * commonBlocks);
            if (dt == (int32_t)at::kFloat) {
                launch_foreach_norm_reduce_float(xPtrs[i], curWs, totalLength, blockLengths[i], tileElems, p, mode,
                                                 commonBlocks, stream);
            } else if (dt == (int32_t)at::kHalf) {
                launch_foreach_norm_reduce_half(xPtrs[i], curWs, totalLength, blockLengths[i], tileElems, p, mode,
                                                commonBlocks, stream);
            } else if (dt == (int32_t)at::kBFloat16) {
                launch_foreach_norm_reduce_bf16(xPtrs[i], curWs, totalLength, blockLengths[i], tileElems, p, mode,
                                                commonBlocks, stream);
            }
        }
        // Single finalize for the whole list, after every partial has been written.
        if (dt == (int32_t)at::kFloat) {
            launch_foreach_norm_finalize_float(wsPtr, yPtr, listLen, commonBlocks, p, mode, stream);
        } else if (dt == (int32_t)at::kHalf) {
            launch_foreach_norm_finalize_half(wsPtr, yPtr, listLen, commonBlocks, p, mode, stream);
        } else if (dt == (int32_t)at::kBFloat16) {
            launch_foreach_norm_finalize_bf16(wsPtr, yPtr, listLen, commonBlocks, p, mode, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ForeachNorm", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("foreach_norm", foreach_norm_npu);
}

} // namespace cann_bench
