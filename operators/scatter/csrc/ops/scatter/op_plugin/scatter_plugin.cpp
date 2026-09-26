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
 * \file scatter_plugin.cpp
 * \brief Scatter API layer - torch bindings (compiled with g++)
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include <cstdint>
#include <string>

#include "../op_kernel/scatter_launch.h"

namespace cann_bench {

// The task declares the operator as `Scatter` and its python helper as `scatter`; both spellings
// are registered so either lookup resolves to the same kernel.
TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor");
    m.def("Scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor");
}

static int64_t NormalizeDim(int64_t dim, int64_t rank)
{
    int64_t d = dim < 0 ? dim + rank : dim;
    TORCH_CHECK(d >= 0 && d < rank, "scatter: dim out of range");
    return d;
}

static int64_t ParseReduce(const c10::optional<std::string> &reduce)
{
    if (!reduce.has_value()) {
        return SC_RM_UPDATE;
    }
    const std::string &s = reduce.value();
    if (s.empty() || s == "update" || s == "none" || s == "null") {
        return SC_RM_UPDATE;
    }
    if (s == "add" || s == "sum") {
        return SC_RM_ADD;
    }
    if (s == "multiply" || s == "prod") {
        return SC_RM_MUL;
    }
    if (s == "amin" || s == "min") {
        return SC_RM_AMIN;
    }
    if (s == "amax" || s == "max") {
        return SC_RM_AMAX;
    }
    TORCH_CHECK(false, "scatter: unsupported reduce mode ", s);
    return SC_RM_UPDATE;
}

static void CheckInputs(const torch::Tensor &data, const torch::Tensor &indices,
                        const torch::Tensor &updates, int64_t d)
{
    const int64_t rank = data.dim();
    TORCH_CHECK(rank >= 1, "scatter: data must be at least 1-D");
    TORCH_CHECK(indices.dim() == rank && updates.dim() == rank,
                "scatter: data/indices/updates must have the same rank");
    TORCH_CHECK(indices.sizes() == updates.sizes(), "scatter: updates shape must equal indices shape");
    TORCH_CHECK(data.scalar_type() == updates.scalar_type(),
                "scatter: data and updates must share the same dtype");
    const auto idxType = indices.scalar_type();
    TORCH_CHECK(idxType == at::kInt || idxType == at::kLong,
                "scatter: indices dtype must be int32 or int64");
    const auto dt = data.scalar_type();
    TORCH_CHECK(dt == at::kFloat || dt == at::kHalf || dt == at::kBFloat16 || dt == at::kInt ||
                    dt == at::kLong,
                "scatter: unsupported data dtype ", dt);
    // Every non-scatter dimension of indices/updates addresses data directly, so it has to coincide
    // with data on those dimensions for the flat [outer, axis, inner] view used by the kernel to be
    // exact.  Scatter semantics allow it to be smaller; that general case is not implemented here.
    for (int64_t i = 0; i < rank; ++i) {
        if (i == d) {
            continue;
        }
        TORCH_CHECK(indices.size(i) == data.size(i),
                    "scatter: indices/updates non-scatter dim ", i, " must match data");
    }
}

torch::Tensor scatter_meta(const torch::Tensor &data, int64_t dim, const torch::Tensor &indices,
                           const torch::Tensor &updates, c10::optional<std::string> reduce)
{
    const int64_t d = NormalizeDim(dim, data.dim());
    CheckInputs(data, indices, updates, d);
    return torch::empty_like(data);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("scatter", scatter_meta);
    m.impl("Scatter", scatter_meta);
}

torch::Tensor scatter_npu(const torch::Tensor &dataIn, int64_t dim, const torch::Tensor &indicesIn,
                          const torch::Tensor &updatesIn, c10::optional<std::string> reduce)
{
    const c10::OptionalDeviceGuard guard(dataIn.device());
    const int64_t rank = dataIn.dim();
    const int64_t d = NormalizeDim(dim, rank);
    CheckInputs(dataIn, indicesIn, updatesIn, d);

    const int64_t mode = ParseReduce(reduce);
    auto data = dataIn.contiguous();
    auto indices = indicesIn.contiguous();
    auto updates = updatesIn.contiguous();
    auto y = torch::empty_like(data);
    if (data.numel() == 0) {
        return y;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < d; ++i) {
        outer *= data.size(i);
    }
    const int64_t kLen = data.size(d);
    const int64_t kuLen = indices.size(d);
    int64_t inner = 1;
    for (int64_t i = d + 1; i < rank; ++i) {
        inner *= data.size(i);
    }

    int64_t dtypeCode = SC_DT_F32;
    int64_t esLT = 4;
    switch (data.scalar_type()) {
        case at::kHalf:
            dtypeCode = SC_DT_F16;
            esLT = 2;
            break;
        case at::kBFloat16:
            dtypeCode = SC_DT_BF16;
            esLT = 2;
            break;
        case at::kFloat:
            dtypeCode = SC_DT_F32;
            esLT = 4;
            break;
        case at::kInt:
            dtypeCode = SC_DT_I32;
            esLT = 4;
            break;
        default:
            dtypeCode = SC_DT_I64;
            esLT = 8;
            break;
    }
    const int64_t idxCode = (indices.scalar_type() == at::kInt) ? 0 : 1;
    const int64_t idxBytes = (idxCode == 0) ? 4 : 8;
    // 16 bit payloads cannot be combined by the aicore scalar unit, so the arithmetic modes
    // promote the resident block and the update rows to fp32 through the vector Cast.
    const int64_t promote = (esLT == 2 && mode != SC_RM_UPDATE) ? 1 : 0;
    const int64_t esCT = promote ? 4 : esLT;

    const ScatterTiling t =
        calc_scatter_tiling(outer, kLen, kuLen, inner, esLT, esCT, idxBytes, promote);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto dPtr = (GM_ADDR)data.data_ptr();
    auto iPtr = (GM_ADDR)indices.data_ptr();
    auto uPtr = (GM_ADDR)updates.data_ptr();
    auto yPtr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        launch_scatter(dPtr, iPtr, uPtr, yPtr, outer, kLen, kuLen, inner, t.tileL, t.tileK, t.ju,
                       t.nL, t.nK, t.units, t.unitsPerBlock, t.numBlocks, t.offBlk, t.offRaw,
                       t.offIdx, t.offUpd, t.totalBytes, dtypeCode, idxCode, mode,
                       (void *)stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Scatter", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("scatter", scatter_npu);
    m.impl("Scatter", scatter_npu);
}

} // namespace cann_bench
