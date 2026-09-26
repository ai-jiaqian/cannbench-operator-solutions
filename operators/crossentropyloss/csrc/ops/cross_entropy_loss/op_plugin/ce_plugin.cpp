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
 * \file ce_plugin.cpp
 * \brief CrossEntropyLoss API layer - torch bindings (compiled with g++)
 */

#include <string>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/ce_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("cross_entropy_loss(Tensor input, Tensor target, str reduction=\"mean\", int ignore_index=-100) -> Tensor loss");
}

static int64_t CeRedMode(c10::string_view reduction)
{
    if (reduction == "mean") {
        return 1;
    }
    if (reduction == "sum") {
        return 2;
    }
    if (reduction == "none") {
        return 0;
    }
    TORCH_CHECK(false, "cross_entropy_loss: unsupported reduction '", std::string(reduction), "'");
    return 1;
}

static void CeCheckInputs(const torch::Tensor &input, const torch::Tensor &target)
{
    TORCH_CHECK(input.dim() >= 2, "cross_entropy_loss: input must have at least 2 dimensions.");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16 ||
                    input.scalar_type() == torch::kBFloat16,
                "cross_entropy_loss: input dtype must be float32, float16 or bfloat16.");
    TORCH_CHECK(target.scalar_type() == torch::kInt64 || target.scalar_type() == torch::kInt32,
                "cross_entropy_loss: integral class-index targets are required.");
    TORCH_CHECK(target.dim() == input.dim() - 1,
                "cross_entropy_loss: hard-label target rank must be input rank - 1.");
}

torch::Tensor ce_meta(const torch::Tensor &input, const torch::Tensor &target, c10::string_view reduction,
                      int64_t ignore_index)
{
    (void)ignore_index;
    CeCheckInputs(input, target);
    if (CeRedMode(reduction) == 0) {
        return torch::empty(target.sizes(), input.options());
    }
    return torch::empty({}, input.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("cross_entropy_loss", ce_meta);
}

torch::Tensor ce_npu(const torch::Tensor &input, const torch::Tensor &target, c10::string_view reduction,
                     int64_t ignore_index)
{
    const c10::OptionalDeviceGuard guard(input.device());
    CeCheckInputs(input, target);

    auto x = input.contiguous();
    auto t = target.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    int64_t P = 1;
    for (int64_t i = 2; i < x.dim(); ++i) {
        P *= x.size(i);
    }
    TORCH_CHECK(t.numel() == N * P, "cross_entropy_loss: target element count (", t.numel(),
                ") must match N*prod(trailing dims) (", N * P, ").");

    const int64_t redMode = CeRedMode(reduction);
    const int64_t esz = static_cast<int64_t>(x.element_size());
    const int64_t eszT = static_cast<int64_t>(t.element_size());

    // A (N, C) matrix whose class axis is the innermost one can also be walked as
    // (1, C, N): the class axis then has stride N and the N rows become the vector
    // lanes.  For a tiny class count the row kernel would need one scalar
    // reduction (plus one tiny DMA) per row, which is latency bound; the column
    // walk instead keeps a whole lane tile busy and is fully vectorized.
    const bool useTrans = (P == 1 && C <= 8 && N >= 8192);
    const int64_t vN = useTrans ? 1 : N;
    const int64_t vC = C;
    const int64_t vP = useTrans ? N : P;

    int64_t plan[12] = {0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    ce_query_plan(vN, vC, vP, esz, eszT, redMode, plan);
    const int64_t mode = plan[0];
    const int64_t numBlocks = plan[1] < 1 ? 1 : plan[1];
    const int64_t tile = plan[2] < 1 ? 1 : plan[2];
    const int64_t cb = plan[4] < 1 ? 1 : plan[4];
    const int64_t nsplit = plan[5] < 1 ? 1 : plan[5];
    const int64_t mergeBlocks = plan[6] < 1 ? 1 : plan[6];
    const int64_t partialElems = plan[8];
    // When the class axis is split the column kernel only publishes per-lane partials and the
    // merge kernel performs the final log-sum-exp fold, so it owns the scalar reduction.
    const bool split = (mode == 1 && nsplit > 1);
    const int64_t foldBlocks = split ? mergeBlocks : numBlocks;

    torch::Tensor out;
    if (redMode == 0) {
        out = torch::empty(t.sizes(), x.options());
    } else {
        out = torch::empty({}, x.options());
    }

    torch::Tensor partT;
    void *partPtr = nullptr;
    if (redMode != 0) {
        const int64_t cap = (numBlocks > foldBlocks) ? numBlocks : foldBlocks;
        partT = at::empty({2 * cap + 8}, x.options().dtype(torch::kFloat32));
        partPtr = reinterpret_cast<void *>(partT.data_ptr<float>());
    }

    torch::Tensor lp;
    void *lpPtr = nullptr;
    if (split && partialElems > 0) {
        lp = at::empty({partialElems}, x.options().dtype(torch::kFloat32));
        lpPtr = reinterpret_cast<void *>(lp.data_ptr<float>());
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    void *xPtr = reinterpret_cast<void *>(x.data_ptr());
    void *tPtr = reinterpret_cast<void *>(t.data_ptr());
    void *oPtr = reinterpret_cast<void *>(out.data_ptr());
    const auto xType = x.scalar_type();
    const auto tType = t.scalar_type();
    const bool xF = (xType == torch::kFloat32);
    const bool xH = (xType == torch::kFloat16);
    const bool t64 = (tType == torch::kInt64);

    auto acl_call = [=]() -> int {
        if (mode == 0) {
            const int64_t R = N;
            if (xF) {
                if (t64) {
                    launch_ce_row_float_i64(xPtr, tPtr, oPtr, partPtr, R, C, ignore_index, redMode, numBlocks,
                                            tile, stream);
                } else {
                    launch_ce_row_float_i32(xPtr, tPtr, oPtr, partPtr, R, C, ignore_index, redMode, numBlocks,
                                            tile, stream);
                }
            } else if (xH) {
                if (t64) {
                    launch_ce_row_half_i64(xPtr, tPtr, oPtr, partPtr, R, C, ignore_index, redMode, numBlocks,
                                           tile, stream);
                } else {
                    launch_ce_row_half_i32(xPtr, tPtr, oPtr, partPtr, R, C, ignore_index, redMode, numBlocks,
                                           tile, stream);
                }
            } else {
                if (t64) {
                    launch_ce_row_bf16_i64(xPtr, tPtr, oPtr, partPtr, R, C, ignore_index, redMode, numBlocks,
                                           tile, stream);
                } else {
                    launch_ce_row_bf16_i32(xPtr, tPtr, oPtr, partPtr, R, C, ignore_index, redMode, numBlocks,
                                           tile, stream);
                }
            }
        } else {
            if (xF) {
                if (t64) {
                    launch_ce_col_float_i64(xPtr, tPtr, oPtr, partPtr, lpPtr, vN, vC, vP, ignore_index, redMode,
                                            numBlocks, tile, cb, nsplit, stream);
                } else {
                    launch_ce_col_float_i32(xPtr, tPtr, oPtr, partPtr, lpPtr, vN, vC, vP, ignore_index, redMode,
                                            numBlocks, tile, cb, nsplit, stream);
                }
            } else if (xH) {
                if (t64) {
                    launch_ce_col_half_i64(xPtr, tPtr, oPtr, partPtr, lpPtr, vN, vC, vP, ignore_index, redMode,
                                           numBlocks, tile, cb, nsplit, stream);
                } else {
                    launch_ce_col_half_i32(xPtr, tPtr, oPtr, partPtr, lpPtr, vN, vC, vP, ignore_index, redMode,
                                           numBlocks, tile, cb, nsplit, stream);
                }
            } else {
                if (t64) {
                    launch_ce_col_bf16_i64(xPtr, tPtr, oPtr, partPtr, lpPtr, vN, vC, vP, ignore_index, redMode,
                                           numBlocks, tile, cb, nsplit, stream);
                } else {
                    launch_ce_col_bf16_i32(xPtr, tPtr, oPtr, partPtr, lpPtr, vN, vC, vP, ignore_index, redMode,
                                           numBlocks, tile, cb, nsplit, stream);
                }
            }
            if (split) {
                if (xF) {
                    launch_ce_merge_float(lpPtr, oPtr, partPtr, vN, vC, vP, redMode, mergeBlocks, tile, nsplit,
                                          stream);
                } else if (xH) {
                    launch_ce_merge_half(lpPtr, oPtr, partPtr, vN, vC, vP, redMode, mergeBlocks, tile, nsplit,
                                         stream);
                } else {
                    launch_ce_merge_bf16(lpPtr, oPtr, partPtr, vN, vC, vP, redMode, mergeBlocks, tile, nsplit,
                                         stream);
                }
            }
        }

        if (redMode != 0 && partPtr != nullptr) {
            if (xF) {
                launch_ce_final_float(partPtr, oPtr, foldBlocks, redMode, stream);
            } else if (xH) {
                launch_ce_final_half(partPtr, oPtr, foldBlocks, redMode, stream);
            } else {
                launch_ce_final_bf16(partPtr, oPtr, foldBlocks, redMode, stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("CrossEntropyLoss", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("cross_entropy_loss", ce_npu);
}

} // namespace cann_bench
