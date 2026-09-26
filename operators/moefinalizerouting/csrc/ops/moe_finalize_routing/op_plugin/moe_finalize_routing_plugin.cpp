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
 * \file moe_finalize_routing_plugin.cpp
 * \brief MoeFinalizeRouting torch bindings (g++ TU).
 *
 * Only shape / dtype metadata is inspected on the host; every arithmetic and data movement step
 * happens inside the custom device kernel.
 */

#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/moe_finalize_routing_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("moe_finalize_routing(Tensor expanded_permuted_rows, Tensor expanded_src_to_dst_row, "
          "Tensor? skip1=None, Tensor? skip2=None, Tensor? bias=None, Tensor? scales=None, "
          "Tensor? expert_for_source_row=None, int drop_pad_mode=0) -> Tensor out");
}

static bool mfr_present(const c10::optional<torch::Tensor>& t)
{
    return t.has_value() && t->defined();
}

/*! \brief output shape derivation - pure metadata, shared by the Meta and NPU kernels. */
static void mfr_resolve(const torch::Tensor& epr, const torch::Tensor& esdr,
                        const c10::optional<torch::Tensor>& scales, int64_t& numRows, int64_t& K, int64_t& H)
{
    TORCH_CHECK(epr.dim() >= 2, "moe_finalize_routing: expanded_permuted_rows must have rank >= 2.");
    TORCH_CHECK(esdr.scalar_type() == torch::kInt32,
                "moe_finalize_routing: expanded_src_to_dst_row must be int32.");
    H = epr.size(-1);
    const int64_t nk = esdr.numel();
    K = 1;
    if (mfr_present(scales)) {
        TORCH_CHECK(scales->dim() == 2, "moe_finalize_routing: scales must be 2-D (NUM_ROWS, K).");
        K = scales->size(1);
    }
    TORCH_CHECK(K > 0, "moe_finalize_routing: K must be positive.");
    TORCH_CHECK(nk % K == 0, "moe_finalize_routing: expanded_src_to_dst_row length must be K*NUM_ROWS.");
    numRows = nk / K;
}

torch::Tensor moe_finalize_routing_meta(const torch::Tensor& expanded_permuted_rows,
                                        const torch::Tensor& expanded_src_to_dst_row,
                                        c10::optional<torch::Tensor> skip1, c10::optional<torch::Tensor> skip2,
                                        c10::optional<torch::Tensor> bias, c10::optional<torch::Tensor> scales,
                                        c10::optional<torch::Tensor> expert_for_source_row,
                                        int64_t drop_pad_mode)
{
    (void)skip1;
    (void)skip2;
    (void)bias;
    (void)expert_for_source_row;
    (void)drop_pad_mode;
    int64_t numRows = 0;
    int64_t K = 1;
    int64_t H = 0;
    mfr_resolve(expanded_permuted_rows, expanded_src_to_dst_row, scales, numRows, K, H);
    return torch::empty({numRows, H}, expanded_permuted_rows.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("moe_finalize_routing", moe_finalize_routing_meta);
}

static int mfr_dtype_code(const torch::Tensor& t)
{
    switch (t.scalar_type()) {
        case torch::kFloat16:
            return 0;
        case torch::kFloat32:
            return 1;
        case torch::kBFloat16:
            return 2;
        default:
            return -1;
    }
}

torch::Tensor moe_finalize_routing_npu(const torch::Tensor& expanded_permuted_rows,
                                       const torch::Tensor& expanded_src_to_dst_row,
                                       c10::optional<torch::Tensor> skip1, c10::optional<torch::Tensor> skip2,
                                       c10::optional<torch::Tensor> bias, c10::optional<torch::Tensor> scales,
                                       c10::optional<torch::Tensor> expert_for_source_row, int64_t drop_pad_mode)
{
    const auto tEpr = expanded_permuted_rows.scalar_type();
    TORCH_CHECK(tEpr == torch::kFloat16 || tEpr == torch::kFloat32 || tEpr == torch::kBFloat16,
                "moe_finalize_routing: expanded_permuted_rows must be float16 / float32 / bfloat16.");

    const c10::OptionalDeviceGuard guard(expanded_permuted_rows.device());

    auto epr = expanded_permuted_rows.contiguous();
    auto esdr = expanded_src_to_dst_row.contiguous();

    int64_t numRows = 0;
    int64_t K = 1;
    int64_t H = 0;
    mfr_resolve(epr, esdr, scales, numRows, K, H);
    if (numRows <= 0 || H <= 0 || K <= 0) {
        return torch::empty({numRows, H}, epr.options());
    }

    const int64_t numDst = epr.numel() / H;
    const bool hasScales = mfr_present(scales);
    const bool hasExpert = mfr_present(expert_for_source_row);
    const bool hasBias = mfr_present(bias) && hasExpert;
    const int64_t E = hasBias ? bias->size(0) : 0;
    const bool hasSkip1 = mfr_present(skip1);
    const bool hasSkip2 = mfr_present(skip2);
    if (hasBias) {
        TORCH_CHECK(bias->size(1) == H, "moe_finalize_routing: bias head dim must equal H.");
    }
    int64_t mode = drop_pad_mode;
    if (mode < 0) {
        mode = 0;
    }
    if (mode > 3) {
        mode = 3;
    }

    const int64_t elemSize = (tEpr == torch::kFloat32) ? 4 : 2;
    int64_t numBlocks = 1;
    int64_t Hc = 1;
    int64_t nb = 1;
    std::tie(numBlocks, Hc, nb) = calc_mfr_tiling(numRows, H, K, E, elemSize);

    const int64_t f1 = hasSkip1 ? 1 : 0;
    const int64_t f2 = hasSkip2 ? 1 : 0;
    const int64_t f3 = hasBias ? 1 : 0;
    const int64_t f4 = hasScales ? 1 : 0;

    const int tCode = mfr_dtype_code(epr);
    const int sCode = hasScales ? mfr_dtype_code(*scales) : tCode;
    TORCH_CHECK(tCode >= 0 && sCode >= 0,
                "moe_finalize_routing: unsupported dtype combination for expanded_permuted_rows / scales.");

    // Materialise (and keep alive) contiguous views of every optional operand.
    torch::Tensor sk1;
    torch::Tensor sk2;
    torch::Tensor bs;
    torch::Tensor sc;
    torch::Tensor ex;
    if (hasSkip1) {
        sk1 = skip1->contiguous();
    }
    if (hasSkip2) {
        sk2 = skip2->contiguous();
    }
    if (hasBias) {
        bs = bias->contiguous();
    }
    if (hasScales) {
        sc = scales->contiguous();
    }
    if (hasExpert) {
        ex = expert_for_source_row->contiguous();
    }

    auto out = torch::empty({numRows, H}, epr.options());

    GM_ADDR eprPtr = reinterpret_cast<GM_ADDR>(epr.data_ptr());
    GM_ADDR esdrPtr = reinterpret_cast<GM_ADDR>(esdr.data_ptr());
    // Absent optional tensors are never dereferenced by the kernel; reuse a valid pointer.
    GM_ADDR skip1Ptr = hasSkip1 ? reinterpret_cast<GM_ADDR>(sk1.data_ptr()) : eprPtr;
    GM_ADDR skip2Ptr = hasSkip2 ? reinterpret_cast<GM_ADDR>(sk2.data_ptr()) : eprPtr;
    GM_ADDR biasPtr = hasBias ? reinterpret_cast<GM_ADDR>(bs.data_ptr()) : eprPtr;
    GM_ADDR scalesPtr = hasScales ? reinterpret_cast<GM_ADDR>(sc.data_ptr()) : eprPtr;
    GM_ADDR expertPtr = hasExpert ? reinterpret_cast<GM_ADDR>(ex.data_ptr()) : esdrPtr;
    GM_ADDR outP = reinterpret_cast<GM_ADDR>(out.data_ptr());

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

#define MFR_LAUNCH(TN, SN)                                                                       \
    launch_mfr_##TN##_##SN(eprPtr, esdrPtr, skip1Ptr, skip2Ptr, biasPtr, scalesPtr, expertPtr, outP, \
                           numRows, H, K, E, numDst, mode, Hc, nb, numBlocks, f1, f2, f3, f4, stream)

    auto acl_call = [=]() -> int {
        if (tCode == 0) {
            if (sCode == 0) {
                MFR_LAUNCH(half, half);
            } else if (sCode == 1) {
                MFR_LAUNCH(half, float);
            } else {
                MFR_LAUNCH(half, bf16);
            }
        } else if (tCode == 1) {
            if (sCode == 0) {
                MFR_LAUNCH(float, half);
            } else if (sCode == 1) {
                MFR_LAUNCH(float, float);
            } else {
                MFR_LAUNCH(float, bf16);
            }
        } else {
            if (sCode == 0) {
                MFR_LAUNCH(bf16, half);
            } else if (sCode == 1) {
                MFR_LAUNCH(bf16, float);
            } else {
                MFR_LAUNCH(bf16, bf16);
            }
        }
        return 0;
    };

#undef MFR_LAUNCH

    at_npu::native::OpCommand::RunOpApi("MoeFinalizeRouting", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("moe_finalize_routing", moe_finalize_routing_npu);
}

}  // namespace cann_bench
