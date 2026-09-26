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
 * \file quant_matmul_plugin.cpp
 * \brief QuantMatmul API layer - torch bindings (compiled with g++)
 */

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/quant_matmul_launch.h"

namespace cann_bench {

namespace {

constexpr int64_t QM_BIAS_NONE = 0;
constexpr int64_t QM_BIAS_I32 = 1;
constexpr int64_t QM_BIAS_F16 = 2;
constexpr int64_t QM_BIAS_BF16 = 3;
constexpr int64_t QM_BIAS_F32 = 4;

inline int64_t ElemBytes(torch::ScalarType t)
{
    return static_cast<int64_t>(torch::elementSize(t));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, Tensor? offset=None, "
          "Tensor? pertoken_scale=None, Tensor? bias=None, str? output_dtype=None) -> Tensor");
}

torch::Tensor quant_matmul_meta(const torch::Tensor& x1, const torch::Tensor& x2,
                                const torch::Tensor& scale,
                                const c10::optional<torch::Tensor>& offset,
                                const c10::optional<torch::Tensor>& pertoken_scale,
                                const c10::optional<torch::Tensor>& bias,
                                const c10::optional<std::string>& output_dtype)
{
    (void)scale;
    (void)offset;
    (void)pertoken_scale;
    (void)bias;
    TORCH_CHECK(x1.dim() >= 2 && x2.dim() >= 2, "quant_matmul requires inputs of rank >= 2.");
    TORCH_CHECK(x1.size(-1) == x2.size(-2), "quant_matmul requires x1.size(-1) == x2.size(-2).");
    bool bf16out = output_dtype.has_value() && output_dtype.value() == "bfloat16";
    const auto dt = bf16out ? torch::kBFloat16 : torch::kFloat16;
    std::vector<int64_t> outShape(x1.sizes().begin(), x1.sizes().end() - 1);
    outShape.push_back(x2.size(-1));
    return torch::empty(outShape, x1.options().dtype(dt));
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("quant_matmul", quant_matmul_meta);
}

torch::Tensor quant_matmul_npu(const torch::Tensor& x1, const torch::Tensor& x2,
                               const torch::Tensor& scale,
                               const c10::optional<torch::Tensor>& offset,
                               const c10::optional<torch::Tensor>& pertoken_scale,
                               const c10::optional<torch::Tensor>& bias,
                               const c10::optional<std::string>& output_dtype)
{
    const c10::OptionalDeviceGuard guard(x1.device());

    TORCH_CHECK(x1.scalar_type() == torch::kChar && x2.scalar_type() == torch::kChar,
                "quant_matmul requires int8 x1 and x2.");
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous() && scale.is_contiguous(),
                "quant_matmul requires contiguous x1/x2/scale.");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32 || scale.scalar_type() == torch::kBFloat16,
                "quant_matmul requires float32 or bfloat16 scale.");
    TORCH_CHECK(x1.dim() >= 2 && x2.dim() >= 2 && x1.dim() == x2.dim(),
                "quant_matmul expects equal ranks >= 2.");
    TORCH_CHECK(x1.size(-1) == x2.size(-2), "quant_matmul requires x1.size(-1) == x2.size(-2).");

    const int64_t K = x1.size(-1);
    const int64_t M = x1.size(-2);
    const int64_t N = x2.size(-1);
    TORCH_CHECK(K > 0 && M > 0 && N > 0, "quant_matmul requires non-empty shapes.");

    int64_t batch = 1;
    for (int64_t d = 0; d + 2 < x1.dim(); ++d) {
        TORCH_CHECK(x1.size(d) == x2.size(d),
                    "quant_matmul expects matching leading batch dimensions.");
        batch *= x1.size(d);
    }

    auto out = quant_matmul_meta(x1, x2, scale, offset, pertoken_scale, bias, output_dtype);
    const bool bf16out = out.scalar_type() == torch::kBFloat16;

    int64_t flags = 0;
    const int64_t scaleNumel = scale.numel();
    TORCH_CHECK(scaleNumel == 1 || scaleNumel == N, "quant_matmul expects scale length 1 or n.");
    const bool scalePerChan = scaleNumel > 1;
    if (scalePerChan) {
        flags |= qm::QM_F_SCALE_PERCHAN;
    }
    if (scale.scalar_type() == torch::kBFloat16) {
        flags |= qm::QM_F_SCALE_BF16;
    }

    bool hasOffset = offset.has_value() && offset->defined();
    if (hasOffset) {
        TORCH_CHECK(offset->scalar_type() == torch::kFloat32 && offset->is_contiguous(),
                    "quant_matmul requires float32 contiguous offset.");
        TORCH_CHECK(offset->numel() == 1 || offset->numel() == N,
                    "quant_matmul expects offset length 1 or n.");
        flags |= qm::QM_F_HAS_OFFSET;
        if (offset->numel() > 1) {
            flags |= qm::QM_F_OFFSET_PERCHAN;
        }
    }

    bool hasPt = pertoken_scale.has_value() && pertoken_scale->defined();
    if (hasPt) {
        TORCH_CHECK(pertoken_scale->scalar_type() == torch::kFloat32 &&
                        pertoken_scale->is_contiguous(),
                    "quant_matmul requires float32 contiguous pertoken_scale.");
        TORCH_CHECK(pertoken_scale->numel() == M,
                    "quant_matmul expects pertoken_scale length m.");
        flags |= qm::QM_F_HAS_PT;
    }

    bool hasBias = bias.has_value() && bias->defined();
    int64_t biasKind = QM_BIAS_NONE;
    bool biasPerChan = false;
    bool biasBatched = false;
    if (hasBias) {
        TORCH_CHECK(bias->is_contiguous(), "quant_matmul requires contiguous bias.");
        const auto bt = bias->scalar_type();
        if (bt == torch::kInt32) {
            biasKind = QM_BIAS_I32;
        } else if (bt == torch::kFloat16) {
            biasKind = QM_BIAS_F16;
        } else if (bt == torch::kBFloat16) {
            biasKind = QM_BIAS_BF16;
        } else if (bt == torch::kFloat32) {
            biasKind = QM_BIAS_F32;
        } else {
            TORCH_CHECK(false, "quant_matmul unsupported bias dtype: ", bt);
        }
        if (bias->dim() == 3 && bias->size(1) == 1 && bias->size(2) == N && batch > 1 &&
            bias->size(0) == batch) {
            biasBatched = true;
            biasPerChan = (N > 1);
        } else {
            TORCH_CHECK(bias->numel() == 1 || bias->numel() == N,
                        "quant_matmul expects bias length 1 or n.");
            biasPerChan = bias->numel() > 1;
        }
        flags |= (biasKind << qm::QM_F_BIAS_SHIFT);
        if (biasPerChan) {
            flags |= qm::QM_F_BIAS_PERCHAN;
        }
    }

    // ---- tiling ------------------------------------------------------------------------------
    int32_t tilingBlob[qm::QM_TILING_INTS];
    int64_t tilingInfo[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int64_t numBlocks = qm_calc_cube_tiling(M, N, K, 0, tilingBlob, tilingInfo);
    int64_t libWsBytes = tilingInfo[0];
    int64_t aivBlocks = tilingInfo[2];
    if (libWsBytes < (32LL << 20)) {
        libWsBytes = 32LL << 20;
    }
    if (aivBlocks <= 0) {
        aivBlocks = 24;
    }

    // ---- device workspaces -------------------------------------------------------------------
    auto i32opts = x1.options().dtype(torch::kInt32);
    auto byteopts = x1.options().dtype(torch::kUInt8);
    auto cWs = torch::empty({batch * M * N}, i32opts);
    auto sysWs = torch::empty({libWsBytes + (8LL << 20)}, byteopts);

    const auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto* x1p = static_cast<GM_ADDR>(x1.data_ptr());
    auto* x2p = static_cast<GM_ADDR>(x2.data_ptr());
    auto* cp = static_cast<GM_ADDR>(cWs.data_ptr());
    auto* wsp = static_cast<GM_ADDR>(sysWs.data_ptr());
    auto* outp = static_cast<GM_ADDR>(out.data_ptr());
    auto* scalep = static_cast<GM_ADDR>(scale.data_ptr());
    auto* offp = hasOffset ? static_cast<GM_ADDR>(offset->data_ptr()) : static_cast<GM_ADDR>(nullptr);
    auto* ptp = hasPt ? static_cast<GM_ADDR>(pertoken_scale->data_ptr())
                      : static_cast<GM_ADDR>(nullptr);
    auto* biasp = hasBias ? static_cast<GM_ADDR>(bias->data_ptr()) : static_cast<GM_ADDR>(nullptr);
    const int64_t biasElemBytes = hasBias ? ElemBytes(bias->scalar_type()) : 4;

    qm::QmCubeArgs cargs;
    for (int32_t i = 0; i < qm::QM_TILING_INTS; ++i) {
        cargs.tiling[i] = tilingBlob[i];
    }
    cargs.M = M;
    cargs.N = N;
    cargs.K = K;
    cargs.batch = batch;
    cargs.numBlocks = batch * numBlocks;
    cargs.nr = tilingInfo[4] > 0 ? tilingInfo[4] : 1;
    cargs.nc = tilingInfo[5] > 0 ? tilingInfo[5] : 1;
    cargs.scM = tilingInfo[6] > 0 ? tilingInfo[6] : M;
    cargs.scN = tilingInfo[7] > 0 ? tilingInfo[7] : N;

    const int64_t cSlice = M * N * 4;
    const int64_t oSlice = M * N * 2;

    qm::QmEpiArgs eargs;
    eargs.M = M;
    eargs.N = N;
    eargs.numBlocks = aivBlocks;
    if (biasBatched) {
        eargs.rows = M;
    } else {
        eargs.rows = batch * M;
        if (batch > 1) {
            flags |= qm::QM_F_PT_MOD_M;
        }
    }
    eargs.flags = flags;

    auto acl_call = [=]() -> int {
        // One cube launch covers every batch: the kernel maps its block index to
        // (batch, row tile, column tile) and strides A / B / C by the batch size.
        qm_launch_cube_i32(x1p, x2p, cp, wsp, &cargs, stream);
        if (biasBatched) {
            for (int64_t b = 0; b < batch; ++b) {
                char* cbase = static_cast<char*>(cp) + b * cSlice;
                char* obase = static_cast<char*>(outp) + b * oSlice;
                char* bbias = static_cast<char*>(biasp) + b * N * biasElemBytes;
                if (bf16out) {
                    qm_launch_epilogue_bf16(cbase, obase, scalep, offp, ptp, bbias, &eargs, stream);
                } else {
                    qm_launch_epilogue_f16(cbase, obase, scalep, offp, ptp, bbias, &eargs, stream);
                }
            }
        } else {
            if (bf16out) {
                qm_launch_epilogue_bf16(cp, outp, scalep, offp, ptp, biasp, &eargs, stream);
            } else {
                qm_launch_epilogue_f16(cp, outp, scalep, offp, ptp, biasp, &eargs, stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("QuantMatmul", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("quant_matmul", quant_matmul_npu);
}

}  // namespace cann_bench
