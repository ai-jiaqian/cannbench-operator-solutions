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
 * \file moe_re_routing_plugin.cpp
 * \brief MoeReRouting API layer - torch bindings (compiled with g++).
 *
 * The whole execution path is device side: the outputs are allocated on the current NPU stream,
 * the tiling is derived from tensor metadata alone, and the kernel is launched through
 * OpCommand::RunOpApi.  There is no host readback and no host-side arithmetic on tensor data.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/moe_re_routing_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def(
        "moe_re_routing(Tensor tokens, Tensor expert_token_num_per_rank, "
        "Tensor? per_token_scales=None, int expert_token_num_type=1, int idx_type=0) -> "
        "(Tensor, Tensor, Tensor, Tensor)");
}

namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_re_routing_meta(
    const torch::Tensor &tokens, const torch::Tensor &expertTokenNumPerRank,
    const c10::optional<torch::Tensor> &perTokenScales, int64_t expertTokenNumType, int64_t idxType)
{
    (void)perTokenScales;
    (void)expertTokenNumType;
    (void)idxType;
    TORCH_CHECK(tokens.dim() == 2, "moe_re_routing: tokens must be 2D (A, H).");
    TORCH_CHECK(expertTokenNumPerRank.dim() == 2,
                "moe_re_routing: expert_token_num_per_rank must be 2D (N, E).");
    const int64_t a = tokens.size(0);
    const int64_t h = tokens.size(1);
    const int64_t e = expertTokenNumPerRank.size(1);
    auto permuteTokens = torch::empty({a, h}, tokens.options());
    auto permuteScales = torch::empty({a}, tokens.options().dtype(torch::kFloat32));
    auto permuteIdx = torch::empty({a}, tokens.options().dtype(torch::kInt32));
    auto expertTokenNum = torch::empty({e}, expertTokenNumPerRank.options());
    return std::make_tuple(permuteTokens, permuteScales, permuteIdx, expertTokenNum);
}

}  // namespace

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("moe_re_routing", moe_re_routing_meta);
}

namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_re_routing_npu(
    const torch::Tensor &tokensIn, const torch::Tensor &expertTokenNumPerRankIn,
    const c10::optional<torch::Tensor> &perTokenScalesIn, int64_t expertTokenNumType, int64_t idxType)
{
    (void)expertTokenNumType;
    TORCH_CHECK(idxType == 0, "moe_re_routing: only gather index mode (idx_type == 0) is supported.");

    const c10::OptionalDeviceGuard guard(tokensIn.device());

    auto tokens = tokensIn.is_contiguous() ? tokensIn : tokensIn.contiguous();
    auto counts = expertTokenNumPerRankIn.is_contiguous() ? expertTokenNumPerRankIn
                                                          : expertTokenNumPerRankIn.contiguous();

    torch::Tensor scales;
    const bool hasScales = perTokenScalesIn.has_value() && perTokenScalesIn->defined();
    if (hasScales) {
        scales = perTokenScalesIn->is_contiguous() ? *perTokenScalesIn : perTokenScalesIn->contiguous();
        TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
                    "moe_re_routing: per_token_scales must be float32.");
    }

    auto outs = moe_re_routing_meta(tokens, counts, perTokenScalesIn, expertTokenNumType, idxType);
    auto &permuteTokens = std::get<0>(outs);
    auto &permuteScales = std::get<1>(outs);
    auto &permuteIdx = std::get<2>(outs);
    auto &expertTokenNum = std::get<3>(outs);

    const int64_t a = tokens.size(0);
    const int64_t h = tokens.size(1);
    const int64_t n = counts.size(0);
    const int64_t e = counts.size(1);
    if (a <= 0 || h <= 0 || n <= 0 || e <= 0) {
        return outs;
    }

    const auto tokType = tokens.scalar_type();
    const auto cntType = counts.scalar_type();
    TORCH_CHECK(tokType == torch::kFloat16 || tokType == torch::kBFloat16 || tokType == torch::kInt8,
                "moe_re_routing: unsupported tokens dtype.");
    TORCH_CHECK(cntType == torch::kInt32 || cntType == torch::kInt64,
                "moe_re_routing: unsupported expert_token_num_per_rank dtype.");
    if (hasScales) {
        TORCH_CHECK(scales.numel() == a, "moe_re_routing: per_token_scales must have A elements.");
    }

    const int64_t tokElem = static_cast<int64_t>(tokens.element_size());
    const int64_t cntElem = static_cast<int64_t>(counts.element_size());
    MoeRRTiling tiling = calc_moe_re_routing_tiling(a, h, n, e, tokElem, cntElem);

    void *stream = c10_npu::getCurrentNPUStream().stream(false);

    GM_ADDR tokPtr = reinterpret_cast<GM_ADDR>(tokens.data_ptr());
    GM_ADDR cntPtr = reinterpret_cast<GM_ADDR>(counts.data_ptr());
    GM_ADDR sclPtr = hasScales ? reinterpret_cast<GM_ADDR>(scales.data_ptr()) : nullptr;
    GM_ADDR oTokPtr = reinterpret_cast<GM_ADDR>(permuteTokens.data_ptr());
    GM_ADDR oSclPtr = reinterpret_cast<GM_ADDR>(permuteScales.data_ptr());
    GM_ADDR oIdxPtr = reinterpret_cast<GM_ADDR>(permuteIdx.data_ptr());
    GM_ADDR oExpPtr = reinterpret_cast<GM_ADDR>(expertTokenNum.data_ptr());

    const int64_t hasScalesFlag = hasScales ? 1 : 0;
    const int64_t numBlocks = tiling.numBlocks;
    const int64_t dstPerBlk = tiling.dstPerBlk;
    const int64_t chunkTokens = tiling.chunkTokens;

    auto acl_call = [=]() -> int {
        if (tokType == torch::kFloat16 && cntType == torch::kInt32) {
            launch_moe_rr_h_i32(tokPtr, cntPtr, sclPtr, oTokPtr, oSclPtr, oIdxPtr, oExpPtr, a, h, n,
                                e, numBlocks, dstPerBlk, chunkTokens, hasScalesFlag, stream);
        } else if (tokType == torch::kFloat16 && cntType == torch::kInt64) {
            launch_moe_rr_h_i64(tokPtr, cntPtr, sclPtr, oTokPtr, oSclPtr, oIdxPtr, oExpPtr, a, h, n,
                                e, numBlocks, dstPerBlk, chunkTokens, hasScalesFlag, stream);
        } else if (tokType == torch::kBFloat16 && cntType == torch::kInt32) {
            launch_moe_rr_bf_i32(tokPtr, cntPtr, sclPtr, oTokPtr, oSclPtr, oIdxPtr, oExpPtr, a, h, n,
                                 e, numBlocks, dstPerBlk, chunkTokens, hasScalesFlag, stream);
        } else if (tokType == torch::kBFloat16 && cntType == torch::kInt64) {
            launch_moe_rr_bf_i64(tokPtr, cntPtr, sclPtr, oTokPtr, oSclPtr, oIdxPtr, oExpPtr, a, h, n,
                                 e, numBlocks, dstPerBlk, chunkTokens, hasScalesFlag, stream);
        } else if (tokType == torch::kInt8 && cntType == torch::kInt32) {
            launch_moe_rr_i8_i32(tokPtr, cntPtr, sclPtr, oTokPtr, oSclPtr, oIdxPtr, oExpPtr, a, h, n,
                                 e, numBlocks, dstPerBlk, chunkTokens, hasScalesFlag, stream);
        } else if (tokType == torch::kInt8 && cntType == torch::kInt64) {
            launch_moe_rr_i8_i64(tokPtr, cntPtr, sclPtr, oTokPtr, oSclPtr, oIdxPtr, oExpPtr, a, h, n,
                                 e, numBlocks, dstPerBlk, chunkTokens, hasScalesFlag, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MoeReRouting", acl_call);

    return outs;
}

}  // namespace

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("moe_re_routing", moe_re_routing_npu);
}

}  // namespace cann_bench
