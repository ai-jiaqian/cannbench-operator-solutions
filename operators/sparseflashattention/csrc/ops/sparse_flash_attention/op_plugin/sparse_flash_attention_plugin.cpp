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
 * \file sparse_flash_attention_plugin.cpp
 * \brief SparseFlashAttention API layer - torch bindings (compiled with g++).
 *
 * Only shape / dtype / stride metadata is inspected here; all arithmetic and data movement happens
 * inside the custom device kernel. BSND and BNSD are handled by reading the real per-axis element
 * strides, so the kernel body is layout agnostic.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/sparse_flash_attention_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparseIndices, "
          "float scaleValue, str inputLayout=\"BSND\", bool is_causal=False) -> Tensor");
}

namespace {

constexpr int64_t SFA_KT_MAXPLUGIN = 64;       // largest key block the vector ops below accept
constexpr int64_t SFA_KT_SMALL = 8;            // fallback for tiny topK (< 16)
constexpr int64_t SFA_DC_PLUGIN = 64;          // fp32 elements per vector repeat
constexpr int64_t SFA_UB_BUDGET = 188 * 1024;  // keep clear of the 192 KiB UB

inline int64_t sfa_align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

struct SfaShape {
    int64_t B = 0, N1 = 0, N2 = 0, S1 = 0, S2 = 0, Dk = 0, Dv = 0, topK = 0, G = 1;
    bool bsnd = true;
    int64_t qSN = 0, qSS = 0, qSB = 0;
    int64_t kSN = 0, kSS = 0, kSB = 0;
    int64_t vSN = 0, vSS = 0, vSB = 0;
    int64_t iSN = 0, iSS = 0, iSB = 0;
};

SfaShape sfa_resolve(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                     const torch::Tensor &si, const std::string &layout)
{
    SfaShape d;
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && si.dim() == 4,
                "SparseFlashAttention: query/key/value/sparseIndices must all be 4-D.");
    std::string up;
    for (char c : layout) {
        up.push_back(static_cast<char>(::toupper(static_cast<unsigned char>(c))));
    }
    TORCH_CHECK(up == "BSND" || up == "BNSD", "SparseFlashAttention: inputLayout must be BSND or BNSD.");
    d.bsnd = (up == "BSND");

    d.B = q.size(0);
    d.Dk = q.size(3);
    d.Dv = v.size(3);
    if (d.bsnd) {
        d.S1 = q.size(1);
        d.N1 = q.size(2);
        d.S2 = k.size(1);
        d.N2 = k.size(2);
        TORCH_CHECK(si.size(0) == d.B && si.size(1) == d.S1 && si.size(2) == d.N2,
                    "SparseFlashAttention: sparseIndices shape mismatch (BSND).");
        TORCH_CHECK(k.size(0) == d.B && v.size(0) == d.B && v.size(1) == d.S2 && v.size(2) == d.N2,
                    "SparseFlashAttention: key/value shape mismatch (BSND).");
    } else {
        d.N1 = q.size(1);
        d.S1 = q.size(2);
        d.N2 = k.size(1);
        d.S2 = k.size(2);
        TORCH_CHECK(si.size(0) == d.B && si.size(1) == d.N2 && si.size(2) == d.S1,
                    "SparseFlashAttention: sparseIndices shape mismatch (BNSD).");
        TORCH_CHECK(k.size(0) == d.B && v.size(0) == d.B && v.size(1) == d.N2 && v.size(2) == d.S2,
                    "SparseFlashAttention: key/value shape mismatch (BNSD).");
    }
    TORCH_CHECK(k.size(3) == d.Dk, "SparseFlashAttention: key head dim must equal the query head dim.");
    TORCH_CHECK(d.N2 > 0 && d.N1 % d.N2 == 0, "SparseFlashAttention: N1 must be a multiple of N2.");
    d.topK = si.size(3);
    TORCH_CHECK(d.topK > 0, "SparseFlashAttention: sparseIndices must not be empty.");
    d.G = d.N1 / d.N2;

    // the innermost head dim must be contiguous: the kernel gathers whole rows with one DataCopyPad
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1 && si.stride(3) == 1,
                "SparseFlashAttention: the innermost dimension of every input must be contiguous.");

    const int64_t headAxis = d.bsnd ? 2 : 1;
    const int64_t seqAxis = d.bsnd ? 1 : 2;
    d.qSN = q.stride(headAxis); d.qSS = q.stride(seqAxis); d.qSB = q.stride(0);
    d.kSN = k.stride(headAxis); d.kSS = k.stride(seqAxis); d.kSB = k.stride(0);
    d.vSN = v.stride(headAxis); d.vSS = v.stride(seqAxis); d.vSB = v.stride(0);
    d.iSN = si.stride(headAxis); d.iSS = si.stride(seqAxis); d.iSB = si.stride(0);
    return d;
}

torch::Tensor sfa_meta(const torch::Tensor &query, const torch::Tensor &key, const torch::Tensor &value,
                       const torch::Tensor &sparseIndices, double scaleValue,
                       c10::string_view inputLayout, bool is_causal)
{
    (void)scaleValue;
    (void)is_causal;
    const std::string layout(inputLayout.data(), inputLayout.size());
    SfaShape d = sfa_resolve(query, key, value, sparseIndices, layout);
    if (d.bsnd) {
        return torch::empty({d.B, d.S1, d.N1, d.Dv}, query.options());
    }
    return torch::empty({d.B, d.N1, d.S1, d.Dv}, query.options());
}

/*!
 * \brief Exact UB footprint of one kernel instance, in bytes, for a given head-block size.
 *
 * Mirrors the InitBuffer list of sfa_kernel one for one: the shared query/odd-parity staging buffer
 * (sized by the larger of the query tile and one K/V key block), per-head buffers, per-key-block
 * buffers, plus 32B rounding slack per buffer.
 */
int64_t sfa_footprint(int64_t HG, int64_t KT, int64_t DkAl, int64_t DvAl, int64_t DmxAl, int64_t topKAl)
{
    const int64_t scr = std::max(HG * DkAl, KT * DmxAl) * 2;
    const int64_t perHead = DkAl * 4 + DvAl * 4 + topKAl * 4 + 8 * 4 + 32;
    const int64_t perBlock = DmxAl * 2 + DmxAl * 4 + SFA_DC_PLUGIN * 4 * 2 + 8 * 4 + 4 + 32 + 4 + 32;
    return scr + HG * perHead + KT * perBlock + topKAl * 8 + DvAl * 2 + 1024 + 20 * 32;
}

}  // namespace

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("sparse_flash_attention", sfa_meta);
}

torch::Tensor sfa_npu(const torch::Tensor &query, const torch::Tensor &key, const torch::Tensor &value,
                      const torch::Tensor &sparseIndices, double scaleValue,
                      c10::string_view inputLayout, bool is_causal)
{
    const c10::OptionalDeviceGuard guard(query.device());

    TORCH_CHECK(query.scalar_type() == torch::kFloat16 || query.scalar_type() == torch::kBFloat16,
                "SparseFlashAttention: only float16 / bfloat16 are supported.");
    TORCH_CHECK(key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type(),
                "SparseFlashAttention: query / key / value dtypes must match.");
    TORCH_CHECK(sparseIndices.scalar_type() == torch::kInt32,
                "SparseFlashAttention: sparseIndices must be int32.");

    const std::string layout(inputLayout.data(), inputLayout.size());
    SfaShape d = sfa_resolve(query, key, value, sparseIndices, layout);

    torch::Tensor y = sfa_meta(query, key, value, sparseIndices, scaleValue, inputLayout, is_causal);
    if (d.B * d.S1 * d.N1 == 0) {
        return y;
    }

    // the output is freshly allocated and contiguous in the input's logical order
    int64_t ySN, ySS, ySB;
    if (d.bsnd) {
        ySN = d.Dv; ySS = d.N1 * d.Dv; ySB = d.S1 * d.N1 * d.Dv;
    } else {
        ySN = d.S1 * d.Dv; ySS = d.Dv; ySB = d.N1 * d.S1 * d.Dv;
    }

    const int64_t DkAl = sfa_align_up(d.Dk, 64);
    const int64_t DvAl = sfa_align_up(d.Dv, 64);
    const int64_t DmxAl = std::max(DkAl, DvAl);
    TORCH_CHECK(DmxAl <= 2040, "SparseFlashAttention: head dim too large for the vector row stride.");

    const int64_t hgCands[5] = {16, 8, 4, 2, 1};
    const int64_t ktCands[4] = {SFA_KT_MAXPLUGIN, 32, 16, SFA_KT_SMALL};

    // Largest key block whose UB footprint fits for this head block. A bigger block puts more of the
    // scattered rows in flight at once, which is what the gather-bound shapes are limited by; it also
    // leaves the per-key vector cost unchanged (every pass is element-rate bound and block-size
    // independent), so this is a free win wherever the UB budget allows it.
    auto pick_kt = [&](int64_t hg) -> int64_t {
        for (int i = 0; i < 4; ++i) {
            const int64_t kt = ktCands[i];
            if (kt > d.topK && kt > SFA_KT_SMALL) {
                continue;
            }
            const int64_t tka = ((d.topK + kt - 1) / kt) * kt;
            if (sfa_footprint(hg, kt, DkAl, DvAl, DmxAl, tka) <= SFA_UB_BUDGET) {
                return kt;
            }
        }
        return SFA_KT_SMALL;
    };

    // The head block is chosen first: it divides the K/V gather work itself, while the key block only
    // hides that gather's latency. Once HG is capped by G the budget it leaves unused goes to KT.
    int64_t HG = 1;
    for (int i = 0; i < 5; ++i) {
        const int64_t kt = pick_kt(hgCands[i]);
        const int64_t tka = ((d.topK + kt - 1) / kt) * kt;
        if (sfa_footprint(hgCands[i], kt, DkAl, DvAl, DmxAl, tka) <= SFA_UB_BUDGET) {
            HG = hgCands[i];
            break;
        }
    }
    if (HG > d.G) {
        HG = d.G;
    }
    const int64_t KT = pick_kt(HG);
    const int64_t numKB = (d.topK + KT - 1) / KT;
    const int64_t topKAl = numKB * KT;
    const int64_t numHeadBlk = (d.G + HG - 1) / HG;
    const int64_t totalTasks = d.B * d.N2 * d.S1 * numHeadBlk;

    SfaParams p{};
    p.qSN = d.qSN; p.qSS = d.qSS; p.qSB = d.qSB;
    p.kSN = d.kSN; p.kSS = d.kSS; p.kSB = d.kSB;
    p.vSN = d.vSN; p.vSS = d.vSS; p.vSB = d.vSB;
    p.iSN = d.iSN; p.iSS = d.iSS; p.iSB = d.iSB;
    p.ySN = ySN;   p.ySS = ySS;   p.ySB = ySB;
    p.B = d.B; p.N1 = d.N1; p.N2 = d.N2; p.S1 = d.S1; p.S2 = d.S2;
    p.Dk = d.Dk; p.Dv = d.Dv; p.topK = d.topK; p.G = d.G;
    p.HG = HG; p.KT = KT; p.numKB = numKB; p.topKAl = topKAl;
    p.DkAl = DkAl; p.DvAl = DvAl; p.DmxAl = DmxAl;
    p.numHeadBlk = numHeadBlk; p.totalTasks = totalTasks;
    p.isCausal = is_causal ? 1 : 0;
    p.s2m1 = d.S2 - d.S1;
    p.scale = static_cast<float>(scaleValue);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto qPtr = reinterpret_cast<GM_ADDR>(query.data_ptr());
    auto kPtr = reinterpret_cast<GM_ADDR>(key.data_ptr());
    auto vPtr = reinterpret_cast<GM_ADDR>(value.data_ptr());
    auto iPtr = reinterpret_cast<GM_ADDR>(sparseIndices.data_ptr());
    auto yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());

    const bool isF16 = (query.scalar_type() == torch::kFloat16);
    auto acl_call = [=]() -> int {
        if (isF16) {
            launch_sfa_half(qPtr, kPtr, vPtr, iPtr, yPtr, p, totalTasks, stream);
        } else {
            launch_sfa_bfloat16(qPtr, kPtr, vPtr, iPtr, yPtr, p, totalTasks, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("SparseFlashAttention", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("sparse_flash_attention", sfa_npu);
}

}  // namespace cann_bench
