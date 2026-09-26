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
 * \file maximum_plugin.cpp
 * \brief Maximum API layer - torch bindings (compiled with g++)
 *
 * The host side only performs shape / stride bookkeeping (pure integer arithmetic); every
 * output element is produced inside the custom NPU kernels.  When the second operand cannot
 * be expressed as a periodic block, it is materialised on the device by the helper kernels.
 */

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/maximum_launch.h"

namespace cann_bench {

namespace {

constexpr int64_t kMaxDims = 8;

std::vector<int64_t> maximum_broadcast_shape(const torch::Tensor &x1, const torch::Tensor &x2)
{
    const int64_t r = std::max(x1.dim(), x2.dim());
    std::vector<int64_t> shape(static_cast<size_t>(r), 1);
    for (int64_t i = 0; i < r; ++i) {
        const int64_t s1 = (i < r - x1.dim()) ? 1 : x1.size(i - (r - x1.dim()));
        const int64_t s2 = (i < r - x2.dim()) ? 1 : x2.size(i - (r - x2.dim()));
        TORCH_CHECK(s1 == s2 || s1 == 1 || s2 == 1, "maximum: input shapes are not broadcastable");
        shape[static_cast<size_t>(i)] = std::max(s1, s2);
    }
    return shape;
}

// dtypeCode: 0 fp16, 1 bf16, 2 fp32, 3 int8, 4 int32, 5 int64
int64_t maximum_dtype_code(const c10::ScalarType st)
{
    switch (st) {
        case torch::kFloat16: return 0;
        case torch::kBFloat16: return 1;
        case torch::kFloat32: return 2;
        case torch::kInt8: return 3;
        case torch::kInt32: return 4;
        case torch::kInt64: return 5;
        default: return -1;
    }
}

// The replication helper keeps one inner run in UB, so a plan whose run block does not fit is
// routed to the per-element broadcast fallback instead of overflowing the UB allocation.
constexpr int64_t kExpandMaxInner = 8192;
constexpr int64_t kExpandMaxLc = 8192;

struct MaximumPlan {
    bool fullA = false;   // A must be materialised to the output shape on the device
    bool fullB = false;   // B must be materialised to the output shape on the device
    bool expandB = false; // B's run block must be materialised (per-run replication)
    bool swapAB = false;  // the owning operand is x2
    int64_t numel = 0;
    int64_t runLen = 1;
    int64_t rows = 1;
    int64_t lc = 1;
    int64_t linner = 1;
    std::vector<int64_t> aSizes; // right aligned sizes of the owning operand
    std::vector<int64_t> bSizes; // right aligned sizes of the periodic operand
    std::vector<int64_t> outSizes;
};

MaximumPlan maximum_build_plan(const torch::Tensor &x1, const torch::Tensor &x2)
{
    MaximumPlan p;
    p.outSizes = maximum_broadcast_shape(x1, x2);
    int64_t r = static_cast<int64_t>(p.outSizes.size());
    if (r == 0) {
        p.outSizes.push_back(1);
        r = 1;
    }
    p.numel = 1;
    for (int64_t d = 0; d < r; ++d) {
        p.numel *= p.outSizes[static_cast<size_t>(d)];
    }
    std::vector<int64_t> as(static_cast<size_t>(r), 1);
    std::vector<int64_t> bs(static_cast<size_t>(r), 1);
    for (int64_t i = 0; i < x1.dim(); ++i) {
        as[static_cast<size_t>(r - x1.dim() + i)] = x1.size(i);
    }
    for (int64_t i = 0; i < x2.dim(); ++i) {
        bs[static_cast<size_t>(r - x2.dim() + i)] = x2.size(i);
    }
    bool aOwns = true;
    bool bOwns = true;
    for (int64_t d = 0; d < r; ++d) {
        if (as[static_cast<size_t>(d)] != p.outSizes[static_cast<size_t>(d)]) {
            aOwns = false;
        }
        if (bs[static_cast<size_t>(d)] != p.outSizes[static_cast<size_t>(d)]) {
            bOwns = false;
        }
    }
    if (!aOwns) {
        if (bOwns) {
            p.swapAB = true;
            std::swap(as, bs);
        } else {
            p.fullA = true;
            p.fullB = true;
        }
    }
    p.aSizes = as;
    p.bSizes = bs;
    if (p.fullA || p.fullB) {
        p.runLen = p.numel;
        p.rows = 1;
        return p;
    }

    // Locate the innermost run: the outermost dimension c0 from which the periodic operand
    // owns a non-trivial extent, with everything outside being broadcast.
    int64_t c0 = -1;
    bool expand = false;
    int64_t lc = 1;
    int64_t linner = 1;
    for (int64_t c = 0; c < r; ++c) {
        bool ok = true;
        for (int64_t d = 0; d < c; ++d) {
            if (p.outSizes[static_cast<size_t>(d)] > 1 && bs[static_cast<size_t>(d)] > 1) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }
        bool seenFree = false;
        bool good = true;
        bool anyOwned = false;
        int64_t lastOwned = -1;
        for (int64_t d = c; d < r; ++d) {
            if (p.outSizes[static_cast<size_t>(d)] <= 1) {
                continue;
            }
            if (bs[static_cast<size_t>(d)] > 1) {
                anyOwned = true;
                if (seenFree) {
                    good = false;
                    break;
                }
                lastOwned = d;
            } else {
                seenFree = true;
            }
        }
        if (!good || !anyOwned) {
            continue;
        }
        c0 = c;
        if (seenFree) {
            expand = true;
            for (int64_t d = c0; d <= lastOwned; ++d) {
                lc *= p.outSizes[static_cast<size_t>(d)];
            }
            for (int64_t d = lastOwned + 1; d < r; ++d) {
                linner *= p.outSizes[static_cast<size_t>(d)];
            }
        }
        break;
    }
    if (c0 < 0) {
        p.fullB = true;
        p.runLen = p.numel;
        p.rows = 1;
        return p;
    }
    if (expand && linner <= kExpandMaxInner && lc <= kExpandMaxLc) {
        p.expandB = true;
        p.lc = lc;
        p.linner = linner;
        p.runLen = lc * linner;
    } else if (expand) {
        // Run block too large for the replication helper's UB buffers: materialise the operand
        // with the per-element broadcast kernel and run the flat max.
        p.fullB = true;
        p.runLen = p.numel;
        p.rows = 1;
        return p;
    } else {
        int64_t rl = 1;
        for (int64_t d = c0; d < r; ++d) {
            rl *= p.outSizes[static_cast<size_t>(d)];
        }
        p.runLen = rl;
    }
    p.rows = (p.runLen > 0) ? (p.numel / p.runLen) : 1;
    return p;
}

void maximum_launch_expand(int64_t esz, GM_ADDR s, GM_ADDR d, int64_t lc, int64_t linner,
                           int64_t nb, void *st)
{
    if (esz == 1) {
        launch_maximum_expand_b8(s, d, lc, linner, nb, st);
    } else if (esz == 2) {
        launch_maximum_expand_b16(s, d, lc, linner, nb, st);
    } else if (esz == 4) {
        launch_maximum_expand_b32(s, d, lc, linner, nb, st);
    } else {
        launch_maximum_expand_b64(s, d, lc, linner, nb, st);
    }
}

void maximum_launch_bcast(int64_t esz, GM_ADDR s, GM_ADDR d, int64_t numel, int32_t r,
                          const int64_t *pk, void *st)
{
    if (esz == 1) {
        launch_maximum_bcast_b8(s, d, numel, r, pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6],
                                pk[7], st);
    } else if (esz == 2) {
        launch_maximum_bcast_b16(s, d, numel, r, pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6],
                                 pk[7], st);
    } else if (esz == 4) {
        launch_maximum_bcast_b32(s, d, numel, r, pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6],
                                 pk[7], st);
    } else {
        launch_maximum_bcast_b64(s, d, numel, r, pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6],
                                 pk[7], st);
    }
}

void maximum_launch_main(int64_t code, GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t rows,
                         int64_t runLen, int64_t tile, int64_t tilesPerRow, int64_t nb, void *st)
{
    switch (code) {
        case 0:
            launch_maximum_fp16(a, b, y, rows, runLen, tile, tilesPerRow, nb, st);
            break;
        case 1:
            launch_maximum_bf16(a, b, y, rows, runLen, tile, tilesPerRow, nb, st);
            break;
        case 2:
            launch_maximum_fp32(a, b, y, rows, runLen, tile, tilesPerRow, nb, st);
            break;
        case 3:
            launch_maximum_int8(a, b, y, rows, runLen, tile, tilesPerRow, nb, st);
            break;
        case 4:
            launch_maximum_int32(a, b, y, rows, runLen, tile, tilesPerRow, nb, st);
            break;
        default:
            launch_maximum_int64(a, b, y, rows, runLen, tile, tilesPerRow, nb, st);
            break;
    }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("maximum(Tensor x1, Tensor x2) -> Tensor");
}

torch::Tensor maximum_meta(const torch::Tensor &x1, const torch::Tensor &x2)
{
    auto shape = maximum_broadcast_shape(x1, x2);
    return torch::empty(shape, x1.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("maximum", maximum_meta);
}

torch::Tensor maximum_npu(const torch::Tensor &x1, const torch::Tensor &x2)
{
    const auto st = x1.scalar_type();
    TORCH_CHECK(st == x2.scalar_type(), "maximum: x1 and x2 must share one dtype");
    const int64_t code = maximum_dtype_code(st);
    TORCH_CHECK(code >= 0, "maximum: unsupported dtype ", st);
    TORCH_CHECK(x1.dim() <= kMaxDims && x2.dim() <= kMaxDims, "maximum: rank above ", kMaxDims,
                " is not supported");
    const c10::OptionalDeviceGuard guard(x1.device());

    auto a = x1.contiguous();
    auto b = x2.contiguous();
    MaximumPlan plan = maximum_build_plan(a, b);
    if (plan.swapAB) {
        std::swap(a, b);
    }
    auto out = torch::empty(plan.outSizes, a.options());
    if (plan.numel == 0) {
        return out;
    }
    const int64_t esz = a.element_size();
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    torch::Tensor aWork;
    torch::Tensor bWork;
    torch::Tensor bBlk;
    bool aWorkTouched = false;
    bool bWorkTouched = false;
    bool bBlkTouched = false;

    if (plan.fullA || plan.fullB) {
        if (plan.fullA) {
            aWork = torch::empty(plan.outSizes, a.options());
            aWorkTouched = true;
        }
        if (plan.fullB) {
            bWork = torch::empty(plan.outSizes, b.options());
            bWorkTouched = true;
        }
    } else if (plan.expandB) {
        bBlk = torch::empty({plan.runLen}, b.options());
        bBlkTouched = true;
    }

    auto a_ptr = (GM_ADDR)a.data_ptr();
    auto b_ptr = (GM_ADDR)b.data_ptr();
    auto y_ptr = (GM_ADDR)out.data_ptr();
    auto aWork_ptr = aWorkTouched ? (GM_ADDR)aWork.data_ptr() : nullptr;
    auto bWork_ptr = bWorkTouched ? (GM_ADDR)bWork.data_ptr() : nullptr;
    auto bBlk_ptr = bBlkTouched ? (GM_ADDR)bBlk.data_ptr() : nullptr;

    std::vector<int64_t> pkA(static_cast<size_t>(kMaxDims), 1);
    std::vector<int64_t> pkB(static_cast<size_t>(kMaxDims), 1);
    const int32_t r = static_cast<int32_t>(plan.outSizes.size());
    for (int32_t d = 0; d < r; ++d) {
        const uint64_t os = static_cast<uint64_t>(plan.outSizes[static_cast<size_t>(d)]);
        pkA[static_cast<size_t>(d)] =
            static_cast<int64_t>((os << 32) | (static_cast<uint64_t>(plan.aSizes[static_cast<size_t>(d)]) & 0xFFFFFFFFu));
        pkB[static_cast<size_t>(d)] =
            static_cast<int64_t>((os << 32) | (static_cast<uint64_t>(plan.bSizes[static_cast<size_t>(d)]) & 0xFFFFFFFFu));
    }

    int64_t tileElems = 0;
    int64_t numBlocks = 0;
    int64_t ubSize = 0;
    std::tie(tileElems, numBlocks, ubSize) =
        calc_maximum_tiling_params(plan.numel, plan.rows, plan.runLen, esz, code);
    const int64_t tilesPerRow = (plan.runLen + tileElems - 1) / tileElems;
    const int64_t expandBlocks = std::min<int64_t>(std::max<int64_t>(plan.lc, 1), 40);

    auto acl_call = [=]() -> int {
        if (plan.fullA) {
            maximum_launch_bcast(esz, a_ptr, aWork_ptr, plan.numel, r, pkA.data(), stream);
        }
        if (plan.fullB) {
            maximum_launch_bcast(esz, b_ptr, bWork_ptr, plan.numel, r, pkB.data(), stream);
        }
        GM_ADDR aSrc = plan.fullA ? aWork_ptr : a_ptr;
        GM_ADDR bSrc = plan.fullB ? bWork_ptr : b_ptr;
        if (plan.expandB) {
            maximum_launch_expand(esz, bSrc, bBlk_ptr, plan.lc, plan.linner, expandBlocks, stream);
            bSrc = bBlk_ptr;
        }
        maximum_launch_main(code, aSrc, bSrc, y_ptr, plan.rows, plan.runLen, tileElems,
                            tilesPerRow, numBlocks, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Maximum", acl_call);
    (void)ubSize;
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("maximum", maximum_npu);
}

} // namespace cann_bench
