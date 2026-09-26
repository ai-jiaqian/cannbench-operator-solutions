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
 * \file gather_plugin.cpp
 * \brief Gather (torch.gather semantics) python bindings + tiling (compiled with g++)
 *
 * Host work is metadata-only: shapes/strides/dtypes and the launch/tiling selection.
 */

#include <vector>
#include <cstdint>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "platform/platform_ascendc.h"

#include "../op_kernel/gather_launch.h"

namespace cann_bench {

namespace {

constexpr int64_t G_MAX_NDIM = 8;
constexpr int64_t G_MAX_BLOCKS = 2048; // keep multi-block DataCopyPad blockCount within the a2 limit
constexpr int64_t G_RB_MIN = 8;        // rows per inner iteration to keep the per-iteration cost amortized

inline int64_t gcd64(int64_t a, int64_t b)
{
    if (a < 0) { a = -a; }
    if (b < 0) { b = -b; }
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

inline int64_t lcm64(int64_t a, int64_t b)
{
    if (a == 0 || b == 0) { return 0; }
    return a / gcd64(a, b) * b;
}

struct GatherPlan {
    bool useFast = false;
    bool useFast64 = false;
    int64_t n = 0;
    int64_t dim = 0;
    int64_t xs[G_MAX_NDIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t os[G_MAX_NDIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t xMid = 1;
    int64_t xRowStride = 1;
    int64_t xNumel = 1;
    int64_t totalOut = 0;
    int64_t outerOut = 1;
    int64_t mid = 1;
    int64_t innerOut = 1;
    int64_t numBlocks = 1;
    // fast path
    int64_t Cc = 1;
    int64_t RB = 1;
    int64_t unitsPerA = 1;
    int64_t numTasks = 1;
    // slow path (always tiled independently of the fast path)
    int64_t slowBlocks = 1;
    int64_t slowBlockLength = 1;
    int64_t slowTileElems = 1;
};

// Choose the windowed-gather tiling (chunk width Cc, rows per inner iteration RB).
// The per-chunk DMA block count is (xMid + 2*mid) regardless of the chunk width, so a wider
// chunk (fewer chunks) is cheaper as long as the grid still fills the vector cores: aim for
// roughly one chunk per core. On top of that, the chunk is capped so that at least G_RB_MIN rows
// fit per inner iteration, otherwise a chunk whose window nearly fills UB leaves a pathologically
// small RB and the per-iteration (barrier/DMA) cost dominates.
bool plan_windowed(GatherPlan &p, int64_t uB, int64_t iB, int64_t perElem, int64_t coreNum,
                   int64_t ubBudget)
{
    if (p.innerOut < 1 || p.xMid <= 0 || p.mid <= 0 || ubBudget <= 0) { return false; }
    const int64_t g = lcm64(32 / gcd64(uB, 32), 32 / gcd64(iB, 32));
    if (g <= 0) { return false; }
    const int64_t CcMax = ubBudget / (p.xMid * uB + perElem);
    if (CcMax < 1) { return false; }

    const bool alignedW = ((p.innerOut * uB) % 32 == 0 && (p.innerOut * iB) % 32 == 0);
    int64_t Cc = 0;
    if (p.innerOut <= CcMax && (alignedW || p.innerOut == 1)) {
        Cc = p.innerOut; // whole inner extent in one chunk: block copies stay contiguous
    } else {
        int64_t CcAl = (CcMax / g) * g;
        if (CcAl >= p.innerOut) { CcAl = ((p.innerOut - 1) / g) * g; }
        if (CcAl >= g) {
            int64_t unitsTarget = (coreNum + p.outerOut - 1) / p.outerOut;
            if (unitsTarget < 1) { unitsTarget = 1; }
            if (unitsTarget > p.innerOut) { unitsTarget = p.innerOut; }
            int64_t CcT = (p.innerOut + unitsTarget - 1) / unitsTarget;
            CcT = ((CcT + g - 1) / g) * g;
            Cc = (CcT < CcAl) ? CcT : CcAl;
            if (Cc >= p.innerOut) { Cc = CcAl; }
            if (p.mid > G_RB_MIN) {
                int64_t CcRb = ubBudget / (p.xMid * uB + perElem * G_RB_MIN);
                int64_t CcRbAl = (CcRb / g) * g;
                if (CcRbAl >= g && CcRbAl < Cc) { Cc = CcRbAl; }
            }
            if (Cc < g) { Cc = 0; }
        }
    }
    if (Cc < 1) { return false; }

    const int64_t avail = ubBudget - p.xMid * Cc * uB;
    if (perElem * Cc <= 0) { return false; }
    int64_t RB = avail / (perElem * Cc);
    if (RB > p.mid) { RB = p.mid; }
    if (RB > G_MAX_BLOCKS) { RB = G_MAX_BLOCKS; }
    if (RB < 1) { return false; }

    p.Cc = Cc;
    p.RB = RB;
    p.unitsPerA = (p.innerOut + Cc - 1) / Cc;
    p.numTasks = p.outerOut * p.unitsPerA;
    p.numBlocks = (coreNum < p.numTasks) ? coreNum : p.numTasks;
    return p.numBlocks >= 1;
}

// Tiling decisions are recomputed on every call; nothing is cached across calls.
void build_plan(const torch::Tensor &x, const torch::Tensor &index, int64_t dim, GatherPlan &p)
{
    const int64_t n = x.dim();
    p.n = n;
    p.dim = dim;
    p.totalOut = index.numel();
    if (p.totalOut <= 0 || n <= 0) { return; }

    for (int64_t j = 0; j < n; ++j) {
        p.os[j] = index.size(j);
    }
    int64_t acc = 1;
    for (int64_t j = n - 1; j >= 0; --j) {
        p.xs[j] = acc;
        acc *= x.size(j);
    }
    for (int64_t j = n; j < G_MAX_NDIM; ++j) {
        p.xs[j] = 0;
        p.os[j] = 1;
    }

    p.xMid = x.size(dim);
    p.xNumel = x.numel();
    p.mid = p.os[dim];
    p.innerOut = 1;
    for (int64_t j = dim + 1; j < n; ++j) { p.innerOut *= p.os[j]; }
    p.outerOut = 1;
    for (int64_t j = 0; j < dim; ++j) { p.outerOut *= p.os[j]; }
    p.xRowStride = p.xs[dim];

    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    if (plat != nullptr) { plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize); }
    int64_t coreNum = (plat != nullptr) ? static_cast<int64_t>(plat->GetCoreNumAiv()) : 0;
    if (coreNum <= 0) { coreNum = 1; }
    if (ubSize == 0) { ubSize = 192 * 1024; }

    const int64_t uB = static_cast<int64_t>(x.element_size());
    const int64_t iB = static_cast<int64_t>(index.element_size());

    // ---- slow path tiling (always computed; used for int8 index / ineligible shapes) ----
    {
        // Fine granularity: every core should get work even for small outputs (a small output
        // would otherwise land on a single core because of the per-element DMA cost).
        const int64_t MIN_ELEMS_PER_CORE = 16;
        int64_t nb = (p.totalOut + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
        if (nb > coreNum) { nb = coreNum; }
        if (nb < 1) { nb = 1; }
        p.slowBlocks = nb;
        p.slowBlockLength = (p.totalOut + nb - 1) / nb;
        int64_t tile = 4096;
        const int64_t maxTileBuf = 32768;
        if (tile * iB > maxTileBuf) { tile = maxTileBuf / iB; }
        if (tile > p.totalOut) { tile = p.totalOut; }
        if (tile < 1) { tile = 1; }
        p.slowTileElems = tile;
    }

    // ---- windowed-path eligibility (metadata only) ----
    bool base = true;
    int64_t s = 1;
    for (int64_t j = n - 1; j >= 0; --j) {
        if (x.stride(j) != s) { base = false; }
        s *= x.size(j);
    }
    s = 1;
    for (int64_t j = n - 1; j >= 0; --j) {
        if (index.stride(j) != s) { base = false; }
        s *= index.size(j);
    }
    if (p.xMid > 65535) { base = false; }
    for (int64_t j = dim + 1; j < n; ++j) {
        if (p.os[j] != x.size(j)) { base = false; } // needs index.size == x.size below dim
    }
    const bool okIdx = (index.scalar_type() == torch::kInt ||
                        index.scalar_type() == torch::kLong);
    const bool isI64 = (x.scalar_type() == torch::kLong);

    const int64_t ubBudget = static_cast<int64_t>(ubSize) - 16384;
    if (base && okIdx && !isI64) {
        // buffers: index + out + offsets + 3 patterns
        p.useFast = plan_windowed(p, uB, iB, iB + uB + 16, coreNum, ubBudget);
    }
    if (base && okIdx && isI64 && !p.useFast) {
        // paired path: index + duplicated index + offsets + out + 3 patterns (all int32 sized)
        p.useFast64 = plan_windowed(p, 8, iB, iB + 40, coreNum, ubBudget);
    }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("gather(Tensor x, Tensor index, int dim=0) -> Tensor");
}

torch::Tensor gather_meta(const torch::Tensor &x, const torch::Tensor &index, int64_t dim)
{
    TORCH_CHECK(x.dim() == index.dim(), "x and index must have the same number of dimensions.");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension.");
    TORCH_CHECK(dim >= -x.dim() && dim < x.dim(), "dim out of range.");
    return torch::empty(index.sizes(), x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("gather", gather_meta);
}

#define G_DO_SLOW(SUF)                                                                         \
    gather_launch_slow2_##SUF(                                                                 \
        x_ptr, ix_ptr, y_ptr, plan.n, plan.dim,                                                \
        plan.xs[0], plan.xs[1], plan.xs[2], plan.xs[3],                                        \
        plan.xs[4], plan.xs[5], plan.xs[6], plan.xs[7],                                        \
        plan.os[0], plan.os[1], plan.os[2], plan.os[3],                                        \
        plan.os[4], plan.os[5], plan.os[6], plan.os[7],                                        \
        plan.xMid, plan.xNumel, plan.totalOut, plan.slowBlockLength,                           \
        plan.slowTileElems, plan.slowBlocks, stream)

#define G_DO_FAST(SUF)                                                                         \
    gather_launch_fast_##SUF(                                                                  \
        x_ptr, ix_ptr, y_ptr, plan.n, plan.dim,                                                \
        plan.xs[0], plan.xs[1], plan.xs[2], plan.xs[3],                                        \
        plan.xs[4], plan.xs[5], plan.xs[6], plan.xs[7],                                        \
        plan.os[0], plan.os[1], plan.os[2], plan.os[3],                                        \
        plan.os[4], plan.os[5], plan.os[6], plan.os[7],                                        \
        plan.xMid, plan.xRowStride, plan.outerOut, plan.mid, plan.innerOut,                    \
        plan.Cc, plan.RB, plan.unitsPerA, plan.numTasks, plan.numBlocks, stream)

#define G_DO_FAST64(SUF)                                                                       \
    gather_launch_fast64_##SUF(                                                                \
        x_ptr, ix_ptr, y_ptr, plan.n, plan.dim,                                                \
        plan.xs[0], plan.xs[1], plan.xs[2], plan.xs[3],                                        \
        plan.xs[4], plan.xs[5], plan.xs[6], plan.xs[7],                                        \
        plan.os[0], plan.os[1], plan.os[2], plan.os[3],                                        \
        plan.os[4], plan.os[5], plan.os[6], plan.os[7],                                        \
        plan.xMid, plan.xRowStride, plan.outerOut, plan.mid, plan.innerOut,                    \
        plan.Cc, plan.RB, plan.unitsPerA, plan.numTasks, plan.numBlocks, stream)

#define G_DO(SUF)                                                                              \
    do {                                                                                       \
        if (plan.useFast) { G_DO_FAST(SUF); } else { G_DO_SLOW(SUF); }                          \
    } while (0)

torch::Tensor gather_npu(const torch::Tensor &x, const torch::Tensor &index, int64_t dim)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto y = gather_meta(x, index, dim);
    if (dim < 0) { dim += x.dim(); }

    GatherPlan plan;
    build_plan(x, index, dim, plan);
    if (plan.totalOut <= 0) { return y; }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto ix_ptr = (GM_ADDR)index.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    const auto xdtype = x.scalar_type();
    const auto idtype = index.scalar_type();

    auto acl_call = [=]() -> int {
        if (xdtype == torch::kFloat32) {
            if (idtype == torch::kChar)       { G_DO_SLOW(f32_i8); }
            else if (idtype == torch::kInt)   { G_DO(f32_i32); }
            else if (idtype == torch::kLong)  { G_DO(f32_i64); }
            else { TORCH_CHECK(false, "gather: unsupported index dtype for float32 x"); }
        } else if (xdtype == torch::kFloat16) {
            if (idtype == torch::kChar)       { G_DO_SLOW(f16_i8); }
            else if (idtype == torch::kInt)   { G_DO(f16_i32); }
            else if (idtype == torch::kLong)  { G_DO(f16_i64); }
            else { TORCH_CHECK(false, "gather: unsupported index dtype for float16 x"); }
        } else if (xdtype == torch::kBFloat16) {
            if (idtype == torch::kChar)       { G_DO_SLOW(bf16_i8); }
            else if (idtype == torch::kInt)   { G_DO(bf16_i32); }
            else if (idtype == torch::kLong)  { G_DO(bf16_i64); }
            else { TORCH_CHECK(false, "gather: unsupported index dtype for bfloat16 x"); }
        } else if (xdtype == torch::kChar) {
            if (idtype == torch::kChar)       { G_DO_SLOW(i8_i8); }
            else if (idtype == torch::kInt)   { G_DO(i8_i32); }
            else if (idtype == torch::kLong)  { G_DO(i8_i64); }
            else { TORCH_CHECK(false, "gather: unsupported index dtype for int8 x"); }
        } else if (xdtype == torch::kInt) {
            if (idtype == torch::kChar)       { G_DO_SLOW(i32_i8); }
            else if (idtype == torch::kInt)   { G_DO(i32_i32); }
            else if (idtype == torch::kLong)  { G_DO(i32_i64); }
            else { TORCH_CHECK(false, "gather: unsupported index dtype for int32 x"); }
        } else if (xdtype == torch::kLong) {
            if (idtype == torch::kChar)       { G_DO_SLOW(i64_i8); }
            else if (idtype == torch::kInt)   { if (plan.useFast64) { G_DO_FAST64(i32); } else { G_DO_SLOW(i64_i32); } }
            else if (idtype == torch::kLong)  { if (plan.useFast64) { G_DO_FAST64(i64); } else { G_DO_SLOW(i64_i64); } }
            else { TORCH_CHECK(false, "gather: unsupported index dtype for int64 x"); }
        } else {
            TORCH_CHECK(false, "gather: unsupported x dtype");
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Gather", acl_call);

    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("gather", gather_npu);
}

} // namespace cann_bench
