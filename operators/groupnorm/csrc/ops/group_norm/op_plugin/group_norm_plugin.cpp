/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file group_norm_plugin.cpp
 * \brief GroupNorm API layer - torch bindings (compiled with g++)
 *
 * Golden semantics = torch.nn.functional.group_norm:
 *   y = (x - mu_g) / sqrt(var_g + eps) * gamma[c] + beta[c]
 * All tensor math happens inside the custom NPU kernels (dual-kernel
 * stats+norm pipeline); workspace is device memory via torch::empty.
 * No host memcpy/item/cpu; no at::/torch:: compute delegation.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/group_norm_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("group_norm(Tensor x, Tensor gamma, Tensor beta, int num_groups, float epsilon) -> Tensor");
}

static torch::Tensor group_norm_meta(const torch::Tensor &x, const torch::Tensor &gamma,
    const torch::Tensor &beta, int64_t num_groups, double epsilon)
{
    (void)epsilon;
    TORCH_CHECK(x.dim() >= 2, "group_norm expects x with dim >= 2.");
    const int64_t C = x.size(1);
    TORCH_CHECK(num_groups >= 1 && C % num_groups == 0,
        "group_norm: num_groups (", num_groups, ") must divide channels (", C, ").");
    TORCH_CHECK(gamma.numel() == C && beta.numel() == C,
        "group_norm: gamma and beta must have ", C, " elements.");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(),
        "group_norm: all inputs must be contiguous.");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
        "group_norm only supports float32/float16/bfloat16, got: ", x.scalar_type());
    TORCH_CHECK(gamma.scalar_type() == x.scalar_type() && beta.scalar_type() == x.scalar_type(),
        "group_norm: gamma/beta dtype must match x.");
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("group_norm", group_norm_meta);
}

static torch::Tensor group_norm_npu(const torch::Tensor &x, const torch::Tensor &gamma,
    const torch::Tensor &beta, int64_t num_groups, double epsilon)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto y = group_norm_meta(x, gamma, beta, num_groups, epsilon);
    const int64_t numel = x.numel();
    if (numel == 0) {
        return y;
    }
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t S = numel / (N * C);

    // Tiling computed on host BEFORE the lambda (captured vars are const).
    GroupNormTilingParams tp = calc_group_norm_tiling(N, C, num_groups, S, x.element_size());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const float eps = (float)epsilon;
    // float GM workspace only needed by the two-kernel path (stats partials)
    auto ws = (tp.modeS != 0)
        ? torch::empty({1}, x.options().dtype(at::kFloat))
        : torch::empty({2 * tp.tasks1}, x.options().dtype(at::kFloat));
    // mode F (z1 solve 8e372571b661a71e...): per-task EXCLUSIVE 32-float
    // (128B) publish regions [sum@0, sumsq@8, arrival@16], zero-padded - one
    // 128B atomic add per task. MUST be zero-initialized on every mode-F
    // launch (declared counter_init stage of the fused mechanism; the
    // arrival-flag and slot semantics rely on the zeroed base).
    auto cnt = (tp.modeF != 0)
        ? torch::zeros({tp.cntFloats}, x.options().dtype(at::kFloat))
        : torch::empty({1}, x.options().dtype(at::kFloat));

    auto acl_call = [=]() -> int {
        GM_ADDR xp = (GM_ADDR)x.data_ptr();
        GM_ADDR gp = (GM_ADDR)gamma.data_ptr();
        GM_ADDR bp = (GM_ADDR)beta.data_ptr();
        GM_ADDR yp = (GM_ADDR)y.data_ptr();
        GM_ADDR wsp = (GM_ADDR)ws.data_ptr();
        if (tp.modeF != 0) {
            // fused single-launch path (z1 solve 8e372571b661a71e...): bounded
            // poll budget 4096 attempts, deterministic local fallback inside
            // the kernel on exhaustion.
            constexpr int64_t GN_F_POLL_BUDGET = 4096;
            GM_ADDR cntp = (GM_ADDR)cnt.data_ptr();
            if (x.scalar_type() == at::kFloat) {
                launch_group_norm_fused_float(xp, gp, bp, yp, cntp, tp.tasksF,
                    tp.P1F, tp.G, tp.Cp, tp.shardLenF, tp.slabLen, tp.S, tp.WF,
                    tp.TILEF, tp.invCnt, eps, GN_F_POLL_BUDGET, tp.blocksF, stream);
            } else if (x.scalar_type() == at::kHalf) {
                launch_group_norm_fused_half(xp, gp, bp, yp, cntp, tp.tasksF,
                    tp.P1F, tp.G, tp.Cp, tp.shardLenF, tp.slabLen, tp.S, tp.WF,
                    tp.TILEF, tp.invCnt, eps, GN_F_POLL_BUDGET, tp.blocksF, stream);
            } else {
                launch_group_norm_fused_bf16(xp, gp, bp, yp, cntp, tp.tasksF,
                    tp.P1F, tp.G, tp.Cp, tp.shardLenF, tp.slabLen, tp.S, tp.WF,
                    tp.TILEF, tp.invCnt, eps, GN_F_POLL_BUDGET, tp.blocksF, stream);
            }
            return 0;
        }
        if (tp.modeS != 0) {
            // single-kernel whole-group path: one launch, no workspace traffic.
            // batch_groups (z1 solve e8fe9640...): tasks = ceil(NG/gpt), each
            // task covers gpt consecutive groups. expS (z2 solve 154df1a3...)
            // selects the non-flat concatenated-segment affine expansion body
            // inside kernel S (public region: c6 only).
            const int64_t NGtotal = N * num_groups;
            if (x.scalar_type() == at::kFloat) {
                launch_group_norm_single_float(xp, gp, bp, yp, tp.tasksS, NGtotal,
                    tp.gpt, tp.G, tp.Cp, tp.slabLen, tp.S, tp.W, tp.TILES, tp.expS,
                    tp.leanS, tp.resS, tp.winS, tp.winCh, tp.invCnt, eps, tp.blocksS, stream);
            } else if (x.scalar_type() == at::kHalf) {
                launch_group_norm_single_half(xp, gp, bp, yp, tp.tasksS, NGtotal,
                    tp.gpt, tp.G, tp.Cp, tp.slabLen, tp.S, tp.W, tp.TILES, tp.expS,
                    tp.leanS, tp.resS, tp.winS, tp.winCh, tp.invCnt, eps, tp.blocksS, stream);
            } else {
                launch_group_norm_single_bf16(xp, gp, bp, yp, tp.tasksS, NGtotal,
                    tp.gpt, tp.G, tp.Cp, tp.slabLen, tp.S, tp.W, tp.TILES, tp.expS,
                    tp.leanS, tp.resS, tp.winS, tp.winCh, tp.invCnt, eps, tp.blocksS, stream);
            }
            return 0;
        }
        if (x.scalar_type() == at::kFloat) {
            launch_group_norm_stats_float(xp, wsp, tp.tasks1, tp.P1, tp.shardLen1,
                tp.slabLen, tp.TILE1, tp.blocks1, stream);
            launch_group_norm_norm_float(xp, gp, bp, yp, wsp, tp.tasks1, tp.tasks2,
                tp.P1, tp.P2, tp.G, tp.Cp, tp.cs, tp.slabLen, tp.S, tp.W, tp.TILE2,
                tp.flat, tp.invCnt, eps, tp.blocks2, stream);
        } else if (x.scalar_type() == at::kHalf) {
            launch_group_norm_stats_half(xp, wsp, tp.tasks1, tp.P1, tp.shardLen1,
                tp.slabLen, tp.TILE1, tp.blocks1, stream);
            launch_group_norm_norm_half(xp, gp, bp, yp, wsp, tp.tasks1, tp.tasks2,
                tp.P1, tp.P2, tp.G, tp.Cp, tp.cs, tp.slabLen, tp.S, tp.W, tp.TILE2,
                tp.flat, tp.invCnt, eps, tp.blocks2, stream);
        } else {
            launch_group_norm_stats_bf16(xp, wsp, tp.tasks1, tp.P1, tp.shardLen1,
                tp.slabLen, tp.TILE1, tp.blocks1, stream);
            launch_group_norm_norm_bf16(xp, gp, bp, yp, wsp, tp.tasks1, tp.tasks2,
                tp.P1, tp.P2, tp.G, tp.Cp, tp.cs, tp.slabLen, tp.S, tp.W, tp.TILE2,
                tp.flat, tp.invCnt, eps, tp.blocks2, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GroupNorm", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("group_norm", group_norm_npu);
}

} // namespace cann_bench
