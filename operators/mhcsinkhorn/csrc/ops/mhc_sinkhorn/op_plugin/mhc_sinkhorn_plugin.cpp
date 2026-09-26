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
 * \file mhc_sinkhorn_plugin.cpp
 * \brief MhcSinkhorn API layer - torch bindings (compiled with g++)
 *
 * Everything on the execution path is device side: the output tensor is
 * allocated on the current NPU stream from the input metadata, the tiling is
 * derived from the shape only, and the kernel is launched through
 * OpCommand::RunOpApi.  No host readback, no host arithmetic.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/mhc_sinkhorn_launch.h"
#include "../op_kernel/mhc_diag.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("mhc_sinkhorn(Tensor comb, int iter_step=20, float eps=1e-6) -> Tensor comb_out");
}

torch::Tensor mhc_sinkhorn_meta(const torch::Tensor &comb, int64_t iterStep, double eps)
{
    (void)iterStep;
    (void)eps;
    return torch::empty_like(comb);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("mhc_sinkhorn", mhc_sinkhorn_meta);
}

torch::Tensor mhc_sinkhorn_npu(const torch::Tensor &comb, int64_t iterStep, double eps)
{
#if MHC_DIAG_SKIP_LAUNCH
    (void)iterStep;
    (void)eps;
    return comb;
#else
    const c10::OptionalDeviceGuard guard(comb.device());
    TORCH_CHECK(comb.dim() == 3, "mhc_sinkhorn: comb must be a 3D tensor [B, hc_mult, hc_mult].");
    TORCH_CHECK(comb.scalar_type() == torch::kFloat32, "mhc_sinkhorn: only float32 is supported.");
    TORCH_CHECK(comb.is_contiguous(), "mhc_sinkhorn: comb must be contiguous.");
    const int64_t hcMult = comb.size(1);
    TORCH_CHECK(comb.size(2) == hcMult, "mhc_sinkhorn: the two inner dimensions must be equal.");
    TORCH_CHECK(hcMult >= 1 && hcMult <= 16, "mhc_sinkhorn: hc_mult must be in [1, 16].");
    TORCH_CHECK(iterStep >= 1, "mhc_sinkhorn: iter_step must be >= 1.");

    auto out = mhc_sinkhorn_meta(comb, iterStep, eps);
    const int64_t totalMats = comb.size(0);
    if (comb.numel() <= 0) {
        return out;
    }

    int64_t numBlocks, matPerCore, tileMats;
    std::tie(numBlocks, matPerCore, tileMats) =
        calc_mhc_sinkhorn_tiling_params(totalMats, hcMult);

    auto xPtr = (GM_ADDR)comb.data_ptr();
    auto yPtr = (GM_ADDR)out.data_ptr();
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const float epsF = (float)eps;
    const int32_t mI = (int32_t)hcMult;
    const int32_t itI = (int32_t)iterStep;
    const int32_t tileI = (int32_t)tileMats;
    const int64_t nblk = numBlocks;
    const int64_t mpc = matPerCore;

    auto acl_call = [=]() -> int {
        launch_mhc_sinkhorn(xPtr, yPtr, totalMats, mI, itI, epsF, nblk, mpc, tileI, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MhcSinkhorn", acl_call);
    return out;
#endif
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("mhc_sinkhorn", mhc_sinkhorn_npu);
}

} // namespace cann_bench
