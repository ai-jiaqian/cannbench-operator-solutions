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
 * \file wqmm_plugin.cpp
 * \brief WeightQuantBatchMatmul API layer - torch bindings (compiled with g++)
 *
 * Device pipeline (three steps on one stream):
 *   1. wqmm_dequant (AIV) writes ANTIQUANT(weight) into a T workspace,
 *   2. wqmm_mm (pure cube) computes y = x @ workspace + bias and stores it directly in T.
 *
 * The host-side tiling is produced by calc_wqmm_plan, which lives in the bisheng translation unit so that
 * both sides use the very same TCubeTiling type.
 */

#include <tuple>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/wqmm_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("weight_quant_batch_matmul(Tensor x, Tensor weight, Tensor antiquantScale, "
          "Tensor? antiquantOffset=None, Tensor? bias=None) -> Tensor");
}

/* TEMPORARY diagnostic switch: when 1 the plugin dumps the generated cube tiling for every case and
 * aborts the call.  Set back to 0 (and rebuilt) once the shape/tiling table has been read off. */
#define WQMM_DUMP_TILING 0

static torch::Tensor wqmm_meta(const torch::Tensor &x, const torch::Tensor &weight,
                               const torch::Tensor &antiquantScale,
                               const c10::optional<torch::Tensor> &antiquantOffset,
                               const c10::optional<torch::Tensor> &bias)
{
    (void)antiquantScale;
    (void)antiquantOffset;
    (void)bias;
    TORCH_CHECK(x.dim() == 2, "weight_quant_batch_matmul requires a 2D x [M, K].");
    TORCH_CHECK(weight.dim() == 2, "weight_quant_batch_matmul requires a 2D weight [K, N].");
    TORCH_CHECK(x.size(1) == weight.size(0), "K dimensions of x and weight must match.");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "weight_quant_batch_matmul supports float16 / bfloat16 x.");
    TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8.");
    return torch::empty({x.size(0), weight.size(1)}, x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("weight_quant_batch_matmul", wqmm_meta);
}

static torch::Tensor wqmm_npu(const torch::Tensor &x, const torch::Tensor &weight,
                              const torch::Tensor &antiquantScale,
                              const c10::optional<torch::Tensor> &antiquantOffset,
                              const c10::optional<torch::Tensor> &bias)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto y = wqmm_meta(x, weight, antiquantScale, antiquantOffset, bias);

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight.size(1);
    if (M <= 0 || N <= 0 || K <= 0) {
        return y;
    }

    const bool isFp16 = (x.scalar_type() == at::kHalf);
    const bool hasOff = antiquantOffset.has_value() && antiquantOffset->defined();
    const bool hasBias = bias.has_value() && bias->defined();
    const int64_t hasOffI = hasOff ? 1 : 0;

    WqmmTilingPOD tp;
    int64_t dqCores = 1;
    int64_t dqRows = 1;
    int64_t dqCol = 1;
    int64_t sysWs = 0;
    if (calc_wqmm_plan(M, N, K, isFp16 ? 0 : 1, hasBias ? 1 : 0, 0, &tp,
                       &dqCores, &dqRows, &dqCol, &sysWs) != 0) {
        TORCH_CHECK(false, "weight_quant_batch_matmul: tiling failed code=", tp.diag[0],
                    " aic=", tp.diag[1], " nSlab=", tp.diag[2], " szCube=", tp.diag[3],
                    " szPOD=", tp.diag[4], " nSlabLast=", tp.diag[6], " attempts=", tp.diag[7],
                    " hM=", tp.diag[8], " hN=", tp.diag[9], " hKa=", tp.diag[10], " hKb=", tp.diag[11],
                    " hScM=", tp.diag[12], " hScN=", tp.diag[13], " hUsed=", tp.diag[14],
                    " M=", M, " N=", N, " K=", K);
    }
    const int64_t numBlocks = static_cast<int64_t>(tp.mBlocks) * static_cast<int64_t>(tp.nBlocks);
#if WQMM_DUMP_TILING
    TORCH_CHECK(false, "WQMMDUMP",
                " M=", M, " N=", N, " K=", K, " fp16=", isFp16 ? 1 : 0, " bias=", hasBias ? 1 : 0,
                " | aic=", tp.diag[1], " nslab=", tp.diag[2], " bM=", tp.diag[3], " bN=", tp.diag[4],
                " bK=", tp.diag[5], " scN=", tp.diag[6], " scK=", tp.diag[7], " dA1=", tp.diag[8],
                " dB1=", tp.diag[9], " stKa=", tp.diag[10], " stKb=", tp.diag[11], " stM=", tp.diag[12],
                " stN=", tp.diag[13], " used=", tp.diag[14], " transLen=", tp.diag[15],
                " | mb=", tp.mBlocks, " nb=", tp.nBlocks, " dqCores=", dqCores, " dqRows=", dqRows,
                " dqCol=", dqCol);
#endif

    auto xc = x.contiguous();
    auto wc = weight.contiguous();
    auto sc = antiquantScale.contiguous();
    auto wd = torch::empty({K, N}, xc.options());
    auto ws = torch::empty({sysWs > 0 ? sysWs : 1}, xc.options().dtype(at::kByte));

    torch::Tensor oc;
    torch::Tensor bc;
    GM_ADDR oPtr = nullptr;
    GM_ADDR bPtr = nullptr;
    if (hasOff) {
        oc = antiquantOffset->contiguous();
        oPtr = reinterpret_cast<GM_ADDR>(oc.data_ptr());
    }
    if (hasBias) {
        bc = bias->contiguous();
        bPtr = reinterpret_cast<GM_ADDR>(bc.data_ptr());
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    GM_ADDR xPtr = reinterpret_cast<GM_ADDR>(xc.data_ptr());
    GM_ADDR wPtr = reinterpret_cast<GM_ADDR>(wc.data_ptr());
    GM_ADDR sPtr = reinterpret_cast<GM_ADDR>(sc.data_ptr());
    GM_ADDR wdPtr = reinterpret_cast<GM_ADDR>(wd.data_ptr());
    GM_ADDR wsPtr = reinterpret_cast<GM_ADDR>(ws.data_ptr());
    GM_ADDR yPtr = reinterpret_cast<GM_ADDR>(y.data_ptr());

    auto acl_call = [=]() -> int {
        if (isFp16) {
            launch_wqmm_dequant_half(wPtr, sPtr, oPtr, wdPtr, K, N, hasOffI, dqCores, dqRows, dqCol, stream);
            launch_wqmm_mm_half(xPtr, wdPtr, bPtr, yPtr, wsPtr, tp, numBlocks, stream);
        } else {
            launch_wqmm_dequant_bf16(wPtr, sPtr, oPtr, wdPtr, K, N, hasOffI, dqCores, dqRows, dqCol, stream);
            launch_wqmm_mm_bf16(xPtr, wdPtr, bPtr, yPtr, wsPtr, tp, numBlocks, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("WeightQuantBatchMatmul", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("weight_quant_batch_matmul", wqmm_npu);
}

} // namespace cann_bench
