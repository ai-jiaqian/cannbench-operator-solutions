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
 * \file transpose_plugin.cpp
 * \brief Transpose API layer - torch bindings (compiled with g++)
 *
 * The host side only inspects tensor metadata (shape / dtype) and launches the kernel; no
 * device data is read and no host-side conversion happens.
 */

#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/transpose_launch.h"

namespace cann_bench {

namespace {

void CheckPerm(const torch::Tensor& x, at::IntArrayRef perm)
{
    const int64_t nd = x.dim();
    TORCH_CHECK(static_cast<int64_t>(perm.size()) == nd,
                "transpose: perm length must equal the input rank");
    std::vector<int64_t> seen(static_cast<size_t>(nd), 0);
    for (int64_t i = 0; i < nd; ++i) {
        const int64_t d = perm[static_cast<size_t>(i)];
        TORCH_CHECK(d >= 0 && d < nd, "transpose: perm entry out of range");
        TORCH_CHECK(seen[static_cast<size_t>(d)] == 0, "transpose: perm is not a permutation");
        seen[static_cast<size_t>(d)] = 1;
    }
}

std::vector<int64_t> OutSizes(const torch::Tensor& x, at::IntArrayRef perm)
{
    const int64_t nd = x.dim();
    std::vector<int64_t> osz(static_cast<size_t>(nd));
    for (int64_t i = 0; i < nd; ++i) {
        osz[static_cast<size_t>(i)] = x.size(perm[static_cast<size_t>(i)]);
    }
    return osz;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("transpose(Tensor x, int[] perm) -> Tensor y");
}

torch::Tensor transpose_meta(const torch::Tensor& x, at::IntArrayRef perm)
{
    CheckPerm(x, perm);
    return torch::empty(OutSizes(x, perm), x.options());
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("transpose", transpose_meta);
}

torch::Tensor transpose_npu(const torch::Tensor& x, at::IntArrayRef perm)
{
    const c10::OptionalDeviceGuard guard(x.device());
    CheckPerm(x, perm);
    const int64_t nd = x.dim();

    torch::Tensor in = x;
    if (!in.is_contiguous()) {
        in = x.contiguous();
    }

    auto output = torch::empty(OutSizes(in, perm), in.options());
    if (in.numel() == 0 || nd == 0) {
        return output;
    }
    TORCH_CHECK(nd <= 8, "transpose: rank above 8 is not supported");

    int64_t sizes[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t prm[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int64_t i = 0; i < nd; ++i) {
        sizes[i] = in.size(i);
        prm[i] = perm[static_cast<size_t>(i)];
    }

    TPParams P;
    calc_transpose_params(nd, static_cast<int64_t>(in.element_size()), in.numel(), sizes, prm, &P);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto xPtr = (GM_ADDR)in.data_ptr();
    auto yPtr = (GM_ADDR)output.data_ptr();
    const auto dtype = in.scalar_type();

    auto acl_call = [=]() -> int {
        switch (dtype) {
            case torch::kFloat32:
                launch_transpose_f32(xPtr, yPtr, P.n, P.d0, P.d1, P.d2, P.d3, P.pp, P.mode, P.TC,
                                     P.K, P.Pk, P.Pj, P.nch, P.ncc, P.nlc, P.nTiles, P.nBlocks,
                                     P.RA, P.numel, P.Ta, P.Tb, P.packed, P.overA, P.NB, stream);
                break;
            case torch::kFloat16:
                launch_transpose_f16(xPtr, yPtr, P.n, P.d0, P.d1, P.d2, P.d3, P.pp, P.mode, P.TC,
                                     P.K, P.Pk, P.Pj, P.nch, P.ncc, P.nlc, P.nTiles, P.nBlocks,
                                     P.RA, P.numel, P.Ta, P.Tb, P.packed, P.overA, P.NB, stream);
                break;
            case torch::kBFloat16:
                launch_transpose_bf16(xPtr, yPtr, P.n, P.d0, P.d1, P.d2, P.d3, P.pp, P.mode, P.TC,
                                      P.K, P.Pk, P.Pj, P.nch, P.ncc, P.nlc, P.nTiles, P.nBlocks,
                                      P.RA, P.numel, P.Ta, P.Tb, P.packed, P.overA, P.NB, stream);
                break;
            case torch::kInt8:
                launch_transpose_i8(xPtr, yPtr, P.n, P.d0, P.d1, P.d2, P.d3, P.pp, P.mode, P.TC,
                                    P.K, P.Pk, P.Pj, P.nch, P.ncc, P.nlc, P.nTiles, P.nBlocks,
                                    P.RA, P.numel, P.Ta, P.Tb, P.packed, P.overA, P.NB, stream);
                break;
            case torch::kInt16:
                launch_transpose_i16(xPtr, yPtr, P.n, P.d0, P.d1, P.d2, P.d3, P.pp, P.mode, P.TC,
                                     P.K, P.Pk, P.Pj, P.nch, P.ncc, P.nlc, P.nTiles, P.nBlocks,
                                     P.RA, P.numel, P.Ta, P.Tb, P.packed, P.overA, P.NB, stream);
                break;
            case torch::kInt32:
                launch_transpose_i32(xPtr, yPtr, P.n, P.d0, P.d1, P.d2, P.d3, P.pp, P.mode, P.TC,
                                     P.K, P.Pk, P.Pj, P.nch, P.ncc, P.nlc, P.nTiles, P.nBlocks,
                                     P.RA, P.numel, P.Ta, P.Tb, P.packed, P.overA, P.NB, stream);
                break;
            case torch::kInt64:
                launch_transpose_i64(xPtr, yPtr, P.n, P.d0, P.d1, P.d2, P.d3, P.pp, P.mode, P.TC,
                                     P.K, P.Pk, P.Pj, P.nch, P.ncc, P.nlc, P.nTiles, P.nBlocks,
                                     P.RA, P.numel, P.Ta, P.Tb, P.packed, P.overA, P.NB, stream);
                break;
            default:
                TORCH_CHECK(false, "transpose: unsupported dtype");
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Transpose", acl_call);
    return output;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("transpose", transpose_npu);
}

}  // namespace cann_bench
