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
 * \file gru_plugin.cpp
 * \brief GRU API layer - torch bindings (compiled with g++).
 *
 * gru(Tensor x, Tensor[] weight_ih, Tensor[] weight_hh, int inputSize, int hiddenSize, int numLayers,
 *     bool bias=True, bool batchFirst=False, float dropout=0.0, bool bidirectional=False,
 *     Tensor?[]? bias_ih=None, Tensor?[]? bias_hh=None, Tensor? h0=None) -> (Tensor y, Tensor hn)
 *
 * The host only inspects metadata (shapes / dtypes / raw device pointers) and schedules one device
 * kernel; every data movement and every arithmetic operation happens inside the custom kernel.
 */

#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/gru_launch.h"

namespace cann_bench {

static inline int64_t GruAlign64(int64_t v)
{
    return ((v + 63) / 64) * 64;
}

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("gru(Tensor x, Tensor[] weight_ih, Tensor[] weight_hh, int inputSize, int hiddenSize, "
          "int numLayers, bool bias=True, bool batchFirst=False, float dropout=0.0, "
          "bool bidirectional=False, Tensor?[]? bias_ih=None, Tensor?[]? bias_hh=None, Tensor? h0=None) "
          "-> (Tensor y, Tensor hn)");
}

static std::tuple<at::Tensor, at::Tensor> gru_meta_impl(const at::Tensor& x, int64_t hiddenSize,
                                                        int64_t numLayers, bool batchFirst,
                                                        bool bidirectional)
{
    const int64_t ND = bidirectional ? 2 : 1;
    TORCH_CHECK(x.dim() == 3, "gru: x must be a 3D tensor (S, B, input_size) or (B, S, input_size)");
    const int64_t S = batchFirst ? x.size(1) : x.size(0);
    const int64_t B = batchFirst ? x.size(0) : x.size(1);
    (void)S;
    std::vector<int64_t> ySizes(x.sizes().begin(), x.sizes().end());
    ySizes[2] = ND * hiddenSize;
    auto y = at::empty(ySizes, x.options());
    auto hn = at::empty({numLayers * ND, B, hiddenSize}, x.options());
    return std::make_tuple(y, hn);
}

static std::tuple<at::Tensor, at::Tensor> gru_meta(
    const at::Tensor& x, const c10::List<at::Tensor>& weight_ih, const c10::List<at::Tensor>& weight_hh,
    int64_t inputSize, int64_t hiddenSize, int64_t numLayers, bool bias, bool batchFirst, double dropout,
    bool bidirectional, const c10::optional<c10::List<c10::optional<at::Tensor>>>& bias_ih,
    const c10::optional<c10::List<c10::optional<at::Tensor>>>& bias_hh, const c10::optional<at::Tensor>& h0)
{
    (void)weight_ih;
    (void)weight_hh;
    (void)inputSize;
    (void)bias;
    (void)dropout;
    (void)bias_ih;
    (void)bias_hh;
    (void)h0;
    return gru_meta_impl(x, hiddenSize, numLayers, batchFirst, bidirectional);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("gru", gru_meta);
}

static void GruFillList(GM_ADDR* dst, const c10::List<at::Tensor>& src, int64_t n)
{
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = (GM_ADDR)src[i].data_ptr();
    }
}

static std::tuple<at::Tensor, at::Tensor> gru_npu(
    const at::Tensor& xIn, const c10::List<at::Tensor>& weight_ih, const c10::List<at::Tensor>& weight_hh,
    int64_t inputSize, int64_t hiddenSize, int64_t numLayers, bool bias, bool batchFirst, double dropout,
    bool bidirectional, const c10::optional<c10::List<c10::optional<at::Tensor>>>& bias_ih,
    const c10::optional<c10::List<c10::optional<at::Tensor>>>& bias_hh, const c10::optional<at::Tensor>& h0)
{
    const c10::OptionalDeviceGuard guard(xIn.device());
    (void)dropout;  /* GRU eval mode: dropout is an identity between layers */

    const auto st = xIn.scalar_type();
    TORCH_CHECK(st == at::kFloat || st == at::kHalf || st == at::kBFloat16,
                "gru: only float32 / float16 / bfloat16 are supported");
    TORCH_CHECK(xIn.dim() == 3, "gru: x must be a 3D tensor");
    TORCH_CHECK(inputSize >= 1 && hiddenSize >= 1, "gru: inputSize / hiddenSize must be positive");
    TORCH_CHECK(numLayers >= 1, "gru: numLayers must be positive");

    auto x = xIn;
    if (!x.is_contiguous()) {
        x = xIn.contiguous();
    }
    const int64_t ND = bidirectional ? 2 : 1;
    const int64_t S = batchFirst ? x.size(1) : x.size(0);
    const int64_t B = batchFirst ? x.size(0) : x.size(1);
    const int64_t lnd = numLayers * ND;
    const int64_t DH = ND * hiddenSize;

    TORCH_CHECK(static_cast<int64_t>(weight_ih.size()) == lnd,
                "gru: weight_ih length must be numLayers * num_directions");
    TORCH_CHECK(static_cast<int64_t>(weight_hh.size()) == lnd,
                "gru: weight_hh length must be numLayers * num_directions");
    TORCH_CHECK(lnd <= GRU_MAX_LND, "gru: at most ", GRU_MAX_LND, " layer/direction pairs supported");
    TORCH_CHECK(x.size(2) == inputSize, "gru: x last dim must equal inputSize");

    const bool useBias = bias && bias_ih.has_value() && bias_hh.has_value();
    if (bias) {
        TORCH_CHECK(useBias, "gru: bias_ih / bias_hh must be provided when bias is true");
    }
    const bool useH0 = h0.has_value();

    std::vector<int64_t> ySizes(x.sizes().begin(), x.sizes().end());
    ySizes[2] = DH;
    auto y = at::empty(ySizes, x.options());
    auto hn = at::empty({lnd, B, hiddenSize}, x.options());

    /* fp32 workspace: x plane + two layer planes */
    const int64_t xOff = 0;
    const int64_t paOff = GruAlign64(S * B * inputSize);
    const int64_t pbOff = paOff + GruAlign64(S * B * DH);
    const int64_t wsFloats = pbOff + GruAlign64(S * B * DH);
    auto ws = at::empty({wsFloats > 0 ? wsFloats : 1}, x.options().dtype(at::kFloat));

    GruArgs a{};
    a.x = (GM_ADDR)x.data_ptr();
    a.y = (GM_ADDR)y.data_ptr();
    a.hn = (GM_ADDR)hn.data_ptr();
    a.wsc = (GM_ADDR)ws.data_ptr();
    a.S = S;
    a.B = B;
    a.inSz = inputSize;
    a.H = hiddenSize;
    a.L = numLayers;
    a.D = ND;
    a.LD = lnd;
    a.batchFirst = batchFirst ? 1 : 0;
    a.hasBias = useBias ? 1 : 0;
    a.hasH0 = useH0 ? 1 : 0;
    a.xOff = xOff;
    a.paOff = paOff;
    a.pbOff = pbOff;

    /* keep every host-side contiguous view alive until after the kernel is launched */
    std::vector<at::Tensor> keep;
    keep.reserve(2 * static_cast<size_t>(lnd) + 3);
    keep.push_back(x);
    if (useH0) {
        at::Tensor h0c = *h0;
        if (!h0c.is_contiguous()) {
            h0c = h0->contiguous();
        }
        keep.push_back(h0c);
        a.h0 = (GM_ADDR)h0c.data_ptr();
    } else {
        a.h0 = nullptr;
    }

    for (int64_t i = 0; i < lnd; ++i) {
        at::Tensor wi = weight_ih[i];
        if (!wi.is_contiguous()) {
            wi = wi.contiguous();
        }
        at::Tensor wh = weight_hh[i];
        if (!wh.is_contiguous()) {
            wh = wh.contiguous();
        }
        keep.push_back(wi);
        keep.push_back(wh);
        const int64_t kIn = (i / ND == 0) ? inputSize : DH;
        TORCH_CHECK(wi.dim() == 2 && wi.size(0) == 3 * hiddenSize && wi.size(1) == kIn,
                    "gru: bad weight_ih shape for entry ", i);
        TORCH_CHECK(wh.dim() == 2 && wh.size(0) == 3 * hiddenSize && wh.size(1) == hiddenSize,
                    "gru: bad weight_hh shape for entry ", i);
        TORCH_CHECK(wi.scalar_type() == st && wh.scalar_type() == st,
                    "gru: weight dtype must match x");
        a.wih[i] = (GM_ADDR)wi.data_ptr();
        a.whh[i] = (GM_ADDR)wh.data_ptr();
    }
    if (useBias) {
        for (int64_t i = 0; i < lnd; ++i) {
            const auto biOpt = bias_ih->get(i);
            TORCH_CHECK(biOpt.has_value(), "gru: bias_ih item is missing");
            at::Tensor bi = *biOpt;
            if (!bi.is_contiguous()) {
                bi = bi.contiguous();
            }
            const auto bhOpt = bias_hh->get(i);
            TORCH_CHECK(bhOpt.has_value(), "gru: bias_hh item is missing");
            at::Tensor bh = *bhOpt;
            if (!bh.is_contiguous()) {
                bh = bh.contiguous();
            }
            keep.push_back(bi);
            keep.push_back(bh);
            TORCH_CHECK(bi.numel() == 3 * hiddenSize && bh.numel() == 3 * hiddenSize,
                        "gru: bad bias shape for entry ", i);
            TORCH_CHECK(bi.scalar_type() == st && bh.scalar_type() == st,
                        "gru: bias dtype must match x");
            a.bih[i] = (GM_ADDR)bi.data_ptr();
            a.bhh[i] = (GM_ADDR)bh.data_ptr();
        }
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() -> int {
        if (st == at::kFloat) {
            launch_gru_float(a, stream);
        } else if (st == at::kHalf) {
            launch_gru_half(a, stream);
        } else {
            launch_gru_bfloat16(a, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Gru", acl_call);
    return std::make_tuple(y, hn);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("gru", gru_npu);
}

}  // namespace cann_bench
