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
 * \file lstm_plugin.cpp
 * \brief LSTM API layer - torch bindings (compiled with g++).
 *
 * Only metadata (sizes / dtypes / raw device pointers) is inspected on the host.  Every data movement and
 * every arithmetic operation happens inside the custom device kernels; the fp32 workspace is a single
 * device allocation whose sub-regions are addressed by the kernels through absolute device pointers.
 */

#include <tuple>
#include <vector>
#include <cstring>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/lstm_launch.h"

namespace cann_bench {

static inline int64_t A64(int64_t x) { return (x + 63) / 64 * 64; }

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("lstm(Tensor x, Tensor[] weight_ih, Tensor[] weight_hh, int inputSize, int hiddenSize, "
          "int numLayers, bool bias=True, bool batchFirst=False, float dropout=0.0, "
          "bool bidirectional=False, int projSize=0, Tensor?[]? bias_ih=None, Tensor?[]? bias_hh=None, "
          "Tensor?[]? weight_hr=None, Tensor? h0=None, Tensor? c0=None) -> (Tensor y, Tensor hn, Tensor cn)");
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_meta_impl(const at::Tensor &x, int64_t hiddenSize,
                                                                    int64_t numLayers, bool batchFirst,
                                                                    bool bidirectional, int64_t projSize)
{
    const int64_t ND = bidirectional ? 2 : 1;
    const int64_t eff = projSize > 0 ? projSize : hiddenSize;
    TORCH_CHECK(x.dim() == 3, "lstm: x must be a 3D tensor");
    const int64_t B = batchFirst ? x.size(0) : x.size(1);
    auto ySizes = x.sizes().vec();
    ySizes[ySizes.size() - 1] = ND * eff;
    auto y = at::empty(ySizes, x.options());
    auto hn = at::empty({numLayers * ND, B, eff}, x.options());
    auto cn = at::empty({numLayers * ND, B, hiddenSize}, x.options());
    return std::make_tuple(y, hn, cn);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_meta(
    const at::Tensor &x, c10::List<at::Tensor> weight_ih, c10::List<at::Tensor> weight_hh, int64_t inputSize,
    int64_t hiddenSize, int64_t numLayers, bool bias, bool batchFirst, double dropout, bool bidirectional,
    int64_t projSize, const c10::optional<c10::List<c10::optional<at::Tensor>>> &bias_ih,
    const c10::optional<c10::List<c10::optional<at::Tensor>>> &bias_hh,
    const c10::optional<c10::List<c10::optional<at::Tensor>>> &weight_hr, const c10::optional<at::Tensor> &h0,
    const c10::optional<at::Tensor> &c0)
{
    (void)weight_ih;
    (void)weight_hh;
    (void)inputSize;
    (void)bias;
    (void)dropout;
    (void)bias_ih;
    (void)bias_hh;
    (void)weight_hr;
    (void)h0;
    (void)c0;
    return lstm_meta_impl(x, hiddenSize, numLayers, batchFirst, bidirectional, projSize);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("lstm", lstm_meta);
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_npu(
    const at::Tensor &xIn, c10::List<at::Tensor> weight_ih, c10::List<at::Tensor> weight_hh, int64_t inputSize,
    int64_t hiddenSize, int64_t numLayers, bool bias, bool batchFirst, double dropout, bool bidirectional,
    int64_t projSize, const c10::optional<c10::List<c10::optional<at::Tensor>>> &bias_ih,
    const c10::optional<c10::List<c10::optional<at::Tensor>>> &bias_hh,
    const c10::optional<c10::List<c10::optional<at::Tensor>>> &weight_hr, const c10::optional<at::Tensor> &h0,
    const c10::optional<at::Tensor> &c0)
{
    const c10::OptionalDeviceGuard guard(xIn.device());
    (void)dropout;  // inference semantics: the reference disables inter-layer dropout

    const auto st = xIn.scalar_type();
    TORCH_CHECK(st == at::kFloat || st == at::kHalf || st == at::kBFloat16,
                "lstm: only float32 / float16 / bfloat16 are supported");
    TORCH_CHECK(xIn.dim() == 3, "lstm: x must be a 3D tensor");
    TORCH_CHECK(numLayers >= 1, "lstm: numLayers must be >= 1");

    at::Tensor x = xIn.contiguous();

    const int64_t S = batchFirst ? x.size(1) : x.size(0);
    const int64_t B = batchFirst ? x.size(0) : x.size(1);
    const int64_t D = bidirectional ? 2 : 1;
    const int64_t L = numLayers;
    const int64_t LD = L * D;
    TORCH_CHECK(LD <= LSTM_MAX_LD, "lstm: numLayers*numDirections must be <= ", LSTM_MAX_LD);
    TORCH_CHECK(projSize == 0 || projSize <= 64, "lstm: projSize must be 0 or <= 64");

    const int64_t H = hiddenSize;
    const int64_t P = projSize > 0 ? projSize : 0;
    const int64_t eff = P > 0 ? P : H;
    const bool useHr = (P > 0);
    if (useHr) {
        TORCH_CHECK(weight_hr.has_value(), "lstm: projSize>0 requires weight_hr");
    }

    auto outs = lstm_meta_impl(x, H, L, batchFirst, bidirectional, P);
    at::Tensor y = std::get<0>(outs);
    at::Tensor hn = std::get<1>(outs);
    at::Tensor cn = std::get<2>(outs);

    if (x.numel() == 0 || B == 0 || S == 0) {
        return std::make_tuple(y, hn, cn);
    }

    TORCH_CHECK(static_cast<int64_t>(weight_ih.size()) == LD, "lstm: weight_ih length must be numLayers*numDir");
    TORCH_CHECK(static_cast<int64_t>(weight_hh.size()) == LD, "lstm: weight_hh length must be numLayers*numDir");
    const bool hasBias = bias && bias_ih.has_value() && bias_hh.has_value();
    TORCH_CHECK(!bias || hasBias, "lstm: bias=True requires bias_ih and bias_hh");

    std::vector<at::Tensor> keepAlive;
    keepAlive.reserve(8 * static_cast<size_t>(LD) + 8);

    auto cvt = [&](const at::Tensor &t) -> at::Tensor {
        TORCH_CHECK(t.scalar_type() == st, "lstm: weight / bias dtype must match x dtype");
        at::Tensor c = t.contiguous();
        keepAlive.push_back(c);
        return c;
    };

    std::vector<at::Tensor> wih(LD), whh(LD), bih(LD), bhh(LD), whr(LD);
    for (int64_t i = 0; i < LD; ++i) {
        wih[i] = cvt(weight_ih.get(i));
        whh[i] = cvt(weight_hh.get(i));
    }
    if (hasBias) {
        for (int64_t i = 0; i < LD; ++i) {
            const auto bi = bias_ih->get(i);
            const auto bh = bias_hh->get(i);
            TORCH_CHECK(bi.has_value() && bh.has_value(), "lstm: required bias item is missing");
            bih[i] = cvt(*bi);
            bhh[i] = cvt(*bh);
        }
    }
    if (useHr) {
        for (int64_t i = 0; i < LD; ++i) {
            const auto hr = weight_hr->get(i);
            TORCH_CHECK(hr.has_value(), "lstm: required projection item is missing");
            whr[i] = cvt(*hr);
        }
    }
    const bool hasH0 = h0.has_value();
    const bool hasC0 = c0.has_value();
    at::Tensor h0c, c0c;
    if (hasH0) {
        h0c = cvt(h0.value());
    }
    if (hasC0) {
        c0c = cvt(c0.value());
    }

    /* ---------------- geometry ---------------- */
    const int64_t G = A64(H);
    const int64_t kHh = A64(eff);
    const int64_t sw = A64(D * eff);
    const int64_t inPad0 = A64(inputSize);

    LstmArgs a;
    std::memset(&a, 0, sizeof(LstmArgs));
    a.S = static_cast<int32_t>(S);
    a.B = static_cast<int32_t>(B);
    a.inSz = static_cast<int32_t>(inputSize);
    a.H = static_cast<int32_t>(H);
    a.P = static_cast<int32_t>(P);
    a.D = static_cast<int32_t>(D);
    a.L = static_cast<int32_t>(L);
    a.LD = static_cast<int32_t>(LD);
    a.G = static_cast<int32_t>(G);
    a.kHh = static_cast<int32_t>(kHh);
    a.sw = static_cast<int32_t>(sw);
    a.inPad0 = static_cast<int32_t>(inPad0);
    a.eff = static_cast<int32_t>(eff);
    a.batchFirst = batchFirst ? 1 : 0;
    a.hasBias = hasBias ? 1 : 0;
    a.hasH0 = hasH0 ? 1 : 0;
    a.hasC0 = hasC0 ? 1 : 0;
    a.useHr = useHr ? 1 : 0;
    for (int64_t ld = 0; ld < LD; ++ld) {
        const int64_t inDim = (ld / D == 0) ? inputSize : D * eff;
        a.inDim[ld] = static_cast<int32_t>(inDim);
        a.inPad[ld] = static_cast<int32_t>(A64(inDim));
    }

    /* ---------------- workspace ---------------- */
    int64_t off = 0;
    int64_t offWih[LSTM_MAX_LD] = {0}, offWhh[LSTM_MAX_LD] = {0}, offWhr[LSTM_MAX_LD] = {0},
            offBias[LSTM_MAX_LD] = {0};
    for (int64_t ld = 0; ld < LD; ++ld) {
        offWih[ld] = off;
        off += 4 * G * a.inPad[ld];
    }
    for (int64_t ld = 0; ld < LD; ++ld) {
        offWhh[ld] = off;
        off += 4 * G * kHh;
    }
    if (useHr) {
        for (int64_t ld = 0; ld < LD; ++ld) {
            offWhr[ld] = off;
            off += kHh * G;
        }
    }
    for (int64_t ld = 0; ld < LD; ++ld) {
        offBias[ld] = off;
        off += 4 * G;
    }
    const int64_t offX = off;
    off += S * B * inPad0;
    const int64_t offS0 = off;
    off += S * B * sw;
    const int64_t offS1 = off;
    off += S * B * sw;
    const int64_t offH0 = off;
    off += LD * B * kHh;
    const int64_t offC0 = off;
    off += LD * B * G;
    const int64_t wsFloats = off;
    TORCH_CHECK(wsFloats > 0, "lstm: empty workspace");

    at::Tensor ws = at::empty({wsFloats}, x.options().dtype(at::kFloat));
    keepAlive.push_back(ws);

    float *wp = ws.data_ptr<float>();
    auto blk = [&](int64_t o) -> GM_ADDR { return reinterpret_cast<GM_ADDR>(wp + o); };
    for (int64_t ld = 0; ld < LD; ++ld) {
        a.wihW[ld] = blk(offWih[ld]);
        a.whhW[ld] = blk(offWhh[ld]);
        if (useHr) {
            a.whrW[ld] = blk(offWhr[ld]);
        }
        a.biasW[ld] = blk(offBias[ld]);
    }
    a.xW = blk(offX);
    a.seqW[0] = blk(offS0);
    a.seqW[1] = blk(offS1);
    a.h0fW = blk(offH0);
    a.c0fW = blk(offC0);

    a.x = reinterpret_cast<GM_ADDR>(x.data_ptr());
    a.y = reinterpret_cast<GM_ADDR>(y.data_ptr());
    a.hn = reinterpret_cast<GM_ADDR>(hn.data_ptr());
    a.cn = reinterpret_cast<GM_ADDR>(cn.data_ptr());
    if (hasH0) {
        a.h0 = reinterpret_cast<GM_ADDR>(h0c.data_ptr());
    }
    if (hasC0) {
        a.c0 = reinterpret_cast<GM_ADDR>(c0c.data_ptr());
    }
    for (int64_t ld = 0; ld < LD; ++ld) {
        a.wih[ld] = reinterpret_cast<GM_ADDR>(wih[ld].data_ptr());
        a.whh[ld] = reinterpret_cast<GM_ADDR>(whh[ld].data_ptr());
        if (hasBias) {
            a.bih[ld] = reinterpret_cast<GM_ADDR>(bih[ld].data_ptr());
            a.bhh[ld] = reinterpret_cast<GM_ADDR>(bhh[ld].data_ptr());
        }
        if (useHr) {
            a.whr[ld] = reinterpret_cast<GM_ADDR>(whr[ld].data_ptr());
        }
    }

    int64_t pm = inPad0;
    if (G > pm) {
        pm = G;
    }
    if (kHh > pm) {
        pm = kHh;
    }
    for (int64_t ld = 0; ld < LD; ++ld) {
        if (a.inPad[ld] > pm) {
            pm = a.inPad[ld];
        }
    }
    a.prepMaxElems = static_cast<int32_t>(pm);
    a.nbPrep = static_cast<int32_t>(lstm_core_num());
    a.ubBytes = static_cast<int32_t>(lstm_ub_bytes());

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto dtype = st;

    auto acl_call = [=]() -> int {
        if (dtype == at::kFloat) {
            launch_lstm_prep_float(a, stream);
            for (int64_t l = 0; l < L; ++l) {
                launch_lstm_rec_float(a, l, stream);
            }
        } else if (dtype == at::kHalf) {
            launch_lstm_prep_half(a, stream);
            for (int64_t l = 0; l < L; ++l) {
                launch_lstm_rec_half(a, l, stream);
            }
        } else {
            launch_lstm_prep_bf16(a, stream);
            for (int64_t l = 0; l < L; ++l) {
                launch_lstm_rec_bf16(a, l, stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Lstm", acl_call);
    return std::make_tuple(y, hn, cn);
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("lstm", lstm_npu);
}

} // namespace cann_bench
