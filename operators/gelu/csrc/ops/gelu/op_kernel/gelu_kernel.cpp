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
 * \file gelu_kernel.cpp
 * \brief Gelu host code - kernel launch and tiling (compiled with bisheng + -xasc)
 */

#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

// ---------------------------------------------------------------------------
// Bound theta constants (contract 7e2f6adb0e6a82a0d2a3eefde071c82c2060d8df9af1f225d98b6e7dceb73604).
// The erf-mode direct-output tail chain, its per-dtype polynomial
// degree/coefficients, the argument-compression constant, the direct-output
// store and the host tiling budget are all fixed by the solve; none of these
// are re-chosen here.
// ---------------------------------------------------------------------------
constexpr static float ERF_CMAP_FP32 = 2.85f;  // erf_cmap_fp32
constexpr static float ERF_CMAP_FP16 = 2.75f;  // erf_cmap_fp16
constexpr static float ERF_CMAP_BF16 = 2.0f;   // erf_cmap_bf16
constexpr static int32_t ERF_DEGREE_FP32 = 3;  // erf_degree_fp32
constexpr static int32_t ERF_DEGREE_FP16 = 3;  // erf_degree_fp16
constexpr static int32_t ERF_DEGREE_BF16 = 2;  // erf_degree_bf16
// The argument-compression constant is added directly (no sqrt(2) folding):
//   w = a + erf_cmap_<dtype>   (a single Adds)
//   v = a / w
// The bound theta coefficients were fitted against this exact chain; folding a
// sqrt(2) factor into the added constant would change the argument transform
// and the approximation error.
// erf_coef_<dtype> holds c'_1..c'_d in ascending power with c'_0 = 0
// (Q'(0) = 0). Q' = -Q, where Q approximates
// 0.5*a*erfc(a/sqrt2)*exp(a^2/2); the coefficients are stored already negated
// so the kernel evaluates Q' directly by Horner and emits no separate negation.
constexpr static float ERF_COEF_FP32[ERF_DEGREE_FP32] = {
    -1.425461513007655f, 1.8303465324674066f,
    -0.8492939696473478f}; // erf_coef_fp32 (c'_1..c'_3)
constexpr static float ERF_COEF_FP16[ERF_DEGREE_FP16] = {
    -1.3768157174042401f, 1.6791176094454898f,
    -0.7271089970810044f}; // erf_coef_fp16 (c'_1..c'_3)
constexpr static float ERF_COEF_BF16[ERF_DEGREE_BF16] = {
    -0.9994122795663141f, 0.6485493519613369f}; // erf_coef_bf16 (c'_1..c'_2)
constexpr static int64_t UB_MARGIN_BYTES = 1024;     // ub_margin_bytes
constexpr static int64_t MIN_ELEMS_PER_CORE = 4096;  // min_elems_per_core
constexpr static int64_t TILE_ALIGNMENT_ELEMENTS = 128;  // tile_alignment_elements
// UB work-buffer budget (bytes/elem) for the float16-storage erf instantiation
// (T==half, MODE==0), solved this round (was the inherited 16). The actual
// fp16-erf work buffers are three half VECCALC TBufs = 6 B/elem.
constexpr static int64_t ERF_WORK_BUF_BYTES_FP16 = 8;  // erf_work_buf_bytes_fp16
// UB work-buffer budget (bytes/elem) for the float32-storage and
// bfloat16-storage erf instantiations, solved this round (were 12 and 16).
// The fp32-erf path now uses two fp32 VECCALC TBufs (a/t) = 8 B/elem; the
// bf16-erf path uses those two fp32 TBufs plus the fp32 up-cast buffer
// (a/t + xBuf) = 12 B/elem. Both replace the previous three-work-buffer
// topology by computing s = x*x instead of s = a*a, so `a` is dead after
// v = a/w.
constexpr static int64_t ERF_WORK_BUF_BYTES_FP32 = 8;   // erf_work_buf_bytes_fp32
// erf_domain_bf16 selector (solved this round): true = the bfloat16-storage erf
// instantiation uses the fp16 internal domain ("float16"), false = the fp32
// internal domain incumbent ("float32"). The bf16-erf UB work-buffer budget is
// derived from this selector and is 12 B/elem for both values (fp16 domain: one
// fp32 TBuf + four half TBufs; fp32 domain: two fp32 TBufs + fp32 xBuf); it is
// not a separate theta key.
constexpr static bool BF16_ERF_FP16_DOMAIN = true;  // erf_domain_bf16 == "float16"
constexpr static int64_t ERF_WORK_BUF_BYTES_BF16 = 12;  // derived from erf_domain_bf16 (12 for both values)
// UB work-buffer budget (bytes/elem) for the tanh-mode instantiations, solved
// this round. The fp32 tanh path drops its second fp32 VECCALC buffer (8 -> 4):
// the final Div writes the VECOUT tensor directly and the trailing Muls copy is
// removed, so only tBuf is needed. The 16-bit tanh path drops its second fp32
// buffer (12 -> 8): the final Div writes the existing fp32 up-cast buffer in
// place, so only tBuf and xBuf are needed.
constexpr static int64_t TANH_WORK_BUF_BYTES_FP32 = 4;   // tanh_work_buf_bytes_fp32
constexpr static int64_t TANH_WORK_BUF_BYTES_16BIT = 8;  // tanh_work_buf_bytes_16bit

// Kernel implementation
// MODE: compile-time mode selector, 0 = erf mode, 1 = tanh mode. The mode is a
// template parameter, so exactly one arithmetic chain and one buffer topology
// are compiled into each (T, MODE) instantiation; no runtime mode value is
// carried into the kernel. The erf mode uses the fp16 internal domain for
// T==half (z4 [S2]) and for T==bfloat16_t (this round's [L1] decision), and the
// fp32 internal domain for T==float; the tanh mode uses the fp32 internal
// domain for all dtypes.
//
// [S10] single fixed-size masked tile loop: the whole per-core block is covered
// by one loop of ceil(currentBlockLength / tileElementCount) iterations. Every
// iteration uses the same fixed tileElementCount; the final iteration may cover
// fewer valid elements (n = min(tileElementCount, remaining)) and is handled by
// a masked/padded copy and a masked store inside the same loop. There is no
// separate tail path and no bulk/tail branch.
template <typename T, int32_t MODE>
__global__ __aicore__ void gelu_kernel(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t blockLength,
    uint32_t tileElementCount)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, yGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tBuf;   // work buffer 1 (fp32, or half for fp16-domain erf)
    AscendC::TBuf<AscendC::TPosition::VECCALC> uBuf;   // work buffer 2 (half, fp16-domain erf only)
    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuf;   // work buffer 3 (fp32, or half for fp16-domain erf)
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf;   // up-cast of x (fp32 for fp32-erf/tanh; half for fp16-domain erf)
    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf;   // half Horner accumulator / x+t' (bf16-erf fp16 domain only)

    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileElementCount * sizeof(T));
    pipe.InitBuffer(outQueueY, PIPELINE_DEPTH, tileElementCount * sizeof(T));
    if constexpr (std::is_same_v<T, half>) {
        if constexpr (MODE == 0) {
            // z4 [S2] fp16 internal-domain erf path (erf_domain_fp16 =
            // "float16"): three half VECCALC TBufs (a/t/u) and no fp32 up-cast
            // buffer. The unchanged 16 B/elem 16-bit erf tiling budget remains a
            // valid upper bound (6 B/elem of work buffers used here).
            pipe.InitBuffer(tBuf, tileElementCount * sizeof(half));
            pipe.InitBuffer(uBuf, tileElementCount * sizeof(half));
            pipe.InitBuffer(aBuf, tileElementCount * sizeof(half));
        } else {
            // tanh mode: tBuf + fp32 up-cast buffer only; no uBuf (the final
            // Div writes the up-cast buffer in place).
            pipe.InitBuffer(tBuf, tileElementCount * sizeof(float));
            pipe.InitBuffer(xBuf, tileElementCount * sizeof(float));
        }
    } else {
        if constexpr (MODE == 0) {
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                // bf16-erf fp16 internal domain (erf_domain_bf16 == "float16"),
                // two-step cast through fp32: one fp32 VECCALC TBuf (tBuf, the
                // bf16->fp32 up-cast and fp16->fp32 down-cast intermediate) plus
                // four half VECCALC TBufs (xBuf = fp16 up-cast of x, aBuf = |x|,
                // uBuf = w/v/s/e/t', wBuf = Horner accumulator then x+t').
                pipe.InitBuffer(tBuf, tileElementCount * sizeof(float));
                pipe.InitBuffer(xBuf, tileElementCount * sizeof(half));
                pipe.InitBuffer(aBuf, tileElementCount * sizeof(half));
                pipe.InitBuffer(uBuf, tileElementCount * sizeof(half));
                pipe.InitBuffer(wBuf, tileElementCount * sizeof(half));
            } else {
                // fp32-erf: two fp32 VECCALC TBufs (a/t) only; no uBuf. The
                // two-buffer chain computes s = x*x, so `a` is dead after v = a/w.
                pipe.InitBuffer(tBuf, tileElementCount * sizeof(float));
                pipe.InitBuffer(aBuf, tileElementCount * sizeof(float));
                if constexpr (sizeof(T) < sizeof(float)) {
                    pipe.InitBuffer(xBuf, tileElementCount * sizeof(float));
                }
            }
        } else {
            // tanh mode: tBuf only for T==float; plus the fp32 up-cast buffer for
            // bfloat16_t. No uBuf (the final Div writes VECOUT directly for
            // T==float and the up-cast buffer in place for bfloat16_t).
            pipe.InitBuffer(tBuf, tileElementCount * sizeof(float));
            if constexpr (sizeof(T) < sizeof(float)) {
                pipe.InitBuffer(xBuf, tileElementCount * sizeof(float));
            }
        }
    }

    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx());

    int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) currentBlockLength = blockLength;

    // [S10] single fixed-size masked tile loop. loopCount = ceil(currentBlockLength
    // / tileElementCount); the final iteration covers n = min(tileElementCount,
    // remaining) valid elements. One DataCopyExtParams and one
    // DataCopyPadExtParams are constructed per iteration from that iteration's n,
    // and the unchanged arithmetic chain runs with element count n.
    int64_t loopCount = (currentBlockLength + (int64_t)tileElementCount - 1) / (int64_t)tileElementCount;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t offset = i * (int64_t)tileElementCount;
        int64_t remaining = currentBlockLength - offset;
        int64_t n = ((int64_t)tileElementCount < remaining) ? (int64_t)tileElementCount : remaining;

        AscendC::DataCopyExtParams copyParams{1,
            static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        auto xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<T>();

        AscendC::LocalTensor<float> tF32;
        AscendC::LocalTensor<float> aF32;          // aBuf tensor (fp32 erf modes only)
        AscendC::LocalTensor<float> xF32;
        AscendC::LocalTensor<half> tH;             // fp16 internal-domain erf buffers
        AscendC::LocalTensor<half> uH;
        AscendC::LocalTensor<half> aH;
        AscendC::LocalTensor<half> xF16;           // fp16 up-cast of bf16 x (bf16-erf fp16 domain)
        if constexpr (std::is_same_v<T, half>) {
            if constexpr (MODE == 0) {
                // fp16 internal-domain erf path: half work tensors, no upcast.
                tH = tBuf.Get<half>();
                uH = uBuf.Get<half>();
                aH = aBuf.Get<half>();
            } else {
                tF32 = tBuf.Get<float>();
                xF32 = xBuf.Get<float>();
                AscendC::Cast(xF32, xLocal, AscendC::RoundMode::CAST_NONE, n);
            }
        } else {
            if constexpr (MODE == 0) {
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    // bf16-erf fp16 internal domain: tBuf is the fp32 two-step
                    // cast intermediate; xBuf/aBuf/uBuf/wBuf are the half work
                    // tensors (wBuf is the Horner accumulator / x+t').
                    tF32 = tBuf.Get<float>();
                    xF16 = xBuf.Get<half>();
                    aH = aBuf.Get<half>();
                    uH = uBuf.Get<half>();
                    tH = wBuf.Get<half>();
                } else {
                    // fp32-erf: A = aBuf, B = tBuf; only two work buffers.
                    tF32 = tBuf.Get<float>();
                    aF32 = aBuf.Get<float>();
                    if constexpr (sizeof(T) < sizeof(float)) {
                        xF32 = xBuf.Get<float>();
                    } else {
                        xF32 = xLocal;
                    }
                }
            } else {
                tF32 = tBuf.Get<float>();
                if constexpr (sizeof(T) < sizeof(float)) {
                    xF32 = xBuf.Get<float>();
                    AscendC::Cast(xF32, xLocal, AscendC::RoundMode::CAST_NONE, n);
                } else {
                    xF32 = xLocal;
                }
            }
        }

        // direct_output=true: the VECOUT tensor is allocated before the
        // arithmetic so the final fp32 Add can write into it directly.
        auto yLocal = outQueueY.AllocTensor<T>();

        if constexpr (MODE == 0) {
            if constexpr (std::is_same_v<T, half>) {
                // z4 [S2] fp16 internal-domain erf chain (erf_domain_fp16 =
                // "float16"): the unchanged [S5] direct-output tail evaluated on
                // half tensors with no fp32 upcast/downcast. The erf_cmap_fp16
                // scalar and each erf_coef_fp16 element are cast to half at use.
                //   a = Abs(x)
                //   w = a + erf_cmap_fp16        (single Adds)
                //   v = a / w
                //   Q' = Horner(erf_coef_fp16, v), degree erf_degree_fp16 = 3,
                //        c'_0 = 0: p = c'_3*v; for j = 2,1: p = p + c'_j; p = p*v
                //   s = a*a; e = Exp(-0.5*s); t' = e*Q'
                //   y = Max(x + t', t')  -> written directly into the VECOUT half tensor
                AscendC::Abs(aH, xLocal, n);                        // a = |x|
                AscendC::Adds(tH, aH, (half)ERF_CMAP_FP16, n);      // w = a + cmap_fp16
                AscendC::Div(uH, aH, tH, n);                        // v = a/w
                AscendC::Muls(tH, uH, (half)ERF_COEF_FP16[2], n);   // p = c'_3*v
                AscendC::Adds(tH, tH, (half)ERF_COEF_FP16[1], n);   // + c'_2
                AscendC::Mul(tH, tH, uH, n);                        // * v
                AscendC::Adds(tH, tH, (half)ERF_COEF_FP16[0], n);   // + c'_1
                AscendC::Mul(tH, tH, uH, n);                        // -> Q'(v)
                AscendC::Mul(uH, aH, aH, n);                        // s = a*a
                AscendC::Muls(uH, uH, (half)(-0.5f), n);            // -0.5*s
                AscendC::Exp(uH, uH, n);                            // e = exp(-0.5*s)
                AscendC::Mul(uH, uH, tH, n);                        // t' = e*Q'
                AscendC::Add(tH, xLocal, uH, n);                    // x + t'
                AscendC::Max(yLocal, tH, uH, n);                    // y = Max(x+t', t')
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                // bf16-erf fp16 internal domain (erf_domain_bf16 == "float16"):
                // up-cast bf16 -> fp16, run the z4 [S2] fp16-erf chain on half
                // tensors, then down-cast fp16 -> bf16 into the VECOUT tensor.
                //   x16 = Cast(x) [bf16 -> fp16]
                //   a = Abs(x16); w = a + erf_cmap_fp16; v = a/w
                //   Q' = Horner(erf_coef_fp16, v), degree 3, c'_0 = 0
                //   s = a*a; e = Exp(-0.5*s); t' = e*Q'
                //   y = Max(x16 + t', t')  -> down-cast to bf16
                AscendC::Cast(tF32, xLocal, AscendC::RoundMode::CAST_NONE, n);  // bf16 -> fp32
                AscendC::Cast(xF16, tF32, AscendC::RoundMode::CAST_NONE, n);    // fp32 -> fp16
                AscendC::Abs(aH, xF16, n);                        // a = |x|
                AscendC::Adds(tH, aH, (half)ERF_CMAP_FP16, n);    // w = a + cmap_fp16
                AscendC::Div(uH, aH, tH, n);                      // v = a/w
                AscendC::Muls(tH, uH, (half)ERF_COEF_FP16[2], n); // p = c'_3*v
                AscendC::Adds(tH, tH, (half)ERF_COEF_FP16[1], n); // + c'_2
                AscendC::Mul(tH, tH, uH, n);                      // * v
                AscendC::Adds(tH, tH, (half)ERF_COEF_FP16[0], n); // + c'_1
                AscendC::Mul(tH, tH, uH, n);                      // -> Q'(v)
                AscendC::Mul(uH, aH, aH, n);                      // s = a*a
                AscendC::Muls(uH, uH, (half)(-0.5f), n);          // -0.5*s
                AscendC::Exp(uH, uH, n);                          // e = exp(-0.5*s)
                AscendC::Mul(uH, uH, tH, n);                      // t' = e*Q'
                AscendC::Add(tH, xF16, uH, n);                    // x16 + t'
                AscendC::Max(tH, tH, uH, n);                      // y (result in tH)
                AscendC::Cast(tF32, tH, AscendC::RoundMode::CAST_NONE, n);      // fp16 -> fp32
                AscendC::Cast(yLocal, tF32, AscendC::RoundMode::CAST_RINT, n);  // fp32 -> bf16
            } else {
            // erf mode (approximate='none'), bound [S5] direct-output tail with
            // the exact final-combination rearrangement (contract
            // 9ec22b02fcea537373a258a96d78beac08565fd6206234799ab62c7dce5b21c1):
            //   a = Abs(x)
            //   w = a + erf_cmap_<dtype>   (single Adds)
            //   v = a / w
            //   Q' = Horner(erf_coef_<dtype>, v), degree erf_degree_<dtype>, c'_0 = 0:
            //       p = c'_d*v; for j = d-1..1: p = p + c'_j; p = p*v
            //   s = a*a; e = Exp(-0.5*s); t' = e*Q'
            //   y = Max(x + t', t')
            // All arithmetic is in the fp32 internal domain; upcast on load and
            // downcast on store for half/bfloat16_t are unchanged from the bound
            // source. This round's two-buffer topology uses aBuf and tBuf only
            // (no uBuf): s = x*x replaces s = a*a, so `a` is dead after v = a/w.
            // The final Max takes (x+t') and t' as its two operands.
            if constexpr (sizeof(T) < sizeof(float)) {
                AscendC::Cast(xF32, xLocal, AscendC::RoundMode::CAST_NONE, n);  // upcast bf16 -> fp32
            }
            AscendC::Abs(aF32, xF32, n);                        // A = a = |x|
            // w = a + cmap as a single Adds; v = a/w. The compression constant
            // is per-dtype (fp32/fp16/bf16), selected by the instantiation type.
            if constexpr (std::is_same_v<T, float>) {
                AscendC::Adds(tF32, aF32, ERF_CMAP_FP32, n);  // B = w = a + cmap_fp32
            } else {
                AscendC::Adds(tF32, aF32, ERF_CMAP_BF16, n);  // B = w = a + cmap_bf16
            }
            AscendC::Div(aF32, aF32, tF32, n);               // A = v = a/w
            if constexpr (std::is_same_v<T, float>) {
                // Horner for degree ERF_DEGREE_FP32 (=3) with c'_0 = 0:
                // p = c'_3*v; then j=2,1: p = p + c'_j; p = p*v.
                AscendC::Muls(tF32, aF32, ERF_COEF_FP32[2], n); // p = c'_3*v
                AscendC::Adds(tF32, tF32, ERF_COEF_FP32[1], n); // + c'_2
                AscendC::Mul(tF32, tF32, aF32, n);              // * v
                AscendC::Adds(tF32, tF32, ERF_COEF_FP32[0], n); // + c'_1
                AscendC::Mul(tF32, tF32, aF32, n);              // B = Q'(v)
            } else {
                // Horner for degree ERF_DEGREE_BF16 (=2) with c'_0 = 0:
                // p = c'_2*v; then j=1: p = p + c'_j; p = p*v.
                AscendC::Muls(tF32, aF32, ERF_COEF_BF16[1], n); // p = c'_2*v
                AscendC::Adds(tF32, tF32, ERF_COEF_BF16[0], n); // + c'_1
                AscendC::Mul(tF32, tF32, aF32, n);              // B = Q'(v)
            }
            AscendC::Mul(aF32, xF32, xF32, n);              // A = s = x*x
            AscendC::Muls(aF32, aF32, -0.5f, n);            // -0.5*s
            AscendC::Exp(aF32, aF32, n);                    // e = exp(-0.5*s)
            AscendC::Mul(aF32, aF32, tF32, n);              // A = t' = e*Q'
            // y = Max(x + t', t'). The Add computes x + t' into the VECCALC
            // fp32 buffer; the final Max takes (x+t') and t' as its two
            // operands. No literal-zero operand and no Duplicate are used.
            AscendC::Add(tF32, xF32, aF32, n);              // B = x + t'
            if constexpr (sizeof(T) == 4) {
                // final fp32 result written directly into the VECOUT tensor.
                AscendC::Max(yLocal, tF32, aF32, n);        // y = Max(x+t', t')
            } else {
                // 16-bit: the final Max writes the VECCALC buffer; the
                // post-branch Cast(CAST_RINT) writes VECOUT.
                AscendC::Max(tF32, tF32, aF32, n);          // y (result in tF32)
            }
            }
        } else {
            // tanh mode: y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            // Canonical CANN decomposition (docs: Gelu 8-step Mul/Muls/Add/Exp/
            // Adds/Div) avoiding the 1+tanh(z) rounding cliff at z ~ -9:
            // y = x / (1 + exp(-2*beta*(x + kappa*x^3)))
            //   beta = 0.7978845608028654, kappa = 0.044715
            // Factored exp argument (exact rearrangement, removes the x^3 Mul):
            //   arg = -2*beta*(x + kappa*x^3) = x*(-2*beta - 2*beta*kappa*x^2)
            //   two_beta = -2*beta = -1.5957691216057308
            //   two_beta_kappa = -2*beta*kappa = -0.0713548162726
            AscendC::Mul(tF32, xF32, xF32, n);                 // x^2
            AscendC::Muls(tF32, tF32, -0.0713548162726f, n);   // -2*beta*kappa*x^2
            AscendC::Adds(tF32, tF32, -1.5957691216057308f, n); // -2*beta - 2*beta*kappa*x^2
            AscendC::Mul(tF32, xF32, tF32, n);                 // arg = x*(-2*beta - 2*beta*kappa*x^2)
            AscendC::Exp(tF32, tF32, n);                       // framework tmp
            AscendC::Adds(tF32, tF32, 1.0f, n);
            if constexpr (sizeof(T) < sizeof(float)) {
                // 16-bit: final Div writes the fp32 up-cast buffer in place; the
                // existing Cast(CAST_RINT) below stores it to VECOUT.
                AscendC::Div(xF32, xF32, tF32, n);             // final result in xF32
            } else {
                // fp32: final Div writes the VECOUT tensor directly (no trailing
                // Muls copy).
                AscendC::Div(yLocal, xF32, tF32, n);           // final result in yLocal
            }
        }

        if constexpr (MODE == 0) {
            // The bf16-erf result is down-cast inside the chain (fp16 -> fp32 ->
            // bf16 two-step); the fp32 and fp16 results were already written
            // directly into yLocal (direct_output). No post-branch cast.
        } else {
            if constexpr (sizeof(T) < sizeof(float)) {
                AscendC::Cast(yLocal, xF32, AscendC::RoundMode::CAST_RINT, n);
            }
            // fp32 tanh: the final Div already wrote yLocal directly; no copy.
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }
}

// Tiling function
// returns: numBlocks, blockLength, tileElementCount
// Erf/Tanh use the framework-allocated tmp overloads (no user tmp buffer needed).
std::tuple<int64_t, int64_t, int64_t> calc_gelu_tiling_params(int64_t totalLength, int32_t mode,
    int64_t typeSize, int32_t storageDtype)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    if (numBlocks <= 0) numBlocks = 1;
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;

    // per-element UB cost: VECIN x2 + VECOUT x2 (4*typeSize), fp32 work buffers,
    // plus xF32 up-cast buffer (4B) for 16-bit dtypes. Exp tmp is framework-allocated.
    // erf mode (mode==0): storage-dtype-dependent work-buffer budget.
    //   fp16-erf uses three half VECCALC TBufs (6 B/elem) -> budget 8.
    //   fp32-erf now uses two fp32 VECCALC TBufs (a/t) -> budget 8 (was 12).
    //   bf16-erf uses four half VECCALC TBufs (t/u/a/x) -> budget 8 (was 12).
    // tanh mode: this round's solved budgets (4 for fp32, 8 for 16-bit); the
    // fp32 path drops the second fp32 TBuf and the 16-bit path drops the second
    // fp32 TBuf (final Div writes in place / directly to VECOUT).
    int64_t workBufBytes;
    if (mode == 0) {
        if (storageDtype == 1) {
            workBufBytes = ERF_WORK_BUF_BYTES_FP16;  // erf_work_buf_bytes_fp16 (float16 storage)
        } else if (storageDtype == 2) {
            workBufBytes = ERF_WORK_BUF_BYTES_BF16;  // erf_work_buf_bytes_bf16 (bfloat16 storage)
        } else {
            workBufBytes = ERF_WORK_BUF_BYTES_FP32;  // erf_work_buf_bytes_fp32 (float32 storage)
        }
    } else {
        workBufBytes = (typeSize < 4) ? TANH_WORK_BUF_BYTES_16BIT : TANH_WORK_BUF_BYTES_FP32;
    }
    int64_t perElem = 4 * typeSize + workBufBytes;
    int64_t usable = (int64_t)ubSize - UB_MARGIN_BYTES;
    if (usable < perElem * 128) usable = perElem * 128;
    int64_t tileElementCount = usable / perElem;
    tileElementCount = tileElementCount / TILE_ALIGNMENT_ELEMENTS * TILE_ALIGNMENT_ELEMENTS;
    if (tileElementCount < 128) tileElementCount = 128;

    return std::make_tuple(numBlocks, blockLength, tileElementCount);
}

// Launch wrappers - regular C functions callable from g++.
// One wrapper per compile-time (storage dtype, mode) specialization; the host
// selects the matching wrapper after resolving `approximate`, so no runtime mode
// value is carried into the kernel.
extern "C" {

void launch_gelu_kernel_float_erf(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElementCount, void* stream)
{
    gelu_kernel<float, 0><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElementCount);
}

void launch_gelu_kernel_float_tanh(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElementCount, void* stream)
{
    gelu_kernel<float, 1><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElementCount);
}

void launch_gelu_kernel_half_erf(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElementCount, void* stream)
{
    gelu_kernel<half, 0><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElementCount);
}

void launch_gelu_kernel_half_tanh(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElementCount, void* stream)
{
    gelu_kernel<half, 1><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElementCount);
}

void launch_gelu_kernel_bfloat16_erf(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElementCount, void* stream)
{
    gelu_kernel<bfloat16_t, 0><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElementCount);
}

void launch_gelu_kernel_bfloat16_tanh(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElementCount, void* stream)
{
    gelu_kernel<bfloat16_t, 1><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElementCount);
}

}
