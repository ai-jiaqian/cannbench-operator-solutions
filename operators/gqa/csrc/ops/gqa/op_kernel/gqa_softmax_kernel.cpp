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
 * \file gqa_softmax_kernel.cpp
 * \brief GQA stage 3 (AIV): scaled causal softmax of the score tiles, computed in place.
 *
 * Row order of the packed score matrix: within one (batch, kv-head) block the row index is
 * r = i*S + s (head-in-group outer, query position inner), so the query position of a row is
 * s = r % S.  The valid key range is the prefix
 *   0 .. lim    with lim = min(S_kv - 1, s + (S_kv - S))   when causal, else S_kv - 1.
 * The reference guarantees S <= S_kv for causal runs, so lim >= 0 and no row is fully masked.
 *
 * Numerics: the reference computes  scores_T = round_T(matmul)            (the cube stage does that)
 *                                  scores_T = round_T(scores_T * scale)
 *                                  p        = round_T(softmax_fp32(scores_T))
 * The intermediate rounding to T after the scale multiply is reproduced explicitly with a
 * Cast-to-T / Cast-to-float round trip, and the final probability is rounded to T with CAST_RINT
 * (round to nearest even), which is the same rounding torch applies.
 *
 * The masked suffix is neutralised by writing a huge negative sentinel into the float tile before
 * the exponentiation, so it contributes exp(-huge) = 0 to the denominator and 0 to the numerator
 * exactly like the reference's -inf masked_fill, without ever forming an infinity.
 *
 * Row maximum: softmax is shift invariant, so the maximum subtraction only exists to bound the
 * exponent range.  The host proves |scaleValue| * D <= 60 from the operator's own shape and scale,
 * in which case exp() cannot overflow fp32 (exp(60) = 1.1e26) and the two extra vector ops plus one
 * vector-to-scalar barrier per row are skipped.  When that bound does not hold the classic
 * max-subtracted form runs instead, so the kernel stays defined for any scaleValue.
 *
 * Alignment: a vector instruction may only touch a Unified Buffer base address that is 32-byte
 * aligned, so the sentinel fill cannot start at lim+1.  It starts at the next 16-element boundary at
 * or below lim+1 (16 float elements = 64 bytes, and the matching 16 dtype elements = 32 bytes) and
 * the handful of valid elements that this over-writes are restored by re-widening the quantised tile.
 *
 * Rows are moved in batches of R (chosen so the tile fits the UB budget) so that the DMA traffic is
 * counted in tiles rather than rows; the scalar read-back of the row sum needs an explicit vector to
 * scalar barrier, which is what the PipeBarrier<PIPE_ALL> below is for - PipeBarrier<PIPE_V> alone
 * does not publish a vector store to the scalar pipe.
 */

#include <algorithm>

#include "kernel_operator.h"
#include "gqa_launch.h"

using namespace AscendC;

namespace {
constexpr float GQA_SM_NEG = -3.0e38f;
constexpr int32_t GQA_SM_BATCH = 20 * 1024; /* T-bytes budget for one row batch slot */
}  // namespace

template <typename T>
__global__ __aicore__ void gqa_softmax_kernel(GM_ADDR scAddr, int64_t totalRows, int64_t RS,
                                              int64_t G, int64_t S, int64_t Skv, float scale,
                                              int32_t causal, int32_t useMax, int64_t rowsPerCore)
{
    (void)G;
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t r0 = blk * rowsPerCore;
    if (r0 >= totalRows) {
        return;
    }
    int64_t r1 = r0 + rowsPerCore;
    if (r1 > totalRows) {
        r1 = totalRows;
    }

    int64_t R = GQA_SM_BATCH / (Skv * static_cast<int64_t>(sizeof(T)));
    if (R < 1) {
        R = 1;
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> f32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> redBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    pipe.InitBuffer(inQue, 2, R * Skv * sizeof(T));
    pipe.InitBuffer(outQue, 2, R * Skv * sizeof(T));
    pipe.InitBuffer(f32Buf, R * Skv * sizeof(float));
    pipe.InitBuffer(redBuf, 32 * sizeof(float));
    pipe.InitBuffer(tmpBuf, Skv * sizeof(float));

    AscendC::GlobalTensor<T> scG;
    scG.SetGlobalBuffer((__gm__ T *)scAddr);

    auto f32 = f32Buf.Get<float>();
    auto red = redBuf.Get<float>();
    auto tmp = tmpBuf.Get<float>();
    const int64_t limShift = Skv - S;
    constexpr int64_t ALIGN = 16;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    for (int64_t g0 = r0; g0 < r1; g0 += R) {
        int64_t cnt = r1 - g0;
        if (cnt > R) {
            cnt = R;
        }
        const int32_t nElems = static_cast<int32_t>(cnt * Skv);
        const uint32_t rowBytes = static_cast<uint32_t>(Skv * static_cast<int64_t>(sizeof(T)));

        auto tin = inQue.AllocTensor<T>();
        AscendC::DataCopyExtParams cpIn{static_cast<uint16_t>(cnt), rowBytes, 0u, 0u, 0u};
        AscendC::DataCopyPad(tin, scG[g0 * Skv], cpIn, padParams);
        inQue.EnQue(tin);
        auto in = inQue.DeQue<T>();
        AscendC::Cast(f32, in, AscendC::RoundMode::CAST_NONE, nElems);
        inQue.FreeTensor(in);

        /* scores_T = round_T(scores_T * scale), then widen back to float. */
        AscendC::Muls(f32, f32, scale, nElems);
        auto quant = outQue.AllocTensor<T>();
        AscendC::Cast(quant, f32, AscendC::RoundMode::CAST_RINT, nElems);
        AscendC::Cast(f32, quant, AscendC::RoundMode::CAST_NONE, nElems);

        for (int64_t r = 0; r < cnt; ++r) {
            const int64_t rr = (g0 + r) % RS;
            int64_t lim = Skv - 1;
            if (causal != 0) {
                lim = (rr % S) + limShift;
                if (lim > Skv - 1) {
                    lim = Skv - 1;
                }
                if (lim < 0) {
                    lim = 0;
                }
            }
            auto row = f32[r * Skv];
            if (lim < Skv - 1) {
                const int64_t keep = lim + 1;
                const int64_t base = keep & ~(ALIGN - 1);
                AscendC::Duplicate(row[base], GQA_SM_NEG, static_cast<int32_t>(Skv - base));
                if (base < keep) {
                    AscendC::Cast(row[base], quant[r * Skv + base],
                                  AscendC::RoundMode::CAST_NONE,
                                  static_cast<int32_t>(keep - base));
                }
            }
            if (useMax != 0) {
                AscendC::ReduceMax(red, row, tmp, static_cast<int32_t>(Skv));
                AscendC::PipeBarrier<PIPE_ALL>();
                const float rowMax = red.GetValue(0);
                AscendC::Adds(row, row, -rowMax, static_cast<int32_t>(Skv));
            }
            AscendC::Exp(row, row, static_cast<int32_t>(Skv));
            AscendC::ReduceSum(red, row, tmp, static_cast<int32_t>(Skv));
            AscendC::PipeBarrier<PIPE_ALL>();
            const float rowSum = red.GetValue(0);
            AscendC::Muls(row, row, 1.0f / rowSum, static_cast<int32_t>(Skv));
        }

        AscendC::Cast(quant, f32, AscendC::RoundMode::CAST_RINT, nElems);
        outQue.EnQue(quant);
        auto out = outQue.DeQue<T>();
        AscendC::DataCopyExtParams cpOut{static_cast<uint16_t>(cnt), rowBytes, 0u, 0u, 0u};
        AscendC::DataCopyPad(scG[g0 * Skv], out, cpOut);
        outQue.FreeTensor(out);
    }
}

extern "C" void launch_gqa_softmax_f16(GM_ADDR scores, int64_t totalRows, int64_t RS, int64_t G,
                                       int64_t S, int64_t Skv, float scale, int32_t causal,
                                       int32_t useMax, int64_t nBlocks, int64_t rowsPerCore,
                                       void *stream)
{
    gqa_softmax_kernel<half><<<nBlocks, nullptr, stream>>>(scores, totalRows, RS, G, S, Skv, scale,
                                                           causal, useMax, rowsPerCore);
}

extern "C" void launch_gqa_softmax_bf16(GM_ADDR scores, int64_t totalRows, int64_t RS, int64_t G,
                                        int64_t S, int64_t Skv, float scale, int32_t causal,
                                        int32_t useMax, int64_t nBlocks, int64_t rowsPerCore,
                                        void *stream)
{
    gqa_softmax_kernel<bfloat16_t><<<nBlocks, nullptr, stream>>>(scores, totalRows, RS, G, S, Skv,
                                                                 scale, causal, useMax, rowsPerCore);
}
