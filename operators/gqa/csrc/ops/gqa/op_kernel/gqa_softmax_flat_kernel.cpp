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
 * \file gqa_softmax_flat_kernel.cpp
 * \brief GQA stage 3 (AIV) of the flat pipeline: scaled softmax over the expanded score row.
 *
 * The flat pipeline evaluates one batch at a time directly on the operator's own tensors:
 *      A = Q[b]                    [S*Nq, D]        row r = s*Nq + h
 *      B = K[b]                    [Skv*Nkv, D]     row j = jj*Nkv + i
 *      scores = A @ B^T            [S*Nq, Skv*Nkv]
 *      P      = softmax(scores)    (normalised over the query's own kv head only)
 *      y[b]   = P @ V[b]           [S*Nq, D]        row r = s*Nq + h  (already the output layout)
 *
 * so the score row of query (s, h) holds the dot product of that query with *every* key of *every*
 * kv head.  Only the columns j' = j*Nkv + g(h), g(h) = h / G, belong to the query's own kv head, so
 * this stage normalises over that sub lattice and leaves exact zeros everywhere else; stage 4 then
 * multiplies the result by the flat V[b], which selects precisely those keys.  The whole pipeline
 * therefore needs no K/V packing, no query repacking and no output permutation - the operator's own
 * tensors are consumed and produced in place.
 *
 * Valid columns of row (s, h):
 *      keep = lim*Nkv + g + 1,   lim = causal ? s + (Skv - S) : Skv - 1
 *      j' < keep            and      j' % Nkv == g
 * The causal part is a prefix, so it is honoured by restricting the denominator and the divide to
 * [0, keep) and by writing exact zeros beyond it - no sentinel fill and no re-widening dance is
 * needed, because the output tile is a scratch buffer that is zeroed wholesale before the prefix is
 * cast into it.
 *
 * The periodic head mask is applied by subtracting a huge bias from every column whose index is not
 * congruent to g.  That bias vector is built once per kv head: a 64 element pattern is assembled from
 * scalar writes (its period Nkv divides 64, so it tiles exactly), transformed into {0, HUGE} with four
 * vector instructions and then replicated up to the chunk length by log2 copy doubling.  Because the
 * row length L = Skv*Nkv is a multiple of Nkv, the same pattern serves every chunk of every row.
 *
 * Rows are moved in tiles of R rows (sized so that a tile holds about 20 KB of operator-typed data)
 * so the DMA traffic is counted in tiles rather than rows; a decode shape has very narrow score rows,
 * and one DMA per row would be latency bound.
 *
 * Numerics mirror the reference exactly:
 *      scores_T = round_T(matmul)                      (done by the cube stage)
 *      scores_T = round_T(scores_T * scale)            (Cast-to-T / Cast-to-float round trip below)
 *      p        = round_T(softmax_fp32(scores_T))      (fp32 throughout, CAST_RINT at the end)
 * The row maximum is not subtracted: the host only selects this path when |scaleValue| * D <= 60, so
 * exp() provably cannot overflow (softmax is shift invariant, so this is mathematically identical).
 */

#include "kernel_operator.h"
#include "gqa_launch.h"

using namespace AscendC;

namespace {
constexpr float GQA_SF_BIG = 1.0e30f;
constexpr int64_t GQA_SF_PAT = 64;           /* scalar-built head pattern period; Nkv divides it */
constexpr int64_t GQA_SF_ROWBUF = 20 * 1024; /* operator-typed bytes staged per tile */
}  // namespace

template <typename T>
__global__ __aicore__ void gqa_softmax_flat_kernel(GM_ADDR scAddr, int64_t totalRows, int64_t S,
                                                   int64_t Nq, int64_t G, int64_t Nkv, int64_t Skv,
                                                   float scale, int32_t causal,
                                                   int64_t rowsPerCore)
{
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t r0 = blk * rowsPerCore;
    if (r0 >= totalRows) {
        return;
    }
    int64_t r1 = r0 + rowsPerCore;
    if (r1 > totalRows) {
        r1 = totalRows;
    }

    const int64_t L = Skv * Nkv;
    const int64_t esz = static_cast<int64_t>(sizeof(T));
    int64_t R = GQA_SF_ROWBUF / (L * esz);
    if (R < 1) {
        R = 1;
    }

    /* Working chunk: a multiple of 16 (32 byte alignment for both fp32 and the 2-byte operator type)
     * and of Nkv, so the periodic head pattern is identical in every chunk. */
    int64_t step = (Nkv > 16) ? Nkv : 16;
    int64_t KQ = (1024 / step) * step;
    if (KQ < GQA_SF_PAT) {
        KQ = GQA_SF_PAT;
    }
    if (KQ > L) {
        KQ = L;
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> f32Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> quantBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> laneBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> redBuf;
    pipe.InitBuffer(inQue, 1, R * L * esz);
    pipe.InitBuffer(outQue, 1, R * L * esz);
    pipe.InitBuffer(f32Buf, R * L * sizeof(float));
    pipe.InitBuffer(quantBuf, KQ * esz);
    pipe.InitBuffer(biasBuf, KQ * sizeof(float));
    pipe.InitBuffer(laneBuf, GQA_SF_PAT * sizeof(float));
    pipe.InitBuffer(tmpBuf, KQ * sizeof(float));
    pipe.InitBuffer(redBuf, 64 * sizeof(float));

    auto f32 = f32Buf.Get<float>();
    auto quant = quantBuf.Get<T>();
    auto bias = biasBuf.Get<float>();
    auto lane = laneBuf.Get<float>();
    auto tmp = tmpBuf.Get<float>();
    auto red = redBuf.Get<float>();
    auto acc = red[0];
    auto part = red[8];

    for (int64_t i = 0; i < GQA_SF_PAT; ++i) {
        lane.SetValue(i, static_cast<float>(i % Nkv));
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::GlobalTensor<T> scG;
    scG.SetGlobalBuffer((__gm__ T *)scAddr);
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    const int64_t SN = S * Nq;
    const int64_t limShift = Skv - S;
    int64_t curG = -1;

    for (int64_t t0 = r0; t0 < r1; t0 += R) {
        int64_t cnt = r1 - t0;
        if (cnt > R) {
            cnt = R;
        }
        const int32_t nElems = static_cast<int32_t>(cnt * L);
        const uint32_t rowBytes = static_cast<uint32_t>(L * esz);

        auto tin = inQue.AllocTensor<T>();
        AscendC::DataCopyExtParams cpIn{static_cast<uint16_t>(cnt), rowBytes, 0u, 0u, 0u};
        AscendC::DataCopyPad(tin, scG[t0 * L], cpIn, padParams);
        inQue.EnQue(tin);
        auto in = inQue.DeQue<T>();
        AscendC::Cast(f32, in, AscendC::RoundMode::CAST_NONE, nElems);
        inQue.FreeTensor(in);

        /* scores_T = round_T(scores_T * scale), then widen back to float. */
        AscendC::Muls(f32, f32, scale, nElems);
        for (int64_t off = 0; off < nElems; off += KQ) {
            const int64_t rem = nElems - off;
            const int32_t c = static_cast<int32_t>((KQ < rem) ? KQ : rem);
            AscendC::Cast(quant, f32[off], AscendC::RoundMode::CAST_RINT, c);
            AscendC::Cast(f32[off], quant, AscendC::RoundMode::CAST_NONE, c);
        }

        auto q = outQue.AllocTensor<T>();
        AscendC::Duplicate(q, static_cast<T>(0.0f), nElems);

        for (int64_t r = 0; r < cnt; ++r) {
            const int64_t posQ = (t0 + r) % SN;
            const int64_t sPos = posQ / Nq;
            const int64_t hPos = posQ % Nq;
            const int64_t gHead = hPos / G;
            int64_t lim = Skv - 1;
            if (causal != 0) {
                lim = sPos + limShift;
                if (lim > Skv - 1) {
                    lim = Skv - 1;
                }
                if (lim < 0) {
                    lim = 0;
                }
            }
            int64_t keep = lim * Nkv + gHead + 1;
            if (keep > L) {
                keep = L;
            }

            if (gHead != curG) {
                /* bias[j] = 0 for j % Nkv == gHead, HUGE otherwise; period Nkv divides GQA_SF_PAT. */
                const int32_t pat = static_cast<int32_t>(GQA_SF_PAT);
                AscendC::Adds(bias, lane, -static_cast<float>(gHead), pat);
                AscendC::Abs(bias, bias, pat);
                AscendC::Mins(bias, bias, 1.0f, pat);
                AscendC::Muls(bias, bias, GQA_SF_BIG, pat);
                int64_t len = GQA_SF_PAT;
                while (len < KQ) {
                    const int64_t dbl = len * 2;
                    const int64_t cp = (dbl <= KQ) ? len : (KQ - len);
                    AscendC::Muls(bias[len], bias[0], 1.0f, static_cast<int32_t>(cp));
                    len += cp;
                }
                curG = gHead;
            }

            auto row = f32[r * L];
            /* head mask: every column belonging to another kv head is pushed to -HUGE. */
            for (int64_t off = 0; off < L; off += KQ) {
                const int64_t rem = L - off;
                const int32_t c = static_cast<int32_t>((KQ < rem) ? KQ : rem);
                AscendC::Sub(row[off], row[off], bias, c);
            }
            AscendC::Exp(row, row, static_cast<int32_t>(L));

            /* denominator over the valid prefix [0, keep); the masked columns already contribute 0. */
            AscendC::Duplicate(acc, 0.0f, 8);
            for (int64_t off = 0; off < keep; off += KQ) {
                const int64_t rem = keep - off;
                const int32_t c = static_cast<int32_t>((KQ < rem) ? KQ : rem);
                AscendC::ReduceSum(part, row[off], tmp, c);
                AscendC::Add(acc, acc, part, 8);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            const float inv = 1.0f / acc.GetValue(0);
            for (int64_t off = 0; off < keep; off += KQ) {
                const int64_t rem = keep - off;
                const int32_t c = static_cast<int32_t>((KQ < rem) ? KQ : rem);
                AscendC::Muls(row[off], row[off], inv, c);
            }
            /* write back: probabilities over the prefix, exact zeros over the masked tail. */
            auto qr = q[r * L];
            for (int64_t off = 0; off < keep; off += KQ) {
                const int64_t rem = keep - off;
                const int32_t c = static_cast<int32_t>((KQ < rem) ? KQ : rem);
                AscendC::Cast(qr[off], row[off], AscendC::RoundMode::CAST_RINT, c);
            }
        }

        outQue.EnQue(q);
        auto qo = outQue.DeQue<T>();
        AscendC::DataCopyExtParams cpOut{static_cast<uint16_t>(cnt), rowBytes, 0u, 0u, 0u};
        AscendC::DataCopyPad(scG[t0 * L], qo, cpOut);
        outQue.FreeTensor(qo);
    }
}

extern "C" void launch_gqa_softmax_flat_f16(GM_ADDR scores, int64_t totalRows, int64_t S, int64_t Nq,
                                            int64_t G, int64_t Nkv, int64_t Skv, float scale,
                                            int32_t causal, int64_t nBlocks, int64_t rowsPerCore,
                                            void *stream)
{
    gqa_softmax_flat_kernel<half><<<nBlocks, nullptr, stream>>>(scores, totalRows, S, Nq, G, Nkv,
                                                                Skv, scale, causal, rowsPerCore);
}

extern "C" void launch_gqa_softmax_flat_bf16(GM_ADDR scores, int64_t totalRows, int64_t S,
                                             int64_t Nq, int64_t G, int64_t Nkv, int64_t Skv,
                                             float scale, int32_t causal, int64_t nBlocks,
                                             int64_t rowsPerCore, void *stream)
{
    gqa_softmax_flat_kernel<bfloat16_t><<<nBlocks, nullptr, stream>>>(
        scores, totalRows, S, Nq, G, Nkv, Skv, scale, causal, rowsPerCore);
}
