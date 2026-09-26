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
 * \file mhc_helpers.h
 * \brief Device-side helpers of the MhcSinkhorn kernel.  Included by the kernel only
 *        (the plugin must not see them), after "kernel_operator.h".
 *
 * All helpers work on the column-chunked layout  X[c][i][r][lane]:
 *   c    column chunk, lane = j - 8*c
 *   i    row index inside the matrix
 *   r    matrix index inside the tile
 * so one (c,i,r) triple is exactly one 32B DataBlock.
 *
 * Everything here is a plain vector call on VECCALC buffers.  A helper only
 * inserts the PipeBarrier<PIPE_V> that its own consumer genuinely needs; there
 * is deliberately NO barrier inside the loops whose iterations are independent
 * of each other (the m column multiplies of one chunk all read the same C and
 * write disjoint X slices, and the per-chunk Brcb/Brcb/Mul groups are
 * independent as well) - a barrier per iteration there would be pure overhead.
 */

#ifndef MHC_HELPERS_H
#define MHC_HELPERS_H

namespace mhcdev {

/* -inf without touching the host */
__aicore__ inline float MhcNegInf()
{
    union { unsigned int u; float f; } v;
    v.u = 0xFF800000u;
    return v.f;
}

/* one MAX per 32B DataBlock, written compactly (cnt/8 values) */
__aicore__ inline void MhcRowMax(const AscendC::LocalTensor<float>& dst,
                                 const AscendC::LocalTensor<float>& src, int32_t cnt)
{
    AscendC::BlockReduceMax<float>(dst, src, (int32_t)(cnt >> 6), (int32_t)64, (int32_t)1,
                                   (int32_t)1, (int32_t)8);
    AscendC::PipeBarrier<PIPE_V>();
}

/* one SUM per 32B DataBlock, written compactly (cnt/8 values) */
__aicore__ inline void MhcRowSum(const AscendC::LocalTensor<float>& dst,
                                 const AscendC::LocalTensor<float>& src, int32_t cnt)
{
    AscendC::BlockReduceSum<float>(dst, src, (int32_t)(cnt >> 6), (int32_t)64, (int32_t)1,
                                   (int32_t)1, (int32_t)8);
    AscendC::PipeBarrier<PIPE_V>();
}

/* broadcast S[0..sz-1] (one value per 8-lane row) over every column chunk.
   The chunks are written independently, so only the last Brcb needs a barrier. */
__aicore__ inline void MhcBroadcast(const AscendC::LocalTensor<float>& dst,
                                    const AscendC::LocalTensor<float>& src, int32_t sz, int32_t nc)
{
    AscendC::BrcbRepeatParams bp(1, 8);
    const int32_t reps = sz >> 3;
    for (int32_t c = 0; c < nc; ++c) {
        AscendC::Brcb(dst[c * sz * 8], src, (uint8_t)reps, bp);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

/* Set every padding lane (lane >= w of each 32B row) of one chunk to 0.
   The bitwise mask selects exactly those lanes, so the real lanes keep the
   `+ eps` that was written over the whole tile just before. */
__aicore__ inline void MhcClearPadChunk(const AscendC::LocalTensor<float>& X, int32_t w, int32_t cnt)
{
    const uint64_t bytePat = (((uint64_t)0xFFu) << (uint32_t)w) & 0xFFu;
    uint64_t pat = 0;
    for (int32_t b = 0; b < 8; ++b) {
        pat |= (bytePat << (8 * b));
    }
    uint64_t mp[2] = {pat, pat};
    AscendC::Duplicate<float>(X, 0.0f, mp, cnt >> 6, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
}

/* w is < 8 only for the last chunk; earlier chunks are completely full */
__aicore__ inline void MhcClearPad(const AscendC::LocalTensor<float>& X, int32_t m, int32_t NC,
                                   int32_t RB)
{
    for (int32_t c = 0; c < NC; ++c) {
        int32_t w = m - 8 * c;
        if (w > 8) {
            w = 8;
        }
        if (w < 8) {
            MhcClearPadChunk(X[c * m * RB], w, m * RB);
        }
    }
}

/* column sums: C[c*RB + r*8 + lane] = sum over i of X[c*m*RB + i*RB + r*8 + lane].
   The m-1 accumulation steps are a genuine dependency chain, so each one needs
   its own barrier; the last one is left to the caller. */
__aicore__ inline void MhcColSum(const AscendC::LocalTensor<float>& C,
                                 const AscendC::LocalTensor<float>& X, int32_t m, int32_t NC,
                                 int32_t RB)
{
    for (int32_t c = 0; c < NC; ++c) {
        AscendC::Add(C[c * RB], X[c * m * RB], X[(c * m + 1) * RB], RB);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t i = 2; i < m; ++i) {
            AscendC::Add(C[c * RB], C[c * RB], X[(c * m + i) * RB], RB);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

/* column normalisation: X[c][i][r][lane] *= C[c][r][lane].
   Every multiply of one chunk reads the same C slice and writes a different X
   slice, so the m multiplies are independent and need no barrier between them;
   one barrier after the chunk is enough for the next consumer. */
__aicore__ inline void MhcColScale(const AscendC::LocalTensor<float>& X,
                                   const AscendC::LocalTensor<float>& C, int32_t m, int32_t NC,
                                   int32_t RB)
{
    for (int32_t c = 0; c < NC; ++c) {
        for (int32_t i = 0; i < m; ++i) {
            AscendC::Mul(X[(c * m + i) * RB], X[(c * m + i) * RB], C[c * RB], RB);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

} // namespace mhcdev

#endif /* MHC_HELPERS_H */
