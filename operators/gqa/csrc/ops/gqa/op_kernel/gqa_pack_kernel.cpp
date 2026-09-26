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
 * \file gqa_pack_kernel.cpp
 * \brief GQA stage 1 (AIV): gather the grouped query/key/value into per (batch, kv-head) contiguous
 *        matrices so that the cube stages can consume plain [rows, D] operands.
 *
 *   Qp[bg][i*S + s][:] = q[b, s, g*G + i, :]        bg = b*Nkv + g
 *   Kp[bg][j][:]       = k[b, j, g, :]
 *   Vp[bg][j][:]       = v[b, j, g, :]
 *
 * Every one of the three copies is a 2D strided DMA: `blockCount` rows of D contiguous elements
 * whose source rows are separated by a constant gap.  DataCopyPad is used with the units that the
 * platform runbooks agree on: the Global-Memory side stride field is a byte count and the Unified
 * Buffer side stride field counts 32-byte blocks.  Because the UB rows are written back to back and
 * every blockLen here is a multiple of 32 bytes, the UB stride is always zero and the two possible
 * readings of the field cannot diverge.
 *
 * The staging buffer is walked in row chunks bounded by the UB budget; each chunk is one copy in and
 * one copy out, and an explicit PipeBarrier<PIPE_ALL> separates them so the MTE3 read of the staging
 * buffer can never overtake the MTE2 fill (the staging tensor lives in a VECIN queue, whose queue
 * events only order MTE2 against the vector pipe).
 */

#include <algorithm>

#include "kernel_operator.h"
#include "gqa_launch.h"

using namespace AscendC;

namespace {
constexpr int32_t GQA_PACK_DEPTH = 2;
constexpr int32_t GQA_PACK_BYTES = 32 * 1024;  /* staging budget per slot */
}  // namespace

template <typename T>
__aicore__ inline void GqaPackCopy(AscendC::TQue<AscendC::QuePosition::VECIN, GQA_PACK_DEPTH> &que,
                                   const AscendC::GlobalTensor<T> &src, int64_t srcOff,
                                   int64_t srcGapElems, const AscendC::GlobalTensor<T> &dst,
                                   int64_t dstOff, int64_t dstGapElems, int64_t rows, int64_t len,
                                   int64_t chunkRows)
{
    const uint32_t lenBytes = static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(T)));
    const uint32_t srcGapBytes = static_cast<uint32_t>(srcGapElems * static_cast<int64_t>(sizeof(T)));
    const uint32_t dstGapBytes = static_cast<uint32_t>(dstGapElems * static_cast<int64_t>(sizeof(T)));
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    int64_t done = 0;
    while (done < rows) {
        int64_t chunk = rows - done;
        if (chunk > chunkRows) {
            chunk = chunkRows;
        }
        auto staged = que.AllocTensor<T>();
        AscendC::DataCopyExtParams inParams{static_cast<uint16_t>(chunk), lenBytes, srcGapBytes, 0u, 0u};
        AscendC::DataCopyPad(staged, src[srcOff + done * (len + srcGapElems)], inParams, padParams);
        que.EnQue(staged);
        auto ready = que.DeQue<T>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams outParams{static_cast<uint16_t>(chunk), lenBytes, 0u, dstGapBytes,
                                             0u};
        AscendC::DataCopyPad(dst[dstOff + done * (len + dstGapElems)], ready, outParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        que.FreeTensor(ready);
        done += chunk;
    }
}

template <typename T>
__global__ __aicore__ void gqa_pack_kernel(GM_ADDR qAddr, GM_ADDR kAddr, GM_ADDR vAddr,
                                           GM_ADDR qpAddr, GM_ADDR kpAddr, GM_ADDR vpAddr,
                                           int64_t B, int64_t S, int64_t Nq, int64_t Nkv,
                                           int64_t Skv, int64_t D, int64_t chunkRows)
{
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t BG = B * Nkv;
    if (blk >= BG) {
        return;
    }
    const int64_t b = blk / Nkv;
    const int64_t g = blk % Nkv;
    const int64_t G = Nq / Nkv;
    const int64_t RS = G * S;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, GQA_PACK_DEPTH> que;
    pipe.InitBuffer(que, GQA_PACK_DEPTH, GQA_PACK_BYTES);

    AscendC::GlobalTensor<T> qG, kG, vG, qpG, kpG, vpG;
    qG.SetGlobalBuffer((__gm__ T *)qAddr);
    kG.SetGlobalBuffer((__gm__ T *)kAddr);
    vG.SetGlobalBuffer((__gm__ T *)vAddr);
    qpG.SetGlobalBuffer((__gm__ T *)qpAddr);
    kpG.SetGlobalBuffer((__gm__ T *)kpAddr);
    vpG.SetGlobalBuffer((__gm__ T *)vpAddr);

    /* --- query: one 2D copy per query-in-group index i, rows s strided by Nq*D --- */
    const int64_t qSrcBase = ((b * S) * Nq + g * G) * D;
    const int64_t qDstBase = blk * RS * D;
    const int64_t qSrcGap = (Nq - 1) * D;
    for (int64_t i = 0; i < G; ++i) {
        GqaPackCopy<T>(que, qG, qSrcBase + i * D, qSrcGap, qpG, qDstBase + (i * S) * D, 0, S, D,
                       chunkRows);
    }

    /* --- key / value: rows j strided by Nkv*D --- */
    const int64_t kvSrcBase = ((b * Skv) * Nkv + g) * D;
    const int64_t kvDstBase = blk * Skv * D;
    const int64_t kvSrcGap = (Nkv - 1) * D;
    GqaPackCopy<T>(que, kG, kvSrcBase, kvSrcGap, kpG, kvDstBase, 0, Skv, D, chunkRows);
    GqaPackCopy<T>(que, vG, kvSrcBase, kvSrcGap, vpG, kvDstBase, 0, Skv, D, chunkRows);
}

extern "C" void launch_gqa_pack_f16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qp, GM_ADDR kp,
                                    GM_ADDR vp, int64_t B, int64_t S, int64_t Nq, int64_t Nkv,
                                    int64_t Skv, int64_t D, int64_t chunkRows, void *stream)
{
    gqa_pack_kernel<half><<<B * Nkv, nullptr, stream>>>(q, k, v, qp, kp, vp, B, S, Nq, Nkv, Skv, D,
                                                        chunkRows);
}

extern "C" void launch_gqa_pack_bf16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qp, GM_ADDR kp,
                                     GM_ADDR vp, int64_t B, int64_t S, int64_t Nq, int64_t Nkv,
                                     int64_t Skv, int64_t D, int64_t chunkRows, void *stream)
{
    gqa_pack_kernel<bfloat16_t><<<B * Nkv, nullptr, stream>>>(q, k, v, qp, kp, vp, B, S, Nq, Nkv,
                                                              Skv, D, chunkRows);
}
