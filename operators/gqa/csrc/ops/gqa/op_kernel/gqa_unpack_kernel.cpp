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
 * \file gqa_unpack_kernel.cpp
 * \brief GQA stage 5 (AIV): scatter the packed output tiles back into the [B, S, Nq, D] tensor.
 *
 *   y[b, s, g*G + i, :] = Yp[b*Nkv + g][i*S + s][:]
 *
 * For a fixed (b, g, i) the source is S contiguous rows of D elements and the destination rows are
 * Nq*D elements apart, so this is again a 2D strided DMA in row chunks.
 */

#include <algorithm>

#include "kernel_operator.h"
#include "gqa_launch.h"

using namespace AscendC;

namespace {
constexpr int32_t GQA_UNPACK_DEPTH = 2;
constexpr int32_t GQA_UNPACK_BYTES = 32 * 1024;
}  // namespace

template <typename T>
__global__ __aicore__ void gqa_unpack_kernel(GM_ADDR ypAddr, GM_ADDR yAddr, int64_t B, int64_t S,
                                             int64_t Nq, int64_t Nkv, int64_t D, int64_t chunkRows)
{
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t G = Nq / Nkv;
    const int64_t BG = B * Nkv;
    if (blk >= BG * G) {
        return;
    }
    const int64_t bg = blk / G;
    const int64_t i = blk % G;
    const int64_t b = bg / Nkv;
    const int64_t g = bg % Nkv;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, GQA_UNPACK_DEPTH> que;
    pipe.InitBuffer(que, GQA_UNPACK_DEPTH, GQA_UNPACK_BYTES);

    AscendC::GlobalTensor<T> ypG, yG;
    ypG.SetGlobalBuffer((__gm__ T *)ypAddr);
    yG.SetGlobalBuffer((__gm__ T *)yAddr);

    const int64_t RS = G * S;
    const int64_t srcBase = (bg * RS + i * S) * D;
    const int64_t dstBase = ((b * S) * Nq + g * G + i) * D;
    const int64_t dstGap = (Nq - 1) * D;

    const uint32_t lenBytes = static_cast<uint32_t>(D * static_cast<int64_t>(sizeof(T)));
    const uint32_t dstGapBytes = static_cast<uint32_t>(dstGap * static_cast<int64_t>(sizeof(T)));

    int64_t done = 0;
    while (done < S) {
        int64_t chunk = S - done;
        if (chunk > chunkRows) {
            chunk = chunkRows;
        }
        auto staged = que.AllocTensor<T>();
        AscendC::DataCopyExtParams inParams{static_cast<uint16_t>(chunk), lenBytes, 0u, 0u, 0u};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(staged, ypG[srcBase + done * D], inParams, padParams);
        que.EnQue(staged);
        auto ready = que.DeQue<T>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams outParams{static_cast<uint16_t>(chunk), lenBytes, 0u, dstGapBytes,
                                             0u};
        AscendC::DataCopyPad(yG[dstBase + done * Nq * D], ready, outParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        que.FreeTensor(ready);
        done += chunk;
    }
}

extern "C" void launch_gqa_unpack_f16(GM_ADDR yp, GM_ADDR y, int64_t B, int64_t S, int64_t Nq,
                                      int64_t Nkv, int64_t D, int64_t chunkRows, void *stream)
{
    gqa_unpack_kernel<half><<<B * Nq, nullptr, stream>>>(yp, y, B, S, Nq, Nkv, D, chunkRows);
}

extern "C" void launch_gqa_unpack_bf16(GM_ADDR yp, GM_ADDR y, int64_t B, int64_t S, int64_t Nq,
                                       int64_t Nkv, int64_t D, int64_t chunkRows, void *stream)
{
    gqa_unpack_kernel<bfloat16_t><<<B * Nq, nullptr, stream>>>(yp, y, B, S, Nq, Nkv, D, chunkRows);
}
