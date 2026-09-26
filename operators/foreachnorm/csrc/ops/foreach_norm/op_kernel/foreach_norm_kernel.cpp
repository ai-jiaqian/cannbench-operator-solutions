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
 * \file foreach_norm_kernel.cpp
 * \brief ForeachNorm device kernels + host tiling/launch (bisheng + -xasc), dav-2201.
 *
 * y_i = (sum_j |x_i[j]|^p)^(1/p)   (p == 1, 2, +-inf, 0 handled as special cases)
 *
 * Two kernels per input tensor, launched back-to-back on the same stream:
 *   1) foreach_norm_reduce_kernel<T>   : grid = numBlocks, one contiguous chunk per block,
 *                                        writes numBlocks fp32 partial scalars into workspace.
 *   2) foreach_norm_finalize_kernel<T> : grid = 1, reduces the partials, applies the p-root,
 *                                        casts to the output dtype and writes one scalar.
 *
 * All numerics run on device in fp32; fp16/bf16 inputs are widened with Cast(CAST_NONE).
 */

#include <type_traits>
#include <algorithm>
#include <tuple>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "foreach_norm_launch.h"

namespace {
constexpr int64_t FN_DEPTH = 2;
// All simultaneously live UB bytes per element (fp32 worst case):
//   qRaw: 2 * sizeof(T)          -> 8 for fp32, 4 for fp16/bf16
//   bF/bT/bAcc: 3 * 4B           -> 12
// fp32 worst case = 20 B/elem.
constexpr int64_t FN_BYTES_PER_ELEM = 20;
constexpr uint32_t FN_RED_BYTES = 4096;  // ReduceSum/ReduceMax/ReduceMin shared tmp
constexpr float FN_MAXF = 3.4028234663852886e+38f;
constexpr float FN_L0_SCALE = 1.0e30f;
constexpr int64_t FN_MIN_ELEMS_PER_BLOCK = 4096;
}  // namespace

// ---------------------------------------------------------------------------------------------
// Stage 1: per-block partial reduction
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void foreach_norm_reduce_kernel(GM_ADDR x, GM_ADDR ws, int64_t totalLength,
                                                      int64_t blockLength, uint32_t tileElems, float p,
                                                      int32_t mode)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, FN_DEPTH> qRaw;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qOut;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bF;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bT;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bAcc;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bRed;

    pipe.InitBuffer(qRaw, FN_DEPTH, tileElems * (uint32_t)sizeof(T));
    pipe.InitBuffer(qOut, 1, 64);
    pipe.InitBuffer(bF, tileElems * 4u);
    pipe.InitBuffer(bT, tileElems * 4u);
    pipe.InitBuffer(bAcc, tileElems * 4u);
    pipe.InitBuffer(bRed, FN_RED_BYTES);

    int64_t blk = (int64_t)AscendC::GetBlockIdx();
    AscendC::GlobalTensor<float> wsGm;
    wsGm.SetGlobalBuffer((__gm__ float *)ws + blk);

    const bool isMin = (mode == FN_MODE_MIN);
    const float neutral = isMin ? FN_MAXF : 0.0f;

    AscendC::LocalTensor<float> outL = qOut.AllocTensor<float>();

    int64_t start = blk * blockLength;
    int64_t remaining = totalLength - start;
    if (remaining > blockLength) {
        remaining = blockLength;
    }

    if (remaining <= 0) {
        // Idle block: publish the accumulation neutral so the finalize stage is correct.
        AscendC::Duplicate(outL, neutral, 8);
        qOut.EnQue(outL);
        outL = qOut.DeQue<float>();
        AscendC::DataCopyExtParams wp{1, 4u, 0, 0, 0};
        AscendC::DataCopyPad(wsGm, outL, wp);
        qOut.FreeTensor(outL);
        return;
    }

    AscendC::GlobalTensor<T> xGm;
    xGm.SetGlobalBuffer((__gm__ T *)x + start);

    AscendC::LocalTensor<float> fBuf = bF.Get<float>();
    AscendC::LocalTensor<float> tBuf = bT.Get<float>();
    AscendC::LocalTensor<float> acc = bAcc.Get<float>();
    AscendC::LocalTensor<float> red = bRed.Get<float>();

    AscendC::Duplicate(acc, neutral, (int32_t)tileElems);

    int64_t offset = 0;
    while (offset < remaining) {
        int64_t cnt64 = remaining - offset;
        if (cnt64 > (int64_t)tileElems) {
            cnt64 = (int64_t)tileElems;
        }
        int32_t cnt = (int32_t)cnt64;

        AscendC::DataCopyExtParams cp{1, (uint32_t)(cnt64 * (int64_t)sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};
        AscendC::LocalTensor<T> raw = qRaw.AllocTensor<T>();
        if (cnt64 == (int64_t)tileElems) {
            // Full tile: both the GM offset (block start rounded to 16 elements) and the byte
            // length are 32B aligned, so the plain (faster) continuous DataCopy path applies.
            AscendC::DataCopy(raw, xGm[offset], tileElems);
        } else {
            AscendC::DataCopyPad(raw, xGm[offset], cp, pp);
        }
        qRaw.EnQue(raw);
        raw = qRaw.DeQue<T>();

        AscendC::LocalTensor<float> srcF;
        if constexpr (std::is_same<T, float>::value) {
            srcF = raw;
        } else {
            srcF = tBuf;
            AscendC::Cast(tBuf, raw, AscendC::RoundMode::CAST_NONE, (uint32_t)cnt);
        }

        // Accumulate the per-element contribution of the current mode into `acc`.
        // L2 uses a single fused multiply-accumulate (strictly fewer UB accesses).
        bool directAcc = false;
        if (mode == FN_MODE_L2) {
            AscendC::MulAddDst(acc, srcF, srcF, cnt);
            directAcc = true;
        } else if (mode == FN_MODE_L1) {
            AscendC::Abs(fBuf, srcF, cnt);
        } else if (mode == FN_MODE_MAX || mode == FN_MODE_MIN) {
            AscendC::Abs(fBuf, srcF, cnt);
        } else if (mode == FN_MODE_L0) {
            AscendC::Abs(fBuf, srcF, cnt);
            AscendC::Muls(tBuf, fBuf, FN_L0_SCALE, cnt);
            AscendC::Mins(fBuf, tBuf, 1.0f, cnt);
        } else {
            AscendC::Abs(fBuf, srcF, cnt);
            AscendC::Ln(tBuf, fBuf, cnt);          // tBuf = ln|x|
            AscendC::Muls(fBuf, tBuf, p, cnt);     // fBuf = p * ln|x|
            AscendC::Exp(tBuf, fBuf, cnt);         // tBuf = |x|^p
        }
        qRaw.FreeTensor(raw);

        if (!directAcc) {
            if (mode == FN_MODE_MAX) {
                AscendC::Max(acc, acc, fBuf, cnt);
            } else if (mode == FN_MODE_MIN) {
                AscendC::Min(acc, acc, fBuf, cnt);
            } else if (mode == FN_MODE_GENERAL) {
                AscendC::Add(acc, acc, tBuf, cnt);
            } else {
                AscendC::Add(acc, acc, fBuf, cnt);
            }
        }
        offset += cnt64;
    }

    if (mode == FN_MODE_MAX) {
        AscendC::ReduceMax(outL, acc, red, (int32_t)tileElems);
    } else if (mode == FN_MODE_MIN) {
        AscendC::ReduceMin(outL, acc, red, (int32_t)tileElems);
    } else {
        AscendC::ReduceSum(outL, acc, red, (int32_t)tileElems);
    }

    qOut.EnQue(outL);
    outL = qOut.DeQue<float>();
    AscendC::DataCopyExtParams wp{1, 4u, 0, 0, 0};
    AscendC::DataCopyPad(wsGm, outL, wp);
    qOut.FreeTensor(outL);
}

// ---------------------------------------------------------------------------------------------
// Stage 2: single-block finalize for the WHOLE tensor list (p-root + dtype cast)
// ---------------------------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void foreach_norm_finalize_kernel(GM_ADDR ws, GM_ADDR y, int64_t listLen, int64_t numBlocks,
                                                        float p, int32_t mode)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qIn;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qOut;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bRed;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bScr;

    pipe.InitBuffer(qIn, 1, (uint32_t)(numBlocks * 4));
    pipe.InitBuffer(qOut, 1, 64);
    pipe.InitBuffer(qY, 1, 64);
    pipe.InitBuffer(bRed, FN_RED_BYTES);
    pipe.InitBuffer(bScr, 64);

    AscendC::GlobalTensor<float> wsGm;
    wsGm.SetGlobalBuffer((__gm__ float *)ws);
    AscendC::GlobalTensor<T> yGm;
    yGm.SetGlobalBuffer((__gm__ T *)y);

    AscendC::LocalTensor<float> red = bRed.Get<float>();
    AscendC::LocalTensor<float> scr = bScr.Get<float>();

    for (int64_t i = 0; i < listLen; ++i) {
        AscendC::LocalTensor<float> inL = qIn.AllocTensor<float>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(numBlocks * 4), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pp{false, 0, 0, 0};
        AscendC::DataCopyPad(inL, wsGm[i * numBlocks], cp, pp);
        qIn.EnQue(inL);
        inL = qIn.DeQue<float>();

        AscendC::LocalTensor<float> outL = qOut.AllocTensor<float>();
        if (mode == FN_MODE_MAX) {
            AscendC::ReduceMax(outL, inL, red, (int32_t)numBlocks);
        } else if (mode == FN_MODE_MIN) {
            AscendC::ReduceMin(outL, inL, red, (int32_t)numBlocks);
        } else {
            AscendC::ReduceSum(outL, inL, red, (int32_t)numBlocks);
        }
        qIn.FreeTensor(inL);

        // Apply the p-th root where needed.
        if (mode == FN_MODE_L2) {
            AscendC::Sqrt(scr, outL, 1);
            AscendC::Adds(outL, scr, 0.0f, 1);
        } else if (mode == FN_MODE_GENERAL) {
            AscendC::Ln(scr, outL, 1);
            AscendC::Muls(scr, scr, 1.0f / p, 1);
            AscendC::Exp(outL, scr, 1);
        }

        if constexpr (std::is_same<T, float>::value) {
            qOut.EnQue(outL);
            outL = qOut.DeQue<float>();
            AscendC::DataCopyExtParams yp{1, 4u, 0, 0, 0};
            AscendC::DataCopyPad(yGm[i], outL, yp);
            qOut.FreeTensor(outL);
        } else {
            AscendC::LocalTensor<T> outT = qY.AllocTensor<T>();
            AscendC::Cast(outT, outL, AscendC::RoundMode::CAST_RINT, (uint32_t)1);
            qY.EnQue(outT);
            outT = qY.DeQue<T>();
            AscendC::DataCopyExtParams yp{1, (uint32_t)sizeof(T), 0, 0, 0};
            AscendC::DataCopyPad(yGm[i], outT, yp);
            qY.FreeTensor(outT);
            qOut.FreeTensor(outL);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host tiling
// ---------------------------------------------------------------------------------------------
ForeachNormTiling calc_foreach_norm_tiling(int64_t totalLength)
{
    ForeachNormTiling tiling;
    tiling.numBlocks = 1;
    tiling.blockLength = (totalLength > 0) ? totalLength : 1;
    tiling.tileElems = 1024;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    int64_t coreNum = 1;
    if (ascendcPlatform != nullptr) {
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        coreNum = (int64_t)ascendcPlatform->GetCoreNumAiv();
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    if (ubSize < 65536) {
        ubSize = 196608;
    }

    int64_t numBlocks = (totalLength + FN_MIN_ELEMS_PER_BLOCK - 1) / FN_MIN_ELEMS_PER_BLOCK;
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    if (numBlocks > coreNum) {
        numBlocks = coreNum;
    }
    // Keep numBlocks a multiple of 8 so the finalize ReduceSum needs no padding.
    numBlocks = ((numBlocks + 7) / 8) * 8;
    if (numBlocks < 8) {
        numBlocks = 8;
    }

    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    if (blockLength < 1) {
        blockLength = 1;
    }
    // Round the per-block span up to a multiple of 16 elements so every block start is 32B
    // aligned for both fp32 (64B) and fp16/bf16 (32B), enabling the aligned DataCopy path.
    blockLength = ((blockLength + 15) / 16) * 16;

    uint64_t avail = (ubSize > 65536) ? (ubSize - 32768) : (ubSize / 2);
    uint64_t maxByUb = avail / (uint64_t)FN_BYTES_PER_ELEM;
    uint32_t tileElems = (uint32_t)((maxByUb / 64) * 64);
    if (tileElems > 8192) {
        tileElems = 8192;
    }
    if (tileElems < 64) {
        tileElems = 64;
    }

    tiling.numBlocks = numBlocks;
    tiling.blockLength = blockLength;
    tiling.tileElems = tileElems;
    return tiling;
}

// ---------------------------------------------------------------------------------------------
// Launch wrappers (callable from g++)
// ---------------------------------------------------------------------------------------------
extern "C" {

void launch_foreach_norm_reduce_float(GM_ADDR x, GM_ADDR ws, int64_t totalLength, int64_t blockLength,
                                      uint32_t tileElems, float p, int32_t mode, int64_t numBlocks, void *stream)
{
    foreach_norm_reduce_kernel<float><<<numBlocks, nullptr, stream>>>(x, ws, totalLength, blockLength, tileElems, p,
                                                                      mode);
}

void launch_foreach_norm_reduce_half(GM_ADDR x, GM_ADDR ws, int64_t totalLength, int64_t blockLength,
                                     uint32_t tileElems, float p, int32_t mode, int64_t numBlocks, void *stream)
{
    foreach_norm_reduce_kernel<half><<<numBlocks, nullptr, stream>>>(x, ws, totalLength, blockLength, tileElems, p,
                                                                     mode);
}

void launch_foreach_norm_reduce_bf16(GM_ADDR x, GM_ADDR ws, int64_t totalLength, int64_t blockLength,
                                     uint32_t tileElems, float p, int32_t mode, int64_t numBlocks, void *stream)
{
    foreach_norm_reduce_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, ws, totalLength, blockLength, tileElems,
                                                                           p, mode);
}

void launch_foreach_norm_finalize_float(GM_ADDR ws, GM_ADDR y, int64_t listLen, int64_t numBlocks, float p,
                                        int32_t mode, void *stream)
{
    foreach_norm_finalize_kernel<float><<<1, nullptr, stream>>>(ws, y, listLen, numBlocks, p, mode);
}

void launch_foreach_norm_finalize_half(GM_ADDR ws, GM_ADDR y, int64_t listLen, int64_t numBlocks, float p,
                                       int32_t mode, void *stream)
{
    foreach_norm_finalize_kernel<half><<<1, nullptr, stream>>>(ws, y, listLen, numBlocks, p, mode);
}

void launch_foreach_norm_finalize_bf16(GM_ADDR ws, GM_ADDR y, int64_t listLen, int64_t numBlocks, float p,
                                       int32_t mode, void *stream)
{
    foreach_norm_finalize_kernel<bfloat16_t><<<1, nullptr, stream>>>(ws, y, listLen, numBlocks, p, mode);
}
}
