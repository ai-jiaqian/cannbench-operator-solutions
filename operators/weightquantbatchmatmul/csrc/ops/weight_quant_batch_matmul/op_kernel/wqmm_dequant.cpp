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
 * \file wqmm_dequant.cpp
 * \brief WeightQuantBatchMatmul: AIV weight dequantisation prologue (vector / AIV kernel).
 *
 *   wd[k, n] = ANTIQUANT(weight)[k, n] = (T(weight[k, n]) + offset[n]) * scale[n]
 *
 * Rounding chain per output dtype (matches the reference):
 *   float16 : int8 -> half (exact), the add and the multiply are evaluated in half and the half result is
 *             stored; this is bit-for-bit the reference chain `(w.to(T) + off) * scale`.
 *   bfloat16: the coefficients are bf16 and DAV_2201 has neither bf16 arithmetic nor a bf16<->half Cast, so
 *             int8 -> half -> fp32, the add and the multiply are evaluated in fp32 and the result is
 *             rounded to bf16 exactly once.  The deviation from the reference is bounded by the
 *             reference's own intermediate bf16 rounding, well inside the bf16 tolerance.
 *
 * Data movement: the K axis (rows) is split into contiguous row ranges, one per vector core, and each core
 * walks its rows once per column chunk, reusing that chunk's per-column coefficients for every row.
 *
 * Every transfer is a single *multi-row* DataCopyPad: blockCount = rows, blockLen = the chunk width in
 * bytes, and the row pitch is carried by the transfer itself.  For a GM operand a stride is expressed in
 * bytes, for a UB operand in 32 byte data blocks, so the GM side gets (N - cnt) [* element size] and the
 * UB side gets 0 (which means "the natural 32B aligned row pitch").  Each row of a block is then processed
 * as a plain contiguous range, so nothing is assumed about the UB pitch beyond the hardware's own
 * alignment rule.  Issuing one descriptor per *block* instead of one per row is the whole point of this
 * kernel: the previous per-row form spent most of its wall clock on MTE2/MTE3 round trips.
 */

#include "kernel_operator.h"
#include "wqmm_launch.h"

namespace {

constexpr int64_t WQMM_DQ_ALIGN = 32;
constexpr int64_t WQMM_DQ_BUDGET = 128 * 1024;
constexpr int64_t WQMM_DQ_MAX_ROWS = 64;

__aicore__ inline int64_t WqmmDqAlign32(int64_t bytes)
{
    return ((bytes + WQMM_DQ_ALIGN - 1) / WQMM_DQ_ALIGN) * WQMM_DQ_ALIGN;
}

template <typename T, bool IS_BF16>
__global__ __aicore__ void wqmm_dequant_kernel(GM_ADDR wPtr, GM_ADDR sPtr, GM_ADDR oPtr, GM_ADDR outPtr,
                                               int64_t K, int64_t N, int64_t hasOff, int64_t rowsPerCore,
                                               int64_t colTileArg)
{
    using namespace AscendC;
    const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t r0 = blk * rowsPerCore;
    if (r0 >= K || N <= 0 || colTileArg <= 0 || rowsPerCore <= 0) {
        return;
    }
    int64_t r1 = r0 + rowsPerCore;
    if (r1 > K) {
        r1 = K;
    }
    const int64_t colTile = (colTileArg > N) ? N : colTileArg;
    const int64_t esT = static_cast<int64_t>(sizeof(T));

    GlobalTensor<int8_t> wGm;
    GlobalTensor<T> sGm, oGm, yGm;
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(wPtr));
    sGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(sPtr));
    if (hasOff != 0) {
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(oPtr));
    }
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outPtr));

    /* ---- UB budget: two in-flight tiles of `rowBlock` rows plus the coefficients ---- */
    const int64_t coefTBytes = WqmmDqAlign32(colTile * esT);
    const int64_t coefFBytes = WqmmDqAlign32(colTile * 4);
    int64_t fixedBytes = 3 * coefTBytes;               // qS + qO + the half pivot
    if constexpr (IS_BF16) {
        fixedBytes += 3 * coefFBytes;                  // float scale + float offset + float pivot
    }
    const int64_t pitchI = WqmmDqAlign32(colTile);      // int8 row pitch, bytes (== elements)
    const int64_t pitchT = WqmmDqAlign32(colTile * esT); // T row pitch, bytes
    int64_t rowBlock = (WQMM_DQ_BUDGET - fixedBytes) / (2 * (pitchI + pitchT));
    if (rowBlock > WQMM_DQ_MAX_ROWS) {
        rowBlock = WQMM_DQ_MAX_ROWS;
    }
    if (rowBlock < 1) {
        rowBlock = 1;
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> qIn;     // int8 weight block
    TQue<QuePosition::VECOUT, 2> qOut;   // dequantised block
    TQue<QuePosition::VECIN, 1> qS;      // scale chunk
    TQue<QuePosition::VECIN, 1> qO;      // offset chunk
    TBuf<TPosition::VECCALC> bH;         // half pivot of one row
    TBuf<TPosition::VECCALC> bSF;        // float scale chunk (bf16 path)
    TBuf<TPosition::VECCALC> bOF;        // float offset chunk (bf16 path)
    TBuf<TPosition::VECCALC> bF;         // float pivot of one row (bf16 path)

    pipe.InitBuffer(qIn, 2, rowBlock * pitchI);
    pipe.InitBuffer(qOut, 2, rowBlock * pitchT);
    pipe.InitBuffer(qS, 1, coefTBytes);
    pipe.InitBuffer(qO, 1, coefTBytes);
    pipe.InitBuffer(bH, coefTBytes);
    if constexpr (IS_BF16) {
        pipe.InitBuffer(bSF, coefFBytes);
        pipe.InitBuffer(bOF, coefFBytes);
        pipe.InitBuffer(bF, coefFBytes);
    }

    DataCopyPadExtParams<int8_t> wPad{false, 0, 0, 0};
    DataCopyPadExtParams<T> tPad{false, 0, 0, 0};

    for (int64_t n0 = 0; n0 < N; n0 += colTile) {
        int64_t cnt = N - n0;
        if (cnt > colTile) {
            cnt = colTile;
        }
        const uint32_t cntU = static_cast<uint32_t>(cnt);
        const int64_t pI = WqmmDqAlign32(cnt);                 // int8 row pitch of this chunk
        const int64_t pTb = WqmmDqAlign32(cnt * esT);          // T row pitch, bytes
        const int64_t pTe = pTb / esT;                         // T row pitch, elements

        /* per-column coefficients of this chunk, fetched once and reused for every row */
        auto sLoc = qS.AllocTensor<T>();
        const DataCopyExtParams sParams{1, static_cast<uint32_t>(cnt * esT), 0, 0, 0};
        DataCopyPad(sLoc, sGm[n0], sParams, tPad);
        qS.EnQue(sLoc);
        LocalTensor<T> sRaw = qS.DeQue<T>();

        LocalTensor<T> oRaw;
        if (hasOff != 0) {
            auto oLoc = qO.AllocTensor<T>();
            DataCopyPad(oLoc, oGm[n0], sParams, tPad);
            qO.EnQue(oLoc);
            oRaw = qO.DeQue<T>();
        }

        LocalTensor<float> sFloat, oFloat;
        if constexpr (IS_BF16) {
            sFloat = bSF.Get<float>();
            Cast(sFloat, sRaw, RoundMode::CAST_NONE, cntU);
            qS.FreeTensor(sRaw);
            if (hasOff != 0) {
                oFloat = bOF.Get<float>();
                Cast(oFloat, oRaw, RoundMode::CAST_NONE, cntU);
                qO.FreeTensor(oRaw);
            }
        }

        for (int64_t rr = r0; rr < r1; rr += rowBlock) {
            int64_t rb = r1 - rr;
            if (rb > rowBlock) {
                rb = rowBlock;
            }
            const uint16_t rbU = static_cast<uint16_t>(rb);

            auto wLoc = qIn.AllocTensor<int8_t>();
            const DataCopyExtParams wp{rbU, static_cast<uint32_t>(cnt),
                                       static_cast<uint32_t>(N - cnt), 0, 0};
            DataCopyPad(wLoc, wGm[rr * N + n0], wp, wPad);
            qIn.EnQue(wLoc);
            LocalTensor<int8_t> wTile = qIn.DeQue<int8_t>();

            auto yLoc = qOut.AllocTensor<T>();
            for (int64_t r = 0; r < rb; ++r) {
                const int32_t iOff = static_cast<int32_t>(r * pI);
                const int32_t oOff = static_cast<int32_t>(r * pTe);
                if constexpr (IS_BF16) {
                    LocalTensor<half> hRow = bH.Get<half>();
                    LocalTensor<float> fRow = bF.Get<float>();
                    Cast(hRow, wTile[iOff], RoundMode::CAST_NONE, cntU);
                    Cast(fRow, hRow, RoundMode::CAST_NONE, cntU);
                    if (hasOff != 0) {
                        Add(fRow, fRow, oFloat, cntU);
                    }
                    Mul(fRow, fRow, sFloat, cntU);
                    Cast(yLoc[oOff], fRow, RoundMode::CAST_RINT, cntU);
                } else {
                    LocalTensor<half> hRow = bH.Get<half>();
                    Cast(hRow, wTile[iOff], RoundMode::CAST_NONE, cntU);
                    if (hasOff != 0) {
                        Add(hRow, hRow, oRaw, cntU);
                    }
                    Mul(yLoc[oOff], hRow, sRaw, cntU);
                }
            }
            qIn.FreeTensor(wTile);

            qOut.EnQue(yLoc);
            LocalTensor<T> yTile = qOut.DeQue<T>();
            const DataCopyExtParams yp{rbU, static_cast<uint32_t>(cnt * esT), 0,
                                       static_cast<uint32_t>((N - cnt) * esT), 0};
            DataCopyPad(yGm[rr * N + n0], yTile, yp);
            qOut.FreeTensor(yTile);
        }

        if constexpr (!IS_BF16) {
            qS.FreeTensor(sRaw);
            if (hasOff != 0) {
                qO.FreeTensor(oRaw);
            }
        }
    }
}

} // namespace

extern "C" {

void launch_wqmm_dequant_half(GM_ADDR w, GM_ADDR s, GM_ADDR o, GM_ADDR out,
                              int64_t K, int64_t N, int64_t hasOff, int64_t numCores,
                              int64_t rowsPerCore, int64_t colTile, void *stream)
{
    if (numCores <= 0 || rowsPerCore <= 0) {
        return;
    }
    wqmm_dequant_kernel<half, false><<<static_cast<uint32_t>(numCores), nullptr, stream>>>(
        w, s, o, out, K, N, hasOff, rowsPerCore, colTile);
}

void launch_wqmm_dequant_bf16(GM_ADDR w, GM_ADDR s, GM_ADDR o, GM_ADDR out,
                              int64_t K, int64_t N, int64_t hasOff, int64_t numCores,
                              int64_t rowsPerCore, int64_t colTile, void *stream)
{
    if (numCores <= 0 || rowsPerCore <= 0) {
        return;
    }
    wqmm_dequant_kernel<bfloat16_t, true><<<static_cast<uint32_t>(numCores), nullptr, stream>>>(
        w, s, o, out, K, N, hasOff, rowsPerCore, colTile);
}

} // extern "C"
