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
 * \file dqsq_kernel.cpp
 * \brief DequantSwigluQuant: fused dequant + SwiGLU + dynamic per-token int8 quantisation.
 *
 *   dequant : int32 path   d = (x.float() * weight_scale) * activation_scale[row]   (whole 2H row)
 *             fp16/bf16    d = x.float()
 *   swiglu  : A = d[:, :H], B = d[:, H:]
 *             activate_left == 0 -> out = silu(B) * A
 *             activate_left != 0 -> out = silu(A) * B
 *   smooth  : out *= quant_scale[0:H]      (AFTER the gate)
 *   quant   : s = clamp(max_j |out| / 127, 1e-12) ; y = clamp(round(out / s), -128, 127) int8
 *
 * Ordering of the optional operands is load bearing:
 *   * activation_scale multiplies the dequantised row, so it sits *inside* the SiLU argument of the gate
 *     operand and also multiplies the linear operand.
 *   * quant_scale multiplies the swiglu result, so it multiplies whichever half is NOT the gate input.
 * Getting either of these on the wrong side of the gate silently changes the operator.
 *
 * Layout / movement policy
 * ------------------------
 * Each core owns a contiguous span of token rows, processed in tiles of `tr` rows.  A tile is fetched with
 * one DataCopyPad per half: blockCount = rows, blockLen = H * sizeof(T), srcStride = (N2 - H) * sizeof(T)
 * bytes, dstStride = 0.  `rightPadding` (<= 15 elements, always inside the 32 byte padding budget) zero
 * fills the tail so the raw panel is fully written up to RL = align32(H*sizeof(T)) / sizeof(T).
 *
 * Two compute layouts are used, selected by the row length:
 *
 *   * BAT (RL <= 1984): the tile row pitch is padded up to a multiple of 64 (`P`), which is also a
 *     multiple of 16, so every buffer row starts on a 32 byte boundary and the row pitch in data blocks
 *     fits in the 8 bit repeat stride of the high dimension vector forms.  Every per-column and per-row
 *     stage is then a handful of tile wide strided calls (one per 64 column chunk) instead of one call per
 *     token row.  The per token maximum is produced by folding the row down to one 32 byte block with
 *     strided Max calls and then collapsing the eight lanes of that block with one Brcb plus three block
 *     stride Max folds.  This removes the per-row issue overhead that dominates small-H / large-M shapes.
 *
 *   * otherwise: rows are longer than the repeat stride range, so every stage uses the stride free
 *     "tensor first n" forms, evaluated per row.
 *
 * In both layouts the padding columns [RL, P) of the fp32 panels are zero, so they can be processed by
 * the elementwise chain (0 -> 0) and never influence an output: the row reduction counts them (harmless),
 * and the int8 store only covers the first H columns.
 *
 * float -> int8 is not a supported Cast pair on this product line, so the quantisation tail goes
 * float -> half (CAST_RINT) -> int8 (CAST_RINT), which rounds the same way as torch.round.
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "dqsq_launch.h"

using namespace AscendC;

namespace dqsq {

constexpr int64_t DQSQ_MAX_TR = 63;
constexpr float DQSQ_INV127 = 1.0f / 127.0f;
constexpr float DQSQ_SMIN = 1e-12f;
constexpr float DQSQ_CMAX = 127.0f;
constexpr float DQSQ_CMIN = -128.0f;
constexpr float DQSQ_NEG_HUGE = -3.402823466e38f;

__aicore__ inline int64_t DqUpTo(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

}  // namespace dqsq

template <typename Tin, bool IS_INT32>
__global__ __aicore__ void DqsqKernel(GM_ADDR xPtr, GM_ADDR yPtr, GM_ADDR sPtr,
                                      GM_ADDR wsPtr, GM_ADDR asPtr, GM_ADDR qsPtr,
                                      int64_t M, int64_t N2, int64_t numBlocks, int64_t rowsPerCore,
                                      int64_t tr, int64_t activateLeft, int64_t hasQ)
{
    using namespace dqsq;

    const int64_t H = N2 >> 1;
    const int64_t es = (int64_t)sizeof(Tin);
    const int64_t RL = DqUpTo(H * es, 32) / es;          // raw valid row length (elements)
    const int64_t KW = DqUpTo(H * 4, 32) / 4;            // length of the 4 byte resident side vectors
    const int64_t KWP = DqUpTo(KW, 64);                  // padded resident vector length
    // High dimension vector forms carry the repeat stride in 8 bits, so a tile can only be driven by
    // tile wide strided calls while the row pitch stays within 255 data blocks.
    const bool BAT = (RL <= 1984);
    const int64_t P = BAT ? DqUpTo(RL, 64) : RL;         // physical row pitch (all tile buffers)
    const int64_t PH = DqUpTo(P, 16);                    // half buffer row pitch (half elements)
    const int64_t PI = DqUpTo(P, 32);                    // int8 buffer row pitch (bytes)
    const int64_t RP = RL - H;                           // raw right padding (elements)
    const int64_t RQ = KW - H;                           // quant_scale right padding (elements)
    const int64_t ySS = (PI - DqUpTo(H, 32)) >> 5;       // int8 store srcStride (32B units)
    const int64_t rSS = (P * es - DqUpTo(H * es, 32)) >> 5;  // raw load dstStride (32B units)
    const uint8_t rp8 = (uint8_t)(BAT ? (P >> 3) : 8);   // row pitch in data blocks
    const int64_t NCH = (P + 63) >> 6;                   // 64 wide column chunks
    const int64_t LM = RL - ((NCH - 1) << 6);            // valid elements of the last raw->fp32 chunk

    const int64_t blk = (int64_t)GetBlockIdx();
    if (blk >= numBlocks) {
        return;
    }
    const int64_t rowStart = blk * rowsPerCore;
    if (rowStart >= M) {
        return;
    }
    int64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > M) {
        rowEnd = M;
    }
    const int64_t nRows = rowEnd - rowStart;
    if (nRows <= 0) {
        return;
    }

    GlobalTensor<Tin> xGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<float> sGm;
    GlobalTensor<float> wsGm;
    GlobalTensor<float> asGm;
    GlobalTensor<float> qsGm;
    xGm.SetGlobalBuffer((__gm__ Tin *)xPtr);
    yGm.SetGlobalBuffer((__gm__ int8_t *)yPtr);
    sGm.SetGlobalBuffer((__gm__ float *)sPtr);
    if (IS_INT32) {
        wsGm.SetGlobalBuffer((__gm__ float *)wsPtr);
        asGm.SetGlobalBuffer((__gm__ float *)asPtr);
    }
    if (hasQ != 0) {
        qsGm.SetGlobalBuffer((__gm__ float *)qsPtr);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qRaw;
    TQue<QuePosition::VECIN, 1> qAs;
    TQue<QuePosition::VECIN, 1> qWA;
    TQue<QuePosition::VECIN, 1> qWB;
    TQue<QuePosition::VECIN, 1> qQS;
    TQue<QuePosition::VECOUT, 1> qY;
    TQue<QuePosition::VECOUT, 1> qS;
    TBuf<TPosition::VECCALC> bFA;
    TBuf<TPosition::VECCALC> bFB;
    TBuf<TPosition::VECCALC> bHalf;
    TBuf<TPosition::VECCALC> bRtmp;
    TBuf<TPosition::VECCALC> bAsBlk;
    TBuf<TPosition::VECCALC> bAs4;
    TBuf<TPosition::VECCALC> bBF;
    TBuf<TPosition::VECCALC> bG1;
    TBuf<TPosition::VECCALC> bG2;
    TBuf<TPosition::VECCALC> bG4;

    pipe.InitBuffer(qRaw, (uint8_t)1, (uint32_t)(tr * P * es + 512));
    pipe.InitBuffer(qY, (uint8_t)1, (uint32_t)(tr * PI + 512));
    pipe.InitBuffer(qS, (uint8_t)1, (uint32_t)(tr * 32 + 64));
    pipe.InitBuffer(bFA, (uint32_t)(tr * P * 4 + 512));
    pipe.InitBuffer(bFB, (uint32_t)(tr * P * 4 + 512));
    pipe.InitBuffer(bHalf, (uint32_t)(tr * PH * 2 + 512));
    pipe.InitBuffer(bRtmp, (uint32_t)8192);
    if (hasQ != 0) {
        pipe.InitBuffer(qQS, (uint8_t)1, (uint32_t)(KWP * 4 + 256));
    }
    if (IS_INT32) {
        pipe.InitBuffer(qAs, (uint8_t)1, (uint32_t)(DqUpTo(tr, 8) * 4 + 128));
        pipe.InitBuffer(qWA, (uint8_t)1, (uint32_t)(KWP * 4 + 256));
        pipe.InitBuffer(qWB, (uint8_t)1, (uint32_t)(KWP * 4 + 256));
        pipe.InitBuffer(bAsBlk, (uint32_t)(DqUpTo(tr, 8) * 32 + 256));
        pipe.InitBuffer(bAs4, (uint32_t)(tr * 64 * 4 + 256));
    }
    if (BAT) {
        pipe.InitBuffer(bBF, (uint32_t)(tr * 256 + 256));
        pipe.InitBuffer(bG1, (uint32_t)(tr * 128 + 256));
        pipe.InitBuffer(bG2, (uint32_t)(tr * 64 + 256));
        pipe.InitBuffer(bG4, (uint32_t)(tr * 256 + 256));
    }

    LocalTensor<float> faBase = bFA.Get<float>();
    LocalTensor<float> fbBase = bFB.Get<float>();
    LocalTensor<half> halfBase = bHalf.Get<half>();
    LocalTensor<float> rtmp = bRtmp.Get<float>();
    LocalTensor<float> asBlkBase;
    LocalTensor<float> as4Base;
    if (IS_INT32) {
        asBlkBase = bAsBlk.Get<float>();
        as4Base = bAs4.Get<float>();
    }

    if (BAT && (P != RL)) {
        // The padding columns [RL, P) of the fp32 panels are never written by the raw->fp32 cast; they are
        // zeroed once here and the elementwise chain preserves zeros (0 * w = 0, silu(0) = 0, 0 / s = 0).
        // Nothing to do when the pitch already equals the valid row length (the cast covers every column).
        Duplicate(faBase, 0.0f, (int32_t)(tr * P));
        Duplicate(fbBase, 0.0f, (int32_t)(tr * P));
    }

    // ---------------------------------------------------------------- resident side vectors
    LocalTensor<float> wqA;
    LocalTensor<float> wqB;
    LocalTensor<float> qsRes;
    if (IS_INT32) {
        DataCopyExtParams wcp;
        wcp.blockCount = 1;
        wcp.blockLen = (uint32_t)(H * 4);
        wcp.srcStride = 0;
        wcp.dstStride = 0;
        wcp.rsv = 0;
        DataCopyPadExtParams<float> wpp{true, 0, (uint8_t)RP, 0.0f};

        {
            LocalTensor<float> t = qWA.AllocTensor<float>();
            DataCopyPad(t, wsGm[0], wcp, wpp);
            qWA.EnQue(t);
            wqA = qWA.DeQue<float>();
        }
        {
            LocalTensor<float> t = qWB.AllocTensor<float>();
            DataCopyPad(t, wsGm[H], wcp, wpp);
            qWB.EnQue(t);
            wqB = qWB.DeQue<float>();
        }
        if (KW < KWP) {
            Duplicate(wqA[KW], 0.0f, (int32_t)(KWP - KW));
            Duplicate(wqB[KW], 0.0f, (int32_t)(KWP - KW));
        }
        if (hasQ != 0) {
            LocalTensor<float> t = qQS.AllocTensor<float>();
            DataCopyPadExtParams<float> qpp{true, 0, (uint8_t)RQ, 0.0f};
            DataCopyPad(t, qsGm[0], wcp, qpp);
            qQS.EnQue(t);
            qsRes = qQS.DeQue<float>();
        }
    } else if (hasQ != 0) {
        DataCopyExtParams wcp;
        wcp.blockCount = 1;
        wcp.blockLen = (uint32_t)(H * 4);
        wcp.srcStride = 0;
        wcp.dstStride = 0;
        wcp.rsv = 0;
        DataCopyPadExtParams<float> qpp{true, 0, (uint8_t)RQ, 0.0f};
        LocalTensor<float> t = qQS.AllocTensor<float>();
        DataCopyPad(t, qsGm[0], wcp, qpp);
        qQS.EnQue(t);
        qsRes = qQS.DeQue<float>();
    }
    if (hasQ != 0 && KW < KWP) {
        Duplicate(qsRes[KW], 0.0f, (int32_t)(KWP - KW));
    }

    float mbv[DQSQ_MAX_TR];
    float ivv[DQSQ_MAX_TR];

    for (int64_t t0 = 0; t0 < nRows; t0 += tr) {
        int64_t trows = nRows - t0;
        if (trows > tr) {
            trows = tr;
        }
        const int64_t grow = rowStart + t0;
        const int64_t N = trows * P;
        const uint8_t rep = (uint8_t)trows;
        // Tile wide strided calls cover `NCH` chunks of 64 columns for all rows at once, i.e. NCH calls per
        // stage; the per row forms need trows calls.  Prefer whichever issues fewer instructions, because
        // the per instruction issue overhead dominates the small-tile wall clock on this platform.
        const bool CHK = BAT && (NCH < trows);
        // The block fold used to obtain the row maximum halves the block count per level, so it is only well
        // defined when a row owns a power of two number of data blocks, and the level 2 repeat count (a byte)
        // has to stay representable: repeat = trows * blocks_after / 16 <= 255.
        const int64_t NBF = P >> 3;
        const bool FOLDF = CHK && ((NBF & (NBF - 1)) == 0) && (trows * NBF <= 4080);

        // ------------------------------------------------------------ load A half
        {
            LocalTensor<Tin> raw = qRaw.AllocTensor<Tin>();
            DataCopyExtParams cp;
            cp.blockCount = (uint16_t)trows;
            cp.blockLen = (uint32_t)(H * es);
            cp.srcStride = (uint32_t)((N2 - H) * es);
            cp.dstStride = (uint32_t)rSS;
            cp.rsv = 0;
            DataCopyPadExtParams<Tin> pp{true, 0, (uint8_t)RP, (Tin)0};
            DataCopyPad(raw, xGm[grow * N2], cp, pp);
            qRaw.EnQue(raw);
        }
        {
            LocalTensor<Tin> raw = qRaw.DeQue<Tin>();
            if (CHK) {
                for (int64_t c = 0; c < NCH; ++c) {
                    const int64_t off = c << 6;
                    const uint64_t m = (uint64_t)((c == NCH - 1) ? LM : 64);
                    Cast(faBase[off], raw[off], RoundMode::CAST_NONE, m, rep,
                         {(uint16_t)1, (uint16_t)1, (uint8_t)(P >> 3), (uint8_t)((P * es) >> 5)});
                }
            } else {
                Cast(faBase, raw, RoundMode::CAST_NONE, (uint32_t)N);
            }
            qRaw.FreeTensor(raw);
        }
        // ------------------------------------------------------------ load B half
        {
            LocalTensor<Tin> raw = qRaw.AllocTensor<Tin>();
            DataCopyExtParams cp;
            cp.blockCount = (uint16_t)trows;
            cp.blockLen = (uint32_t)(H * es);
            cp.srcStride = (uint32_t)((N2 - H) * es);
            cp.dstStride = (uint32_t)rSS;
            cp.rsv = 0;
            DataCopyPadExtParams<Tin> pp{true, 0, (uint8_t)RP, (Tin)0};
            DataCopyPad(raw, xGm[grow * N2 + H], cp, pp);
            qRaw.EnQue(raw);
        }
        {
            LocalTensor<Tin> raw = qRaw.DeQue<Tin>();
            if (CHK) {
                for (int64_t c = 0; c < NCH; ++c) {
                    const int64_t off = c << 6;
                    const uint64_t m = (uint64_t)((c == NCH - 1) ? LM : 64);
                    Cast(fbBase[off], raw[off], RoundMode::CAST_NONE, m, rep,
                         {(uint16_t)1, (uint16_t)1, (uint8_t)(P >> 3), (uint8_t)((P * es) >> 5)});
                }
            } else {
                Cast(fbBase, raw, RoundMode::CAST_NONE, (uint32_t)N);
            }
            qRaw.FreeTensor(raw);
        }

        // ------------------------------------------------------------ dequant weights
        if (IS_INT32) {
            if (CHK) {
                for (int64_t c = 0; c < NCH; ++c) {
                    const int64_t off = c << 6;
                    Mul(faBase[off], faBase[off], wqA[off], (uint64_t)64, rep, {1, 1, 1, rp8, rp8, 0});
                    Mul(fbBase[off], fbBase[off], wqB[off], (uint64_t)64, rep, {1, 1, 1, rp8, rp8, 0});
                }
            } else {
                for (int64_t i = 0; i < trows; ++i) {
                    Mul(faBase[i * P], faBase[i * P], wqA, (uint32_t)KW);
                    Mul(fbBase[i * P], fbBase[i * P], wqB, (uint32_t)KW);
                }
            }
        }
        // ------------------------------------------------------------ smooth quant coefficient
        // quant_scale multiplies the swiglu result, i.e. whichever half is NOT the gate input, so it can be
        // folded into that half before the gate is formed.
        if (hasQ != 0) {
            LocalTensor<float> tgt = (activateLeft != 0) ? fbBase : faBase;
            if (CHK) {
                for (int64_t c = 0; c < NCH; ++c) {
                    const int64_t off = c << 6;
                    Mul(tgt[off], tgt[off], qsRes[off], (uint64_t)64, rep, {1, 1, 1, rp8, rp8, 0});
                }
            } else {
                for (int64_t i = 0; i < trows; ++i) {
                    Mul(tgt[i * P], tgt[i * P], qsRes, (uint32_t)KW);
                }
            }
        }

        // ------------------------------------------------------------ activation_scale tile load
        if (IS_INT32) {
            LocalTensor<float> asL = qAs.AllocTensor<float>();
            DataCopyExtParams acp;
            acp.blockCount = 1;
            acp.blockLen = (uint32_t)(trows * 4);
            acp.srcStride = 0;
            acp.dstStride = 0;
            acp.rsv = 0;
            DataCopyPadExtParams<float> app{false, 0, 0, 0.0f};
            DataCopyPad(asL, asGm[grow], acp, app);
            qAs.EnQue(asL);
            asL = qAs.DeQue<float>();
            const uint8_t nc8 = (uint8_t)((trows + 7) / 8);
            Brcb(asBlkBase, asL, nc8, {1, 8});
            Brcb(as4Base, asBlkBase, rep, {1, 8});
            qAs.FreeTensor(asL);
        }

        // -------------------------------------------------- activation_scale on the dequant output
        // The reference dequantises the WHOLE row, so activation_scale multiplies both the A and the B half
        // before the SiLU gate is formed.  Applying it afterwards would put it outside silu().
        if (IS_INT32) {
            if (CHK) {
                for (int64_t c = 0; c < NCH; ++c) {
                    const int64_t off = c << 6;
                    Mul(faBase[off], faBase[off], as4Base, (uint64_t)64, rep, {1, 1, 1, rp8, rp8, 8});
                    Mul(fbBase[off], fbBase[off], as4Base, (uint64_t)64, rep, {1, 1, 1, rp8, rp8, 8});
                }
            } else {
                const int64_t p64 = P >> 6;
                const int64_t r64 = P - (p64 << 6);
                for (int64_t i = 0; i < trows; ++i) {
                    if (p64 > 0) {
                        Mul(faBase[i * P], faBase[i * P], as4Base[i * 64], (uint64_t)64, (uint8_t)p64,
                            {1, 1, 1, 8, 8, 0});
                        Mul(fbBase[i * P], fbBase[i * P], as4Base[i * 64], (uint64_t)64, (uint8_t)p64,
                            {1, 1, 1, 8, 8, 0});
                    }
                    if (r64 > 0) {
                        Mul(faBase[i * P + (p64 << 6)], faBase[i * P + (p64 << 6)], as4Base[i * 64],
                            (uint64_t)r64, (uint8_t)1, {1, 1, 1, 1, 1, 0});
                        Mul(fbBase[i * P + (p64 << 6)], fbBase[i * P + (p64 << 6)], as4Base[i * 64],
                            (uint64_t)r64, (uint8_t)1, {1, 1, 1, 1, 1, 0});
                    }
                }
            }
        }

        // ------------------------------------------------------------ swiglu (flat)
        if (activateLeft == 0) {
            Mul(faBase, faBase, fbBase, (uint32_t)N);
            Muls(fbBase, fbBase, -1.0f, (uint32_t)N);
            Exp(fbBase, fbBase, (uint32_t)N);
            Adds(fbBase, fbBase, 1.0f, (uint32_t)N);
            Div(faBase, faBase, fbBase, (uint32_t)N);
        } else {
            Mul(fbBase, faBase, fbBase, (uint32_t)N);
            Muls(faBase, faBase, -1.0f, (uint32_t)N);
            Exp(faBase, faBase, (uint32_t)N);
            Adds(faBase, faBase, 1.0f, (uint32_t)N);
            Div(fbBase, fbBase, faBase, (uint32_t)N);
        }
        LocalTensor<float> outB = (activateLeft != 0) ? fbBase : faBase;
        LocalTensor<float> absB = (activateLeft != 0) ? faBase : fbBase;

        // ------------------------------------------------------------ per row |out| max
        LocalTensor<float> sOut = qS.AllocTensor<float>();
        Abs(absB, outB, (uint32_t)N);
        if (CHK) {
            // Row maximum without any per row reduction call.  absB is [trows, P] with a row pitch of NBF = P/8
            // data blocks, P is a multiple of 64 so NBF is a multiple of 8, and FOLDF has already checked that
            // NBF is a power of two, so every row owns a whole number of block pairs and no pair ever straddles
            // a row.  `Max(dst, src, src+s, mask, repeat)` folds the block count in half per level:
            //   * mask 64 (NBF a multiple of 16): dst block b pairs src blocks 16r+j and 16r+8+j, i.e. a factor
            //     16 of source blocks per repeat;
            //   * mask 8: dst block b pairs src blocks 2b and 2b+1.
            // Either way the destination is written strictly below the blocks being read, so the fold is safe
            // in place.  After log2(NBF) levels each row is one block whose lane j holds the maximum over
            // columns {j, j+8, j+16, ...}; a Brcb spreads those eight partial maxima into whole blocks, three
            // block stride Max folds collapse them to one uniform block per row (all eight lanes hold the exact
            // row maximum), and one more Brcb turns that into the per row divisor of the strided divide.
            LocalTensor<float> bfT = bBF.Get<float>();
            LocalTensor<float> g1T = bG1.Get<float>();
            LocalTensor<float> g2T = bG2.Get<float>();
            LocalTensor<float> g4T = bG4.Get<float>();

            if (FOLDF) {
                int64_t nnb = NBF;
                while (nnb > 1) {
                    if ((nnb & 15) == 0) {
                        Max(absB, absB, absB[64], (uint64_t)64, (uint8_t)(trows * (nnb >> 4)),
                            {1, 1, 1, 8, 16, 16});
                    } else {
                        Max(absB, absB, absB[8], (uint64_t)8, (uint8_t)(trows * (nnb >> 1)),
                            {1, 1, 1, 1, 2, 2});
                    }
                    nnb >>= 1;
                }
                Brcb(bfT, absB, rep, {1, 8});
            } else {
                Duplicate(sOut, DQSQ_NEG_HUGE, (int32_t)(trows * 8));
                for (int64_t i = 0; i < trows; ++i) {
                    ReduceMax<float>(sOut[i * 8], absB[i * P], rtmp, (int32_t)P, false);
                }
                Brcb(bfT, sOut, rep, {1, 8});
            }
            Max(g1T, bfT, bfT[8], (uint64_t)8, (uint8_t)(trows * 4), {1, 1, 1, 1, 2, 2});
            Max(g2T, g1T, g1T[8], (uint64_t)8, (uint8_t)(trows * 2), {1, 1, 1, 1, 2, 2});
            Max(sOut, g2T, g2T[8], (uint64_t)8, rep, {1, 1, 1, 1, 2, 2});
            Muls(sOut, sOut, DQSQ_INV127, (uint64_t)8, rep, {1, 1, 1, 1});
            Maxs(sOut, sOut, DQSQ_SMIN, (uint64_t)8, rep, {1, 1, 1, 1});
            Brcb(g4T, sOut, rep, {1, 8});

            for (int64_t c = 0; c < NCH; ++c) {
                const int64_t off = c << 6;
                Div(outB[off], outB[off], g4T, (uint64_t)64, rep, {1, 1, 1, rp8, rp8, 8});
            }
        } else {
            for (int64_t i = 0; i < trows; ++i) {
                ReduceMax<float>(sOut[i * 8], absB[i * P], rtmp, (int32_t)P, false);
            }
            for (int64_t i = 0; i < trows; ++i) {
                mbv[i] = sOut.GetValue((int32_t)(i * 8));
            }
            for (int64_t i = 0; i < trows; ++i) {
                float mb = mbv[i];
                if (!(mb > 0.0f)) {
                    mb = 0.0f;
                }
                float sv = mb * DQSQ_INV127;
                if (sv < DQSQ_SMIN) {
                    sv = DQSQ_SMIN;
                }
                ivv[i] = 1.0f / sv;
                sOut.SetValue((int32_t)(i * 8), sv);
            }
            for (int64_t i = 0; i < trows; ++i) {
                Muls(outB[i * P], outB[i * P], ivv[i], (uint32_t)P);
            }
        }

        // ------------------------------------------------------------ clamp + cast + store
        Mins(outB, outB, DQSQ_CMAX, (uint32_t)N);
        Maxs(outB, outB, DQSQ_CMIN, (uint32_t)N);

        LocalTensor<int8_t> yOut = qY.AllocTensor<int8_t>();
        if (CHK) {
            for (int64_t c = 0; c < NCH; ++c) {
                const int64_t off = c << 6;
                Cast(halfBase[off], outB[off], RoundMode::CAST_RINT, (uint64_t)64, rep,
                     {(uint16_t)1, (uint16_t)1, (uint8_t)(PH >> 4), (uint8_t)(P >> 3)});
                Cast(yOut[off], halfBase[off], RoundMode::CAST_RINT, (uint64_t)64, rep,
                     {(uint16_t)1, (uint16_t)1, (uint8_t)(PI >> 5), (uint8_t)(PH >> 4)});
            }
        } else {
            for (int64_t i = 0; i < trows; ++i) {
                Cast(halfBase[i * PH], outB[i * P], RoundMode::CAST_RINT, (uint32_t)P);
                Cast(yOut[i * PI], halfBase[i * PH], RoundMode::CAST_RINT, (uint32_t)P);
            }
        }
        qY.EnQue(yOut);
        {
            LocalTensor<int8_t> yD = qY.DeQue<int8_t>();
            DataCopyExtParams ycp;
            ycp.blockCount = (uint16_t)trows;
            ycp.blockLen = (uint32_t)H;
            ycp.srcStride = (uint32_t)ySS;
            ycp.dstStride = 0;
            ycp.rsv = 0;
            DataCopyPad(yGm[grow * H], yD, ycp);
            qY.FreeTensor(yD);
        }

        qS.EnQue(sOut);
        {
            LocalTensor<float> sD = qS.DeQue<float>();
            DataCopyExtParams scp;
            scp.blockCount = (uint16_t)trows;
            scp.blockLen = (uint32_t)4;
            scp.srcStride = 0;
            scp.dstStride = 0;
            scp.rsv = 0;
            DataCopyPad(sGm[grow], sD, scp);
            qS.FreeTensor(sD);
        }
    }
}

/* --------------------------------------------------------------------------------------------- */
/* host tiling                                                                                    */
/* --------------------------------------------------------------------------------------------- */

std::tuple<int64_t, int64_t, int64_t> calc_dqsq_tiling(int64_t M, int64_t N2, int64_t elemSize,
                                                       int64_t isInt32, int64_t hasQScale)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    if (ascendPlatform != nullptr) {
        ascendPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    if (ubSize < 1024) {
        ubSize = 196608u;
    }
    int64_t coreNum = 1;
    if (ascendPlatform != nullptr) {
        coreNum = ascendPlatform->GetCoreNumAiv();
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }

    const int64_t H = (N2 > 1) ? (N2 >> 1) : 1;
    const int64_t RL = ((H * elemSize + 31) / 32) * 32 / elemSize;
    const int64_t KW = ((H * 4 + 31) / 32) * 32 / 4;
    const int64_t KWP = ((KW + 63) / 64) * 64;
    const bool BAT = (RL <= 1984);
    const int64_t P = BAT ? (((RL + 63) / 64) * 64) : RL;
    const int64_t PH = ((P + 15) / 16) * 16;
    const int64_t PI = ((P + 31) / 32) * 32;

    // Per tile row cost, mirroring the InitBuffer sizes in the kernel exactly.  Charging the fixed slack per
    // row (as a generic fudge factor) shrinks `tr` badly for short rows, which is where the per tile issue
    // overhead matters most, so the row and shape independent parts are kept separate here.
    int64_t perRow = P * elemSize;  // raw panel
    perRow += 8 * P;                // fa / fb fp32
    perRow += 2 * PH;               // half staging panel
    perRow += PI;                   // int8 output panel
    perRow += 32;                   // per token scale panel
    if (BAT) {
        perRow += 704;              // fold scratch (bBF 256 + bG1 128 + bG2 64 + bG4 256)
    }
    if (isInt32 != 0) {
        perRow += 256;              // activation_scale per row broadcast staging (bAs4)
    }
    if (perRow < 1) {
        perRow = 1;
    }

    int64_t fixed = 12288;                   // head room
    fixed += 8192;                           // reduce workspace
    fixed += 5 * 512 + 64;                   // per queue / per buffer slack
    fixed += 4 * 256;                        // fold scratch slack
    if (isInt32 != 0) {                      // resident weight_scale halves + activation_scale staging
        fixed += 2 * (KWP * 4 + 256) + 64 * 4 + 128 + 64 * 32 + 256;
    }
    if (hasQScale != 0) {                    // resident quant_scale
        fixed += KWP * 4 + 256;
    }
    int64_t budget = (int64_t)ubSize - fixed;
    if (budget < perRow) {
        budget = perRow;
    }
    int64_t tr = budget / perRow;
    if (tr < 1) {
        tr = 1;
    }
    if (tr > 63) {
        tr = 63;
    }

    int64_t numBlocks = (M < coreNum) ? M : coreNum;
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t rowsPerCore = (M + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, rowsPerCore, tr);
}

/* --------------------------------------------------------------------------------------------- */
/* launch wrappers                                                                                */
/* --------------------------------------------------------------------------------------------- */

extern "C" {

void launch_dqsq_half(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR wscale, GM_ADDR ascale,
                      GM_ADDR qscale, int64_t M, int64_t N2, int64_t numBlocks, int64_t rowsPerCore,
                      int64_t tr, int64_t activateLeft, int64_t hasQScale, void *stream)
{
    DqsqKernel<half, false><<<numBlocks, nullptr, stream>>>(
        x, y, scale, wscale, ascale, qscale, M, N2, numBlocks, rowsPerCore, tr, activateLeft, hasQScale);
}

void launch_dqsq_bfloat16(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR wscale, GM_ADDR ascale,
                          GM_ADDR qscale, int64_t M, int64_t N2, int64_t numBlocks, int64_t rowsPerCore,
                          int64_t tr, int64_t activateLeft, int64_t hasQScale, void *stream)
{
    DqsqKernel<bfloat16_t, false><<<numBlocks, nullptr, stream>>>(
        x, y, scale, wscale, ascale, qscale, M, N2, numBlocks, rowsPerCore, tr, activateLeft, hasQScale);
}

void launch_dqsq_int32(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR wscale, GM_ADDR ascale,
                       GM_ADDR qscale, int64_t M, int64_t N2, int64_t numBlocks, int64_t rowsPerCore,
                       int64_t tr, int64_t activateLeft, int64_t hasQScale, void *stream)
{
    DqsqKernel<int32_t, true><<<numBlocks, nullptr, stream>>>(
        x, y, scale, wscale, ascale, qscale, M, N2, numBlocks, rowsPerCore, tr, activateLeft, hasQScale);
}

}  // extern "C"
