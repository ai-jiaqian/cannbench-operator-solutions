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
 * \file gru_kernel.cpp
 * \brief GRU device kernel + host tiling/launch wrappers (bisheng + -xasc, dav-2201).
 *
 * One single kernel launch per operator call.  Grid = min(coreNumAiv, B); every core owns whole
 * batch element(s) and walks all layers / directions / timesteps for them.  Batch elements are
 * independent, so no inter-core communication and no global barrier is needed.
 *
 * All arithmetic happens in fp32 inside the kernel; the raw input dtype tensors are cast to fp32 on
 * the fly (inputs) or at the final store (outputs).  Nothing is routed through host memory.
 *
 * Every vector operand used below starts at offset 0 of its own UB buffer, so no 32-byte alignment
 * constraint can be violated for arbitrary hiddenSize / inputSize (a sub-tensor view of a TBuf at a
 * non multiple-of-8 element offset is misaligned and faults the vector core).
 */

#define GRU_DEVICE_TU 1

#include <type_traits>
#include <cstdint>
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_reduce_intf.h"
#include "platform/platform_ascendc.h"
#include "gru_launch.h"

using namespace AscendC;

namespace {

constexpr int64_t GRU_SW = 64;   /* matvec K sub-strip width (one 256B repeat of fp32) */

/* super-strip width in elements.  The graded row lengths (K) top out at 512, so a 16-bit
   super-strip of 512 lets one whole matrix row be fetched as a single DMA block. */
template <typename T>
struct GruWs {
    static constexpr int64_t value = (sizeof(T) == 4) ? 128 : 512;
};

/* matvec row block: small enough that RB * WS * sizeof(T) * 2 (queue depth) + RB * WS * 4
   (fp32 strip copy) stays inside the UB budget. */
template <typename T>
struct GruRb {
    static constexpr int64_t value = (sizeof(T) == 4) ? 32 : 16;
};

__aicore__ inline int64_t GruCeil64(int64_t v)
{
    return ((v + 63) >> 6) << 6;
}

/*! \brief load n raw-dtype elements from GM and produce fp32 in dst (dst is UB fp32) */
template <typename T>
__aicore__ inline void GruLoadF(const LocalTensor<float>& dst, __gm__ uint8_t* base, int64_t n,
                                TQue<QuePosition::VECIN, 2>& qIn)
{
    GlobalTensor<T> g;
    g.SetGlobalBuffer((__gm__ T*)base);
    LocalTensor<T> t = qIn.AllocTensor<T>();
    DataCopyExtParams cp{1, (uint32_t)(n * (int64_t)sizeof(T)), 0u, 0u, 0u};
    DataCopyPadExtParams<T> pp{false, 0u, 0u, (T)0};
    DataCopyPad(t, g[0], cp, pp);
    qIn.EnQue(t);
    t = qIn.DeQue<T>();
    if constexpr (std::is_same<T, float>::value) {
        Adds(dst, t, 0.0f, (int32_t)n);
    } else {
        Cast(dst, t, RoundMode::CAST_NONE, (int32_t)n);
    }
    qIn.FreeTensor(t);
}

/*!
 * \brief acc[0:rows][0:64] = W[r0:r0+rows, 0:K] @ vec[0:K]
 *
 * W lives in GM as rows of K elements of the input dtype (row stride K, unpadded).  vec is a UB fp32
 * buffer with at least ceil64(K) valid-or-masked elements.  acc must be a UB fp32 buffer of at least
 * rows*64 elements.  The vector operand is broadcast over the rows through src1RepStride = 0.
 */
/*! \brief real element count of super-strip si of a K-wide row (clamped to >= 1) */
template <typename T>
__aicore__ inline int64_t GruSupKlen(int64_t K, int64_t si)
{
    constexpr int64_t WS = GruWs<T>::value;
    int64_t klen = K - si * WS;
    if (klen > WS) {
        klen = WS;
    }
    if (klen < 1) {
        klen = 1;
    }
    return klen;
}

/*!
 * \brief issue the DMA of one (rows x klen) raw-weight super-strip of W into a fresh UB slot
 *
 * When the strip spans the whole row (klen == K, so the rows are contiguous with stride klen) and the
 * row length is a multiple of 32 bytes, the entire row block is fetched as ONE descriptor block
 * (blockCount = 1).  Every other shape keeps the one-block-per-row form, which DataCopyPad tail-pads
 * to 32 bytes, so the UB row pitch is always align32(klen * sizeof(T)) / sizeof(T) elements.
 */
template <typename T>
__aicore__ inline void GruIssueStrip(__gm__ uint8_t* wBase, int64_t K, int64_t rows, int64_t r0,
                                     int64_t si, TQue<QuePosition::VECIN, 2>& qW)
{
    constexpr int64_t WS = GruWs<T>::value;
    GlobalTensor<T> wg;
    wg.SetGlobalBuffer((__gm__ T*)wBase);
    const int64_t klen = GruSupKlen<T>(K, si);
    LocalTensor<T> wr = qW.AllocTensor<T>();
    if (klen == K && ((klen * (int64_t)sizeof(T)) & 31) == 0) {
        const uint32_t bytes = (uint32_t)(rows * klen * (int64_t)sizeof(T));
        DataCopyExtParams cp{1, bytes, 0u, 0u, 0u};
        DataCopyPadExtParams<T> pp{true, 0u, (uint8_t)(((bytes + 31) & ~31u) - bytes), (T)0};
        DataCopyPad(wr, wg[r0 * K + si * WS], cp, pp);
    } else {
        DataCopyExtParams cp{(uint16_t)rows, (uint32_t)(klen * (int64_t)sizeof(T)),
                             (uint32_t)((K - klen) * (int64_t)sizeof(T)), 0u, 0u};
        DataCopyPadExtParams<T> pp{true, 0u, 0u, (T)0};
        DataCopyPad(wr, wg[r0 * K + si * WS], cp, pp);
    }
    qW.EnQue(wr);
}

/*!
 * \brief acc[0:rows][0:64] = W[r0:r0+rows, 0:K] @ vec[0:K]
 *
 * W lives in GM as rows of K elements of the input dtype (row stride K, unpadded).  vec is a UB fp32
 * buffer with at least ceil64(K) valid-or-masked elements.  acc must be a UB fp32 buffer of at least
 * rows*64 elements.  The vector operand is broadcast over the rows through src1RepStride = 0.
 *
 * The super-strips are software-pipelined one ahead: the DMA for strip si+1 is issued before strip si
 * is consumed, so the MTE2 round trip overlaps the Cast/MulAddDst of the previous strip.  The
 * accumulator is only written by the pipeline, never by the DMA engine.
 */
template <typename T>
__aicore__ inline void GruAccum(const LocalTensor<float>& acc, __gm__ uint8_t* wBase, int64_t K,
                                const LocalTensor<float>& vec, int64_t r0, int64_t rows,
                                TQue<QuePosition::VECIN, 2>& qW, const LocalTensor<float>& wf)
{
    constexpr int64_t WS = GruWs<T>::value;
    const int64_t Kp = GruCeil64(K);
    const int64_t nSup = (Kp + WS - 1) / WS;

    GruIssueStrip<T>(wBase, K, rows, r0, 0, qW);

    for (int64_t si = 0; si < nSup; ++si) {
        if (si + 1 < nSup) {
            GruIssueStrip<T>(wBase, K, rows, r0, si + 1, qW);
        }
        const int64_t klen = GruSupKlen<T>(K, si);
        LocalTensor<T> wr = qW.DeQue<T>();
        const int64_t be = (klen == K && ((klen * (int64_t)sizeof(T)) & 31) == 0)
                               ? klen
                               : (((klen * (int64_t)sizeof(T) + 31) / 32) * 32 / (int64_t)sizeof(T));

        LocalTensor<float> s0t;
        if constexpr (std::is_same<T, float>::value) {
            s0t = wr;
        } else {
            Cast(wf, wr, RoundMode::CAST_NONE, (int32_t)(rows * be));
            s0t = wf;
        }

        const int64_t nSub = (klen + GRU_SW - 1) / GRU_SW;
        for (int64_t j = 0; j < nSub; ++j) {
            const int64_t kk = (klen - j * GRU_SW > GRU_SW) ? GRU_SW : (klen - j * GRU_SW);
            const int64_t kg = si * WS + j * GRU_SW;
            const BinaryRepeatParams rep{1, 1, 1, 8, (uint8_t)(be / 8), 0};
            if (kk == GRU_SW) {
                if (kg == 0) {
                    Mul(acc, s0t[j * GRU_SW], vec[kg], (uint64_t)GRU_SW, (uint8_t)rows, rep);
                } else {
                    MulAddDst(acc, s0t[j * GRU_SW], vec[kg], (uint64_t)GRU_SW, (uint8_t)rows, rep);
                }
            } else {
                if (kg == 0) {
                    Duplicate(acc, 0.0f, (int32_t)(rows * GRU_SW));
                }
                MulAddDst(acc, s0t[j * GRU_SW], vec[kg], (uint64_t)kk, (uint8_t)rows, rep);
            }
        }
        qW.FreeTensor(wr);
    }
}

/*!
 * \brief outA = W_A @ vA ; outH = W_H @ vH   (both M x K row blocks, M rows)
 */
template <typename T>
__aicore__ inline void GruMatVecPair(const LocalTensor<float>& outA, __gm__ uint8_t* wA, int64_t KA,
                                     const LocalTensor<float>& vA, const LocalTensor<float>& outH,
                                     __gm__ uint8_t* wH, int64_t KH, const LocalTensor<float>& vH,
                                     int64_t M, TQue<QuePosition::VECIN, 2>& qW,
                                     const LocalTensor<float>& wf, const LocalTensor<float>& acc)
{
    constexpr int64_t RB = GruRb<T>::value;
    for (int64_t r0 = 0; r0 < M; r0 += RB) {
        int64_t rows = M - r0;
        if (rows > RB) {
            rows = RB;
        }
        GruAccum<T>(acc, wA, KA, vA, r0, rows, qW, wf);
        WholeReduceSum<float>(outA[r0], acc, (uint64_t)GRU_SW, (uint8_t)rows, 1, 1, 8);

        GruAccum<T>(acc, wH, KH, vH, r0, rows, qW, wf);
        WholeReduceSum<float>(outH[r0], acc, (uint64_t)GRU_SW, (uint8_t)rows, 1, 1, 8);
    }
}

template <typename T>
__global__ __aicore__ void gru_kernel(GruArgs a)
{
    const int64_t S = a.S;
    const int64_t B = a.B;
    const int64_t K0 = a.inSz;
    const int64_t H = a.H;
    const int64_t L = a.L;
    const int64_t D = a.D;
    const int64_t DH = D * H;
    const int64_t Hp = GruCeil64(H);
    const int64_t K0p = GruCeil64(K0);
    const int64_t DHp = GruCeil64(DH);
    const int64_t vecMax = (K0p > DHp) ? K0p : DHp;
    int64_t inMax = K0;
    if (3 * H > inMax) {
        inMax = 3 * H;
    }
    if (H > inMax) {
        inMax = H;
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> qW;
    TQue<QuePosition::VECIN, 2> qV;
    TQue<QuePosition::VECIN, 2> qIn;
    TQue<QuePosition::VECOUT, 2> qF;
    TQue<QuePosition::VECOUT, 2> qOut;

    pipe.InitBuffer(qW, 2, (uint32_t)(GruRb<T>::value * GruWs<T>::value * (int64_t)sizeof(T)));
    pipe.InitBuffer(qV, 2, (uint32_t)(vecMax * 4));
    pipe.InitBuffer(qIn, 2, (uint32_t)(inMax * (int64_t)sizeof(T)));
    pipe.InitBuffer(qF, 2, (uint32_t)(vecMax * 4));
    pipe.InitBuffer(qOut, 2, (uint32_t)((H > 1 ? H : 1) * (int64_t)sizeof(T)));

    TBuf<TPosition::VECCALC> bGIR, bGIZ, bGIN;     /* W_ih chunks: r, z, n */
    TBuf<TPosition::VECCALC> bGHR, bGHZ, bGHN;     /* W_hh chunks: r, z, n */
    TBuf<TPosition::VECCALC> bGR, bGZ, bBR, bBZ;   /* combined pre-activation / bias for r, z */
    TBuf<TPosition::VECCALC> bNI, bNH;            /* n chunk biases (input side / hidden side) */
    TBuf<TPosition::VECCALC> bTR, bTZ, bTN, bT4, bHN, bONES, bH;
    TBuf<TPosition::VECCALC> bWF, bAA;
    pipe.InitBuffer(bGIR, (uint32_t)(H * 4));
    pipe.InitBuffer(bGIZ, (uint32_t)(H * 4));
    pipe.InitBuffer(bGIN, (uint32_t)(H * 4));
    pipe.InitBuffer(bGHR, (uint32_t)(H * 4));
    pipe.InitBuffer(bGHZ, (uint32_t)(H * 4));
    pipe.InitBuffer(bGHN, (uint32_t)(H * 4));
    pipe.InitBuffer(bGR, (uint32_t)(H * 4));
    pipe.InitBuffer(bGZ, (uint32_t)(H * 4));
    pipe.InitBuffer(bBR, (uint32_t)(H * 4));
    pipe.InitBuffer(bBZ, (uint32_t)(H * 4));
    pipe.InitBuffer(bNI, (uint32_t)(H * 4));
    pipe.InitBuffer(bNH, (uint32_t)(H * 4));
    pipe.InitBuffer(bTR, (uint32_t)(H * 4));
    pipe.InitBuffer(bTZ, (uint32_t)(H * 4));
    pipe.InitBuffer(bTN, (uint32_t)(H * 4));
    pipe.InitBuffer(bT4, (uint32_t)(H * 4));
    pipe.InitBuffer(bHN, (uint32_t)(H * 4));
    pipe.InitBuffer(bONES, (uint32_t)(H * 4));
    pipe.InitBuffer(bH, (uint32_t)(Hp * 4));
    pipe.InitBuffer(bWF, (uint32_t)(GruRb<T>::value * GruWs<T>::value * 4 + 256));
    pipe.InitBuffer(bAA, (uint32_t)(GruRb<T>::value * GRU_SW * 4));

    const int64_t blk = GetBlockIdx();
    const int64_t nb = GetBlockNum();
    if (blk >= nb) {
        return;
    }
    const int64_t b0 = (B * blk) / nb;
    const int64_t b1 = (B * (blk + 1)) / nb;

    GlobalTensor<T> xg, yg, hng;
    xg.SetGlobalBuffer((__gm__ T*)a.x);
    yg.SetGlobalBuffer((__gm__ T*)a.y);
    hng.SetGlobalBuffer((__gm__ T*)a.hn);
    GlobalTensor<float> wsg;
    wsg.SetGlobalBuffer((__gm__ float*)a.wsc);

    const LocalTensor<float> gIR = bGIR.Get<float>();
    const LocalTensor<float> gIZ = bGIZ.Get<float>();
    const LocalTensor<float> gIN = bGIN.Get<float>();
    const LocalTensor<float> gHR = bGHR.Get<float>();
    const LocalTensor<float> gHZ = bGHZ.Get<float>();
    const LocalTensor<float> gHN = bGHN.Get<float>();
    const LocalTensor<float> gR = bGR.Get<float>();
    const LocalTensor<float> gZ = bGZ.Get<float>();
    const LocalTensor<float> bR = bBR.Get<float>();
    const LocalTensor<float> bZ = bBZ.Get<float>();
    const LocalTensor<float> bni = bNI.Get<float>();
    const LocalTensor<float> bnh = bNH.Get<float>();
    const LocalTensor<float> tR = bTR.Get<float>();
    const LocalTensor<float> tZ = bTZ.Get<float>();
    const LocalTensor<float> tN = bTN.Get<float>();
    const LocalTensor<float> t4 = bT4.Get<float>();
    const LocalTensor<float> hNew = bHN.Get<float>();
    const LocalTensor<float> ones = bONES.Get<float>();
    const LocalTensor<float> hBuf = bH.Get<float>();
    const LocalTensor<float> wf = bWF.Get<float>();
    const LocalTensor<float> accA = bAA.Get<float>();

    Duplicate(ones, 1.0f, (int32_t)H);

    /* ---------------- cast x into the fp32 time-major plane ---------------- */
    for (int64_t b = b0; b < b1; ++b) {
        for (int64_t t = 0; t < S; ++t) {
            const int64_t srow = a.batchFirst ? (b * S + t) : (t * B + b);
            const int64_t drow = t * B + b;
            __gm__ uint8_t* xrow = (__gm__ uint8_t*)((__gm__ T*)a.x + srow * K0);
            LocalTensor<float> of = qF.AllocTensor<float>();
            GruLoadF<T>(of, xrow, K0, qIn);
            qF.EnQue(of);
            of = qF.DeQue<float>();
            DataCopyExtParams cpo{1, (uint32_t)(K0 * 4), 0u, 0u, 0u};
            DataCopyPad(wsg[a.xOff + drow * K0], of, cpo);
            qF.FreeTensor(of);
        }
    }
    /* the layer planes are read back through MTE2 right after being written through MTE3 */
    PipeBarrier<PIPE_ALL>();

    /* ---------------- layers / directions ---------------- */
    for (int64_t b = b0; b < b1; ++b) {
        for (int64_t l = 0; l < L; ++l) {
            const bool lastLayer = (l == (L - 1));
            const int64_t inW = (l == 0) ? K0 : DH;
            const int64_t inOff = (l == 0) ? a.xOff : (((l & 1) == 1) ? a.paOff : a.pbOff);
            const int64_t outOff = ((l & 1) == 0) ? a.paOff : a.pbOff;

            for (int64_t d = 0; d < D; ++d) {
                const int64_t lnd = l * D + d;
                __gm__ uint8_t* wihBase = (__gm__ uint8_t*)a.wih[lnd];
                __gm__ uint8_t* whhBase = (__gm__ uint8_t*)a.whh[lnd];

                /* r / z combined bias = b_ih + b_hh; n chunk keeps both halves separate */
                if (a.hasBias) {
                    const int64_t hs = (int64_t)H * (int64_t)sizeof(T);
                    __gm__ uint8_t* bi = (__gm__ uint8_t*)a.bih[lnd];
                    __gm__ uint8_t* bh = (__gm__ uint8_t*)a.bhh[lnd];
                    GruLoadF<T>(bR, bi, H, qIn);
                    GruLoadF<T>(t4, bh, H, qIn);
                    Add(bR, bR, t4, (int32_t)H);
                    GruLoadF<T>(bZ, bi + hs, H, qIn);
                    GruLoadF<T>(t4, bh + hs, H, qIn);
                    Add(bZ, bZ, t4, (int32_t)H);
                    GruLoadF<T>(bni, bi + 2 * hs, H, qIn);
                    GruLoadF<T>(bnh, bh + 2 * hs, H, qIn);
                } else {
                    Duplicate(bR, 0.0f, (int32_t)H);
                    Duplicate(bZ, 0.0f, (int32_t)H);
                    Duplicate(bni, 0.0f, (int32_t)H);
                    Duplicate(bnh, 0.0f, (int32_t)H);
                }

                /* initial hidden state */
                if (a.hasH0) {
                    __gm__ uint8_t* h0b = (__gm__ uint8_t*)((__gm__ T*)a.h0 + (lnd * B + b) * H);
                    GruLoadF<T>(hBuf, h0b, H, qIn);
                    if (Hp > H) {
                        Duplicate(hBuf[H], 0.0f, (int32_t)(Hp - H));
                    }
                } else {
                    Duplicate(hBuf, 0.0f, (int32_t)Hp);
                }

                const int64_t tBeg = (d == 0) ? 0 : (S - 1);
                const int64_t tEnd = (d == 0) ? S : -1;
                const int64_t tStep = (d == 0) ? 1 : -1;

                for (int64_t t = tBeg; t != tEnd; t += tStep) {
                    /* input row vector */
                    const int64_t irow = t * B + b;
                    LocalTensor<float> v = qV.AllocTensor<float>();
                    {
                        DataCopyExtParams cpi{1, (uint32_t)(inW * 4), 0u, 0u, 0u};
                        DataCopyPadExtParams<float> pp{false, 0u, 0u, 0.0f};
                        DataCopyPad(v, wsg[inOff + irow * inW], cpi, pp);
                    }
                    qV.EnQue(v);
                    v = qV.DeQue<float>();

                    GruMatVecPair<T>(gIR, wihBase, inW, v, gHR, whhBase, H, hBuf, H,
                                     qW, wf, accA);
                    GruMatVecPair<T>(gIZ, wihBase + (int64_t)H * inW * (int64_t)sizeof(T), inW, v,
                                     gHZ, whhBase + (int64_t)H * H * (int64_t)sizeof(T), H, hBuf, H,
                                     qW, wf, accA);
                    GruMatVecPair<T>(gIN, wihBase + 2 * (int64_t)H * inW * (int64_t)sizeof(T), inW, v,
                                     gHN, whhBase + 2 * (int64_t)H * H * (int64_t)sizeof(T), H, hBuf, H,
                                     qW, wf, accA);
                    PipeBarrier<PIPE_ALL>();
                    qV.FreeTensor(v);

                    /* r = sigmoid(gi_r + gh_r + bias_r), z likewise; n keeps gi/gh separate */
                    Add(gR, gIR, gHR, (int32_t)H);
                    Add(gR, gR, bR, (int32_t)H);
                    Add(gZ, gIZ, gHZ, (int32_t)H);
                    Add(gZ, gZ, bZ, (int32_t)H);
                    Add(gIN, gIN, bni, (int32_t)H);
                    Add(gHN, gHN, bnh, (int32_t)H);

                    Muls(tR, gR, -1.0f, (int32_t)H);
                    Exp(t4, tR, (int32_t)H);
                    Adds(t4, t4, 1.0f, (int32_t)H);
                    Div(tR, ones, t4, (int32_t)H);          /* tR = r */

                    Muls(tZ, gZ, -1.0f, (int32_t)H);
                    Exp(t4, tZ, (int32_t)H);
                    Adds(t4, t4, 1.0f, (int32_t)H);
                    Div(tZ, ones, t4, (int32_t)H);          /* tZ = z */

                    /* n = tanh(gi_n + r * gh_n) = 2*sigmoid(2y) - 1 */
                    Mul(tN, tR, gHN, (int32_t)H);
                    Add(tN, tN, gIN, (int32_t)H);
                    Muls(t4, tN, 2.0f, (int32_t)H);
                    Muls(tN, t4, -1.0f, (int32_t)H);
                    Exp(t4, tN, (int32_t)H);
                    Adds(t4, t4, 1.0f, (int32_t)H);
                    Div(tN, ones, t4, (int32_t)H);          /* sigmoid(2y) */
                    Muls(t4, tN, 2.0f, (int32_t)H);
                    Adds(tN, t4, -1.0f, (int32_t)H);        /* tN = n */

                    /* h_new = n + z * (h_old - n) */
                    Sub(t4, hBuf, tN, (int32_t)H);
                    Mul(t4, t4, tZ, (int32_t)H);
                    Add(hNew, tN, t4, (int32_t)H);

                    /* store the sequence value */
                    if (lastLayer) {
                        LocalTensor<T> ot = qOut.AllocTensor<T>();
                        if constexpr (std::is_same<T, float>::value) {
                            Adds(ot, hNew, 0.0f, (int32_t)H);
                        } else {
                            Cast(ot, hNew, RoundMode::CAST_RINT, (int32_t)H);
                        }
                        qOut.EnQue(ot);
                        ot = qOut.DeQue<T>();
                        const int64_t yrow = a.batchFirst ? (b * S + t) : (t * B + b);
                        DataCopyExtParams cpo{1, (uint32_t)(H * (int64_t)sizeof(T)), 0u, 0u, 0u};
                        DataCopyPad(yg[yrow * DH + d * H], ot, cpo);
                        qOut.FreeTensor(ot);
                    } else {
                        LocalTensor<float> of = qF.AllocTensor<float>();
                        Adds(of, hNew, 0.0f, (int32_t)H);
                        qF.EnQue(of);
                        of = qF.DeQue<float>();
                        const int64_t prow = t * B + b;
                        DataCopyExtParams cpo{1, (uint32_t)(H * 4), 0u, 0u, 0u};
                        DataCopyPad(wsg[outOff + prow * DH + d * H], of, cpo);
                        qF.FreeTensor(of);
                    }

                    Adds(hBuf, hNew, 0.0f, (int32_t)H);
                }

                /* final hidden state of this layer/direction */
                LocalTensor<T> ot = qOut.AllocTensor<T>();
                if constexpr (std::is_same<T, float>::value) {
                    Adds(ot, hBuf, 0.0f, (int32_t)H);
                } else {
                    Cast(ot, hBuf, RoundMode::CAST_RINT, (int32_t)H);
                }
                qOut.EnQue(ot);
                ot = qOut.DeQue<T>();
                DataCopyExtParams cpo{1, (uint32_t)(H * (int64_t)sizeof(T)), 0u, 0u, 0u};
                DataCopyPad(hng[(lnd * B + b) * H], ot, cpo);
                qOut.FreeTensor(ot);
            }

            /* the plane written by this layer is read back (MTE2) by the next layer */
            PipeBarrier<PIPE_ALL>();
        }
    }
}

}  // namespace

/* ---------------- host-side tiling + launch wrappers ---------------- */
int64_t calc_gru_blocks(int64_t B)
{
    if (B < 1) {
        B = 1;
    }
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = 0;
    if (plat != nullptr) {
        coreNum = (int64_t)plat->GetCoreNumAiv();
    }
    if (coreNum <= 0) {
        coreNum = 40;
    }
    return (B < coreNum) ? B : coreNum;
}

extern "C" {

void launch_gru_float(GruArgs a, void* stream)
{
    int64_t nb = calc_gru_blocks(a.B);
    gru_kernel<float><<<nb, nullptr, stream>>>(a);
}

void launch_gru_half(GruArgs a, void* stream)
{
    int64_t nb = calc_gru_blocks(a.B);
    gru_kernel<half><<<nb, nullptr, stream>>>(a);
}

void launch_gru_bfloat16(GruArgs a, void* stream)
{
    int64_t nb = calc_gru_blocks(a.B);
    gru_kernel<bfloat16_t><<<nb, nullptr, stream>>>(a);
}

}  // extern "C"
