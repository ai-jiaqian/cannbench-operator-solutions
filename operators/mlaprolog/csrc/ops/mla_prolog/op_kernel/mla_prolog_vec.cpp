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
 * \file mla_prolog_vec.cpp
 * \brief MlaProlog vector stages (bisheng + -xasc): RMSNorm, RoPE, bf16 splitting and layout transforms.
 *
 * The buffers are plain TBuf scratch, so every cross-pipe dependency is inserted explicitly:
 *   MTE2 -> V after each GM->UB copy (and before the vector work consuming it),
 *   V -> MTE2 before a copy overwrites a buffer the vector unit has been reading,
 *   V -> MTE3 before a UB->GM store,
 *   V -> S / S -> V around the scalar reduce and broadcast steps.
 * Missing these is a silent data race that still "works" on all-zero inputs.
 *
 * RMSNorm is evaluated as gamma * x * inv where inv = 1/sqrt(mean(x^2)+eps).  The hardware Rsqrt is only an
 * approximation (~1e-3 relative, and its error is common to a whole row so it would appear as a systematic
 * scale error), so two Newton-Raphson refinements are applied: each squaring of the residual error takes
 * 1e-3 -> 1e-6 -> 1e-12.  Division is never approximated.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "mla_prolog_launch.h"

using namespace AscendC;

namespace mlav {

constexpr int32_t kBlkBytes = 32;

__aicore__ inline void SyncMte2ToV()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
__aicore__ inline void SyncVToMte2()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}
__aicore__ inline void SyncVToMte3()
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}
__aicore__ inline void SyncMte3ToV()
{
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}
__aicore__ inline void SyncMte3ToMte2()
{
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
__aicore__ inline void SyncVToS()
{
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
}
__aicore__ inline void SyncSToV()
{
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
}

/*! \brief build a DataCopyExtParams with the documented field names (strides stay non-negative). */
__aicore__ inline DataCopyExtParams MkExt(uint32_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                          uint32_t dstStride)
{
    DataCopyExtParams cp;
    cp.blockCount = (uint16_t)blockCount;
    cp.blockLen = blockLen;
    cp.srcStride = srcStride;
    cp.dstStride = dstStride;
    cp.rsv = 0;
    return cp;
}

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> MkPad()
{
    DataCopyPadExtParams<T> pad;
    pad.isPad = false;
    pad.leftPadding = 0;
    pad.rightPadding = 0;
    pad.paddingValue = 0;
    return pad;
}

/*! \brief one data block (not the whole tensor) */
template <typename T>
__aicore__ inline void CopyBlockIn(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                   uint32_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                   uint32_t dstStride)
{
    DataCopyPad(dst, src, MkExt(blockCount, blockLen, srcStride, dstStride), MkPad<T>());
}

template <typename T>
__aicore__ inline void CopyBlockOut(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                    uint32_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                    uint32_t dstStride)
{
    DataCopyPad(dst, src, MkExt(blockCount, blockLen, srcStride, dstStride));
}

/*! \brief one Newton-Raphson step for r = 1/sqrt(m):  r <- r*(1.5 - 0.5*m*r*r) */
__aicore__ inline void NrStep(const LocalTensor<float>& dst, const LocalTensor<float>& tmp,
                              const LocalTensor<float>& r, const LocalTensor<float>& m)
{
    Mul(tmp, r, r, (int32_t)8);
    Mul(tmp, tmp, m, (int32_t)8);
    Muls(tmp, tmp, -0.5f, (int32_t)8);
    Muls(dst, r, 1.5f, (int32_t)8);
    Add(dst, dst, tmp, (int32_t)8);
}

/*! \brief log-tree sum: afterwards t[0..7] holds 8 partial sums whose total equals the sum of t[0..len).
 *  Each step folds the upper half onto the lower one, so every operand address stays 32 B aligned and no
 *  vector instruction ever handles fewer than 8 f32 lanes.  The last 8 lanes are folded on the scalar unit.
 *  The caller must have zeroed t[count..len) beforehand. */
__aicore__ inline void SumTree(const LocalTensor<float>& t, int64_t len)
{
    while (len > 8) {
        const int64_t h = len / 2;
        Add(t, t, t[h], (int32_t)h);
        len = h;
    }
}

__aicore__ inline float ScalarFold8(const LocalTensor<float>& t)
{
    float s = t.GetValue(0);
    s += t.GetValue(1);
    s += t.GetValue(2);
    s += t.GetValue(3);
    s += t.GetValue(4);
    s += t.GetValue(5);
    s += t.GetValue(6);
    s += t.GetValue(7);
    return s;
}

/*! \brief exact 3-term bf16 split: hi + mid + lo reproduces an fp32 value bit-exactly.
 *  hi keeps the top 8 significand bits, mid the next 8 (the residual has at most 16 significant bits), and lo
 *  then holds the remaining <=8 bits, so the last cast is lossless. */
__aicore__ inline void Split3(const LocalTensor<bfloat16_t>& hi, const LocalTensor<bfloat16_t>& mid,
                              const LocalTensor<bfloat16_t>& lo, const LocalTensor<float>& v,
                              const LocalTensor<float>& s1, const LocalTensor<float>& s2, int32_t n)
{
    Cast(hi, v, RoundMode::CAST_RINT, n);
    Cast(s1, hi, RoundMode::CAST_NONE, n);
    Sub(s1, v, s1, n);
    Cast(mid, s1, RoundMode::CAST_RINT, n);
    Cast(s2, mid, RoundMode::CAST_NONE, n);
    Sub(s2, s1, s2, n);
    Cast(lo, s2, RoundMode::CAST_RINT, n);
}

}  // namespace mlav

// =========================================================================
// K2:  csplit[3M, Hcq]bf16 = 3-term bf16 split of RMSNorm(cq_raw, g_cq)
//      c_kv[M, Hckv]bf16   = RMSNorm(dkv[:, :Hckv], g_ckv)
//      k_rope[M, Dr]bf16   = RoPE(dkv[:, Hckv:], cos, sin)
// =========================================================================
__global__ __aicore__ void mla_v_norm_kernel(GM_ADDR cqPtr, GM_ADDR gcqPtr, GM_ADDR dkvPtr,
                                             GM_ADDR gckvPtr, GM_ADDR cosPtr, GM_ADDR sinPtr,
                                             GM_ADDR csplitPtr, GM_ADDR ckvPtr, GM_ADDR krPtr,
                                             int64_t M, int64_t Hcq, int64_t HkvDr, int64_t Hckv,
                                             int64_t Dr, float epsCq, float epsCkv, int64_t nBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace mlav;

    const int64_t blk = (int64_t)GetBlockIdx();
    if (blk >= nBlocks) {
        return;
    }
    const int64_t per = (M + nBlocks - 1) / nBlocks;
    const int64_t r0 = blk * per;
    int64_t r1 = r0 + per;
    if (r1 > M) {
        r1 = M;
    }
    if (r0 >= M) {
        return;
    }
    const int64_t halfDr = Dr / 2;
    const int64_t kPadX = 2048;
    const int64_t kPadD = 1024;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bX;   // f32 Hcq       row staging
    TBuf<TPosition::VECCALC> bSq;  // f32 HkvDr     squares / scratch
    TBuf<TPosition::VECCALC> bRd;  // f32 32        reduce result + rsqrt
    TBuf<TPosition::VECCALC> bNr;  // f32 32        newton scratch
    TBuf<TPosition::VECCALC> bG1;  // f32 Hcq       gamma_cq
    TBuf<TPosition::VECCALC> bG2;  // f32 Hckv      gamma_ckv
    TBuf<TPosition::VECCALC> bHi;  // bf16 Hcq
    TBuf<TPosition::VECCALC> bMd;  // bf16 Hcq
    TBuf<TPosition::VECCALC> bLo;  // bf16 Hcq
    TBuf<TPosition::VECCALC> bD;   // f32 HkvDr
    TBuf<TPosition::VECCALC> bO;   // f32 Hckv
    TBuf<TPosition::VECCALC> bR;   // f32 Dr        rope input
    TBuf<TPosition::VECCALC> bTv;  // f32 64        RMSNorm argument t, kept alive across Rsqrt
    TBuf<TPosition::VECCALC> bRot; // f32 Dr        rotate_half(input)
    TBuf<TPosition::VECCALC> bCf;  // f32 Dr
    TBuf<TPosition::VECCALC> bSf;  // f32 Dr        sin with the first half negated
    TBuf<TPosition::VECCALC> bY;   // f32 Dr
    TBuf<TPosition::VECCALC> bYb;  // bf16 Dr
    TBuf<TPosition::VECCALC> bCh;  // bf16 Dr       cos staging
    TBuf<TPosition::VECCALC> bSh;  // bf16 Dr       sin staging
    TBuf<TPosition::VECCALC> bGh;  // bf16 Hcq      gamma staging

    pipe.InitBuffer(bX, 2048 * 4);
    pipe.InitBuffer(bSq, 2048 * 4);
    pipe.InitBuffer(bRd, 32 * 4);
    pipe.InitBuffer(bNr, 32 * 4);
    pipe.InitBuffer(bG1, MLA_MAX_HCQ * 4);
    pipe.InitBuffer(bG2, MLA_MAX_HCKV * 4);
    pipe.InitBuffer(bHi, MLA_MAX_HCQ * 2);
    pipe.InitBuffer(bMd, MLA_MAX_HCQ * 2);
    pipe.InitBuffer(bLo, MLA_MAX_HCQ * 2);
    pipe.InitBuffer(bD, 1024 * 4);
    pipe.InitBuffer(bO, MLA_MAX_HCKV * 4);
    pipe.InitBuffer(bR, MLA_MAX_DR * 4);
    pipe.InitBuffer(bTv, 64 * 4);
    pipe.InitBuffer(bRot, MLA_MAX_DR * 4);
    pipe.InitBuffer(bCf, MLA_MAX_DR * 4);
    pipe.InitBuffer(bSf, MLA_MAX_DR * 4);
    pipe.InitBuffer(bY, 2048 * 4);
    pipe.InitBuffer(bYb, MLA_MAX_HCKV * 2);
    pipe.InitBuffer(bCh, MLA_MAX_DR * 2);
    pipe.InitBuffer(bSh, MLA_MAX_DR * 2);
    pipe.InitBuffer(bGh, MLA_MAX_HCQ * 2);

    LocalTensor<float> x = bX.Get<float>();
    LocalTensor<float> sq = bSq.Get<float>();
    LocalTensor<float> red = bRd.Get<float>();
    LocalTensor<float> nr = bNr.Get<float>();
    LocalTensor<float> g1 = bG1.Get<float>();
    LocalTensor<float> g2 = bG2.Get<float>();
    LocalTensor<bfloat16_t> hi = bHi.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> md = bMd.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lo = bLo.Get<bfloat16_t>();
    LocalTensor<float> dk = bD.Get<float>();
    LocalTensor<float> ckvf = bO.Get<float>();
    LocalTensor<float> rr = bR.Get<float>();
    LocalTensor<float> tv = bTv.Get<float>();
    LocalTensor<float> rot = bRot.Get<float>();
    LocalTensor<float> cf = bCf.Get<float>();
    LocalTensor<float> sf = bSf.Get<float>();
    LocalTensor<float> yy = bY.Get<float>();
    LocalTensor<bfloat16_t> yb = bYb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> ch = bCh.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> sh = bSh.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> gh = bGh.Get<bfloat16_t>();

    GlobalTensor<float> gCq;
    GlobalTensor<float> gDkv;
    GlobalTensor<bfloat16_t> gG1;
    GlobalTensor<bfloat16_t> gG2;
    GlobalTensor<bfloat16_t> gCos;
    GlobalTensor<bfloat16_t> gSin;
    GlobalTensor<bfloat16_t> gSplit;
    GlobalTensor<bfloat16_t> gCkv;
    GlobalTensor<bfloat16_t> gKr;
    gCq.SetGlobalBuffer((__gm__ float*)cqPtr);
    gDkv.SetGlobalBuffer((__gm__ float*)dkvPtr);
    gG1.SetGlobalBuffer((__gm__ bfloat16_t*)gcqPtr);
    gG2.SetGlobalBuffer((__gm__ bfloat16_t*)gckvPtr);
    gCos.SetGlobalBuffer((__gm__ bfloat16_t*)cosPtr);
    gSin.SetGlobalBuffer((__gm__ bfloat16_t*)sinPtr);
    gSplit.SetGlobalBuffer((__gm__ bfloat16_t*)csplitPtr);
    gCkv.SetGlobalBuffer((__gm__ bfloat16_t*)ckvPtr);
    gKr.SetGlobalBuffer((__gm__ bfloat16_t*)krPtr);

    // gamma tables -> fp32, once per block
    CopyBlockIn(gh, gG1, 1, (uint32_t)(Hcq * 2), 0, 0);
    CopyBlockIn(md, gG2, 1, (uint32_t)(Hckv * 2), 0, 0);
    SyncMte2ToV();
    Cast(g1, gh, RoundMode::CAST_NONE, (int32_t)Hcq);
    Cast(g2, md, RoundMode::CAST_NONE, (int32_t)Hckv);
    SyncVToMte2();

    for (int64_t m = r0; m < r1; ++m) {
        // close the previous iteration's MTE3 stores and vector reads before reloading the scratch
        SyncVToMte2();
        SyncMte3ToV();
        // ---------------- c_q = RMSNorm(cq_raw) ----------------
        CopyBlockIn(x, gCq[m * Hcq], 1, (uint32_t)(Hcq * 4), 0, 0);
        SyncMte2ToV();
        Mul(sq, x, x, (int32_t)Hcq);
        Duplicate(sq[Hcq], 0.0f, (int32_t)(kPadX - Hcq));
        SumTree(sq, kPadX);
        SyncVToS();
        float ssum = ScalarFold8(sq);
        SyncSToV();
        red.SetValue(0, ssum / (float)Hcq + epsCq);
        SyncSToV();
        Rsqrt(tv, red, (int32_t)8);  // tv = rsqrt(t), red keeps t untouched
        SyncVToS();
        float rqc = tv.GetValue(0);
        SyncSToV();
        // Newton-Raphson for 1/sqrt(t) in plain scalar fp32: t is already a scalar, so no vector op is
        // involved and there is no chance of feeding the step the wrong buffer.  Two iterations take the
        // hardware Rsqrt's ~1e-3 relative error to ~1e-12.
        float tq = ssum / (float)Hcq + epsCq;
        float invCq = rqc * (1.5f - 0.5f * tq * rqc * rqc);
        invCq = invCq * (1.5f - 0.5f * tq * invCq * invCq);
        SyncSToV();
        Muls(yy, x, invCq, (int32_t)Hcq);
        Mul(yy, yy, g1, (int32_t)Hcq);
        Split3(hi, md, lo, yy, sq, x, (int32_t)Hcq);
        SyncVToMte3();
        CopyBlockOut(gSplit[m * Hcq], hi, 1, (uint32_t)(Hcq * 2), 0, 0);
        CopyBlockOut(gSplit[(M + m) * Hcq], md, 1, (uint32_t)(Hcq * 2), 0, 0);
        CopyBlockOut(gSplit[(2 * M + m) * Hcq], lo, 1, (uint32_t)(Hcq * 2), 0, 0);

        // ---------------- c_kv + k_rope from dkv ----------------
        SyncMte3ToMte2();
        CopyBlockIn(dk, gDkv[m * HkvDr], 1, (uint32_t)(HkvDr * 4), 0, 0);
        SyncMte2ToV();
        Mul(sq, dk, dk, (int32_t)Hckv);
        Duplicate(sq[Hckv], 0.0f, (int32_t)(kPadD - Hckv));
        SumTree(sq, kPadD);
        SyncVToS();
        float ssum2 = ScalarFold8(sq);
        SyncSToV();
        red.SetValue(0, ssum2 / (float)Hckv + epsCkv);
        SyncSToV();
        Rsqrt(tv, red, (int32_t)8);
        SyncVToS();
        float rkv = tv.GetValue(0);
        SyncSToV();
        float tkv = ssum2 / (float)Hckv + epsCkv;
        float invCkv = rkv * (1.5f - 0.5f * tkv * rkv * rkv);
        invCkv = invCkv * (1.5f - 0.5f * tkv * invCkv * invCkv);
        SyncSToV();
        Muls(ckvf, dk, invCkv, (int32_t)Hckv);
        Mul(ckvf, ckvf, g2, (int32_t)Hckv);
        Cast(yb, ckvf, RoundMode::CAST_RINT, (int32_t)Hckv);
        SyncVToMte3();
        CopyBlockOut(gCkv[m * Hckv], yb, 1, (uint32_t)(Hckv * 2), 0, 0);

        // RoPE on dkv[:, Hckv:]:  y[j] = r[j]*cos[j] + rot[j]*sin'[j]
        //   rot = [rB | rA]  (swap halves),  sin' = [-sinA | sinB]
        SyncMte3ToMte2();
        CopyBlockIn(ch, gCos[m * Dr], 1, (uint32_t)(Dr * 2), 0, 0);
        CopyBlockIn(sh, gSin[m * Dr], 1, (uint32_t)(Dr * 2), 0, 0);
        SyncMte2ToV();
        Cast(cf, ch, RoundMode::CAST_NONE, (int32_t)Dr);
        Cast(sf, sh, RoundMode::CAST_NONE, (int32_t)Dr);
        // rr = [rA | rB] taken straight out of dk
        Muls(rr, dk[Hckv], 1.0f, (int32_t)Dr);
        Muls(rot, dk[Hckv + halfDr], 1.0f, (int32_t)halfDr);
        Muls(rot[halfDr], dk[Hckv], 1.0f, (int32_t)halfDr);
        // sin' : negate only the first half
        Muls(sf, sf, -1.0f, (int32_t)halfDr);
        Mul(yy, rr, cf, (int32_t)Dr);
        Mul(sq, rot, sf, (int32_t)Dr);
        Add(yy, yy, sq, (int32_t)Dr);
        Cast(yb, yy, RoundMode::CAST_RINT, (int32_t)Dr);
        SyncVToMte3();
        CopyBlockOut(gKr[m * Dr], yb, 1, (uint32_t)(Dr * 2), 0, 0);
        SyncMte3ToMte2();
    }
}

// =========================================================================
// K3:  wuk3[N][3D][Hckv]bf16 <- wuk[N][D][Hckv]bf16 repeated three times along the D axis
// =========================================================================
__global__ __aicore__ void mla_v_dupuk_kernel(GM_ADDR wukPtr, GM_ADDR wuk3Ptr, int64_t N, int64_t D,
                                              int64_t Hckv, int64_t nBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace mlav;

    const int64_t blk = (int64_t)GetBlockIdx();
    if (blk >= nBlocks) {
        return;
    }
    const int32_t kChunk = 8192;  // bf16 elements per copy = 16 KiB

    TPipe pipe;
    TBuf<TPosition::VECCALC> bT;
    pipe.InitBuffer(bT, kChunk * 2);
    LocalTensor<bfloat16_t> t = bT.Get<bfloat16_t>();

    GlobalTensor<bfloat16_t> gIn;
    GlobalTensor<bfloat16_t> gOut;
    gIn.SetGlobalBuffer((__gm__ bfloat16_t*)wukPtr);
    gOut.SetGlobalBuffer((__gm__ bfloat16_t*)wuk3Ptr);

    const int64_t per = (N + nBlocks - 1) / nBlocks;
    const int64_t n0 = blk * per;
    int64_t n1 = n0 + per;
    if (n1 > N) {
        n1 = N;
    }
    const int64_t blockElems = D * Hckv;
    for (int64_t n = n0; n < n1; ++n) {
        const int64_t srcBase = n * blockElems;
        const int64_t dstBase = n * 3 * blockElems;
        for (int64_t e = 0; e < blockElems; e += kChunk) {
            int64_t cnt = blockElems - e;
            if (cnt > kChunk) {
                cnt = kChunk;
            }
            CopyBlockIn(t, gIn[srcBase + e], 1, (uint32_t)(cnt * 2), 0, 0);
            SyncMte2ToV();
            SyncVToMte3();
            CopyBlockOut(gOut[dstBase + e], t, 1, (uint32_t)(cnt * 2), 0, 0);
            CopyBlockOut(gOut[dstBase + blockElems + e], t, 1, (uint32_t)(cnt * 2), 0, 0);
            CopyBlockOut(gOut[dstBase + 2 * blockElems + e], t, 1, (uint32_t)(cnt * 2), 0, 0);
            SyncMte3ToMte2();
        }
    }
}

// =========================================================================
// K5:  qr == qr2[0:M] + qr2[M:2M] + qr2[2M:3M]   (exact fp32 fold)
//      qcs[n][m][0:3D]bf16 = 3-term split of q_c[m, n, :]     (q_c = qr[:, n*PH : n*PH+D])
//      qropeIn[m][n*Dr + r]f32 = qr[m, n*PH + D + r]
// =========================================================================
__global__ __aicore__ void mla_v_fold_kernel(GM_ADDR qr2Ptr, GM_ADDR qcsPtr, GM_ADDR qropePtr,
                                             int64_t M, int64_t N, int64_t D, int64_t Dr, int64_t NW,
                                             int64_t nBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace mlav;

    const int64_t blk = (int64_t)GetBlockIdx();
    if (blk >= nBlocks) {
        return;
    }
    const int64_t per = (M + nBlocks - 1) / nBlocks;
    const int64_t r0 = blk * per;
    int64_t r1 = r0 + per;
    if (r1 > M) {
        r1 = M;
    }
    if (r0 >= M) {
        return;
    }
    const int64_t PH = D + Dr;
    const int64_t D3 = 3 * D;
    const int32_t kChunk = 2048;  // f32 elements per gather

    TPipe pipe;
    TBuf<TPosition::VECCALC> bP0;
    TBuf<TPosition::VECCALC> bP1;
    TBuf<TPosition::VECCALC> bP2;
    TBuf<TPosition::VECCALC> bHi;
    TBuf<TPosition::VECCALC> bMd;
    TBuf<TPosition::VECCALC> bLo;
    pipe.InitBuffer(bP0, kChunk * 4);
    pipe.InitBuffer(bP1, kChunk * 4);
    pipe.InitBuffer(bP2, kChunk * 4);
    pipe.InitBuffer(bHi, kChunk * 2);
    pipe.InitBuffer(bMd, kChunk * 2);
    pipe.InitBuffer(bLo, kChunk * 2);

    LocalTensor<float> p0 = bP0.Get<float>();
    LocalTensor<float> p1 = bP1.Get<float>();
    LocalTensor<float> p2 = bP2.Get<float>();
    LocalTensor<bfloat16_t> shi = bHi.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> smd = bMd.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> slo = bLo.Get<bfloat16_t>();

    GlobalTensor<float> gQr;
    GlobalTensor<bfloat16_t> gQcs;
    GlobalTensor<float> gQrope;
    gQr.SetGlobalBuffer((__gm__ float*)qr2Ptr);
    gQcs.SetGlobalBuffer((__gm__ bfloat16_t*)qcsPtr);
    gQrope.SetGlobalBuffer((__gm__ float*)qropePtr);

    const int64_t nChunks = (N * D + kChunk - 1) / kChunk;
    const int64_t chPerChunk = kChunk / D;  // heads handled by one q_c chunk

    for (int64_t m = r0; m < r1; ++m) {
        SyncVToMte2();
        SyncMte3ToV();
        // ---- q_c part ----
        for (int64_t ci = 0; ci < nChunks; ++ci) {
            const int64_t n0 = ci * chPerChunk;
            int64_t heads = N - n0;
            if (heads > chPerChunk) {
                heads = chPerChunk;
            }
            if (heads <= 0) {
                break;
            }
            const uint32_t blen = (uint32_t)(D * 4);
            const uint32_t sstride = (uint32_t)((PH - D) * 4);
            CopyBlockIn(p0, gQr[m * NW + n0 * PH], (uint32_t)heads, blen, sstride, 0);
            CopyBlockIn(p1, gQr[(M + m) * NW + n0 * PH], (uint32_t)heads, blen, sstride, 0);
            CopyBlockIn(p2, gQr[(2 * M + m) * NW + n0 * PH], (uint32_t)heads, blen, sstride, 0);
            SyncMte2ToV();
            Add(p0, p0, p1, (int32_t)(heads * D));
            Add(p0, p0, p2, (int32_t)(heads * D));
            Split3(shi, smd, slo, p0, p1, p2, (int32_t)(heads * D));
            SyncVToMte3();
            // qcs[n][3M][D]: the three terms are stacked along M (not along D) so the w_uk matmul can
            // run at K = D with the UNTRIPLED weight.  Each term is one contiguous D-wide run per
            // (head, m) pair, separated by 3M*D elements between consecutive heads.
            const uint32_t dstStride = (uint32_t)((3 * M * D - D) * 2);
            const uint32_t oblen = (uint32_t)(D * 2);
            const int64_t dstBase = (n0 * 3 * M + m) * D;
            CopyBlockOut(gQcs[dstBase], shi, (uint32_t)heads, oblen, 0, dstStride);
            CopyBlockOut(gQcs[dstBase + M * D], smd, (uint32_t)heads, oblen, 0, dstStride);
            CopyBlockOut(gQcs[dstBase + 2 * M * D], slo, (uint32_t)heads, oblen, 0, dstStride);
            SyncMte3ToMte2();
        }
        // ---- q_r part ----
        const int64_t rTot = N * Dr;
        for (int64_t k0 = 0; k0 < rTot; k0 += kChunk) {
            int64_t cnt = rTot - k0;
            if (cnt > kChunk) {
                cnt = kChunk;
            }
            const int64_t heads = cnt / Dr;
            const uint32_t blen = (uint32_t)(Dr * 4);
            const uint32_t sstride = (uint32_t)((PH - Dr) * 4);
            const int64_t off = k0 / Dr * PH + D;  // head index * PH + D
            CopyBlockIn(p0, gQr[m * NW + off], (uint32_t)heads, blen, sstride, 0);
            CopyBlockIn(p1, gQr[(M + m) * NW + off], (uint32_t)heads, blen, sstride, 0);
            CopyBlockIn(p2, gQr[(2 * M + m) * NW + off], (uint32_t)heads, blen, sstride, 0);
            SyncMte2ToV();
            Add(p0, p0, p1, (int32_t)(heads * Dr));
            Add(p0, p0, p2, (int32_t)(heads * Dr));
            SyncVToMte3();
            CopyBlockOut(gQrope[m * rTot + k0], p0, 1, (uint32_t)(cnt * 4), 0, 0);
            SyncMte3ToMte2();
        }
    }
}

// =========================================================================
// K6b: qtmp[N][M][Hckv]bf16 = qtmp3[n][m] + qtmp3[n][M+m] + qtmp3[n][2M+m]   (exact fp32 fold)
//
// The three q_c terms were multiplied by the untripled w_uk as three separate row bands, so the
// per-head products are summed here in fp32 before rounding once to bf16 - exactly the same
// arithmetic as the K-tripled variant, with none of its weight traffic.
// =========================================================================
__global__ __aicore__ void mla_v_foldq_kernel(GM_ADDR q3Ptr, GM_ADDR qoutPtr, int64_t M, int64_t N,
                                              int64_t Hckv, int64_t nBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace mlav;

    const int64_t blk = (int64_t)GetBlockIdx();
    if (blk >= nBlocks) {
        return;
    }
    // qtmp3[n] is [3M][Hckv] and qtmp[n] is [M][Hckv], i.e. the three terms are three CONTIGUOUS
    // planes of M*Hckv elements.  Iterating per (head, row) - as an earlier version did - costs M*N
    // tiny 512-element transfers instead of N*M*Hckv/(kRows*Hckv) large ones, which made this
    // kernel the dominant cost for every M above ~16 (measured: M=512 case +1012 us vs the
    // K-tripled variant).  Chunk the contiguous per-head planes instead.
    //
    // The transfer is issued as kRows separate 32B-aligned blocks (blockLen = Hckv*4 = 2048 B,
    // blockCount <= kRows, zero gap on both sides) rather than one large block: a single 32 KB
    // DataCopyPad block was measured to hang the vector core (aivec timeout) even though the
    // documented blockLen limit is 2^21-1, while blockLen = 2048 with blockCount <= 16 is already
    // exercised by the fold kernel.
    const int64_t plane = M * Hckv;
    const int64_t row3 = 3 * plane;
    const int32_t kRows = 8;
    const int32_t chunkElems = kRows * (int32_t)Hckv;

    TPipe pipe;
    TBuf<TPosition::VECCALC> bA;
    TBuf<TPosition::VECCALC> bB;
    TBuf<TPosition::VECCALC> bC;
    TBuf<TPosition::VECCALC> bO;
    pipe.InitBuffer(bA, chunkElems * 4);
    pipe.InitBuffer(bB, chunkElems * 4);
    pipe.InitBuffer(bC, chunkElems * 4);
    pipe.InitBuffer(bO, chunkElems * 2);
    LocalTensor<float> a = bA.Get<float>();
    LocalTensor<float> b = bB.Get<float>();
    LocalTensor<float> c = bC.Get<float>();
    LocalTensor<bfloat16_t> o = bO.Get<bfloat16_t>();

    GlobalTensor<float> gIn;
    GlobalTensor<bfloat16_t> gOut;
    gIn.SetGlobalBuffer((__gm__ float*)q3Ptr);
    gOut.SetGlobalBuffer((__gm__ bfloat16_t*)qoutPtr);

    const int64_t nCh = (M + kRows - 1) / kRows;  // row-chunks per head
    const int64_t totalCh = N * nCh;
    const int64_t per = (totalCh + nBlocks - 1) / nBlocks;
    const int64_t c0 = blk * per;
    int64_t c1 = c0 + per;
    if (c1 > totalCh) {
        c1 = totalCh;
    }
    const uint32_t row_bytes = (uint32_t)(Hckv * 4);
    const uint32_t out_bytes = (uint32_t)(Hckv * 2);
    for (int64_t ci = c0; ci < c1; ++ci) {
        const int64_t n = ci / nCh;
        const int64_t k = ci - n * nCh;
        const int64_t r0 = k * kRows;
        int64_t rows = M - r0;
        if (rows > kRows) {
            rows = kRows;
        }
        const int64_t off = r0 * Hckv;
        const int64_t src = n * row3 + off;
        const int32_t cnt = (int32_t)(rows * Hckv);
        SyncVToMte2();
        CopyBlockIn(a, gIn[src], (uint32_t)rows, row_bytes, 0, 0);
        CopyBlockIn(b, gIn[src + plane], (uint32_t)rows, row_bytes, 0, 0);
        CopyBlockIn(c, gIn[src + 2 * plane], (uint32_t)rows, row_bytes, 0, 0);
        SyncMte2ToV();
        Add(a, a, b, cnt);
        Add(a, a, c, cnt);
        Cast(o, a, RoundMode::CAST_RINT, cnt);
        SyncVToMte3();
        CopyBlockOut(gOut[n * plane + off], o, (uint32_t)rows, out_bytes, 0, 0);
        SyncMte3ToMte2();
    }
}

// =========================================================================
// K7:  query[M][N*Hckv]bf16 = transpose(qtmp[N][M][Hckv])
//      query_rope[M][N*Dr]bf16 = RoPE(qropeIn[M][N*Dr])
// =========================================================================
__global__ __aicore__ void mla_v_final_kernel(GM_ADDR qtmpPtr, GM_ADDR qropePtr, GM_ADDR cosPtr,
                                              GM_ADDR sinPtr, GM_ADDR queryPtr, GM_ADDR qropeOutPtr,
                                              int64_t M, int64_t N, int64_t Hckv, int64_t Dr,
                                              int64_t nBlocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace mlav;

    const int64_t blk = (int64_t)GetBlockIdx();
    if (blk >= nBlocks) {
        return;
    }
    const int64_t per = (M + nBlocks - 1) / nBlocks;
    const int64_t r0 = blk * per;
    int64_t r1 = r0 + per;
    if (r1 > M) {
        r1 = M;
    }
    if (r0 >= M) {
        return;
    }
    const int64_t halfDr = Dr / 2;
    const int64_t rTot = N * Dr;

    const int32_t kTransHeads = 32;  // heads staged per transpose copy
    const int32_t kRopeHeads = 64;   // heads per RoPE chunk

    TPipe pipe;
    TBuf<TPosition::VECCALC> bT;    // bf16 transpose staging
    TBuf<TPosition::VECCALC> bQ;    // f32 [kRopeHeads, Dr]
    TBuf<TPosition::VECCALC> bRot;  // f32 [kRopeHeads, Dr]
    TBuf<TPosition::VECCALC> bY;    // f32 [kRopeHeads, Dr]
    TBuf<TPosition::VECCALC> bYb;   // bf16 [kRopeHeads, Dr]
    TBuf<TPosition::VECCALC> bRr;   // f32 [kRopeHeads, Dr]  rope input copy
    TBuf<TPosition::VECCALC> bCf;   // f32 Dr
    TBuf<TPosition::VECCALC> bSf;   // f32 Dr
    TBuf<TPosition::VECCALC> bCh;   // bf16 Dr
    TBuf<TPosition::VECCALC> bSh;   // bf16 Dr
    pipe.InitBuffer(bT, kTransHeads * MLA_MAX_HCKV * 2);
    pipe.InitBuffer(bQ, kRopeHeads * MLA_MAX_DR * 4);
    pipe.InitBuffer(bRot, kRopeHeads * MLA_MAX_DR * 4);
    pipe.InitBuffer(bY, kRopeHeads * MLA_MAX_DR * 4);
    pipe.InitBuffer(bYb, kRopeHeads * MLA_MAX_DR * 2);
    pipe.InitBuffer(bRr, kRopeHeads * MLA_MAX_DR * 4);
    pipe.InitBuffer(bCf, MLA_MAX_DR * 4);
    pipe.InitBuffer(bSf, MLA_MAX_DR * 4);
    pipe.InitBuffer(bCh, MLA_MAX_DR * 2);
    pipe.InitBuffer(bSh, MLA_MAX_DR * 2);

    LocalTensor<bfloat16_t> tbuf = bT.Get<bfloat16_t>();
    LocalTensor<float> qv = bQ.Get<float>();
    LocalTensor<float> rv = bRot.Get<float>();
    LocalTensor<float> yv = bY.Get<float>();
    LocalTensor<bfloat16_t> ybv = bYb.Get<bfloat16_t>();
    LocalTensor<float> rrv = bRr.Get<float>();
    LocalTensor<float> cf = bCf.Get<float>();
    LocalTensor<float> sf = bSf.Get<float>();
    LocalTensor<bfloat16_t> ch = bCh.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> sh = bSh.Get<bfloat16_t>();

    GlobalTensor<bfloat16_t> gTmp;
    GlobalTensor<float> gQrope;
    GlobalTensor<bfloat16_t> gCos;
    GlobalTensor<bfloat16_t> gSin;
    GlobalTensor<bfloat16_t> gQuery;
    GlobalTensor<bfloat16_t> gQropeOut;
    gTmp.SetGlobalBuffer((__gm__ bfloat16_t*)qtmpPtr);
    gQrope.SetGlobalBuffer((__gm__ float*)qropePtr);
    gCos.SetGlobalBuffer((__gm__ bfloat16_t*)cosPtr);
    gSin.SetGlobalBuffer((__gm__ bfloat16_t*)sinPtr);
    gQuery.SetGlobalBuffer((__gm__ bfloat16_t*)queryPtr);
    gQropeOut.SetGlobalBuffer((__gm__ bfloat16_t*)qropeOutPtr);

    // NOTE: deliberately only count-form vector ops here, mirroring the k_rope path that was measured exact.
    for (int64_t m = r0; m < r1; ++m) {
        SyncVToMte2();
        SyncMte3ToV();
        // ---- transpose qtmp[N][M][Hckv] into query[m][N*Hckv] ----
        for (int64_t n0 = 0; n0 < N; n0 += kTransHeads) {
            int64_t heads = N - n0;
            if (heads > kTransHeads) {
                heads = kTransHeads;
            }
            const uint32_t blen = (uint32_t)(Hckv * 2);
            const uint32_t sstride = (uint32_t)(M * Hckv * 2 - Hckv * 2);
            CopyBlockIn(tbuf, gTmp[n0 * M * Hckv + m * Hckv], (uint32_t)heads, blen, sstride, 0);
            SyncMte2ToV();
            SyncVToMte3();
            CopyBlockOut(gQuery[m * N * Hckv + n0 * Hckv], tbuf, 1, (uint32_t)(heads * Hckv * 2), 0, 0);
            SyncMte3ToMte2();
        }
        // ---- RoPE ----
        CopyBlockIn(ch, gCos[m * Dr], 1, (uint32_t)(Dr * 2), 0, 0);
        CopyBlockIn(sh, gSin[m * Dr], 1, (uint32_t)(Dr * 2), 0, 0);
        SyncMte2ToV();
        Cast(cf, ch, RoundMode::CAST_NONE, (int32_t)Dr);
        Cast(sf, sh, RoundMode::CAST_NONE, (int32_t)Dr);
        Muls(sf, sf, -1.0f, (int32_t)halfDr);
        // Chunked RoPE.  Within a chunk ONLY count-form vector ops on 32B-aligned offset views are
        // used, so the whole chunk needs exactly one GM load of q, one strided pair of loads for the
        // swapped halves, one cast and one GM store.  That removes the per-head GM round trip and the
        // per-head pipe syncs that dominated the small-M cases.
        for (int64_t n0 = 0; n0 < N; n0 += kRopeHeads) {
            int64_t heads = N - n0;
            if (heads > kRopeHeads) {
                heads = kRopeHeads;
            }
            const int64_t base = m * rTot + n0 * Dr;
            CopyBlockIn(qv, gQrope[base], 1, (uint32_t)(heads * Dr * 4), 0, 0);
            CopyBlockIn(rv, gQrope[base + halfDr], (uint32_t)heads, (uint32_t)(halfDr * 4),
                        (uint32_t)((Dr - halfDr) * 4), (uint32_t)(halfDr * 4 / kBlkBytes));
            CopyBlockIn(rv[halfDr], gQrope[base], (uint32_t)heads, (uint32_t)(halfDr * 4),
                        (uint32_t)((Dr - halfDr) * 4), (uint32_t)(halfDr * 4 / kBlkBytes));
            SyncMte2ToV();
            for (int64_t h = 0; h < heads; ++h) {
                Mul(yv[h * Dr], qv[h * Dr], cf, (int32_t)Dr);
                Mul(rrv, rv[h * Dr], sf, (int32_t)Dr);
                Add(yv[h * Dr], yv[h * Dr], rrv, (int32_t)Dr);
            }
            Cast(ybv, yv, RoundMode::CAST_RINT, (int32_t)(heads * Dr));
            SyncVToMte3();
            CopyBlockOut(gQropeOut[base], ybv, 1, (uint32_t)(heads * Dr * 2), 0, 0);
            SyncMte3ToMte2();
        }
    }
}

// =========================================================================
// launch wrappers
// =========================================================================
extern "C" {

void launch_mla_v_norm(GM_ADDR cq, GM_ADDR gcq, GM_ADDR dkv, GM_ADDR gckv, GM_ADDR cosG, GM_ADDR sinG,
                       GM_ADDR csplit, GM_ADDR ckvOut, GM_ADDR krOut,
                       int64_t M, int64_t Hcq, int64_t HkvDr, int64_t Hckv, int64_t Dr,
                       float epsCq, float epsCkv, int64_t nBlocks, void* stream)
{
    mla_v_norm_kernel<<<nBlocks, nullptr, stream>>>((GM_ADDR)cq, (GM_ADDR)gcq, (GM_ADDR)dkv,
                                                    (GM_ADDR)gckv, (GM_ADDR)cosG, (GM_ADDR)sinG,
                                                    (GM_ADDR)csplit, (GM_ADDR)ckvOut, (GM_ADDR)krOut,
                                                    M, Hcq, HkvDr, Hckv, Dr, epsCq, epsCkv, nBlocks);
}

void launch_mla_v_dupuk(GM_ADDR wuk, GM_ADDR wuk3, int64_t N, int64_t D, int64_t Hckv,
                        int64_t nBlocks, void* stream)
{
    mla_v_dupuk_kernel<<<nBlocks, nullptr, stream>>>((GM_ADDR)wuk, (GM_ADDR)wuk3, N, D, Hckv, nBlocks);
}

void launch_mla_v_fold(GM_ADDR qr2, GM_ADDR qcs, GM_ADDR qropeIn, int64_t M, int64_t N, int64_t D,
                       int64_t Dr, int64_t NW, int64_t nBlocks, void* stream)
{
    mla_v_fold_kernel<<<nBlocks, nullptr, stream>>>((GM_ADDR)qr2, (GM_ADDR)qcs, (GM_ADDR)qropeIn,
                                                    M, N, D, Dr, NW, nBlocks);
}

void launch_mla_v_foldq(GM_ADDR qtmp3, GM_ADDR qtmp, int64_t M, int64_t N, int64_t Hckv,
                        int64_t nBlocks, void* stream)
{
    mla_v_foldq_kernel<<<nBlocks, nullptr, stream>>>((GM_ADDR)qtmp3, (GM_ADDR)qtmp, M, N, Hckv, nBlocks);
}

void launch_mla_v_final(GM_ADDR qtmp, GM_ADDR qropeIn, GM_ADDR cosG, GM_ADDR sinG, GM_ADDR query,
                        GM_ADDR queryRope, int64_t M, int64_t N, int64_t Hckv, int64_t Dr,
                        int64_t nBlocks, void* stream)
{
    mla_v_final_kernel<<<nBlocks, nullptr, stream>>>((GM_ADDR)qtmp, (GM_ADDR)qropeIn, (GM_ADDR)cosG,
                                                     (GM_ADDR)sinG, (GM_ADDR)query, (GM_ADDR)queryRope,
                                                     M, N, Hckv, Dr, nBlocks);
}

}  // extern "C"
