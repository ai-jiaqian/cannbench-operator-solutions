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
 * \file engram_launch.h
 * \brief Launch / tiling declarations shared between the bisheng kernel TU and the g++ plugin TU.
 *
 * EngramGateFusion golden semantics (fp32 internal, bf16 in/out):
 *   nk  = keys   / sqrt(mean_d(keys^2) + eps) * w1
 *   nh  = hidden / sqrt(mean_d(hid^2)  + eps) * w2
 *   raw = sum_d(nk*nh) / sqrt(D)
 *   gate= sigmoid(sqrt(clamp(|raw|,1e-6)) * sign(raw))
 *   vg  = gate * value
 *   nv  = rms_norm(vg, wc)                       <- the ShortConv input (fp32)
 *   xc  = cat([conv_state, nv]) along L
 *   y[l]= sum_k cw[hc*D+d, 0, k] * xc[l + k*dil]
 *   out = vg + SiLU(y)
 *   conv_state_out = xc[..., L:L+H, :],  H = (K-1)*dil
 *
 * CARRIER SCALING.  The kernel's circular window stores `w = xc / wc = value * gate / rms_vg`
 * rather than `xc` itself, and each main block builds its conv weights already scaled to
 * `cw' = cw * wc`.  Since sum_k cw'[k]*w == sum_k cw[k]*xc exactly up to rounding, this removes
 * one full width multiply from every row of the hot loop.  The places where that matters:
 *   * the H seed rows of a decode call arrive as real `xc` values and are divided by wc in the
 *     main prologue;
 *   * the H rows handed to the state output are multiplied back by wc before they are stored.
 *
 * THREE KERNELS, NOT FOUR.  The weight prep (the [d][K] -> [K][d] transpose plus w1*w2) used to be
 * its own kernel, but a whole extra launch cost about as much as the entire fixed part of a call,
 * and the transpose is only K Gathers per main block.  Each main block now builds its own cwT in
 * UB.  Only the two transposes that genuinely need a separate device pass (the [d][j] <-> [j][d]
 * shuffles of conv_state / conv_state_out) are still separate kernels.
 */

#ifndef ENGRAM_LAUNCH_H
#define ENGRAM_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/* Host side tiling description (plain POD). */
struct EngramTiling {
    int64_t nDC;            /* (unused, kept for layout stability)             */
    int64_t dnPrep;         /* (unused)                                        */
    int64_t nDT;            /* channel tiles per (b, hc) for the state kernels */
    int64_t dtState;        /* channels per state kernel tile                  */
    int64_t nChunks;        /* time chunks per (b, hc)                         */
    int64_t chunkLt;        /* rows owned by one time chunk                    */
    int64_t numBlocksPrep;
    int64_t numBlocksState;
    int64_t numBlocksMain;
};

EngramTiling calc_engram_tiling(int64_t B, int64_t L, int64_t HC, int64_t D, int64_t K, int64_t dil);
int64_t calc_engram_state_rows(int64_t K, int64_t dil);

extern "C" {

/* Kernel 1 (decode only): transpose conv_state [B, HC*D, H] into wsi [B, HC, H, D] fp32. */
void launch_engram_state_in(GM_ADDR convState, GM_ADDR wsi, int64_t B, int64_t HC, int64_t D,
                            int64_t H, int64_t dt, int64_t nDT, void* stream);

/* Kernel 2: the fused operator.  It builds its own cwT = cw*wc and w12 = w1*w2 in UB. */
void launch_engram_main(GM_ADDR keys, GM_ADDR hidden, GM_ADDR value, GM_ADDR w1, GM_ADDR w2,
                        GM_ADDR wc, GM_ADDR cw, GM_ADDR wsi, GM_ADDR ws, GM_ADDR out,
                        int64_t B, int64_t L, int64_t HC, int64_t D, int64_t K, int64_t dil,
                        int64_t H, int64_t hasState, float eps,
                        int64_t nChunks, int64_t chunkLt, void* stream);

/* Kernel 3: transpose ws [B, HC, H, Dpad] fp32 into conv_state_out [B, HC*D, H] bf16. */
void launch_engram_state_out(GM_ADDR ws, GM_ADDR sout, int64_t B, int64_t HC, int64_t D,
                             int64_t H, int64_t dt, int64_t nDT, void* stream);
}

#endif  // ENGRAM_LAUNCH_H
