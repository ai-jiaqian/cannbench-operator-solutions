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
 * \file depthwise_conv_2d_launch.h
 * \brief Launch / tiling declarations shared by the bisheng kernel TU and the g++ plugin TU.
 *
 * DepthwiseConv2D:
 *   y[n,c,ho,wo] = bias[c] + sum_{kh,kw} x[n,c,ho*sh+kh*dh-ph, wo*sw+kw*dw-pw] * weight[c,kh,kw]
 * with x[N,C,H,W], weight[C,Kh,Kw], bias[C] and
 *   Hout = floor((H + 2*ph - dh*(Kh-1) - 1)/sh) + 1,  Wout likewise.
 *
 * The whole plan is handed to the kernel by value (no GM tiling blob).
 * Accumulation is always fp32; the single rounded Cast back to the input dtype happens on store.
 * Extended output width Wz = (Wout-1)*sw + 1 lets the stride be applied once, at the very end:
 * with z[i] = sum_{kh,kw} w[kh,kw]*xp[i + kw*dw] we have y[wo] = z[wo*sw].
 */

#ifndef DEPTHWISE_CONV_2D_LAUNCH_H
#define DEPTHWISE_CONV_2D_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

struct DWConvParams {
    int64_t N;
    int64_t C;
    int64_t H;
    int64_t W;
    int64_t Kh;
    int64_t Kw;
    int64_t sh;
    int64_t sw;
    int64_t ph;
    int64_t pw;
    int64_t dh;
    int64_t dw;
    int64_t Hout;
    int64_t Wout;
    int64_t Wz;        // (Wout-1)*sw + 1
    int64_t rowa;      // fp32 row pitch (elements), multiple of 16
    int64_t outa;      // decimated row pitch (elements), multiple of 16
    int64_t R;         // output rows per work item
    int64_t S;         // slab rows (slabMode), = R + (Kh-1)*dh
    int64_t slabMode;  // 1 => one DMA per kw over a vertical span, 0 => one DMA per (kh,kw)
    int64_t numRowBlocks;
    int64_t totalItems;
    int64_t numBlocks;
    int64_t elemSize;  // sizeof(T) of x / weight / bias / y
};

// Host side tiling (defined in the bisheng kernel TU).
DWConvParams calc_depthwise_conv_2d_tiling(int64_t N, int64_t C, int64_t H, int64_t W,
                                           int64_t Kh, int64_t Kw, int64_t sh, int64_t sw,
                                           int64_t ph, int64_t pw, int64_t dh, int64_t dw,
                                           int64_t Hout, int64_t Wout, int64_t elemSize);

extern "C" {

void launch_depthwise_conv_2d_float(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, DWConvParams p,
                                    void* stream);
void launch_depthwise_conv_2d_half(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, DWConvParams p,
                                   void* stream);
void launch_depthwise_conv_2d_bfloat16(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, DWConvParams p,
                                       void* stream);

} // extern "C"

#endif // DEPTHWISE_CONV_2D_LAUNCH_H
