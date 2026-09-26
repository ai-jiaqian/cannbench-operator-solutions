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
 * \file conv2d_launch.h
 * \brief Conv2D tiling / launch declarations shared by the bisheng kernel TU and the g++ plugin TU.
 *
 * conv_2d(x[N,Cin,H,W], filter[Cout,Cin,Kh,Kw], bias[Cout], strides, pads, dilations) -> y[N,Cout,Hout,Wout]
 *   y[n,co,ho,wo] = bias[co] + sum_{ci,kh,kw} x[n,ci,ho*sh-pt+kh*dh, wo*sw-pl+kw*dw] * filter[co,ci,kh,kw]
 * pads are [pad_top, pad_bottom, pad_left, pad_right].
 */

#ifndef CONV2D_LAUNCH_H
#define CONV2D_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// dtype codes shared between host tiling and the plugin
#define CONV2D_DTYPE_HALF 0
#define CONV2D_DTYPE_BF16 1
#define CONV2D_DTYPE_FLOAT 2

// Host side tiling result. Plain POD so both compilers agree on the layout.
struct Conv2dTiling {
    int64_t CT;         // output channels per work item
    int64_t HBT;        // output rows per work item (upper bound)
    int64_t Pr;         // row pitch (elements) of the accumulator / staged input rows
    int64_t PitW;       // row pitch (elements) of the staged weight slab
    int64_t ciChunk;    // input channels staged per weight slab
    int64_t numCoBlk;
    int64_t numHoBlk;
    int64_t totalItems;
    int64_t numBlocks;
    int64_t gsP;        // row pitch (elements) of the strided-path span buffer
    int64_t gsRep;      // GatherMask repeatTimes per row
    int64_t khGroup;    // how many kh rows are covered by one multi-row descriptor
};

// Computes the tiling above for one concrete shape. Never depends on values.
Conv2dTiling calc_conv2d_tiling(int64_t N, int64_t Cin, int64_t H, int64_t W,
                                int64_t Cout, int64_t Kh, int64_t Kw,
                                int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                                int64_t pt, int64_t pb, int64_t pl, int64_t ppR,
                                int64_t Hout, int64_t Wout, int64_t dtypeCode);

extern "C" {

void launch_conv2d_kernel_half(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               int64_t N, int64_t Cin, int64_t H, int64_t W,
                               int64_t Cout, int64_t Kh, int64_t Kw,
                               int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                               int64_t pt, int64_t pl, int64_t ppR, int64_t Hout, int64_t Wout,
                               int64_t CT, int64_t HBT, int64_t Pr, int64_t PitW, int64_t ciChunk,
                               int64_t numCoBlk, int64_t numHoBlk, int64_t totalItems,
                               int64_t gsP, int64_t gsRep, int64_t kg, int64_t numBlocks, void *stream);

void launch_conv2d_kernel_bfloat16(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                   int64_t N, int64_t Cin, int64_t H, int64_t W,
                                   int64_t Cout, int64_t Kh, int64_t Kw,
                                   int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                                   int64_t pt, int64_t pl, int64_t ppR, int64_t Hout, int64_t Wout,
                                   int64_t CT, int64_t HBT, int64_t Pr, int64_t PitW, int64_t ciChunk,
                                   int64_t numCoBlk, int64_t numHoBlk, int64_t totalItems,
                                   int64_t gsP, int64_t gsRep, int64_t kg, int64_t numBlocks, void *stream);

void launch_conv2d_kernel_float(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                                int64_t N, int64_t Cin, int64_t H, int64_t W,
                                int64_t Cout, int64_t Kh, int64_t Kw,
                                int64_t sh, int64_t sw, int64_t dh, int64_t dw,
                                int64_t pt, int64_t pl, int64_t ppR, int64_t Hout, int64_t Wout,
                                int64_t CT, int64_t HBT, int64_t Pr, int64_t PitW, int64_t ciChunk,
                                int64_t numCoBlk, int64_t numHoBlk, int64_t totalItems,
                                int64_t gsP, int64_t gsRep, int64_t kg, int64_t numBlocks, void *stream);

} // extern "C"

#endif // CONV2D_LAUNCH_H
