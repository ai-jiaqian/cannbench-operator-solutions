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
 * \file conv_3d_bpf_launch.h
 * \brief Launch/tiling declarations shared by the bisheng kernel TU and the g++ plugin TU.
 */

#ifndef CONV_3D_BPF_LAUNCH_H
#define CONV_3D_BPF_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

struct C3BpfTiling {
    int64_t numBlocks;
    int64_t CT;
    int64_t NT;
    int64_t RT;
    int64_t RW;
    int64_t chunkLen;
    int64_t nColTilesPerCi;
    int64_t ncpg;
    int64_t numItems;
    int64_t rowStride;
    int64_t subPl;
    int64_t jmax;
    int64_t b1x1;   // 1: Kd=Kh=Kw=1, no pad, unit stride -> flat contiguous contraction
};

C3BpfTiling calc_conv_3d_bpf_tiling(
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Dout, int64_t Hout, int64_t Wout,
    int64_t Kd, int64_t Kh, int64_t Kw,
    int64_t sd, int64_t sh, int64_t sw,
    int64_t pd, int64_t ph, int64_t pw,
    int64_t dd, int64_t dh, int64_t dw,
    int64_t groups, int64_t elemSize);

extern "C" {

void launch_conv3d_bpf_main_half(
    GM_ADDR x, GM_ADDR g, GM_ADDR y, GM_ADDR xs,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Dout, int64_t Hout, int64_t Wout,
    int64_t Kd, int64_t Kh, int64_t Kw,
    int64_t sd, int64_t sh, int64_t sw,
    int64_t pd, int64_t ph, int64_t pw,
    int64_t dd, int64_t dh, int64_t dw,
    int64_t Cin_g, int64_t Cout_g, int64_t groups,
    int64_t rowStride, int64_t subPl, int64_t RW,
    int64_t CT, int64_t NT, int64_t RT,
    int64_t ncpg, int64_t ntpc, int64_t numItems, int64_t numBlocks,
    int64_t b1x1,
    void* stream);

void launch_conv3d_bpf_main_bf16(
    GM_ADDR x, GM_ADDR g, GM_ADDR y, GM_ADDR xs,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Dout, int64_t Hout, int64_t Wout,
    int64_t Kd, int64_t Kh, int64_t Kw,
    int64_t sd, int64_t sh, int64_t sw,
    int64_t pd, int64_t ph, int64_t pw,
    int64_t dd, int64_t dh, int64_t dw,
    int64_t Cin_g, int64_t Cout_g, int64_t groups,
    int64_t rowStride, int64_t subPl, int64_t RW,
    int64_t CT, int64_t NT, int64_t RT,
    int64_t ncpg, int64_t ntpc, int64_t numItems, int64_t numBlocks,
    int64_t b1x1,
    void* stream);

void launch_conv3d_bpf_split_half(
    GM_ADDR x, GM_ADDR xs, int64_t rows, int64_t W, int64_t jmax,
    int64_t sw, int64_t DH, int64_t numBlocks, void* stream);

void launch_conv3d_bpf_split_bf16(
    GM_ADDR x, GM_ADDR xs, int64_t rows, int64_t W, int64_t jmax,
    int64_t sw, int64_t DH, int64_t numBlocks, void* stream);

} // extern "C"

#endif // CONV_3D_BPF_LAUNCH_H
