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
 * \file unique_launch.h
 * \brief Launch declarations shared between the bisheng kernel TU and the g++ plugin TU.
 *
 * All value computation happens inside the device kernels declared below.  The plugin only
 * allocates device workspaces, launches kernels, and reads back shape metadata (number of
 * distinct values D, and for the integer dtypes the reduced key range) because torch.unique's
 * output length is data dependent and cannot be derived from the input shape alone.
 */

#ifndef UNIQUE_LAUNCH_H
#define UNIQUE_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// dtype codes shared with the plugin
constexpr int64_t UK_DT_U8 = 0;
constexpr int64_t UK_DT_I8 = 1;
constexpr int64_t UK_DT_F16 = 2;
constexpr int64_t UK_DT_BF16 = 3;
constexpr int64_t UK_DT_I32 = 4;
constexpr int64_t UK_DT_I64 = 5;
constexpr int64_t UK_DT_F32 = 6;

// params[] slot layout (device int64 buffer, 8 slots)
constexpr int64_t UK_P_MINKEY = 0;
constexpr int64_t UK_P_CAP = 1;
constexpr int64_t UK_P_MODE = 2; // 0 = dense bitmap path, 1 = radix path
constexpr int64_t UK_P_NPASS = 3;
constexpr int64_t UK_P_D = 4;
constexpr int64_t UK_P_NUMBLK = 5;
constexpr int64_t UK_P_BLOCKLEN = 6;

// dense path capacity limit (codes); above this the radix path is selected on device
constexpr int64_t UK_MAXCAP = 262144;
// radix digit bins
constexpr int64_t UK_BINS = 256;

// host side tiling: number of AIV blocks to use for n elements
int64_t calc_unique_blocks(int64_t n);

extern "C" {

// ---------- setup ----------
void ukL_setparams(GM_ADDR params, int64_t minKey, int64_t cap, int64_t mode, int64_t npass, int64_t numBlocks,
                   int64_t blockLength, void* stream);
void ukL_mm32(GM_ADDR x, GM_ADDR mm, int64_t n, int64_t blockLength, void* stream);
void ukL_mm64(GM_ADDR x, GM_ADDR mm, int64_t n, int64_t blockLength, void* stream);
void ukL_prep(GM_ADDR mm, GM_ADDR params, int64_t numBlocks, int64_t keyBase, int64_t nv64, void* stream);

// ---------- dense path ----------
void ukL_dpres(GM_ADDR x, GM_ADDR bitmaps, GM_ADDR params, int64_t n, int64_t blockLength, int64_t dt,
               int64_t minKey, int64_t cap, void* stream);
void ukL_dscan(GM_ADDR bitmaps, GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR params, int64_t numBlocks, int64_t cap,
               void* stream);
void ukL_dy(GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR yBig, GM_ADDR params, int64_t cap, int64_t numBlocks,
            int64_t dt, int64_t minKey, void* stream);
void ukL_dinv(GM_ADDR x, GM_ADDR comb, GM_ADDR wordPrefix, GM_ADDR inv, GM_ADDR params, int64_t n,
              int64_t blockLength, int64_t dt, int64_t minKey, int64_t cap, void* stream);

// ---------- radix path ----------
void ukL_rpack(GM_ADDR x, GM_ADDR keys, GM_ADDR pay, int64_t n, int64_t blockLength, int64_t dt, int64_t minKey,
               void* stream);
void ukL_rhist(GM_ADDR keys, GM_ADDR hist, int64_t n, int64_t blockLength, int64_t pass, void* stream);
void ukL_rscan(GM_ADDR hist, GM_ADDR start, int64_t numBlocks, void* stream);
void ukL_rscatter(GM_ADDR kIn, GM_ADDR pIn, GM_ADDR kOut, GM_ADDR pOut, GM_ADDR start, int64_t n,
                  int64_t blockLength, int64_t pass, void* stream);
void ukL_rflag(GM_ADDR keys, GM_ADDR flags, GM_ADDR cnts, int64_t n, int64_t blockLength, void* stream);
void ukL_rbase(GM_ADDR cnts, GM_ADDR base, GM_ADDR params, int64_t numBlocks, void* stream);
void ukL_remit(GM_ADDR keys, GM_ADDR flags, GM_ADDR base, GM_ADDR yBig, GM_ADDR rank, int64_t n,
               int64_t blockLength, int64_t dt, int64_t minKey, void* stream);
void ukL_rwiden(GM_ADDR rankSorted, GM_ADDR inv, int64_t n, int64_t blockLength, void* stream);
}

#endif // UNIQUE_LAUNCH_H
