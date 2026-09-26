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
 * \file wqmm_launch.h
 * \brief WeightQuantBatchMatmul: shared declarations for the bisheng kernel TUs and the g++ plugin TU.
 *
 *   y = x @ ANTIQUANT(weight) + bias ,   ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
 *
 *   x       [M, K]        float16 / bfloat16
 *   weight  [K, N]        int8
 *   scale   [N] or [1,N]  same dtype as x
 *   offset  [N] or [1,N]  optional, same dtype as scale
 *   bias    [N] or [1,N]  optional; float16 when x is float16, float32 when x is bfloat16
 *   y       [M, N]        same dtype as x
 *
 * Implementation shape (two device kernels on one stream):
 *
 *   1. wqmm_dequant : AIV kernel. Materialises ANTIQUANT(weight) into a T workspace, reproducing the
 *                     reference rounding chain per dtype (fp16 does the add and the multiply in half,
 *                     bfloat16 evaluates the chain in fp32 from the bf16 coefficients and rounds once).
 *   2. wqmm_mm      : pure-cube (ASCENDC_CUBE_ONLY) Matmul over the dequantised workspace.  The M/N axes
 *                     are split across the cube cores by the host tiling; the kernel adds the bias and
 *                     writes y directly in the output dtype, so no vector epilogue is needed.
 *
 * The cube tiling struct (AscendC::tiling::TCubeTiling, 50 x int32_t = 200 bytes) is transported to the
 * device by value inside WqmmTilingPOD, so the layout of that blob is part of the ABI between the two
 * translation units.
 */

#ifndef WQMM_LAUNCH_H
#define WQMM_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/*!
 * \brief Kernel-argument bundle: everything the two device kernels need besides the raw GM pointers.
 *
 * cube[] holds the serialised TCubeTiling blob (only the first 50 int32_t are meaningful).
 */
struct WqmmTilingPOD {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t singleCoreM;
    int32_t singleCoreN;
    int32_t mBlocks;
    int32_t nBlocks;
    int32_t hasBias;
    int32_t isBf16;
    int32_t reserved;
    int32_t diag[16];
    int32_t cube[64];
};

/*!
 * \brief Host tiling entry point, implemented in the bisheng translation unit and called from the plugin.
 *
 * \param numAic   requested cube core budget (<=0 means "use the platform maximum")
 * \param out      filled with the cube tiling blob and the block decomposition
 * \param dqCores / dqRowsPerCore / dqColTile   dequant kernel launch geometry
 * \param sysWsBytes  bytes of system workspace the Matmul API needs
 * \return 0 on success, negative when the Matmul tiling engine could not produce a legal tiling
 */
int64_t calc_wqmm_plan(int64_t M, int64_t N, int64_t K, int64_t isBf16, int64_t hasBias,
                       int64_t numAic, WqmmTilingPOD *out,
                       int64_t *dqCores, int64_t *dqRowsPerCore, int64_t *dqColTile,
                       int64_t *sysWsBytes);

extern "C" {

void launch_wqmm_dequant_half(GM_ADDR w, GM_ADDR s, GM_ADDR o, GM_ADDR out,
                              int64_t K, int64_t N, int64_t hasOff, int64_t numCores,
                              int64_t rowsPerCore, int64_t colTile, void *stream);

void launch_wqmm_dequant_bf16(GM_ADDR w, GM_ADDR s, GM_ADDR o, GM_ADDR out,
                              int64_t K, int64_t N, int64_t hasOff, int64_t numCores,
                              int64_t rowsPerCore, int64_t colTile, void *stream);

void launch_wqmm_mm_half(GM_ADDR x, GM_ADDR wd, GM_ADDR bias, GM_ADDR y, GM_ADDR ws,
                         WqmmTilingPOD tp, int64_t numBlocks, void *stream);

void launch_wqmm_mm_bf16(GM_ADDR x, GM_ADDR wd, GM_ADDR bias, GM_ADDR y, GM_ADDR ws,
                         WqmmTilingPOD tp, int64_t numBlocks, void *stream);

} // extern "C"

#endif // WQMM_LAUNCH_H
