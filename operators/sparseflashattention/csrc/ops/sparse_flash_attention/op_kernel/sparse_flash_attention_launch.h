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
 * \file sparse_flash_attention_launch.h
 * \brief Launch / tiling interface shared between the g++ plugin and the bisheng kernel TU.
 */

#ifndef SPARSE_FLASH_ATTENTION_LAUNCH_H
#define SPARSE_FLASH_ATTENTION_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

/*!
 * POD description of one SparseFlashAttention invocation.
 *
 * All strides are in ELEMENTS and are read back from the actual tensors, so BSND and BNSD (and any
 * legal non-contiguous outer layout) run through exactly the same kernel body. Indexing is always by
 * (b, n1|n2, s1|s2, d) with a per-tensor element stride for each axis.
 *
 * Work decomposition: one task handles one (b, n2, s1) group and HG query heads of its GQA group.
 * The HG heads share one sparse index row, so the gathered K/V rows are loaded once and reused.
 */
struct SfaParams {
    int64_t qSN, qSS, qSB;   // query      stride over (head n1, seq s1, batch b)
    int64_t kSN, kSS, kSB;   // key        stride over (head n2, seq s2, batch b)
    int64_t vSN, vSS, vSB;   // value      stride over (head n2, seq s2, batch b)
    int64_t iSN, iSS, iSB;   // sparseIdx  stride over (head n2, seq s1, batch b)
    int64_t ySN, ySS, ySB;   // output     stride over (head n1, seq s1, batch b)

    int64_t B, N1, N2, S1, S2, Dk, Dv, topK, G;

    int64_t HG;              // query heads handled by one task (<= G)
    int64_t KT;              // keys per key block (8 or 16, power of two)
    int64_t numKB;           // ceil(topK / KT)
    int64_t topKAl;          // numKB * KT  (score-row length, multiple of KT)
    int64_t DkAl;            // Dk rounded up to a 32B block
    int64_t DvAl;            // Dv rounded up to a 32B block
    int64_t DmxAl;           // max(DkAl, DvAl), the shared K/V staging row pitch
    int64_t numHeadBlk;      // ceil(G / HG)
    int64_t totalTasks;      // B * N2 * S1 * numHeadBlk
    int64_t isCausal;
    int64_t s2m1;            // S2 - S1 (bottom-right causal alignment offset)
    float   scale;
};

extern "C" {

void launch_sfa_half(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR idx, GM_ADDR y,
                     SfaParams p, int64_t numBlocks, void *stream);
void launch_sfa_bfloat16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR idx, GM_ADDR y,
                         SfaParams p, int64_t numBlocks, void *stream);

}

#endif // SPARSE_FLASH_ATTENTION_LAUNCH_H
