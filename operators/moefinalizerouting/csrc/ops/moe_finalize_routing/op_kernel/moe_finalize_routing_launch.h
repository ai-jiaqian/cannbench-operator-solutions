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
 * \file moe_finalize_routing_launch.h
 * \brief Host side declarations shared by the bisheng kernel TU and the g++ plugin TU.
 */

#ifndef MOE_FINALIZE_ROUTING_LAUNCH_H
#define MOE_FINALIZE_ROUTING_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Host side tiling (platform sizes only). Returns (numBlocks, Hc, NB).
// Hc is the H-chunk length; it is always a positive multiple of 16 so that
// Hc * elemSize is a multiple of 32 bytes (one UB data block) and the resident
// bias table has a uniform row pitch. NB is the number of (row, k) gathers that
// share one queue element (batched, double-buffered random reads).
std::tuple<int64_t, int64_t, int64_t> calc_mfr_tiling(int64_t numRows, int64_t H, int64_t K, int64_t E,
                                                      int64_t elemSize);

#define MFR_ARGS                                                                              \
    GM_ADDR epr, GM_ADDR esdr, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias, GM_ADDR scales,     \
        GM_ADDR expert, GM_ADDR out, int64_t numRows, int64_t H, int64_t K, int64_t E,         \
        int64_t numDst, int64_t mode, int64_t Hc, int64_t NB, int64_t numBlocks, int64_t hasSkip1, \
        int64_t hasSkip2, int64_t hasBias, int64_t hasScales

extern "C" {

void launch_mfr_half_half(MFR_ARGS, void* stream);
void launch_mfr_half_float(MFR_ARGS, void* stream);
void launch_mfr_half_bf16(MFR_ARGS, void* stream);
void launch_mfr_float_half(MFR_ARGS, void* stream);
void launch_mfr_float_float(MFR_ARGS, void* stream);
void launch_mfr_float_bf16(MFR_ARGS, void* stream);
void launch_mfr_bf16_half(MFR_ARGS, void* stream);
void launch_mfr_bf16_float(MFR_ARGS, void* stream);
void launch_mfr_bf16_bf16(MFR_ARGS, void* stream);

}  // extern "C"

#endif  // MOE_FINALIZE_ROUTING_LAUNCH_H
