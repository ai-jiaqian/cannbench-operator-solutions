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
 * \file nms_launch.h
 * \brief Launch function declarations for the NMS operator (visible to g++)
 */

#ifndef NMS_LAUNCH_H
#define NMS_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

extern "C" {

/* Kernel A: de-interleave boxes[N,4] into four packed fp32 arrays (each nPad elements long). */
void launch_nms_layout_kernel(GM_ADDR boxes, GM_ADDR x1, GM_ADDR y1, GM_ADDR x2, GM_ADDR y2,
                              int64_t n, int64_t numBlocks, void* stream);

/* Kernel B: greedy NMS over the highest-score-first order; writes nPad int64 slots, of which the
 * first M hold the kept original indices (the rest are zeroed). */
void launch_nms_apply_kernel(GM_ADDR x1, GM_ADDR y1, GM_ADDR x2, GM_ADDR y2, GM_ADDR scores,
                             GM_ADDR outIdx, int64_t n, int64_t nPad, float iouThreshold,
                             void* stream);
}

#endif // NMS_LAUNCH_H
