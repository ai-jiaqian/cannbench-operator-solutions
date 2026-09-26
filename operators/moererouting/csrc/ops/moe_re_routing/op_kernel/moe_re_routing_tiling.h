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
 * \file moe_re_routing_tiling.h
 * \brief Plain-C++ tiling bundle shared by the bisheng kernel TU and the g++ plugin TU.
 *        Deliberately free of AscendC types and of GM_ADDR so it can be included on both sides.
 *        Every tunable policy number lives here so the kernel TU never needs editing to retune.
 */

#ifndef MOE_RE_ROUTING_TILING_H
#define MOE_RE_ROUTING_TILING_H

#include <cstdint>

/* Index staging capacity, in elements.  The ramp for one whole (rank, expert) segment has to fit
 * so that a segment costs one vector op and one DMA instead of one per streaming chunk. */
static const int64_t MOE_RR_IDX_CAP = 4096;

/* Cap of the raw count staging buffer. */
static const int64_t MOE_RR_MAX_CNT_RAW_BYTES = 32 * 1024;

/* Minimum token traffic (bytes, one direction) that justifies one launched block.  Every block
 * redundantly DMAs and walks the whole (N, E) count matrix and pays a fixed dispatch, so for the
 * small shapes an extra block costs more than it saves: measured on the old (scalar-prologue)
 * kernel, case 1 went 8.80us at 32 blocks -> 9.30us at 48, case 8 5.48 -> 6.38, case 14 9.08 ->
 * 9.64.  64 KiB per block keeps those shapes at 16-32 blocks while the large shapes stay at the
 * core count. */
static const int64_t MOE_RR_BYTES_PER_BLOCK = 65536;

/* Parallelism floor: never launch fewer than this many blocks (nor more than there are cells),
 * because a block owns one contiguous destination window and a window narrower than the average
 * cell splits that cell into several individually synchronised segments.  Measured: collapsing to
 * 1-2 blocks costs 1.3-3.1 us on the smallest shapes. */
static const int64_t MOE_RR_CELL_FLOOR = 16;

/* UB left unclaimed as safety margin. */
static const int64_t MOE_RR_UB_SAFETY = 4096;

struct MoeRRTiling {
    int64_t numBlocks;   /* launched AIV blocks                                  */
    int64_t dstPerBlk;   /* contiguous destination token rows owned by one block */
    int64_t chunkTokens; /* streaming chunk granularity, in tokens               */
    int64_t idxCap;      /* index ramp staging capacity, in elements             */
};

#endif  // MOE_RE_ROUTING_TILING_H
