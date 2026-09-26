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
 * \file scatter_launch.h
 * \brief Tiling description + launch declarations for the Scatter direct-launch kernel (visible to g++)
 */

#ifndef SCATTER_LAUNCH_H
#define SCATTER_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

// reduce modes, kept in sync with the plugin parser
enum ScatterReduceModeC : int64_t {
    SC_RM_UPDATE = 0,
    SC_RM_ADD = 1,
    SC_RM_MUL = 2,
    SC_RM_AMIN = 3,
    SC_RM_AMAX = 4,
};

// data dtype codes used by the launcher switch
enum ScatterDtypeCodeC : int64_t {
    SC_DT_F16 = 0,
    SC_DT_BF16 = 1,
    SC_DT_F32 = 2,
    SC_DT_I32 = 3,
    SC_DT_I64 = 4,
};

/* Plain POD tiling description handed from the (bisheng compiled) tiling helper to the g++ plugin.
 *
 * Output-domain decomposition: data / y are viewed as [outer, K, inner] and indices / updates as
 * [outer, Ku, inner] (every non-scatter dimension of indices/updates equals data's, which the plugin
 * asserts).  A work unit owns one (outer index o, inner chunk [i0, i0+tileL), scatter chunk
 * [k0, k0+tileK)) block of the output; the kernel loads that block from `data`, applies every update
 * whose target row falls inside [k0, k0+tileK) in row-major source order (overwrite => last wins),
 * and stores the block to `y`.  Every output element therefore has exactly one owner (the last inner
 * chunk may overlap its predecessor, but overlapping columns recompute the identical value).
 */
struct ScatterTiling {
    int64_t tileL;         // inner columns per unit (multiple of the 32 B alignment step, or inner)
    int64_t tileK;         // scatter-axis rows resident per unit
    int64_t ju;            // update rows staged per DMA chunk
    int64_t nL;            // number of inner chunks
    int64_t nK;            // number of scatter-axis chunks
    int64_t units;         // total work units
    int64_t unitsPerBlock; // work units handled by one AI core
    int64_t numBlocks;     // AI core grid size

    int64_t offBlk;   // byte offset of the resident compute block (CT)
    int64_t offRaw;   // byte offset of the raw (LT) staging area (promotion path only)
    int64_t offIdx;   // byte offset of the index staging area (IDX)
    int64_t offUpd;   // byte offset of the update staging area (CT)
    int64_t totalBytes; // UB bytes the kernel allocates
};

ScatterTiling calc_scatter_tiling(int64_t outer, int64_t kLen, int64_t kuLen, int64_t inner,
                                  int64_t esLT, int64_t esCT, int64_t idxBytes, int64_t promote,
                                  int64_t mode = SC_RM_UPDATE);

extern "C" {

void launch_scatter(GM_ADDR data, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                    int64_t outer, int64_t kLen, int64_t kuLen, int64_t inner,
                    int64_t tileL, int64_t tileK, int64_t ju,
                    int64_t nL, int64_t nK, int64_t units, int64_t unitsPerBlock, int64_t numBlocks,
                    int64_t offBlk, int64_t offRaw, int64_t offIdx, int64_t offUpd,
                    int64_t totalBytes,
                    int64_t dtypeCode, int64_t idxCode, int64_t mode, void *stream);
}

#endif // SCATTER_LAUNCH_H
