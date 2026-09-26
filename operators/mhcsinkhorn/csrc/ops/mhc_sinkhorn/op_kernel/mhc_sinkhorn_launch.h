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
 * \file mhc_sinkhorn_launch.h
 * \brief Launch function declarations for g++
 */

#ifndef MHC_SINKHORN_LAUNCH_H
#define MHC_SINKHORN_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling: returns (numBlocks, matPerCore, tileMats)
std::tuple<int64_t, int64_t, int64_t> calc_mhc_sinkhorn_tiling_params(int64_t totalMats, int64_t hcMult);

extern "C" {
void launch_mhc_sinkhorn(GM_ADDR x, GM_ADDR y, int64_t totalMats, int32_t hcMult, int32_t iterStep,
                         float eps, int64_t numBlocks, int64_t matPerCore, int32_t tileMats,
                         void* stream);
}

#endif // MHC_SINKHORN_LAUNCH_H
