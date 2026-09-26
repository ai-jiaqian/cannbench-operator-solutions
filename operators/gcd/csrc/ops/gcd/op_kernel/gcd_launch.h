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
 * \file gcd_launch.h
 * \brief Launch / tiling declarations for g++ (Gcd: y = gcd(x1, x2) with numpy broadcasting)
 *
 * The broadcast output is decomposed as
 *
 *      [ kOuter outer dims enumerated by an odometer ]  x  [ innerLen elements ]
 *
 * where the trailing run is the maximal suffix of dimensions over which each input
 * behaves uniformly as either "affine" (source offset advances one element per output
 * element) or "constant" (one single source element reused).  The plan is computed on
 * the host and handed to the launch wrappers, which expand it into flat scalar kernel
 * arguments, so no host pointer is ever dereferenced on the device.
 *
 * When the dimension just outside that run has x1 at the run stride and x2 at a stride
 * that is neither zero nor the run stride, the run is merged with it and x2 becomes
 * "repeat": the innerLen of the plan is then P periods of repR elements where x1 stays
 * contiguous and x2 is one single value per period, advancing by repStep elements per
 * period.  That is what turns a broadcast such as [.., 687, 37] x [.., 687, 1] from
 * 687 short rows into one long contiguous run.
 */

#ifndef GCD_LAUNCH_H
#define GCD_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

constexpr int32_t GCD_MAX_RANK = 8;

// Offset class of one operand inside the trailing run.
constexpr int32_t GCD_CLS_AFFINE = 1;
constexpr int32_t GCD_CLS_CONST = 2;
constexpr int32_t GCD_CLS_REPEAT = 3;

struct GcdPlan {
    int32_t innerLen;
    int32_t rowsTotal;
    int32_t kOuter;
    int32_t cls1;
    int32_t cls2;
    int32_t repR;    // period of a merged "repeat" run (0 when cls2 != REPEAT)
    int32_t repStep; // x2 elements advanced per period
    int32_t oShape[GCD_MAX_RANK];
    int32_t oS1[GCD_MAX_RANK];
    int32_t oS2[GCD_MAX_RANK];
};

// Returns (numBlocks, blockLength, tileElems).  tileElems is a power of two.
std::tuple<int64_t, int64_t, int64_t> calc_gcd_tiling(int64_t total, int64_t elemSize);

extern "C" {

void launch_gcd_kernel_int16(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdPlan* plan,
                             int64_t total, int64_t numBlocks, int64_t blockLength,
                             int64_t tileElems, void* stream);
void launch_gcd_kernel_int32(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdPlan* plan,
                             int64_t total, int64_t numBlocks, int64_t blockLength,
                             int64_t tileElems, void* stream);
void launch_gcd_kernel_int64(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GcdPlan* plan,
                             int64_t total, int64_t numBlocks, int64_t blockLength,
                             int64_t tileElems, void* stream);
}

#endif // GCD_LAUNCH_H
