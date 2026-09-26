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
 * \file gqa_dev_tiling.h
 * \brief Device side conversion of the transported tiling POD into the AscendC TCubeTiling.
 *        Included by the cube kernel translation units only - the plugin must not see AscendC types.
 */

#ifndef GQA_DEV_TILING_H
#define GQA_DEV_TILING_H

#include "kernel_operator.h"
#include "gqa_launch.h"

/* Assignment by member name: immune to any layout difference between the host optiling::TCubeTiling
 * produced by the Tiling API and the device side struct consumed by the Matmul API. */
__aicore__ inline void GqaDevFill(const GqaCubeTiling &s, AscendC::tiling::TCubeTiling &d)
{
    d.usedCoreNum = s.usedCoreNum;
    d.M = s.M;
    d.N = s.N;
    d.Ka = s.Ka;
    d.Kb = s.Kb;
    d.singleCoreM = s.singleCoreM;
    d.singleCoreN = s.singleCoreN;
    d.singleCoreK = s.singleCoreK;
    d.baseM = s.baseM;
    d.baseN = s.baseN;
    d.baseK = s.baseK;
    d.depthA1 = s.depthA1;
    d.depthB1 = s.depthB1;
    d.stepM = s.stepM;
    d.stepN = s.stepN;
    d.isBias = s.isBias;
    d.transLength = s.transLength;
    d.iterateOrder = s.iterateOrder;
    d.shareMode = s.shareMode;
    d.shareL1Size = s.shareL1Size;
    d.shareL0CSize = s.shareL0CSize;
    d.shareUbSize = s.shareUbSize;
    d.batchM = s.batchM;
    d.batchN = s.batchN;
    d.BatchNum = s.batchNum;
    d.stepKa = s.stepKa;
    d.stepKb = s.stepKb;
}

#endif  // GQA_DEV_TILING_H
