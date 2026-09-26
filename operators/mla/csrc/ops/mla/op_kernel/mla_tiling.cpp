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
 * \file mla_tiling.cpp
 * \brief MLA host-side Matmul tiling and platform queries (bisheng host pass).
 *
 * The Matmul tiling headers only exist in the toolchain's kernel-facing include tree, so the
 * tiling is produced here and handed to the g++ plugin as a self-describing MlaTiling POD
 * through the launch ABI.  The kernel rebuilds its own AscendC::tiling::TCubeTiling from that
 * POD field by field.
 *
 * Every tiling describes exactly ONE complete [M, N, K] matmul computed by ONE block
 * (single-core tiling); parallelism comes from launching several blocks, each with its own
 * operand pointers.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#if defined(__has_include)
#  if __has_include("lib/matmul/matmul_tiling.h")
#    include "lib/matmul/matmul_tiling.h"
#    define MLA_TILING_HDR_OK 1
#  elif __has_include("matrix/matmul_tiling.h")
#    include "matrix/matmul_tiling.h"
#    define MLA_TILING_HDR_OK 1
#  elif __has_include("matmul_tiling.h")
#    include "matmul_tiling.h"
#    define MLA_TILING_HDR_OK 1
#  elif __has_include("adv_api/matmul/matmul_tiling.h")
#    include "adv_api/matmul/matmul_tiling.h"
#    define MLA_TILING_HDR_OK 1
#  endif
#endif

#ifndef MLA_TILING_HDR_OK
#error "mla: no matmul tiling header found in the kernel include tree"
#endif

#include "mla_launch.h"

extern "C" int64_t mla_sys_ws_bytes(void)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return 16 * 1024 * 1024;
    }
    int64_t n = (int64_t)plat->GetLibApiWorkSpaceSize();
    if (n <= 0) {
        n = 16 * 1024 * 1024;
    }
    return n;
}

extern "C" int64_t mla_aic_num(void)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return 0;
    }
    int64_t n = (int64_t)plat->GetCoreNumAic();
    return (n > 0) ? n : 0;
}

extern "C" int64_t mla_aiv_num(void)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return 0;
    }
    int64_t n = (int64_t)plat->GetCoreNumAiv();
    return (n > 0) ? n : 0;
}

extern "C" int64_t mla_tiling_mm(int64_t dt, int64_t M, int64_t N, int64_t K, int64_t bTrans,
                                 int64_t cF32, MlaTiling* pod)
{
    if (pod == nullptr || M <= 0 || N <= 0 || K <= 0) {
        return -1;
    }
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return -1;
    }

    const auto abDt = (dt != 0) ? matmul_tiling::DataType::DT_BFLOAT16
                                : matmul_tiling::DataType::DT_FLOAT16;
    const auto cDt = (cF32 != 0) ? matmul_tiling::DataType::DT_FLOAT : abDt;

    matmul_tiling::MatmulApiTiling t(*plat);
    t.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, abDt, false);
    t.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, abDt, bTrans != 0);
    t.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, cDt);
    t.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_FLOAT);
    t.SetShape(M, N, K);
    t.SetOrgShape(M, N, K);
    t.EnableBias(false);
    t.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling td;
    if (t.GetTiling(td) == -1) {
        return -1;
    }

    pod->usedCoreNum = (int32_t)td.get_usedCoreNum();
    pod->M = (int32_t)td.get_M();
    pod->N = (int32_t)td.get_N();
    pod->Ka = (int32_t)td.get_Ka();
    pod->Kb = (int32_t)td.get_Kb();
    pod->singleCoreM = (int32_t)td.get_singleCoreM();
    pod->singleCoreN = (int32_t)td.get_singleCoreN();
    pod->singleCoreK = (int32_t)td.get_singleCoreK();
    pod->baseM = (int32_t)td.get_baseM();
    pod->baseN = (int32_t)td.get_baseN();
    pod->baseK = (int32_t)td.get_baseK();
    pod->depthA1 = (int32_t)td.get_depthA1();
    pod->depthB1 = (int32_t)td.get_depthB1();
    pod->stepM = (int32_t)td.get_stepM();
    pod->stepN = (int32_t)td.get_stepN();
    pod->stepKa = (int32_t)td.get_stepKa();
    pod->stepKb = (int32_t)td.get_stepKb();
    pod->isBias = (int32_t)td.get_isBias();
    pod->transLength = (int32_t)td.get_transLength();
    pod->iterateOrder = (int32_t)td.get_iterateOrder();
    pod->dbL0A = (int32_t)td.get_dbL0A();
    pod->dbL0B = (int32_t)td.get_dbL0B();
    pod->dbL0C = (int32_t)td.get_dbL0C();
    pod->shareMode = (int32_t)td.get_shareMode();
    pod->shareL1Size = (int32_t)td.get_shareL1Size();
    pod->shareL0CSize = (int32_t)td.get_shareL0CSize();
    pod->shareUbSize = (int32_t)td.get_shareUbSize();
    pod->batchM = (int32_t)td.get_batchM();
    pod->batchN = (int32_t)td.get_batchN();
    pod->singleBatchM = (int32_t)td.get_singleBatchM();
    pod->singleBatchN = (int32_t)td.get_singleBatchN();
    pod->mxTypePara = (int32_t)td.get_mxTypePara();
    return 0;
}
