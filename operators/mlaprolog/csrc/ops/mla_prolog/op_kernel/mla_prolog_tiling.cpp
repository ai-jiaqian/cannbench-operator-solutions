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
 * \file mla_prolog_tiling.cpp
 * \brief MlaProlog host-side Matmul tiling (bisheng host pass).
 *
 * The Matmul tiling headers belong to the kernel-facing include tree, so the tiling is produced here (a
 * bisheng TU) and handed to the g++ plugin as a raw serialized kernel-side TCubeTiling blob.  Both sides use
 * AscendC::tiling::TCubeTiling, so a byte copy is layout safe.
 */

#include <cstdint>
#include <cstring>
#include <cstdio>

#include "platform/platform_ascendc.h"

#if defined(__has_include)
#  if __has_include("lib/matmul/matmul_tiling.h")
#    include "lib/matmul/matmul_tiling.h"
#    define MLA_TILING_HDR 1
#  elif __has_include("adv_api/matmul/matmul_tiling.h")
#    include "adv_api/matmul/matmul_tiling.h"
#    define MLA_TILING_HDR 1
#  elif __has_include("matmul_tiling.h")
#    include "matmul_tiling.h"
#    define MLA_TILING_HDR 1
#  elif __has_include("tiling/matrix/matmul_tiling.h")
#    include "tiling/matrix/matmul_tiling.h"
#    define MLA_TILING_HDR 1
#  elif __has_include("lib/matmul/bmm_tiling.h")
#    include "lib/matmul/bmm_tiling.h"
#    define MLA_TILING_HDR 1
#  elif __has_include("bmm_tiling.h")
#    include "bmm_tiling.h"
#    define MLA_TILING_HDR 1
#  endif
#endif

#if !defined(MLA_TILING_HDR)
#error "mla_prolog: no Matmul tiling header found in the kernel include path"
#endif

#include "mla_prolog_launch.h"

extern "C" int64_t mla_aic_num()
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return 1;
    }
    int64_t n = (int64_t)plat->GetCoreNumAic();
    return n > 0 ? n : 1;
}

extern "C" int64_t mla_aiv_num()
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return 1;
    }
    int64_t n = (int64_t)plat->GetCoreNumAiv();
    return n > 0 ? n : 1;
}

extern "C" int64_t mla_sys_workspace_bytes()
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return 1 << 20;
    }
    uint32_t sz = plat->GetLibApiWorkSpaceSize();
    if (sz == 0xFFFFFFFFu || sz == 0) {
        return 1 << 20;
    }
    return (int64_t)sz;
}

extern "C" int32_t mla_make_tiling(int64_t singleM, int64_t singleN, int64_t singleK,
                                   int64_t orgM, int64_t orgN, int64_t orgKa, int64_t orgKb,
                                   int64_t cBf16, MlaTiling* pod)
{
    if (pod == nullptr || singleM <= 0 || singleN <= 0 || singleK <= 0 ||
        orgM <= 0 || orgN <= 0 || orgKa <= 0 || orgKb <= 0) {
        return -1;
    }
    if ((int64_t)sizeof(AscendC::tiling::TCubeTiling) > (int64_t)MLA_TILING_BYTES) {
        return -1;
    }
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat == nullptr) {
        return -1;
    }

    for (int32_t i = 0; i < MLA_TILING_BYTES; ++i) {
        pod->v[i] = 0;
    }

    matmul_tiling::MatmulApiTiling t(*plat);
    t.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16);
    t.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16);
    t.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
               cBf16 != 0 ? matmul_tiling::DataType::DT_BFLOAT16 : matmul_tiling::DataType::DT_FLOAT);
    t.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    t.SetShape((int32_t)singleM, (int32_t)singleN, (int32_t)singleK);
    t.SetOrgShape((int32_t)orgM, (int32_t)orgN, (int32_t)orgKa, (int32_t)orgKb);
    t.EnableBias(false);
    t.SetBufferSpace(-1, -1, -1);

    AscendC::tiling::TCubeTiling td;
    if (t.GetTiling(td) == -1) {
        return -1;
    }
    uint8_t* dst = pod->v;
    const uint8_t* src = (const uint8_t*)&td;
    for (int64_t i = 0; i < (int64_t)sizeof(AscendC::tiling::TCubeTiling); ++i) {
        dst[i] = src[i];
    }
    return 0;
}
