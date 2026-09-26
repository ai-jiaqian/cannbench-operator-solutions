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
 * \file wqmm_qmm.cpp
 * \brief NOT COMPILED / NOT PART OF THE SUBMITTED OPERATOR -- investigation record only.
 *
 * The obvious way to make this operator cheap is to let the Matmul itself dequantise the int8 weight while
 * it copies B from GM to L1 (`SetAntiQuantVector` + the MDL `hasAntiQuantOffset` / `isPerTensor` config
 * parameters).  That would read the 1 byte/element int8 weight and nothing else, i.e. the minimum possible
 * traffic, instead of the 5 bytes/element the materialised-workspace design needs
 * (read int8 + write T + read T).
 *
 * It was implemented and does NOT build on this target: the CANN 9.0.0 DAV_2201 Matmul client
 * (`AscendC::MatmulClient<...>`) has no member named `SetAntiQuantVector`, and the API card carries
 * `platforms: [310p]`.  The A16W8 anti-quant scenario is therefore 310P only and cannot be used here.
 *
 * This file is deliberately NOT listed in CMakeLists.txt; the real implementation lives in
 * wqmm_dequant.cpp (AIV dequantisation into a T workspace) and wqmm_mm.cpp (pure-cube Matmul consuming it).
 */
