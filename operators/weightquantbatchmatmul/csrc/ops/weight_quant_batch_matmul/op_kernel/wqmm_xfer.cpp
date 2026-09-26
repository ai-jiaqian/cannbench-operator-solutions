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
 * \file wqmm_xfer.cpp
 * \brief NOT COMPILED / NOT PART OF THE SUBMITTED OPERATOR -- investigation record only.
 *
 * These two AIV helpers belonged to the abandoned A16W8 anti-quant pipeline (see wqmm_qmm.cpp):
 * a bfloat16 -> half conversion of x (the anti-quant scenario needs a half A matrix and DAV_2201 has no
 * bf16<->half Cast) and a y = T(y32 + bias) epilogue for the fp32 accumulator that pipeline produces.
 *
 * The submitted design needs neither: the materialised-workspace Matmul runs with A/B/C in the output
 * dtype, so the fixpipe writes y directly and the bias is folded in by the Matmul's own bias path.
 *
 * This file is deliberately NOT listed in CMakeLists.txt.
 */
