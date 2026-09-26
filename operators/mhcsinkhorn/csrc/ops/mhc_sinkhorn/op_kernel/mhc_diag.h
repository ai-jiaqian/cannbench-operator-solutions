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
 * \file mhc_diag.h
 * \brief Diagnostic build switches for the MhcSinkhorn kernel (temporary).
 */

#ifndef MHC_DIAG_H
#define MHC_DIAG_H

/* 1 -> kernel returns immediately (tests plugin/tiling/launch plumbing only) */
#define MHC_DIAG_NOOP 0

/* 1 -> do the DMA in/out but skip every vector op (tests the DMA geometry) */
#define MHC_DIAG_DMA_ONLY 0

/* 1 -> never launch the kernel; the plugin returns the input untouched. */
#define MHC_DIAG_SKIP_LAUNCH 0

/* how the reference `comb / row_sum + eps` of iteration 0 is emulated:
   0 = not at all (padding lanes stay exactly 0)  [case 19 then misses MARE]
   1 = Adds(X, X, eps) over the whole tile (padding lanes get the eps and are
       then amplified by their own column normalisation -> row sums polluted)
   2 = Adds(X, X, eps) followed by a masked Duplicate that clears the padding
       lanes again                                              */
#define MHC_DIAG_EPS0 2

#endif /* MHC_DIAG_H */
