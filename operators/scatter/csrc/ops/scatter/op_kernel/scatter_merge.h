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
 * \file scatter_merge.h
 * \brief The scalar read-modify-write merge of the scatter kernel (included by scatter_kernel.cpp).
 *
 * This target has no vector scatter primitive, so the merge is a per-update-element scalar RMW and it
 * dominates the kernel's runtime: every case whose tiling exercises the batch loops measures the same
 * ~13 core cycles per update element, i.e. about 4.3 cycles per scalar UB access for the three
 * accesses of an overwrite (index read, value read, resident-block write).  The single lever left is
 * ILP: the compiler must assume that a store into the resident block may alias the following reads of
 * the index and update staging buffers, so a loop written "read, write, read, write" serialises at one
 * full UB latency per element.  Every batch here therefore reads its indices, values and (where legal)
 * resident-block values into locals *before* the first store of the batch.
 *
 * Legality of hoisting the resident-block read:
 *   - within one update row the destinations of distinct columns are distinct, because
 *     k1*tileL + t1 == k2*tileL + t2 with |t1-t2| < tileL forces k1 == k2 and t1 == t2; so the wide
 *     batch may always hoist the block reads;
 *   - across update rows (tileL == 1, i.e. one column per row) two rows may target the same block
 *     slot, so the block read must stay ordered behind the previous store.  The one exception is the
 *     distinct-target fast path below, which first proves the eight targets of a batch pairwise
 *     different and only then hoists the block reads; when the targets are distinct the batch is
 *     bit-exactly equivalent to the strictly ordered form, so the check costs nothing but ILP.
 *
 * Batch widths: 8 for the wide shape, then 4 (which covers integer dtypes whose 32 B row alignment
 * step is only four elements), then one-at-a-time for the leftover.
 */

#ifndef SCATTER_MERGE_H
#define SCATTER_MERGE_H

#include <cstdint>

#include "kernel_operator.h"

namespace scat {

template <int MODE, typename CT>
__aicore__ inline CT CombineCT(CT cur, CT val)
{
    if constexpr (MODE == 0) {
        return val;
    } else if constexpr (MODE == 1) {
        return static_cast<CT>(cur + val);
    } else if constexpr (MODE == 2) {
        return static_cast<CT>(cur * val);
    } else if constexpr (MODE == 3) {
        // torch amin propagates NaN
        if (cur != cur) {
            return cur;
        }
        return (cur < val) ? cur : val;
    } else {
        if (cur != cur) {
            return cur;
        }
        return (cur > val) ? cur : val;
    }
}

/* apply one update at row k of a one-column block (the row pitch is 1) */
template <int MODE, typename CT>
__aicore__ inline void ApplyK(__ubuf__ CT *blk, int64_t k, CT v, int64_t k0, int64_t kb, bool fullK)
{
    if (!fullK) {
        k -= k0;
        if (k < 0 || k >= kb) {
            return;
        }
    }
    if constexpr (MODE == 0) {
        blk[k] = v;
    } else {
        blk[k] = CombineCT<MODE, CT>(blk[k], v);
    }
}

/* apply one update at (row k, column t) of a wide block */
template <int MODE, typename CT>
__aicore__ inline void ApplyW(__ubuf__ CT *blk, int64_t k, int64_t pitch, int64_t t, CT v,
                              int64_t k0, int64_t kb, bool fullK)
{
    if (!fullK) {
        k -= k0;
        if (k < 0 || k >= kb) {
            return;
        }
    }
    __ubuf__ CT *d = blk + k * pitch + t;
    if constexpr (MODE == 0) {
        *d = v;
    } else {
        *d = CombineCT<MODE, CT>(*d, v);
    }
}

/* merge a staged update chunk [jc rows x tileL columns] into the resident block */
template <int MODE, typename CT, typename IDX>
__aicore__ inline void MergeChunk(__ubuf__ CT *blkp, __ubuf__ IDX *idxp, __ubuf__ CT *updp,
                                  int64_t jc, int64_t cl, int64_t tileL, int64_t k0, int64_t kb,
                                  bool fullK)
{
    if (cl == 1) {
        // One element per update row, so the parallelism has to come from the row dimension.
        if constexpr (MODE != 0) {
            if (fullK) {
                // Fast path: read the eight resident-block values up front, but only after proving
                // the eight targets pairwise different.  A duplicate is rare, and when the check
                // succeeds the hoisted form is bit-exactly the ordered form.
                int64_t j = 0;
                for (; j + 8 <= jc; j += 8) {
                    const int64_t a0 = static_cast<int64_t>(idxp[j + 0]);
                    const int64_t a1 = static_cast<int64_t>(idxp[j + 1]);
                    const int64_t a2 = static_cast<int64_t>(idxp[j + 2]);
                    const int64_t a3 = static_cast<int64_t>(idxp[j + 3]);
                    const int64_t a4 = static_cast<int64_t>(idxp[j + 4]);
                    const int64_t a5 = static_cast<int64_t>(idxp[j + 5]);
                    const int64_t a6 = static_cast<int64_t>(idxp[j + 6]);
                    const int64_t a7 = static_cast<int64_t>(idxp[j + 7]);
                    const CT v0 = updp[j + 0];
                    const CT v1 = updp[j + 1];
                    const CT v2 = updp[j + 2];
                    const CT v3 = updp[j + 3];
                    const CT v4 = updp[j + 4];
                    const CT v5 = updp[j + 5];
                    const CT v6 = updp[j + 6];
                    const CT v7 = updp[j + 7];
                    const bool dup = (a0 == a1) || (a0 == a2) || (a0 == a3) || (a0 == a4) ||
                                     (a0 == a5) || (a0 == a6) || (a0 == a7) || (a1 == a2) ||
                                     (a1 == a3) || (a1 == a4) || (a1 == a5) || (a1 == a6) ||
                                     (a1 == a7) || (a2 == a3) || (a2 == a4) || (a2 == a5) ||
                                     (a2 == a6) || (a2 == a7) || (a3 == a4) || (a3 == a5) ||
                                     (a3 == a6) || (a3 == a7) || (a4 == a5) || (a4 == a6) ||
                                     (a4 == a7) || (a5 == a6) || (a5 == a7) || (a6 == a7);
                    if (!dup) {
                        const CT c0 = blkp[a0];
                        const CT c1 = blkp[a1];
                        const CT c2 = blkp[a2];
                        const CT c3 = blkp[a3];
                        const CT c4 = blkp[a4];
                        const CT c5 = blkp[a5];
                        const CT c6 = blkp[a6];
                        const CT c7 = blkp[a7];
                        blkp[a0] = CombineCT<MODE, CT>(c0, v0);
                        blkp[a1] = CombineCT<MODE, CT>(c1, v1);
                        blkp[a2] = CombineCT<MODE, CT>(c2, v2);
                        blkp[a3] = CombineCT<MODE, CT>(c3, v3);
                        blkp[a4] = CombineCT<MODE, CT>(c4, v4);
                        blkp[a5] = CombineCT<MODE, CT>(c5, v5);
                        blkp[a6] = CombineCT<MODE, CT>(c6, v6);
                        blkp[a7] = CombineCT<MODE, CT>(c7, v7);
                    } else {
                        ApplyK<MODE, CT>(blkp, a0, v0, k0, kb, fullK);
                        ApplyK<MODE, CT>(blkp, a1, v1, k0, kb, fullK);
                        ApplyK<MODE, CT>(blkp, a2, v2, k0, kb, fullK);
                        ApplyK<MODE, CT>(blkp, a3, v3, k0, kb, fullK);
                        ApplyK<MODE, CT>(blkp, a4, v4, k0, kb, fullK);
                        ApplyK<MODE, CT>(blkp, a5, v5, k0, kb, fullK);
                        ApplyK<MODE, CT>(blkp, a6, v6, k0, kb, fullK);
                        ApplyK<MODE, CT>(blkp, a7, v7, k0, kb, fullK);
                    }
                }
                for (; j < jc; ++j) {
                    ApplyK<MODE, CT>(blkp, static_cast<int64_t>(idxp[j]), updp[j], k0, kb, fullK);
                }
                return;
            }
        }
        int64_t j = 0;
        for (; j + 8 <= jc; j += 8) {
            const int64_t a0 = static_cast<int64_t>(idxp[j + 0]);
            const int64_t a1 = static_cast<int64_t>(idxp[j + 1]);
            const int64_t a2 = static_cast<int64_t>(idxp[j + 2]);
            const int64_t a3 = static_cast<int64_t>(idxp[j + 3]);
            const int64_t a4 = static_cast<int64_t>(idxp[j + 4]);
            const int64_t a5 = static_cast<int64_t>(idxp[j + 5]);
            const int64_t a6 = static_cast<int64_t>(idxp[j + 6]);
            const int64_t a7 = static_cast<int64_t>(idxp[j + 7]);
            const CT v0 = updp[j + 0];
            const CT v1 = updp[j + 1];
            const CT v2 = updp[j + 2];
            const CT v3 = updp[j + 3];
            const CT v4 = updp[j + 4];
            const CT v5 = updp[j + 5];
            const CT v6 = updp[j + 6];
            const CT v7 = updp[j + 7];
            ApplyK<MODE, CT>(blkp, a0, v0, k0, kb, fullK);
            ApplyK<MODE, CT>(blkp, a1, v1, k0, kb, fullK);
            ApplyK<MODE, CT>(blkp, a2, v2, k0, kb, fullK);
            ApplyK<MODE, CT>(blkp, a3, v3, k0, kb, fullK);
            ApplyK<MODE, CT>(blkp, a4, v4, k0, kb, fullK);
            ApplyK<MODE, CT>(blkp, a5, v5, k0, kb, fullK);
            ApplyK<MODE, CT>(blkp, a6, v6, k0, kb, fullK);
            ApplyK<MODE, CT>(blkp, a7, v7, k0, kb, fullK);
        }
        for (; j < jc; ++j) {
            ApplyK<MODE, CT>(blkp, static_cast<int64_t>(idxp[j]), updp[j], k0, kb, fullK);
        }
        return;
    }

    for (int64_t j = 0; j < jc; ++j) {
        __ubuf__ IDX *ir = idxp + j * tileL;
        __ubuf__ CT *ur = updp + j * tileL;
        int64_t t = 0;
        for (; t + 8 <= cl; t += 8) {
            const int64_t a0 = static_cast<int64_t>(ir[t + 0]);
            const int64_t a1 = static_cast<int64_t>(ir[t + 1]);
            const int64_t a2 = static_cast<int64_t>(ir[t + 2]);
            const int64_t a3 = static_cast<int64_t>(ir[t + 3]);
            const int64_t a4 = static_cast<int64_t>(ir[t + 4]);
            const int64_t a5 = static_cast<int64_t>(ir[t + 5]);
            const int64_t a6 = static_cast<int64_t>(ir[t + 6]);
            const int64_t a7 = static_cast<int64_t>(ir[t + 7]);
            const CT v0 = ur[t + 0];
            const CT v1 = ur[t + 1];
            const CT v2 = ur[t + 2];
            const CT v3 = ur[t + 3];
            const CT v4 = ur[t + 4];
            const CT v5 = ur[t + 5];
            const CT v6 = ur[t + 6];
            const CT v7 = ur[t + 7];
            if (fullK) {
                __ubuf__ CT *d0 = blkp + a0 * tileL + (t + 0);
                __ubuf__ CT *d1 = blkp + a1 * tileL + (t + 1);
                __ubuf__ CT *d2 = blkp + a2 * tileL + (t + 2);
                __ubuf__ CT *d3 = blkp + a3 * tileL + (t + 3);
                __ubuf__ CT *d4 = blkp + a4 * tileL + (t + 4);
                __ubuf__ CT *d5 = blkp + a5 * tileL + (t + 5);
                __ubuf__ CT *d6 = blkp + a6 * tileL + (t + 6);
                __ubuf__ CT *d7 = blkp + a7 * tileL + (t + 7);
                if constexpr (MODE == 0) {
                    *d0 = v0;
                    *d1 = v1;
                    *d2 = v2;
                    *d3 = v3;
                    *d4 = v4;
                    *d5 = v5;
                    *d6 = v6;
                    *d7 = v7;
                } else {
                    const CT c0 = *d0;
                    const CT c1 = *d1;
                    const CT c2 = *d2;
                    const CT c3 = *d3;
                    const CT c4 = *d4;
                    const CT c5 = *d5;
                    const CT c6 = *d6;
                    const CT c7 = *d7;
                    *d0 = CombineCT<MODE, CT>(c0, v0);
                    *d1 = CombineCT<MODE, CT>(c1, v1);
                    *d2 = CombineCT<MODE, CT>(c2, v2);
                    *d3 = CombineCT<MODE, CT>(c3, v3);
                    *d4 = CombineCT<MODE, CT>(c4, v4);
                    *d5 = CombineCT<MODE, CT>(c5, v5);
                    *d6 = CombineCT<MODE, CT>(c6, v6);
                    *d7 = CombineCT<MODE, CT>(c7, v7);
                }
            } else {
                ApplyW<MODE, CT>(blkp, a0, tileL, t + 0, v0, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a1, tileL, t + 1, v1, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a2, tileL, t + 2, v2, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a3, tileL, t + 3, v3, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a4, tileL, t + 4, v4, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a5, tileL, t + 5, v5, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a6, tileL, t + 6, v6, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a7, tileL, t + 7, v7, k0, kb, fullK);
            }
        }
        for (; t + 4 <= cl; t += 4) {
            const int64_t a0 = static_cast<int64_t>(ir[t + 0]);
            const int64_t a1 = static_cast<int64_t>(ir[t + 1]);
            const int64_t a2 = static_cast<int64_t>(ir[t + 2]);
            const int64_t a3 = static_cast<int64_t>(ir[t + 3]);
            const CT v0 = ur[t + 0];
            const CT v1 = ur[t + 1];
            const CT v2 = ur[t + 2];
            const CT v3 = ur[t + 3];
            if (fullK) {
                __ubuf__ CT *d0 = blkp + a0 * tileL + (t + 0);
                __ubuf__ CT *d1 = blkp + a1 * tileL + (t + 1);
                __ubuf__ CT *d2 = blkp + a2 * tileL + (t + 2);
                __ubuf__ CT *d3 = blkp + a3 * tileL + (t + 3);
                if constexpr (MODE == 0) {
                    *d0 = v0;
                    *d1 = v1;
                    *d2 = v2;
                    *d3 = v3;
                } else {
                    const CT c0 = *d0;
                    const CT c1 = *d1;
                    const CT c2 = *d2;
                    const CT c3 = *d3;
                    *d0 = CombineCT<MODE, CT>(c0, v0);
                    *d1 = CombineCT<MODE, CT>(c1, v1);
                    *d2 = CombineCT<MODE, CT>(c2, v2);
                    *d3 = CombineCT<MODE, CT>(c3, v3);
                }
            } else {
                ApplyW<MODE, CT>(blkp, a0, tileL, t + 0, v0, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a1, tileL, t + 1, v1, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a2, tileL, t + 2, v2, k0, kb, fullK);
                ApplyW<MODE, CT>(blkp, a3, tileL, t + 3, v3, k0, kb, fullK);
            }
        }
        for (; t < cl; ++t) {
            ApplyW<MODE, CT>(blkp, static_cast<int64_t>(ir[t]), tileL, t, ur[t], k0, kb, fullK);
        }
    }
}

} // namespace scat

#endif // SCATTER_MERGE_H
