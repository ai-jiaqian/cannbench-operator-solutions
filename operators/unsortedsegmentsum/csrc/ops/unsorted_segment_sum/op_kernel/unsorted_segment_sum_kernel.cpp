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
 * \file unsorted_segment_sum_kernel.cpp
 * \brief UnsortedSegmentSum device kernels + host tiling (bisheng, -xasc, dav-2201).
 *
 *   y[i] = sum_{j : segment_ids[j] == i} data[j]
 *
 * `data` is viewed as a flat matrix [N, inner] and the output as [numSeg, inner]; empty segments
 * produce zeros.  Accumulation happens on device in a wider / supported type:
 *     float16 / bfloat16 / float32 -> float accumulator
 *     int32   / int64              -> int32 accumulator (restored to int64 with an exact cast;
 *                                     exact for every admitted value range: |sum| < 2^31)
 *
 * Two deterministic (non atomic) execution modes exist; the host tiling chooses from the shape alone:
 *   mode 0 owner     - see uss_launch.h
 *   mode 1 partition - see uss_launch.h
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "uss_launch.h"

using namespace AscendC;

namespace {

template <typename T>
struct UssAccOf {
    using type = float;
};
template <>
struct UssAccOf<int32_t> {
    using type = int32_t;
};
template <>
struct UssAccOf<int64_t> {
    using type = int32_t;
};

__aicore__ inline int64_t UssUp32(int64_t b)
{
    if (b <= 0) {
        return 32;
    }
    return ((b + 31) / 32) * 32;
}

// accumulator -> destination dtype rounding mode (matches the reference round-to-nearest-even)
template <typename T, typename A>
__aicore__ inline RoundMode UssBackRound()
{
    if constexpr (std::is_same_v<A, float> &&
                  (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>)) {
        return RoundMode::CAST_RINT;
    } else {
        return RoundMode::CAST_NONE;
    }
}

// scalar read of one segment id, normalised to int32 (ids are < numSeg <= 32768)
template <typename IdT>
__aicore__ inline int32_t UssIdAt(const LocalTensor<IdT> &t, int32_t j)
{
    if constexpr (std::is_same_v<IdT, int32_t>) {
        return t.GetValue(j);
    } else {
        return (int32_t)t.GetValue(j);
    }
}

// append one in-window row to its segment bucket (capacity limited)
__aicore__ inline void UssPut(LocalTensor<int32_t> &sortL, LocalTensor<int32_t> &curL, int32_t cap,
                              int32_t k, int32_t row)
{
    int32_t p = curL.GetValue(k);
    if (p < cap) {
        sortL.SetValue(k * cap + p, row);
    }
    curL.SetValue(k, p + 1);
}

} // namespace

// host side helper (never called from device code)
static inline int64_t UssUp32Host(int64_t b)
{
    if (b <= 0) {
        return 32;
    }
    return ((b + 31) / 32) * 32;
}

// =================================================================================================
// mode 0 : unit owner with a per-segment bucketed row index
// =================================================================================================
template <typename T, typename IdT>
__global__ __aicore__ void uss_owner_kernel(GM_ADDR x, GM_ADDR ids, GM_ADDR y, int64_t N, int64_t inner,
                                            int64_t numSeg, int64_t chunkElems, int64_t nChunks,
                                            int64_t unitsPerCore, int64_t segArrMax, int64_t idTile,
                                            int64_t cap)
{
    using A = typename UssAccOf<T>::type;
    constexpr bool SAME = std::is_same_v<T, A>;

    GlobalTensor<T> xGm;
    GlobalTensor<IdT> idGm;
    GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    idGm.SetGlobalBuffer(reinterpret_cast<__gm__ IdT *>(ids));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    const int32_t blk = (int32_t)GetBlockIdx();
    const int32_t totalUnits = (int32_t)(numSeg * nChunks);
    const int32_t u0 = blk * (int32_t)unitsPerCore;
    if (u0 >= totalUnits) {
        return;
    }
    int32_t u1 = u0 + (int32_t)unitsPerCore;
    if (u1 > totalUnits) {
        u1 = totalUnits;
    }
    const int32_t nCh = (int32_t)nChunks;
    const int32_t segA = u0 / nCh;
    const int32_t segB = (u1 - 1) / nCh;
    const int32_t winLen = segB - segA + 1;
    const int32_t cap32 = (int32_t)cap;
    const int32_t inner32 = (int32_t)inner;
    const int32_t ce32 = (int32_t)chunkElems;
    const int32_t N32 = (int32_t)N;

    TPipe pipe;
    TBuf<TPosition::VECCALC> cntBuf;
    TBuf<TPosition::VECCALC> offBuf;
    TBuf<TPosition::VECCALC> curBuf;
    TBuf<TPosition::VECCALC> sortBuf;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> castBuf;
    TQue<QuePosition::VECIN, 2> idQ;
    TQue<QuePosition::VECIN, 2> dataQ;
    TQue<QuePosition::VECOUT, 2> outQ;

    int64_t sortElems = N;
    int64_t bucketElems = (int64_t)winLen * cap32 + 2;
    if (bucketElems > sortElems) {
        sortElems = bucketElems;
    }

    pipe.InitBuffer(cntBuf, UssUp32(segArrMax * (int64_t)sizeof(int32_t)));
    pipe.InitBuffer(offBuf, UssUp32(segArrMax * (int64_t)sizeof(int32_t)));
    pipe.InitBuffer(curBuf, UssUp32(segArrMax * (int64_t)sizeof(int32_t)));
    pipe.InitBuffer(sortBuf, UssUp32(sortElems * (int64_t)sizeof(int32_t)));
    pipe.InitBuffer(idQ, 2, UssUp32(idTile * (int64_t)sizeof(IdT)));
    pipe.InitBuffer(dataQ, 2, UssUp32(chunkElems * (int64_t)sizeof(T)));
    pipe.InitBuffer(outQ, 2, UssUp32(chunkElems * (int64_t)sizeof(T)));
    if constexpr (!SAME) {
        pipe.InitBuffer(accBuf, UssUp32(chunkElems * (int64_t)sizeof(A)));
        pipe.InitBuffer(castBuf, UssUp32(chunkElems * (int64_t)sizeof(A)));
    }

    LocalTensor<int32_t> cntL = cntBuf.Get<int32_t>();
    LocalTensor<int32_t> offL = offBuf.Get<int32_t>();
    LocalTensor<int32_t> curL = curBuf.Get<int32_t>();
    LocalTensor<int32_t> sortL = sortBuf.Get<int32_t>();

    // ---- single pass : bucket the window rows with a per-segment capacity ----
    for (int32_t k = 0; k < winLen; ++k) {
        curL.SetValue(k, 0);
    }
    const uint32_t winLim = (uint32_t)(winLen - 1);
    for (int32_t off = 0; off < N32; off += (int32_t)idTile) {
        int32_t n = N32 - off;
        if (n > (int32_t)idTile) {
            n = (int32_t)idTile;
        }
        DataCopyExtParams cp{1, (uint32_t)(n * (int64_t)sizeof(IdT)), 0, 0, 0};
        DataCopyPadExtParams<IdT> pp{false, 0, 0, 0};
        LocalTensor<IdT> idl = idQ.AllocTensor<IdT>();
        DataCopyPad(idl, idGm[off], cp, pp);
        idQ.EnQue(idl);
        idl = idQ.DeQue<IdT>();
        int32_t j = 0;
        for (; j + 8 <= n; j += 8) {
            const int32_t s0 = UssIdAt(idl, j) - segA;
            const int32_t s1 = UssIdAt(idl, j + 1) - segA;
            const int32_t s2 = UssIdAt(idl, j + 2) - segA;
            const int32_t s3 = UssIdAt(idl, j + 3) - segA;
            const int32_t s4 = UssIdAt(idl, j + 4) - segA;
            const int32_t s5 = UssIdAt(idl, j + 5) - segA;
            const int32_t s6 = UssIdAt(idl, j + 6) - segA;
            const int32_t s7 = UssIdAt(idl, j + 7) - segA;
            if ((uint32_t)s0 <= winLim) {
                UssPut(sortL, curL, cap32, s0, off + j);
            }
            if ((uint32_t)s1 <= winLim) {
                UssPut(sortL, curL, cap32, s1, off + j + 1);
            }
            if ((uint32_t)s2 <= winLim) {
                UssPut(sortL, curL, cap32, s2, off + j + 2);
            }
            if ((uint32_t)s3 <= winLim) {
                UssPut(sortL, curL, cap32, s3, off + j + 3);
            }
            if ((uint32_t)s4 <= winLim) {
                UssPut(sortL, curL, cap32, s4, off + j + 4);
            }
            if ((uint32_t)s5 <= winLim) {
                UssPut(sortL, curL, cap32, s5, off + j + 5);
            }
            if ((uint32_t)s6 <= winLim) {
                UssPut(sortL, curL, cap32, s6, off + j + 6);
            }
            if ((uint32_t)s7 <= winLim) {
                UssPut(sortL, curL, cap32, s7, off + j + 7);
            }
        }
        for (; j + 4 <= n; j += 4) {
            const int32_t s0 = UssIdAt(idl, j) - segA;
            const int32_t s1 = UssIdAt(idl, j + 1) - segA;
            const int32_t s2 = UssIdAt(idl, j + 2) - segA;
            const int32_t s3 = UssIdAt(idl, j + 3) - segA;
            if ((uint32_t)s0 <= winLim) {
                UssPut(sortL, curL, cap32, s0, off + j);
            }
            if ((uint32_t)s1 <= winLim) {
                UssPut(sortL, curL, cap32, s1, off + j + 1);
            }
            if ((uint32_t)s2 <= winLim) {
                UssPut(sortL, curL, cap32, s2, off + j + 2);
            }
            if ((uint32_t)s3 <= winLim) {
                UssPut(sortL, curL, cap32, s3, off + j + 3);
            }
        }
        for (; j < n; ++j) {
            const int32_t s = UssIdAt(idl, j) - segA;
            if ((uint32_t)s <= winLim) {
                UssPut(sortL, curL, cap32, s, off + j);
            }
        }
        idQ.FreeTensor(idl);
    }

    int32_t overflow = 0;
    for (int32_t k = 0; k < winLen; ++k) {
        int32_t c = curL.GetValue(k);
        offL.SetValue(k, c);
        if (c > cap32) {
            overflow = 1;
        }
    }

    if (overflow != 0) {
        // ---- fallback : exact counting + exclusive prefix sum placement ----
        for (int32_t k = 0; k < winLen; ++k) {
            cntL.SetValue(k, 0);
        }
        for (int32_t off = 0; off < N32; off += (int32_t)idTile) {
            int32_t n = N32 - off;
            if (n > (int32_t)idTile) {
                n = (int32_t)idTile;
            }
            DataCopyExtParams cp{1, (uint32_t)(n * (int64_t)sizeof(IdT)), 0, 0, 0};
            DataCopyPadExtParams<IdT> pp{false, 0, 0, 0};
            LocalTensor<IdT> idl = idQ.AllocTensor<IdT>();
            DataCopyPad(idl, idGm[off], cp, pp);
            idQ.EnQue(idl);
            idl = idQ.DeQue<IdT>();
            for (int32_t j = 0; j < n; ++j) {
                int32_t s = UssIdAt(idl, j);
                if (s >= segA && s <= segB) {
                    int32_t k = s - segA;
                    cntL.SetValue(k, cntL.GetValue(k) + 1);
                }
            }
            idQ.FreeTensor(idl);
        }
        int32_t run = 0;
        for (int32_t k = 0; k < winLen; ++k) {
            offL.SetValue(k, run);
            curL.SetValue(k, run);
            run += cntL.GetValue(k);
        }
        for (int32_t off = 0; off < N32; off += (int32_t)idTile) {
            int32_t n = N32 - off;
            if (n > (int32_t)idTile) {
                n = (int32_t)idTile;
            }
            DataCopyExtParams cp{1, (uint32_t)(n * (int64_t)sizeof(IdT)), 0, 0, 0};
            DataCopyPadExtParams<IdT> pp{false, 0, 0, 0};
            LocalTensor<IdT> idl = idQ.AllocTensor<IdT>();
            DataCopyPad(idl, idGm[off], cp, pp);
            idQ.EnQue(idl);
            idl = idQ.DeQue<IdT>();
            for (int32_t j = 0; j < n; ++j) {
                int32_t s = UssIdAt(idl, j);
                if (s >= segA && s <= segB) {
                    int32_t k = s - segA;
                    int32_t p = curL.GetValue(k);
                    sortL.SetValue(p, off + j);
                    curL.SetValue(k, p + 1);
                }
            }
            idQ.FreeTensor(idl);
        }
    }

    LocalTensor<A> accL;
    LocalTensor<A> castL;
    if constexpr (!SAME) {
        accL = accBuf.Get<A>();
        castL = castBuf.Get<A>();
    }

    // ---- per owned (segment, chunk) unit : accumulate the segment rows and store ----
    int32_t curSeg = -1;
    int32_t rStart = 0;
    int32_t rEnd = 0;
    for (int32_t u = u0; u < u1; ++u) {
        const int32_t s = u / nCh;
        const int32_t c = u - s * nCh;
        if (s != curSeg) {
            curSeg = s;
            const int32_t k = s - segA;
            if (overflow != 0) {
                rStart = offL.GetValue(k);
                rEnd = rStart + cntL.GetValue(k);
            } else {
                rStart = k * cap32;
                rEnd = rStart + offL.GetValue(k);
            }
        }
        const int32_t base = c * ce32;
        int32_t cn = inner32 - base;
        if (cn > ce32) {
            cn = ce32;
        }
        const int32_t nrows = rEnd - rStart;

        LocalTensor<T> outT = outQ.AllocTensor<T>();
        if constexpr (SAME) {
            if (nrows == 0) {
                Duplicate(outT, (T)0, cn);
            } else {
                for (int32_t r = rStart; r < rEnd; ++r) {
                    const int32_t row = sortL.GetValue(r);
                    DataCopyExtParams cp{1, (uint32_t)(cn * (int64_t)sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
                    LocalTensor<T> d = dataQ.AllocTensor<T>();
                    DataCopyPad(d, xGm[(int64_t)row * inner + base], cp, pp);
                    dataQ.EnQue(d);
                    d = dataQ.DeQue<T>();
                    if (r == rStart) {
                        Adds(outT, d, (T)0, cn);
                    } else {
                        Add(outT, outT, d, cn);
                    }
                    dataQ.FreeTensor(d);
                }
            }
        } else {
            if (nrows == 0) {
                Duplicate(accL, (A)0, cn);
            } else {
                for (int32_t r = rStart; r < rEnd; ++r) {
                    const int32_t row = sortL.GetValue(r);
                    DataCopyExtParams cp{1, (uint32_t)(cn * (int64_t)sizeof(T)), 0, 0, 0};
                    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
                    LocalTensor<T> d = dataQ.AllocTensor<T>();
                    DataCopyPad(d, xGm[(int64_t)row * inner + base], cp, pp);
                    dataQ.EnQue(d);
                    d = dataQ.DeQue<T>();
                    if (r == rStart) {
                        Cast(accL, d, RoundMode::CAST_NONE, cn);
                    } else {
                        Cast(castL, d, RoundMode::CAST_NONE, cn);
                        Add(accL, accL, castL, cn);
                    }
                    dataQ.FreeTensor(d);
                }
            }
            Cast(outT, accL, UssBackRound<T, A>(), cn);
        }
        outQ.EnQue(outT);
        outT = outQ.DeQue<T>();
        DataCopyExtParams cpo{1, (uint32_t)(cn * (int64_t)sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[(int64_t)s * inner + base], outT, cpo);
        outQ.FreeTensor(outT);
    }
}

// =================================================================================================
// mode 1 : row partition + accumulation (inner == 1)
// =================================================================================================
template <typename T, typename IdT>
__global__ __aicore__ void uss_part_kernel(GM_ADDR x, GM_ADDR ids, GM_ADDR ws, int64_t N, int64_t numSeg,
                                           int64_t rowsPerCore, int64_t rowTile)
{
    using A = typename UssAccOf<T>::type;
    constexpr bool SAME = std::is_same_v<T, A>;
    constexpr bool FLOATACC = std::is_same_v<A, float>;

    GlobalTensor<T> xGm;
    GlobalTensor<IdT> idGm;
    GlobalTensor<A> wsGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    idGm.SetGlobalBuffer(reinterpret_cast<__gm__ IdT *>(ids));
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ A *>(ws));

    const int32_t blk = (int32_t)GetBlockIdx();
    const int32_t r0 = blk * (int32_t)rowsPerCore;
    int32_t rows = 0;
    if ((int64_t)r0 < N) {
        int64_t r1 = (int64_t)r0 + rowsPerCore;
        if (r1 > N) {
            r1 = N;
        }
        rows = (int32_t)(r1 - (int64_t)r0);
    }
    const int32_t rt = (int32_t)rowTile;
    const int32_t ns = (int32_t)numSeg;

    TPipe pipe;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> castBuf;
    TBuf<TPosition::VECCALC> vecBuf;
    TBuf<TPosition::VECCALC> workBuf;
    TQue<QuePosition::VECIN, 2> idQ;
    TQue<QuePosition::VECIN, 2> dataQ;
    TQue<QuePosition::VECOUT, 1> redQ;

    pipe.InitBuffer(accBuf, UssUp32((int64_t)ns * (int64_t)sizeof(A)));
    pipe.InitBuffer(idQ, 2, UssUp32((int64_t)rt * (int64_t)sizeof(IdT)));
    pipe.InitBuffer(dataQ, 2, UssUp32((int64_t)rt * (int64_t)sizeof(T)));
    if constexpr (!SAME) {
        pipe.InitBuffer(castBuf, UssUp32((int64_t)rt * (int64_t)sizeof(A)));
    }
    if constexpr (FLOATACC) {
        if (ns == 1) {
            pipe.InitBuffer(vecBuf, UssUp32((int64_t)rt * (int64_t)sizeof(A)));
            pipe.InitBuffer(workBuf, UssUp32((int64_t)rt * (int64_t)sizeof(A)));
            pipe.InitBuffer(redQ, 1, 32);
        }
    }

    LocalTensor<A> accL = accBuf.Get<A>();
    LocalTensor<A> castL;
    if constexpr (!SAME) {
        castL = castBuf.Get<A>();
    }

    if constexpr (FLOATACC) {
        if (ns == 1) {
            // whole-core vector reduction of a single segment
            LocalTensor<A> accV = vecBuf.Get<A>();
            LocalTensor<A> workV = workBuf.Get<A>();
            Duplicate(accV, (A)0, rt);
            for (int32_t t = 0; t < rows; t += rt) {
                int32_t n = rows - t;
                if (n > rt) {
                    n = rt;
                }
                DataCopyExtParams cpd{1, (uint32_t)((int64_t)n * (int64_t)sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> ppd{false, 0, 0, 0};
                LocalTensor<T> dl = dataQ.AllocTensor<T>();
                DataCopyPad(dl, xGm[(int64_t)r0 + t], cpd, ppd);
                dataQ.EnQue(dl);
                dl = dataQ.DeQue<T>();
                if constexpr (SAME) {
                    Add(accV, accV, dl, n);
                } else {
                    Cast(castL, dl, RoundMode::CAST_NONE, n);
                    Add(accV, accV, castL, n);
                }
                dataQ.FreeTensor(dl);
            }
            LocalTensor<A> o = redQ.AllocTensor<A>();
            ReduceSum(o, accV, workV, rt);
            redQ.EnQue(o);
            o = redQ.DeQue<A>();
            DataCopyExtParams cpo{1, (uint32_t)sizeof(A), 0, 0, 0};
            DataCopyPad(wsGm[(int64_t)blk], o, cpo);
            redQ.FreeTensor(o);
            return;
        }
    }

    Duplicate(accL, (A)0, ns);

    for (int32_t t = 0; t < rows; t += rt) {
        int32_t n = rows - t;
        if (n > rt) {
            n = rt;
        }
        DataCopyExtParams cpi{1, (uint32_t)((int64_t)n * (int64_t)sizeof(IdT)), 0, 0, 0};
        DataCopyPadExtParams<IdT> ppi{false, 0, 0, 0};
        LocalTensor<IdT> idl = idQ.AllocTensor<IdT>();
        DataCopyPad(idl, idGm[(int64_t)r0 + t], cpi, ppi);
        idQ.EnQue(idl);
        idl = idQ.DeQue<IdT>();

        DataCopyExtParams cpd{1, (uint32_t)((int64_t)n * (int64_t)sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> ppd{false, 0, 0, 0};
        LocalTensor<T> dl = dataQ.AllocTensor<T>();
        DataCopyPad(dl, xGm[(int64_t)r0 + t], cpd, ppd);
        dataQ.EnQue(dl);
        dl = dataQ.DeQue<T>();

        if constexpr (SAME) {
            int32_t j = 0;
            for (; j + 4 <= n; j += 4) {
                const int32_t s0 = UssIdAt(idl, j);
                const T v0 = dl.GetValue(j);
                const int32_t s1 = UssIdAt(idl, j + 1);
                const T v1 = dl.GetValue(j + 1);
                const int32_t s2 = UssIdAt(idl, j + 2);
                const T v2 = dl.GetValue(j + 2);
                const int32_t s3 = UssIdAt(idl, j + 3);
                const T v3 = dl.GetValue(j + 3);
                accL.SetValue(s0, accL.GetValue(s0) + v0);
                accL.SetValue(s1, accL.GetValue(s1) + v1);
                accL.SetValue(s2, accL.GetValue(s2) + v2);
                accL.SetValue(s3, accL.GetValue(s3) + v3);
            }
            for (; j < n; ++j) {
                int32_t s = UssIdAt(idl, j);
                accL.SetValue(s, accL.GetValue(s) + dl.GetValue(j));
            }
        } else {
            Cast(castL, dl, RoundMode::CAST_NONE, n);
            int32_t j = 0;
            for (; j + 4 <= n; j += 4) {
                const int32_t s0 = UssIdAt(idl, j);
                const A v0 = castL.GetValue(j);
                const int32_t s1 = UssIdAt(idl, j + 1);
                const A v1 = castL.GetValue(j + 1);
                const int32_t s2 = UssIdAt(idl, j + 2);
                const A v2 = castL.GetValue(j + 2);
                const int32_t s3 = UssIdAt(idl, j + 3);
                const A v3 = castL.GetValue(j + 3);
                accL.SetValue(s0, accL.GetValue(s0) + v0);
                accL.SetValue(s1, accL.GetValue(s1) + v1);
                accL.SetValue(s2, accL.GetValue(s2) + v2);
                accL.SetValue(s3, accL.GetValue(s3) + v3);
            }
            for (; j < n; ++j) {
                int32_t s = UssIdAt(idl, j);
                accL.SetValue(s, accL.GetValue(s) + castL.GetValue(j));
            }
        }
        dataQ.FreeTensor(dl);
        idQ.FreeTensor(idl);
    }

    DataCopyExtParams cpo{1, (uint32_t)((int64_t)ns * (int64_t)sizeof(A)), 0, 0, 0};
    DataCopyPad(wsGm[(int64_t)blk * numSeg], accL, cpo);
}

// =================================================================================================
// mode 1 fold kernel
// =================================================================================================
template <typename A, typename T>
__global__ __aicore__ void uss_reduce_kernel(GM_ADDR ws, GM_ADDR y, int64_t numSeg, int64_t parts)
{
    constexpr bool SAME = std::is_same_v<A, T>;

    GlobalTensor<A> wsGm;
    GlobalTensor<T> yGm;
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ A *>(ws));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    const int64_t blk = GetBlockIdx();
    const int64_t nblk = GetBlockNum();
    int64_t per = (numSeg + nblk - 1) / nblk;
    if (per < 1) {
        per = 1;
    }
    const int64_t s0 = blk * per;
    if (s0 >= numSeg) {
        return;
    }
    int64_t s1 = s0 + per;
    if (s1 > numSeg) {
        s1 = numSeg;
    }
    const int64_t n = s1 - s0;
    const int32_t n32 = (int32_t)n;

    TPipe pipe;
    TBuf<TPosition::VECCALC> accBuf;
    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
    pipe.InitBuffer(inQ, 2, UssUp32(n * (int64_t)sizeof(A)));
    pipe.InitBuffer(outQ, 2, UssUp32(n * (int64_t)sizeof(T)));
    if constexpr (!SAME) {
        pipe.InitBuffer(accBuf, UssUp32(n * (int64_t)sizeof(A)));
    }

    if constexpr (SAME) {
        LocalTensor<T> outT = outQ.AllocTensor<T>();
        bool first = true;
        for (int64_t b = 0; b < parts; ++b) {
            DataCopyExtParams cp{1, (uint32_t)(n * (int64_t)sizeof(A)), 0, 0, 0};
            DataCopyPadExtParams<A> pp{false, 0, 0, 0};
            LocalTensor<A> d = inQ.AllocTensor<A>();
            DataCopyPad(d, wsGm[b * numSeg + s0], cp, pp);
            inQ.EnQue(d);
            d = inQ.DeQue<A>();
            if (first) {
                Adds(outT, d, (T)0, n32);
                first = false;
            } else {
                Add(outT, outT, d, n32);
            }
            inQ.FreeTensor(d);
        }
        if (first) {
            Duplicate(outT, (T)0, n32);
        }
        outQ.EnQue(outT);
        outT = outQ.DeQue<T>();
        DataCopyExtParams cpo{1, (uint32_t)(n * (int64_t)sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[s0], outT, cpo);
        outQ.FreeTensor(outT);
    } else {
        LocalTensor<A> accL = accBuf.Get<A>();
        Duplicate(accL, (A)0, n32);
        for (int64_t b = 0; b < parts; ++b) {
            DataCopyExtParams cp{1, (uint32_t)(n * (int64_t)sizeof(A)), 0, 0, 0};
            DataCopyPadExtParams<A> pp{false, 0, 0, 0};
            LocalTensor<A> d = inQ.AllocTensor<A>();
            DataCopyPad(d, wsGm[b * numSeg + s0], cp, pp);
            inQ.EnQue(d);
            d = inQ.DeQue<A>();
            Add(accL, accL, d, n32);
            inQ.FreeTensor(d);
        }
        LocalTensor<T> outT = outQ.AllocTensor<T>();
        Cast(outT, accL, UssBackRound<T, A>(), n32);
        outQ.EnQue(outT);
        outT = outQ.DeQue<T>();
        DataCopyExtParams cpo{1, (uint32_t)(n * (int64_t)sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[s0], outT, cpo);
        outQ.FreeTensor(outT);
    }
}

// =================================================================================================
// host tiling
// =================================================================================================
UssTiling calc_uss_tiling(int64_t N, int64_t inner, int64_t numSeg, int64_t szT, int64_t szId, int64_t szA)
{
    UssTiling t;
    t.mode = 0;
    t.N = N;
    t.inner = inner;
    t.numSeg = numSeg;
    t.numBlocks = 1;
    t.chunkElems = 1;
    t.nChunks = 1;
    t.unitsPerCore = 1;
    t.segArrMax = 1;
    t.idTile = 1;
    t.cap = 1;
    t.rowsPerCore = 1;
    t.rowTile = 1;
    t.wsElems = 0;
    t.redBlocks = 1;

    int64_t coreNum = 1;
    int64_t ubSize = 192 * 1024;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        int64_t cn = plat->GetCoreNumAiv();
        if (cn > 0) {
            coreNum = cn;
        }
        uint64_t ub = 0;
        plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        if (ub > 0) {
            ubSize = (int64_t)ub;
        }
    }

    const int64_t OWNER_MAX_N = 8192;
    if (inner == 1 && N > OWNER_MAX_N) {
        t.mode = 1;
        int64_t nb = coreNum;
        if (nb > N) {
            nb = N;
        }
        if (nb < 1) {
            nb = 1;
        }
        t.numBlocks = nb;
        t.rowsPerCore = (N + nb - 1) / nb;
        int64_t rt = (96 * 1024) / (2 * szId + 2 * szT + 1);
        if (rt > 4096) {
            rt = 4096;
        }
        if (rt < 256) {
            rt = 256;
        }
        const int64_t accBytes = numSeg * szA;
        for (int it = 0; it < 32; ++it) {
            int64_t per = 2 * szId + 2 * szT + (szT != szA ? szA : 0);
            if (accBytes + rt * per <= ubSize - 8192 || rt <= 256) {
                break;
            }
            int64_t nrt = rt / 2;
            if (nrt < 256) {
                nrt = 256;
            }
            if (nrt >= rt) {
                break;
            }
            rt = nrt;
        }
        t.rowTile = rt;
        t.wsElems = nb * numSeg;
        t.redBlocks = coreNum < numSeg ? coreNum : numSeg;
        if (t.redBlocks < 1) {
            t.redBlocks = 1;
        }
        return t;
    }

    t.mode = 0;
    int64_t idTile = N < 2048 ? N : 2048;
    if (idTile < 1) {
        idTile = 1;
    }
    t.idTile = idTile;

    const int64_t perElem = 4 * szT + (szT != szA ? 2 * szA : 0);
    const int64_t idBytes = 2 * UssUp32Host(idTile * szId);
    int64_t ce = inner < 8192 ? inner : 8192;
    if (ce < 1) {
        ce = 1;
    }
    int64_t meanRows = (N + numSeg - 1) / numSeg;
    if (meanRows < 1) {
        meanRows = 1;
    }
    for (int it = 0; it < 40; ++it) {
        int64_t nCh = (inner + ce - 1) / ce;
        if (nCh < 1) {
            nCh = 1;
        }
        int64_t tu = numSeg * nCh;
        int64_t upc = (tu + coreNum - 1) / coreNum;
        if (upc < 1) {
            upc = 1;
        }
        int64_t winLenMax = (upc + nCh - 1) / nCh + 1;
        int64_t segArrMax = winLenMax + 1;
        int64_t cap = 4 * meanRows + 16;
        const int64_t maxBucket = 12288;
        if (winLenMax * cap > maxBucket) {
            cap = maxBucket / (winLenMax > 0 ? winLenMax : 1);
        }
        if (cap < 8) {
            cap = 8;
        }
        int64_t bucketElems = winLenMax * cap + 2;
        int64_t sortElems = N > bucketElems ? N : bucketElems;
        int64_t sortBytes = UssUp32Host(sortElems * 4);
        int64_t segBytes = 3 * UssUp32Host(segArrMax * 4);
        int64_t tileBytes = ce * perElem;
        if (sortBytes + segBytes + idBytes + tileBytes <= ubSize - 8192) {
            t.chunkElems = ce;
            t.nChunks = nCh;
            t.unitsPerCore = upc;
            t.segArrMax = segArrMax;
            t.cap = cap;
            t.numBlocks = coreNum;
            return t;
        }
        int64_t nce = ce / 2;
        if (nce > 64) {
            nce = (nce / 64) * 64;
        }
        if (nce < 64) {
            nce = 64;
        }
        if (nce >= ce) {
            break;
        }
        ce = nce;
    }
    // last resort : minimal tiling (never UB-overflows)
    {
        int64_t nCh = inner;
        int64_t tu = numSeg * nCh;
        int64_t upc = (tu + coreNum - 1) / coreNum;
        if (upc < 1) {
            upc = 1;
        }
        int64_t winLenMax = (upc + nCh - 1) / nCh + 1;
        t.chunkElems = 1;
        t.nChunks = nCh;
        t.unitsPerCore = upc;
        t.segArrMax = winLenMax + 1;
        t.cap = 8;
        t.numBlocks = coreNum;
    }
    return t;
}

// =================================================================================================
// launch wrappers (callable from g++)
// =================================================================================================
extern "C" {

#define USS_OWNER_DEF(TAG, TD, IDD)                                                                          \
    void launch_uss_owner_##TAG(GM_ADDR x, GM_ADDR ids, GM_ADDR y, int64_t N, int64_t inner, int64_t numSeg,  \
                                int64_t chunkElems, int64_t nChunks, int64_t unitsPerCore, int64_t segArrMax, \
                                int64_t idTile, int64_t cap, void *stream)                                     \
    {                                                                                                        \
        int64_t totalUnits = numSeg * nChunks;                                                                \
        int64_t nblk = (totalUnits + unitsPerCore - 1) / (unitsPerCore > 0 ? unitsPerCore : 1);               \
        if (nblk < 1) {                                                                                       \
            nblk = 1;                                                                                         \
        }                                                                                                     \
        uss_owner_kernel<TD, IDD><<<nblk, nullptr, stream>>>(x, ids, y, N, inner, numSeg, chunkElems,          \
                                                             nChunks, unitsPerCore, segArrMax, idTile, cap);  \
    }

#define USS_PART_DEF(TAG, TD, IDD)                                                                        \
    void launch_uss_part_##TAG(GM_ADDR x, GM_ADDR ids, GM_ADDR ws, int64_t N, int64_t numSeg,              \
                               int64_t numBlocks, int64_t rowsPerCore, int64_t rowTile, void *stream)      \
    {                                                                                                     \
        uss_part_kernel<TD, IDD><<<numBlocks, nullptr, stream>>>(x, ids, ws, N, numSeg, rowsPerCore, rowTile); \
    }

#define USS_RED_DEF(TAG, AD, TD)                                                                            \
    void launch_uss_red_##TAG(GM_ADDR ws, GM_ADDR y, int64_t numSeg, int64_t parts, int64_t redBlocks,       \
                              void *stream)                                                                  \
    {                                                                                                        \
        uss_reduce_kernel<AD, TD><<<redBlocks, nullptr, stream>>>(ws, y, numSeg, parts);                      \
    }

USS_OWNER_DEF(f16_i32, half, int32_t)
USS_OWNER_DEF(f16_i64, half, int64_t)
USS_OWNER_DEF(bf16_i32, bfloat16_t, int32_t)
USS_OWNER_DEF(bf16_i64, bfloat16_t, int64_t)
USS_OWNER_DEF(f32_i32, float, int32_t)
USS_OWNER_DEF(f32_i64, float, int64_t)
USS_OWNER_DEF(i32_i32, int32_t, int32_t)
USS_OWNER_DEF(i32_i64, int32_t, int64_t)
USS_OWNER_DEF(i64_i32, int64_t, int32_t)
USS_OWNER_DEF(i64_i64, int64_t, int64_t)

USS_PART_DEF(f16_i32, half, int32_t)
USS_PART_DEF(f16_i64, half, int64_t)
USS_PART_DEF(bf16_i32, bfloat16_t, int32_t)
USS_PART_DEF(bf16_i64, bfloat16_t, int64_t)
USS_PART_DEF(f32_i32, float, int32_t)
USS_PART_DEF(f32_i64, float, int64_t)
USS_PART_DEF(i32_i32, int32_t, int32_t)
USS_PART_DEF(i32_i64, int32_t, int64_t)
USS_PART_DEF(i64_i32, int64_t, int32_t)
USS_PART_DEF(i64_i64, int64_t, int64_t)

USS_RED_DEF(f32_f16, float, half)
USS_RED_DEF(f32_bf16, float, bfloat16_t)
USS_RED_DEF(f32_f32, float, float)
USS_RED_DEF(i32_i32, int32_t, int32_t)
USS_RED_DEF(i32_i64, int32_t, int64_t)

} // extern "C"
