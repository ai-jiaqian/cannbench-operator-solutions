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
 * \file moe_finalize_routing_kernel.cpp
 * \brief MoeFinalizeRouting kernel + tiling + launch (bisheng, -xasc, dav-2201).
 *
 *   out[i, c] = skip1[i, c] + skip2[i, c]
 *             + sum_k scales[i, k] * (expanded_permuted_rows[p(i, k), c] + bias[eid(i, k), c])
 *
 *   p(i, k) = expanded_src_to_dst_row[k * numRows + i]   (drop_pad_mode 0/1, column arranged)
 *           = expanded_src_to_dst_row[i * K + k]         (drop_pad_mode 2/3, row arranged)
 *   p == -1 (or outside [0, numDst)) -> the whole term contributes 0 (bias included)
 *   eid outside [0, E)               -> only the bias part is dropped
 *
 * Accumulation is fp32 in ascending k, matching the reference (which upcasts low precision inputs,
 * accumulates and casts back). The narrowing store uses CAST_RINT (round half to even), which is what
 * torch's `.to(dtype)` does.
 *
 * Performance model
 * -----------------
 * Two properties drive the layout:
 *
 * 1. The expert table bias[E, H] is tiny (at most 128 KiB for the graded shapes) while shipping one bias
 *    row per (row, k) term would cost N*K*H bytes - as much as the whole expanded-permuted-rows gather.
 *    The whole table is therefore brought into UB once per block per H-chunk and addressed by eid with the
 *    scalar pipe, which removes every per-term bias DMA. Output decomposes over H, so H is chunked into Hc
 *    columns; Hc is chosen by the host tiling from the UB budget.
 *
 * 2. The random epr row gathers are latency bound. A single queue element covers NB whole (row, k) tasks:
 *    NB DataCopyPad calls land in NB rowBytes-pitched slots of one TQue tensor and cost a single
 *    EnQue/DeQue pair. With TQue depth 2 the next batch is already in flight while the current one is
 *    consumed, so the effective in-flight depth is NB instead of the 8-buffer queue ceiling, and the
 *    per-task synchronisation overhead drops by NB. Hc and NB are picked jointly by the host tiling.
 *
 * skip1 / skip2 travel through a second, depth-2 VECIN TQue issued one row ahead of the consumer; the
 * store uses a depth-1 VECOUT TQue. Metadata (esdr / expert / scales) is staged into plain TBufs once per
 * 32-row tile and guarded by one MTE2->S fence.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "moe_finalize_routing_launch.h"

using namespace AscendC;

namespace {

constexpr int32_t MFR_BQD = 2;    // TQue depth of the batched gather queue (double buffering)
constexpr int64_t MFR_R = 32;     // output rows covered by one staged metadata tile
constexpr int64_t MFR_UNIT = 16;  // Hc granularity: keeps Hc*elemSize a multiple of 32B
constexpr int64_t MFR_NB_MAX = 32;
constexpr int64_t MFR_SG = 4;     // output rows merged into one UB->GM store
// Bytes of one contiguous row of a skip sub-tile: rs = MFR_SKIP_BUDGET / rowBytes, so the whole
// (skip1, skip2) sub-tile arena is 4 * rs * rowBytes <= 16 KiB. Rows are only worth merging while a
// row is small; for a 4 KiB row (Hc = 2048 fp32) rs collapses to 1 and the old per-row path is used.
constexpr int64_t MFR_SKIP_BUDGET = 4096;
constexpr int32_t MFR_EVT_META = 0;

__aicore__ inline int64_t MfrAlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

__aicore__ inline int64_t MfrAlign32(int64_t b)
{
    return ((b + 31) / 32) * 32;
}

/*! \brief output rows carried by one skip sub-tile (shared by the host tiling and the kernel). */
__aicore__ inline int64_t MfrSkipRows(int64_t rowBytes)
{
    int64_t rs = MFR_SKIP_BUDGET / rowBytes;
    if (rs < 1) {
        rs = 1;
    }
    if (rs > MFR_R) {
        rs = MFR_R;
    }
    return rs;
}

/*! \brief widen (or copy) a low precision / fp32 tile into the fp32 working buffer. */
template <typename U>
__aicore__ inline void MfrToFp32(const LocalTensor<float>& dst, const LocalTensor<U>& src, int32_t n)
{
    if constexpr (std::is_same<U, float>::value) {
        Adds(dst, src, 0.0f, n);
    } else {
        Cast<float, U>(dst, src, RoundMode::CAST_NONE, n);
    }
}

/*! \brief scalar low precision -> fp32. bfloat16 must use the official scalar overload: the bisheng
 *         backend rejects a raw bf16 scalar cast outright. */
template <typename U>
__aicore__ inline float MfrScalarFp32(U v)
{
    if constexpr (std::is_same<U, float>::value) {
        return v;
    } else if constexpr (std::is_same<U, half>::value) {
        return static_cast<float>(v);
    } else {
        return AscendC::ToFloat(v);
    }
}

/*! \brief one random epr row gather into one slot of the batch buffer. */
template <typename T>
__aicore__ inline void MfrIssueGather(const LocalTensor<T>& slot, const GlobalTensor<T>& eprGm,
                                      const LocalTensor<int32_t>& esdrT, int64_t dIdx, int64_t H, int64_t cs,
                                      const DataCopyExtParams& cp, const DataCopyPadExtParams<T>& pd,
                                      int64_t numDst)
{
    const int64_t idx = static_cast<int64_t>(esdrT.GetValue(static_cast<uint32_t>(dIdx)));
    const int64_t row = (idx >= 0 && idx < numDst) ? idx : 0;
    DataCopyPad(slot, eprGm[row * H + cs], cp, pd);
}

/*! \brief consume one gathered (row, k) slot: term = fp32(epr)[+fp32(bias)]; term *= scale; acc += term. */
template <typename T, typename S>
__aicore__ inline void MfrConsumeSlot(const LocalTensor<T>& slot, const LocalTensor<float>& accT,
                                      const LocalTensor<float>& termT, const LocalTensor<float>& tmpT,
                                      const LocalTensor<T>& biasT, const LocalTensor<int32_t>& esdrT,
                                      const LocalTensor<int32_t>& expT, const LocalTensor<S>& scT, int64_t dIdx,
                                      int64_t eidIdx, int64_t Hc, int64_t E, int32_t n, int64_t hasBias,
                                      int64_t hasScales, int64_t numDst)
{
    const int64_t idx = static_cast<int64_t>(esdrT.GetValue(static_cast<uint32_t>(dIdx)));
    if (idx < 0 || idx >= numDst) {
        return;
    }
    int64_t eid = -1;
    if (hasBias != 0) {
        eid = static_cast<int64_t>(expT.GetValue(static_cast<uint32_t>(eidIdx)));
    }
    if (hasBias != 0 && eid >= 0 && eid < E) {
        if constexpr (std::is_same<T, float>::value) {
            Add(termT, slot, biasT[static_cast<uint32_t>(eid * Hc)], n);
            if (hasScales != 0) {
                Muls(termT, termT, MfrScalarFp32<S>(scT.GetValue(static_cast<uint32_t>(eidIdx))), n);
            }
            Add(accT, accT, termT, n);
        } else {
            MfrToFp32(termT, slot, n);
            MfrToFp32(tmpT, biasT[static_cast<uint32_t>(eid * Hc)], n);
            Add(termT, termT, tmpT, n);
            if (hasScales != 0) {
                Muls(termT, termT, MfrScalarFp32<S>(scT.GetValue(static_cast<uint32_t>(eidIdx))), n);
            }
            Add(accT, accT, termT, n);
        }
        return;
    }
    if (hasScales != 0) {
        if constexpr (std::is_same<T, float>::value) {
            Muls(termT, slot, MfrScalarFp32<S>(scT.GetValue(static_cast<uint32_t>(eidIdx))), n);
        } else {
            MfrToFp32(termT, slot, n);
            Muls(termT, termT, MfrScalarFp32<S>(scT.GetValue(static_cast<uint32_t>(eidIdx))), n);
        }
        Add(accT, accT, termT, n);
    } else {
        if constexpr (std::is_same<T, float>::value) {
            Add(accT, accT, slot, n);
        } else {
            MfrToFp32(termT, slot, n);
            Add(accT, accT, termT, n);
        }
    }
}

/*! \brief GM->UB copy of one skip sub-tile: nrows consecutive output rows, each Hl columns wide.
 *         skip rows are H elements apart in GM, so the inter-block gap is (H - Hl) * elemSize. */
template <typename T>
__aicore__ inline void MfrIssueSkipRows(const LocalTensor<T>& dst, const GlobalTensor<T>& gm, int64_t row0,
                                        int64_t nrows, int64_t H, int64_t cs, int64_t Hl, int64_t es,
                                        uint32_t rowBytes)
{
    DataCopyExtParams cp{static_cast<uint16_t>(nrows),
                         static_cast<uint32_t>(Hl * es),
                         static_cast<uint32_t>((H - Hl) * es),
                         static_cast<uint32_t>(((int64_t)rowBytes - MfrAlign32(Hl * es)) / 32), 0};
    DataCopyPadExtParams<T> pd;
    pd.isPad = false;
    pd.leftPadding = 0;
    pd.rightPadding = 0;
    DataCopyPad(dst, gm[row0 * H + cs], cp, pd);
}

/*! \brief enqueue one skip sub-tile (rows [sub*rs, sub*rs+nrows) of both skip tensors) as a single
 *         queue element: the rs skip1 rows first, then the rs skip2 rows. One Alloc/EnQue pair per
 *         sub-tile instead of one per row, and one strided DMA per tensor instead of one per row. */
template <typename T>
__aicore__ inline void MfrEnqSkipSub(TQue<QuePosition::VECIN, 2>& skipQ, const GlobalTensor<T>& s1Gm,
                                     const GlobalTensor<T>& s2Gm, int64_t t0, int64_t sub, int64_t rs, int64_t Rt,
                                     int64_t H, int64_t cs, int64_t Hl, int64_t es, int64_t slotElems,
                                     uint32_t rowBytes, int64_t hasSkip1, int64_t hasSkip2)
{
    const int64_t row0 = sub * rs;
    int64_t nrows = Rt - row0;
    if (nrows > rs) {
        nrows = rs;
    }
    if (nrows <= 0) {
        return;
    }
    auto buf = skipQ.template AllocTensor<T>();
    if (hasSkip1 != 0) {
        MfrIssueSkipRows(buf, s1Gm, t0 + row0, nrows, H, cs, Hl, es, rowBytes);
    }
    if (hasSkip2 != 0) {
        MfrIssueSkipRows(buf[static_cast<uint32_t>(rs * slotElems)], s2Gm, t0 + row0, nrows, H, cs, Hl, es, rowBytes);
    }
    skipQ.template EnQue(buf);
}

/*! \brief start the accumulator of one output row: skip1 + skip2 (fp32), zero when both are absent. */
template <typename T>
__aicore__ inline void MfrBuildAcc(const LocalTensor<T>& sbuf, int64_t ri, int64_t rs, int64_t slotElems,
                                   const LocalTensor<float>& accT, const LocalTensor<float>& tmpT, int32_t n,
                                   int64_t hasSkip1, int64_t hasSkip2)
{
    if (hasSkip1 != 0) {
        MfrToFp32(accT, sbuf[static_cast<uint32_t>(ri * slotElems)], n);
    } else {
        Duplicate(accT, 0.0f, n);
    }
    if (hasSkip2 != 0) {
        MfrToFp32(tmpT, sbuf[static_cast<uint32_t>((rs + ri) * slotElems)], n);
        Add(accT, accT, tmpT, n);
    }
}

}  // namespace

template <typename T, typename S>
__global__ __aicore__ void mfr_kernel(MFR_ARGS)
{
    if (numRows <= 0 || H <= 0 || K <= 0 || Hc <= 0) {
        return;
    }
    const int64_t blk = static_cast<int64_t>(GetBlockIdx());
    if (blk >= numBlocks) {
        return;
    }
    const int64_t r0 = numRows * blk / numBlocks;
    const int64_t r1 = numRows * (blk + 1) / numBlocks;
    if (r0 >= r1) {
        return;
    }

    const int64_t es = static_cast<int64_t>(sizeof(T));
    const uint32_t rowBytes = static_cast<uint32_t>(MfrAlign32(Hc * es));
    const int64_t slotElems = static_cast<int64_t>(rowBytes) / es;
    const int64_t rs = MfrSkipRows(static_cast<int64_t>(rowBytes));
    const int64_t fp32Bytes = MfrAlign32(Hc * 4);
    const int64_t metaBytes = MfrAlign32(MFR_R * K * 4);
    const int64_t biasBytes = (E > 0) ? MfrAlign32(E * static_cast<int64_t>(rowBytes)) : 32;
    const int64_t nb = (NB < 1) ? 1 : NB;
    const bool anySkip = (hasSkip1 != 0) || (hasSkip2 != 0);

    TPipe pipe;
    TQue<QuePosition::VECIN, MFR_BQD> gatherQ;
    TQue<QuePosition::VECIN, 2> skipQ;
    TQue<QuePosition::VECOUT, 1> outQ;
    TBuf<TPosition::VECCALC> accBuf, termBuf, tmpBuf, biasBuf, esdrBuf, expBuf, scBuf;
    pipe.InitBuffer(gatherQ, MFR_BQD, nb * rowBytes);
    pipe.InitBuffer(skipQ, 2, 2 * rs * slotElems * es);
    pipe.InitBuffer(outQ, 1, MFR_SG * rowBytes);
    pipe.InitBuffer(accBuf, fp32Bytes);
    pipe.InitBuffer(termBuf, fp32Bytes);
    pipe.InitBuffer(tmpBuf, fp32Bytes);
    pipe.InitBuffer(biasBuf, biasBytes);
    pipe.InitBuffer(esdrBuf, metaBytes);
    pipe.InitBuffer(expBuf, metaBytes);
    pipe.InitBuffer(scBuf, metaBytes);

    const LocalTensor<float> accT = accBuf.Get<float>();
    const LocalTensor<float> termT = termBuf.Get<float>();
    const LocalTensor<float> tmpT = tmpBuf.Get<float>();
    const LocalTensor<T> biasT = biasBuf.Get<T>();
    const LocalTensor<int32_t> esdrT = esdrBuf.Get<int32_t>();
    const LocalTensor<int32_t> expT = expBuf.Get<int32_t>();
    const LocalTensor<S> scT = scBuf.Get<S>();

    GlobalTensor<T> eprGm, skip1Gm, skip2Gm, biasGm, outGm;
    GlobalTensor<int32_t> esdrGm, expGm;
    GlobalTensor<S> scGm;
    eprGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(epr));
    esdrGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(esdr));
    skip1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(skip1));
    skip2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(skip2));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    scGm.SetGlobalBuffer(reinterpret_cast<__gm__ S*>(scales));
    expGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(expert));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

    const bool hasB = hasBias != 0;
    const bool hasSc = hasScales != 0;

    const int64_t nChunks = (H + Hc - 1) / Hc;
    for (int64_t c = 0; c < nChunks; ++c) {
        const int64_t cs = c * Hc;
        const int64_t Hl = (H - cs < Hc) ? (H - cs) : Hc;
        const uint32_t hlBytes = static_cast<uint32_t>(Hl * es);
        const int32_t nHl = static_cast<int32_t>(Hl);
        const bool colMode = (mode <= 1);
        DataCopyExtParams gcp{1, hlBytes, 0, 0, 0};
        DataCopyPadExtParams<T> gpd;
        gpd.isPad = false;
        gpd.leftPadding = 0;
        gpd.rightPadding = 0;

        if (hasB && E > 0) {
            DataCopyExtParams cp{static_cast<uint16_t>(E), hlBytes,
                                 static_cast<uint32_t>((H - Hl) * es),
                                 static_cast<uint32_t>(((int64_t)rowBytes - MfrAlign32(hlBytes)) / 32), 0};
            DataCopyPadExtParams<T> pd;
            pd.isPad = false;
            pd.leftPadding = 0;
            pd.rightPadding = 0;
            DataCopyPad(biasT, biasGm[cs], cp, pd);
        }

        for (int64_t t0 = r0; t0 < r1; t0 += MFR_R) {
            const int64_t Rt = (r1 - t0 < MFR_R) ? (r1 - t0) : MFR_R;
            DataCopyPadExtParams<int32_t> pdI;
            pdI.isPad = false;
            pdI.leftPadding = 0;
            pdI.rightPadding = 0;
            if (mode <= 1) {
                DataCopyExtParams cp{static_cast<uint16_t>(K), static_cast<uint32_t>(Rt * 4),
                                     static_cast<uint32_t>((numRows - Rt) * 4),
                                     static_cast<uint32_t>(((int64_t)MFR_R * 4 - MfrAlign32(Rt * 4)) / 32), 0};
                DataCopyPad(esdrT, esdrGm[t0], cp, pdI);
            } else {
                DataCopyExtParams cp{1, static_cast<uint32_t>(Rt * K * 4), 0, 0, 0};
                DataCopyPad(esdrT, esdrGm[t0 * K], cp, pdI);
            }
            if (hasB) {
                DataCopyExtParams cp{1, static_cast<uint32_t>(Rt * K * 4), 0, 0, 0};
                DataCopyPad(expT, expGm[t0 * K], cp, pdI);
            }
            if (hasSc) {
                DataCopyPadExtParams<S> pdS;
                pdS.isPad = false;
                pdS.leftPadding = 0;
                pdS.rightPadding = 0;
                DataCopyExtParams cp{1, static_cast<uint32_t>(Rt * K * static_cast<int64_t>(sizeof(S))), 0, 0, 0};
                DataCopyPad(scT, scGm[t0 * K], cp, pdS);
            }
            SetFlag<HardEvent::MTE2_S>(MFR_EVT_META);
            WaitFlag<HardEvent::MTE2_S>(MFR_EVT_META);

            const int64_t nTasks = Rt * K;
            const int64_t nBatches = (nTasks + nb - 1) / nb;

            int64_t curSub = -1;
            int64_t stageCnt = 0;
            LocalTensor<T> skipBuf;
            LocalTensor<T> obuf;
            if (anySkip) {
                MfrEnqSkipSub(skipQ, skip1Gm, skip2Gm, t0, 0, rs, Rt, H, cs, Hl, es, slotElems, rowBytes,
                              hasSkip1, hasSkip2);
            }

            for (int64_t b = 0; b <= nBatches; ++b) {
                if (b < nBatches) {
                    const int64_t base = b * nb;
                    int64_t cnt = nTasks - base;
                    if (cnt > nb) {
                        cnt = nb;
                    }
                    int64_t rr = base / K;
                    int64_t k = base - rr * K;
                    auto buf = gatherQ.template AllocTensor<T>();
                    for (int64_t j = 0; j < cnt; ++j) {
                        const int64_t dIdx = colMode ? (k * MFR_R + rr) : (rr * K + k);
                        MfrIssueGather(buf[j * slotElems], eprGm, esdrT, dIdx, H, cs, gcp, gpd, numDst);
                        if (++k == K) {
                            k = 0;
                            ++rr;
                        }
                    }
                    gatherQ.template EnQue(buf);
                }
                if (b > 0) {
                    const int64_t base = (b - 1) * nb;
                    int64_t cnt = nTasks - base;
                    if (cnt > nb) {
                        cnt = nb;
                    }
                    int64_t rr = base / K;
                    int64_t k = base - rr * K;
                    auto pb = gatherQ.template DeQue<T>();
                    for (int64_t j = 0; j < cnt; ++j) {
                        const int64_t dIdx = colMode ? (k * MFR_R + rr) : (rr * K + k);
                        if (k == 0) {
                            if (anySkip) {
                                if (rr - curSub * rs >= rs) {
                                    if (curSub >= 0) {
                                        skipQ.FreeTensor(skipBuf);
                                    }
                                    skipBuf = skipQ.template DeQue<T>();
                                    curSub = rr / rs;
                                    MfrEnqSkipSub(skipQ, skip1Gm, skip2Gm, t0, curSub + 1, rs, Rt, H, cs, Hl, es,
                                                  slotElems, rowBytes, hasSkip1, hasSkip2);
                                }
                                MfrBuildAcc(skipBuf, rr - curSub * rs, rs, slotElems, accT, tmpT, nHl, hasSkip1,
                                            hasSkip2);
                            } else {
                                Duplicate(accT, 0.0f, nHl);
                            }
                        }
                        MfrConsumeSlot(pb[j * slotElems], accT, termT, tmpT, biasT, esdrT, expT, scT, dIdx,
                                       rr * K + k, Hc, E, nHl, hasBias, hasScales, numDst);
                        if (k == K - 1) {
                            if (stageCnt == 0) {
                                obuf = outQ.template AllocTensor<T>();
                            }
                            if constexpr (std::is_same<T, float>::value) {
                                Adds(obuf[static_cast<uint32_t>(stageCnt * slotElems)], accT, 0.0f, nHl);
                            } else {
                                Cast<T, float>(obuf[static_cast<uint32_t>(stageCnt * slotElems)], accT,
                                               RoundMode::CAST_RINT, nHl);
                            }
                            ++stageCnt;
                            if (stageCnt == MFR_SG || rr + 1 == Rt) {
                                outQ.template EnQue(obuf);
                                auto got = outQ.template DeQue<T>();
                                DataCopyExtParams scp{static_cast<uint16_t>(stageCnt), hlBytes,
                                                      static_cast<uint32_t>(((int64_t)rowBytes -
                                                                             MfrAlign32((int64_t)hlBytes)) /
                                                                            32),
                                                      static_cast<uint32_t>((H - Hl) * es), 0};
                                DataCopyPad(outGm[(t0 + rr + 1 - stageCnt) * H + cs], got, scp);
                                outQ.FreeTensor(got);
                                stageCnt = 0;
                            }
                        }
                        if (++k == K) {
                            k = 0;
                            ++rr;
                        }
                    }
                    gatherQ.FreeTensor(pb);
                }
            }

            if (anySkip && curSub >= 0) {
                skipQ.FreeTensor(skipBuf);
            }

            SetFlag<HardEvent::S_MTE2>(MFR_EVT_META);
            WaitFlag<HardEvent::S_MTE2>(MFR_EVT_META);
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Host side tiling
// ---------------------------------------------------------------------------------------------------

namespace {

inline int64_t MfrHAlignUp(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

inline int64_t MfrHAlign32(int64_t b)
{
    return ((b + 31) / 32) * 32;
}

}  // namespace

std::tuple<int64_t, int64_t, int64_t> calc_mfr_tiling(int64_t numRows, int64_t H, int64_t K, int64_t E,
                                                      int64_t elemSize)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 1;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        coreNum = static_cast<int64_t>(plat->GetCoreNumAiv());
    }
    if (ubSize == 0) {
        ubSize = 192u * 1024u;
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    int64_t numBlocks = std::min<int64_t>(coreNum, numRows);
    if (numBlocks <= 0) {
        numBlocks = 1;
    }

    const int64_t budget = static_cast<int64_t>(ubSize) - 16384;
    const int64_t metaBytes = 3 * MfrHAlign32(MFR_R * K * 4);
    // Total UB for a given H-chunk and batch size. Must mirror the kernel's InitBuffer calls:
    //   gather arena 2*nb*rb, skip sub-tile arena 4*rs*rb, store stage MFR_SG*rb,
    //   three fp32 working tiles 3*align32(Hc*4), metadata 3*align32(MFR_R*K*4), bias table E*rb.
    auto totalFor = [&](int64_t hc, int64_t nbat) -> int64_t {
        const int64_t rb = MfrHAlign32(hc * elemSize);
        const int64_t bias = (E > 0) ? MfrHAlign32(E * rb) : 32;
        // Same rule as the device side MfrSkipRows(): keep the skip arena at most 16 KiB.
        int64_t rs = MFR_SKIP_BUDGET / rb;
        if (rs < 1) {
            rs = 1;
        }
        if (rs > MFR_R) {
            rs = MFR_R;
        }
        return 2 * nbat * rb + 4 * rs * rb + MFR_SG * rb + 3 * MfrHAlign32(hc * 4) + metaBytes + bias;
    };

    // Largest Hc that fits with a single-task batch.
    int64_t hcMax = MfrHAlignUp(H, MFR_UNIT);
    for (int32_t it = 0; it < 256; ++it) {
        if (hcMax <= MFR_UNIT) {
            break;
        }
        if (totalFor(hcMax, 1) <= budget) {
            break;
        }
        int64_t nxt = (hcMax * 7) / 8;
        nxt = (nxt / MFR_UNIT) * MFR_UNIT;
        if (nxt >= hcMax) {
            nxt = hcMax - MFR_UNIT;
        }
        if (nxt < MFR_UNIT) {
            hcMax = MFR_UNIT;
            break;
        }
        hcMax = nxt;
    }

    // Trade chunk length against gather batch depth. Measured per (row,k) cost is about L/nb + C
    // cycles (L = exposed random-gather latency, C = fixed scalar issue cost), so the total work of
    // one block scales as nChunks * (L/nb + C). Minimise that: an extra chunk doubles the task count
    // and only pays off when it buys enough in-flight depth.
    const int64_t latCycles = 2000;
    const int64_t fixCycles = 300;
    int64_t bestHc = hcMax;
    int64_t bestNb = 1;
    int64_t bestCost = -1;
    for (int64_t hc = hcMax;; hc = hc / 2) {
        const int64_t hcA = (hc / MFR_UNIT) * MFR_UNIT;
        if (hcA < MFR_UNIT) {
            break;
        }
        const int64_t rb = MfrHAlign32(hcA * elemSize);
        int64_t nbat = (budget - totalFor(hcA, 0)) / (2 * rb);
        if (nbat < 1) {
            nbat = 1;
        }
        if (nbat > MFR_NB_MAX) {
            nbat = MFR_NB_MAX;
        }
        const int64_t nChunks = (H + hcA - 1) / hcA;
        const int64_t cost = nChunks * (latCycles / nbat + fixCycles);
        if (bestCost < 0 || cost < bestCost) {
            bestCost = cost;
            bestHc = hcA;
            bestNb = nbat;
        }
        if (hcA == MFR_UNIT) {
            break;
        }
    }

    return std::make_tuple(numBlocks, bestHc, bestNb);
}

// ---------------------------------------------------------------------------------------------------
// Launch wrappers (regular C functions callable from g++)
// ---------------------------------------------------------------------------------------------------

extern "C" {

#define MFR_DEFINE_LAUNCH(NAME, TT, SS)                                                          \
    void NAME(MFR_ARGS, void* stream)                                                            \
    {                                                                                            \
        mfr_kernel<TT, SS><<<numBlocks, nullptr, stream>>>(                                      \
            epr, esdr, skip1, skip2, bias, scales, expert, out, numRows, H, K, E, numDst, mode,  \
            Hc, NB, numBlocks, hasSkip1, hasSkip2, hasBias, hasScales);                          \
    }

MFR_DEFINE_LAUNCH(launch_mfr_half_half, half, half)
MFR_DEFINE_LAUNCH(launch_mfr_half_float, half, float)
MFR_DEFINE_LAUNCH(launch_mfr_half_bf16, half, bfloat16_t)
MFR_DEFINE_LAUNCH(launch_mfr_float_half, float, half)
MFR_DEFINE_LAUNCH(launch_mfr_float_float, float, float)
MFR_DEFINE_LAUNCH(launch_mfr_float_bf16, float, bfloat16_t)
MFR_DEFINE_LAUNCH(launch_mfr_bf16_half, bfloat16_t, half)
MFR_DEFINE_LAUNCH(launch_mfr_bf16_float, bfloat16_t, float)
MFR_DEFINE_LAUNCH(launch_mfr_bf16_bf16, bfloat16_t, bfloat16_t)

#undef MFR_DEFINE_LAUNCH

}  // extern "C"
