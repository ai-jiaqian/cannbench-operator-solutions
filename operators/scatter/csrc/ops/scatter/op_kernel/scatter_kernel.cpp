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
 * \file scatter_kernel.cpp
 * \brief Scatter (ScatterUpdate) device kernel, tiling and launch wrapper (bisheng + -xasc).
 *
 * Semantics, matching task/reference.py:
 *   y = data.clone();  y[..., indices[...], ...] = combine(y, updates)   (update/add/mul/amin/amax)
 *
 * Decomposition.  The scatter axis `dim` splits every tensor as
 *   data / y  : [outer, K , inner]
 *   indices   : [outer, Ku, inner]
 *   updates   : [outer, Ku, inner]
 * and an element of the update domain at (o, j, i) targets output element (o, indices[o,j,i], i).
 * The kernel walks the OUTPUT domain so the data -> y copy is fused with the merge and every output
 * element is written exactly once; the only non contiguous GM traffic is the strided multi-row
 * DataCopyPad used for a partial inner chunk.
 *
 * One work unit owns (outer index o) x (inner chunk of `tileL` columns, all chunks the same width -
 * the last one is shifted left and recomputes its overlap, which is harmless because the merge of one
 * column is a pure function of that column) x (scatter chunk of `tileK` rows).  Inside a unit the
 * resident output block is kept in UB and the update stream is scanned in row-major source order, so
 * duplicate targets are resolved in one deterministic place: last update wins for reduce=None, and
 * the arithmetic modes are order independent.
 *
 * 16 bit payloads cannot be combined by the aicore scalar unit ("half/bfloat16_t precision operation
 * is not allowed in aicore function") and the scalar 16 bit float cast is not emittable, so:
 *   - reduce=None moves the 16 bit payload as a raw uint16_t bit pattern (bit identical, also the only
 *     way to preserve NaN payloads / denormals);
 *   - the arithmetic modes on 16 bit data promote the resident block and the update rows to fp32 with
 *     the *vector* Cast, merge in fp32, and cast the block back once before the store.
 *
 * UB layout / DataCopyPad contract:
 *   - every multi-row transfer uses dstStride/srcStride 0 on the UB side, so rows land at the hardware
 *     32 B aligned row pitch.  `tileL` is always chosen so that tileL*sizeof(elements) and
 *     tileL*sizeof(indices) are both multiples of 32 B (or `tileL == inner`, where the transfer is a
 *     single contiguous run), hence the packed row pitch is exactly tileL and no explicit gap is ever
 *     needed.
 *   - a chunk that spans the whole `inner` extent is contiguous in GM and in UB, so it is moved as a
 *     plain contiguous run of elements (blockCount = 1); otherwise the transfer is a multi-row
 *     strided copy with blockCount = min(rows, 4095)  (the DataCopyPad burst counter is 12 bits).
 *   - scalar UB accesses are not covered by the queue oriented events of TQue, so the kernel uses
 *     TBuf + PipeBarrier<PIPE_ALL>() to order MTE2(load) -> S(merge) -> MTE3(store) -> buffer reuse.
 *
 * The merge itself (scatter_merge.h) is a scalar read-modify-write and dominates the runtime.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "scatter_launch.h"
#include "scatter_merge.h"

namespace scat {

constexpr int64_t kMaxRowsDma = 4095;      // DataCopyPad blockCount is a 12 bit field
constexpr int64_t kMaxContigElems = 16384; // elements per contiguous DMA call

template <typename T>
__aicore__ inline void CopyInBlock(const AscendC::LocalTensor<T> &dst,
                                   const AscendC::GlobalTensor<T> &src, int64_t gmBase,
                                   int64_t inner, int64_t rows, int64_t cols, int64_t pitch,
                                   bool contig)
{
    using namespace AscendC;
    constexpr int64_t ES = static_cast<int64_t>(sizeof(T));
    if (contig) {
        int64_t total = rows * cols;
        int64_t done = 0;
        while (done < total) {
            int64_t n = total - done;
            if (n > kMaxContigElems) {
                n = kMaxContigElems;
            }
            DataCopyExtParams cp{1, static_cast<uint32_t>(n * ES), 0, 0, 0};
            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            DataCopyPad(dst[static_cast<uint32_t>(done)], src[gmBase + done], cp, pp);
            done += n;
        }
    } else {
        int64_t done = 0;
        while (done < rows) {
            int64_t n = rows - done;
            if (n > kMaxRowsDma) {
                n = kMaxRowsDma;
            }
            DataCopyExtParams cp{static_cast<uint16_t>(n), static_cast<uint32_t>(cols * ES),
                                 static_cast<uint32_t>((inner - cols) * ES), 0, 0};
            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            DataCopyPad(dst[static_cast<uint32_t>(done * pitch)], src[gmBase + done * inner], cp,
                        pp);
            done += n;
        }
    }
}

template <typename T>
__aicore__ inline void CopyOutBlock(const AscendC::GlobalTensor<T> &dst,
                                    const AscendC::LocalTensor<T> &src, int64_t gmBase,
                                    int64_t inner, int64_t rows, int64_t cols, int64_t pitch,
                                    bool contig)
{
    using namespace AscendC;
    constexpr int64_t ES = static_cast<int64_t>(sizeof(T));
    if (contig) {
        int64_t total = rows * cols;
        int64_t done = 0;
        while (done < total) {
            int64_t n = total - done;
            if (n > kMaxContigElems) {
                n = kMaxContigElems;
            }
            DataCopyExtParams cp{1, static_cast<uint32_t>(n * ES), 0, 0, 0};
            DataCopyPad(dst[gmBase + done], src[static_cast<uint32_t>(done)], cp);
            done += n;
        }
    } else {
        int64_t done = 0;
        while (done < rows) {
            int64_t n = rows - done;
            if (n > kMaxRowsDma) {
                n = kMaxRowsDma;
            }
            DataCopyExtParams cp{static_cast<uint16_t>(n), static_cast<uint32_t>(cols * ES), 0,
                                 static_cast<uint32_t>((inner - cols) * ES), 0};
            DataCopyPad(dst[gmBase + done * inner], src[static_cast<uint32_t>(done * pitch)], cp);
            done += n;
        }
    }
}

template <typename CT, typename IDX, typename LT, int MODE>
__global__ __aicore__ void scatter_kernel(GM_ADDR gData, GM_ADDR gIdx, GM_ADDR gUpd, GM_ADDR gY,
                                          int64_t outer, int64_t K, int64_t Ku, int64_t inner,
                                          int64_t tileL, int64_t tileK, int64_t ju, int64_t nL,
                                          int64_t nK, int64_t units, int64_t unitsPerBlock,
                                          int64_t offBlk, int64_t offRaw, int64_t offIdx,
                                          int64_t offUpd, int64_t totalBytes)
{
    using namespace AscendC;
    constexpr bool PROMOTE = !std::is_same<CT, LT>::value;

    int64_t u0 = unitsPerBlock * static_cast<int64_t>(GetBlockIdx());
    if (u0 >= units) {
        return;
    }
    int64_t u1 = u0 + unitsPerBlock;
    if (u1 > units) {
        u1 = units;
    }

    GlobalTensor<LT> dataGm;
    GlobalTensor<IDX> idxGm;
    GlobalTensor<LT> updGm;
    GlobalTensor<LT> yGm;
    dataGm.SetGlobalBuffer((__gm__ LT *)gData);
    idxGm.SetGlobalBuffer((__gm__ IDX *)gIdx);
    updGm.SetGlobalBuffer((__gm__ LT *)gUpd);
    yGm.SetGlobalBuffer((__gm__ LT *)gY);

    TPipe pipe;
    TBuf<TPosition::VECCALC> work;
    pipe.InitBuffer(work, static_cast<uint32_t>(totalBytes));

    const int64_t blkElems = tileK * tileL;
    const int64_t juElems = (ju > 0 ? ju : 1) * tileL;
    const int64_t rawElems = (blkElems > juElems) ? blkElems : juElems;

    LocalTensor<CT> blk = work.GetWithOffset<CT>(blkElems, offBlk);
    LocalTensor<IDX> idxb = work.GetWithOffset<IDX>(juElems, offIdx);
    LocalTensor<CT> updF = work.GetWithOffset<CT>(juElems, offUpd);

    LocalTensor<LT> blkLT;
    LocalTensor<LT> updLT;
    if constexpr (PROMOTE) {
        LocalTensor<LT> rawb = work.GetWithOffset<LT>(rawElems, offRaw);
        blkLT = rawb;
        updLT = rawb;
    } else {
        blkLT = blk;
        updLT = updF;
    }

    __ubuf__ CT *blkp = (__ubuf__ CT *)blk.GetPhyAddr();
    __ubuf__ IDX *idxp = (__ubuf__ IDX *)idxb.GetPhyAddr();
    __ubuf__ CT *updp = (__ubuf__ CT *)updF.GetPhyAddr();

    const bool fullK = (nK == 1);

    for (int64_t u = u0; u < u1; ++u) {
        int64_t kc = u % nK;
        int64_t rem = u / nK;
        int64_t icc = rem % nL;
        int64_t o = rem / nL;

        int64_t k0 = kc * tileK;
        int64_t kb = tileK;
        if (k0 + kb > K) {
            kb = K - k0;
        }
        if (kb <= 0) {
            continue;
        }
        int64_t cl = tileL;
        int64_t i0 = icc * tileL;
        if (i0 > inner - tileL) {
            i0 = inner - tileL;
        }
        if (i0 < 0) {
            i0 = 0;
        }
        const bool contig = (cl == inner);

        int64_t dBase = o * K * inner + k0 * inner + i0;
        int64_t uBase = o * Ku * inner + i0;

        CopyInBlock<LT>(blkLT, dataGm, dBase, inner, kb, cl, tileL, contig);
        PipeBarrier<PIPE_ALL>();
        if constexpr (PROMOTE) {
            Cast(blk, blkLT, RoundMode::CAST_NONE, static_cast<uint32_t>(kb * tileL));
            PipeBarrier<PIPE_ALL>();
        }

        for (int64_t j0 = 0; j0 < Ku; j0 += ju) {
            int64_t jc = ju;
            if (j0 + jc > Ku) {
                jc = Ku - j0;
            }
            CopyInBlock<IDX>(idxb, idxGm, uBase + j0 * inner, inner, jc, cl, tileL, contig);
            CopyInBlock<LT>(updLT, updGm, uBase + j0 * inner, inner, jc, cl, tileL, contig);
            PipeBarrier<PIPE_ALL>();
            if constexpr (PROMOTE) {
                Cast(updF, updLT, RoundMode::CAST_NONE, static_cast<uint32_t>(jc * tileL));
                PipeBarrier<PIPE_ALL>();
            }
            MergeChunk<MODE, CT, IDX>(blkp, idxp, updp, jc, cl, tileL, k0, kb, fullK);
        }

        PipeBarrier<PIPE_ALL>();
        if constexpr (PROMOTE) {
            Cast(blkLT, blk, RoundMode::CAST_RINT, static_cast<uint32_t>(kb * tileL));
            PipeBarrier<PIPE_ALL>();
        }
        CopyOutBlock<LT>(yGm, blkLT, dBase, inner, kb, cl, tileL, contig);
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace scat

/* ===================================================================== *
 * Host tiling
 * ===================================================================== */
namespace {

constexpr int64_t kHostMaxRowsDma = scat::kMaxRowsDma;
constexpr int64_t kHostMaxContigElems = scat::kMaxContigElems;

inline int64_t HAUp(int64_t v, int64_t a)
{
    return (v + a - 1) / a * a;
}

/* smallest tileL step that keeps both the element and the index row 32 B aligned */
inline int64_t LStep(int64_t esLT, int64_t idxBytes)
{
    int64_t a = (32 + esLT - 1) / esLT;
    int64_t b = (32 + idxBytes - 1) / idxBytes;
    return (a > b) ? a : b;
}

} // namespace

ScatterTiling calc_scatter_tiling(int64_t outer, int64_t kLen, int64_t kuLen, int64_t inner,
                                  int64_t esLT, int64_t esCT, int64_t idxBytes, int64_t promote,
                                  int64_t mode)
{
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = static_cast<int64_t>(plat->GetCoreNumAiv());
    if (coreNum <= 0) {
        coreNum = 1;
    }
    int64_t budget = static_cast<int64_t>(ubSize) - 16384;
    if (budget < 8192) {
        budget = 8192;
    }
    if (inner < 1) {
        inner = 1;
    }
    if (kLen < 1) {
        kLen = 1;
    }
    if (outer < 1) {
        outer = 1;
    }
    const int64_t maxRowsDma = kHostMaxRowsDma;
    const int64_t step = LStep(esLT, idxBytes);
    const bool modeOw = (mode == SC_RM_UPDATE);

    // ---- candidate ladders -------------------------------------------------------------
    int64_t Lc[300];
    int64_t nLc = 0;
    Lc[nLc++] = inner; // whole-inner chunk (contiguous DMA, any width)
    for (int64_t i = 1; i <= 64 && nLc < 290; ++i) {
        int64_t v = step * i;
        if (v < inner) {
            Lc[nLc++] = v;
        }
    }
    for (int64_t v = step; v <= inner && v <= 262144 && nLc < 300; v *= 2) {
        if (v < inner) {
            Lc[nLc++] = v;
        }
    }

    const int64_t kBc[32] = {1,     2,     3,     4,     6,     8,     12,    16,    24,   32,
                             48,    64,    96,    128,   192,   256,   384,   512,   768,  1024,
                             1536,  2048,  3072,  4096,  6144,  8192,  12288, 16384, 24576,
                             32768, 65536, 131072};
    const int64_t nKBc = 32;
    const int64_t kJc[13] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int64_t nJc = 13;

    // ---- cost model weights (AI-core cycles) ------------------------------------------
    // Calibrated against measurement: the scalar read-modify-write of the resident block costs
    // roughly kScanCycles per update element (three or four scalar UB accesses) and dominates the
    // measured runtime.  A one-column block with an arithmetic mode is the slowest shape, because its
    // resident-block read has to stay ordered behind the previous store.
    const double kScanWide = 13.0;
    const double kScanCol1Ow = 13.0;
    const double kScanCol1Rw = 26.0;
    const double kRowCycles = 3.0;       // per DMA row descriptor (strided copies only)
    const double kBytesPerCycle = 140.0; // ~280 GB/s of DRAM bandwidth
    const double kCallCycles = 60.0;
    const double kUnitCycles = 2000.0;   // fixed cost of one work unit

    ScatterTiling best;
    best.tileL = 1;
    best.tileK = 1;
    best.ju = 1;
    best.nL = 1;
    best.nK = 1;
    best.units = 1;
    best.unitsPerBlock = 1;
    best.numBlocks = 1;
    best.offBlk = 0;
    best.offRaw = 0;
    best.offIdx = 0;
    best.offUpd = 0;
    best.totalBytes = 0;
    double bestTime = -1.0;
    bool haveBest = false;

    for (int64_t li = 0; li < nLc; ++li) {
        int64_t L = Lc[li];
        if (L > inner) {
            L = inner;
        }
        if (L < 1) {
            continue;
        }
        const bool contig = (L == inner);
        if (!contig && (L % step) != 0) {
            continue;
        }
        int64_t pitch = L;
        const double scanCycles = (L == 1) ? (modeOw ? kScanCol1Ow : kScanCol1Rw) : kScanWide;
        for (int64_t ki = 0; ki < nKBc; ++ki) {
            int64_t KB = kBc[ki];
            if (KB > kLen) {
                KB = kLen;
            }
            if (KB < 1) {
                continue;
            }
            for (int64_t ji = 0; ji < nJc; ++ji) {
                int64_t JU = kJc[ji];
                if (kuLen > 0 && JU > kuLen) {
                    JU = kuLen;
                }
                if (JU < 1) {
                    JU = 1;
                }
                // ---- UB budget ---------------------------------------------------------
                int64_t blkBytes = HAUp(KB * pitch * esCT, 32);
                int64_t rawArea = promote ? HAUp(std::max(KB, JU) * pitch * esLT, 32) : 0;
                int64_t idxArea = HAUp(JU * pitch * idxBytes, 32);
                int64_t updArea = HAUp(JU * pitch * esCT, 32);
                int64_t total = blkBytes + rawArea + idxArea + updArea;
                if (total > budget) {
                    continue;
                }
                int64_t nLv = (inner + L - 1) / L;
                int64_t nKv = (kLen + KB - 1) / KB;
                int64_t units = outer * nLv * nKv;
                if (units < 1) {
                    continue;
                }
                int64_t nb = std::min<int64_t>(coreNum, units);
                int64_t per = (units + nb - 1) / nb;

                double scanOps = static_cast<double>(outer) * nKv * kuLen * inner;
                double dmaBytes =
                    static_cast<double>(outer) *
                    (2.0 * kLen * inner * esLT +
                     static_cast<double>(nKv) * kuLen * inner * (idxBytes + esLT));
                // rows moved as multi-row blocks: one descriptor per matrix row for strided copies,
                // one per contiguous run when the chunk spans the whole inner extent
                double rowsPerUnit;
                if (contig) {
                    rowsPerUnit =
                        2.0 * ((KB + kHostMaxContigElems - 1) / kHostMaxContigElems) + 2.0;
                } else {
                    rowsPerUnit = 2.0 * KB + 2.0 * kuLen;
                }
                double dmaRows = static_cast<double>(units) * rowsPerUnit;
                double calls = static_cast<double>(units) *
                               (2.0 * ((KB + maxRowsDma - 1) / maxRowsDma) +
                                2.0 * ((kuLen + JU - 1) / JU) + 2.0);
                double work = scanOps * scanCycles + dmaRows * kRowCycles +
                              dmaBytes / kBytesPerCycle + calls * kCallCycles;
                // The scalar merge and the DMA do not overlap much (pipe barriers separate them),
                // so the two terms add.
                double time = static_cast<double>(per) * (kUnitCycles + work / units);

                if (!haveBest || time < bestTime) {
                    haveBest = true;
                    bestTime = time;
                    best.tileL = L;
                    best.tileK = KB;
                    best.ju = JU;
                    best.nL = nLv;
                    best.nK = nKv;
                    best.units = units;
                    best.unitsPerBlock = per;
                    best.numBlocks = nb;
                    best.offBlk = 0;
                    best.offRaw = (promote ? blkBytes : 0);
                    best.offIdx = (promote ? blkBytes + rawArea : blkBytes);
                    best.offUpd = best.offIdx + idxArea;
                    best.totalBytes = best.offUpd + updArea;
                }
            }
        }
    }

    if (!haveBest) {
        // Degenerate fallback: one column per unit with one scatter row resident.
        int64_t L = 1;
        if (inner < L) {
            L = inner;
        }
        best.tileL = L;
        best.tileK = 1;
        best.ju = 1;
        best.nL = (inner + L - 1) / L;
        best.nK = kLen;
        best.units = outer * best.nL * best.nK;
        best.numBlocks = std::min<int64_t>(coreNum, best.units);
        best.unitsPerBlock = (best.units + best.numBlocks - 1) / best.numBlocks;
        best.offBlk = 0;
        best.offRaw = 0;
        best.offIdx = HAUp(L * esCT, 32);
        best.offUpd = best.offIdx + HAUp(L * idxBytes, 32);
        best.totalBytes = best.offUpd + HAUp(L * esCT, 32);
        if (best.totalBytes > static_cast<int64_t>(ubSize)) {
            best.totalBytes = static_cast<int64_t>(ubSize);
        }
    }
    return best;
}

/* ===================================================================== *
 * Launch wrapper (extern "C", called by the g++ plugin)
 * ===================================================================== */
namespace scat {

template <typename CT, typename IDX, typename LT>
void launch_mode(int64_t mode, GM_ADDR data, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                 int64_t outer, int64_t kLen, int64_t kuLen, int64_t inner, int64_t tileL,
                 int64_t tileK, int64_t ju, int64_t nL, int64_t nK, int64_t units,
                 int64_t unitsPerBlock, int64_t numBlocks, int64_t offBlk, int64_t offRaw,
                 int64_t offIdx, int64_t offUpd, int64_t totalBytes, void *stream)
{
#define SC_LAUNCH(M)                                                                               \
    scatter_kernel<CT, IDX, LT, M><<<numBlocks, nullptr, stream>>>(                                \
        data, indices, updates, y, outer, kLen, kuLen, inner, tileL, tileK, ju, nL, nK, units,     \
        unitsPerBlock, offBlk, offRaw, offIdx, offUpd, totalBytes)

    switch (mode) {
        case 0:
            SC_LAUNCH(0);
            break;
        case 1:
            SC_LAUNCH(1);
            break;
        case 2:
            SC_LAUNCH(2);
            break;
        case 3:
            SC_LAUNCH(3);
            break;
        default:
            SC_LAUNCH(4);
            break;
    }
#undef SC_LAUNCH
}

} // namespace scat

extern "C" {

void launch_scatter(GM_ADDR data, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, int64_t outer,
                    int64_t kLen, int64_t kuLen, int64_t inner, int64_t tileL, int64_t tileK,
                    int64_t ju, int64_t nL, int64_t nK, int64_t units, int64_t unitsPerBlock,
                    int64_t numBlocks, int64_t offBlk, int64_t offRaw, int64_t offIdx,
                    int64_t offUpd, int64_t totalBytes, int64_t dtypeCode, int64_t idxCode,
                    int64_t mode, void *stream)
{
#define SC_DISPATCH(CT, LT)                                                                        \
    do {                                                                                           \
        if (idxCode == 0) {                                                                        \
            scat::launch_mode<CT, int32_t, LT>(mode, data, indices, updates, y, outer, kLen,       \
                                               kuLen, inner, tileL, tileK, ju, nL, nK, units,      \
                                               unitsPerBlock, numBlocks, offBlk, offRaw, offIdx,   \
                                               offUpd, totalBytes, stream);                        \
        } else {                                                                                   \
            scat::launch_mode<CT, int64_t, LT>(mode, data, indices, updates, y, outer, kLen,       \
                                               kuLen, inner, tileL, tileK, ju, nL, nK, units,      \
                                               unitsPerBlock, numBlocks, offBlk, offRaw, offIdx,   \
                                               offUpd, totalBytes, stream);                        \
        }                                                                                          \
    } while (0)

    if (dtypeCode == SC_DT_F16 || dtypeCode == SC_DT_BF16) {
        // 16 bit: raw bit movement for overwrite, fp32 promotion for the arithmetic modes.
        if (mode == 0) {
            SC_DISPATCH(uint16_t, uint16_t);
        } else if (dtypeCode == SC_DT_F16) {
            SC_DISPATCH(float, half);
        } else {
            SC_DISPATCH(float, bfloat16_t);
        }
    } else if (dtypeCode == SC_DT_F32) {
        SC_DISPATCH(float, float);
    } else if (dtypeCode == SC_DT_I32) {
        SC_DISPATCH(int32_t, int32_t);
    } else {
        SC_DISPATCH(int64_t, int64_t);
    }
#undef SC_DISPATCH
}

} // extern "C"
