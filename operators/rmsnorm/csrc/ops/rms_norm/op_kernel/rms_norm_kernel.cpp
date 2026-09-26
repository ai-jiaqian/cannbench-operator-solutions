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
 * \file rms_norm_kernel.cpp
 * \brief RmsNorm PADDED-PITCH FLAT-COMBINED-FIELD kernel + tiling + launch (bisheng + -xasc)
 *
 *   y = x / sqrt(mean(x^2) + eps) * gamma          (x: (...,D), gamma: (D,))
 *
 * Frozen z26 PADDED-PITCH FLAT-COMBINED-FIELD structure: ONE kernel launch per call.
 * There is NO packed path and NO z10 fallback: EVERY shape uses the padded-pitch binding.
 * Each core owns a contiguous row range [c*blockRows, min((c+1)*blockRows, numRow)) and
 * walks it in padded-pitch row-block tiles of tileRows rows.  The per-row statistic is
 * computed and consumed on the resident block and is NEVER written to global memory; no
 * statistic workspace exists.
 *
 * On-chip pitch (derived.pitch, pure function of D/elemSize/theta.pitch_align):
 *   P32 = align32(D*elemSize)/elemSize
 *   P   = P32 when theta.pitch_align == 32, else align64(P32)
 * Each row occupies P elements on chip; the real D elements sit at the start of the pitch
 * slot and the remaining P-D lanes are pad.  Pad lanes are never stored.
 *
 * Per tile of R consecutive rows:
 *   (1) load : ONE multi-block DataCopyPad (blockCount=rows, blockLen=D*elemSize,
 *              srcStride=0, dstStride=gapBlocks) placing row r at on-chip offset r*P.
 *   (2) widen: 16-bit input widened once to a single fp32 work representation
 *              (Cast CAST_NONE) over rows*P elements; fp32 input aliases the input tile.
 *   (3) square: one flat Mul into a SEPARATE squared fp32 buffer for 16-bit input, or
 *              into the output tile for fp32 input.  It is NEVER written in place into
 *              the work tile that holds x.
 *   (4) reduce: per-row sum of squares over the REAL column count D on the padded block,
 *              method chosen by the shape-only rule (segmented WholeReduceSum when
 *              P <= 384, else one ReduceSum per row).  Pad lanes are never read.
 *   (5) rstd  : Muls(invD) -> Adds(epsilon) -> Sqrt -> Div(ones, .) on rows*8 lanes,
 *              NO positive-denominator guard.
 *   (6) field : flat per-element COMBINED scale field F of rows*P fp32 over the padded
 *              block, F[r,i] = rstd[r]*gamma[i], method chosen by the shape-only rule
 *              (flat Brcb + column-chunk gamma multiply when
 *              14*ceil(rows/64)*(ceil(P/64)+1) < 13*rows, else one Muls per row).  The
 *              shared gamma is folded in here; there is NO separate gamma fold pass.
 *   (7) apply : ONE flat Mul(yF, xF, F, rows*P) over the whole padded block.
 *   (8) narrow: 16-bit output narrowed with Cast CAST_RINT over rows*P elements.
 *   (9) store : ONE multi-block DataCopyPad (blockCount=rows, blockLen=D*elemSize,
 *              srcStride=gapBlocks, dstStride=0) storing only the real D elements of
 *              each row.  Pad lanes are never stored.
 *
 * theta bindings realized here:
 *   tile_rows_cap = 1024 (kMaxTileRows)
 *   pitch_align   = 32   (kPitchAlign)
 *
 * Numerics (matching torch F.rms_norm, which accumulates in fp32):
 *   - fp32 accumulation for every dtype; a single down-cast at the end.
 *   - rstd = 1.0f / sqrt(mean(x^2) + eps): a plain IEEE divide of 1.0 by the sqrt, so
 *     1/+inf is exactly 0 while 1/NaN stays NaN.  There is deliberately NO
 *     positive-denominator guard.
 *   - the combined field is one fp32 rounding of the product rstd*gamma and the apply is
 *     one fp32 multiply per element, preserving the exact per-element product semantics.
 *
 * UB budget note: tiling is derived from exactly the buffers allocated below and the host
 * tiling function mirrors the very same pitch and tile-height rule.
 */

#include <tuple>
#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

namespace rms_norm_detail {

constexpr int32_t kQueDepth = 2;
constexpr int32_t kReduceTmpBytes = 8192;
// derived.tile_rows: fixed UB size of the target Ascend910B2.  The tiling budget is a
// pure function of the shape/dtype built on this constant, so host and kernel agree.
constexpr uint32_t kUbSize = 196608u;
constexpr int32_t kSlotFloats = 8;             // 8 fp32 = 32B per row
// theta.tile_rows_cap: upper clamp on the number of rows processed per tile.
constexpr int64_t kMaxTileRows = 1024;
// WholeReduceSum processes one 256B (64 fp32) block per repeat; row-chunks are
// capped at 255 repeats, the documented maximum for the repeat field.
constexpr int32_t kSegChunkElems = 64;
constexpr int32_t kSegMaxRepeats = 255;
// Flat-field row chunk: one Brcb per <=64 rows.
constexpr int32_t kFieldChunkRows = 64;
// 64 rows * 8 blocks * 32B = 16384 B flat-field row-chunk scratch.
constexpr uint32_t kRblkBytes = 64u * 8u * 32u;
// derived.reduction: enable the segmented per-row reduction when the padded pitch P <= this.
constexpr int32_t kSegMaxPitch = 384;
// theta.pitch_align: on-chip row-pitch alignment in bytes (32 or 64).
constexpr int32_t kPitchAlign = 32;

__aicore__ inline uint32_t AlignUp32(uint32_t v)
{
    return (v + 31u) & ~31u;
}

__aicore__ inline uint32_t AlignUp64(uint32_t v)
{
    return (v + 63u) & ~63u;
}

__aicore__ inline int64_t TileRowsClamped(int64_t myRows, int64_t tileRowsLocal, int64_t t)
{
    int64_t rowsThisTile = myRows - t * tileRowsLocal;
    if (rowsThisTile > tileRowsLocal) {
        rowsThisTile = tileRowsLocal;
    }
    return rowsThisTile;
}

} // namespace rms_norm_detail

// ---------------------------------------------------------------------------
// Fused single-pass RmsNorm kernel (padded-pitch flat-combined-field binding)
// ---------------------------------------------------------------------------
template <typename T>
__global__ __aicore__ void rms_norm_kernel(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                           int64_t numRow, int64_t numCol,
                                           float epsilon, float invD,
                                           int64_t blockRows, int64_t tileRows)
{
    using namespace rms_norm_detail;
    constexpr bool kIsF32 = std::is_same<T, float>::value;

    const int64_t rowBegin = static_cast<int64_t>(AscendC::GetBlockIdx()) * blockRows;
    if (rowBegin >= numRow) {
        return;
    }
    int64_t rowEnd = rowBegin + blockRows;
    if (rowEnd > numRow) {
        rowEnd = numRow;
    }
    const int64_t myRows = rowEnd - rowBegin;

    const int32_t numColI = static_cast<int32_t>(numCol);
    const uint32_t elemSize = static_cast<uint32_t>(sizeof(T));
    const uint32_t realRowBytes = static_cast<uint32_t>(numCol) * elemSize;

    // derived.pitch: pure function of D, elemSize and theta.pitch_align; mirrored exactly
    // by the host tiling.  P = align32(D*elemSize)/elemSize, optionally 64-aligned.
    const uint32_t p32 = AlignUp32(realRowBytes) / elemSize;
    const uint32_t rowPitch = (kPitchAlign == 32) ? p32 : AlignUp64(p32);
    const uint32_t rowBytesPad = rowPitch * elemSize;
    const uint32_t rowF32Bytes = rowPitch * 4u;
    // DataCopyPad pads each blockLen up to 32B, so the destination row pitch is
    // align32(D*elemSize) + gapBlocks*32; the gap is measured from the 32B-aligned real row.
    const uint32_t gapBlocks = (rowBytesPad - AlignUp32(realRowBytes)) / 32u;
    const int32_t pitchI = static_cast<int32_t>(rowPitch);

    // derived.tile_rows UB-feasibility guard: the flat-field row-chunk scratch (rblkBuf,
    // 16384 B) is allocated only when a one-row plan that includes it fits the UB budget.
    // When it does not fit, the scratch is dropped and the flat-field mechanism is forced
    // off (see derived.combined_field).  This is a pure function of D/elemSize and is
    // mirrored exactly by calc_rms_norm_tiling_params.  Without it, case 18 (D=8192 bf16)
    // overflowed UB and faulted with "VEC instruction error: the ub address out of bounds".
    const uint32_t perRowBytes = 4u * rowBytesPad + 96u +
        ((elemSize != 4u) ? (2u * rowF32Bytes) : 0u);
    uint32_t fixedWithBytes = rowF32Bytes + static_cast<uint32_t>(kReduceTmpBytes) + kRblkBytes;
    if (elemSize != 4u) {
        fixedWithBytes += rowBytesPad;
    }
    const bool rblkAlloc = (perRowBytes + fixedWithBytes) <= kUbSize;

    // Shape-only selection rules (no model constants).
    const int32_t numChunks = (numColI + kSegChunkElems - 1) / kSegChunkElems;
    const int32_t chunksP = (pitchI + kSegChunkElems - 1) / kSegChunkElems;

    int64_t tileRowsLocal = tileRows;
    if (tileRowsLocal > myRows) {
        tileRowsLocal = myRows;
    }
    if (tileRowsLocal < 1) {
        tileRowsLocal = 1;
    }

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> gammaGm;
    AscendC::GlobalTensor<T> yGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    AscendC::TQue<AscendC::QuePosition::VECIN, kQueDepth> inQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQueDepth> outQ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> fBuf;     // fp32 work tile (16-bit only)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sqBuf;    // fp32 squared tile (16-bit only)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gBuf;     // fp32 gamma row
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gRawBuf;  // raw gamma row (16-bit only)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> slotBuf;  // per-row sum / rstd slots
    AscendC::TBuf<AscendC::QuePosition::VECCALC> oneBuf;   // constant 1.0 lanes
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;   // reduction scratch
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rblkBuf;  // flat-field row-chunk scratch

    const uint32_t slotBytes = static_cast<uint32_t>(tileRowsLocal) * kSlotFloats * sizeof(float);
    const uint32_t tileBytes = static_cast<uint32_t>(tileRowsLocal) * rowBytesPad;
    pipe.InitBuffer(inQ, kQueDepth, tileBytes);
    pipe.InitBuffer(outQ, kQueDepth, tileBytes);
    pipe.InitBuffer(gBuf, rowF32Bytes);
    pipe.InitBuffer(slotBuf, slotBytes);
    pipe.InitBuffer(oneBuf, slotBytes);
    pipe.InitBuffer(tmpBuf, kReduceTmpBytes);
    // derived.tile_rows: allocate the flat-field row-chunk scratch only when it fits UB.
    if (rblkAlloc) {
        pipe.InitBuffer(rblkBuf, kRblkBytes);
    }
    if constexpr (!kIsF32) {
        // 16-bit: x and x^2 both need an fp32 tile, and the squared tile must not
        // alias the work tile.
        pipe.InitBuffer(fBuf, static_cast<uint32_t>(tileRowsLocal) * rowF32Bytes);
        pipe.InitBuffer(sqBuf, static_cast<uint32_t>(tileRowsLocal) * rowF32Bytes);
        pipe.InitBuffer(gRawBuf, rowBytesPad);
    }

    // gamma is loaded once per core and widened to fp32 for the 16-bit dtypes.
    auto gammaF = gBuf.Get<float>();
    if constexpr (kIsF32) {
        AscendC::DataCopyExtParams gammaParams{1, static_cast<uint32_t>(numCol) * 4u, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> gammaPad{false, 0, 0, 0};
        AscendC::DataCopyPad(gammaF, gammaGm, gammaParams, gammaPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    } else {
        auto gammaRaw = gRawBuf.Get<T>();
        AscendC::DataCopyExtParams gammaParams{1, static_cast<uint32_t>(numCol) * elemSize, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> gammaPad{false, 0, 0, 0};
        AscendC::DataCopyPad(gammaRaw, gammaGm, gammaParams, gammaPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(gammaF, gammaRaw, AscendC::RoundMode::CAST_NONE, numColI);
    }

    auto slots = slotBuf.Get<float>();
    auto ones = oneBuf.Get<float>();
    auto reduceTmp = tmpBuf.Get<float>();
    const int32_t slotCap = static_cast<int32_t>(tileRowsLocal) * kSlotFloats;
    AscendC::Duplicate(ones, 1.0f, slotCap);

    const int64_t numTiles = (myRows + tileRowsLocal - 1) / tileRowsLocal;

    AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, 0};

    // ---------------- software pipeline prologue: issue the load of tile 0 --------
    {
        auto xT0 = inQ.AllocTensor<T>();
        AscendC::DataCopyExtParams copyIn{
            static_cast<uint16_t>(tileRowsLocal), realRowBytes, 0, gapBlocks, 0};
        AscendC::DataCopyPad(xT0, xGm[rowBegin * numCol], copyIn, padIn);
        inQ.EnQue(xT0);
    }

    for (int64_t t = 0; t < numTiles; ++t) {
        const int64_t rowStart = rowBegin + t * tileRowsLocal;
        const int32_t rows = static_cast<int32_t>(TileRowsClamped(myRows, tileRowsLocal, t));
        const int32_t tileElems = rows * pitchI;

        // ---------------- phase 1: take this tile's data ----------------------
        auto xTile = inQ.DeQue<T>();

        // Issue the NEXT tile's MTE2 right away: it then runs concurrently with the
        // whole body of this tile, which is what makes the double buffering overlap.
        if (t + 1 < numTiles) {
            const int64_t nextStart = rowStart + tileRowsLocal;
            const int64_t nextRows = TileRowsClamped(myRows, tileRowsLocal, t + 1);
            auto xTn = inQ.AllocTensor<T>();
            AscendC::DataCopyExtParams copyIn{
                static_cast<uint16_t>(nextRows), realRowBytes, 0, gapBlocks, 0};
            AscendC::DataCopyPad(xTn, xGm[nextStart * numCol], copyIn, padIn);
            inQ.EnQue(xTn);
        }

        // ---------------- phase 2: widen to fp32 (one call for the whole tile) ----
        AscendC::LocalTensor<float> xF;
        if constexpr (kIsF32) {
            xF = xTile;
        } else {
            xF = fBuf.Get<float>();
            AscendC::Cast(xF, xTile, AscendC::RoundMode::CAST_NONE, tileElems);
            inQ.FreeTensor(xTile);
        }

        auto yTile = outQ.AllocTensor<T>();
        // squared-tile scratch: fp32 reuses the output tile (same byte size / pitch),
        // 16-bit uses the dedicated fp32 buffer
        AscendC::LocalTensor<float> sqTile;
        if constexpr (kIsF32) {
            sqTile = yTile;
        } else {
            sqTile = sqBuf.Get<float>();
        }

        // ---------------- phase 3: flat square + per-row reduce --------------
        AscendC::Mul(sqTile, xF, xF, tileElems);
        const int32_t slotCount = rows * kSlotFloats;

        // derived.reduction: segmented WholeReduceSum when P <= kSegMaxPitch, else one
        // ReduceSum per row.  The mask is the REAL column count D, never a padded pitch,
        // so UB padding lanes are never read.  srcRepStride = P/8 (32B blocks) walks rows.
        if (pitchI <= kSegMaxPitch) {
            // Segmented per-row reduction: row-chunks outer, column-chunks inner.
            // Each 64-element chunk of every row is reduced by one WholeReduceSum issued
            // over up to 255 rows at once; the chunk scratch is the fixed tmpBuf.
            for (int32_t r0 = 0; r0 < rows; r0 += kSegMaxRepeats) {
                int32_t c = rows - r0;
                if (c > kSegMaxRepeats) {
                    c = kSegMaxRepeats;
                }
                for (int32_t j = 0; j < numChunks; ++j) {
                    const int32_t remaining = numColI - j * kSegChunkElems;
                    const int32_t maskJ = (remaining < kSegChunkElems) ? remaining : kSegChunkElems;
                    if (j == 0) {
                        AscendC::WholeReduceSum<float>(
                            slots[static_cast<int64_t>(r0) * kSlotFloats],
                            sqTile[static_cast<int64_t>(r0) * pitchI + j * kSegChunkElems],
                            maskJ, c, kSlotFloats, 1, pitchI / 8);
                    } else {
                        AscendC::WholeReduceSum<float>(
                            reduceTmp,
                            sqTile[static_cast<int64_t>(r0) * pitchI + j * kSegChunkElems],
                            maskJ, c, kSlotFloats, 1, pitchI / 8);
                        AscendC::Add(slots[static_cast<int64_t>(r0) * kSlotFloats],
                                     slots[static_cast<int64_t>(r0) * kSlotFloats],
                                     reduceTmp, c * kSlotFloats);
                    }
                }
            }
        } else {
            // per-row path: one ReduceSum per row over the REAL D columns.
            for (int32_t r = 0; r < rows; ++r) {
                AscendC::ReduceSum<float>(slots[static_cast<int64_t>(r) * kSlotFloats],
                                          sqTile[static_cast<int64_t>(r) * pitchI],
                                          reduceTmp, numColI);
            }
        }

        // ---------------- phase 4: mean -> +eps -> sqrt -> reciprocal ---------
        AscendC::Muls(slots, slots, invD, slotCount);
        AscendC::Adds(slots, slots, epsilon, slotCount);
        AscendC::Sqrt(slots, slots, slotCount);
        AscendC::Div(slots, ones, slots, slotCount);
        AscendC::LocalTensor<float> rstdSrc = slots;

        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

        // ---------------- phase 5: materialize the COMBINED scale field ------
        // The flat per-element field F holds the COMBINED scale F[r,i] = rstd[r]*gamma[i]
        // over the padded pitch.  The shared gamma is folded in here, so there is no
        // separate full-tile gamma fold pass.  F aliases the output tile for fp32 and the
        // squared tile for 16-bit (both are free fp32 tile buffers after the reduction).
        AscendC::LocalTensor<float> F;
        if constexpr (kIsF32) {
            F = yTile;
        } else {
            F = sqTile;
        }
        const int32_t fieldRowChunks = (rows + kFieldChunkRows - 1) / kFieldChunkRows;
        // derived.combined_field shape-only rule: flat iff the rblkBuf scratch is allocated
        // AND 14*ceil(rows/64)*(chunksP+1) < 13*rows.  When rblkAlloc is false the flat
        // mechanism must not be selected (its scratch does not exist).
        const bool useFlatField = rblkAlloc &&
            (14 * fieldRowChunks * (chunksP + 1) < 13 * rows);
        if (useFlatField) {
            // Flat expansion: one Brcb per <=64-row chunk materializes block 8r =
            // [rstd[r]]*8, then one low-level Mul per column chunk combines it with the
            // shared gamma chunk over the padded pitch.  src0 = gammaF + j*64 is read with
            // src0BlkStride = 1 and src0RepStride = 0 (the same gamma column chunk is
            // reused for every row repeat), and src1 = rblk is read with src1BlkStride = 0
            // / src1RepStride = 8 blocks (block 8r of rblk is broadcast across the
            // 64-element repeat).  dstRepStride = P/8 walks the padded rows.
            auto rblk = rblkBuf.Get<float>();
            for (int32_t r0 = 0; r0 < rows; r0 += kFieldChunkRows) {
                int32_t c = rows - r0;
                if (c > kFieldChunkRows) {
                    c = kFieldChunkRows;
                }
                AscendC::Brcb(rblk, rstdSrc[static_cast<int64_t>(r0) * kSlotFloats],
                              static_cast<uint8_t>(c), AscendC::BrcbRepeatParams{1, 8});
                for (int32_t j = 0; j < chunksP; ++j) {
                    const int32_t remaining = pitchI - j * kSegChunkElems;
                    const int32_t maskJ = (remaining < kSegChunkElems) ? remaining : kSegChunkElems;
                    AscendC::Mul(
                        F[static_cast<int64_t>(r0) * pitchI + j * kSegChunkElems],
                        gammaF[j * kSegChunkElems],
                        rblk, maskJ, static_cast<uint8_t>(c),
                        AscendC::BinaryRepeatParams{1, 1, 0,
                            static_cast<uint8_t>(pitchI / 8), 0, 8});
                }
            }
        } else {
            // per-row mechanism: one vector-scalar multiply of the shared gamma row by
            // the row's rstd produces the combined scale in one pass over the padded pitch.
            for (int32_t r = 0; r < rows; ++r) {
                AscendC::Muls(F[static_cast<int64_t>(r) * pitchI], gammaF,
                              rstdSrc.GetValue(static_cast<int32_t>(r * kSlotFloats)),
                              pitchI);
            }
        }

        // ---------------- phase 6: apply -------------------------------------
        AscendC::LocalTensor<float> yF;
        if constexpr (kIsF32) {
            yF = yTile;     // fp32 writes the separate output tile
        } else {
            yF = xF;        // 16-bit writes the work tile, narrowed afterwards
        }
        AscendC::Mul(yF, xF, F, tileElems);

        // ---------------- phase 7: narrow 16-bit output -----------------------
        if constexpr (!kIsF32) {
            AscendC::Cast(yTile, xF, AscendC::RoundMode::CAST_RINT, tileElems);
        }
        if constexpr (kIsF32) {
            inQ.FreeTensor(xTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);

        // ---------------- phase 8: store --------------------------------------
        outQ.EnQue(yTile);
        yTile = outQ.DeQue<T>();
        {
            AscendC::DataCopyExtParams copyOut{
                static_cast<uint16_t>(rows), realRowBytes, gapBlocks, 0, 0};
            AscendC::DataCopyPad(yGm[rowStart * numCol], yTile, copyOut);
        }
        outQ.FreeTensor(yTile);
    }
}

// ---------------------------------------------------------------------------
// Host side tiling: mirrors the kernel's UB allocation exactly.
// ---------------------------------------------------------------------------
std::tuple<int64_t, int64_t, int64_t> calc_rms_norm_tiling_params(int64_t numRow, int64_t numCol,
                                                                  int64_t elemSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    // derived.tile_rows: the UB size is the fixed target constant, so the host and the
    // kernel share the same pure function of shape/dtype.
    const int64_t ubSize = static_cast<int64_t>(rms_norm_detail::kUbSize);
    int64_t coreNum = (ascendcPlatform != nullptr) ? ascendcPlatform->GetCoreNumAiv() : 1;
    if (coreNum <= 0) {
        coreNum = 1;
    }
    if (numRow <= 0 || numCol <= 0) {
        return std::make_tuple<int64_t, int64_t, int64_t>(1, 1, 1);
    }

    int64_t numBlocks = std::min(coreNum, numRow);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t blockRows = (numRow + numBlocks - 1) / numBlocks;
    if (blockRows < 1) {
        blockRows = 1;
    }

    const int64_t D = numCol;
    // derived.pitch: mirrored exactly by the kernel.
    const int64_t p32 = ((D * elemSize + 31) / 32 * 32) / elemSize;
    const int64_t P = (rms_norm_detail::kPitchAlign == 32) ? p32 : ((p32 + 63) / 64 * 64);
    const int64_t rowF32Bytes = P * 4;
    const int64_t rowBytesPad = P * elemSize;

    // derived.tile_rows: per tile the kernel holds inQ depth2 + outQ depth2 (T elements),
    // the per-row slots and ones (8 fp32 each) and, for 16-bit, the fp32 work tile and the
    // fp32 squared tile.  Once per core it holds the fp32 gamma, the ReduceSum scratch and
    // the flat-field row-chunk scratch, plus the raw gamma staging for 16-bit.
    const int64_t perRow = 4 * rowBytesPad + 96 + ((elemSize != 4) ? (2 * rowF32Bytes) : 0);
    // derived.tile_rows UB-feasibility guard, mirrored exactly by the kernel: keep the
    // 16384 B flat-field row-chunk scratch only when a one-row plan that includes it fits.
    int64_t fixedWith = rowF32Bytes + 8192 + static_cast<int64_t>(rms_norm_detail::kRblkBytes);
    if (elemSize != 4) {
        fixedWith += rowBytesPad;
    }
    const bool rblkAlloc = (perRow + fixedWith) <= ubSize;
    int64_t fixed = rblkAlloc ? fixedWith
                              : (fixedWith - static_cast<int64_t>(rms_norm_detail::kRblkBytes));

    int64_t budget = static_cast<int64_t>(ubSize) - 8192;
    int64_t tileRows = 1;
    if (budget > fixed && perRow > 0) {
        tileRows = (budget - fixed) / perRow;
    }
    if (tileRows < 1) {
        tileRows = 1;
    }
    if (tileRows > rms_norm_detail::kMaxTileRows) {
        tileRows = rms_norm_detail::kMaxTileRows;
    }
    if (tileRows > blockRows) {
        tileRows = blockRows;
    }
    if (tileRows < 1) {
        tileRows = 1;
    }
    return std::make_tuple(numBlocks, blockRows, tileRows);
}

// ---------------------------------------------------------------------------
// Launch wrappers - plain C entry points callable from the g++ plugin layer.
// ---------------------------------------------------------------------------
extern "C" {

void launch_rms_norm_kernel_float(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t numRow, int64_t numCol, float epsilon, float invD,
    int64_t numBlocks, int64_t blockRows, int64_t tileRows, void* stream)
{
    rms_norm_kernel<float><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(
        x, gamma, y, numRow, numCol, epsilon, invD, blockRows, tileRows);
}

void launch_rms_norm_kernel_half(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t numRow, int64_t numCol, float epsilon, float invD,
    int64_t numBlocks, int64_t blockRows, int64_t tileRows, void* stream)
{
    rms_norm_kernel<half><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(
        x, gamma, y, numRow, numCol, epsilon, invD, blockRows, tileRows);
}

void launch_rms_norm_kernel_bfloat16(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
    int64_t numRow, int64_t numCol, float epsilon, float invD,
    int64_t numBlocks, int64_t blockRows, int64_t tileRows, void* stream)
{
    rms_norm_kernel<bfloat16_t><<<static_cast<uint32_t>(numBlocks), nullptr, stream>>>(
        x, gamma, y, numRow, numCol, epsilon, invD, blockRows, tileRows);
}

} // extern "C"
