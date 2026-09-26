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
 * \file quant_matmul_epilogue_kernel.cpp
 * \brief QuantMatmul AIV stage (bisheng + -xasc, DAV_2201): generic dequant epilogue.
 *
 * Golden semantics (task/reference.py):
 *     mm = x1 @ x2                                  (int32, produced by the AIC kernel)
 *     if bias is int32:  mm = mm + bias             (pre-scale)
 *     y  = mm * scale
 *     if offset:         y  = y + offset
 *     if pertoken:       y  = y * pertoken_scale[m]
 *     if bias is float:  y  = y + bias              (post-scale)
 *     out = cast(y, float16 | bfloat16)
 *
 * float32 is the widest supported type, so the whole chain runs in float32 and only the final
 * narrowing cast rounds.
 *
 * Parallel layout: the row axis is split across the AIV blocks; each block walks its rows and
 * chunks the N axis with a UB tile. Per-channel vectors are re-read per row from (L2-cached)
 * GM, per-row scalars are fetched with a single-element DataCopyPad.
 *
 * This translation unit must NOT contain an AIC kernel (the build helper picks one kernel task
 * type per file); leaving it unmarked keeps the default AIV (Vector) task type.
 */

#include <cstdint>

#include "kernel_operator.h"

#include "quant_matmul_launch.h"

namespace qm {
static constexpr int32_t QM_EPI_TILE = 2048;   // elements per UB chunk on the N axis
static constexpr int32_t QM_AUX_BYTES = 128;   // scalar staging slot
}  // namespace qm

// scalar dtype codes for qm_read_scalar
constexpr int64_t QM_SK_F32 = 0;
constexpr int64_t QM_SK_I32 = 1;
constexpr int64_t QM_SK_BF16 = 2;
constexpr int64_t QM_SK_F16 = 3;

/*!
 * \brief Read one scalar from GM through UB and widen/normalize it to float32.
 */
__aicore__ inline float qm_read_scalar(GM_ADDR base, int64_t elemIdx, int64_t kind,
                                       AscendC::TQue<AscendC::QuePosition::VECIN, 1>& qAux,
                                       AscendC::TBuf<AscendC::QuePosition::VECCALC>& bTmp)
{
    const int64_t esize = (kind == QM_SK_BF16 || kind == QM_SK_F16) ? 2 : 4;
    AscendC::GlobalTensor<uint8_t> g;
    g.SetGlobalBuffer((__gm__ uint8_t*)base);
    auto raw = qAux.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(esize), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint8_t> pp{false, 0, 0, 0};
    AscendC::DataCopyPad(raw, g[elemIdx * esize], cp, pp);
    qAux.EnQue(raw);
    raw = qAux.DeQue<uint8_t>();
    auto dst = bTmp.Get<float>();
    if (kind == QM_SK_BF16) {
        AscendC::Cast(dst, raw.ReinterpretCast<bfloat16_t>(), AscendC::RoundMode::CAST_NONE, 8);
    } else if (kind == QM_SK_F16) {
        AscendC::Cast(dst, raw.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, 8);
    } else if (kind == QM_SK_I32) {
        AscendC::Cast(dst, raw.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, 8);
    } else {
        AscendC::Adds(dst, raw.ReinterpretCast<float>(), 0.0f, 8);
    }
    qAux.FreeTensor(raw);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return dst.GetValue(0);
}

/*!
 * \brief Generic dequant epilogue: int32 C -> (float16 | bfloat16) out.
 *
 * Loop order is column-chunk outer / row inner so that the per-channel scale / offset / bias
 * vectors of a chunk are read from GM exactly once per block instead of once per row (the naive
 * row-outer order re-read the whole scale vector for every row, doubling the epilogue traffic of
 * every per-channel case).
 */
template <typename OutT>
__global__ __aicore__ void qm_epi_kernel(GM_ADDR cIn, GM_ADDR out, GM_ADDR scaleGm,
                                         GM_ADDR offGm, GM_ADDR ptGm, GM_ADDR biasGm,
                                         qm::QmEpiArgs a)
{
    const int64_t blk = AscendC::GetBlockIdx();
    if (blk >= a.numBlocks) {
        return;
    }
    const int64_t rows = a.rows;
    const int64_t N = a.N;
    const int64_t rowsPer = (rows + a.numBlocks - 1) / a.numBlocks;
    const int64_t rBeg = blk * rowsPer;
    int64_t rEnd = rBeg + rowsPer;
    if (rEnd > rows) {
        rEnd = rows;
    }
    if (rBeg >= rEnd || N <= 0) {
        return;
    }

    const int64_t flags = a.flags;
    const bool scalePerChan = (flags & qm::QM_F_SCALE_PERCHAN) != 0;
    const bool scaleBf16 = (flags & qm::QM_F_SCALE_BF16) != 0;
    const bool hasOffset = (flags & qm::QM_F_HAS_OFFSET) != 0;
    const bool offsetPerChan = (flags & qm::QM_F_OFFSET_PERCHAN) != 0;
    const bool hasPt = (flags & qm::QM_F_HAS_PT) != 0;
    const int64_t biasKind = (flags & qm::QM_F_BIAS_MASK) >> qm::QM_F_BIAS_SHIFT;
    const bool biasPerChan = (flags & qm::QM_F_BIAS_PERCHAN) != 0;
    const bool ptModM = (flags & qm::QM_F_PT_MOD_M) != 0;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> qC;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qSt;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qY;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qAux;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bF;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bScl;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bOff;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bBia;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bSc;

    pipe.InitBuffer(qC, 2, static_cast<uint32_t>(qm::QM_EPI_TILE * 4));
    pipe.InitBuffer(qSt, 1, static_cast<uint32_t>(qm::QM_EPI_TILE * 4));
    pipe.InitBuffer(qY, 2, static_cast<uint32_t>(qm::QM_EPI_TILE * sizeof(OutT) + 64));
    pipe.InitBuffer(qAux, 1, static_cast<uint32_t>(qm::QM_AUX_BYTES));
    pipe.InitBuffer(bF, static_cast<uint32_t>(qm::QM_EPI_TILE * 4));
    pipe.InitBuffer(bScl, static_cast<uint32_t>(qm::QM_EPI_TILE * 4));
    pipe.InitBuffer(bOff, static_cast<uint32_t>(qm::QM_EPI_TILE * 4));
    pipe.InitBuffer(bBia, static_cast<uint32_t>(qm::QM_EPI_TILE * 4));
    pipe.InitBuffer(bSc, static_cast<uint32_t>(qm::QM_AUX_BYTES));

    AscendC::GlobalTensor<int32_t> cGm;
    AscendC::GlobalTensor<OutT> yGm;
    cGm.SetGlobalBuffer((__gm__ int32_t*)cIn);
    yGm.SetGlobalBuffer((__gm__ OutT*)out);

    auto fBuf = bF.Get<float>();

    // ---- block-invariant scalars -----------------------------------------------------------
    int64_t scaleKind = QM_SK_F32;
    if (scaleBf16) {
        scaleKind = QM_SK_BF16;
    }
    float scaleSc = 1.0f;
    if (!scalePerChan) {
        scaleSc = qm_read_scalar(scaleGm, 0, scaleKind, qAux, bSc);
    }
    float offSc = 0.0f;
    if (hasOffset && !offsetPerChan) {
        offSc = qm_read_scalar(offGm, 0, QM_SK_F32, qAux, bSc);
    }
    int64_t biasScalarKind = QM_SK_F32;
    if (biasKind == qm::QM_BIAS_I32) {
        biasScalarKind = QM_SK_I32;
    } else if (biasKind == qm::QM_BIAS_BF16) {
        biasScalarKind = QM_SK_BF16;
    } else if (biasKind == qm::QM_BIAS_F16) {
        biasScalarKind = QM_SK_F16;
    }
    float biasSc = 0.0f;
    if (biasKind != qm::QM_BIAS_NONE && !biasPerChan) {
        biasSc = qm_read_scalar(biasGm, 0, biasScalarKind, qAux, bSc);
    }

    const bool biasIsI32 = (biasKind == qm::QM_BIAS_I32);
    const bool biasIsFloat = (biasKind == qm::QM_BIAS_F16 || biasKind == qm::QM_BIAS_BF16 ||
                              biasKind == qm::QM_BIAS_F32);
    const bool biasPerChanActive = biasPerChan && (biasIsI32 || biasIsFloat);

    for (int64_t c0 = 0; c0 < N; c0 += qm::QM_EPI_TILE) {
        int64_t cnt = N - c0;
        if (cnt > qm::QM_EPI_TILE) {
            cnt = qm::QM_EPI_TILE;
        }
        const int32_t n = static_cast<int32_t>(cnt);

        // ---- per-channel vectors of this chunk, read once and reused for every row ----------
        if (scalePerChan) {
            auto st = qSt.AllocTensor<uint8_t>();
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * (scaleBf16 ? 2 : 4)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> pp{false, 0, 0, 0};
            AscendC::GlobalTensor<uint8_t> sg;
            sg.SetGlobalBuffer((__gm__ uint8_t*)scaleGm);
            AscendC::DataCopyPad(st, sg[c0 * (scaleBf16 ? 2 : 4)], cp, pp);
            qSt.EnQue(st);
            st = qSt.DeQue<uint8_t>();
            auto dst = bScl.Get<float>();
            if (scaleBf16) {
                AscendC::Cast(dst, st.ReinterpretCast<bfloat16_t>(), AscendC::RoundMode::CAST_NONE, n);
            } else {
                AscendC::Adds(dst, st.ReinterpretCast<float>(), 0.0f, n);
            }
            qSt.FreeTensor(st);
        }
        if (hasOffset && offsetPerChan) {
            auto st = qSt.AllocTensor<uint8_t>();
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * 4), 0, 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> pp{false, 0, 0, 0};
            AscendC::GlobalTensor<uint8_t> og;
            og.SetGlobalBuffer((__gm__ uint8_t*)offGm);
            AscendC::DataCopyPad(st, og[c0 * 4], cp, pp);
            qSt.EnQue(st);
            st = qSt.DeQue<uint8_t>();
            auto dst = bOff.Get<float>();
            AscendC::Adds(dst, st.ReinterpretCast<float>(), 0.0f, n);
            qSt.FreeTensor(st);
        }
        if (biasPerChanActive) {
            const int64_t eb = (biasKind == qm::QM_BIAS_BF16 || biasKind == qm::QM_BIAS_F16) ? 2 : 4;
            auto st = qSt.AllocTensor<uint8_t>();
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * eb), 0, 0, 0};
            AscendC::DataCopyPadExtParams<uint8_t> pp{false, 0, 0, 0};
            AscendC::GlobalTensor<uint8_t> bg;
            bg.SetGlobalBuffer((__gm__ uint8_t*)biasGm);
            AscendC::DataCopyPad(st, bg[c0 * eb], cp, pp);
            qSt.EnQue(st);
            st = qSt.DeQue<uint8_t>();
            auto dst = bBia.Get<float>();
            if (biasKind == qm::QM_BIAS_I32) {
                AscendC::Cast(dst, st.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, n);
            } else if (biasKind == qm::QM_BIAS_BF16) {
                AscendC::Cast(dst, st.ReinterpretCast<bfloat16_t>(), AscendC::RoundMode::CAST_NONE, n);
            } else if (biasKind == qm::QM_BIAS_F16) {
                AscendC::Cast(dst, st.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, n);
            } else {
                AscendC::Adds(dst, st.ReinterpretCast<float>(), 0.0f, n);
            }
            qSt.FreeTensor(st);
        }

        for (int64_t r = rBeg; r < rEnd; ++r) {
            float ptSc = 1.0f;
            if (hasPt) {
                int64_t ptIdx = r;
                if (ptModM && a.M > 0) {
                    ptIdx = r % a.M;
                }
                ptSc = qm_read_scalar(ptGm, ptIdx, QM_SK_F32, qAux, bSc);
            }

            // ---- C tile ---------------------------------------------------------------------
            auto ct = qC.AllocTensor<int32_t>();
            AscendC::DataCopyExtParams cpC{1, static_cast<uint32_t>(cnt * 4), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> ppC{false, 0, 0, 0};
            AscendC::DataCopyPad(ct, cGm[r * N + c0], cpC, ppC);
            qC.EnQue(ct);
            ct = qC.DeQue<int32_t>();

            AscendC::Cast(fBuf, ct, AscendC::RoundMode::CAST_NONE, n);
            qC.FreeTensor(ct);

            // ---- pre-scale integer bias ------------------------------------------------------
            if (biasIsI32) {
                if (biasPerChanActive) {
                    AscendC::Add(fBuf, fBuf, bBia.Get<float>(), n);
                } else {
                    AscendC::Adds(fBuf, fBuf, biasSc, n);
                }
            }

            // ---- scale ----------------------------------------------------------------------
            if (scalePerChan) {
                AscendC::Mul(fBuf, fBuf, bScl.Get<float>(), n);
            } else {
                AscendC::Muls(fBuf, fBuf, scaleSc, n);
            }

            // ---- offset (post-scale add) -----------------------------------------------------
            if (hasOffset) {
                if (offsetPerChan) {
                    AscendC::Add(fBuf, fBuf, bOff.Get<float>(), n);
                } else {
                    AscendC::Adds(fBuf, fBuf, offSc, n);
                }
            }

            // ---- pertoken (per row) ----------------------------------------------------------
            if (hasPt) {
                AscendC::Muls(fBuf, fBuf, ptSc, n);
            }

            // ---- post-scale float bias -------------------------------------------------------
            if (biasIsFloat) {
                if (biasPerChanActive) {
                    AscendC::Add(fBuf, fBuf, bBia.Get<float>(), n);
                } else {
                    AscendC::Adds(fBuf, fBuf, biasSc, n);
                }
            }

            // ---- narrowing cast + store ------------------------------------------------------
            auto yt = qY.AllocTensor<OutT>();
            AscendC::Cast(yt, fBuf, AscendC::RoundMode::CAST_RINT, n);
            qY.EnQue(yt);
            AscendC::DataCopyExtParams cpY{1, static_cast<uint32_t>(cnt * sizeof(OutT)), 0, 0, 0};
            yt = qY.DeQue<OutT>();
            AscendC::DataCopyPad(yGm[r * N + c0], yt, cpY);
            qY.FreeTensor(yt);
        }
    }
}

extern "C" void qm_launch_epilogue_f16(GM_ADDR cIn, GM_ADDR out, GM_ADDR scale, GM_ADDR offset,
                                       GM_ADDR pt, GM_ADDR bias, const qm::QmEpiArgs* args,
                                       void* stream)
{
    qm_epi_kernel<half><<<static_cast<uint32_t>(args->numBlocks), nullptr, stream>>>(
        cIn, out, scale, offset, pt, bias, *args);
}

extern "C" void qm_launch_epilogue_bf16(GM_ADDR cIn, GM_ADDR out, GM_ADDR scale, GM_ADDR offset,
                                        GM_ADDR pt, GM_ADDR bias, const qm::QmEpiArgs* args,
                                        void* stream)
{
    qm_epi_kernel<bfloat16_t><<<static_cast<uint32_t>(args->numBlocks), nullptr, stream>>>(
        cIn, out, scale, offset, pt, bias, *args);
}
