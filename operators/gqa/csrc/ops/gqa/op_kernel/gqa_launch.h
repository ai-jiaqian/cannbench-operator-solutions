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
 * \file gqa_launch.h
 * \brief Plain-C++ declarations shared between the bisheng-compiled kernel translation units and the
 *        g++ compiled plugin translation unit.  No AscendC type may appear here.
 *
 * GQA grouping:
 *   q   [B, S, Nq, D]      k/v [B, Skv, Nkv, D]      G = Nq / Nkv
 *   out[b, i, h, :] = softmax_j( q[b,i,h,:] . k[b,j,h/G,:] * scale ) * v[b,j,h/G,:]
 *
 * Two equivalent formulations are provided; the host picks the cheaper one from a bandwidth model.
 *
 * PACKED pipeline (used when the score matrix is small relative to K/V):
 *   stage 1 (AIV)  : pack  q/k/v into per (batch, kv-head) contiguous matrices
 *                    Qp[bg] = [G*S, D] (row r = i*S + s), Kp[bg] = [Skv, D], Vp[bg] = [Skv, D]
 *   stage 2 (AIC)  : scores[bg] = Qp[bg] @ Kp[bg]^T                     [G*S, Skv]
 *   stage 3 (AIV)  : scaled softmax with right-bottom aligned causal mask, in place
 *   stage 4 (AIC)  : Yp[bg] = P[bg] @ Vp[bg]                            [G*S, D]
 *   stage 5 (AIV)  : scatter Yp back into y[b, s, g*G + i, :]
 *   traffic = 3Q + 3K + 3V + 3S + 2Y
 *
 * FLAT pipeline (used when the expanded score matrix is small relative to K/V - decode / MTP shapes):
 *   stage 2 (AIC)  : scores[b] = Q[b] @ K[b]^T          [S*Nq, Skv*Nkv]   (native layouts, no pack)
 *   stage 3 (AIV)  : softmax over the columns j' % Nkv == h/G only
 *   stage 4 (AIC)  : y[b] = P[b] @ V[b]                 [S*Nq, D]         (native output layout)
 *   traffic = Q + K + V + 3*Nkv*S + Y
 * The expanded score matrix costs Nkv times more bytes but saves two full passes over K and V plus
 * the query repack and the output permutation, which dominates whenever S is small.
 */

#ifndef GQA_LAUNCH_H
#define GQA_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

/* ---------------------------------------------------------------------------
 * Layout independent transport of the Matmul tiling.
 *
 * The host builds optiling::TCubeTiling with the Matmul Tiling API, the device side needs
 * AscendC::tiling::TCubeTiling.  They are different types, so the values are carried through
 * this member-name-identical POD and handed over as plain kernel scalars.
 * ------------------------------------------------------------------------- */
struct GqaCubeTiling {
    int32_t usedCoreNum;
    int32_t M;
    int32_t N;
    int32_t Ka;
    int32_t Kb;
    int32_t singleCoreM;
    int32_t singleCoreN;
    int32_t singleCoreK;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t depthA1;
    int32_t depthB1;
    int32_t stepM;
    int32_t stepN;
    int32_t isBias;
    int32_t transLength;
    int32_t iterateOrder;
    int32_t shareMode;
    int32_t shareL1Size;
    int32_t shareL0CSize;
    int32_t shareUbSize;
    int32_t batchM;
    int32_t batchN;
    int32_t batchNum;
    int32_t stepKa;
    int32_t stepKb;
    int32_t pad0;
    int32_t pad1;
};

/* Host side helpers implemented in the bisheng kernel translation units. */
bool gqa_build_qk_tiling(int64_t m, int64_t n, int64_t k, int64_t isBf16, GqaCubeTiling &out);
bool gqa_build_pv_tiling(int64_t m, int64_t n, int64_t k, int64_t isBf16, GqaCubeTiling &out);
int64_t gqa_lib_workspace_size();
int64_t gqa_aiv_core_num();

extern "C" {

/* ------------------------------------------------------------------ stage 1 */
void launch_gqa_pack_f16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qp, GM_ADDR kp, GM_ADDR vp,
                         int64_t B, int64_t S, int64_t Nq, int64_t Nkv, int64_t Skv, int64_t D,
                         int64_t chunkRows, void *stream);
void launch_gqa_pack_bf16(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qp, GM_ADDR kp, GM_ADDR vp,
                          int64_t B, int64_t S, int64_t Nq, int64_t Nkv, int64_t Skv, int64_t D,
                          int64_t chunkRows, void *stream);

/* ------------------------------------------------------------------ stage 2 */
void launch_gqa_qk_f16(GM_ADDR qp, GM_ADDR kp, GM_ADDR scores, GM_ADDR sysWs, int64_t BG,
                       int64_t RS, int64_t Skv, int64_t D, int64_t MT, int64_t nMT,
                       GqaCubeTiling t, void *stream);
void launch_gqa_qk_bf16(GM_ADDR qp, GM_ADDR kp, GM_ADDR scores, GM_ADDR sysWs, int64_t BG,
                        int64_t RS, int64_t Skv, int64_t D, int64_t MT, int64_t nMT,
                        GqaCubeTiling t, void *stream);

/* ------------------------------------------------------------------ stage 3 */
void launch_gqa_softmax_f16(GM_ADDR scores, int64_t totalRows, int64_t RS, int64_t G, int64_t S,
                            int64_t Skv, float scale, int32_t causal, int32_t useMax,
                            int64_t nBlocks, int64_t rowsPerCore, void *stream);
void launch_gqa_softmax_bf16(GM_ADDR scores, int64_t totalRows, int64_t RS, int64_t G, int64_t S,
                             int64_t Skv, float scale, int32_t causal, int32_t useMax,
                             int64_t nBlocks, int64_t rowsPerCore, void *stream);

/* Flat pipeline softmax: rows are (s, h) = t % (S*Nq), and the row is Skv*Nkv wide. */
void launch_gqa_softmax_flat_f16(GM_ADDR scores, int64_t totalRows, int64_t S, int64_t Nq,
                                 int64_t G, int64_t Nkv, int64_t Skv, float scale, int32_t causal,
                                 int64_t nBlocks, int64_t rowsPerCore, void *stream);
void launch_gqa_softmax_flat_bf16(GM_ADDR scores, int64_t totalRows, int64_t S, int64_t Nq,
                                  int64_t G, int64_t Nkv, int64_t Skv, float scale, int32_t causal,
                                  int64_t nBlocks, int64_t rowsPerCore, void *stream);

/* ------------------------------------------------------------------ stage 4 */
void launch_gqa_pv_f16(GM_ADDR p, GM_ADDR vp, GM_ADDR yp, GM_ADDR sysWs, int64_t BG, int64_t RS,
                       int64_t Skv, int64_t D, int64_t MT, int64_t nMT, GqaCubeTiling t,
                       void *stream);
void launch_gqa_pv_bf16(GM_ADDR p, GM_ADDR vp, GM_ADDR yp, GM_ADDR sysWs, int64_t BG, int64_t RS,
                        int64_t Skv, int64_t D, int64_t MT, int64_t nMT, GqaCubeTiling t,
                        void *stream);

/* ------------------------------------------------------------------ stage 5 */
void launch_gqa_unpack_f16(GM_ADDR yp, GM_ADDR y, int64_t B, int64_t S, int64_t Nq, int64_t Nkv,
                           int64_t D, int64_t chunkRows, void *stream);
void launch_gqa_unpack_bf16(GM_ADDR yp, GM_ADDR y, int64_t B, int64_t S, int64_t Nq, int64_t Nkv,
                            int64_t D, int64_t chunkRows, void *stream);
}

#endif  // GQA_LAUNCH_H
