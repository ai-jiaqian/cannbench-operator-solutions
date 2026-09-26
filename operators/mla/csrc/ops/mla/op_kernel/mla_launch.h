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
 * \file mla_launch.h
 * \brief MLA (Multi-head Latent Attention) launch / tiling declarations shared by the
 *        bisheng kernel translation units and the g++ plugin translation unit.
 *
 *   Q = concat(q_nope, q_rope)   K = concat(k_nope, k_rope)   V = k_nope
 *   y = softmax(Q K^T * scale) V
 *
 * Pipeline (three kernels, all arithmetic on device, intermediates in a GM workspace):
 *   K1 qk      : s[totalRows, Skv] = qa @ kb^T            (float32)   -- two launches
 *   K2 softmax : p[totalRows, Skv] = rownorm(exp(scale*s)) (input dtype)
 *   K3 pv      : y[totalRows, Dn]  = p @ v                (input dtype)
 *
 * Layout handling: with numKVHeads == 1 the per-batch query / output arrays of both
 * BSND ([B,S,Nq,D]) and BNSD ([B,Nq,S,D]) are one and the same contiguous [S*Nq, D]
 * block, and k_nope / k_rope / v are plain [Skv, D] matrices per batch.  Nothing is
 * permuted anywhere; only the decoding of a flat row index into a key position differs
 * and that is needed only for the causal mask (BSND s = r / Nq, BNSD s = r % S).
 */

#ifndef MLA_LAUNCH_H
#define MLA_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/*!
 * \brief Kernel-side copy of the fields of AscendC::tiling::TCubeTiling that the MLA
 *        matmuls need.
 *
 * The host side (bisheng host pass) owns the matmul tiling headers and produces the
 * values from matmul_tiling::MatmulApiTiling; the g++ plugin cannot see those headers,
 * so the tiling travels through the launch ABI as this self-describing POD.  The kernel
 * rebuilds its own AscendC::tiling::TCubeTiling field by field (never by memcpy - the two
 * declarations live in different include trees and their layouts are not guaranteed to
 * agree beyond the documented field list).
 */
struct MlaTiling {
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
    int32_t stepKa;
    int32_t stepKb;
    int32_t isBias;
    int32_t transLength;
    int32_t iterateOrder;
    int32_t dbL0A;
    int32_t dbL0B;
    int32_t dbL0C;
    int32_t shareMode;
    int32_t shareL1Size;
    int32_t shareL0CSize;
    int32_t shareUbSize;
    int32_t batchM;
    int32_t batchN;
    int32_t singleBatchM;
    int32_t singleBatchN;
    int32_t mxTypePara;
};

extern "C" {

/*! \brief bytes of system workspace the AscendC library APIs (Matmul) need per block. */
int64_t mla_sys_ws_bytes(void);

/*! \brief number of AIC (cube) cores of the current platform. */
int64_t mla_aic_num(void);

/*! \brief number of AIV (vector) cores of the current platform. */
int64_t mla_aiv_num(void);

/*!
 * \brief single-core Matmul tiling for  C[M,N] = A[M,K] * B[K,N].
 *   dt     : 0 = float16, 1 = bfloat16
 *   bTrans : B is physically [N,K] (i.e. logical B^T)
 *   cF32   : C elements are float32 (otherwise the input dtype)
 * returns 0 on success, -1 on failure.
 */
int64_t mla_tiling_mm(int64_t dt, int64_t M, int64_t N, int64_t K, int64_t bTrans, int64_t cF32,
                      MlaTiling* pod);

/* ---- K1: s[startRows+0 .. startRows+RQ, 0..Skv) = qa * kb^T ----
 *
 * Launched twice for one MLA: (q_nope, k_nope, Dc = Dn, enAtomic = 0) fills s, then
 * (q_rope, k_rope, Dc = Dr, enAtomic = 1) adds the rope half of the dot product.  Two
 * sequential launches guarantee that the AtomicAdd cannot race the plain store of the
 * other pass (a kernel boundary is a full memory barrier).
 */
void launch_mla_qk_half(GM_ADDR qa, GM_ADDR kb, GM_ADDR s, GM_ADDR ws, MlaTiling t, int64_t B,
                        int64_t R, int64_t Skv, int64_t Dc, int64_t RQ, int64_t perWs,
                        int64_t nBlocks, int64_t nBandsTot, int64_t enAtomic, void* stream);
void launch_mla_qk_bfloat16(GM_ADDR qa, GM_ADDR kb, GM_ADDR s, GM_ADDR ws, MlaTiling t, int64_t B,
                            int64_t R, int64_t Skv, int64_t Dc, int64_t RQ, int64_t perWs,
                            int64_t nBlocks, int64_t nBandsTot, int64_t enAtomic, void* stream);

/* ---- K2: p = rownorm(exp(scale * s)) with the causal mask ---- */
void launch_mla_softmax_half(GM_ADDR s, GM_ADDR p, int64_t totalRows, int64_t R, int64_t Skv,
                             int64_t S, int64_t Nq, int64_t bnMode, int64_t causal, float scale,
                             int64_t BR, int64_t nBlocks, void* stream);
void launch_mla_softmax_bfloat16(GM_ADDR s, GM_ADDR p, int64_t totalRows, int64_t R, int64_t Skv,
                                 int64_t S, int64_t Nq, int64_t bnMode, int64_t causal, float scale,
                                 int64_t BR, int64_t nBlocks, void* stream);

/* ---- K3: y[R rows of a batch, 0..Dn) = p * v ---- */
void launch_mla_pv_half(GM_ADDR p, GM_ADDR v, GM_ADDR y, GM_ADDR ws, MlaTiling t, int64_t B,
                        int64_t R, int64_t Skv, int64_t Dn, int64_t RQ, int64_t perWs,
                        int64_t nBlocks, int64_t nBandsTot, void* stream);
void launch_mla_pv_bfloat16(GM_ADDR p, GM_ADDR v, GM_ADDR y, GM_ADDR ws, MlaTiling t, int64_t B,
                            int64_t R, int64_t Skv, int64_t Dn, int64_t RQ, int64_t perWs,
                            int64_t nBlocks, int64_t nBandsTot, void* stream);

} // extern "C"

#endif // MLA_LAUNCH_H
