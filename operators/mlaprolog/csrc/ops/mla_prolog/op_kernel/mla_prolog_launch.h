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
 * \file mla_prolog_launch.h
 * \brief Declarations shared by the bisheng kernel translation units and the g++ plugin TU.
 *
 * MlaProlog (Multi-Head Latent Attention prolog):
 *   c_q_raw = x @ W_DQ ;  c_q = RMSNorm(c_q_raw, g_cq, eps_cq)
 *   qr      = c_q @ W_UQ_QR ;  q_c, q_r_raw = split(qr)      (per head)
 *   query   = q_c @ W_UK (per head) ;  query_rope = RoPE(q_r_raw, cos, sin)
 *   dkv     = x @ W_DKV_KR ; c_kv = RMSNorm(dkv[:, :Hckv], g_ckv, eps_ckv) ; k_rope = RoPE(dkv[:, Hckv:])
 *
 * PRECISION NOTE
 * --------------
 * The reference computes the whole pipeline in fp32 and only rounds the four outputs to bf16.  Rounding an
 * intermediate activation (c_q, q_c) to bf16 therefore costs ~2^-9 relative per stage, which the calibrated
 * tolerance rejects on the query path.  Both activations are carried as an EXACT 3-term bf16 split
 * (hi = bf16(v), mid = bf16(v-hi), lo = bf16(v-hi-mid)); three bf16 significands (8+8+8 bits) reproduce a
 * 24-bit fp32 mantissa bit-exactly, so the cube matmuls that consume them are exact up to fp32 accumulation
 * order.  The bf16 weights are inputs already, so they need no refinement.
 */

#ifndef MLA_PROLOG_LAUNCH_H
#define MLA_PROLOG_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

/*! \brief size of the serialized kernel-side AscendC::tiling::TCubeTiling carried through the launch ABI */
constexpr int32_t MLA_TILING_BYTES = 320;

struct MlaTiling {
    uint8_t v[MLA_TILING_BYTES];
};

/* Compile-time UB sizing constants (the operator's supported shape set fixes these). */
constexpr int64_t MLA_MAX_HCQ = 1536;    /* Hcq */
constexpr int64_t MLA_MAX_HKVDR = 576;   /* Hckv + Dr */
constexpr int64_t MLA_MAX_HCKV = 512;    /* Hckv */
constexpr int64_t MLA_MAX_DR = 64;       /* Dr */

extern "C" {

/*! \brief host-side Matmul tiling. Fills *pod with a serialized kernel-side TCubeTiling.
 *  singleM/N/K describe ONE core's block; org* are the original row extents used for address offsets.
 *  Returns 0 on success, -1 on failure. */
int32_t mla_make_tiling(int64_t singleM, int64_t singleN, int64_t singleK,
                        int64_t orgM, int64_t orgN, int64_t orgKa, int64_t orgKb,
                        int64_t cBf16, MlaTiling* pod);

int64_t mla_aic_num();
int64_t mla_aiv_num();
int64_t mla_sys_workspace_bytes();

/* ---- K1: cq[M,Hcq]f32 = x[M,He] @ wdq[He,Hcq] ;  dkv[M,HkvDr]f32 = x[M,He] @ wdkv[He,HkvDr] ---- */
void launch_mla_mm_pair(GM_ADDR x, GM_ADDR wdq, GM_ADDR cq, GM_ADDR wdkv, GM_ADDR dkv, GM_ADDR ws,
                        const MlaTiling* t1, const MlaTiling* t2,
                        int64_t M, int64_t He, int64_t Hcq, int64_t HkvDr,
                        int64_t sN1, int64_t nb1, int64_t sN2, void* stream);

/* ---- K2: csplit[3M,Hcq]bf16 (hi/mid/lo of c_q) ; ckv[M,Hckv]bf16 ; krope[M,Dr]bf16 ---- */
void launch_mla_v_norm(GM_ADDR cq, GM_ADDR gcq, GM_ADDR dkv, GM_ADDR gckv,
                       GM_ADDR cosG, GM_ADDR sinG,
                       GM_ADDR csplit, GM_ADDR ckvOut, GM_ADDR krOut,
                       int64_t M, int64_t Hcq, int64_t HkvDr, int64_t Hckv, int64_t Dr,
                       float epsCq, float epsCkv, int64_t nBlocks, void* stream);

/* ---- K3: wuk3[N][3D][Hckv]bf16 <- wuk[N][D][Hckv]bf16 repeated 3x along the D axis ---- */
void launch_mla_v_dupuk(GM_ADDR wuk, GM_ADDR wuk3,
                        int64_t N, int64_t D, int64_t Hckv, int64_t nBlocks, void* stream);

/* ---- K4: qr2[3M,NW]f32 = csplit[3M,Hcq] @ wuqqr[Hcq,NW] , N split across blocks ---- */
void launch_mla_mm_big(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, const MlaTiling* t,
                       int64_t sN, int64_t nb, void* stream);

/* ---- K5: fold the 3 qr2 planes -> fp32; split q_c -> qcs[N][3M][D]bf16 ; q_r_raw -> qropeIn[M][N*Dr]f32 ---- */
void launch_mla_v_fold(GM_ADDR qr2, GM_ADDR qcs, GM_ADDR qropeIn,
                       int64_t M, int64_t N, int64_t D, int64_t Dr, int64_t NW,
                       int64_t nBlocks, void* stream);

/* ---- K6: qtmp3[N][3M][Hckv]f32 = qcs[N][3M][D] @ wuk[N][D][Hckv] , one block per head ---- */
void launch_mla_mm_qk(GM_ADDR qcs, GM_ADDR wuk, GM_ADDR qtmp3, GM_ADDR ws, const MlaTiling* t,
                      int64_t M3, int64_t D, int64_t Hckv, int64_t nHeads, void* stream);

/* ---- K6b: qtmp[N][M][Hckv]bf16 = sum of the three qtmp3 row bands (exact fp32 fold) ---- */
void launch_mla_v_foldq(GM_ADDR qtmp3, GM_ADDR qtmp, int64_t M, int64_t N, int64_t Hckv,
                        int64_t nBlocks, void* stream);

/* ---- K7: query[M][N*Hckv]bf16 = transpose(qtmp) ; queryrope[M][N*Dr]bf16 = RoPE(qropeIn) ---- */
void launch_mla_v_final(GM_ADDR qtmp, GM_ADDR qropeIn, GM_ADDR cosG, GM_ADDR sinG,
                        GM_ADDR query, GM_ADDR queryRope,
                        int64_t M, int64_t N, int64_t Hckv, int64_t Dr,
                        int64_t nBlocks, void* stream);

} /* extern "C" */

#endif /* MLA_PROLOG_LAUNCH_H */
