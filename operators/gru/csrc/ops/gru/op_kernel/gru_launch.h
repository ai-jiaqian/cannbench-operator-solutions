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
 * \file gru_launch.h
 * \brief GRU launch declarations shared by the bisheng (op_kernel) TU and the g++ (op_plugin) TU.
 *
 * GRU (PyTorch nn.GRU semantics).  Weight gate order inside every (3H, K) matrix is
 * [reset(r), update(z), new(n)] - i.e. rows [0,H) = r, [H,2H) = z, [2H,3H) = n:
 *
 *   gi = W_ih x_t + b_ih                (3H)
 *   gh = W_hh h_{t-1} + b_hh            (3H)
 *   r  = sigmoid(gi[0:H)  + gh[0:H))
 *   z  = sigmoid(gi[H:2H) + gh[H:2H))
 *   n  = tanh(gi[2H:3H) + r * gh[2H:3H))
 *   h_t = (1 - z) * n + z * h_{t-1}
 *
 * The whole forward is evaluated on device in fp32; inputs are cast to fp32 inside the kernel and
 * the results are cast back to the input dtype at the very end.  Nothing is routed through host memory.
 */

#ifndef GRU_LAUNCH_H
#define GRU_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

constexpr int32_t GRU_MAX_LND = 16;

struct GruArgs {
    /* raw input / output tensors in the input dtype */
    GM_ADDR x;                  /* (S,B,inSz) or (B,S,inSz) contiguous */
    GM_ADDR h0;                 /* (LD,B,H) or null */
    GM_ADDR y;                  /* (S,B,D*H) or (B,S,D*H) */
    GM_ADDR hn;                 /* (LD,B,H) */

    /* fp32 workspace base (single allocation) */
    GM_ADDR wsc;

    /* raw weight / bias tensors, input dtype, one entry per layer-direction pair */
    GM_ADDR wih[GRU_MAX_LND];
    GM_ADDR whh[GRU_MAX_LND];
    GM_ADDR bih[GRU_MAX_LND];
    GM_ADDR bhh[GRU_MAX_LND];

    int64_t S;
    int64_t B;
    int64_t inSz;
    int64_t H;
    int64_t L;
    int64_t D;
    int64_t LD;
    int64_t batchFirst;
    int64_t hasBias;
    int64_t hasH0;

    /* fp32 element offsets inside the workspace */
    int64_t xOff;
    int64_t paOff;
    int64_t pbOff;
};

/* host-side helper implemented in the bisheng TU */
int64_t calc_gru_blocks(int64_t B);

extern "C" {
void launch_gru_float(GruArgs a, void* stream);
void launch_gru_half(GruArgs a, void* stream);
void launch_gru_bfloat16(GruArgs a, void* stream);
}

#endif  // GRU_LAUNCH_H
