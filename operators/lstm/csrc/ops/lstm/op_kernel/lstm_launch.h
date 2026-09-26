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
 * \file lstm_launch.h
 * \brief LSTM launch declarations shared by the bisheng (op_kernel) TU and the g++ (op_plugin) TU.
 *
 * LSTM (PyTorch nn.LSTM semantics, gate order [i, f, g, o]):
 *   gates = W_ih x_t + b_ih + b_hh + W_hh h_{t-1}   (4H rows)
 *   i = sigmoid(gates[0:H]); f = sigmoid(gates[H:2H])
 *   g = tanh(gates[2H:3H]);  o = sigmoid(gates[3H:4H])
 *   c_t = f .* c_{t-1} + i .* g
 *   h'_t = o .* tanh(c_t)
 *   h_t  = W_hr h'_t    (LSTMP, only when projSize > 0; otherwise h_t = h'_t)
 *
 * All arithmetic runs on device in fp32.  Every input tensor of dtype T is cast exactly once by the
 * preparation kernel into an fp32 workspace whose rows are already zero padded to the geometry that the
 * kernels consume; the recurrence kernel casts the final results back to T.
 *
 * Layout discipline (element counts, all fp32 unless noted):
 *   align8(x)  = (x + 7) / 8 * 8      align64(x) = (x + 63) / 64 * 64
 *   eff        = projSize > 0 ? projSize : hiddenSize
 *   G          = align64(H)            -- one "gate segment" width
 *   kHh        = align64(eff)          -- W_hh reduction width AND the width of the recurrent state h
 *   inDim[ld]  = layer == 0 ? inputSize : D * eff
 *   inPad[ld]  = align64(inDim[ld])
 *   sw         = align64(D * eff)      -- width of one sequence row in the ping-pong layer buffers
 *   Ppad       = align64(projSize)     -- == kHh when projSize > 0
 *
 * Gate aware weight storage: the 4H raw rows of W_ih / W_hh are stored as 4*G rows, gate g occupying
 * rows [g*G, g*G+G) with rows [g*G+H, g*G+G) zero.  The row block reduction therefore produces the four
 * gates already separated at G aligned offsets and the padding lanes stay inert through the recurrence
 * (their pre-activations are exactly 0 => i=f=o=0.5, g=0 => c_pad = 0.5*c_pad which stays 0 and h_pad = 0).
 *
 * Every raw row written by the preparation kernel is either real data (cast, right padded to its full
 * destination width) or an explicit zero row, so no region of the workspace relies on pre-existing memory
 * content.  Every read of the workspace is a full width aligned DataCopyPad.
 */

#ifndef LSTM_LAUNCH_H
#define LSTM_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

constexpr int32_t LSTM_MAX_LD = 8;

struct LstmArgs {
    /* ---------------- raw inputs, dtype T ---------------- */
    GM_ADDR wih[LSTM_MAX_LD];
    GM_ADDR whh[LSTM_MAX_LD];
    GM_ADDR bih[LSTM_MAX_LD];
    GM_ADDR bhh[LSTM_MAX_LD];
    GM_ADDR whr[LSTM_MAX_LD];
    GM_ADDR x;
    GM_ADDR h0;
    GM_ADDR c0;
    /* ---------------- outputs, dtype T ---------------- */
    GM_ADDR y;
    GM_ADDR hn;
    GM_ADDR cn;
    /* ---------------- fp32 workspace regions ---------------- */
    GM_ADDR wihW[LSTM_MAX_LD];
    GM_ADDR whhW[LSTM_MAX_LD];
    GM_ADDR whrW[LSTM_MAX_LD];
    GM_ADDR biasW[LSTM_MAX_LD];
    GM_ADDR xW;
    GM_ADDR seqW[2];
    GM_ADDR h0fW;
    GM_ADDR c0fW;
    /* ---------------- per layer/direction geometry ---------------- */
    int32_t inDim[LSTM_MAX_LD];
    int32_t inPad[LSTM_MAX_LD];
    /* ---------------- global geometry ---------------- */
    int32_t S;
    int32_t B;
    int32_t inSz;
    int32_t H;
    int32_t P;
    int32_t D;
    int32_t L;
    int32_t LD;
    int32_t G;
    int32_t kHh;
    int32_t sw;
    int32_t inPad0;
    int32_t eff;
    int32_t batchFirst;
    int32_t hasBias;
    int32_t hasH0;
    int32_t hasC0;
    int32_t useHr;
    int32_t nbPrep;
    int32_t prepMaxElems;
    int32_t ubBytes;
};

/* host side helpers implemented in the bisheng TU */
int64_t lstm_core_num();
int64_t lstm_ub_bytes();

extern "C" {
void launch_lstm_prep_float(LstmArgs a, void* stream);
void launch_lstm_prep_half(LstmArgs a, void* stream);
void launch_lstm_prep_bf16(LstmArgs a, void* stream);
void launch_lstm_rec_float(LstmArgs a, int64_t layer, void* stream);
void launch_lstm_rec_half(LstmArgs a, int64_t layer, void* stream);
void launch_lstm_rec_bf16(LstmArgs a, int64_t layer, void* stream);
}

#endif // LSTM_LAUNCH_H
