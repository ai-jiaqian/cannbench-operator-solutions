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
 * \file engram_kernel.cpp
 * \brief EngramGateFusion direct-launch kernels (bisheng / AscendC, dav-2201).
 *
 * DESIGN NOTES
 * ------------
 * 1. Everything is fp32 internally; only the operator inputs/outputs are bf16.  No low precision
 *    transcendentals (Rsqrt / Reciprocal are ~9 bit / ~1e-3 on this part): every inverse is
 *    Sqrt + Div, and every division is the accurate vector Div.
 *
 * 2. The gate needs three reductions per (b,l,hc) row plus sum(value^2), and the whole 7 step chain
 *    collapses to two scalars per row:
 *        gate = sigmoid(sqrt(clamp(|raw|,1e-6)) * sign(raw)),   s = gate / sqrt(gate^2*mean(v^2)+eps)
 *        vg = value * gate,                                     nv = value * wc * s
 *
 * 3. THE CARRIER SCALING.  The circular window does NOT store nv; it stores
 *        w = nv / wc = value * s
 *    and the conv weights are pre-scaled to cw' = cw * wc.  Because
 *        sum_k cw'[k] * w == sum_k cw[k] * nv
 *    this is exact up to rounding and deletes one full width multiply from every row of the hot
 *    loop - which matters, because the measured row cost is proportional to the NUMBER of
 *    D-element vector ops (1.27 us/row at D = 1024, K = 4 vs 1.42 us/row at K = 8, i.e. the
 *    measured 1.12 ratio matches the predicted 26/22 op ratio).  The two places the scaling shows
 *    through are the decode seed rows (divided by wc in the main prologue, since conv_state
 *    arrives as real nv) and the state output rows (multiplied back by wc before the store).
 *    Both are D-element vector ops paid once per block, not once per row.
 *
 * 4. THE WEIGHT PREP IS NOT A KERNEL.  Building cwT = transpose(cw)*wc and w12 = w1*w2 used to be
 *    a separate launch, but a launch costs about as much as the entire fixed part of a call, while
 *    the transpose is only K Gathers per main block.  Each main block now builds both in UB,
 *    borrowing the (still unused at that point) bf16 staging buffer as the [d][K] cw staging area
 *    and a corner of the lane grid as the Gather offset ramp - so the UB budget, and therefore
 *    `tr`, is exactly what it was.  A call is now THREE kernels (prefill) or FOUR (decode), the
 *    extra one being the conv_state transpose.
 *
 *    Watch out for the Gather signature: `Gather(dst, src, srcOffset, srcBaseAddr, count)` takes
 *    the BYTE OFFSET TENSOR as its third argument.  Passing the raw index ramp instead silently
 *    makes every tap use tap 0 - which shows up as conv_state_out being perfect (it does not touch
 *    cwT at all) while every output element is wrong.
 *
 * 5. The ShortConv runs on a circular fp32 window of W = H+1 rows held in UB.  Row l stores its own
 *    w into slot (l mod W); slot q holds xc[q + H] once the walk has reached it and it holds the
 *    seed value xc[q-1] before that.  Tap k of output row l reads slot (l + k*dil - H) mod W, and
 *    the two identities agree because -H == +1 (mod H+1): for l + k*dil >= H the slot has already
 *    been written by the walk and holds xc[l+k*dil], and for l + k*dil < H the slot is still the seed
 *    and holds exactly the state entry xc[l+k*dil].  That is why ANY chunk whose first row index is
 *    below H must load the conv_state seed (not just chunk 0) and why the seed is loaded from
 *    `rowStart == 0` onward.
 *
 *    The k = 0 tap of row l needs xc[l], which lives in slot (l+1) mod W - the very slot that row
 *    l+1 writes.  So the row order is load bearing: write(l) must precede conv(l) and conv(l) must
 *    precede write(l+1).  Phase 2 therefore CANNOT be software-pipelined over a group of rows
 *    (verified on paper for every group size >= 2 while W = H+1).
 *
 * 6. A work item owns rows [l0, l1) and additionally walks the halo rows [l0-H, l0) purely to seed
 *    the window.  Those rows must NOT write the output: the same rows belong to an earlier chunk and
 *    both blocks would race on them, and their own conv would read window slots they have not filled
 *    yet.
 *
 * 7. Chunking and state parallelism are chosen on the host by cost models, and the measured shape of
 *    this operator is what drives them:
 *      elapsed ~= FIXED + (rows owned by ONE block, halo included) * waves * per_row_cost
 *    with FIXED in the low tens of microseconds for a 3-launch call.  So (a) the main grid must fill
 *    the cores once (or a whole multiple of them) and then make each block as short as possible, and
 *    (b) every auxiliary kernel must be spread over ALL cores - at 4..16 blocks they were leaving the
 *    machine mostly idle.
 *
 * 8. Rows are processed in tiles of `tr` rows.  One multi block DataCopyPad per input tensor fetches
 *    a whole tile and one multi block DataCopyPad writes the tile's output rows, so the DMA
 *    instruction count and the barrier count per row drop by a factor `tr` (`tr` is derived inside
 *    the kernel from the leftover UB budget, so the widest shapes fall back to a small tile while the
 *    common ones get a large one).
 *
 * 9. The gate chain is BATCHED over the whole tile.  A tile's per-row four reductions are written
 *    into a strided lane grid (quantity q of row r lives at fp32 lane q*CHW + r*EN_NL with
 *    CHW = 8*EN_NL = 64), and the ~25 step chain then runs ONCE with a count of CHW lanes, i.e. once
 *    per tile instead of once per row.  The strided lane grid exists because a reduction destination
 *    must start on a 32B boundary, so the 8 rows of a tile are 8 fp32 lanes apart.  (Reducing several
 *    rows in ONE ReduceSum call is not an option: on Ascend950 a multi-row AR reduce returns
 *    per-row-wrong results.)
 *
 * 10. WHAT COSTS THE TIME IS THE NUMBER OF FULL WIDTH VECTOR OPS, NOT LATENCY AND NOT BYTES.
 *    Established by measurements:
 *      * A row costs the same at D = 769 (12 fp32 repeats per op) as at D = 1024 (16 repeats).
 *      * Replacing the four full width ReduceSums with a binary fold (5 extra instructions per row)
 *        cost 15..20%, which is roughly the extra issue slots and nothing more.
 *      * Batching phase 1 over four rows per instruction (gr = 4) was worth nothing at all, because
 *        it forced `tr` down from 8 to 4 and the extra per-tile barriers ate the whole gain.
 *      * The K = 8 case costs 1.42 us/row against 1.27 us/row for K = 4, matching the op ratio
 *        (26 vs 22) rather than the MAC ratio.
 *      * Folding wc into cwT and using MulAddDst for the taps (4 fewer full width ops per row)
 *        improved exactly the row dominated cases: case 6 271.95 -> 254.67 us, case 7 243.15 ->
 *        230.52 us, case 8 309.49 -> 294.75 us.
 *    What DID pay, hugely, was fanning the auxiliary kernels out over all cores and replacing their
 *    SCALAR transpose loops with vector Gathers: a scalar UB round trip costs about ten cycles of
 *    scalar pipe latency, so 2*dn*K (prep), 2*dn*H (state_in) and 2*dn*H (state_out) scalar accesses
 *    completely dominated the fixed cost.  Measured progression of this kernel: 57.42 -> 58.40
 *    (prep + state_in Gathers) -> 59.19 (state_out Gather) -> 59.99 (hoist the periodic gather ramp
 *    pattern out of the chunk loop).
 *
 * 11. The tile loop uses exactly THREE `EnSyncAll()` per tile, and the staging DMA of tile t+1 sits
 *    between two of them rather than at the top of the loop:
 *
 *        prologue: weights + window seed + stage(tile 0) ;  SYNC
 *        loop:    phase1 + chain ; SYNC(V->S) ; readback + phase2 ; SYNC(V->MTE3, V->MTE2)
 *                 store(tile) ; stage(tile+1) ; SYNC(MTE2->V)
 *
 *    The middle SYNC is what orders phase 2's vector reads of the staging buffer before the next
 *    tile's MTE2 write, and it is why the staging copy must be issued *after* the store.  Issuing
 *    the stage at the top of the loop instead (with the trailing SYNC deleted) was measured to
 *    corrupt the result: an MTE2 write is not ordered against earlier V reads by a barrier that comes
 *    after it, and the resulting error was a ~0.5% wobble - bf16-quantum sized.
 *
 *    A PIPE_ALL barrier drains every pipe, so this single primitive covers the MTE2->S, S->V, V->S
 *    and S->MTE3 crossings that a bare `PipeBarrier` is not documented to cover.  Hard coded
 *    SetFlag/WaitFlag pairs held across loop iterations were measured to hang this kernel (vector
 *    core timeout, 507034): FetchEventID does not reserve an id, so two live Set/Wait pairs of the
 *    same pipe end up sharing one id.  A SetFlag/WaitFlag pair that is issued and consumed adjacent
 *    to each other in a straight line of code is however safe and cheap, and that is what the EnM2V
 *    / EnV2M2 helpers below are for.  Note that the number of full barriers on the FIXED path matters
 *    as much as on the row path: two stray `EnSyncAll()` calls in the state-output epilogue measurably
 *    cost every one of the 20 cases about 0.5 us.
 *
 * 12. Buffer geometry rules learned on this part:
 *    - A multi block DataCopyPad's srcStride applies to the SOURCE side and dstStride to the
 *      DESTINATION side (both are byte gaps), and the UB row advance is alignUp(blockLen, 32) plus
 *      the UB side gap.  With a blockLen of D*2 bytes that advance is exactly PS*2 bytes where
 *      PS = align(D, 16), so staging rows are laid out at pitch PS.  GM->UB therefore puts the GM
 *      gap in srcStride and 0 in dstStride; UB->GM is the mirror image.  A UB row pitch is always
 *      alignUp(rowBytes, 32) / elemSize - NOT the tile's nominal width - which is why the state
 *      kernels index their row buffer with align(dn, 8) and not with align(dt, 8).
 *    - The granule rounding provably applies to the UB side but the GM side behaves differently, so
 *      a multi block copy whose source rows sit at exactly alignUp(rowBytes,32) is ambiguous there.
 *      Two attempts at a multi block cwT load produced 19/20 with the failure isolated to D = 769
 *      (the only D that is not a multiple of 8); the conv weight rows are therefore transposed from
 *      a plain contiguous [d][K] block with Gathers and no multi block copy with a stride is used
 *      for them.
 *    - A block DMA cannot transpose: a block is contiguous on BOTH sides, and a transpose maps the
 *      fast axis of one operand onto the slow axis of the other, so no stride choice works.  That is
 *      why the transposes are Gathers.  A Gather's destination is contiguous, so a transposed layout
 *      whose flat index i maps to source (j, d) with j = i % Hp, d = i / Hp needs an interleaved
 *      offset ramp; it is built by dividing the index by a POWER OF TWO Hp with a float multiply and
 *      a truncating Cast (exact, and no Floor needed because every index is non-negative).
 *    - A Gather / CreateVecIndex pair is only known to be safe at the small counts used elsewhere in
 *      this file (<= 128).  Driving it with a count of dn*Hp (over a thousand) faulted on the first
 *      try with "VEC instruction error: the ub address out of bounds", so the interleaved ramp is
 *      both built and consumed in chunks of 128.
 *    - Because the ramp pattern has period Hp and Hp divides the 128 element chunk, the pattern is
 *      IDENTICAL in every chunk; it is therefore built once per block and each subsequent chunk costs
 *      only an Adds (the chunk's block index times 4) plus the Gather.  Rebuilding the whole 9
 *      instruction ramp per chunk instead was the dominant cost of the state output for large
 *      channel tiles (up to ~56 us of a 333 us case).
 *    - `LocalTensor::operator[]` yields a sub-view used as a *vector operand*; a single element must
 *      be written with `SetValue` (assigning through `[]` is a type error on this toolchain).
 *    - Vector operands, including a Gather's destination and its offset tensor, must sit on 32B
 *      boundaries, so a tail zeroing cannot start at an arbitrary element index.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

#include "engram_launch.h"

using namespace AscendC;

namespace {

constexpr int64_t EN_DT_MAX = 128;    /* state kernels: channel tile upper bound      */
constexpr int64_t EN_NL = 8;          /* fp32 lanes between two rows in the lane grid */
constexpr int32_t EN_CHW = (int32_t)(8 * EN_NL); /* chain lane width: 8 rows per tile */
constexpr int64_t EN_CHAINS = 16 * EN_CHW;       /* fp32 slots of the batched chain  */
constexpr int64_t EN_SLACK = 8;       /* trailing pad so vector granule rounding is safe */
constexpr int64_t EN_TMP = 8192;      /* ReduceSum shared work buffer (bytes)        */
constexpr int64_t EN_TR_MAX = 8;      /* rows staged per tile, upper bound (lane grid) */
constexpr int64_t EN_UB_BUDGET = 176 * 1024;
constexpr int64_t EN_GCH = 128;       /* vector GATHER / ramp chunk, in elements     */

__aicore__ inline int64_t EnAl(int64_t x, int64_t a)
{
    return (x + a - 1) / a * a;
}

__aicore__ inline int64_t EnMod(int64_t x, int64_t m)
{
    int64_t r = x - (x / m) * m;
    if (r < 0) {
        r += m;
    }
    return r;
}

/* A single adjacent Set/Wait pair: exact ordering for one pipe crossing, at the price of one
 * hardware event rather than a full pipe drain.  Safe with a constant event id because the two are
 * issued back to back in a straight line and never live across a loop iteration. */
__aicore__ inline void EnM2V()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
__aicore__ inline void EnV2M2()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

/* Full pipeline barrier plus the scalar crossings a plain PipeBarrier is not reliably documented
 * to cover (MTE2->S, S->V, V->S, S->MTE3).  This is the only heavy synchronisation primitive, and
 * it is used three times per tile. */
__aicore__ inline void EnSyncAll()
{
    PipeBarrier<PIPE_ALL>();
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(e);
    WaitFlag<HardEvent::MTE2_S>(e);
    e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e);
    WaitFlag<HardEvent::S_V>(e);
    e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e);
    WaitFlag<HardEvent::V_S>(e);
    e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(e);
    WaitFlag<HardEvent::S_MTE3>(e);
}

/* Interleaved gather of a transposed tile, in chunks of EN_GCH elements:
 *      dst[i] = src[ (i % Hp) * strided + i / Hp ]      (element units)
 * The period Hp divides EN_GCH, so the first EN_GCH offset values are the same for every chunk;
 * that pattern (pat) is built once and each chunk costs one Adds and one Gather.  The first index
 * of a chunk is a multiple of EN_GCH, hence chunk base c0 maps to the pattern offset (c0/Hp). */
__aicore__ inline void EnPatGather(LocalTensor<float> dst, LocalTensor<float> src,
                                   LocalTensor<int32_t> idx, LocalTensor<float> idxF,
                                   LocalTensor<int32_t> quo, LocalTensor<int32_t> pat,
                                   LocalTensor<int32_t> oft, int64_t n, int64_t Hp,
                                   int64_t strided)
{
    const int32_t cw = (int32_t)EN_GCH;
    CreateVecIndex(idx, (int32_t)0, cw);
    Cast(idxF, idx, RoundMode::CAST_NONE, cw);
    Muls(idxF, idxF, 1.0f / (float)Hp, cw);
    Cast(quo, idxF, RoundMode::CAST_TRUNC, cw); /* local / Hp                */
    Muls(pat, quo, (int32_t)Hp, cw);
    Sub(pat, idx, pat, cw);                     /* local % Hp                */
    Muls(oft, quo, (int32_t)4, cw);             /* (local / Hp) * 4          */
    Muls(pat, pat, (int32_t)(strided * 4), cw); /* (local % Hp) * strided * 4 */
    Add(pat, pat, oft, cw);
    for (int64_t c0 = 0; c0 < n; c0 += EN_GCH) {
        int64_t cn = n - c0;
        if (cn > EN_GCH) {
            cn = EN_GCH;
        }
        const int32_t k = (int32_t)cn;
        Adds(oft, pat, (int32_t)((c0 / Hp) * 4), k);
        Gather(dst[(uint32_t)c0], src, oft.ReinterpretCast<uint32_t>(), (uint32_t)0, k);
    }
}

}  // namespace

/* =====================================================================================
 * Kernel 1 : conv_state [B, HC*D, H] -> wsi [B, HC, H, D] fp32   (decode only)
 * The load is one contiguous block; each of the H transposed rows is one Gather.
 * ===================================================================================== */
__global__ __aicore__ void engram_state_in_kernel(GM_ADDR convState, GM_ADDR wsi, int64_t B,
                                                  int64_t HC, int64_t D, int64_t H, int64_t dt,
                                                  int64_t nDT)
{
    const int64_t blk = (int64_t)GetBlockIdx();
    const int64_t bhc = blk / nDT;
    const int64_t dtc = blk % nDT;
    if (H <= 0 || dt <= 0 || bhc >= B * HC) {
        return;
    }
    const int64_t d0 = dtc * dt;
    if (d0 >= D) {
        return;
    }
    int64_t dn = D - d0;
    if (dn > dt) {
        dn = dt;
    }
    const int64_t n = dn * H;
    const int64_t nc = EnAl(n, 64);
    const int64_t rp = EnAl(dn, 8); /* the pitch the DMA engine actually uses */
    const int64_t rq = EnAl(dt, 8); /* allocation bound (>= rp)              */

    TPipe pipe;
    TBuf<TPosition::VECCALC> sB, fB, rB, iB, oB;
    pipe.InitBuffer(sB, (nc + EN_SLACK) * sizeof(bfloat16_t));
    pipe.InitBuffer(fB, (nc + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(rB, (H * rq + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(iB, (dn + EN_SLACK) * sizeof(int32_t));
    pipe.InitBuffer(oB, (dn + EN_SLACK) * sizeof(int32_t));
    auto sL = sB.Get<bfloat16_t>();
    auto fL = fB.Get<float>();
    auto rL = rB.Get<float>();
    auto idxI = iB.Get<int32_t>();
    auto oftI = oB.Get<int32_t>();
    auto oftU = oB.Get<uint32_t>();

    GlobalTensor<bfloat16_t> csG;
    GlobalTensor<float> wsiG;
    csG.SetGlobalBuffer((__gm__ bfloat16_t*)convState);
    wsiG.SetGlobalBuffer((__gm__ float*)wsi);

    DataCopyExtParams cp{1, (uint32_t)(n * sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> ppB{false, 0, 0, 0};
    DataCopyPad(sL, csG[(bhc * D + d0) * H], cp, ppB);
    EnSyncAll();

    Cast(fL, sL, RoundMode::CAST_NONE, (int32_t)n);

    /* wsi layout [j][d]: Gather row j from the [d][j] source with a byte ramp of H*4. */
    CreateVecIndex(idxI, (int32_t)0, (int32_t)dn);
    Muls(idxI, idxI, (int32_t)(H * 4), (int32_t)dn);
    for (int64_t j = 0; j < H; ++j) {
        Adds(oftI, idxI, (int32_t)(j * 4), (int32_t)dn);
        Gather(rL[(uint32_t)(j * rp)], fL, oftU, (uint32_t)0, (int32_t)dn);
    }
    EnSyncAll();

    /* UB -> GM: the UB gap is srcStride (0, the rows already sit at the natural pitch) and the GM
     * gap between wsi rows is dstStride. */
    DataCopyExtParams cw{(uint16_t)H, (uint32_t)(dn * sizeof(float)), 0,
                         (uint32_t)((D - dn) * (int64_t)sizeof(float)), 0};
    DataCopyPad(wsiG[bhc * H * D + d0], rL, cw);
}

/* =====================================================================================
 * Kernel 3 : ws [B, HC, H, Dpad] fp32 -> conv_state_out [B, HC*D, H] bf16
 *
 * The output is [d][j] with H contiguous bf16 per channel, i.e. a 2-byte row of H elements whose
 * UB pitch is alignUp(H*2, 32) - 32 bytes for H <= 16, 64 bytes for H = 21.  Hp is that pitch in
 * ELEMENTS, and the transposed tile is built directly in the [d][Hp] layout by a Gather whose
 * offset ramp is
 *      off[i] = (i % Hp) * rp * 4 + (i / Hp) * 4,        i in [0, dn*Hp)
 * so that  oL[d*Hp + j] = iL[j*rp + d]  for j < H  (the lanes j >= H are gathered too, from
 * padding rows of the source buffer, and are simply never written to GM because the store's
 * blockLen is H*2 bytes).  The source buffer is allocated for Hp rows rather than H so that the
 * padding lanes stay in bounds.
 * ===================================================================================== */
__global__ __aicore__ void engram_state_out_kernel(GM_ADDR ws, GM_ADDR sout, int64_t B, int64_t HC,
                                                   int64_t D, int64_t H, int64_t dt, int64_t nDT)
{
    const int64_t blk = (int64_t)GetBlockIdx();
    const int64_t bhc = blk / nDT;
    const int64_t dtc = blk % nDT;
    if (H <= 0 || dt <= 0 || bhc >= B * HC) {
        return;
    }
    const int64_t d0 = dtc * dt;
    if (d0 >= D) {
        return;
    }
    int64_t dn = D - d0;
    if (dn > dt) {
        dn = dt;
    }
    const int64_t Dpad = EnAl(D, 16);
    const int64_t Hp = (H * 2 <= 32) ? 16 : 32;
    const int64_t n = dn * Hp;
    const int64_t nc = EnAl(n, 64);
    const int64_t rp = EnAl(dn, 8);
    const int64_t rq = EnAl(dt, 8);

    TPipe pipe;
    TBuf<TPosition::VECCALC> iB, oB, bB, xB, yB, zB, uB;
    pipe.InitBuffer(iB, (Hp * rq + 64) * sizeof(float));
    pipe.InitBuffer(oB, (nc + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(bB, (nc + EN_SLACK) * sizeof(bfloat16_t));
    pipe.InitBuffer(xB, (EN_GCH + EN_SLACK) * sizeof(int32_t));
    pipe.InitBuffer(yB, (EN_GCH + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(zB, (EN_GCH + EN_SLACK) * sizeof(int32_t));
    pipe.InitBuffer(uB, (EN_GCH + EN_SLACK) * sizeof(int32_t));
    auto iL = iB.Get<float>();
    auto oL = oB.Get<float>();
    auto bL = bB.Get<bfloat16_t>();
    auto xI = xB.Get<int32_t>();
    auto yF = yB.Get<float>();
    auto zI = zB.Get<int32_t>();
    auto uI = uB.Get<int32_t>();

    GlobalTensor<float> wsG;
    GlobalTensor<bfloat16_t> soG;
    wsG.SetGlobalBuffer((__gm__ float*)ws);
    soG.SetGlobalBuffer((__gm__ bfloat16_t*)sout);

    /* GM -> UB: the GM gap between ws rows is srcStride; the UB rows land at pitch
     * alignUp(dn*4, 32) because dstStride is 0. */
    DataCopyExtParams cr{(uint16_t)H, (uint32_t)(dn * sizeof(float)),
                         (uint32_t)((Dpad - dn) * (int64_t)sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> ppF{false, 0, 0, 0};
    DataCopyPad(iL, wsG[bhc * H * Dpad + d0], cr, ppF);
    EnSyncAll();

    EnPatGather(oL, iL, xI, yF, zI, uI, zI, n, Hp, rp);
    Cast(bL, oL, RoundMode::CAST_RINT, (int32_t)n);
    EnSyncAll();

    /* UB -> GM: the UB rows advance by alignUp(H*2, 32) = Hp*2 bytes (srcStride 0) and the GM rows
     * are exactly H elements apart (dstStride 0), so this is the whole transposed state. */
    DataCopyExtParams cp{(uint16_t)dn, (uint32_t)(H * 2), 0, 0, 0};
    DataCopyPad(soG[(bhc * D + d0) * H], bL, cp);
}

/* =====================================================================================
 * Kernel 2 : the fused operator.  Grid = B*HC*nChunks.
 * ===================================================================================== */
__global__ __aicore__ void engram_main_kernel(GM_ADDR keys, GM_ADDR hidden, GM_ADDR value,
                                              GM_ADDR w1, GM_ADDR w2, GM_ADDR wc, GM_ADDR cw,
                                              GM_ADDR wsi, GM_ADDR ws, GM_ADDR out, int64_t B,
                                              int64_t L, int64_t HC, int64_t D, int64_t K,
                                              int64_t dil, int64_t H, int64_t hasState, float eps,
                                              int64_t nChunks, int64_t chunkLt)
{
    const int64_t blk = (int64_t)GetBlockIdx();
    const int64_t bhc = blk / nChunks;
    const int64_t c = blk % nChunks;
    if (bhc >= B * HC) {
        return;
    }
    const int64_t b = bhc / HC;
    const int64_t hc = bhc % HC;
    const int64_t l0 = c * chunkLt;
    if (l0 >= L) {
        return;
    }
    int64_t l1 = l0 + chunkLt;
    if (l1 > L) {
        l1 = L;
    }
    const int64_t rowStart = (l0 >= H) ? (l0 - H) : 0;

    const int64_t W = H + 1;
    const int64_t Dp = EnAl(D, 64);  /* window / fp32 vector operand pitch */
    const int64_t Dpc = EnAl(D, 8);  /* cwT row pitch                     */
    const int64_t Dpad = EnAl(D, 16);
    const int64_t PS = EnAl(D, 16);  /* bf16 staging row pitch            */
    const float invD = 1.0f / (float)D;
    const float fD = (float)D;
    const int32_t nD = (int32_t)D;

    /* Rows carried per vector instruction in phase 1, and rows staged per tile.  `gr` is reduced
     * until a FULL tr = 8 row tile still fits, because the per-tile barrier count is worth more than
     * the instruction count (measured).  When the bf16 staging rows are not contiguous in UB
     * (PS != D, i.e. D = 769) the batched cast form is unavailable and gr is forced to 1. */
    const bool contig = (PS == D && Dp == D);
    int64_t gr = contig ? 4 : 1;
    int64_t fixed = 0;
    while (true) {
        fixed = (W * Dp + EN_SLACK) * 4                        /* conv window        */
                + (Dp + EN_SLACK) * 4                          /* V                  */
                + 3 * gr * (Dp + EN_SLACK) * 4                 /* A / Bb / C         */
                + gr * (Dp + EN_SLACK) * 4                     /* replicated w12     */
                + (Dp + EN_SLACK) * 4                          /* wc                 */
                + K * (Dpc + EN_SLACK) * 4                     /* cwT                */
                + (EN_CHAINS + EN_SLACK) * 4                   /* lane grid          */
                + (EN_TMP + EN_SLACK);                         /* reduce work buffer */
        if (gr <= 1 || fixed + 8 * 8 * PS <= EN_UB_BUDGET) {
            break;
        }
        gr /= 2;
    }
    int64_t tr = (EN_UB_BUDGET - fixed) / (8 * PS);
    if (tr < 1) {
        tr = 1;
    }
    if (tr > EN_TR_MAX) {
        tr = EN_TR_MAX;
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> winB, aB, bB, cB, vB, wcB, w12B, cwTB, slotB, tmpB, stgB, ostgB;
    pipe.InitBuffer(winB, (W * Dp + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(aB, (gr * (Dp + EN_SLACK) + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(bB, (gr * (Dp + EN_SLACK) + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(cB, (gr * (Dp + EN_SLACK) + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(vB, (Dp + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(wcB, (Dp + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(w12B, (gr * (Dp + EN_SLACK) + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(cwTB, K * (Dpc + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(slotB, (EN_CHAINS + EN_SLACK) * sizeof(float));
    pipe.InitBuffer(tmpB, EN_TMP + EN_SLACK);
    pipe.InitBuffer(stgB, (3 * tr * PS + EN_SLACK) * sizeof(bfloat16_t));
    pipe.InitBuffer(ostgB, (tr * PS + EN_SLACK) * sizeof(bfloat16_t));

    auto win = winB.Get<float>();
    auto A = aB.Get<float>();
    auto Bb = bB.Get<float>();
    auto C = cB.Get<float>();
    auto V = vB.Get<float>();
    auto wcv = wcB.Get<float>();
    auto w12r = w12B.Get<float>();
    auto cwTl = cwTB.Get<float>();
    auto sl = slotB.Get<float>();
    auto tmpW = tmpB.Get<float>();
    auto stg = stgB.Get<bfloat16_t>();
    auto ostg = ostgB.Get<bfloat16_t>();

    GlobalTensor<float> w1G, w2G, wcG, cwG, wsiG, wsG;
    GlobalTensor<bfloat16_t> kG, hG, vG, oG;
    w1G.SetGlobalBuffer((__gm__ float*)w1);
    w2G.SetGlobalBuffer((__gm__ float*)w2);
    wcG.SetGlobalBuffer((__gm__ float*)wc);
    cwG.SetGlobalBuffer((__gm__ float*)cw);
    wsG.SetGlobalBuffer((__gm__ float*)ws);
    kG.SetGlobalBuffer((__gm__ bfloat16_t*)keys);
    hG.SetGlobalBuffer((__gm__ bfloat16_t*)hidden);
    vG.SetGlobalBuffer((__gm__ bfloat16_t*)value);
    oG.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    if (hasState != 0) {
        wsiG.SetGlobalBuffer((__gm__ float*)wsi);
    }

    DataCopyPadExtParams<bfloat16_t> ppB{false, 0, 0, 0};
    const uint32_t rowBytes = (uint32_t)(D * (int64_t)sizeof(bfloat16_t));
    const uint32_t rowGapKeys = (uint32_t)((HC - 1) * D * (int64_t)sizeof(bfloat16_t));

    /* ---- prologue : weights + window seed + the first tile's staging ----
     * The conv weights are transposed AND pre-scaled right here instead of by a separate prep
     * kernel: the [cn, K] block is staged in the (still unused) bf16 staging buffer and each of the
     * K transposed rows is emitted by chunked Gathers with a linear byte ramp of K*4.  The byte
     * OFFSET tensor lives next to the ramp in a corner of the lane grid, which the chain does not
     * touch until tile 0.
     * The seed is needed by every chunk that starts inside the warm-up region (rowStart == 0),
     * because rows below H read the seeded slots for their early taps; it arrives as real nv values
     * and is divided by wc below, since the window carries xc/wc. */
    {
        DataCopyExtParams cp1{1, (uint32_t)(D * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> ppF{false, 0, 0, 0};
        DataCopyPad(wcv, wcG[hc * D], cp1, ppF);
        DataCopyPad(w12r, w1G[hc * D], cp1, ppF); /* w1 -> w12r[0, D)            */
        DataCopyPad(V, w2G[hc * D], cp1, ppF);    /* w2 -> V (free until phase 2) */

        auto cwStage = stg.ReinterpretCast<float>();
        auto rampI = sl.ReinterpretCast<int32_t>();
        auto offtI = rampI[(uint32_t)EN_GCH];
        auto rampU = rampI.ReinterpretCast<uint32_t>();
        auto offtU = rampU[(uint32_t)EN_GCH];
        CreateVecIndex(rampI, (int32_t)0, (int32_t)EN_GCH);
        Muls(rampI, rampI, (int32_t)(K * 4), (int32_t)EN_GCH);
        int64_t cwd = ((3 * tr * PS * 2) / 4 / K) / 8 * 8; /* channels that fit the staging area */
        if (cwd < 8) {
            cwd = 8;
        }
        if (cwd > D) {
            cwd = D;
        }
        for (int64_t c0 = 0; c0 < D; c0 += cwd) {
            int64_t cn = D - c0;
            if (cn > cwd) {
                cn = cwd;
            }
            DataCopyExtParams ccp{1, (uint32_t)(cn * K * 4), 0, 0, 0};
            DataCopyPad(cwStage, cwG[(hc * D + c0) * K], ccp, ppF);
            EnM2V();
            for (int64_t k = 0; k < K; ++k) {
                for (int64_t c1 = 0; c1 < cn; c1 += EN_GCH) {
                    int64_t cc = cn - c1;
                    if (cc > EN_GCH) {
                        cc = EN_GCH;
                    }
                    /* byte offset of source element (c1 + i)*K + k */
                    Adds(offtI, rampI, (int32_t)(k * 4 + c1 * K * 4), (int32_t)cc);
                    Gather(cwTl[(uint32_t)(k * Dpc + c0 + c1)], cwStage, offtU, (uint32_t)0,
                           (int32_t)cc);
                }
            }
            EnV2M2(); /* the next chunk's DMA must not overwrite the staging area under a read */
        }

        if (rowStart == 0 && H > 0) {
            if (hasState != 0) {
                /* one multi block copy: wsi rows are D floats apart, the UB rows want pitch Dp */
                DataCopyExtParams cps{(uint16_t)H, (uint32_t)(D * sizeof(float)), 0,
                                      (uint32_t)((Dp - EnAl(D, 8)) * (int64_t)sizeof(float)), 0};
                DataCopyPad(win[(uint32_t)Dp], wsiG[bhc * H * D], cps, ppF);
            } else {
                Duplicate(win[(uint32_t)Dp], 0.0f, (int32_t)(H * Dp));
            }
        }
    }

    int64_t t0 = rowStart;
    int64_t cnt = l1 - t0;
    if (cnt > tr) {
        cnt = tr;
    }
    {
        DataCopyExtParams cpK{(uint16_t)cnt, rowBytes, rowGapKeys, 0, 0};
        DataCopyExtParams cpV{(uint16_t)cnt, rowBytes, 0, 0, 0};
        DataCopyPad(stg[0], kG[((b * L + t0) * HC + hc) * D], cpK, ppB);
        DataCopyPad(stg[(uint32_t)(tr * PS)], hG[((b * L + t0) * HC + hc) * D], cpK, ppB);
        DataCopyPad(stg[(uint32_t)(2 * tr * PS)], vG[(b * L + t0) * D], cpV, ppB);
    }
    EnSyncAll();

    /* scale the transposed conv weights by wc, and finish w12 = w1*w2 (+ the gr-way replication:
     * a counted Mul reads `count` elements of both sources, so the product vector has to exist gr
     * times over for a batched multiply) */
    for (int64_t k = 0; k < K; ++k) {
        Mul(cwTl[(uint32_t)(k * Dpc)], cwTl[(uint32_t)(k * Dpc)], wcv, nD);
    }
    Mul(w12r, w12r, V, nD);
    for (int64_t r = 1; r < gr; ++r) {
        Adds(w12r[(uint32_t)(r * Dp)], w12r, 0.0f, nD);
    }
    if (rowStart == 0 && H > 0 && hasState != 0) {
        for (int64_t j = 0; j < H; ++j) {
            Div(win[(uint32_t)((j + 1) * Dp)], win[(uint32_t)((j + 1) * Dp)], wcv, nD);
        }
    }

    while (true) {
        /* ---- phase 1: casts and elementwise multiplies over whole GROUPS of rows ---- */
        const int64_t geff = contig ? gr : 1;
        for (int64_t g0 = 0; g0 < cnt; g0 += geff) {
            int64_t gn = cnt - g0;
            if (gn > geff) {
                gn = geff;
            }
            const int32_t nk = (int32_t)(gn * D);
            Cast(A, stg[(uint32_t)(g0 * PS)], RoundMode::CAST_NONE, nk);
            Cast(Bb, stg[(uint32_t)(tr * PS + g0 * PS)], RoundMode::CAST_NONE, nk);
            Mul(C, A, Bb, nk);   /* keys * hidden            */
            Mul(A, A, A, nk);    /* keys^2                   */
            Mul(Bb, Bb, Bb, nk); /* hidden^2                 */
            Mul(C, C, w12r, nk); /* keys * hidden * w1 * w2  */
            for (int64_t r = 0; r < gn; ++r) {
                const uint32_t lane = (uint32_t)((g0 + r) * EN_NL);
                ReduceSum<float>(sl[0 * EN_CHW + lane], A[(uint32_t)(r * D)], tmpW, nD);
                ReduceSum<float>(sl[1 * EN_CHW + lane], Bb[(uint32_t)(r * D)], tmpW, nD);
                ReduceSum<float>(sl[2 * EN_CHW + lane], C[(uint32_t)(r * D)], tmpW, nD);
            }
            Cast(C, stg[(uint32_t)(2 * tr * PS + g0 * PS)], RoundMode::CAST_NONE, nk);
            Mul(C, C, C, nk);    /* value^2 */
            for (int64_t r = 0; r < gn; ++r) {
                const uint32_t lane = (uint32_t)((g0 + r) * EN_NL);
                ReduceSum<float>(sl[3 * EN_CHW + lane], C[(uint32_t)(r * D)], tmpW, nD);
            }
        }

        /* ---- the whole chain, once per tile, on EN_CHW lanes ---- */
        Muls(sl[4 * EN_CHW], sl[0 * EN_CHW], invD, EN_CHW); /* mk per row            */
        Adds(sl[4 * EN_CHW], sl[4 * EN_CHW], eps, EN_CHW);
        Muls(sl[5 * EN_CHW], sl[1 * EN_CHW], invD, EN_CHW); /* mh per row            */
        Adds(sl[5 * EN_CHW], sl[5 * EN_CHW], eps, EN_CHW);
        Mul(sl[6 * EN_CHW], sl[4 * EN_CHW], sl[5 * EN_CHW], EN_CHW);
        Muls(sl[6 * EN_CHW], sl[6 * EN_CHW], fD, EN_CHW);
        Sqrt(sl[6 * EN_CHW], sl[6 * EN_CHW], EN_CHW);       /* den = sqrt(mk*mh*D)   */
        Div(sl[7 * EN_CHW], sl[2 * EN_CHW], sl[6 * EN_CHW], EN_CHW); /* raw          */
        Abs(sl[8 * EN_CHW], sl[7 * EN_CHW], EN_CHW);
        Maxs(sl[9 * EN_CHW], sl[8 * EN_CHW], 1e-6f, EN_CHW);
        Sqrt(sl[9 * EN_CHW], sl[9 * EN_CHW], EN_CHW);       /* a = sqrt(clamp(|raw|)) */
        Maxs(sl[10 * EN_CHW], sl[8 * EN_CHW], 1e-30f, EN_CHW);
        Div(sl[10 * EN_CHW], sl[7 * EN_CHW], sl[10 * EN_CHW], EN_CHW); /* sign(raw)  */
        Mul(sl[11 * EN_CHW], sl[10 * EN_CHW], sl[9 * EN_CHW], EN_CHW);
        Muls(sl[11 * EN_CHW], sl[11 * EN_CHW], -1.0f, EN_CHW);
        Exp(sl[12 * EN_CHW], sl[11 * EN_CHW], EN_CHW);      /* exp(-z)               */
        Adds(sl[12 * EN_CHW], sl[12 * EN_CHW], 1.0f, EN_CHW);
        Duplicate(sl[13 * EN_CHW], 1.0f, EN_CHW);
        Div(sl[13 * EN_CHW], sl[13 * EN_CHW], sl[12 * EN_CHW], EN_CHW); /* gate      */
        Mul(sl[14 * EN_CHW], sl[13 * EN_CHW], sl[13 * EN_CHW], EN_CHW);
        Mul(sl[14 * EN_CHW], sl[14 * EN_CHW], sl[3 * EN_CHW], EN_CHW);
        Muls(sl[14 * EN_CHW], sl[14 * EN_CHW], invD, EN_CHW);
        Adds(sl[14 * EN_CHW], sl[14 * EN_CHW], eps, EN_CHW);
        Sqrt(sl[14 * EN_CHW], sl[14 * EN_CHW], EN_CHW);     /* rms_vg                */
        Div(sl[15 * EN_CHW], sl[13 * EN_CHW], sl[14 * EN_CHW], EN_CHW); /* s         */

        EnSyncAll(); /* the chain results are visible to the scalar read-backs */

        /* ---- phase 2: per row epilogue ----
         * The window carries w = xc/wc = value*s (one Muls) and the taps are accumulated with
         * MulAddDst, which is the whole point: the measured row cost tracks the number of full
         * width ops.  The row order is load bearing (see the file header): write(l), then conv(l),
         * then write(l+1). */
        for (int64_t r = 0; r < cnt; ++r) {
            const int64_t l = t0 + r;
            const uint32_t slot = (uint32_t)EnMod(l, W);
            const uint32_t lane = (uint32_t)(r * EN_NL);
            auto vRow = stg[(uint32_t)(2 * tr * PS + r * PS)];

            const float gate = sl.GetValue(13 * EN_CHW + lane);
            const float s = sl.GetValue(15 * EN_CHW + lane);

            Cast(C, vRow, RoundMode::CAST_NONE, nD);
            Muls(win[slot * (uint32_t)Dp], C, s, nD);

            if (l >= l0) {
                Muls(A, C, gate, nD);                        /* vg (halo rows skip it) */
                if (H > 0) {
                    const uint32_t s0 = (uint32_t)EnMod(l - H, W);
                    Mul(Bb, cwTl, win[s0 * (uint32_t)Dp], nD);
                    for (int64_t k = 1; k < K; ++k) {
                        const uint32_t sk = (uint32_t)EnMod(l + k * dil - H, W);
                        MulAddDst(Bb, cwTl[(uint32_t)(k * Dpc)], win[sk * (uint32_t)Dp], nD);
                    }
                    /* silu(y) = y * sigmoid(y) = y / (1 + exp(-y)), like the reference */
                    Muls(V, Bb, -1.0f, nD);
                    Exp(V, V, nD);
                    Adds(V, V, 1.0f, nD);
                    Div(C, Bb, V, nD);
                    Add(Bb, A, C, nD);                   /* vg + silu(conv) */
                } else {
                    Adds(Bb, A, 0.0f, nD);               /* H == 0: vg      */
                }
                Cast(ostg[(uint32_t)(r * PS)], Bb, RoundMode::CAST_RINT, nD);
            }
        }

        /* This barrier drains the vector pipe, which orders phase 2's reads of the staging
         * buffer before the next tile's MTE2 write and the output staging writes before the
         * store's MTE3 reads below. */
        EnSyncAll();

        int64_t rOut = (l0 > t0) ? (l0 - t0) : 0;
        int64_t nOut = cnt - rOut;
        if (nOut > 0) {
            DataCopyExtParams cpO{(uint16_t)nOut, rowBytes, 0, rowGapKeys, 0};
            DataCopyPad(oG[((b * L + t0 + rOut) * HC + hc) * D], ostg[(uint32_t)(rOut * PS)], cpO);
        }

        t0 += tr;
        if (t0 >= l1) {
            break;
        }
        cnt = l1 - t0;
        if (cnt > tr) {
            cnt = tr;
        }
        {
            DataCopyExtParams cpK{(uint16_t)cnt, rowBytes, rowGapKeys, 0, 0};
            DataCopyExtParams cpV{(uint16_t)cnt, rowBytes, 0, 0, 0};
            DataCopyPad(stg[0], kG[((b * L + t0) * HC + hc) * D], cpK, ppB);
            DataCopyPad(stg[(uint32_t)(tr * PS)], hG[((b * L + t0) * HC + hc) * D], cpK, ppB);
            DataCopyPad(stg[(uint32_t)(2 * tr * PS)], vG[(b * L + t0) * D], cpV, ppB);
        }
        EnSyncAll();
    }

    /* ---- the state rows belong to the block that owns the last chunk.  The window carries
     * xc/wc, so they are scaled back up by wc here, H ops per block rather than one per row.
     * Note there is deliberately NO barrier before the Mul: it is a vector op that follows the
     * window writes in program order, and phase 2's trailing barrier already drained the pipe. ---- */
    if (l1 == L && H > 0) {
        for (int64_t j = 0; j < H; ++j) {
            const uint32_t sj = (uint32_t)EnMod(L + j - H, W);
            Mul(win[sj * (uint32_t)Dp], win[sj * (uint32_t)Dp], wcv, nD);
        }
        EnSyncAll();
        for (int64_t j = 0; j < H; ++j) {
            const uint32_t sj = (uint32_t)EnMod(L + j - H, W);
            DataCopyExtParams cpS{1, (uint32_t)(D * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsG[bhc * H * Dpad + j * Dpad], win[sj * (uint32_t)Dp], cpS);
        }
    }
}

/* =====================================================================================
 * Tiling + launch wrappers
 * ===================================================================================== */

int64_t calc_engram_state_rows(int64_t K, int64_t dil)
{
    int64_t h = (K - 1) * dil;
    if (h < 0) {
        h = 0;
    }
    return h;
}

EngramTiling calc_engram_tiling(int64_t B, int64_t L, int64_t HC, int64_t D, int64_t K, int64_t dil)
{
    EngramTiling t;
    const int64_t H = calc_engram_state_rows(K, dil);

    int64_t coreNum = 48;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        int64_t cn = (int64_t)plat->GetCoreNumAiv();
        if (cn > 0) {
            coreNum = cn;
        }
    }

    t.nDC = 1;
    t.dnPrep = D;
    t.numBlocksPrep = 0;

    /* State kernels: pick the channel tile that minimises the per-core work
     *      waves * (elements transposed by one block)
     * because these kernels are transposes and a partial last wave costs as much as a full one.
     * Pinning nDT to a single wave (the previous rule) wasted a third of the work as soon as B*HC
     * was large (B*HC = 32 forced nDT = 2, i.e. dn = 512 per block). */
    {
        const int64_t bhc = (B * HC > 0) ? B * HC : 1;
        const int64_t maxnDT = (D + 7) / 8; /* at least 8 channels per block */
        int64_t bestnDT = 1;
        int64_t bestCost = -1;
        for (int64_t n = 1; n <= maxnDT; ++n) {
            int64_t dn = (D + n - 1) / n;
            if (dn > EN_DT_MAX) {
                dn = EN_DT_MAX;
            }
            int64_t nb = bhc * n;
            int64_t waves = (nb + coreNum - 1) / coreNum;
            int64_t cost = waves * dn;
            if (bestCost < 0 || cost < bestCost) {
                bestCost = cost;
                bestnDT = n;
            }
        }
        int64_t dt = (D + bestnDT - 1) / bestnDT;
        if (dt < 8) {
            dt = 8;
        }
        if (dt > EN_DT_MAX) {
            dt = EN_DT_MAX;
        }
        int64_t nDT = (D + dt - 1) / dt;
        if (nDT < 1) {
            nDT = 1;
        }
        t.dtState = dt;
        t.nDT = nDT;
        t.numBlocksState = B * HC * t.nDT;
    }

    /* Main kernel chunking.  Measured shape of this operator: elapsed scales with the row count of
     * ONE work item and a partial last wave of blocks costs as much as a full one.  So the chunk
     * length is picked by explicitly evaluating
     *     waves * (rows owned + H)
     * over every candidate chunk count.  A single chunk needs no halo at all, hence the special
     * case. */
    const int64_t segs = (B * HC > 0) ? B * HC : 1;
    int64_t bestLt = L;
    int64_t bestCost = -1;
    for (int64_t nCh = 1; nCh <= 128; ++nCh) {
        int64_t Lt = (L + nCh - 1) / nCh;
        if (Lt < 1) {
            Lt = 1;
        }
        const int64_t nck = (L + Lt - 1) / Lt;
        const int64_t blocks = segs * nck;
        const int64_t waves = (blocks + coreNum - 1) / coreNum;
        const int64_t per = (nck == 1) ? L : (Lt + H);
        const int64_t cost = waves * per;
        if (bestCost < 0 || cost < bestCost) {
            bestCost = cost;
            bestLt = Lt;
        }
    }
    int64_t Lt = bestLt;
    if (Lt < 1) {
        Lt = 1;
    }
    t.chunkLt = Lt;
    t.nChunks = (L + Lt - 1) / Lt;
    if (t.nChunks < 1) {
        t.nChunks = 1;
    }
    t.numBlocksMain = B * HC * t.nChunks;
    return t;
}

extern "C" {

void launch_engram_state_in(GM_ADDR convState, GM_ADDR wsi, int64_t B, int64_t HC, int64_t D,
                            int64_t H, int64_t dt, int64_t nDT, void* stream)
{
    if (convState == nullptr || wsi == nullptr || H <= 0) {
        return;
    }
    const int64_t nb = B * HC * nDT;
    if (nb <= 0) {
        return;
    }
    engram_state_in_kernel<<<nb, nullptr, stream>>>(convState, wsi, B, HC, D, H, dt, nDT);
}

void launch_engram_main(GM_ADDR keys, GM_ADDR hidden, GM_ADDR value, GM_ADDR w1, GM_ADDR w2,
                        GM_ADDR wc, GM_ADDR cw, GM_ADDR wsi, GM_ADDR ws, GM_ADDR out, int64_t B,
                        int64_t L, int64_t HC, int64_t D, int64_t K, int64_t dil, int64_t H,
                        int64_t hasState, float eps, int64_t nChunks, int64_t chunkLt, void* stream)
{
    const int64_t nb = B * HC * nChunks;
    if (nb <= 0) {
        return;
    }
    engram_main_kernel<<<nb, nullptr, stream>>>(keys, hidden, value, w1, w2, wc, cw, wsi, ws, out,
                                                B, L, HC, D, K, dil, H, hasState, eps, nChunks,
                                                chunkLt);
}

void launch_engram_state_out(GM_ADDR ws, GM_ADDR sout, int64_t B, int64_t HC, int64_t D, int64_t H,
                             int64_t dt, int64_t nDT, void* stream)
{
    if (ws == nullptr || sout == nullptr || H <= 0) {
        return;
    }
    const int64_t nb = B * HC * nDT;
    if (nb <= 0) {
        return;
    }
    engram_state_out_kernel<<<nb, nullptr, stream>>>(ws, sout, B, HC, D, H, dt, nDT);
}

}  // extern "C"
