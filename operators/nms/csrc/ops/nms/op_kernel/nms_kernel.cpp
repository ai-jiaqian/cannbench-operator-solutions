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
 * \file nms_kernel.cpp
 * \brief NMS (non-maximum suppression) kernels for dav-2201 (910B), bisheng -xasc.
 *
 *   keep_indices = nms(boxes[N,4] fp32, scores[N] fp32, iou_threshold)
 *
 * Reference semantics (torch golden) that this kernel reproduces:
 *   areas   = (x2-x1)*(y2-y1)
 *   order   = scores.sort(descending=True)      # greedy starts from the highest score
 *   while order not empty:
 *       i = order[0]; keep.append(i)
 *       xx1 = max(boxes[order[1:],0], boxes[i,0]); yy1 = max(boxes[order[1:],1], boxes[i,1])
 *       xx2 = min(boxes[order[1:],2], boxes[i,2]); yy2 = min(boxes[order[1:],3], boxes[i,3])
 *       w = clamp(xx2-xx1, 0); h = clamp(yy2-yy1, 0); inter = w*h
 *       iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
 *       order = order[order[1:][iou <= thr] + 1]
 *   Candidates whose iou is NaN (NaN boxes) remove nobody => they are kept and everything else is
 *   suppressed afterwards, so NaN must SUPPRESS (`iou <= thr` is false for NaN).
 *
 * Execution shape
 * ----------------
 * Kernel A de-interleaves boxes[N,4] into four packed fp32 arrays so the IoU tile maths can run at
 * full vector width (a multi-row DataCopyPad with a 4B block, a 12B GM gap and a 0B UB gap lands
 * every row on a 32B block boundary; a second DataCopyPad drops the filler and packs).
 *
 * Kernel B runs the greedy itself.  The descending-score order is obtained WITHOUT any sort
 * primitive: the kernel keeps a per-32-element-block maximum of the still-alive scores in UB and
 * each round selects the best block (strict `>`) and then the best element inside that block
 * (strict `>`).  Strict comparisons make ties go to the smaller index, i.e. the stable descending
 * order the reference's loop relies on.  Every round then runs one vectorised IoU pass over the
 * whole candidate array and kills the non-survivors.
 *
 * Suppression uses `CompareScalar(iou, thr, CMPMODE::LE)` + `Select` + `Muls` instead of arithmetic
 * clamps: for NaN iou the comparison is false and the candidate is removed, exactly like the
 * reference's `(iou <= thr)` mask.  Arithmetic clamp tricks cannot express this because NaN would
 * survive them.
 *
 * All state lives in ONE TBuf addressed through GetWithOffset (on dav-2201 independent TBuf
 * allocations start overlapping once more than a handful of buffers are live).
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "nms_launch.h"

using namespace AscendC;

namespace {

constexpr int64_t NMS_BLOCK = 32;                 /* per-block argmax granularity */
constexpr float   NMS_NEG = -3.402823466e38f;     /* "no candidate" sentinel */
constexpr float   NMS_EPS = 1e-6f;
constexpr int64_t NMS_TILE = 512;                 /* IoU vector tile */
constexpr int64_t NMS_CHUNK = 512;                /* de-interleave chunk (rows) */
constexpr int64_t NMS_RESIDENT_MAX_NPAD = 4096;   /* above this the streamed kernel is used */

__aicore__ inline uint32_t nmsAlign32(uint32_t v)
{
    return (v + 31u) & ~31u;
}

}  // namespace

/* =====================================================================================
 * Kernel A: de-interleave boxes[N,4] into four packed arrays.
 * ===================================================================================== */
__global__ __aicore__ void nms_layout_kernel(GM_ADDR boxes, GM_ADDR x1g, GM_ADDR y1g,
                                             GM_ADDR x2g, GM_ADDR y2g, int64_t n,
                                             int64_t numBlocks)
{
    int64_t perCore = (n + numBlocks - 1) / numBlocks;
    if (perCore <= 0) {
        return;
    }
    int64_t start = (int64_t)AscendC::GetBlockIdx() * perCore;
    int64_t end = start + perCore;
    if (end > n) {
        end = n;
    }
    if (start >= end) {
        return;
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> stageBuf;
    pipe.InitBuffer(stageBuf, (uint32_t)(NMS_CHUNK * 32));
    LocalTensor<float> ub = stageBuf.Get<float>();

    GlobalTensor<float> boxesGm, g0, g1, g2, g3;
    boxesGm.SetGlobalBuffer((__gm__ float *)boxes, n * 4);
    g0.SetGlobalBuffer((__gm__ float *)x1g, n);
    g1.SetGlobalBuffer((__gm__ float *)y1g, n);
    g2.SetGlobalBuffer((__gm__ float *)x2g, n);
    g3.SetGlobalBuffer((__gm__ float *)y2g, n);

    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    int64_t pos = start;
    while (pos < end) {
        int64_t cnt = end - pos;
        if (cnt > NMS_CHUNK) {
            cnt = NMS_CHUNK;
        }
        for (int64_t c = 0; c < 4; ++c) {
            DataCopyExtParams inCp{(uint16_t)cnt, 4u, 12u, 0u, 0u};
            DataCopyPad(ub, boxesGm[pos * 4 + c], inCp, padParams);
            AscendC::PipeBarrier<PIPE_ALL>();

            DataCopyExtParams outCp{(uint16_t)cnt, 4u, 0u, 0u, 0u};
            if (c == 0) {
                DataCopyPad(g0[pos], ub, outCp);
            } else if (c == 1) {
                DataCopyPad(g1[pos], ub, outCp);
            } else if (c == 2) {
                DataCopyPad(g2[pos], ub, outCp);
            } else {
                DataCopyPad(g3[pos], ub, outCp);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        pos += cnt;
    }
}

/* =====================================================================================
 * IoU tile:  alive_j = (iou(i,j) <= thr) ? alive_j : 0     (NaN iou -> 0, i.e. suppressed)
 * ===================================================================================== */
__aicore__ inline void nmsIouTile(const LocalTensor<float> &t0, const LocalTensor<float> &t1,
                                  const LocalTensor<float> &t2, const LocalTensor<float> &t3,
                                  const LocalTensor<uint8_t> &mk,
                                  const LocalTensor<float> &x1, const LocalTensor<float> &y1,
                                  const LocalTensor<float> &x2, const LocalTensor<float> &y2,
                                  const LocalTensor<float> &alive, float x1i, float y1i,
                                  float x2i, float y2i, float ai, float thr, int32_t cnt)
{
    Maxs(t0, x1, x1i, cnt);
    Mins(t1, x2, x2i, cnt);
    Sub(t1, t1, t0, cnt);
    Maxs(t1, t1, 0.0f, cnt);

    Maxs(t2, y1, y1i, cnt);
    Mins(t3, y2, y2i, cnt);
    Sub(t3, t3, t2, cnt);
    Maxs(t3, t3, 0.0f, cnt);

    Mul(t1, t1, t3, cnt);

    Sub(t2, x2, x1, cnt);
    Sub(t0, y2, y1, cnt);
    Mul(t2, t2, t0, cnt);
    Sub(t2, t2, t1, cnt);
    Adds(t2, t2, ai, cnt);
    Adds(t2, t2, NMS_EPS, cnt);

    Div(t1, t1, t2, cnt);

    CompareScalar(mk, t1, thr, CMPMODE::LE, cnt);
    Select(t0, mk, alive, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, cnt);
    Muls(alive, t0, 1.0f, cnt);
}

/* =====================================================================================
 * Kernel B (resident): boxes and scores both fit in UB (nPad <= 4096, all real cases).
 * ===================================================================================== */
__global__ __aicore__ void nms_apply_resident(GM_ADDR x1g, GM_ADDR y1g, GM_ADDR x2g,
                                              GM_ADDR y2g, GM_ADDR scoreGm, GM_ADDR outGm,
                                              int64_t n, int64_t nPad, float thr)
{
    const uint32_t nP = (uint32_t)nPad;
    const uint32_t T = (uint32_t)NMS_TILE;
    const uint32_t G = nP / (uint32_t)NMS_BLOCK;

    const uint32_t b_x1 = 0u;
    const uint32_t b_y1 = b_x1 + 4u * nP;
    const uint32_t b_x2 = b_y1 + 4u * nP;
    const uint32_t b_y2 = b_x2 + 4u * nP;
    const uint32_t b_alive = b_y2 + 4u * nP;
    const uint32_t b_score = b_alive + 4u * nP;
    const uint32_t b_keep = b_score + 4u * nP;
    const uint32_t b_blk = b_keep + 4u * nP;
    const uint32_t b_t0 = nmsAlign32(b_blk + 4u * G);
    const uint32_t b_t1 = b_t0 + 4u * T;
    const uint32_t b_t2 = b_t1 + 4u * T;
    const uint32_t b_t3 = b_t2 + 4u * T;
    const uint32_t b_mk = b_t3 + 4u * T;
    const uint32_t b_i64 = nmsAlign32(b_mk + T);
    const uint32_t total = b_i64 + 8u * T;

    TPipe pipe;
    TBuf<TPosition::VECCALC> wb;
    pipe.InitBuffer(wb, total);

    LocalTensor<float> x1 = wb.GetWithOffset<float>(nP, b_x1);
    LocalTensor<float> y1 = wb.GetWithOffset<float>(nP, b_y1);
    LocalTensor<float> x2 = wb.GetWithOffset<float>(nP, b_x2);
    LocalTensor<float> y2 = wb.GetWithOffset<float>(nP, b_y2);
    LocalTensor<float> alive = wb.GetWithOffset<float>(nP, b_alive);
    LocalTensor<float> score = wb.GetWithOffset<float>(nP, b_score);
    LocalTensor<int32_t> keep = wb.GetWithOffset<int32_t>(nP, b_keep);
    LocalTensor<float> blkMax = wb.GetWithOffset<float>(G, b_blk);
    LocalTensor<float> t0 = wb.GetWithOffset<float>(T, b_t0);
    LocalTensor<float> t1 = wb.GetWithOffset<float>(T, b_t1);
    LocalTensor<float> t2 = wb.GetWithOffset<float>(T, b_t2);
    LocalTensor<float> t3 = wb.GetWithOffset<float>(T, b_t3);
    LocalTensor<uint8_t> mk = wb.GetWithOffset<uint8_t>(T, b_mk);
    LocalTensor<int64_t> i64 = wb.GetWithOffset<int64_t>(T, b_i64);

    GlobalTensor<float> gX1, gY1, gX2, gY2, gScore;
    gX1.SetGlobalBuffer((__gm__ float *)x1g, nPad);
    gY1.SetGlobalBuffer((__gm__ float *)y1g, nPad);
    gX2.SetGlobalBuffer((__gm__ float *)x2g, nPad);
    gY2.SetGlobalBuffer((__gm__ float *)y2g, nPad);
    gScore.SetGlobalBuffer((__gm__ float *)scoreGm, n);

    /* Whole-extent zeroing before any load: a slice that starts at a non-32B-aligned address makes
     * the VEC unit fault (507035 "UB address accessed by the VEC instruction is not aligned"). */
    Duplicate(x1, 0.0f, (int32_t)nP);
    Duplicate(y1, 0.0f, (int32_t)nP);
    Duplicate(x2, 0.0f, (int32_t)nP);
    Duplicate(y2, 0.0f, (int32_t)nP);
    Duplicate(alive, 0.0f, (int32_t)nP);
    Duplicate(keep, (int32_t)0, (int32_t)nP);
    Duplicate(score, NMS_NEG, (int32_t)nP);
    AscendC::PipeBarrier<PIPE_ALL>();

    DataCopyPadExtParams<float> padParams{false, 0, 0, NMS_NEG};
    DataCopyExtParams cpN{1, (uint32_t)(n * (int64_t)sizeof(float)), 0, 0, 0};
    DataCopyPad(x1, gX1, cpN, padParams);
    DataCopyPad(y1, gY1, cpN, padParams);
    DataCopyPad(x2, gX2, cpN, padParams);
    DataCopyPad(y2, gY2, cpN, padParams);
    DataCopyPad(score, gScore, cpN, padParams);
    AscendC::PipeBarrier<PIPE_ALL>();

    Duplicate(alive, 1.0f, (int32_t)n);
    AscendC::PipeBarrier<PIPE_ALL>();

    /* seed the per-block maxima of the alive scores */
    for (uint32_t b = 0; b < G; ++b) {
        const uint32_t base = b * (uint32_t)NMS_BLOCK;
        float m = NMS_NEG;
        for (uint32_t j = 0; j < (uint32_t)NMS_BLOCK; ++j) {
            const uint32_t idx = base + j;
            if (idx >= nP) {
                break;
            }
            if (alive.GetValue(idx) != 0.0f) {
                const float s = score.GetValue(idx);
                if (s > m) {
                    m = s;
                }
            }
        }
        blkMax.SetValue(b, m);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t kept = 0;
    for (int64_t k = 0; k < n; ++k) {
        float bv = NMS_NEG;
        int32_t bb = -1;
        for (uint32_t b = 0; b < G; ++b) {
            const float m = blkMax.GetValue(b);
            if (m > bv) {
                bv = m;
                bb = (int32_t)b;
            }
        }
        if (bb < 0) {
            break;
        }
        const uint32_t base = (uint32_t)bb * (uint32_t)NMS_BLOCK;
        float ev = NMS_NEG;
        int32_t ei = -1;
        for (uint32_t j = 0; j < (uint32_t)NMS_BLOCK; ++j) {
            const uint32_t idx = base + j;
            if (idx >= nP) {
                break;
            }
            if (alive.GetValue(idx) == 0.0f) {
                continue;
            }
            const float s = score.GetValue(idx);
            if (s > ev) {
                ev = s;
                ei = (int32_t)idx;
            }
        }
        if (ei < 0) {
            break;
        }
        keep.SetValue(kept, ei);
        ++kept;
        alive.SetValue(ei, 0.0f);
        {
            float m = NMS_NEG;
            for (uint32_t j = 0; j < (uint32_t)NMS_BLOCK; ++j) {
                const uint32_t idx = base + j;
                if (idx >= nP) {
                    break;
                }
                if (alive.GetValue(idx) != 0.0f) {
                    const float s = score.GetValue(idx);
                    if (s > m) {
                        m = s;
                    }
                }
            }
            blkMax.SetValue(bb, m);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        const float x1i = x1.GetValue(ei);
        const float y1i = y1.GetValue(ei);
        const float x2i = x2.GetValue(ei);
        const float y2i = y2.GetValue(ei);
        const float ai = (x2i - x1i) * (y2i - y1i);

        for (int64_t j = 0; j < nPad; j += NMS_TILE) {
            const int64_t left = nPad - j;
            const int32_t cnt = (int32_t)(left < NMS_TILE ? left : NMS_TILE);
            nmsIouTile(t0, t1, t2, t3, mk, x1[j], y1[j], x2[j], y2[j], alive[j], x1i, y1i,
                       x2i, y2i, ai, thr, cnt);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::PipeBarrier<PIPE_ALL>();
    GlobalTensor<int64_t> outGmT;
    outGmT.SetGlobalBuffer((__gm__ int64_t *)outGm, n);
    for (int64_t off = 0; off < n; off += NMS_TILE) {
        const int64_t left = n - off;
        const int32_t cnt = (int32_t)(left < NMS_TILE ? left : NMS_TILE);
        Cast(i64, keep[off], RoundMode::CAST_NONE, cnt);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cpO{1, (uint32_t)(cnt * (int64_t)sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(outGmT[off], i64, cpO);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

/* =====================================================================================
 * Kernel B (streamed): only scores / alive / keep stay resident, boxes come from GM.
 * ===================================================================================== */
__global__ __aicore__ void nms_apply_streamed(GM_ADDR x1g, GM_ADDR y1g, GM_ADDR x2g,
                                              GM_ADDR y2g, GM_ADDR scoreGm, GM_ADDR outGm,
                                              int64_t n, int64_t nPad, float thr)
{
    const uint32_t nP = (uint32_t)nPad;
    const uint32_t T = (uint32_t)NMS_TILE;
    const uint32_t G = nP / (uint32_t)NMS_BLOCK;

    const uint32_t b_alive = 0u;
    const uint32_t b_score = b_alive + 4u * nP;
    const uint32_t b_keep = b_score + 4u * nP;
    const uint32_t b_blk = b_keep + 4u * nP;
    const uint32_t b_x1 = nmsAlign32(b_blk + 4u * G);
    const uint32_t b_y1 = b_x1 + 4u * T;
    const uint32_t b_x2 = b_y1 + 4u * T;
    const uint32_t b_y2 = b_x2 + 4u * T;
    const uint32_t b_t0 = b_y2 + 4u * T;
    const uint32_t b_t1 = b_t0 + 4u * T;
    const uint32_t b_t2 = b_t1 + 4u * T;
    const uint32_t b_t3 = b_t2 + 4u * T;
    const uint32_t b_mk = b_t3 + 4u * T;
    const uint32_t b_i64 = nmsAlign32(b_mk + T);
    const uint32_t total = b_i64 + 8u * T;

    TPipe pipe;
    TBuf<TPosition::VECCALC> wb;
    pipe.InitBuffer(wb, total);

    LocalTensor<float> alive = wb.GetWithOffset<float>(nP, b_alive);
    LocalTensor<float> score = wb.GetWithOffset<float>(nP, b_score);
    LocalTensor<int32_t> keep = wb.GetWithOffset<int32_t>(nP, b_keep);
    LocalTensor<float> blkMax = wb.GetWithOffset<float>(G, b_blk);
    LocalTensor<float> x1 = wb.GetWithOffset<float>(T, b_x1);
    LocalTensor<float> y1 = wb.GetWithOffset<float>(T, b_y1);
    LocalTensor<float> x2 = wb.GetWithOffset<float>(T, b_x2);
    LocalTensor<float> y2 = wb.GetWithOffset<float>(T, b_y2);
    LocalTensor<float> t0 = wb.GetWithOffset<float>(T, b_t0);
    LocalTensor<float> t1 = wb.GetWithOffset<float>(T, b_t1);
    LocalTensor<float> t2 = wb.GetWithOffset<float>(T, b_t2);
    LocalTensor<float> t3 = wb.GetWithOffset<float>(T, b_t3);
    LocalTensor<uint8_t> mk = wb.GetWithOffset<uint8_t>(T, b_mk);
    LocalTensor<int64_t> i64 = wb.GetWithOffset<int64_t>(T, b_i64);

    GlobalTensor<float> gX1, gY1, gX2, gY2, gScore;
    gX1.SetGlobalBuffer((__gm__ float *)x1g, nPad);
    gY1.SetGlobalBuffer((__gm__ float *)y1g, nPad);
    gX2.SetGlobalBuffer((__gm__ float *)x2g, nPad);
    gY2.SetGlobalBuffer((__gm__ float *)y2g, nPad);
    gScore.SetGlobalBuffer((__gm__ float *)scoreGm, n);

    Duplicate(alive, 0.0f, (int32_t)nP);
    Duplicate(keep, (int32_t)0, (int32_t)nP);
    Duplicate(score, NMS_NEG, (int32_t)nP);
    AscendC::PipeBarrier<PIPE_ALL>();

    DataCopyPadExtParams<float> padParams{false, 0, 0, NMS_NEG};
    DataCopyExtParams cpN{1, (uint32_t)(n * (int64_t)sizeof(float)), 0, 0, 0};
    DataCopyPad(score, gScore, cpN, padParams);
    Duplicate(alive, 1.0f, (int32_t)n);
    AscendC::PipeBarrier<PIPE_ALL>();

    for (uint32_t b = 0; b < G; ++b) {
        const uint32_t base = b * (uint32_t)NMS_BLOCK;
        float m = NMS_NEG;
        for (uint32_t j = 0; j < (uint32_t)NMS_BLOCK; ++j) {
            const uint32_t idx = base + j;
            if (idx >= nP) {
                break;
            }
            if (alive.GetValue(idx) != 0.0f) {
                const float s = score.GetValue(idx);
                if (s > m) {
                    m = s;
                }
            }
        }
        blkMax.SetValue(b, m);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    int64_t kept = 0;
    for (int64_t k = 0; k < n; ++k) {
        float bv = NMS_NEG;
        int32_t bb = -1;
        for (uint32_t b = 0; b < G; ++b) {
            const float m = blkMax.GetValue(b);
            if (m > bv) {
                bv = m;
                bb = (int32_t)b;
            }
        }
        if (bb < 0) {
            break;
        }
        const uint32_t base = (uint32_t)bb * (uint32_t)NMS_BLOCK;
        float ev = NMS_NEG;
        int32_t ei = -1;
        for (uint32_t j = 0; j < (uint32_t)NMS_BLOCK; ++j) {
            const uint32_t idx = base + j;
            if (idx >= nP) {
                break;
            }
            if (alive.GetValue(idx) == 0.0f) {
                continue;
            }
            const float s = score.GetValue(idx);
            if (s > ev) {
                ev = s;
                ei = (int32_t)idx;
            }
        }
        if (ei < 0) {
            break;
        }
        keep.SetValue(kept, ei);
        ++kept;
        alive.SetValue(ei, 0.0f);
        {
            float m = NMS_NEG;
            for (uint32_t j = 0; j < (uint32_t)NMS_BLOCK; ++j) {
                const uint32_t idx = base + j;
                if (idx >= nP) {
                    break;
                }
                if (alive.GetValue(idx) != 0.0f) {
                    const float s = score.GetValue(idx);
                    if (s > m) {
                        m = s;
                    }
                }
            }
            blkMax.SetValue(bb, m);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        const float x1i = gX1.GetValue(ei);
        const float y1i = gY1.GetValue(ei);
        const float x2i = gX2.GetValue(ei);
        const float y2i = gY2.GetValue(ei);
        const float ai = (x2i - x1i) * (y2i - y1i);

        for (int64_t j = 0; j < nPad; j += NMS_TILE) {
            const int64_t left = nPad - j;
            const int32_t cnt = (int32_t)(left < NMS_TILE ? left : NMS_TILE);
            DataCopyExtParams cpc{1, (uint32_t)(cnt * (int64_t)sizeof(float)), 0, 0, 0};
            DataCopyPad(x1, gX1[j], cpc, padParams);
            DataCopyPad(y1, gY1[j], cpc, padParams);
            DataCopyPad(x2, gX2[j], cpc, padParams);
            DataCopyPad(y2, gY2[j], cpc, padParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            nmsIouTile(t0, t1, t2, t3, mk, x1, y1, x2, y2, alive[j], x1i, y1i, x2i, y2i, ai,
                       thr, cnt);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    AscendC::PipeBarrier<PIPE_ALL>();
    GlobalTensor<int64_t> outGmT;
    outGmT.SetGlobalBuffer((__gm__ int64_t *)outGm, n);
    for (int64_t off = 0; off < n; off += NMS_TILE) {
        const int64_t left = n - off;
        const int32_t cnt = (int32_t)(left < NMS_TILE ? left : NMS_TILE);
        Cast(i64, keep[off], RoundMode::CAST_NONE, cnt);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cpO{1, (uint32_t)(cnt * (int64_t)sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(outGmT[off], i64, cpO);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

/* =====================================================================================
 * Launch wrappers (extern "C" so the g++ plugin can call them)
 * ===================================================================================== */
extern "C" {

void launch_nms_layout_kernel(GM_ADDR boxes, GM_ADDR x1, GM_ADDR y1, GM_ADDR x2, GM_ADDR y2,
                              int64_t n, int64_t numBlocks, void* stream)
{
    nms_layout_kernel<<<numBlocks, nullptr, stream>>>(boxes, x1, y1, x2, y2, n, numBlocks);
}

void launch_nms_apply_kernel(GM_ADDR x1, GM_ADDR y1, GM_ADDR x2, GM_ADDR y2, GM_ADDR scores,
                             GM_ADDR outIdx, int64_t n, int64_t nPad, float iouThreshold,
                             void* stream)
{
    if (nPad <= NMS_RESIDENT_MAX_NPAD) {
        nms_apply_resident<<<1, nullptr, stream>>>(x1, y1, x2, y2, scores, outIdx, n, nPad,
                                                   iouThreshold);
    } else {
        nms_apply_streamed<<<1, nullptr, stream>>>(x1, y1, x2, y2, scores, outIdx, n, nPad,
                                                   iouThreshold);
    }
}
}
