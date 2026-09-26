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
 * \file roi_align_kernel.cpp
 * \brief ROIAlign (torchvision / pure-python reference compatible bilinear ROI pooling) device kernel
 *        plus host tiling.  Compiled with bisheng + -xasc for dav-2201 (Ascend 910B / Atlas A2).
 *
 * The reference arithmetic is reproduced operation by operation:
 *
 *   offset = aligned ? 0.5f : 0.f
 *   rsw = x0*ss - offset ; rsh = y0*ss - offset ; rew = x1*ss - offset ; reh = y1*ss - offset
 *   rw = rew - rsw ; rh = reh - rsh
 *   if (!aligned) { rw = max(rw,1) ; rh = max(rh,1) }
 *   binw = rw/ow ; binh = rh/oh
 *   if (sr > 0) { gh = gw = sr ; cnt = sr*sr ; ghf = gwf = (float)sr }
 *   else        { ghf = ceil(rh/oh) ; gwf = ceil(rw/ow) ; gh = (int)ghf ; gw = (int)gwf ;
 *                 cnt = max(ghf*gwf, 1) }
 *   binhg = binh/ghf ; binwg = binw/gwf
 *   for iy in [0,gh):
 *      y = (rsh + ph*binh) + (iy+0.5f)*binhg ;  y = max(y,0)
 *      yl = (int)y ; ylc = min(yl,H-1) ; yhc = min(yl+1,H-1) ; ly = y - ylc ; hy = 1-ly
 *      for ix in [0,gw):
 *         x = (rsw + pw*binw) + (ix+0.5f)*binwg ; x = max(x,0)
 *         xl = (int)x ; xlc = min(xl,W-1) ; xhc = min(xl+1,W-1) ; lx = x - xlc ; hx = 1-lx
 *         t1 = hy*hx ; t2 = hy*lx ; t3 = ly*hx ; t4 = ly*lx
 *         val = ((t1*v1 + t2*v2) + t3*v3) + t4*v4          (left associative, 7 rounded ops)
 *         acc += val
 *   out = acc / cnt
 *
 * FMA / Axpy are deliberately NOT used and every SCALAR float operation goes through a volatile store
 * helper: the reference evaluates each product and sum as a separately rounded float operation and the
 * small-value / cancellation accuracy gates are sensitive to a 1 ulp shift of the sampling coordinates.
 *
 * Data layout
 * -----------
 *   work unit : (roi n, channel block [c0, c0+Cb))
 *   lane      : l = c*OWp + pw over the CbP channels of the block and the OWp output columns, with
 *               OWp = pow2ceil(ow) so that the lanes carry no more columns than the ROI has.
 *
 *   For every (output row ph, vertical sample iy) the two source rows (ylc,yhc) of the whole channel
 *   block are staged with ONE strided DataCopyPad each.  The four bilinear corners are fetched with four
 *   vector Gathers whose offset vector is c*pitchRun*esz + (icol-xs)*esz.
 *
 *   The x geometry (gather byte offsets, hx, lx) depends only on (roi, ix), so it is evaluated once per
 *   unit into a small UB cache and reused by every (ph, iy, ix) step, leaving 16 vector instructions in
 *   the inner loop.
 *
 *   The store needs the UB source blocks on a 32 byte pitch, which OWp*esz does not always reach (fp16
 *   with OWp=8, fp32 with OWp<8).  In that case the row is repacked once per output row with a single
 *   Gather into a padded [CbP][OP] buffer whose pitch OP*esz is 32 byte aligned, so the narrow lane
 *   layout can be kept without inflating the number of working lanes.
 */

#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "roi_align_launch.h"

using namespace AscendC;

/*! hard cap on the number of lanes (channels x output columns) of one vector pass */
constexpr int64_t RA_LANE_MAX = 2048;
/*! hard cap on the element count of one Gather */
constexpr int64_t RA_GATHER_MAX = 8192;

int64_t RaAlignUpH(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

int64_t RaPow2Ceil(int64_t v)
{
    int64_t r = 1;
    while (r < v) {
        r <<= 1;
    }
    return r;
}

/*! Host tiling. */
RoiAlignTiling calc_roi_align_tiling(int64_t B, int64_t C, int64_t H, int64_t W, int64_t N,
                                     int64_t oh, int64_t ow, int64_t esz, int64_t sr)
{
    (void)B;
    (void)H;
    (void)N;
    (void)oh;
    RoiAlignTiling t;
    t.CbP = 8;
    t.nCb = 1;
    t.OWp = 1;
    t.OP = 1;
    t.perm = 0;
    t.pitchMax = 16;
    t.gwCap = 1;
    t.numUnits = 0;
    t.numBlocks = 1;
    t.unitsPerCore = 1;
    if (C <= 0 || W <= 0 || ow <= 0) {
        return t;
    }

    int64_t OWp = RaPow2Ceil(ow);
    int64_t OP = RaPow2Ceil(RaAlignUpH(ow * esz, 32) / esz);
    int64_t perm = (OWp * esz < RaAlignUpH(ow * esz, 32)) ? 1 : 0;
    t.OWp = OWp;
    t.OP = OP;
    t.perm = perm;

    int64_t pitchMaxEl = RaAlignUpH(W * esz, 32) / esz;
    t.pitchMax = pitchMaxEl;

    int64_t ubSize = 192 * 1024;
    int64_t cores = 48;
    auto plat = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (plat != nullptr) {
        uint64_t sz = 0;
        plat->GetCoreMemSize(platform_ascendc::CoreMemType::UB, sz);
        if (sz > 0) {
            ubSize = (int64_t)sz;
        }
        int64_t cn = plat->GetCoreNumAiv();
        if (cn > 0) {
            cores = cn;
        }
    }
    int64_t budget = (ubSize * 17) / 20;

    /* number of ix slots whose x geometry is kept in UB (sr > 0 => exactly sr, else a small bound) */
    int64_t gwT = (sr > 0) ? sr : 4;
    if (gwT > 16) {
        gwT = 16;
    }
    if (gwT < 1) {
        gwT = 1;
    }

    int64_t cbMax = RA_LANE_MAX / OWp;
    if (cbMax > ((C + 7) / 8) * 8) {
        cbMax = ((C + 7) / 8) * 8;
    }
    if (cbMax < 8) {
        cbMax = 8;
    }
    if (perm && OP * cbMax > RA_GATHER_MAX) {
        cbMax = ((RA_GATHER_MAX / OP) / 8) * 8;
    }

    int64_t bestCb = 0;
    int64_t bestGw = 1;
    int64_t bestCost = -1;
    for (int64_t cb = 8; cb <= cbMax; cb += 8) {
        int64_t lanes = cb * OWp;
        int64_t base = 80 * lanes + 2 * cb * pitchMaxEl * esz + 32;
        if (esz == 2) {
            base += lanes * 2;
        }
        if (perm) {
            base += cb * OP * esz + 16 * cb * OP;
        }
        if (base + 32 * lanes + 8192 > budget) {
            break;
        }
        int64_t gwCap = (budget - base - 8192 - 16 * lanes) / (16 * lanes);
        if (gwCap > gwT) {
            gwCap = gwT;
        }
        if (gwCap < 1) {
            break;
        }
        int64_t nCb = (C + cb - 1) / cb;
        int64_t nOpsCached = (esz == 2) ? 20 : 16;
        int64_t nOpsPlain = (esz == 2) ? 30 : 26;
        int64_t nOps = (gwCap >= gwT) ? nOpsCached : nOpsPlain;
        int64_t cost = nCb * nOps * (32000 + 74 * (lanes / 10));
        if (bestCost < 0 || cost < bestCost) {
            bestCost = cost;
            bestCb = cb;
            bestGw = gwCap;
        }
    }
    if (bestCb == 0) {
        bestCb = 8;
        bestGw = 1;
    }
    t.CbP = bestCb;
    t.nCb = (C + bestCb - 1) / bestCb;
    t.gwCap = bestGw;
    t.numUnits = N * t.nCb;
    t.numBlocks = std::min<int64_t>(cores, t.numUnits);
    if (t.numBlocks < 1) {
        t.numBlocks = 1;
    }
    t.unitsPerCore = (t.numUnits + t.numBlocks - 1) / t.numBlocks;
    return t;
}

/* ------------------------------------------------------------------ *
 * Non contractible scalar float helpers (see the file header).
 * ------------------------------------------------------------------ */
__aicore__ inline float RaMul(float a, float b)
{
    volatile float r = a * b;
    return r;
}
__aicore__ inline float RaAdd(float a, float b)
{
    volatile float r = a + b;
    return r;
}
__aicore__ inline float RaSub(float a, float b)
{
    volatile float r = a - b;
    return r;
}
__aicore__ inline float RaDiv(float a, float b)
{
    volatile float r = a / b;
    return r;
}

__aicore__ inline float RaCeilF(float v)
{
    int32_t i = (int32_t)v;
    if ((float)i < v) {
        i += 1;
    }
    return (float)i;
}

__aicore__ inline int64_t RaAlignUpD(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

__aicore__ inline float RaToFf(half v)
{
    return (float)v;
}

__aicore__ inline float RaToFf(float v)
{
    return v;
}

/*! x geometry of one ix step: clamped column index, fractional weight and its complement. */
__aicore__ inline void RaGeom(LocalTensor<float> pwS, LocalTensor<float> xvS,
                              LocalTensor<int32_t> ivD, LocalTensor<float> hxD, LocalTensor<float> lxD,
                              float binw, float rsw, float xshift, int32_t Wm1i, int32_t LANES)
{
    Muls(xvS, pwS, binw, LANES);
    Adds(xvS, xvS, rsw, LANES);
    Adds(xvS, xvS, xshift, LANES);
    Maxs(xvS, xvS, 0.0f, LANES);
    Cast(ivD, xvS, RoundMode::CAST_FLOOR, LANES);
    Mins(ivD, ivD, Wm1i, LANES);
    Cast(hxD, ivD, RoundMode::CAST_NONE, LANES);
    Sub(lxD, xvS, hxD, LANES);
    Muls(hxD, lxD, -1.0f, LANES);
    Adds(hxD, hxD, 1.0f, LANES);
}

/*! gather byte offsets of the two sampled columns of one ix step. */
__aicore__ inline void RaOffsets(LocalTensor<int32_t> iv, LocalTensor<int32_t> coffS,
                                 LocalTensor<int32_t> offMaxS, LocalTensor<int32_t> oL, LocalTensor<int32_t> oH,
                                 int32_t esz, int32_t LANES)
{
    Muls(oL, iv, esz, LANES);
    Add(oL, oL, coffS, LANES);
    Adds(oH, oL, esz, LANES);
    Min(oH, oH, offMaxS, LANES);
}

template <typename T>
__global__ __aicore__ void roi_align_kernel(
    GM_ADDR x, GM_ADDR boxes, GM_ADDR y,
    int64_t B, int64_t C, int64_t H, int64_t W, int64_t N,
    int64_t oh, int64_t ow, float ss, int64_t sr, int64_t aligned,
    int64_t CbP, int64_t nCb, int64_t OWp, int64_t OP, int64_t perm,
    int64_t pitchMax, int64_t gwCap, int64_t numUnits, int64_t unitsPerCore)
{
    constexpr bool IS_HALF = std::is_same<T, half>::value;
    const int32_t esz = (int32_t)sizeof(T);

    const int32_t LANES = (int32_t)(CbP * OWp);
    const int32_t NP = (int32_t)(CbP * OP);
    const int64_t ROW_BYTES = CbP * pitchMax * (int64_t)sizeof(T);
    const int32_t GWC = (int32_t)gwCap;
    const bool PERM = (perm != 0);

    int64_t bid = (int64_t)GetBlockIdx();
    int64_t uBeg = bid * unitsPerCore;
    int64_t uEnd = uBeg + unitsPerCore;
    if (uEnd > numUnits) {
        uEnd = numUnits;
    }
    if (uBeg >= uEnd) {
        return;
    }

    GlobalTensor<T> xG;
    GlobalTensor<T> bG;
    GlobalTensor<T> yG;
    xG.SetGlobalBuffer((__gm__ T *)x);
    bG.SetGlobalBuffer((__gm__ T *)boxes);
    yG.SetGlobalBuffer((__gm__ T *)y);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 1> outQ;
    TQue<QuePosition::VECOUT, 1> outH;
    TQue<QuePosition::VECOUT, 1> outP;
    TBuf<TPosition::VECCALC> f32B;
    TBuf<TPosition::VECCALC> i32B;
    TBuf<TPosition::VECCALC> hB;
    TBuf<TPosition::VECCALC> cacheF;
    TBuf<TPosition::VECCALC> cacheO;
    TBuf<TPosition::VECCALC> cacheI;
    TBuf<TPosition::VECCALC> permF;
    pipe.InitBuffer(inQ, 2, ROW_BYTES);
    pipe.InitBuffer(outQ, 1, (int64_t)LANES * 4);
    pipe.InitBuffer(f32B, (int64_t)LANES * 4 * 9);
    pipe.InitBuffer(i32B, (int64_t)LANES * 4 * 5);
    pipe.InitBuffer(hB, (int64_t)LANES * 2 * 2);
    pipe.InitBuffer(cacheF, (int64_t)LANES * 4 * (2 * GWC + 3));
    pipe.InitBuffer(cacheO, (int64_t)LANES * 4 * (2 * GWC));
    pipe.InitBuffer(cacheI, (int64_t)LANES * 4);
    pipe.InitBuffer(outH, 1, IS_HALF ? (int64_t)LANES * 2 : 32);
    pipe.InitBuffer(outP, 1, PERM ? (int64_t)NP * esz : 32);
    pipe.InitBuffer(permF, PERM ? (int64_t)NP * 16 : 64);

    LocalTensor<float> f32 = f32B.Get<float>();
    LocalTensor<int32_t> i32 = i32B.Get<int32_t>();
    LocalTensor<half> hb = hB.Get<half>();
    LocalTensor<float> cf = cacheF.Get<float>();
    LocalTensor<int32_t> co = cacheO.Get<int32_t>();
    LocalTensor<int32_t> ci = cacheI.Get<int32_t>();
    LocalTensor<int32_t> pf = permF.Get<int32_t>();

    LocalTensor<float> pwF = f32[0];
    LocalTensor<float> t1v = f32[LANES];
    LocalTensor<float> t2v = f32[LANES * 2];
    LocalTensor<float> t3v = f32[LANES * 3];
    LocalTensor<float> t4v = f32[LANES * 4];
    LocalTensor<float> v1f = f32[LANES * 5];
    LocalTensor<float> v2f = f32[LANES * 6];
    LocalTensor<float> sv = f32[LANES * 7];
    LocalTensor<float> cntV = f32[LANES * 8];
    LocalTensor<int32_t> hiI = i32[0];
    LocalTensor<int32_t> coff = i32[LANES];
    LocalTensor<int32_t> offMax = i32[LANES * 2];
    LocalTensor<int32_t> offXl = i32[LANES * 3];
    LocalTensor<int32_t> offXh = i32[LANES * 4];
    LocalTensor<half> v1h = hb[0];
    LocalTensor<half> v2h = hb[LANES];

    LocalTensor<float> hxArr = cf[0];
    LocalTensor<float> lxArr = cf[(int64_t)LANES * GWC];
    LocalTensor<float> xvS = cf[(int64_t)LANES * 2 * GWC];
    LocalTensor<float> hxS = cf[(int64_t)LANES * (2 * GWC + 1)];
    LocalTensor<float> lxS = cf[(int64_t)LANES * (2 * GWC + 2)];
    LocalTensor<int32_t> offLArr = co[0];
    LocalTensor<int32_t> offHArr = co[(int64_t)LANES * GWC];
    LocalTensor<int32_t> ivS = ci[0];

    /* pw = l % OWp and hi = l / OWp, built once per core with pure vector instructions. */
    CreateVecIndex(ivS, (int32_t)0, LANES);
    Cast(v1f, ivS, RoundMode::CAST_NONE, LANES);
    if (OWp > 1) {
        Muls(pwF, v1f, 1.0f / (float)OWp, LANES);
        Cast(hiI, pwF, RoundMode::CAST_FLOOR, LANES);
        Cast(v2f, hiI, RoundMode::CAST_NONE, LANES);
        Muls(v2f, v2f, (float)OWp, LANES);
        Sub(pwF, v1f, v2f, LANES);
    } else {
        Duplicate(pwF, 0.0f, LANES);
        Cast(hiI, v1f, RoundMode::CAST_FLOOR, LANES);
    }

    if (PERM) {
        /* byte offsets repacking [CbP][OWp] into [CbP][OP], evaluated once per core */
        LocalTensor<int32_t> jv = pf[0];
        LocalTensor<float> fv = pf[(int64_t)NP].ReinterpretCast<float>();
        LocalTensor<int32_t> cv = pf[(int64_t)NP * 2];
        LocalTensor<int32_t> rv = pf[(int64_t)NP * 3];
        CreateVecIndex(jv, (int32_t)0, NP);
        Cast(fv, jv, RoundMode::CAST_NONE, NP);
        Muls(fv, fv, 1.0f / (float)OP, NP);
        Cast(cv, fv, RoundMode::CAST_FLOOR, NP);
        Cast(fv, cv, RoundMode::CAST_NONE, NP);
        Muls(fv, fv, (float)OP, NP);
        Cast(rv, fv, RoundMode::CAST_RINT, NP);
        Sub(rv, jv, rv, NP);
        Mins(rv, rv, (int32_t)(OWp - 1), NP);
        Muls(cv, cv, (int32_t)OWp, NP);
        Add(cv, cv, rv, NP);
        Muls(jv, cv, esz, NP);
    }

    const int32_t Wm1i = (int32_t)(W - 1);
    const int32_t Hm1i = (int32_t)(H - 1);

    for (int64_t u = uBeg; u < uEnd; ++u) {
        int64_t n = u / nCb;
        int64_t cb = u - n * nCb;
        int64_t c0 = cb * CbP;
        int64_t Cb = C - c0;
        if (Cb > CbP) {
            Cb = CbP;
        }

        int64_t bo = n * 5;
        float f0 = RaToFf(bG.GetValue(bo + 0));
        float f1 = RaToFf(bG.GetValue(bo + 1));
        float f2 = RaToFf(bG.GetValue(bo + 2));
        float f3 = RaToFf(bG.GetValue(bo + 3));
        float f4 = RaToFf(bG.GetValue(bo + 4));
        int64_t b = (int64_t)f0;
        if (b < 0) {
            b = 0;
        }
        if (b > B - 1) {
            b = B - 1;
        }

        const float offv = aligned ? 0.5f : 0.0f;
        const float rsw = RaSub(RaMul(f1, ss), offv);
        const float rsh = RaSub(RaMul(f2, ss), offv);
        const float rew = RaSub(RaMul(f3, ss), offv);
        const float reh = RaSub(RaMul(f4, ss), offv);
        float rw = RaSub(rew, rsw);
        float rh = RaSub(reh, rsh);
        if (aligned == 0) {
            if (rw < 1.0f) {
                rw = 1.0f;
            }
            if (rh < 1.0f) {
                rh = 1.0f;
            }
        }
        const float binw = RaDiv(rw, (float)ow);
        const float binh = RaDiv(rh, (float)oh);

        int64_t gh;
        int64_t gw;
        float ghf;
        float gwf;
        float cnt;
        if (sr > 0) {
            gh = sr;
            gw = sr;
            ghf = (float)sr;
            gwf = (float)sr;
            cnt = (float)(sr * sr);
        } else {
            ghf = RaCeilF(RaDiv(rh, (float)oh));
            gwf = RaCeilF(RaDiv(rw, (float)ow));
            gh = (int64_t)ghf;
            gw = (int64_t)gwf;
            if (gh < 1) {
                gh = 1;
            }
            if (gw < 1) {
                gw = 1;
            }
            cnt = RaMul(ghf, gwf);
            if (cnt < 1.0f) {
                cnt = 1.0f;
            }
        }
        const float binhg = RaDiv(binh, ghf);
        const float binwg = RaDiv(binw, gwf);

        /* column window covering every sampled column of the ROI */
        int32_t cmin = Wm1i;
        int32_t cmax = 0;
        for (int64_t pw = 0; pw < ow; ++pw) {
            float bx = RaAdd(RaMul((float)pw, binw), rsw);
            for (int64_t ix = 0; ix < gw; ++ix) {
                float xv = RaAdd(bx, RaMul(RaAdd((float)ix, 0.5f), binwg));
                if (xv < 0.0f) {
                    xv = 0.0f;
                }
                int32_t xl = (int32_t)xv;
                int32_t xlc = xl;
                if (xlc > Wm1i) {
                    xlc = Wm1i;
                }
                int32_t xhc = xl + 1;
                if (xhc > Wm1i) {
                    xhc = Wm1i;
                }
                if (xlc < cmin) {
                    cmin = xlc;
                }
                if (xhc > cmax) {
                    cmax = xhc;
                }
            }
        }
        int32_t unitEl = (int32_t)(32 / esz);
        int32_t WX = ((cmax - cmin + 1 + unitEl - 1) / unitEl) * unitEl;
        if (WX > (int32_t)W) {
            WX = (int32_t)W;
        }
        int32_t xs = cmin;
        if (xs + WX > (int32_t)W) {
            xs = (int32_t)W - WX;
        }
        if (xs < 0) {
            xs = 0;
        }
        int32_t pitchRun = (int32_t)(RaAlignUpD((int64_t)WX * esz, 32) / esz);
        const int32_t rowStrideBytes = pitchRun * esz;
        const int32_t coffS = -(xs * esz);
        const int32_t maxColOff = (Wm1i - xs) * esz;
        Muls(coff, hiI, rowStrideBytes, LANES);
        Adds(coff, coff, coffS, LANES);
        Muls(offMax, hiI, rowStrideBytes, LANES);
        Adds(offMax, offMax, maxColOff, LANES);
        Duplicate(cntV, cnt, LANES);

        const int64_t xChanBase = ((b * C + c0) * H) * W + xs;
        const int64_t srcStrideBytes = (int64_t)(H * W - WX) * esz;
        const int64_t yBias = (int64_t)(oh * ow - ow) * esz;

        /* ---- x geometry for every ix, evaluated once and kept in UB ---- */
        const bool cached = (gw <= (int64_t)GWC);
        if (cached) {
            for (int64_t ix = 0; ix < gw; ++ix) {
                float xshift = RaMul(RaAdd((float)ix, 0.5f), binwg);
                RaGeom(pwF, xvS, ivS, hxArr[ix * LANES], lxArr[ix * LANES],
                       binw, rsw, xshift, Wm1i, LANES);
                RaOffsets(ivS, coff, offMax, offLArr[ix * LANES], offHArr[ix * LANES], esz, LANES);
            }
        }

        for (int64_t ph = 0; ph < oh; ++ph) {
            LocalTensor<float> accF = outQ.AllocTensor<float>();
            Duplicate(accF, 0.0f, LANES);

            for (int64_t iy = 0; iy < gh; ++iy) {
                float yv = RaAdd(RaAdd(rsh, RaMul((float)ph, binh)),
                                 RaMul(RaAdd((float)iy, 0.5f), binhg));
                if (yv < 0.0f) {
                    yv = 0.0f;
                }
                int32_t yl = (int32_t)yv;
                int32_t ylc = yl;
                if (ylc > Hm1i) {
                    ylc = Hm1i;
                }
                int32_t yhc = yl + 1;
                if (yhc > Hm1i) {
                    yhc = Hm1i;
                }
                const float ly = RaSub(yv, (float)ylc);
                const float hy = RaSub(1.0f, ly);

                LocalTensor<T> t0 = inQ.AllocTensor<T>();
                LocalTensor<T> t1 = inQ.AllocTensor<T>();
                AscendC::DataCopyExtParams cp0{(uint16_t)Cb, (uint32_t)(WX * esz),
                                               (uint32_t)srcStrideBytes, 0, 0};
                AscendC::DataCopyPadExtParams<T> pp{false, 0, 0, 0};
                DataCopyPad(t0, xG[xChanBase + (int64_t)ylc * W], cp0, pp);
                DataCopyPad(t1, xG[xChanBase + (int64_t)yhc * W], cp0, pp);
                inQ.EnQue(t0);
                inQ.EnQue(t1);
                t0 = inQ.DeQue<T>();
                t1 = inQ.DeQue<T>();

                for (int64_t ix = 0; ix < gw; ++ix) {
                    LocalTensor<float> hxV;
                    LocalTensor<float> lxV;
                    LocalTensor<int32_t> oLo;
                    LocalTensor<int32_t> oHi;
                    if (cached) {
                        hxV = hxArr[ix * LANES];
                        lxV = lxArr[ix * LANES];
                        oLo = offLArr[ix * LANES];
                        oHi = offHArr[ix * LANES];
                    } else {
                        hxV = hxS;
                        lxV = lxS;
                        oLo = offXl;
                        oHi = offXh;
                        float xshift = RaMul(RaAdd((float)ix, 0.5f), binwg);
                        RaGeom(pwF, xvS, ivS, hxV, lxV, binw, rsw, xshift, Wm1i, LANES);
                        RaOffsets(ivS, coff, offMax, oLo, oHi, esz, LANES);
                    }
                    LocalTensor<uint32_t> offLoU = oLo.ReinterpretCast<uint32_t>();
                    LocalTensor<uint32_t> offHiU = oHi.ReinterpretCast<uint32_t>();

                    Muls(t1v, hxV, hy, LANES);
                    Muls(t2v, lxV, hy, LANES);
                    Muls(t3v, hxV, ly, LANES);
                    Muls(t4v, lxV, ly, LANES);

                    if constexpr (IS_HALF) {
                        Gather(v1h, t0, offLoU, (uint32_t)0, (uint32_t)LANES);
                        Gather(v2h, t0, offHiU, (uint32_t)0, (uint32_t)LANES);
                        Cast(v1f, v1h, RoundMode::CAST_NONE, LANES);
                        Cast(v2f, v2h, RoundMode::CAST_NONE, LANES);
                    } else {
                        Gather(v1f, t0, offLoU, (uint32_t)0, (uint32_t)LANES);
                        Gather(v2f, t0, offHiU, (uint32_t)0, (uint32_t)LANES);
                    }
                    Mul(v1f, v1f, t1v, LANES);
                    Mul(v2f, v2f, t2v, LANES);
                    Add(sv, v1f, v2f, LANES);

                    if constexpr (IS_HALF) {
                        Gather(v1h, t1, offLoU, (uint32_t)0, (uint32_t)LANES);
                        Gather(v2h, t1, offHiU, (uint32_t)0, (uint32_t)LANES);
                        Cast(v1f, v1h, RoundMode::CAST_NONE, LANES);
                        Cast(v2f, v2h, RoundMode::CAST_NONE, LANES);
                    } else {
                        Gather(v1f, t1, offLoU, (uint32_t)0, (uint32_t)LANES);
                        Gather(v2f, t1, offHiU, (uint32_t)0, (uint32_t)LANES);
                    }
                    Mul(v1f, v1f, t3v, LANES);
                    Add(sv, sv, v1f, LANES);
                    Mul(v2f, v2f, t4v, LANES);
                    Add(sv, sv, v2f, LANES);
                    Add(accF, accF, sv, LANES);
                }

                inQ.FreeTensor(t0);
                inQ.FreeTensor(t1);
            }

            Div(accF, accF, cntV, LANES);
            int64_t yOff = ((n * C + c0) * oh + ph) * ow;
            if (PERM) {
                LocalTensor<uint32_t> pOffU = pf.ReinterpretCast<uint32_t>();
                if constexpr (IS_HALF) {
                    LocalTensor<half> o2h = outH.AllocTensor<half>();
                    Cast(o2h, accF, RoundMode::CAST_RINT, LANES);
                    outH.EnQue(o2h);
                    o2h = outH.DeQue<half>();
                    LocalTensor<half> o3h = outP.AllocTensor<half>();
                    Gather(o3h, o2h, pOffU, (uint32_t)0, (uint32_t)NP);
                    outP.EnQue(o3h);
                    o3h = outP.DeQue<half>();
                    AscendC::DataCopyExtParams cp{(uint16_t)Cb, (uint32_t)(ow * 2),
                                                  (uint32_t)((OP * 2 - RaAlignUpD((int64_t)ow * 2, 32)) / 32),
                                                  (uint32_t)yBias, 0};
                    DataCopyPad(yG[yOff], o3h, cp);
                    outP.FreeTensor(o3h);
                    outH.FreeTensor(o2h);
                } else {
                    LocalTensor<float> o3f = outP.AllocTensor<float>();
                    Gather(o3f, accF, pOffU, (uint32_t)0, (uint32_t)NP);
                    outP.EnQue(o3f);
                    o3f = outP.DeQue<float>();
                    AscendC::DataCopyExtParams cp{(uint16_t)Cb, (uint32_t)(ow * 4),
                                                  (uint32_t)((OP * 4 - RaAlignUpD((int64_t)ow * 4, 32)) / 32),
                                                  (uint32_t)yBias, 0};
                    DataCopyPad(yG[yOff], o3f, cp);
                    outP.FreeTensor(o3f);
                }
                outQ.FreeTensor(accF);
            } else if constexpr (IS_HALF) {
                LocalTensor<half> o2h = outH.AllocTensor<half>();
                Cast(o2h, accF, RoundMode::CAST_RINT, LANES);
                outH.EnQue(o2h);
                o2h = outH.DeQue<half>();
                AscendC::DataCopyExtParams cp{(uint16_t)Cb, (uint32_t)(ow * 2),
                                              (uint32_t)((OWp * 2 - RaAlignUpD((int64_t)ow * 2, 32)) / 32),
                                              (uint32_t)yBias, 0};
                DataCopyPad(yG[yOff], o2h, cp);
                outH.FreeTensor(o2h);
                outQ.FreeTensor(accF);
            } else {
                outQ.EnQue(accF);
                LocalTensor<float> o = outQ.DeQue<float>();
                AscendC::DataCopyExtParams cp{(uint16_t)Cb, (uint32_t)(ow * 4),
                                              (uint32_t)((OWp * 4 - RaAlignUpD((int64_t)ow * 4, 32)) / 32),
                                              (uint32_t)yBias, 0};
                DataCopyPad(yG[yOff], o, cp);
                outQ.FreeTensor(o);
            }
        }
    }
}

extern "C" {

void launch_roi_align_float(GM_ADDR x, GM_ADDR boxes, GM_ADDR y,
                            int64_t B, int64_t C, int64_t H, int64_t W, int64_t N,
                            int64_t oh, int64_t ow, float ss, int64_t sr, int64_t aligned,
                            int64_t CbP, int64_t nCb, int64_t OWp, int64_t OP, int64_t perm,
                            int64_t pitchMax, int64_t gwCap,
                            int64_t numUnits, int64_t unitsPerCore,
                            int64_t numBlocks, void *stream)
{
    roi_align_kernel<float><<<numBlocks, nullptr, stream>>>(
        x, boxes, y, B, C, H, W, N, oh, ow, ss, sr, aligned,
        CbP, nCb, OWp, OP, perm, pitchMax, gwCap, numUnits, unitsPerCore);
}

void launch_roi_align_half(GM_ADDR x, GM_ADDR boxes, GM_ADDR y,
                           int64_t B, int64_t C, int64_t H, int64_t W, int64_t N,
                           int64_t oh, int64_t ow, float ss, int64_t sr, int64_t aligned,
                           int64_t CbP, int64_t nCb, int64_t OWp, int64_t OP, int64_t perm,
                           int64_t pitchMax, int64_t gwCap,
                           int64_t numUnits, int64_t unitsPerCore,
                           int64_t numBlocks, void *stream)
{
    roi_align_kernel<half><<<numBlocks, nullptr, stream>>>(
        x, boxes, y, B, C, H, W, N, oh, ow, ss, sr, aligned,
        CbP, nCb, OWp, OP, perm, pitchMax, gwCap, numUnits, unitsPerCore);
}
}
