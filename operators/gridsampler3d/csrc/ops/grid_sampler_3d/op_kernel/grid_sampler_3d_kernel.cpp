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
 * \file grid_sampler_3d_kernel.cpp
 * \brief GridSampler3D (5D grid_sample, NCDHW) device kernels + host tiling/launch.
 *
 *   y[n, c, od, oh, ow] = interp( x[n, c, :, :, :], grid[n, od, oh, ow, :] )
 *
 * grid[...,0] -> W axis, grid[...,1] -> H axis, grid[...,2] -> D axis.  Reference semantics is
 * torch.nn.functional.grid_sample for 5-D input (mode 'bilinear' == trilinear, or 'nearest').
 *
 * The channel axis of the NCDHW input is the outer axis, so an arbitrary-voxel read for all
 * channels is a large-stride gather.  Three device kernels therefore run on one stream:
 *
 *   K1 gs_trans_x : x  (N, C, DHW)   -> xT (N, DHW, Cp)     [DHW = D*H*W, Cp = align(C,16)]
 *   K2 gs_sample  : grid + xT        -> yT (N, OS,  Cp)     [OS = OD*OH*OW]
 *   K3 gs_trans_y : yT (N, OS,  Cp)  -> y  (N, C,   OS)
 *
 * After K1 one voxel owns Cp contiguous channel elements, so a corner read for all channels is
 * a single contiguous DataCopyPad.  K2 issues two of those per (z,y) row pair (x0 and x1 are
 * adjacent in xT) - four loads for the eight trilinear corners - and accumulates in fp32 (torch
 * accumulates in opmath_t = float for half inputs).  A corner that falls outside the source image
 * is *skipped* rather than multiplied by zero, which reproduces torch's within_bounds handling
 * exactly, including the 0*inf = NaN hazard.
 *
 * The 16x16 tile transpose in K1/K3 uses Gather with one offset vector, off[k] = esize *
 * (16*(k&15) + (k>>4)), built once per core with vector ops.  Every UB "row" holds 16 elements so
 * each DMA block is a 32 byte multiple.
 */

#include <algorithm>
#include <tuple>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

using namespace AscendC;

namespace {
constexpr int64_t GS_TR = 16;        // channels per transpose tile
constexpr int64_t GS_TS_MAX = 64;    // spatial positions per transpose tile (upper bound)
constexpr int64_t GS_TR_EL = 256;    // 16 x 16
constexpr int64_t GS_TS_EL_MAX = GS_TS_MAX * GS_TR;  // 1024
constexpr int32_t GS_KMAX = 8;       // output positions sampled per K2 group
constexpr int64_t GS_LDC_DEPTH = 2;  // double buffering: one concatenated corner panel per position
constexpr int64_t GS_GRIDB = 128;    // output positions transported per grid block
}  // namespace

// ---------------------------------------------------------------------------
// device side helpers
// ---------------------------------------------------------------------------
__aicore__ inline int64_t GsAlignUpD(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

__aicore__ inline int64_t GsMinD(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline int64_t GsMaxD(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

/*! \brief number of 16 channel blocks for C (last block overlaps when C % 16 != 0). */
__aicore__ inline int64_t GsChanBlocks(int64_t C)
{
    if (C <= GS_TR) {
        return 1;
    }
    return (C - 1) / GS_TR + 1;
}

__aicore__ inline int64_t GsChanStart(int64_t C, int64_t cb)
{
    if (C <= GS_TR) {
        return 0;
    }
    int64_t c0 = cb * GS_TR;
    int64_t cLast = C - GS_TR;
    return c0 < cLast ? c0 : cLast;
}

__aicore__ inline int64_t GsChanCount(int64_t C)
{
    return C <= GS_TR ? C : GS_TR;
}

/*! \brief floor for the bounded coordinates used here. */
__aicore__ inline float GsFloorF(float x)
{
    int32_t i = (int32_t)x;
    if ((float)i > x) {
        i -= 1;
    }
    return (float)i;
}

/*! \brief std::nearbyint semantics (round half to even). */
__aicore__ inline int32_t GsNearbyIntF(float x)
{
    float f = GsFloorF(x);
    float d = x - f;
    int32_t i = (int32_t)f;
    if (d > 0.5f) {
        i += 1;
    } else if (d == 0.5f) {
        if ((i & 1) != 0) {
            i += 1;
        }
    }
    return i;
}

/*! \brief grid_sampler_unnormalize (identical operation order to ATen). */
__aicore__ inline float GsUnnormF(float g, int64_t size, int64_t alignC)
{
    if (alignC != 0) {
        return ((g + 1.0f) * 0.5f) * (float)(size - 1);
    }
    return ((g + 1.0f) * (float)size - 1.0f) * 0.5f;
}

/*! \brief grid_sampler clip_coordinates: min(limit-1, max(in, 0)). */
__aicore__ inline float GsClipF(float v, int64_t size)
{
    float hi = (float)(size - 1);
    float t = (v < 0.0f) ? 0.0f : v;
    return (t < hi) ? t : hi;
}

/*! \brief grid_sampler reflect_coordinates. */
__aicore__ inline float GsReflectF(float in, int64_t twiceLow, int64_t twiceHigh)
{
    if (twiceLow == twiceHigh) {
        return 0.0f;
    }
    float mn = (float)twiceLow * 0.5f;
    float span = (float)(twiceHigh - twiceLow) * 0.5f;
    float v = in - mn;
    if (v < 0.0f) {
        v = -v;
    }
    float q = GsFloorF(v / span);
    float extra = v - span * q;
    int32_t flips = (int32_t)q;
    if ((flips & 1) == 0) {
        return extra + mn;
    }
    return span - extra + mn;
}

/*! \brief grid_sampler_compute_source_index. */
__aicore__ inline float GsSrcCoordF(float g, int64_t size, int64_t padMode, int64_t alignC)
{
    float c = GsUnnormF(g, size, alignC);
    if (padMode == 1) {
        c = GsClipF(c, size);
    } else if (padMode == 2) {
        if (alignC != 0) {
            c = GsReflectF(c, 0, 2 * (size - 1));
        } else {
            c = GsReflectF(c, -1, 2 * size - 1);
        }
        c = GsClipF(c, size);
    }
    return c;
}

/*! \brief builds the byte offset table of the TR x ts transpose gather. */
__aicore__ inline void GsBuildTransOffsets(const LocalTensor<int32_t> &offI,
                                           const LocalTensor<float> &fA,
                                           const LocalTensor<float> &fB, int64_t esize, int64_t A,
                                           int64_t B, int64_t count)
{
    // offI[k] = esize * ((k % A) * B + (k / A)) = esize * (B*k - (B*A-1)*(k/A)), k in [0,count)
    CreateVecIndex(fA, 0.0f, (uint32_t)count);
    Muls(fB, fA, 1.0f / (float)A, (int32_t)count);              // k / A
    Cast(offI, fB, RoundMode::CAST_FLOOR, (uint32_t)count);     // floor(k / A)
    Cast(fB, offI, RoundMode::CAST_NONE, (uint32_t)count);
    Muls(fB, fB, (float)(B * A - 1), (int32_t)count);
    Muls(fA, fA, (float)B, (int32_t)count);
    Sub(fA, fA, fB, (int32_t)count);
    Cast(offI, fA, RoundMode::CAST_ROUND, (uint32_t)count);
    Muls(offI, offI, (int32_t)esize, (int32_t)count);
}

/*!
 * \brief accumulates one (z,y) row pair: two contiguous corner vectors become one fp32 sum.
 *        The half overload widens to fp32 first (torch accumulates half inputs in float).
 */
template <typename U>
__aicore__ inline void GsDiffAccum(const LocalTensor<float> &castF, const LocalTensor<float> &accF,
                                   const LocalTensor<U> &lp, float w0, float w1, int32_t n, bool u0,
                                   bool u1)
{
    Cast(castF, lp, RoundMode::CAST_NONE, (uint32_t)(2 * n));
    if (u0) {
        Axpy(accF, castF, w0, n);
    }
    if (u1) {
        Axpy(accF, castF[n], w1, n);
    }
}

__aicore__ inline void GsDiffAccum(const LocalTensor<float> &castF, const LocalTensor<float> &accF,
                                   const LocalTensor<float> &lp, float w0, float w1, int32_t n, bool u0,
                                   bool u1)
{
    (void)castF;
    if (u0) {
        Axpy(accF, lp, w0, n);
    }
    if (u1) {
        Axpy(accF, lp[n], w1, n);
    }
}

/*! \brief seeds the fp32 accumulator with the first contributing corner (avoids a zero-fill). */
template <typename U>
__aicore__ inline void GsFirstAccum(const LocalTensor<float> &castF, const LocalTensor<float> &accF,
                                    const LocalTensor<U> &lp, float w, int32_t n)
{
    Cast(castF, lp, RoundMode::CAST_NONE, (uint32_t)n);
    Muls(accF, castF, w, n);
}

__aicore__ inline void GsFirstAccum(const LocalTensor<float> &castF, const LocalTensor<float> &accF,
                                    const LocalTensor<float> &lp, float w, int32_t n)
{
    (void)castF;
    Muls(accF, lp, w, n);
}

/*! \brief writes the fp32 accumulator into the output tile (single rounding for half). */
template <typename U>
__aicore__ inline void GsOutCast(const LocalTensor<U> &ot, const LocalTensor<float> &accF, int32_t n)
{
    Cast(ot, accF, RoundMode::CAST_RINT, (uint32_t)n);
}

__aicore__ inline void GsOutCast(const LocalTensor<float> &ot, const LocalTensor<float> &accF, int32_t n)
{
    Muls(ot, accF, 1.0f, n);
}

/*!
 * \brief K1: x (N, C, DHW) -> xT (N, DHW, Cp): one tile = 16 channels x ts spatial positions.
 */
template <typename T>
__global__ __aicore__ void gs_trans_x_kernel(GM_ADDR xGm_, GM_ADDR xtGm_, int64_t C, int64_t DHW,
                                             int64_t Cp, int64_t padFront, int64_t ts, int64_t numS,
                                             int64_t numCB, int64_t totalTiles, int64_t tilesPerCore)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (totalTiles <= 0) {
        return;
    }
    int64_t t0 = (int64_t)GetBlockIdx() * tilesPerCore;
    int64_t t1 = t0 + tilesPerCore;
    if (t1 > totalTiles) {
        t1 = totalTiles;
    }
    if (t0 >= t1) {
        return;
    }

    GlobalTensor<T> xG;
    GlobalTensor<T> xtG;
    xG.SetGlobalBuffer((__gm__ T *)xGm_);
    xtG.SetGlobalBuffer((__gm__ T *)xtGm_);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
    pipe.InitBuffer(inQ, 2, GS_TS_EL_MAX * sizeof(T));
    pipe.InitBuffer(outQ, 2, GS_TS_EL_MAX * sizeof(T));
    TBuf<TPosition::VECCALC> offBuf;
    TBuf<TPosition::VECCALC> fBufA;
    TBuf<TPosition::VECCALC> fBufB;
    pipe.InitBuffer(offBuf, GS_TS_EL_MAX * (int64_t)sizeof(int32_t));
    pipe.InitBuffer(fBufA, GS_TS_EL_MAX * (int64_t)sizeof(float));
    pipe.InitBuffer(fBufB, GS_TS_EL_MAX * (int64_t)sizeof(float));
    LocalTensor<int32_t> offI = offBuf.Get<int32_t>();
    LocalTensor<float> fA = fBufA.Get<float>();
    LocalTensor<float> fB = fBufB.Get<float>();
    const int64_t cnt = ts * GS_TR;
    GsBuildTransOffsets(offI, fA, fB, (int64_t)sizeof(T), GS_TR, ts, cnt);
    LocalTensor<uint32_t> offU = offI.ReinterpretCast<uint32_t>();

    DataCopyPadExtParams<T> padp{false, 0, 0, 0};
    const int64_t Cm = GsChanCount(C);
    const int64_t sLast = DHW - ts;

    for (int64_t t = t0; t < t1; ++t) {
        int64_t cb = t % numCB;
        int64_t s = (t / numCB) % numS;
        int64_t n = t / (numCB * numS);
        int64_t c0 = GsChanStart(C, cb);
        int64_t s0 = s * ts;
        if (s0 > sLast) {
            s0 = sLast;
        }
        int64_t srcOff = ((n * C + c0) * DHW) + s0;
        int64_t dstOff = padFront + ((n * DHW + s0) * Cp + c0);

        auto sT = inQ.AllocTensor<T>();
        DataCopyExtParams cp{(uint16_t)Cm, (uint32_t)(ts * sizeof(T)),
                             (uint32_t)((DHW - ts) * sizeof(T)), 0, 0};
        DataCopyPad(sT, xG[srcOff], cp, padp);
        inQ.EnQue(sT);
        sT = inQ.DeQue<T>();

        auto dT = outQ.AllocTensor<T>();
        Gather(dT, sT, offU, (uint32_t)0, (uint32_t)cnt);
        inQ.FreeTensor(sT);
        outQ.EnQue(dT);
        dT = outQ.DeQue<T>();

        DataCopyExtParams stp{(uint16_t)ts, (uint32_t)(GS_TR * sizeof(T)), 0,
                              (uint32_t)((Cp - GS_TR) * sizeof(T)), 0};
        DataCopyPad(xtG[dstOff], dT, stp);
        outQ.FreeTensor(dT);
    }
}

/*!
 * \brief K3: yT (N, OS, Cp) -> y (N, C, OS): one tile = ts positions x 16 channels.
 */
template <typename T>
__global__ __aicore__ void gs_trans_y_kernel(GM_ADDR ytGm_, GM_ADDR yGm_, int64_t C, int64_t OS,
                                             int64_t Cp, int64_t ts, int64_t numP, int64_t numCB,
                                             int64_t totalTiles, int64_t tilesPerCore)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (totalTiles <= 0) {
        return;
    }
    int64_t t0 = (int64_t)GetBlockIdx() * tilesPerCore;
    int64_t t1 = t0 + tilesPerCore;
    if (t1 > totalTiles) {
        t1 = totalTiles;
    }
    if (t0 >= t1) {
        return;
    }

    GlobalTensor<T> ytG;
    GlobalTensor<T> yG;
    ytG.SetGlobalBuffer((__gm__ T *)ytGm_);
    yG.SetGlobalBuffer((__gm__ T *)yGm_);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
    pipe.InitBuffer(inQ, 2, GS_TS_EL_MAX * sizeof(T));
    pipe.InitBuffer(outQ, 2, GS_TS_EL_MAX * sizeof(T));
    TBuf<TPosition::VECCALC> offBuf;
    TBuf<TPosition::VECCALC> fBufA;
    TBuf<TPosition::VECCALC> fBufB;
    pipe.InitBuffer(offBuf, GS_TS_EL_MAX * (int64_t)sizeof(int32_t));
    pipe.InitBuffer(fBufA, GS_TS_EL_MAX * (int64_t)sizeof(float));
    pipe.InitBuffer(fBufB, GS_TS_EL_MAX * (int64_t)sizeof(float));
    LocalTensor<int32_t> offI = offBuf.Get<int32_t>();
    LocalTensor<float> fA = fBufA.Get<float>();
    LocalTensor<float> fB = fBufB.Get<float>();
    const int64_t cnt = ts * GS_TR;
    GsBuildTransOffsets(offI, fA, fB, (int64_t)sizeof(T), ts, GS_TR, cnt);
    LocalTensor<uint32_t> offU = offI.ReinterpretCast<uint32_t>();

    DataCopyPadExtParams<T> padp{false, 0, 0, 0};
    const int64_t Cm = GsChanCount(C);
    const int64_t pLast = OS - ts;

    for (int64_t t = t0; t < t1; ++t) {
        int64_t cb = t % numCB;
        int64_t sp = (t / numCB) % numP;
        int64_t n = t / (numCB * numP);
        int64_t c0 = GsChanStart(C, cb);
        int64_t p0 = sp * ts;
        if (p0 > pLast) {
            p0 = pLast;
        }
        int64_t srcOff = (n * OS + p0) * Cp + c0;
        int64_t dstOff = (n * C + c0) * OS + p0;

        auto sT = inQ.AllocTensor<T>();
        DataCopyExtParams cp{(uint16_t)ts, (uint32_t)(GS_TR * sizeof(T)),
                             (uint32_t)((Cp - GS_TR) * sizeof(T)), 0, 0};
        DataCopyPad(sT, ytG[srcOff], cp, padp);
        inQ.EnQue(sT);
        sT = inQ.DeQue<T>();

        auto dT = outQ.AllocTensor<T>();
        Gather(dT, sT, offU, (uint32_t)0, (uint32_t)cnt);
        inQ.FreeTensor(sT);
        outQ.EnQue(dT);
        dT = outQ.DeQue<T>();

        DataCopyExtParams stp{(uint16_t)Cm, (uint32_t)(ts * sizeof(T)), 0,
                              (uint32_t)((OS - ts) * sizeof(T)), 0};
        DataCopyPad(yG[dstOff], dT, stp);
        outQ.FreeTensor(dT);
    }
}

/*! \brief per output position sampling plan carried from the issue to the consume stage. */
struct GsPlan {
    float w[8];
    uint32_t flags;
    int64_t rowOff[4];
    int64_t dstOff;
};

/*!
 * \brief builds the sampling plan of one output position from its grid triplet.
 */
template <typename T>
__aicore__ inline GsPlan GsBuildPlan(const LocalTensor<T> &gT, int64_t gi, int64_t n, int64_t D,
                                     int64_t H, int64_t W, int64_t Cp, int64_t padFront,
                                     int64_t padMode, int64_t alignC, int64_t isNearest)
{
    GsPlan pl;
    pl.dstOff = 0;
    const float gx = (float)gT.GetValue((uint32_t)gi);
    const float gy = (float)gT.GetValue((uint32_t)(gi + 1));
    const float gz = (float)gT.GetValue((uint32_t)(gi + 2));
    const float ix = GsSrcCoordF(gx, W, padMode, alignC);
    const float iy = GsSrcCoordF(gy, H, padMode, alignC);
    const float iz = GsSrcCoordF(gz, D, padMode, alignC);
    if (isNearest != 0) {
        const int32_t xr = (int32_t)GsNearbyIntF(ix);
        const int32_t yr = (int32_t)GsNearbyIntF(iy);
        const int32_t zr = (int32_t)GsNearbyIntF(iz);
        const bool ok = (xr >= 0 && xr < (int32_t)W && yr >= 0 && yr < (int32_t)H && zr >= 0 &&
                         zr < (int32_t)D);
        const int32_t xc = (int32_t)GsMinD(GsMaxD(xr, 0), W - 1);
        const int32_t yc = (int32_t)GsMinD(GsMaxD(yr, 0), H - 1);
        const int32_t zc = (int32_t)GsMinD(GsMaxD(zr, 0), D - 1);
        pl.flags = ok ? 1u : 0u;
        pl.w[0] = 0.0f;
        pl.rowOff[0] = padFront + (((n * D + zc) * H + yc) * W + xc) * Cp;
        return pl;
    }
    const int32_t x0 = (int32_t)GsFloorF(ix);
    const int32_t y0 = (int32_t)GsFloorF(iy);
    const int32_t z0 = (int32_t)GsFloorF(iz);
    const int32_t x1 = x0 + 1;
    const int32_t y1 = y0 + 1;
    const int32_t z1 = z0 + 1;
    const float wx1 = ix - (float)x0;
    const float wx0 = (float)x1 - ix;
    const float wy1 = iy - (float)y0;
    const float wy0 = (float)y1 - iy;
    const float wz1 = iz - (float)z0;
    const float wz0 = (float)z1 - iz;
    int32_t xl = x0;
    if (xl > (int32_t)(W - 1)) {
        xl = (int32_t)(W - 1);
    }
    if (xl < -1) {
        xl = -1;
    }
    const int32_t xv0 = (x0 >= 0 && x0 < (int32_t)W) ? 1 : 0;
    const int32_t xv1 = (x1 >= 0 && x1 < (int32_t)W) ? 1 : 0;
    pl.flags = 0u;
    for (int32_t j = 0; j < 4; ++j) {
        const int32_t z = ((j >> 1) & 1) ? z1 : z0;
        const int32_t y = (j & 1) ? y1 : y0;
        const bool rowOk = (z >= 0 && z < (int32_t)D && y >= 0 && y < (int32_t)H);
        if (rowOk) {
            pl.rowOff[j] = padFront + (((n * D + z) * H + y) * W + xl) * Cp;
        } else {
            pl.rowOff[j] = -1;
        }
        const float wz = ((j >> 1) & 1) ? wz1 : wz0;
        const float wy = (j & 1) ? wy1 : wy0;
        pl.w[2 * j] = wx0 * wy * wz;
        pl.w[2 * j + 1] = wx1 * wy * wz;
        if (rowOk && xv0 != 0) {
            pl.flags |= (1u << (2 * j));
        }
        if (rowOk && xv1 != 0) {
            pl.flags |= (1u << (2 * j + 1));
        }
    }
    return pl;
}

/*! \brief fp32 accumulates straight into the out-queue tile; fp16 accumulates in a float scratch. */
__aicore__ inline LocalTensor<float> GsAccumDstF(const LocalTensor<float> &accF,
                                                 const LocalTensor<float> &ot, int32_t off)
{
    return ot[off];
}

__aicore__ inline LocalTensor<float> GsAccumDstF(const LocalTensor<float> &accF,
                                                 const LocalTensor<half> &ot, int32_t off)
{
    (void)ot;
    (void)off;
    return accF;
}

__aicore__ inline void GsAccumFlush(const LocalTensor<float> &ot, int32_t off,
                                    const LocalTensor<float> &accF, int32_t n)
{
    (void)ot;
    (void)off;
    (void)accF;
    (void)n;
}

__aicore__ inline void GsAccumFlush(const LocalTensor<half> &ot, int32_t off,
                                    const LocalTensor<float> &accF, int32_t n)
{
    GsOutCast(ot[off], accF, n);
}

/*! \brief accumulates one sampled output position into its slice of the out-queue tile. */
template <typename T>
__aicore__ inline void GsAccumPos(const GsPlan &pl, int64_t isNearest, int32_t Cp,
                                  const LocalTensor<T> &lp, int32_t lpOff, const LocalTensor<T> &ot,
                                  int32_t otOff, const LocalTensor<float> &accF,
                                  const LocalTensor<float> &castF)
{
    if (isNearest != 0) {
        if ((pl.flags & 1u) == 0u) {
            Duplicate(ot[otOff], (T)0, Cp);
        } else {
            Adds(ot[otOff], lp[lpOff], (T)0, Cp);
        }
        return;
    }
    LocalTensor<float> dstF = GsAccumDstF(accF, ot, otOff);
    uint32_t fl = pl.flags & 0xFFu;
    if (fl == 0u) {
        Duplicate(dstF, 0.0f, Cp);
    } else {
        // seed the accumulator with the first contributing corner so no zero-fill is needed
        int32_t firstJ = 0;
        int32_t firstX1 = 0;
        for (int32_t k = 0; k < 8; ++k) {
            if (((fl >> k) & 1u) != 0u) {
                firstJ = k >> 1;
                firstX1 = k & 1;
                break;
            }
        }
        const int32_t seedOff = lpOff + 2 * firstJ * Cp + (firstX1 != 0 ? Cp : 0);
        GsFirstAccum(castF, dstF, lp[seedOff], pl.w[2 * firstJ + firstX1], Cp);
        fl &= ~(1u << (2 * firstJ + firstX1));
        for (int32_t j = 0; j < 4; ++j) {
            uint32_t f = (fl >> (2 * j)) & 3u;
            if (f != 0u) {
                GsDiffAccum(castF, dstF, lp[lpOff + 2 * j * Cp], pl.w[2 * j], pl.w[2 * j + 1], Cp,
                            (f & 1u) != 0u, (f & 2u) != 0u);
            }
        }
    }
    GsAccumFlush(ot, otOff, accF, Cp);
}

/*!
 * \brief K2 (grouped): grid + xT -> yT, K consecutive output positions per queue slot.
 *
 * All 8*K corner reads of a group are issued together into one queue tensor, so the (random)
 * DRAM latency of the corner reads is paid once per group instead of once per position, and the
 * queue/DMA bookkeeping is amortised over K positions.
 */
template <typename T>
__global__ __aicore__ void gs_sample_group_kernel(GM_ADDR xT_, GM_ADDR grid_, GM_ADDR yT_, int64_t N,
                                                  int64_t C, int64_t D, int64_t H, int64_t W,
                                                  int64_t OD, int64_t OH, int64_t OW, int64_t Cp,
                                                  int64_t padFront, int64_t padMode, int64_t alignC,
                                                  int64_t isNearest, int64_t posPerCore,
                                                  int64_t totalPos, int64_t kg)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (totalPos <= 0) {
        return;
    }
    int64_t start = (int64_t)GetBlockIdx() * posPerCore;
    int64_t end = start + posPerCore;
    if (end > totalPos) {
        end = totalPos;
    }
    if (start >= end) {
        return;
    }
    const int64_t OS = OD * OH * OW;
    const int64_t totalG = N * OS * 3;
    int32_t K = (int32_t)kg;
    if (K < 1) {
        K = 1;
    }
    if (K > GS_KMAX) {
        K = GS_KMAX;
    }
    const int32_t per = (isNearest == 0) ? (int32_t)(8 * Cp) : (int32_t)Cp;

    GlobalTensor<T> xTG;
    GlobalTensor<T> gG;
    GlobalTensor<T> yTG;
    xTG.SetGlobalBuffer((__gm__ T *)xT_);
    gG.SetGlobalBuffer((__gm__ T *)grid_);
    yTG.SetGlobalBuffer((__gm__ T *)yT_);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECIN, 2> gQ;
    TQue<QuePosition::VECOUT, 2> outQ;
    pipe.InitBuffer(inQ, 2, K * per * sizeof(T));
    pipe.InitBuffer(gQ, 2, 3 * GS_GRIDB * sizeof(T));
    pipe.InitBuffer(outQ, 2, K * (int32_t)Cp * sizeof(T));
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> castBuf;
    pipe.InitBuffer(accBuf, Cp * (int64_t)sizeof(float));
    pipe.InitBuffer(castBuf, 2 * Cp * (int64_t)sizeof(float));
    LocalTensor<float> accF = accBuf.Get<float>();
    LocalTensor<float> castF = castBuf.Get<float>();

    DataCopyPadExtParams<T> padp{false, 0, 0, 0};

    int64_t p = start;
    int64_t n = p / OS;
    int64_t r = p - n * OS;
    int64_t od = r / (OH * OW);
    int64_t r2 = r - od * OH * OW;
    int64_t oh = r2 / OW;
    int64_t ow = r2 - oh * OW;

    int64_t curGb = -1;
    int64_t gbBase = 0;
    LocalTensor<T> gT;
    GsPlan planArr[GS_KMAX];

    while (p < end) {
        // ---- grid block covering the whole group ----
        const int64_t gb = p / GS_GRIDB;
        if (gb != curGb) {
            if (curGb >= 0) {
                gQ.FreeTensor(gT);
            }
            const int64_t base = gb * GS_GRIDB;
            int64_t cnt = totalG - base * 3;
            if (cnt > 3 * GS_GRIDB) {
                cnt = 3 * GS_GRIDB;
            }
            gT = gQ.AllocTensor<T>();
            DataCopyExtParams gp{1, (uint32_t)(cnt * sizeof(T)), 0, 0, 0};
            DataCopyPad(gT, gG[base * 3], gp, padp);
            gQ.EnQue(gT);
            gT = gQ.DeQue<T>();
            curGb = gb;
            gbBase = base;
        }
        int64_t gbEnd = (gb + 1) * GS_GRIDB;
        if (gbEnd > totalPos) {
            gbEnd = totalPos;
        }
        int32_t cnt = (int32_t)K;
        if (gbEnd - p < (int64_t)cnt) {
            cnt = (int32_t)(gbEnd - p);
        }
        if (end - p < (int64_t)cnt) {
            cnt = (int32_t)(end - p);
        }

        // ---- plans for the whole group ----
        {
            int64_t cn = n;
            int64_t cod = od;
            int64_t coh = oh;
            int64_t cow = ow;
            for (int32_t i = 0; i < cnt; ++i) {
                planArr[i] = GsBuildPlan(gT, (p + i - gbBase) * 3, cn, D, H, W, Cp, padFront, padMode,
                                         alignC, isNearest);
                if (++cow == OW) {
                    cow = 0;
                    if (++coh == OH) {
                        coh = 0;
                        if (++cod == OD) {
                            cod = 0;
                            ++cn;
                        }
                    }
                }
            }
            n = cn;
            od = cod;
            oh = coh;
            ow = cow;
        }

        // ---- issue every corner read of the group ----
        auto lp = inQ.AllocTensor<T>();
        for (int32_t i = 0; i < cnt; ++i) {
            const GsPlan &pl = planArr[i];
            if (isNearest == 0) {
                for (int32_t j = 0; j < 4; ++j) {
                    if (pl.rowOff[j] >= 0) {
                        DataCopyExtParams cp{1, (uint32_t)(2 * Cp * sizeof(T)), 0, 0, 0};
                        DataCopyPad(lp[i * per + 2 * j * (int32_t)Cp], xTG[pl.rowOff[j]], cp, padp);
                    }
                }
            } else {
                DataCopyExtParams cp{1, (uint32_t)(Cp * sizeof(T)), 0, 0, 0};
                DataCopyPad(lp[i * per], xTG[pl.rowOff[0]], cp, padp);
            }
        }
        inQ.EnQue(lp);
        lp = inQ.DeQue<T>();

        // ---- accumulate the group and store its contiguous yT rows with one DMA ----
        auto ot = outQ.AllocTensor<T>();
        for (int32_t i = 0; i < cnt; ++i) {
            GsAccumPos<T>(planArr[i], isNearest, (int32_t)Cp, lp, i * per, ot, i * (int32_t)Cp, accF,
                          castF);
        }
        inQ.FreeTensor(lp);
        outQ.EnQue(ot);
        ot = outQ.DeQue<T>();
        DataCopyExtParams sp{1, (uint32_t)(cnt * Cp * sizeof(T)), 0, 0, 0};
        DataCopyPad(yTG[p * Cp], ot, sp);
        outQ.FreeTensor(ot);

        p += cnt;
    }
}

// ---------------------------------------------------------------------------
// host side: tiling + launch
// ---------------------------------------------------------------------------
static int64_t GsAlignUpH(int64_t v, int64_t a)
{
    return ((v + a - 1) / a) * a;
}

static int64_t GsMinH(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

static int64_t GsChanBlocksH(int64_t C)
{
    if (C <= GS_TR) {
        return 1;
    }
    return (C - 1) / GS_TR + 1;
}

/*! \brief spatial/position width of one transpose tile: 64 when possible, else 16. */
static int64_t GsTransTsH(int64_t L)
{
    return (L >= GS_TS_MAX) ? GS_TS_MAX : GS_TR;
}

// diagnostic switch (0 = run everything)
static constexpr int64_t GS_DBG_SKIP = 0;

/*!
 * \brief tiling: (Cp, padFront, nb1, tpc1, nb2, ppc2, nb3, tpc3)
 */
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
calc_grid_sampler_3d_tiling(int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, int64_t OD,
                            int64_t OH, int64_t OW)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }
    const int64_t Cp = (C <= GS_TR) ? GS_TR : GsAlignUpH(C, GS_TR);
    const int64_t DHW = D * H * W;
    const int64_t OS = OD * OH * OW;

    const int64_t tiles1 = N * (GsAlignUpH(DHW, GsTransTsH(DHW)) / GsTransTsH(DHW)) * GsChanBlocksH(C);
    int64_t nb1 = GsMinH(coreNum, tiles1);
    if (nb1 < 1) {
        nb1 = 1;
    }
    const int64_t tpc1 = (tiles1 + nb1 - 1) / nb1;

    const int64_t totalPos = N * OS;
    int64_t nb2 = GsMinH(coreNum, totalPos);
    if (nb2 < 1) {
        nb2 = 1;
    }
    const int64_t ppc2 = (totalPos + nb2 - 1) / nb2;

    const int64_t tiles3 = N * (GsAlignUpH(OS, GsTransTsH(OS)) / GsTransTsH(OS)) * GsChanBlocksH(C);
    int64_t nb3 = GsMinH(coreNum, tiles3);
    if (nb3 < 1) {
        nb3 = 1;
    }
    const int64_t tpc3 = (tiles3 + nb3 - 1) / nb3;

    const int64_t padFront = 4 * Cp;
    return std::make_tuple(Cp, padFront, nb1, tpc1, nb2, ppc2, nb3, tpc3);
}

extern "C" {

void launch_grid_sampler_3d_float(GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR xT, GM_ADDR yT,
                                  int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, int64_t OD,
                                  int64_t OH, int64_t OW, int64_t Cp, int64_t padFront, int64_t mode,
                                  int64_t padMode, int64_t alignC, int64_t nb1, int64_t tpc1, int64_t nb2,
                                  int64_t ppc2, int64_t nb3, int64_t tpc3, int64_t xTpitch, void *stream)
{
    const int64_t DHW = D * H * W;
    const int64_t OS = OD * OH * OW;
    const int64_t ts1 = GsTransTsH(DHW);
    const int64_t ts3 = GsTransTsH(OS);
    const int64_t numS = GsAlignUpH(DHW, ts1) / ts1;
    const int64_t numP = GsAlignUpH(OS, ts3) / ts3;
    const int64_t numCB = GsChanBlocksH(C);
    const int64_t tiles1 = N * numS * numCB;
    const int64_t tiles3 = N * numP * numCB;
    (void)xTpitch;
    int64_t kg = 32768 / (8 * Cp * (int64_t)sizeof(float));
    if (kg > GS_KMAX) {
        kg = GS_KMAX;
    }
    if (kg < 1) {
        kg = 1;
    }
    gs_trans_x_kernel<float><<<(uint32_t)nb1, nullptr, stream>>>(x, xT, C, DHW, Cp, padFront, ts1, numS,
                                                                 numCB, tiles1, tpc1);
    if (GS_DBG_SKIP != 1) {
        gs_sample_group_kernel<float><<<(uint32_t)nb2, nullptr, stream>>>(
            xT, grid, yT, N, C, D, H, W, OD, OH, OW, Cp, padFront, padMode, alignC, mode, ppc2,
            N * OS, kg);
    }
    if (GS_DBG_SKIP != 3) {
        gs_trans_y_kernel<float><<<(uint32_t)nb3, nullptr, stream>>>(yT, y, C, OS, Cp, ts3, numP, numCB,
                                                                     tiles3, tpc3);
    }
}

void launch_grid_sampler_3d_half(GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR xT, GM_ADDR yT,
                                 int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, int64_t OD,
                                 int64_t OH, int64_t OW, int64_t Cp, int64_t padFront, int64_t mode,
                                 int64_t padMode, int64_t alignC, int64_t nb1, int64_t tpc1, int64_t nb2,
                                 int64_t ppc2, int64_t nb3, int64_t tpc3, int64_t xTpitch, void *stream)
{
    const int64_t DHW = D * H * W;
    const int64_t OS = OD * OH * OW;
    const int64_t ts1 = GsTransTsH(DHW);
    const int64_t ts3 = GsTransTsH(OS);
    const int64_t numS = GsAlignUpH(DHW, ts1) / ts1;
    const int64_t numP = GsAlignUpH(OS, ts3) / ts3;
    const int64_t numCB = GsChanBlocksH(C);
    const int64_t tiles1 = N * numS * numCB;
    const int64_t tiles3 = N * numP * numCB;
    (void)xTpitch;
    int64_t kg = 32768 / (8 * Cp * (int64_t)sizeof(half));
    if (kg > GS_KMAX) {
        kg = GS_KMAX;
    }
    if (kg < 1) {
        kg = 1;
    }
    gs_trans_x_kernel<half><<<(uint32_t)nb1, nullptr, stream>>>(x, xT, C, DHW, Cp, padFront, ts1, numS,
                                                                numCB, tiles1, tpc1);
    if (GS_DBG_SKIP != 1) {
        gs_sample_group_kernel<half><<<(uint32_t)nb2, nullptr, stream>>>(
            xT, grid, yT, N, C, D, H, W, OD, OH, OW, Cp, padFront, padMode, alignC, mode, ppc2,
            N * OS, kg);
    }
    if (GS_DBG_SKIP != 3) {
        gs_trans_y_kernel<half><<<(uint32_t)nb3, nullptr, stream>>>(yT, y, C, OS, Cp, ts3, numP, numCB,
                                                                    tiles3, tpc3);
    }
}
}
