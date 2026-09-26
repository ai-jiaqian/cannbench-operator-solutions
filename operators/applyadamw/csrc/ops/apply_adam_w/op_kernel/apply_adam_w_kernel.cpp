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
 * \file apply_adam_w_kernel.cpp
 * \brief ApplyAdamW device kernel + host tiling + launch (compiled with bisheng + -xasc), target DAV_2201.
 *
 * Elementwise fused AdamW. With invA = 1/(1-beta1^step), invB = 1/(1-beta2^step) and
 * lrSigned = (maximize ? +lr : -lr) the whole update is folded into four coefficients:
 *
 *   a1 = beta1 * invA * lrSigned          c1 = (1-beta1) * invA * lrSigned
 *   a2 = beta2 * invB                     c2 = (1-beta2) * invB
 *   scaleX = 1 + lrSigned * weight_decay
 *
 *   t      = grad * grad
 *   v_hat  = a2 * v + c2 * t
 *   m_hat  = a1 * m + c1 * grad                  (already carries lrSigned)
 *   ratio  = m_hat / (sqrt(v_hat) + epsilon)     (epsilon is OUTSIDE the sqrt)
 *   y      = var * scaleX + ratio
 *
 * Argument/semantic order is otherwise identical to task/reference.py:
 *   m_new = beta1*m + (1-beta1)*grad,  v_new = beta2*v + (1-beta2)*grad*grad,
 *   m_hat = m_new/(1-beta1^step),      v_hat = v_new/(1-beta2^step),
 *   update = m_hat/(sqrt(v_hat)+epsilon),  update += weight_decay*var when weight_decay != 0,
 *   result = var - lr*update (minimize) / var + lr*update (maximize).
 *
 * float16 / bfloat16 inputs are widened to float32 for the whole computation and rounded back to the
 * input dtype at the very end. float32 operands are read in place and computed in float32.
 * NOTE: Cast<float, float> is not an identity move on this target, therefore no float -> float Cast
 * is ever issued; a float32 result is produced directly by the vector ops.
 *
 * UB accounting (40 B/element for both dtype classes):
 *   float32 : 5 queues x 2 x 4B                      = 40 B/elem (the VECOUT tensor doubles as scratch)
 *   2-byte  : 5 queues x 2 x 2B + 5 x 4B float bufs  = 20 + 20 = 40 B/elem
 */

#include <tuple>
#include <algorithm>
#include "kernel_operator.h"
#include "basic_api/kernel_operator_vec_ternary_scalar_intf.h"
#include "platform/platform_ascendc.h"
#include "apply_adam_w_launch.h"

namespace {

constexpr static int32_t APPLY_ADAM_W_PIPELINE_DEPTH = 2;
constexpr static int64_t APPLY_ADAM_W_UB_RESERVE = 8192;
constexpr static int64_t APPLY_ADAM_W_MIN_ELEMS_PER_CORE = 2048;
constexpr static int64_t APPLY_ADAM_W_ALIGN_ELEMS = 32;

// Elementwise AdamW step, all tensors float32.
//   fX : var      fG : grad      fM : m (updated in place)      fV : v (updated in place)
//   fT : scratch until the final two ops, then the result
__aicore__ inline void adamw_compute(
    const AscendC::LocalTensor<float>& fX,
    const AscendC::LocalTensor<float>& fG,
    const AscendC::LocalTensor<float>& fM,
    const AscendC::LocalTensor<float>& fV,
    const AscendC::LocalTensor<float>& fT,
    int32_t cnt,
    float a1, float c1, float a2, float c2,
    float epsilon, float scaleX, bool hasWd)
{
    // m_hat = a1 * m + c1 * grad   (a1/c1 already carry lrSigned)
    AscendC::Muls(fM, fM, a1, cnt);
    AscendC::Axpy(fM, fG, c1, cnt);

    // v_hat = a2 * v + c2 * grad * grad
    AscendC::Mul(fT, fG, fG, cnt);
    AscendC::Muls(fV, fV, a2, cnt);
    AscendC::Axpy(fV, fT, c2, cnt);

    // fT = sqrt(v_hat) + epsilon, then fM = m_hat / fT
    AscendC::Sqrt(fT, fV, cnt);
    if (epsilon != 0.0f) {
        AscendC::Adds(fT, fT, epsilon, cnt);
    }
    AscendC::Div(fM, fM, fT, cnt);

    // fT = var * scaleX + m_hat  (a single Add when weight_decay == 0, i.e. scaleX == 1)
    if (hasWd) {
        AscendC::Muls(fT, fX, scaleX, cnt);
        AscendC::Add(fT, fT, fM, cnt);
    } else {
        AscendC::Add(fT, fX, fM, cnt);
    }
}

} // namespace

template <typename T>
__global__ __aicore__ void apply_adam_w_kernel(
    GM_ADDR varGmAddr, GM_ADDR gradGmAddr, GM_ADDR mGmAddr, GM_ADDR vGmAddr, GM_ADDR yGmAddr,
    int64_t totalLength, int64_t blockLength, uint32_t tileElems,
    float a1, float c1, float a2, float c2, float epsilon, float scaleX, int64_t hasWd)
{
    constexpr bool TWO_BYTE = (sizeof(T) < sizeof(float));

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, APPLY_ADAM_W_PIPELINE_DEPTH> qVar;
    AscendC::TQue<AscendC::QuePosition::VECIN, APPLY_ADAM_W_PIPELINE_DEPTH> qGrad;
    AscendC::TQue<AscendC::QuePosition::VECIN, APPLY_ADAM_W_PIPELINE_DEPTH> qM;
    AscendC::TQue<AscendC::QuePosition::VECIN, APPLY_ADAM_W_PIPELINE_DEPTH> qV;
    AscendC::TQue<AscendC::QuePosition::VECOUT, APPLY_ADAM_W_PIPELINE_DEPTH> qY;

    uint32_t tileBytesT = tileElems * static_cast<uint32_t>(sizeof(T));
    uint32_t tileBytesF = tileElems * static_cast<uint32_t>(sizeof(float));

    pipe.InitBuffer(qVar, APPLY_ADAM_W_PIPELINE_DEPTH, tileBytesT);
    pipe.InitBuffer(qGrad, APPLY_ADAM_W_PIPELINE_DEPTH, tileBytesT);
    pipe.InitBuffer(qM, APPLY_ADAM_W_PIPELINE_DEPTH, tileBytesT);
    pipe.InitBuffer(qV, APPLY_ADAM_W_PIPELINE_DEPTH, tileBytesT);
    pipe.InitBuffer(qY, APPLY_ADAM_W_PIPELINE_DEPTH, tileBytesT);

    // float32 scratch buffers, only needed when 2-byte operands have to be widened first.
    // In the float32 case the four VECIN tensors are used directly and the VECOUT tensor
    // doubles as scratch, so no extra UB buffer is required.
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufG;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufM;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufV;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufT;
    if constexpr (TWO_BYTE) {
        pipe.InitBuffer(bufX, tileBytesF);
        pipe.InitBuffer(bufG, tileBytesF);
        pipe.InitBuffer(bufM, tileBytesF);
        pipe.InitBuffer(bufV, tileBytesF);
        pipe.InitBuffer(bufT, tileBytesF);
    }

    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t base = blockIdx * blockLength;
    int64_t curBlockLength = totalLength - base;
    if (curBlockLength > blockLength) {
        curBlockLength = blockLength;
    }
    if (curBlockLength <= 0) {
        return;
    }

    AscendC::GlobalTensor<T> varGm;
    AscendC::GlobalTensor<T> gradGm;
    AscendC::GlobalTensor<T> mGm;
    AscendC::GlobalTensor<T> vGm;
    AscendC::GlobalTensor<T> yGm;
    varGm.SetGlobalBuffer((__gm__ T*)varGmAddr + base);
    gradGm.SetGlobalBuffer((__gm__ T*)gradGmAddr + base);
    mGm.SetGlobalBuffer((__gm__ T*)mGmAddr + base);
    vGm.SetGlobalBuffer((__gm__ T*)vGmAddr + base);
    yGm.SetGlobalBuffer((__gm__ T*)yGmAddr + base);

    const bool hasWdFlag = (hasWd != 0);
    const int64_t tileElemsI = static_cast<int64_t>(tileElems);
    const int64_t tileNum = curBlockLength / tileElemsI;
    const int64_t tailElems = curBlockLength - tileNum * tileElemsI;

    // Process tileNum full tiles plus (when tailElems > 0) one tail tile.
    for (int64_t i = 0; i <= tileNum; ++i) {
        int64_t cnt = (i < tileNum) ? tileElemsI : tailElems;
        if (cnt <= 0) {
            break;
        }
        int64_t offset = i * tileElemsI;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        auto xRaw = qVar.AllocTensor<T>();
        auto gRaw = qGrad.AllocTensor<T>();
        auto mRaw = qM.AllocTensor<T>();
        auto vRaw = qV.AllocTensor<T>();
        AscendC::DataCopyPad(xRaw, varGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(gRaw, gradGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(mRaw, mGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(vRaw, vGm[offset], copyParams, padParams);
        qVar.EnQue(xRaw);
        qGrad.EnQue(gRaw);
        qM.EnQue(mRaw);
        qV.EnQue(vRaw);
        xRaw = qVar.DeQue<T>();
        gRaw = qGrad.DeQue<T>();
        mRaw = qM.DeQue<T>();
        vRaw = qV.DeQue<T>();

        auto yRaw = qY.AllocTensor<T>();

        AscendC::LocalTensor<float> fX;
        AscendC::LocalTensor<float> fG;
        AscendC::LocalTensor<float> fM;
        AscendC::LocalTensor<float> fV;
        AscendC::LocalTensor<float> fT;
        if constexpr (TWO_BYTE) {
            fX = bufX.Get<float>();
            fG = bufG.Get<float>();
            fM = bufM.Get<float>();
            fV = bufV.Get<float>();
            fT = bufT.Get<float>();
            AscendC::Cast(fX, xRaw, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(cnt));
            AscendC::Cast(fG, gRaw, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(cnt));
            AscendC::Cast(fM, mRaw, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(cnt));
            AscendC::Cast(fV, vRaw, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(cnt));
        } else {
            fX = xRaw;
            fG = gRaw;
            fM = mRaw;
            fV = vRaw;
            fT = yRaw;
        }

        adamw_compute(fX, fG, fM, fV, fT, static_cast<int32_t>(cnt),
                      a1, c1, a2, c2, epsilon, scaleX, hasWdFlag);

        if constexpr (TWO_BYTE) {
            AscendC::Cast(yRaw, fT, AscendC::RoundMode::CAST_RINT, static_cast<int32_t>(cnt));
        }

        qY.EnQue(yRaw);
        // Inputs stay alive until the result has been enqueued.
        qVar.FreeTensor(xRaw);
        qGrad.FreeTensor(gRaw);
        qM.FreeTensor(mRaw);
        qV.FreeTensor(vRaw);

        yRaw = qY.DeQue<T>();
        AscendC::DataCopyPad(yGm[offset], yRaw, copyParams);
        qY.FreeTensor(yRaw);
    }
}

// Host tiling: (numBlocks, blockLength, tileElems). blockLength is rounded up to a 32-element
// boundary so that every per-core base address is naturally aligned; tileElems is bounded by
// the bytes-per-element budget of the UB layout described above.
std::tuple<int64_t, int64_t, int64_t> calc_apply_adam_w_tiling_params(int64_t totalLength, int64_t elemBytes)
{
    constexpr static int64_t QUEUE_COUNT = 5;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) {
        coreNum = 1;
    }

    int64_t numBlocks = std::min(coreNum,
                                 (totalLength + APPLY_ADAM_W_MIN_ELEMS_PER_CORE - 1) / APPLY_ADAM_W_MIN_ELEMS_PER_CORE);
    if (numBlocks < 1) {
        numBlocks = 1;
    }
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    blockLength = ((blockLength + APPLY_ADAM_W_ALIGN_ELEMS - 1) / APPLY_ADAM_W_ALIGN_ELEMS) * APPLY_ADAM_W_ALIGN_ELEMS;

    // queue bytes (2-byte: 2B queues) + widened float32 working buffers (2-byte only)
    int64_t bytesPerElem = APPLY_ADAM_W_PIPELINE_DEPTH * QUEUE_COUNT * elemBytes;
    if (elemBytes < 4) {
        bytesPerElem += QUEUE_COUNT * 4;
    }

    int64_t avail = static_cast<int64_t>(ubSize) - APPLY_ADAM_W_UB_RESERVE;
    int64_t tileElems = avail / bytesPerElem;
    tileElems = (tileElems / APPLY_ADAM_W_ALIGN_ELEMS) * APPLY_ADAM_W_ALIGN_ELEMS;
    if (tileElems < APPLY_ADAM_W_ALIGN_ELEMS) {
        tileElems = APPLY_ADAM_W_ALIGN_ELEMS;
    }
    return std::make_tuple(numBlocks, blockLength, tileElems);
}

// Launch wrappers - regular C functions callable from g++
extern "C" {

void launch_apply_adam_w_kernel_float(
    GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems,
    float a1, float c1, float a2, float c2, float epsilon, float scaleX, int64_t hasWd,
    void* stream)
{
    apply_adam_w_kernel<float><<<numBlocks, nullptr, stream>>>(
        var, grad, m, v, y, totalLength, blockLength, tileElems, a1, c1, a2, c2, epsilon, scaleX, hasWd);
}

void launch_apply_adam_w_kernel_half(
    GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems,
    float a1, float c1, float a2, float c2, float epsilon, float scaleX, int64_t hasWd,
    void* stream)
{
    apply_adam_w_kernel<half><<<numBlocks, nullptr, stream>>>(
        var, grad, m, v, y, totalLength, blockLength, tileElems, a1, c1, a2, c2, epsilon, scaleX, hasWd);
}

void launch_apply_adam_w_kernel_bfloat16(
    GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileElems,
    float a1, float c1, float a2, float c2, float epsilon, float scaleX, int64_t hasWd,
    void* stream)
{
    apply_adam_w_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(
        var, grad, m, v, y, totalLength, blockLength, tileElems, a1, c1, a2, c2, epsilon, scaleX, hasWd);
}

} // extern "C"
