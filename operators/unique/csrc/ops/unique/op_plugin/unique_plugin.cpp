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
 * \file unique_plugin.cpp
 * \brief Unique API layer - torch bindings + host dispatch (compiled with g++).
 *
 * Every element of y and of inverse is produced by the Ascend C kernels launched here.  The
 * only scalars that cross back to the host are shape metadata: torch.unique's output length is
 * data dependent and cannot be derived from the input shape, so the output tensor has to be
 * sized on the host side (the same situation the framework's OutputShapeDependOnCompute
 * mechanism handles for NonZero-like operators).  No element of either result is computed,
 * inspected or transformed on the host.
 */

#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/unique_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("unique(Tensor x, bool return_inverse) -> Tensor[]");
}

namespace {

int64_t DtypeCode(torch::ScalarType st)
{
    switch (st) {
        case torch::kUInt8:
            return UK_DT_U8;
        case torch::kInt8:
            return UK_DT_I8;
        case torch::kFloat16:
            return UK_DT_F16;
        case torch::kBFloat16:
            return UK_DT_BF16;
        case torch::kInt32:
            return UK_DT_I32;
        case torch::kInt64:
            return UK_DT_I64;
        case torch::kFloat32:
            return UK_DT_F32;
        default:
            return -1;
    }
}

int64_t P2Ceil(int64_t total, int64_t blocks)
{
    if (blocks < 1) {
        blocks = 1;
    }
    return (total + blocks - 1) / blocks;
}

int64_t NPassFor(int64_t limit)
{
    int64_t bits = 0;
    int64_t r = (limit > 1) ? (limit - 1) : 0;
    while (r > 0) {
        ++bits;
        r >>= 1;
    }
    int64_t np = (bits + 7) / 8;
    if (np < 1) {
        np = 1;
    }
    return np;
}

// shape metadata readback: data dependent output length has to be sized on the host
int64_t ReadDevI64(const torch::Tensor &t, int64_t idx)
{
    auto host = t.to(torch::Device(torch::kCPU));
    return host.data_ptr<int64_t>()[idx];
}

} // namespace

std::vector<torch::Tensor> unique_meta(const torch::Tensor &x, bool return_inverse)
{
    TORCH_CHECK(DtypeCode(x.scalar_type()) >= 0,
                "cann_bench.unique supports uint8/int8/float16/bfloat16/int32/int64/float32 only.");
    std::vector<torch::Tensor> out;
    out.push_back(torch::empty({x.numel()}, x.options()));
    if (return_inverse) {
        out.push_back(torch::empty(x.sizes(), x.options().dtype(torch::kLong)));
    }
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("unique", unique_meta);
}

std::vector<torch::Tensor> unique_npu(const torch::Tensor &x, bool return_inverse)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto xc = x.contiguous();
    int64_t n = xc.numel();
    TORCH_CHECK(n > 0, "cann_bench.unique requires a non-empty input tensor.");
    int64_t dt = DtypeCode(xc.scalar_type());
    TORCH_CHECK(dt >= 0, "cann_bench.unique unsupported dtype.");

    auto optsLong = xc.options().dtype(torch::kLong);
    auto optsI32 = xc.options().dtype(torch::kInt);

    int64_t numBlocks = calc_unique_blocks(n);
    int64_t blockLength = P2Ceil(n, numBlocks);

    auto params = torch::empty({8}, optsLong);
    auto yBig = torch::empty({n}, xc.options());
    torch::Tensor inverse;
    if (return_inverse) {
        inverse = torch::empty(xc.sizes(), optsLong);
    }

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    GM_ADDR xp = reinterpret_cast<GM_ADDR>(xc.data_ptr());
    GM_ADDR yp = reinterpret_cast<GM_ADDR>(yBig.data_ptr());
    GM_ADDR pp = reinterpret_cast<GM_ADDR>(params.data_ptr());
    GM_ADDR invp = return_inverse ? reinterpret_cast<GM_ADDR>(inverse.data_ptr()) : (GM_ADDR)nullptr;

    // ---------------- path selection ----------------
    int64_t mode = 0;
    int64_t cap = 0;
    int64_t npass = 0;
    int64_t minKey = 0;
    if (dt == UK_DT_U8 || dt == UK_DT_I8) {
        mode = 0;
        cap = 256;
        npass = 0;
    } else if (dt == UK_DT_F16 || dt == UK_DT_BF16) {
        mode = 0;
        cap = 65536;
        npass = 0;
    } else if (dt == UK_DT_F32) {
        mode = 1;
        cap = (int64_t)4294967296LL;
        npass = 4;
    } else {
        // int32 / int64: the reduced key range is data dependent, so the whole chain (min/max
        // reduction, range decision and dense bitmap) is issued as ONE device call.  The kernels
        // consume minKey through the shared params block and the bitmap stride is the maximum
        // reduced range, so nothing data dependent has to reach the host before the call.  If the
        // range turns out too wide the dense kernels no-op (mode is then 1) and the radix path below
        // redoes the work.
        const int64_t capWordMax = (UK_MAXCAP + 31) / 32;
        auto mm = torch::empty({numBlocks * 2}, optsLong);
        auto bitmaps = torch::empty({numBlocks * capWordMax}, optsI32);
        auto comb = torch::empty({capWordMax}, optsI32);
        auto wordPrefix = torch::empty({capWordMax}, optsI32);
        GM_ADDR mmp = reinterpret_cast<GM_ADDR>(mm.data_ptr());
        GM_ADDR bmp = reinterpret_cast<GM_ADDR>(bitmaps.data_ptr());
        GM_ADDR cmp = reinterpret_cast<GM_ADDR>(comb.data_ptr());
        GM_ADDR wpp = reinterpret_cast<GM_ADDR>(wordPrefix.data_ptr());
        auto acl_call = [=]() -> int {
            if (dt == UK_DT_I32) {
                ukL_mm32(xp, mmp, n, blockLength, stream);
            } else {
                ukL_mm64(xp, mmp, n, blockLength, stream);
            }
            ukL_prep(mmp, pp, numBlocks, 0, dt, stream);
            ukL_dpres(xp, bmp, pp, n, blockLength, dt, 0, (int64_t)UK_MAXCAP, stream);
            ukL_dscan(bmp, cmp, wpp, pp, numBlocks, (int64_t)UK_MAXCAP, stream);
            ukL_dy(cmp, wpp, yp, pp, (int64_t)UK_MAXCAP, numBlocks, dt, 0, stream);
            if (return_inverse) {
                ukL_dinv(xp, cmp, wpp, invp, pp, n, blockLength, dt, 0, (int64_t)UK_MAXCAP, stream);
            }
            return 0;
        };
        at_npu::native::OpCommand::RunOpApi("Unique", acl_call);
        auto hostParams = params.to(torch::Device(torch::kCPU));
        const int64_t *hp = hostParams.const_data_ptr<int64_t>();
        mode = hp[UK_P_MODE];
        if (mode == 0) {
            int64_t dm = hp[UK_P_D];
            TORCH_CHECK(dm >= 0 && dm <= n, "cann_bench.unique internal error: invalid distinct count.");
            std::vector<torch::Tensor> outv;
            outv.push_back(yBig.narrow(0, 0, dm));
            if (return_inverse) {
                outv.push_back(inverse);
            }
            return outv;
        }
        cap = hp[UK_P_CAP];
        npass = hp[UK_P_NPASS];
        minKey = hp[UK_P_MINKEY];
    }

    if (mode == 0) {
        // ---------------- dense bitmap path ----------------
        int64_t capWords = (cap + 31) / 32;
        auto bitmaps = torch::empty({numBlocks * capWords}, optsI32);
        auto comb = torch::empty({capWords}, optsI32);
        auto wordPrefix = torch::empty({capWords}, optsI32);
        GM_ADDR bmp = reinterpret_cast<GM_ADDR>(bitmaps.data_ptr());
        GM_ADDR cmp = reinterpret_cast<GM_ADDR>(comb.data_ptr());
        GM_ADDR wpp = reinterpret_cast<GM_ADDR>(wordPrefix.data_ptr());
        const int64_t capc = cap;
        const int64_t mkc = minKey;

        auto acl_call = [=]() -> int {
            // publish minKey/cap on device so the dense kernels agree on the reduced range
            ukL_setparams(pp, mkc, capc, 0, 0, numBlocks, blockLength, stream);
            ukL_dpres(xp, bmp, pp, n, blockLength, dt, mkc, capc, stream);
            ukL_dscan(bmp, cmp, wpp, pp, numBlocks, capc, stream);
            ukL_dy(cmp, wpp, yp, pp, capc, numBlocks, dt, mkc, stream);
            if (return_inverse) {
                ukL_dinv(xp, cmp, wpp, invp, pp, n, blockLength, dt, mkc, capc, stream);
            }
            return 0;
        };
        at_npu::native::OpCommand::RunOpApi("Unique", acl_call);
    } else {
        // ---------------- radix path ----------------
        auto b0 = torch::empty({n}, optsI32);
        auto b1 = torch::empty({n}, optsI32);
        auto b2 = torch::empty({n}, optsI32);
        auto b3 = torch::empty({n}, optsI32);
        auto b4 = torch::empty({n}, optsI32);
        auto b5 = torch::empty({n}, optsI32);
        auto hist = torch::empty({numBlocks * 256}, optsI32);
        auto st = torch::empty({numBlocks * 256}, optsI32);
        auto cnts = torch::empty({numBlocks}, optsI32);
        auto basearr = torch::empty({numBlocks}, optsI32);

        GM_ADDR a0 = reinterpret_cast<GM_ADDR>(b0.data_ptr());
        GM_ADDR a1 = reinterpret_cast<GM_ADDR>(b1.data_ptr());
        GM_ADDR a2 = reinterpret_cast<GM_ADDR>(b2.data_ptr());
        GM_ADDR a3 = reinterpret_cast<GM_ADDR>(b3.data_ptr());
        GM_ADDR a4 = reinterpret_cast<GM_ADDR>(b4.data_ptr());
        GM_ADDR a5 = reinterpret_cast<GM_ADDR>(b5.data_ptr());
        GM_ADDR hp = reinterpret_cast<GM_ADDR>(hist.data_ptr());
        GM_ADDR sp = reinterpret_cast<GM_ADDR>(st.data_ptr());
        GM_ADDR cnp = reinterpret_cast<GM_ADDR>(cnts.data_ptr());
        GM_ADDR bap = reinterpret_cast<GM_ADDR>(basearr.data_ptr());

        const int64_t npass2 = NPassFor(n);
        const int64_t mkc = minKey;

        auto acl_call = [=]() -> int {
            ukL_rpack(xp, a0, a1, n, blockLength, dt, mkc, stream);
            GM_ADDR ck = a0;
            GM_ADDR cp = a1;
            GM_ADDR dk = a2;
            GM_ADDR dp = a3;
            for (int64_t pass = 0; pass < npass; ++pass) {
                ukL_rhist(ck, hp, n, blockLength, pass, stream);
                ukL_rscan(hp, sp, numBlocks, stream);
                ukL_rscatter(ck, cp, dk, dp, sp, n, blockLength, pass, stream);
                GM_ADDR tk = ck;
                GM_ADDR tp = cp;
                ck = dk;
                cp = dp;
                dk = tk;
                dp = tp;
            }
            ukL_rflag(ck, a4, cnp, n, blockLength, stream);
            // a4 currently holds flags; rank goes to a5 later, so use it only after the widen step
            GM_ADDR flagsp = a4;
            ukL_rbase(cnp, bap, pp, numBlocks, stream);
            ukL_remit(ck, flagsp, bap, yp, a5, n, blockLength, dt, mkc, stream);
            if (return_inverse) {
                GM_ADDR sK = cp;
                GM_ADDR sP = a5;
                GM_ADDR dK = a2;
                GM_ADDR dP = a3;
                for (int64_t pass = 0; pass < npass2; ++pass) {
                    ukL_rhist(sK, hp, n, blockLength, pass, stream);
                    ukL_rscan(hp, sp, numBlocks, stream);
                    ukL_rscatter(sK, sP, dK, dP, sp, n, blockLength, pass, stream);
                    GM_ADDR tk = sK;
                    GM_ADDR tp = sP;
                    sK = dK;
                    sP = dP;
                    dK = tk;
                    dP = tp;
                }
                ukL_rwiden(sP, invp, n, blockLength, stream);
            }
            return 0;
        };
        at_npu::native::OpCommand::RunOpApi("Unique", acl_call);
    }

    int64_t d = ReadDevI64(params, UK_P_D);
    TORCH_CHECK(d >= 0 && d <= n, "cann_bench.unique internal error: invalid distinct count.");

    std::vector<torch::Tensor> out;
    out.push_back(yBig.narrow(0, 0, d));
    if (return_inverse) {
        out.push_back(inverse);
    }
    return out;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("unique", unique_npu);
}

} // namespace cann_bench
