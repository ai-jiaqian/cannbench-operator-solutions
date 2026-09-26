#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""CANN Bench - Ascend operator benchmarks"""
__version__ = "1.0.0"
import torch

try:
    from . import _C
except ImportError as e:
    raise ImportError(
        "Cannot import _C. Please make sure the `cann_bench` package is properly installed. "
    ) from e

# Direct function calls: cann_bench.add(x, y)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.add(x, y)

def sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.sqrt(x)

def depthwise_conv_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                      kernelSize=None, stride=None, padding=None, dilation=None,
                      groups=None) -> torch.Tensor:
    """Depthwise 2-D convolution.

    y[n,c,ho,wo] = bias[c] + sum_{kh,kw} x[n,c,ho*s_h+kh*d_h-p_h, wo*s_w+kw*d_w-p_w] * weight[c,kh,kw]
    """
    if kernelSize is None:
        kernelSize = [weight.shape[1], weight.shape[2]]
    if stride is None:
        stride = [1, 1]
    if padding is None:
        padding = [0, 0]
    if dilation is None:
        dilation = [1, 1]
    if groups is None:
        groups = x.shape[1]
    return torch.ops.cann_bench.depthwise_conv_2d(
        x, weight, bias, list(kernelSize), list(stride), list(padding), list(dilation),
        int(groups))

# Also accessible via torch.ops.cann_bench.add(), torch.ops.cann_bench.sqrt()
# and torch.ops.cann_bench.depthwise_conv_2d()