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


def lstm(x, weight_ih, weight_hh, inputSize: int, hiddenSize: int, numLayers: int = 1,
         bias: bool = True, batchFirst: bool = False, dropout: float = 0.0,
         bidirectional: bool = False, projSize: int = 0, bias_ih=None, bias_hh=None,
         weight_hr=None, h0=None, c0=None):
    """LSTM forward matching torch.nn.LSTM (TensorList weights).

    weight_ih / weight_hh are lists of length numLayers * num_directions ordered
    [l0, l0_reverse, l1, l1_reverse, ...].  bias_ih / bias_hh are required when bias is true;
    weight_hr is required when projSize > 0.  Returns (y, hn, cn).
    """
    return torch.ops.cann_bench.lstm(x, list(weight_ih), list(weight_hh), inputSize, hiddenSize,
                                     numLayers, bias, batchFirst, dropout, bidirectional, projSize,
                                     bias_ih, bias_hh, weight_hr, h0, c0)

# Also accessible via torch.ops.cann_bench.add() and torch.ops.cann_bench.sqrt()