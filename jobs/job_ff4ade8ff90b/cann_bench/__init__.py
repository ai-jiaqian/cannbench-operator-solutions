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

def grouped_matmul(x: torch.Tensor, weight: torch.Tensor, bias=None, group_list=None,
                   split_item: int = 0, transpose_weight: bool = False):
    """Grouped matmul: y[rows_g] = x[rows_g] @ weight[g] (+ bias[g]).

    group_list is a length-E list of cumulative token counts (last value == M).
    split_item 0/1 returns one tensor per expert, 2/3 returns a single [M, N] tensor.

    The torch schema takes the group list before the optional bias (so that the bias can
    default to None); accept either argument order here.
    """
    if group_list is None and bias is not None and not torch.is_tensor(bias):
        group_list, bias = bias, None
    if group_list is None:
        group_list = []
    return torch.ops.cann_bench.grouped_matmul(x, weight, group_list, bias, split_item, transpose_weight)

# Also accessible via torch.ops.cann_bench.add(), torch.ops.cann_bench.sqrt()
# and torch.ops.cann_bench.grouped_matmul()