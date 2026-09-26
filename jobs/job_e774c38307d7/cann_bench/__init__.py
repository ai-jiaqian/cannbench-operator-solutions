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

def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: int = 0,
    rotaryMode: str = "half",
):
    """Rotary position embedding for query and key.

    layout: 0 -> (B, S, N, D); 1 -> (B, N, S, D)
    rotaryMode: "half" | "interleaved"
    """
    return torch.ops.cann_bench.apply_rotary_pos_emb(
        query, key, cos, sin, int(layout), str(rotaryMode)
    )

def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.ops.cann_bench.unsorted_segment_sum(data, segment_ids, num_segments)

# Also accessible via torch.ops.cann_bench.*()
