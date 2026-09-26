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

def mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    numKVHeads: int = 1,
    scaleValue: float = -1.0,
    inputLayout: str = "BSND",
    is_causal: bool = False,
) -> torch.Tensor:
    """Multi-head Latent Attention core (concat Q/K, scaled dot product, softmax, PV)."""
    return torch.ops.cann_bench.mla(
        q_nope, q_rope, k_nope, k_rope, v, int(numKVHeads), float(scaleValue),
        str(inputLayout), bool(is_causal)
    )

# Also accessible via torch.ops.cann_bench.add(), torch.ops.cann_bench.sqrt()
# and torch.ops.cann_bench.mla()