#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best observed L1 CANN Bench operator bundle."""

import torch

__version__ = "1.0.0"

try:
    from . import _C
except ImportError as exc:
    raise ImportError(
        "Cannot import _C. Please make sure the cann_bench package is properly installed."
    ) from exc


def exp(
    x: torch.Tensor,
    base: float = -1.0,
    scale: float = 1.0,
    shift: float = 0.0,
) -> torch.Tensor:
    return torch.ops.cann_bench.exp(x, base, scale, shift)


def foreach_addcdiv_scalar(
    x1: list[torch.Tensor],
    x2: list[torch.Tensor],
    x3: list[torch.Tensor],
    scalar: float,
) -> list[torch.Tensor]:
    return torch.ops.cann_bench.foreach_addcdiv_scalar(x1, x2, x3, scalar)


def foreach_norm(x: list[torch.Tensor], scalar: float) -> list[torch.Tensor]:
    return torch.ops.cann_bench.foreach_norm(x, scalar)


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return torch.ops.cann_bench.gelu(x, approximate)


def masked_scale(
    x: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    return torch.ops.cann_bench.masked_scale(x, mask, scale)


def mish(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.mish(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.cann_bench.sigmoid(x)


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.ops.cann_bench.swi_glu(input, dim)


__all__ = [
    "exp",
    "foreach_addcdiv_scalar",
    "foreach_norm",
    "gelu",
    "masked_scale",
    "mish",
    "sigmoid",
    "swi_glu",
]
