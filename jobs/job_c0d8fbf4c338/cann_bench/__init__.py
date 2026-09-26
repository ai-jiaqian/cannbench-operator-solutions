#!/usr/bin/env python3
"""Latest available official-tasks v1.1.2 L2/L1 candidates."""
import torch

try:
    from . import _C
except ImportError as exc:
    raise ImportError("Cannot import _C. Build and install the package first.") from exc

__all__ = ['dynamic_quant', 'unsorted_segment_sum', 'group_norm', 'arg_max', 'exp', 'foreach_addcdiv_scalar', 'foreach_norm', 'gelu', 'masked_scale', 'mish', 'sigmoid', 'swi_glu']

def dynamic_quant(x: torch.Tensor):
    """Per-token symmetric dynamic quantization along the last dim.

    Returns (y, scale): y is int8 with the same shape as x, scale is float32
    with shape x.shape[:-1] and equals row_max(abs(x)) / 127.
    """
    return torch.ops.cann_bench.dynamic_quant(x)

def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.ops.cann_bench.unsorted_segment_sum(data, segment_ids, num_segments)

def group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """Group normalization: y = (x - mean_g) / sqrt(var_g + eps) * gamma + beta."""
    return torch.ops.cann_bench.group_norm(x, gamma, beta, num_groups, epsilon)

def arg_max(input: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """ArgMax indices along dim; indices = argmax(input, dim=dim)."""
    return torch.ops.cann_bench.arg_max(input, dim, keepdim)

def exp(x: torch.Tensor, base: float = -1.0, scale: float = 1.0, shift: float = 0.0) -> torch.Tensor:
    """y = exp((x * scale + shift) * ln(base)) for base > 0, else exp(x * scale + shift)."""
    return torch.ops.cann_bench.exp(x, base, scale, shift)

def foreach_addcdiv_scalar(x1, x2, x3, scalar):
    return torch.ops.cann_bench.foreach_addcdiv_scalar(x1, x2, x3, scalar)

def foreach_norm(x, scalar: float):
    return torch.ops.cann_bench.foreach_norm(x, scalar)

def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Gaussian Error Linear Unit: y = x * Phi(x) (or the tanh approximation)."""
    return torch.ops.cann_bench.gelu(x, approximate)

def masked_scale(x: torch.Tensor, mask: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """y = x * mask * scale, output dtype follows x."""
    return torch.ops.cann_bench.masked_scale(x, mask, scale)

def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish activation: y = x * tanh(softplus(x))."""
    return torch.ops.cann_bench.mish(x)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """y = 1 / (1 + exp(-x)), elementwise."""
    return torch.ops.cann_bench.sigmoid(x)

def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """SwiGLU activation: silu(chunk(input, 2, dim)[0]) * chunk(input, 2, dim)[1]."""
    return torch.ops.cann_bench.swi_glu(input, dim)
