import torch
try:
    from . import _C
except ImportError as e:
    raise ImportError("Cannot import _C. Please make sure cann_bench is installed.") from e
__all__ = ["rms_norm"]

def rms_norm(x: torch.Tensor = None, gamma: torch.Tensor = None, epsilon: float = 1e-6, **kwargs) -> torch.Tensor:
    if x is None:
        x = kwargs.pop("input", None)
    if gamma is None:
        gamma = kwargs.pop("weight", None)
    if epsilon == 1e-6:
        eps = kwargs.pop("eps", None)
        if eps is not None:
            epsilon = eps
    if x is None or gamma is None:
        raise TypeError("rms_norm() missing required argument: 'x' or 'gamma'")
    return torch.ops.cann_bench.rms_norm(x, gamma, float(epsilon))
