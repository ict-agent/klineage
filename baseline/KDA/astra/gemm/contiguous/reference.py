"""PyTorch reference for ``Y = X @ W.T``.

The transposed weight contract is intentional: the expert kernels consume a
contiguous row-major ``weight[N, K]`` tensor rather than ``weight[K, N]``.
"""

from __future__ import annotations

import torch

M = 4096
N = 4096
K = 4096
DTYPE = torch.bfloat16


def make_inputs(
    *,
    m: int = M,
    n: int = N,
    k: int = K,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create contiguous ``x[M,K]`` and ``weight[N,K]`` tensors."""

    if min(m, n, k) <= 0:
        raise ValueError("m, n, and k must be positive")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "x": torch.randn(
            (m, k), dtype=dtype, device=device, generator=generator
        ).contiguous(),
        "weight": torch.randn(
            (n, k), dtype=dtype, device=device, generator=generator
        ).contiguous(),
    }


def torch_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``x @ weight.T`` using PyTorch's matmul implementation."""

    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must both be rank-2 tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            "x and weight must have the same K dimension; "
            f"got {x.shape[1]} and {weight.shape[1]}"
        )
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if x.dtype != weight.dtype:
        raise ValueError("x and weight must have the same dtype")
    return torch.matmul(x, weight.transpose(0, 1), out=out)


__all__ = ["DTYPE", "K", "M", "N", "make_inputs", "torch_ref"]
