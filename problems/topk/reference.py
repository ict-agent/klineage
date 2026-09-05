"""PyTorch reference for row-wise Top-K selection."""

from __future__ import annotations

import torch

BATCH = 64
SEQUENCE_LENGTH = 4096
K = 512
DTYPE = torch.float32


def make_inputs(
    *,
    batch: int = BATCH,
    sequence_length: int = SEQUENCE_LENGTH,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create a row-major ``[batch, sequence_length]`` input tensor."""

    if batch <= 0 or sequence_length <= 0:
        raise ValueError("batch and sequence_length must be positive")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "values": torch.randn(
            (batch, sequence_length),
            dtype=dtype,
            device=device,
            generator=generator,
        ).contiguous()
    }


def torch_ref(
    values: torch.Tensor,
    *,
    k: int = K,
    largest: bool = True,
    sorted: bool = False,
) -> torch.return_types.topk:
    """Return PyTorch's ``(values, indices)`` Top-K result.

    The expert kernel emits int32 indices only.  Correctness checks should cast
    those indices to int64, gather the selected values, and compare the sorted
    selected-value sets; order is intentionally unspecified.
    """

    if values.ndim != 2:
        raise ValueError("values must have shape [batch, sequence_length]")
    if k <= 0 or k > values.shape[-1]:
        raise ValueError(f"k must be in [1, {values.shape[-1]}]")
    return torch.topk(values, k, dim=-1, largest=largest, sorted=sorted)


__all__ = [
    "BATCH",
    "DTYPE",
    "SEQUENCE_LENGTH",
    "K",
    "make_inputs",
    "torch_ref",
]
