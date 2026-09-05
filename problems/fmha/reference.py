"""PyTorch SDPA reference for packed-NHD FMHA.

Q, K, and V use the same packed layout as the expert kernel:
``[total_tokens, heads, head_dim]``.  Cumulative sequence-length tensors make
the sequence boundaries explicit and also let the reference cover ragged
inputs without padding them into the kernel ABI.
"""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F

BATCH = 8
HEADS = 64
SEQUENCE_LENGTH = 2048
HEAD_DIM = 128
DTYPE = torch.float16


def make_inputs(
    *,
    batch: int = BATCH,
    heads: int = HEADS,
    sequence_length: int = SEQUENCE_LENGTH,
    head_dim: int = HEAD_DIM,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create fixed-length packed-NHD inputs and their sequence offsets."""

    if min(batch, heads, sequence_length, head_dim) <= 0:
        raise ValueError("all FMHA dimensions must be positive")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    total_tokens = batch * sequence_length
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        sequence_length,
        dtype=torch.int32,
        device=device,
    )
    shape = (total_tokens, heads, head_dim)
    return {
        "q": torch.randn(shape, dtype=dtype, device=device, generator=generator),
        "k": torch.randn(shape, dtype=dtype, device=device, generator=generator),
        "v": torch.randn(shape, dtype=dtype, device=device, generator=generator),
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens.clone(),
    }


def torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Run SDPA per sequence and return packed ``[total_q, H, Dv]`` output."""

    _validate_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()

    outputs: list[torch.Tensor] = []
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1],
        q_offsets[1:],
        k_offsets[:-1],
        k_offsets[1:],
        strict=True,
    ):
        q_batch = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_batch = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_batch = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            q_batch,
            k_batch,
            v_batch,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have packed [tokens, heads, dim] shape")
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError("k and v must have the same token and head dimensions")
    if q.shape[1] != k.shape[1] or q.shape[2] != k.shape[2]:
        raise ValueError(
            "this FMHA workload requires matching Q/K heads and dimensions"
        )
    if len({q.device, k.device, v.device}) != 1:
        raise ValueError("q, k, and v must be on the same device")
    if len({q.dtype, k.dtype, v.dtype}) != 1:
        raise ValueError("q, k, and v must have the same dtype")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cumulative sequence lengths must be rank-1")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q and KV must contain the same number of sequences")
    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()
    if len(q_offsets) < 2 or q_offsets[0] != 0 or k_offsets[0] != 0:
        raise ValueError("cumulative sequence lengths must begin at zero")
    if q_offsets[-1] != q.shape[0] or k_offsets[-1] != k.shape[0]:
        raise ValueError("final cumulative length must equal the packed token count")
    if any(a >= b for a, b in pairwise(q_offsets)):
        raise ValueError("all Q sequences must be non-empty")
    if any(a >= b for a, b in pairwise(k_offsets)):
        raise ValueError("all KV sequences must be non-empty")


__all__ = [
    "BATCH",
    "DTYPE",
    "HEADS",
    "HEAD_DIM",
    "SEQUENCE_LENGTH",
    "make_inputs",
    "torch_ref",
]
