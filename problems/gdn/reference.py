# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# The naive GDN definitions below are adapted from
# https://github.com/fla-org/flash-linear-attention under the MIT license.

"""FLA naive GDN definitions and the paper-workload adapter.

This single module contains FLA's recurrent and chunk-naive mathematical
references, deterministic paper inputs, and the workload's Grouped Value
Attention (GVA) head mapping.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BATCH = 1
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
CHUNK_SIZE = 64
SEQUENCE_LENGTH = 4096
DTYPE = torch.bfloat16
GATE_DTYPE = torch.float32
STATE_DTYPE = torch.float32
L2_EPSILON = 1e-6
SWA_RATIO = 0.75


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference PyTorch implementation of recurrent gated delta rule.

    Args:
        q: ``[B, T, H, K]`` queries.
        k: ``[B, T, H, K]`` keys.
        v: ``[B, T, H, V]`` values.
        beta: ``[B, T, H]`` update strengths.
        g: ``[B, T, H]`` log-space forget gates.
        scale: Optional query scale; defaults to ``1 / sqrt(K)``.
        initial_state: Optional ``[B, H, K, V]`` state.
        output_final_state: Whether to return the final state.
    """

    q, k, v, beta, g = (
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (q, k, v, beta, g)
    )
    batch, heads, sequence_length, key_dim = k.shape
    value_dim = v.shape[-1]
    output = torch.zeros(batch, heads, sequence_length, value_dim).to(v)
    state = torch.zeros(batch, heads, key_dim, value_dim).to(v)
    if initial_state is not None:
        state = initial_state.to(torch.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for token in range(sequence_length):
        q_token = q[:, :, token]
        k_token = k[:, :, token]
        v_token = v[:, :, token].clone()
        state = state.clone() * g[:, :, token].exp()[..., None, None]
        beta_token = beta[:, :, token]
        v_token = v_token - (state.clone() * k_token[..., None]).sum(-2)
        v_token = v_token * beta_token[..., None]
        state = state.clone() + k_token.unsqueeze(-1) * v_token.unsqueeze(-2)
        output[:, :, token] = torch.einsum("bhd,bhdm->bhm", q_token, state)

    if not output_final_state:
        state = None
    output = output.transpose(1, 2).contiguous()
    return output, state


def naive_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference PyTorch implementation of chunk gated delta rule.

    Its tensor contract matches :func:`naive_recurrent_gated_delta_rule`.
    ``g`` precedes ``beta`` here to match FLA's public chunk operator.
    """

    block_tokens = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q, k, v, beta, g = (
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (q, k, v, beta, g)
    )
    original_length = q.shape[-2]
    pad_length = (block_tokens - (original_length % block_tokens)) % block_tokens
    if pad_length > 0:
        q = F.pad(q, (0, 0, 0, pad_length))
        k = F.pad(k, (0, 0, 0, pad_length))
        v = F.pad(v, (0, 0, 0, pad_length))
        beta = F.pad(beta, (0, pad_length))
        g = F.pad(g, (0, pad_length))

    q, k, v, beta, g = (tensor.to(torch.float32) for tensor in (q, k, v, beta, g))
    decay = g
    batch, heads, padded_length, key_dim = q.shape
    value_dim = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    if padded_length % block_tokens != 0:
        raise ValueError("padded sequence length must be divisible by chunk_size")

    diagonal_mask = torch.triu(
        torch.ones(
            block_tokens,
            block_tokens,
            dtype=torch.bool,
            device=q.device,
        ),
        diagonal=0,
    )
    q, k, v, k_beta, decay_with_dim = (
        _split_chunks(tensor, block_tokens)
        for tensor in (q, k, v, k_beta, decay.unsqueeze(-1))
    )
    decay = decay_with_dim.squeeze(-1).cumsum(-1)
    decay_exp = decay.exp()[..., None]
    lower_decay = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()
    ).tril()
    attention = -((k_beta @ k.transpose(-1, -2)) * lower_decay).masked_fill(
        diagonal_mask, 0
    )
    for row in range(1, block_tokens):
        attention[..., row, :row] = attention[..., row, :row].clone() + (
            attention[..., row, :row, None].clone() * attention[..., :row, :row].clone()
        ).sum(-2)
    attention = attention + torch.eye(block_tokens, dtype=torch.float, device=q.device)
    k_cumsum = attention @ v
    k_cumdecay = attention @ (k_beta * decay_exp)
    v = k_cumsum

    state = k.new_zeros(batch, heads, key_dim, value_dim)
    if initial_state is not None:
        state = initial_state.to(torch.float32)

    output = torch.zeros_like(v)
    upper_mask = torch.triu(
        torch.ones(
            block_tokens,
            block_tokens,
            dtype=torch.bool,
            device=q.device,
        ),
        diagonal=1,
    )
    for chunk in range(padded_length // block_tokens):
        q_chunk, k_chunk, v_chunk = (
            q[:, :, chunk],
            k[:, :, chunk],
            v[:, :, chunk],
        )
        attention = (
            (q_chunk @ k_chunk.transpose(-1, -2)) * lower_decay[:, :, chunk]
        ).masked_fill_(upper_mask, 0)
        v_prime = k_cumdecay[:, :, chunk] @ state
        v_new = v_chunk - v_prime
        output_from_state = (q_chunk * decay[:, :, chunk, :, None].exp()) @ state
        output[:, :, chunk] = output_from_state + attention @ v_new
        final_decay = decay[:, :, chunk, -1]
        state = (
            state * final_decay[..., None, None].exp()
            + (
                k_chunk * (final_decay[..., None] - decay[:, :, chunk]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        state = None
    output = output.flatten(2, 3)[:, :, :original_length]
    output = output.transpose(1, 2)
    return output, state


def _split_chunks(tensor: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Equivalent to FLA's ``rearrange('b h (n c) d -> b h n c d')``."""

    batch, heads, sequence_length, dimension = tensor.shape
    return tensor.reshape(
        batch,
        heads,
        sequence_length // chunk_size,
        chunk_size,
        dimension,
    )


def make_inputs(
    *,
    batch: int = BATCH,
    sequence_length: int = SEQUENCE_LENGTH,
    qk_heads: int = QK_HEADS,
    value_heads: int = VALUE_HEADS,
    head_dim: int = HEAD_DIM,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cuda",
    seed: int = 0,
    l2_epsilon: float = L2_EPSILON,
    swa_ratio: float = SWA_RATIO,
) -> dict[str, torch.Tensor]:
    """Create normalized Q/K, log-space gates, beta, and initial state."""

    if min(batch, sequence_length, qk_heads, value_heads, head_dim) <= 0:
        raise ValueError("all GDN dimensions must be positive")
    if value_heads % qk_heads != 0:
        raise ValueError("value_heads must be divisible by qk_heads")
    if l2_epsilon <= 0:
        raise ValueError("l2_epsilon must be positive")
    if not 0.0 <= swa_ratio <= 1.0:
        raise ValueError("swa_ratio must be between zero and one")

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    q = _l2_normalize(
        torch.randn(
            (batch, sequence_length, qk_heads, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        ),
        l2_epsilon,
    )
    k = _l2_normalize(
        torch.randn(
            (batch, sequence_length, qk_heads, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        ),
        l2_epsilon,
    )
    v = torch.randn(
        (batch, sequence_length, value_heads, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    g = (
        F.logsigmoid(
            torch.randn(
                (batch, sequence_length, value_heads),
                dtype=GATE_DTYPE,
                device=device,
                generator=generator,
            )
        )
        / 16
    )
    beta = torch.randn(
        (batch, sequence_length, value_heads),
        dtype=GATE_DTYPE,
        device=device,
        generator=generator,
    ).sigmoid()
    initial_state = torch.randn(
        (batch, value_heads, head_dim, head_dim),
        dtype=STATE_DTYPE,
        device=device,
        generator=generator,
    )

    swa_heads = math.ceil(swa_ratio * value_heads)
    swa_mask = torch.zeros(value_heads, dtype=torch.bool, device=device)
    swa_mask[:swa_heads] = True
    swa_mask = swa_mask[torch.randperm(value_heads, device=device, generator=generator)]
    g[:, :, ~swa_mask] = 0.0

    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "g": g.contiguous(),
        "beta": beta.contiguous(),
        "initial_state": initial_state.contiguous(),
    }


def torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
    scale: float | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply paper-specific GVA mapping, then call FLA's chunk-naive oracle."""

    _validate_inputs(q, k, v, g, beta, initial_state, chunk_size)
    repeats = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(repeats, dim=2)
    k = k.repeat_interleave(repeats, dim=2)
    return naive_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def _l2_normalize(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    inverse_norm = torch.rsqrt((value * value).sum(dim=-1, keepdim=True) + epsilon)
    return (value * inverse_norm).to(value.dtype)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have [B, T, H, D] shape")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shapes")
    if q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must share batch and sequence dimensions")
    if v.shape[2] % q.shape[2] != 0:
        raise ValueError("value-head count must be divisible by Q/K-head count")
    expected_gate_shape = v.shape[:3]
    if g.shape != expected_gate_shape or beta.shape != expected_gate_shape:
        raise ValueError(f"g and beta must have shape {expected_gate_shape}")
    tensors = (q, k, v, g, beta)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all GDN inputs must be on the same device")
    if initial_state is not None:
        expected_state_shape = (
            q.shape[0],
            v.shape[2],
            q.shape[-1],
            v.shape[-1],
        )
        if initial_state.shape != expected_state_shape:
            raise ValueError(f"initial_state must have shape {expected_state_shape}")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")


__all__ = [
    "BATCH",
    "CHUNK_SIZE",
    "DTYPE",
    "GATE_DTYPE",
    "HEAD_DIM",
    "L2_EPSILON",
    "QK_HEADS",
    "SEQUENCE_LENGTH",
    "STATE_DTYPE",
    "SWA_RATIO",
    "VALUE_HEADS",
    "make_inputs",
    "naive_chunk_gated_delta_rule",
    "naive_recurrent_gated_delta_rule",
    "torch_ref",
]
