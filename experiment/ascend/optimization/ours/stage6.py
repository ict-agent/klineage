# Chunk Gated Delta Rule 前向算子 (Ascend NPU Triton Kernel)
#
# 核心递归公式: h_t = exp(g_last) * h_{t-1} + k_t @ ((v_t - w_t @ h_{t-1}) * exp(g_last - g_t))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors
import torch
from vllm.triton_utils import tl, triton

from baseline.fla.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.jit
def safe_exp(x):
    """数值稳定的 exp 计算，当 x > 0 时返回 0"""
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_CHUNK_OFFSETS": lambda args: args["chunk_offsets"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "H", "Hg"])
def chunk_gated_delta_rule_fwd_kernel_h_v128_k64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H,
    Hg,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_CHUNK_OFFSETS: tl.constexpr,
):
    BK: tl.constexpr = 64
    BV: tl.constexpr = 128

    sequence_head_idx = tl.program_id(1)
    sequence_idx = sequence_head_idx // H
    head_idx = sequence_head_idx % H
    kv_head_idx = head_idx // (H // Hg)
    total_tokens = 1 * T

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + sequence_idx).to(tl.int32)
        eos = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int32)
        T = eos - bos
        num_chunks = tl.cdiv(T, BT)
        if USE_CHUNK_OFFSETS:
            chunk_base = tl.load(chunk_offsets + sequence_idx).to(tl.int32)
        else:
            chunk_base = (bos >> 6) + sequence_idx
    else:
        bos = sequence_idx * T
        num_chunks = tl.cdiv(T, BT)
        chunk_base = sequence_idx * num_chunks

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    offs_k = tl.arange(0, BK)
    offs_k_row = offs_k[:, None]
    offs_k_col = offs_k[None, :]
    offs_v = tl.arange(0, BV)[None, :]

    if USE_INITIAL_STATE:
        h0_ptr = h0 + sequence_head_idx * K * V
        b_h_k0 = tl.load(h0_ptr + offs_k_row * V + offs_v).to(tl.float32)
        b_h_k1 = tl.load(
            h0_ptr + (BK + offs_k_row) * V + offs_v
        ).to(tl.float32)
    else:
        b_h_k0 = tl.zeros([BK, BV], dtype=tl.float32)
        b_h_k1 = tl.zeros([BK, BV], dtype=tl.float32)

    k_base = k + bos * Hg * K + kv_head_idx * K
    v_base = v + bos * H * V + head_idx * V
    w_base = w + bos * H * K + head_idx * K
    if USE_G:
        g_base = g + bos + head_idx * total_tokens
    if SAVE_NEW_VALUE:
        v_new_base = v_new + bos * H * V + head_idx * V

    for chunk_idx in range(num_chunks):
        token_start = chunk_idx * BT
        h_base = h + ((chunk_base + chunk_idx) * H + head_idx) * K * V
        p_h_k0 = tl.make_block_ptr(
            h_base,
            (K, V),
            (V, 1),
            (0, 0),
            (BK, BV),
            (1, 0),
        )
        p_h_k1 = tl.make_block_ptr(
            h_base,
            (K, V),
            (V, 1),
            (BK, 0),
            (BK, BV),
            (1, 0),
        )
        tl.store(p_h_k0, b_h_k0.to(p_h_k0.dtype.element_ty))
        tl.store(p_h_k1, b_h_k1.to(p_h_k1.dtype.element_ty))

        b_h_k0_dot = b_h_k0.to(w.dtype.element_ty)
        b_h_k1_dot = b_h_k1.to(w.dtype.element_ty)

        offs_t = (token_start + tl.arange(0, BT))[:, None]
        valid_t = offs_t < T
        b_w_k0 = tl.load(
            w_base + offs_t * stride_w + offs_k_col,
            mask=valid_t,
            other=0.0,
        )
        b_w_k1 = tl.load(
            w_base + offs_t * stride_w + BK + offs_k_col,
            mask=valid_t,
            other=0.0,
        )
        b_u = tl.load(
            v_base + offs_t * stride_v + offs_v,
            mask=valid_t,
            other=0.0,
        )

        if USE_G:
            token_offsets = token_start + tl.arange(0, BT)
            valid_tokens = token_offsets < T
            last_idx = min(token_start + BT, T) - 1
            b_g_last = tl.load(g_base + last_idx)
            b_g = tl.load(
                g_base + token_offsets,
                mask=valid_tokens,
                other=0.0,
            )
            b_g = safe_exp(b_g_last - b_g)
            b_g_last = tl.exp(b_g_last)

        b_wh = tl.dot(b_w_k0, b_h_k0_dot)
        if USE_G:
            b_h_k0_next = b_h_k0 * b_g_last
        else:
            b_h_k0_next = b_h_k0
        b_wh = tl.dot(b_w_k1, b_h_k1_dot, acc=b_wh)
        b_v_new = b_u.to(tl.float32) - b_wh

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(
                v_new_base,
                (T, V),
                (stride_v, 1),
                (token_start, 0),
                (BT, BV),
                (1, 0),
            )
            tl.store(
                p_v_new,
                b_v_new.to(p_v_new.dtype.element_ty),
                boundary_check=(0,),
            )

        if USE_G:
            b_v_new *= b_g[:, None]

        offs_t_k = (token_start + tl.arange(0, BT))[None, :]
        valid_t_k = offs_t_k < T
        b_k_k0 = tl.load(
            k_base + offs_k_row + offs_t_k * stride_k,
            mask=valid_t_k,
            other=0.0,
        )
        b_k_k1 = tl.load(
            k_base + BK + offs_k_row + offs_t_k * stride_k,
            mask=valid_t_k,
            other=0.0,
        )
        b_v_new_dot = b_v_new.to(k.dtype.element_ty)
        b_h_k0 = b_h_k0_next + tl.dot(b_k_k0, b_v_new_dot)
        if USE_G:
            b_h_k1_next = b_h_k1 * b_g_last
        else:
            b_h_k1_next = b_h_k1
        b_h_k1 = b_h_k1_next + tl.dot(b_k_k1, b_v_new_dot)

    if STORE_FINAL_STATE:
        ht_ptr = ht + sequence_head_idx * K * V
        p_ht_k0 = tl.make_block_ptr(
            ht_ptr,
            (K, V),
            (V, 1),
            (0, 0),
            (BK, BV),
            (1, 0),
        )
        p_ht_k1 = tl.make_block_ptr(
            ht_ptr,
            (K, V),
            (V, 1),
            (BK, 0),
            (BK, BV),
            (1, 0),
        )
        tl.store(p_ht_k0, b_h_k0.to(p_ht_k0.dtype.element_ty))
        tl.store(p_ht_k1, b_h_k1.to(p_ht_k1.dtype.element_ty))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    计算带门控的 Delta Rule 前向传播

    Args:
        k: [B, T, Hg, K] Key tensor
        w: [B, T, H, K] Weight tensor
        u: [B, T, H, V] Value tensor
        g: [B, T, H] Gate tensor (可选)
        initial_state: [N, H, K, V] 初始状态，dtype 必须是 torch.float32
        output_final_state: 是否输出最终状态
        chunk_size: Chunk 大小
        save_new_value: 是否保存 v_new
        cu_seqlens: [B+1] 变长序列累积长度

    Returns:
        h: [B, NT, H, K, V] 每个 chunk 的隐藏状态
        v_new: [B, T, H, V] 变换后的 value
        final_state: [N, H, K, V] 最终状态 (如果 output_final_state=True)
    """
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    use_v128_k64 = K == 128 and V == 128 and BT == 64
    if not use_v128_k64:
        from baseline.fla.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h as baseline_fwd_h,
        )

        return baseline_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=BT,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    if cu_seqlens is None:
        NT, chunk_offsets = triton.cdiv(T, BT), None
    else:
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        NT = len(chunk_indices)

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    g_head_first = g.transpose(1, 2).contiguous() if g is not None else None

    def optimized_grid(meta):
        return (1, N * H)

    chunk_gated_delta_rule_fwd_kernel_h_v128_k64[optimized_grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g_head_first,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        num_warps=4,
        num_stages=2,
        multibuffer=True,
    )
    return h, v_new, final_state
