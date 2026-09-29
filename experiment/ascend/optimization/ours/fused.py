# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# mypy: ignore-errors
from __future__ import annotations

import torch
import triton.language.extra.cann.extension as al
from vllm.triton_utils import tl, triton


@triton.jit
def _safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit
def _schedule_grouped_varlen_stage6_task(
    cu_seqlens, sequence_head_idx, total_tasks, H: tl.constexpr
):
    group_sequences: tl.constexpr = 3
    group_tasks: tl.constexpr = group_sequences * H
    num_sequences = total_tasks // H
    group_idx = sequence_head_idx // group_tasks
    group_sequence_start = group_idx * group_sequences
    local_task_idx = sequence_head_idx - group_idx * group_tasks
    group_size = min(group_sequences, num_sequences - group_sequence_start)
    tail_four_sequence_start = num_sequences - 4
    is_tail_pair_task = (num_sequences % group_sequences == 1) & (
        sequence_head_idx >= tail_four_sequence_start * H
    )
    if is_tail_pair_task:
        pair_tasks: tl.constexpr = 2 * H
        tail_task_idx = sequence_head_idx - tail_four_sequence_start * H
        pair_idx = tail_task_idx // pair_tasks
        group_sequence_start = tail_four_sequence_start + 2 * pair_idx
        local_task_idx = tail_task_idx - pair_idx * pair_tasks
        group_size = 2
    scheduled_local_task_idx = local_task_idx
    sequence_tokens0 = tl.load(cu_seqlens + group_sequence_start + 1) - tl.load(
        cu_seqlens + group_sequence_start
    )
    if group_size == 3:
        sequence_tokens1 = tl.load(cu_seqlens + group_sequence_start + 2) - tl.load(
            cu_seqlens + group_sequence_start + 1
        )
        sequence_tokens2 = tl.load(cu_seqlens + group_sequence_start + 3) - tl.load(
            cu_seqlens + group_sequence_start + 2
        )
        swap01 = sequence_tokens1 > sequence_tokens0
        high_tokens = tl.where(swap01, sequence_tokens1, sequence_tokens0)
        high_idx = tl.where(swap01, 1, 0)
        low_tokens = tl.where(swap01, sequence_tokens0, sequence_tokens1)
        low_idx = tl.where(swap01, 0, 1)
        swap12 = sequence_tokens2 > low_tokens
        middle_tokens = tl.where(swap12, sequence_tokens2, low_tokens)
        middle_idx = tl.where(swap12, 2, low_idx)
        short_idx = tl.where(swap12, low_idx, 2)
        swap_top = middle_tokens > high_tokens
        long_idx = tl.where(swap_top, middle_idx, high_idx)
        middle_idx = tl.where(swap_top, high_idx, middle_idx)
        long_tokens = tl.where(swap_top, middle_tokens, high_tokens)
        middle_tokens = tl.where(swap_top, high_tokens, middle_tokens)
        short_tokens = min(low_tokens, sequence_tokens2)
        flat_critical_tokens = long_tokens + middle_tokens
        balanced_critical_tokens = max(long_tokens + short_tokens, 2 * middle_tokens)
        should_balance = 100 * flat_critical_tokens >= 105 * balanced_critical_tokens
        if should_balance:
            is_long_task = local_task_idx < 16
            is_middle_task = (local_task_idx >= 16) & (local_task_idx < 24) | (
                local_task_idx >= 40
            )
            middle_head_idx = tl.where(
                local_task_idx < 24, local_task_idx - 16, local_task_idx - 32
            )
            short_head_idx = local_task_idx - 24
            scheduled_local_task_idx = tl.where(
                is_long_task,
                long_idx * H + local_task_idx,
                tl.where(
                    is_middle_task,
                    middle_idx * H + middle_head_idx,
                    short_idx * H + short_head_idx,
                ),
            )
    elif group_size == 2:
        sequence_tokens1 = tl.load(cu_seqlens + group_sequence_start + 2) - tl.load(
            cu_seqlens + group_sequence_start + 1
        )
        sequence1_is_long = sequence_tokens1 > sequence_tokens0
        long_idx = tl.where(sequence1_is_long, 1, 0)
        short_idx = 1 - long_idx
        long_tokens = tl.where(sequence1_is_long, sequence_tokens1, sequence_tokens0)
        short_tokens = tl.where(sequence1_is_long, sequence_tokens0, sequence_tokens1)
        flat_critical_tokens = long_tokens + short_tokens
        balanced_critical_tokens = max(long_tokens, 2 * short_tokens)
        should_balance = 100 * flat_critical_tokens >= 105 * balanced_critical_tokens
        if should_balance:
            is_long_task = (local_task_idx >= 8) & (local_task_idx < 24)
            long_head_idx = local_task_idx - 8
            short_head_idx = tl.where(
                local_task_idx < 8, local_task_idx, local_task_idx - 16
            )
            scheduled_local_task_idx = tl.where(
                is_long_task,
                long_idx * H + long_head_idx,
                short_idx * H + short_head_idx,
            )
    return group_sequence_start * H + scheduled_local_task_idx


@triton.jit
def _stage7_loop_pipeline_produce_qk_qh(
    q,
    k,
    h,
    qk_workspace,
    qh_workspace,
    bos,
    sequence_tokens,
    chunk_base,
    chunk_idx,
    head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    kv_head_idx = head_idx // (H // Hg)
    token_start = chunk_idx * BT
    stride_qk = Hg * K
    q_base = q + bos * Hg * K + kv_head_idx * K
    k_base = k + bos * Hg * K + kv_head_idx * K
    h_base = h + ((chunk_base + chunk_idx) * H + head_idx) * K * V
    p_q = tl.make_block_ptr(
        q_base, (sequence_tokens, K), (stride_qk, 1), (token_start, 0), (BT, K), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k_base, (K, sequence_tokens), (1, stride_qk), (0, token_start), (K, BT), (0, 1)
    )
    p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    qk_worker_base = qk_workspace + worker_idx * BT * V + buffer_id * BT * BT
    qh_worker_base = qh_workspace + worker_idx * K * V + buffer_id * BT * V
    p_qk = tl.make_block_ptr(
        qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qh = tl.make_block_ptr(qh_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_h = tl.load(p_h, volatile=True)
    b_qk = tl.dot(b_q, b_k)
    b_qh = tl.dot(b_q, b_h)
    al.sync_block_wait(
        "vector",
        "cube",
        buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    tl.store(p_qk, b_qk.to(p_qk.dtype.element_ty))
    tl.store(p_qh, b_qh.to(p_qh.dtype.element_ty))
    al.sync_block_set(
        "cube",
        "vector",
        2 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )


@triton.jit
def _stage7_loop_pipeline_preprocess(
    g,
    qk_workspace,
    qh_workspace,
    gated_qk_workspace,
    bos,
    sequence_tokens,
    token_start,
    sequence_idx,
    head_idx,
    worker_idx,
    buffer_id,
    total_tokens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    qk_worker_base = qk_workspace + worker_idx * BT * V + buffer_id * BT * BT
    qh_worker_base = qh_workspace + worker_idx * K * V + buffer_id * BT * V
    gated_qk_worker_base = (
        gated_qk_workspace + worker_idx * BT * V + buffer_id * BT * BT
    )
    p_qk = tl.make_block_ptr(
        qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qh = tl.make_block_ptr(qh_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    p_gated_qk = tl.make_block_ptr(
        gated_qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    al.sync_block_wait(
        "cube",
        "vector",
        2 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_qk = tl.load(p_qk).to(tl.float32)
    b_qh = tl.load(p_qh).to(tl.float32)
    al.sync_block_set(
        "vector",
        "cube",
        buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    token_offsets = token_start + tl.arange(0, BT)
    valid_tokens = token_offsets < sequence_tokens
    if USE_G:
        if IS_VARLEN:
            g_base = g + bos + head_idx * total_tokens
        else:
            g_base = g + (sequence_idx * H + head_idx) * total_tokens
        b_g = tl.load(g_base + token_offsets, mask=valid_tokens, other=0.0)
        b_qh_scaled = b_qh * tl.exp(b_g)[:, None]
        b_qk *= _safe_exp(b_g[:, None] - b_g[None, :])
    else:
        b_qh_scaled = b_qh
    causal_idx = tl.arange(0, BT)
    causal_mask = causal_idx[:, None] >= causal_idx[None, :]
    b_gated_qk = tl.where(causal_mask, b_qk, 0.0)
    tl.store(p_gated_qk, b_gated_qk.to(p_gated_qk.dtype.element_ty))
    al.sync_block_set(
        "vector",
        "cube",
        4 + buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    return b_qh_scaled


@triton.jit
def _stage7_loop_pipeline_qkv(
    v_new,
    gated_qk_workspace,
    qkv_workspace,
    bos,
    sequence_tokens,
    chunk_idx,
    head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    token_start = chunk_idx * BT
    stride_v = H * V
    v_base = v_new + bos * H * V + head_idx * V
    p_v = tl.make_block_ptr(
        v_base, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    gated_qk_worker_base = (
        gated_qk_workspace + worker_idx * BT * V + buffer_id * BT * BT
    )
    qkv_worker_base = qkv_workspace + worker_idx * V * V + buffer_id * BT * V
    p_gated_qk = tl.make_block_ptr(
        gated_qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qkv = tl.make_block_ptr(qkv_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    al.sync_block_wait(
        "vector",
        "cube",
        4 + buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_gated_qk = tl.load(p_gated_qk)
    b_v = tl.load(p_v, boundary_check=(0, 1), volatile=True)
    b_qkv = tl.dot(b_gated_qk, b_v)
    tl.store(p_qkv, b_qkv.to(p_qkv.dtype.element_ty))
    al.sync_block_set(
        "cube",
        "vector",
        6 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )


@triton.jit
def _stage7_loop_pipeline_finalize(
    o,
    qkv_workspace,
    b_qh_scaled,
    scale,
    bos,
    sequence_tokens,
    chunk_idx,
    head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    token_start = chunk_idx * BT
    stride_v = H * V
    o_base = o + bos * H * V + head_idx * V
    qkv_worker_base = qkv_workspace + worker_idx * V * V + buffer_id * BT * V
    p_qkv = tl.make_block_ptr(qkv_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    p_o = tl.make_block_ptr(
        o_base, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    al.sync_block_wait(
        "cube",
        "vector",
        6 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_qkv = tl.load(p_qkv).to(tl.float32)
    b_o = (b_qh_scaled + b_qkv) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _stage7_head_pair_produce_qk_qh(
    q,
    k,
    h,
    qk_workspace,
    qh_workspace,
    bos,
    sequence_tokens,
    chunk_base,
    chunk_idx,
    kv_head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = H // Hg
    head_idx0 = kv_head_idx * GROUP_SIZE
    head_idx1 = head_idx0 + 1
    token_start = chunk_idx * BT
    stride_qk = Hg * K
    q_base = q + bos * Hg * K + kv_head_idx * K
    k_base = k + bos * Hg * K + kv_head_idx * K
    h_base0 = h + ((chunk_base + chunk_idx) * H + head_idx0) * K * V
    h_base1 = h + ((chunk_base + chunk_idx) * H + head_idx1) * K * V
    p_q = tl.make_block_ptr(
        q_base, (sequence_tokens, K), (stride_qk, 1), (token_start, 0), (BT, K), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k_base, (K, sequence_tokens), (1, stride_qk), (0, token_start), (K, BT), (0, 1)
    )
    p_h0 = tl.make_block_ptr(h_base0, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    p_h1 = tl.make_block_ptr(h_base1, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    qk_worker_base = qk_workspace + worker_idx * 2 * BT * BT + buffer_id * BT * BT
    qh_worker_base = qh_workspace + worker_idx * 4 * BT * V + buffer_id * 2 * BT * V
    p_qk = tl.make_block_ptr(
        qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qh0 = tl.make_block_ptr(qh_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    p_qh1 = tl.make_block_ptr(
        qh_worker_base + BT * V, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_h0 = tl.load(p_h0, volatile=True)
    b_h1 = tl.load(p_h1, volatile=True)
    b_qk = tl.dot(b_q, b_k)
    b_qh0 = tl.dot(b_q, b_h0)
    b_qh1 = tl.dot(b_q, b_h1)
    al.sync_block_wait(
        "vector",
        "cube",
        buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    tl.store(p_qk, b_qk.to(p_qk.dtype.element_ty))
    tl.store(p_qh0, b_qh0.to(p_qh0.dtype.element_ty))
    tl.store(p_qh1, b_qh1.to(p_qh1.dtype.element_ty))
    al.sync_block_set(
        "cube",
        "vector",
        2 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )


@triton.jit
def _stage7_head_pair_preprocess(
    g,
    qk_workspace,
    qh_workspace,
    gated_qk_workspace,
    bos,
    sequence_tokens,
    token_start,
    sequence_idx,
    kv_head_idx,
    worker_idx,
    buffer_id,
    total_tokens,
    H: tl.constexpr,
    Hg: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = H // Hg
    head_idx0 = kv_head_idx * GROUP_SIZE
    head_idx1 = head_idx0 + 1
    qk_worker_base = qk_workspace + worker_idx * 2 * BT * BT + buffer_id * BT * BT
    qh_worker_base = qh_workspace + worker_idx * 4 * BT * V + buffer_id * 2 * BT * V
    gated_worker_base = (
        gated_qk_workspace + worker_idx * 4 * BT * BT + buffer_id * 2 * BT * BT
    )
    p_qk = tl.make_block_ptr(
        qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qh0 = tl.make_block_ptr(qh_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    p_qh1 = tl.make_block_ptr(
        qh_worker_base + BT * V, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    p_gated0 = tl.make_block_ptr(
        gated_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_gated1 = tl.make_block_ptr(
        gated_worker_base + BT * BT, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    al.sync_block_wait(
        "cube",
        "vector",
        2 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_qk = tl.load(p_qk).to(tl.float32)
    b_qh0 = tl.load(p_qh0).to(tl.float32)
    b_qh1 = tl.load(p_qh1).to(tl.float32)
    al.sync_block_set(
        "vector",
        "cube",
        buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    token_offsets = token_start + tl.arange(0, BT)
    valid_tokens = token_offsets < sequence_tokens
    if USE_G:
        if IS_VARLEN:
            g_base0 = g + bos + head_idx0 * total_tokens
            g_base1 = g + bos + head_idx1 * total_tokens
        else:
            g_base0 = g + (sequence_idx * H + head_idx0) * total_tokens
            g_base1 = g + (sequence_idx * H + head_idx1) * total_tokens
        b_g0 = tl.load(g_base0 + token_offsets, mask=valid_tokens, other=0.0)
        b_g1 = tl.load(g_base1 + token_offsets, mask=valid_tokens, other=0.0)
        b_qh0 *= tl.exp(b_g0)[:, None]
        b_qh1 *= tl.exp(b_g1)[:, None]
        b_gated0 = b_qk * _safe_exp(b_g0[:, None] - b_g0[None, :])
        b_gated1 = b_qk * _safe_exp(b_g1[:, None] - b_g1[None, :])
    else:
        b_gated0 = b_qk
        b_gated1 = b_qk
    causal_idx = tl.arange(0, BT)
    causal_mask = causal_idx[:, None] >= causal_idx[None, :]
    b_gated0 = tl.where(causal_mask, b_gated0, 0.0)
    b_gated1 = tl.where(causal_mask, b_gated1, 0.0)
    tl.store(p_gated0, b_gated0.to(p_gated0.dtype.element_ty))
    tl.store(p_gated1, b_gated1.to(p_gated1.dtype.element_ty))
    al.sync_block_set(
        "vector",
        "cube",
        4 + buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    return (b_qh0, b_qh1)


@triton.jit
def _stage7_head_pair_qkv(
    v_new,
    gated_qk_workspace,
    qkv_workspace,
    bos,
    sequence_tokens,
    chunk_idx,
    kv_head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    Hg: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = H // Hg
    head_idx0 = kv_head_idx * GROUP_SIZE
    head_idx1 = head_idx0 + 1
    token_start = chunk_idx * BT
    stride_v = H * V
    v_base0 = v_new + bos * H * V + head_idx0 * V
    v_base1 = v_new + bos * H * V + head_idx1 * V
    p_v0 = tl.make_block_ptr(
        v_base0, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    p_v1 = tl.make_block_ptr(
        v_base1, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    gated_worker_base = (
        gated_qk_workspace + worker_idx * 4 * BT * BT + buffer_id * 2 * BT * BT
    )
    qkv_worker_base = qkv_workspace + worker_idx * 4 * BT * V + buffer_id * 2 * BT * V
    p_gated0 = tl.make_block_ptr(
        gated_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_gated1 = tl.make_block_ptr(
        gated_worker_base + BT * BT, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qkv0 = tl.make_block_ptr(
        qkv_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    p_qkv1 = tl.make_block_ptr(
        qkv_worker_base + BT * V, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    al.sync_block_wait(
        "vector",
        "cube",
        4 + buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_gated0 = tl.load(p_gated0)
    b_gated1 = tl.load(p_gated1)
    b_v0 = tl.load(p_v0, boundary_check=(0, 1), volatile=True)
    b_v1 = tl.load(p_v1, boundary_check=(0, 1), volatile=True)
    b_qkv0 = tl.dot(b_gated0, b_v0)
    b_qkv1 = tl.dot(b_gated1, b_v1)
    tl.store(p_qkv0, b_qkv0.to(p_qkv0.dtype.element_ty))
    tl.store(p_qkv1, b_qkv1.to(p_qkv1.dtype.element_ty))
    al.sync_block_set(
        "cube",
        "vector",
        6 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )


@triton.jit
def _stage7_head_pair_finalize(
    o,
    qkv_workspace,
    b_qh0,
    b_qh1,
    scale,
    bos,
    sequence_tokens,
    chunk_idx,
    kv_head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    Hg: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = H // Hg
    head_idx0 = kv_head_idx * GROUP_SIZE
    head_idx1 = head_idx0 + 1
    token_start = chunk_idx * BT
    stride_v = H * V
    o_base0 = o + bos * H * V + head_idx0 * V
    o_base1 = o + bos * H * V + head_idx1 * V
    qkv_worker_base = qkv_workspace + worker_idx * 4 * BT * V + buffer_id * 2 * BT * V
    p_qkv0 = tl.make_block_ptr(
        qkv_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    p_qkv1 = tl.make_block_ptr(
        qkv_worker_base + BT * V, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    p_o0 = tl.make_block_ptr(
        o_base0, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    p_o1 = tl.make_block_ptr(
        o_base1, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    al.sync_block_wait(
        "cube",
        "vector",
        6 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_qkv0 = tl.load(p_qkv0).to(tl.float32)
    b_qkv1 = tl.load(p_qkv1).to(tl.float32)
    b_o0 = (b_qh0 + b_qkv0) * scale
    b_o1 = (b_qh1 + b_qkv1) * scale
    tl.store(p_o0, b_o0.to(p_o0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_o1, b_o1.to(p_o1.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _run_stage7_head_pair_loop_pipeline(
    q,
    k,
    v_new,
    h,
    g,
    o,
    cu_seqlens,
    qk_workspace,
    qh_workspace,
    gated_qk_workspace,
    qkv_workspace,
    scale,
    T,
    TOTAL_TASKS,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    total_tokens = 1 * T
    num_sequences = TOTAL_TASKS // H
    al.sync_block_set(
        "vector",
        "cube",
        0,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    al.sync_block_set(
        "vector",
        "cube",
        1,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    db_flag = 0
    global_task_base = 0
    for sequence_idx in range(num_sequences):
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + sequence_idx).to(tl.int32)
            eos = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int32)
            sequence_tokens = eos - bos
            chunk_base = bos // BT + sequence_idx
        else:
            bos = sequence_idx * T
            sequence_tokens = T
            chunk_base = sequence_idx * tl.cdiv(T, BT)
        num_chunks = tl.cdiv(sequence_tokens, BT)
        sequence_tasks = num_chunks * Hg
        local_start = (
            worker_idx - global_task_base % num_workers + num_workers
        ) % num_workers
        if local_start < sequence_tasks:
            first_chunk_idx = local_start // Hg
            first_kv_head_idx = local_start % Hg
            _stage7_head_pair_produce_qk_qh(
                q,
                k,
                h,
                qk_workspace,
                qh_workspace,
                bos,
                sequence_tokens,
                chunk_base,
                first_chunk_idx,
                first_kv_head_idx,
                worker_idx,
                db_flag % 2,
                H=H,
                Hg=Hg,
                K=K,
                V=V,
                BT=BT,
            )
            for local_task_idx in range(local_start, sequence_tasks, num_workers):
                chunk_idx = local_task_idx // Hg
                kv_head_idx = local_task_idx % Hg
                buffer_id = db_flag % 2
                (b_qh0, b_qh1) = _stage7_head_pair_preprocess(
                    g,
                    qk_workspace,
                    qh_workspace,
                    gated_qk_workspace,
                    bos,
                    sequence_tokens,
                    chunk_idx * BT,
                    sequence_idx,
                    kv_head_idx,
                    worker_idx,
                    buffer_id,
                    total_tokens,
                    H=H,
                    Hg=Hg,
                    V=V,
                    BT=BT,
                    USE_G=USE_G,
                    IS_VARLEN=IS_VARLEN,
                )
                next_task_idx = local_task_idx + num_workers
                if next_task_idx < sequence_tasks:
                    _stage7_head_pair_produce_qk_qh(
                        q,
                        k,
                        h,
                        qk_workspace,
                        qh_workspace,
                        bos,
                        sequence_tokens,
                        chunk_base,
                        next_task_idx // Hg,
                        next_task_idx % Hg,
                        worker_idx,
                        (db_flag + 1) % 2,
                        H=H,
                        Hg=Hg,
                        K=K,
                        V=V,
                        BT=BT,
                    )
                _stage7_head_pair_qkv(
                    v_new,
                    gated_qk_workspace,
                    qkv_workspace,
                    bos,
                    sequence_tokens,
                    chunk_idx,
                    kv_head_idx,
                    worker_idx,
                    buffer_id,
                    H=H,
                    Hg=Hg,
                    V=V,
                    BT=BT,
                )
                _stage7_head_pair_finalize(
                    o,
                    qkv_workspace,
                    b_qh0,
                    b_qh1,
                    scale,
                    bos,
                    sequence_tokens,
                    chunk_idx,
                    kv_head_idx,
                    worker_idx,
                    buffer_id,
                    H=H,
                    Hg=Hg,
                    V=V,
                    BT=BT,
                )
                db_flag += 1
        global_task_base += sequence_tasks
    al.sync_block_wait(
        "vector",
        "cube",
        0,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    al.sync_block_wait(
        "vector",
        "cube",
        1,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )


@triton.jit
def _run_stage7_loop_pipeline(
    q,
    k,
    v_new,
    h,
    g,
    o,
    cu_seqlens,
    qk_workspace,
    qh_workspace,
    gated_qk_workspace,
    qkv_workspace,
    scale,
    T,
    TOTAL_TASKS,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    total_tokens = 1 * T
    num_sequences = TOTAL_TASKS // H
    al.sync_block_set(
        "vector",
        "cube",
        0,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    al.sync_block_set(
        "vector",
        "cube",
        1,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    db_flag = 0
    global_task_base = 0
    for sequence_idx in range(num_sequences):
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + sequence_idx).to(tl.int32)
            eos = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int32)
            sequence_tokens = eos - bos
            chunk_base = bos // BT + sequence_idx
        else:
            bos = sequence_idx * T
            sequence_tokens = T
            chunk_base = sequence_idx * tl.cdiv(T, BT)
        num_chunks = tl.cdiv(sequence_tokens, BT)
        sequence_tasks = num_chunks * H
        local_start = (
            worker_idx - global_task_base % num_workers + num_workers
        ) % num_workers
        if local_start < sequence_tasks:
            first_chunk_idx = local_start // H
            first_head_idx = local_start % H
            _stage7_loop_pipeline_produce_qk_qh(
                q=q,
                k=k,
                h=h,
                qk_workspace=qk_workspace,
                qh_workspace=qh_workspace,
                bos=bos,
                sequence_tokens=sequence_tokens,
                chunk_base=chunk_base,
                chunk_idx=first_chunk_idx,
                head_idx=first_head_idx,
                worker_idx=worker_idx,
                buffer_id=db_flag % 2,
                H=H,
                Hg=Hg,
                K=K,
                V=V,
                BT=BT,
            )
            for local_task_idx in range(local_start, sequence_tasks, num_workers):
                chunk_idx = local_task_idx // H
                head_idx = local_task_idx % H
                buffer_id = db_flag % 2
                token_start = chunk_idx * BT
                b_qh_scaled = _stage7_loop_pipeline_preprocess(
                    g=g,
                    qk_workspace=qk_workspace,
                    qh_workspace=qh_workspace,
                    gated_qk_workspace=gated_qk_workspace,
                    bos=bos,
                    sequence_tokens=sequence_tokens,
                    token_start=token_start,
                    sequence_idx=sequence_idx,
                    head_idx=head_idx,
                    worker_idx=worker_idx,
                    buffer_id=buffer_id,
                    total_tokens=total_tokens,
                    H=H,
                    K=K,
                    V=V,
                    BT=BT,
                    USE_G=USE_G,
                    IS_VARLEN=IS_VARLEN,
                )
                next_task_idx = local_task_idx + num_workers
                if next_task_idx < sequence_tasks:
                    next_chunk_idx = next_task_idx // H
                    next_head_idx = next_task_idx % H
                    _stage7_loop_pipeline_produce_qk_qh(
                        q=q,
                        k=k,
                        h=h,
                        qk_workspace=qk_workspace,
                        qh_workspace=qh_workspace,
                        bos=bos,
                        sequence_tokens=sequence_tokens,
                        chunk_base=chunk_base,
                        chunk_idx=next_chunk_idx,
                        head_idx=next_head_idx,
                        worker_idx=worker_idx,
                        buffer_id=(db_flag + 1) % 2,
                        H=H,
                        Hg=Hg,
                        K=K,
                        V=V,
                        BT=BT,
                    )
                _stage7_loop_pipeline_qkv(
                    v_new=v_new,
                    gated_qk_workspace=gated_qk_workspace,
                    qkv_workspace=qkv_workspace,
                    bos=bos,
                    sequence_tokens=sequence_tokens,
                    chunk_idx=chunk_idx,
                    head_idx=head_idx,
                    worker_idx=worker_idx,
                    buffer_id=buffer_id,
                    H=H,
                    V=V,
                    BT=BT,
                )
                _stage7_loop_pipeline_finalize(
                    o=o,
                    qkv_workspace=qkv_workspace,
                    b_qh_scaled=b_qh_scaled,
                    scale=scale,
                    bos=bos,
                    sequence_tokens=sequence_tokens,
                    chunk_idx=chunk_idx,
                    head_idx=head_idx,
                    worker_idx=worker_idx,
                    buffer_id=buffer_id,
                    H=H,
                    V=V,
                    BT=BT,
                )
                db_flag += 1
        global_task_base += sequence_tasks
    al.sync_block_wait(
        "vector",
        "cube",
        0,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    al.sync_block_wait(
        "vector",
        "cube",
        1,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "TOTAL_TASKS", "scale"])
def _fused_stage6_7_kernel(
    q,
    k,
    u,
    w,
    g,
    h,
    v_new,
    h0,
    ht,
    o,
    cu_seqlens,
    state_workspace,
    wh_workspace,
    gated_v_workspace,
    kv_workspace,
    scale,
    T,
    TOTAL_TASKS,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr = False,
    STORE_FINAL_STATE: tl.constexpr = False,
    IS_VARLEN: tl.constexpr = False,
    STAGE7_HEAD_PAIR: tl.constexpr = False,
    STAGE6_BALANCE_VARLEN_N: tl.constexpr = 0,
):
    BK: tl.constexpr = 64
    BV: tl.constexpr = 128
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    total_tokens = 1 * T
    offs_k = tl.arange(0, BK)
    offs_k_row = offs_k[:, None]
    offs_v = tl.arange(0, BV)[None, :]
    offs_t_row = tl.arange(0, BT)[:, None]
    state_worker_base = state_workspace + worker_idx * K * V
    wh_worker_base = wh_workspace + worker_idx * BT * V
    gated_v_worker_base = gated_v_workspace + worker_idx * BT * V
    kv_worker_base = kv_workspace + worker_idx * K * V
    if STAGE6_BALANCE_VARLEN_N == 2:
        sequence_tokens0 = tl.load(cu_seqlens + 1) - tl.load(cu_seqlens)
        sequence_tokens1 = tl.load(cu_seqlens + 2) - tl.load(cu_seqlens + 1)
        sequence1_is_long = sequence_tokens1 > sequence_tokens0
        long_sequence_idx = tl.where(sequence1_is_long, 1, 0)
        short_sequence_idx = 1 - long_sequence_idx
        long_sequence_tokens = tl.where(
            sequence1_is_long, sequence_tokens1, sequence_tokens0
        )
        short_sequence_tokens = tl.where(
            sequence1_is_long, sequence_tokens0, sequence_tokens1
        )
        flat_critical_tokens = long_sequence_tokens + short_sequence_tokens
        balanced_critical_tokens = max(long_sequence_tokens, 2 * short_sequence_tokens)
        should_balance_stage6 = (
            100 * flat_critical_tokens >= 105 * balanced_critical_tokens
        )
    elif STAGE6_BALANCE_VARLEN_N == 3:
        sequence_tokens0 = tl.load(cu_seqlens + 1) - tl.load(cu_seqlens)
        sequence_tokens1 = tl.load(cu_seqlens + 2) - tl.load(cu_seqlens + 1)
        sequence_tokens2 = tl.load(cu_seqlens + 3) - tl.load(cu_seqlens + 2)
        swap01 = sequence_tokens1 > sequence_tokens0
        high_tokens = tl.where(swap01, sequence_tokens1, sequence_tokens0)
        high_idx = tl.where(swap01, 1, 0)
        low_tokens = tl.where(swap01, sequence_tokens0, sequence_tokens1)
        low_idx = tl.where(swap01, 0, 1)
        swap12 = sequence_tokens2 > low_tokens
        middle_tokens = tl.where(swap12, sequence_tokens2, low_tokens)
        middle_idx = tl.where(swap12, 2, low_idx)
        short_sequence_idx = tl.where(swap12, low_idx, 2)
        swap_top = middle_tokens > high_tokens
        long_sequence_idx = tl.where(swap_top, middle_idx, high_idx)
        middle_sequence_idx = tl.where(swap_top, high_idx, middle_idx)
        long_sequence_tokens = tl.where(swap_top, middle_tokens, high_tokens)
        middle_sequence_tokens = tl.where(swap_top, high_tokens, middle_tokens)
        short_sequence_tokens = min(low_tokens, sequence_tokens2)
        flat_critical_tokens = long_sequence_tokens + middle_sequence_tokens
        balanced_critical_tokens = max(
            long_sequence_tokens + short_sequence_tokens, 2 * middle_sequence_tokens
        )
        should_balance_stage6 = (
            100 * flat_critical_tokens >= 105 * balanced_critical_tokens
        )
    for sequence_head_idx in range(worker_idx, TOTAL_TASKS, num_workers):
        scheduled_sequence_head_idx = sequence_head_idx
        if STAGE6_BALANCE_VARLEN_N == -1:
            scheduled_sequence_head_idx = _schedule_grouped_varlen_stage6_task(
                cu_seqlens, sequence_head_idx, TOTAL_TASKS, H=H
            )
        elif STAGE6_BALANCE_VARLEN_N == 2:
            if should_balance_stage6:
                is_long_task = (sequence_head_idx >= 8) & (sequence_head_idx < 24)
                long_head_idx = sequence_head_idx - 8
                short_head_idx = tl.where(
                    sequence_head_idx < 8, sequence_head_idx, sequence_head_idx - 16
                )
                scheduled_sequence_head_idx = tl.where(
                    is_long_task,
                    long_sequence_idx * H + long_head_idx,
                    short_sequence_idx * H + short_head_idx,
                )
        elif STAGE6_BALANCE_VARLEN_N == 3:
            if should_balance_stage6:
                is_long_task = sequence_head_idx < 16
                is_middle_task = (sequence_head_idx >= 16) & (
                    sequence_head_idx < 24
                ) | (sequence_head_idx >= 40)
                middle_head_idx = tl.where(
                    sequence_head_idx < 24,
                    sequence_head_idx - 16,
                    sequence_head_idx - 32,
                )
                short_head_idx = sequence_head_idx - 24
                scheduled_sequence_head_idx = tl.where(
                    is_long_task,
                    long_sequence_idx * H + sequence_head_idx,
                    tl.where(
                        is_middle_task,
                        middle_sequence_idx * H + middle_head_idx,
                        short_sequence_idx * H + short_head_idx,
                    ),
                )
        sequence_idx = scheduled_sequence_head_idx // H
        head_idx = scheduled_sequence_head_idx % H
        kv_head_idx = head_idx // (H // Hg)
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + sequence_idx).to(tl.int32)
            eos = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int32)
            sequence_tokens = eos - bos
            num_chunks = tl.cdiv(sequence_tokens, BT)
            chunk_base = bos // BT + sequence_idx
        else:
            bos = sequence_idx * T
            sequence_tokens = T
            num_chunks = tl.cdiv(sequence_tokens, BT)
            chunk_base = sequence_idx * num_chunks
        stride_k = Hg * K
        stride_u = H * V
        stride_w = H * K
        k_base = k + bos * Hg * K + kv_head_idx * K
        u_base = u + bos * H * V + head_idx * V
        w_base = w + bos * H * K + head_idx * K
        v_new_base = v_new + bos * H * V + head_idx * V
        if USE_G:
            if IS_VARLEN:
                g_base = g + bos + head_idx * total_tokens
            else:
                g_base = g + (sequence_idx * H + head_idx) * total_tokens
        if USE_INITIAL_STATE:
            h0_base = h0 + scheduled_sequence_head_idx * K * V
            b_h_k0 = tl.load(h0_base + offs_k_row * V + offs_v).to(tl.float32)
            b_h_k1 = tl.load(h0_base + (BK + offs_k_row) * V + offs_v).to(tl.float32)
        else:
            b_h_k0 = tl.zeros([BK, BV], dtype=tl.float32)
            b_h_k1 = tl.zeros([BK, BV], dtype=tl.float32)
        p_state_k0 = tl.make_block_ptr(
            state_worker_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
        )
        p_state_k1 = tl.make_block_ptr(
            state_worker_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
        )
        tl.store(p_state_k0, b_h_k0.to(p_state_k0.dtype.element_ty))
        tl.store(p_state_k1, b_h_k1.to(p_state_k1.dtype.element_ty))
        first_h_base = h + (chunk_base * H + head_idx) * K * V
        p_first_h_k0 = tl.make_block_ptr(
            first_h_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
        )
        p_first_h_k1 = tl.make_block_ptr(
            first_h_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
        )
        tl.store(p_first_h_k0, b_h_k0.to(p_first_h_k0.dtype.element_ty))
        tl.store(p_first_h_k1, b_h_k1.to(p_first_h_k1.dtype.element_ty))
        al.sync_block_set(
            "vector",
            "cube",
            3,
            sender_pipe=al.PIPE.PIPE_MTE3,
            receiver_pipe=al.PIPE.PIPE_MTE2,
        )
        for chunk_idx in range(num_chunks):
            token_start = chunk_idx * BT
            valid_t_row = offs_t_row + token_start < sequence_tokens
            al.sync_block_wait(
                "vector",
                "cube",
                3,
                sender_pipe=al.PIPE.PIPE_MTE3,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            p_state_full = tl.make_block_ptr(
                state_worker_base, (K, V), (V, 1), (0, 0), (K, BV), (1, 0)
            )
            p_w_full = tl.make_block_ptr(
                w_base,
                (sequence_tokens, K),
                (stride_w, 1),
                (token_start, 0),
                (BT, K),
                (1, 0),
            )
            b_state_full = tl.load(p_state_full)
            b_w_full = tl.load(p_w_full, boundary_check=(0, 1))
            b_wh = tl.dot(b_w_full, b_state_full)
            p_wh = tl.make_block_ptr(
                wh_worker_base, (BT, V), (V, 1), (0, 0), (BT, BV), (1, 0)
            )
            tl.store(p_wh, b_wh.to(p_wh.dtype.element_ty))
            al.sync_block_set(
                "cube",
                "vector",
                0,
                sender_pipe=al.PIPE.PIPE_FIX,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            b_u = tl.load(
                u_base + (token_start + offs_t_row) * stride_u + offs_v,
                mask=valid_t_row,
                other=0.0,
            ).to(tl.float32)
            if USE_G:
                token_offsets = token_start + tl.arange(0, BT)
                valid_tokens = token_offsets < sequence_tokens
                last_idx = min(token_start + BT, sequence_tokens) - 1
                b_g_last_raw = tl.load(g_base + last_idx)
                b_g = tl.load(g_base + token_offsets, mask=valid_tokens, other=0.0)
                b_decay = _safe_exp(b_g_last_raw - b_g)
                b_state_decay = tl.exp(b_g_last_raw)
            al.sync_block_wait(
                "cube",
                "vector",
                0,
                sender_pipe=al.PIPE.PIPE_FIX,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            b_wh = tl.load(p_wh).to(tl.float32)
            b_v_new = b_u - b_wh
            p_v_new = tl.make_block_ptr(
                v_new_base,
                (sequence_tokens, V),
                (stride_u, 1),
                (token_start, 0),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0,))
            if USE_G:
                b_h_k0 *= b_state_decay
                b_h_k1 *= b_state_decay
            p_k_full = tl.make_block_ptr(
                k_base,
                (K, sequence_tokens),
                (1, stride_k),
                (0, token_start),
                (K, BT),
                (0, 1),
            )
            b_k_full = tl.load(p_k_full, boundary_check=(0, 1))
            if USE_G:
                b_gated_v = b_v_new * b_decay[:, None]
            else:
                b_gated_v = b_v_new
            p_gated_v = tl.make_block_ptr(
                gated_v_worker_base, (BT, V), (V, 1), (0, 0), (BT, BV), (1, 0)
            )
            tl.store(p_gated_v, b_gated_v.to(p_gated_v.dtype.element_ty))
            al.sync_block_set(
                "vector",
                "cube",
                1,
                sender_pipe=al.PIPE.PIPE_MTE3,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            al.sync_block_wait(
                "vector",
                "cube",
                1,
                sender_pipe=al.PIPE.PIPE_MTE3,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            b_v_new_cube = tl.load(p_gated_v)
            b_k_full_cube = b_k_full
            p_kv_k0 = tl.make_block_ptr(
                kv_worker_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
            )
            p_kv_k1 = tl.make_block_ptr(
                kv_worker_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
            )
            p_kv_full = tl.make_block_ptr(
                kv_worker_base, (K, V), (V, 1), (0, 0), (K, BV), (1, 0)
            )
            b_kv_full = tl.dot(b_k_full_cube, b_v_new_cube)
            tl.store(p_kv_full, b_kv_full.to(p_kv_full.dtype.element_ty))
            al.sync_block_set(
                "cube",
                "vector",
                2,
                sender_pipe=al.PIPE.PIPE_FIX,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            al.sync_block_wait(
                "cube",
                "vector",
                2,
                sender_pipe=al.PIPE.PIPE_FIX,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            b_h_k0 += tl.load(p_kv_k0).to(tl.float32)
            b_h_k1 += tl.load(p_kv_k1).to(tl.float32)
            if chunk_idx + 1 < num_chunks:
                tl.store(p_state_k0, b_h_k0.to(p_state_k0.dtype.element_ty))
                tl.store(p_state_k1, b_h_k1.to(p_state_k1.dtype.element_ty))
                next_h_base = h + ((chunk_base + chunk_idx + 1) * H + head_idx) * K * V
                p_next_h_k0 = tl.make_block_ptr(
                    next_h_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
                )
                p_next_h_k1 = tl.make_block_ptr(
                    next_h_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
                )
                tl.store(p_next_h_k0, b_h_k0.to(p_next_h_k0.dtype.element_ty))
                tl.store(p_next_h_k1, b_h_k1.to(p_next_h_k1.dtype.element_ty))
            if chunk_idx + 1 < num_chunks:
                al.sync_block_set(
                    "vector",
                    "cube",
                    3,
                    sender_pipe=al.PIPE.PIPE_MTE3,
                    receiver_pipe=al.PIPE.PIPE_MTE2,
                )
        if STORE_FINAL_STATE:
            ht_base = ht + scheduled_sequence_head_idx * K * V
            p_ht_k0 = tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
            )
            p_ht_k1 = tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
            )
            tl.store(p_ht_k0, b_h_k0)
            tl.store(p_ht_k1, b_h_k1)
    al.sync_block_all("all", 0)
    if STAGE7_HEAD_PAIR:
        _run_stage7_head_pair_loop_pipeline(
            q=q,
            k=k,
            v_new=v_new,
            h=h,
            g=g,
            o=o,
            cu_seqlens=cu_seqlens,
            qk_workspace=wh_workspace,
            qh_workspace=state_workspace,
            gated_qk_workspace=gated_v_workspace,
            qkv_workspace=kv_workspace,
            scale=scale,
            T=T,
            TOTAL_TASKS=TOTAL_TASKS,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            USE_G=USE_G,
            IS_VARLEN=IS_VARLEN,
        )
    else:
        stage7_qkv_workspace = kv_workspace
        _run_stage7_loop_pipeline(
            q=q,
            k=k,
            v_new=v_new,
            h=h,
            g=g,
            o=o,
            cu_seqlens=cu_seqlens,
            qk_workspace=wh_workspace,
            qh_workspace=state_workspace,
            gated_qk_workspace=gated_v_workspace,
            qkv_workspace=stage7_qkv_workspace,
            scale=scale,
            T=T,
            TOTAL_TASKS=TOTAL_TASKS,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            USE_G=USE_G,
            IS_VARLEN=IS_VARLEN,
        )


DEFAULT_NUM_AIC = 24


def _use_stage7_head_pair(
    *, total_tokens: int, num_sequences: int, num_heads: int, num_kv_heads: int
) -> bool:
    if num_heads != 2 * num_kv_heads:
        return False
    if num_sequences > 1:
        return total_tokens >= 512
    return total_tokens >= 1024 or (total_tokens >= 256 and total_tokens % 64 == 0)


def chunk_h_o_fused_with_intermediates(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    B, T, Hg, K = q.shape
    H, V = u.shape[-2:]
    BT = chunk_size
    assert K == 128 and V == 128 and BT == 64
    assert H % Hg == 0
    if scale is None:
        scale = K**-0.5

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    num_tasks = N * H
    num_workers = DEFAULT_NUM_AIC
    use_head_pair = _use_stage7_head_pair(
        total_tokens=T,
        num_sequences=N,
        num_heads=H,
        num_kv_heads=Hg,
    )
    balance_varlen_n = 0
    if cu_seqlens is not None and H == 16 and num_workers == 24 and N >= 2:
        balance_varlen_n = N if N in (2, 3) else -1

    chunk_slots = B * triton.cdiv(T, BT) if cu_seqlens is None else (T - 1) // BT + N
    h = k.new_empty(1, chunk_slots, H, K, V)
    v_new = torch.empty_like(u)
    o = torch.empty_like(u)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    g_head_first = g.transpose(1, 2).contiguous() if g is not None else None

    state_workspace = k.new_empty(num_workers, 2 * K if use_head_pair else K, V)
    wh_workspace = k.new_empty(num_workers, BT, V)
    gated_v_workspace = k.new_empty(num_workers, K if use_head_pair else BT, V)
    # FP32 staging for the per-chunk K^T @ v_new update. The state recursion adds
    # this term every chunk, so rounding it to BF16 breaks the 1e-2 final-state
    # gate; FP32 costs nothing measurable (the writes hide behind Cube compute).
    kv_workspace = k.new_empty(
        num_workers, 2 * K if use_head_pair else K, V, dtype=torch.float32
    )

    def grid(meta):
        return (1, num_workers)

    _fused_stage6_7_kernel[grid](
        q=q,
        k=k,
        u=u,
        w=w,
        g=g_head_first,
        h=h,
        v_new=v_new,
        h0=initial_state,
        ht=final_state,
        o=o,
        cu_seqlens=cu_seqlens,
        state_workspace=state_workspace,
        wh_workspace=wh_workspace,
        gated_v_workspace=gated_v_workspace,
        kv_workspace=kv_workspace,
        scale=scale,
        T=T,
        TOTAL_TASKS=num_tasks,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        STAGE7_HEAD_PAIR=use_head_pair,
        STAGE6_BALANCE_VARLEN_N=balance_varlen_n,
        num_warps=4,
        num_stages=2,
        enable_sync_block_lock=True,
        multibuffer=False,
        disable_auto_inject_block_sync=True,
    )
    return o, final_state, h, v_new


def chunk_h_o_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del chunk_indices, chunk_offsets
    o, final_state, _, _ = chunk_h_o_fused_with_intermediates(
        q=q,
        k=k,
        w=w,
        u=u,
        g=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state


def chunk_gated_delta_rule_fwd_h(*args, **kwargs):
    from .stage6 import chunk_gated_delta_rule_fwd_h as stage6_fwd_h

    return stage6_fwd_h(*args, **kwargs)


_varlen_head_pair_chunk_h_o_fused = chunk_h_o_fused


__all__ = [
    "chunk_gated_delta_rule_fwd_h",
    "chunk_h_o_fused",
    "chunk_h_o_fused_with_intermediates",
]
