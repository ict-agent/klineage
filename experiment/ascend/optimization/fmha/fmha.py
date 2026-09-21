"""Packed-NHD FP16 attention with runtime shape dispatch on Ascend 910B1."""
from enum import Enum
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if os.environ.get('ASCEND_OPP_PATH') not in (str(ROOT/'opp_candidate'), str(ROOT/'opp_baseline')):
    raise RuntimeError('Use run_with_opp.sh to select an isolated OPP before Python starts.')

import torch
import torch_npu

Q_TILE = 256
KV_TILE = 512
RESIDENT_DIM = 128
MIN_KV_TILES = 3
DIM_ALIGNMENT = 16
MAX_HEAD_DIM = 256
FULL_CONTEXT = 2147483647


class PathKind(Enum):
    RESIDENT = 'resident_eligible_fia'
    GENERAL = 'general_fusion_attention'


def validate(q, k, v, cuq, cuk):
    for x in (q, k, v):
        if x.ndim != 3 or min(x.shape) <= 0:
            raise ValueError('Q/K/V must have nonempty [tokens, heads, dim] shapes.')
        if x.dtype != torch.float16 or not x.is_contiguous():
            raise ValueError('Q/K/V must be contiguous FP16 tensors.')
        if x.device != q.device or x.device.type != 'npu':
            raise ValueError('Q/K/V must be on the same NPU.')
    if k.shape != v.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError('K/V shapes and Q/K head counts and dimensions must match.')
    dim = q.shape[-1]
    if dim % DIM_ALIGNMENT or dim > MAX_HEAD_DIM:
        raise ValueError('head_dim must be a multiple of 16 in [16, 256].')
    if cuq.numel() != cuk.numel() or cuq.numel() < 2:
        raise ValueError('Q and KV must contain the same nonzero number of sequences.')
    offsets = []
    for cu, tokens in ((cuq, q.shape[0]), (cuk, k.shape[0])):
        if cu.ndim != 1 or cu.dtype != torch.int32 or not cu.is_contiguous():
            raise ValueError('Cumulative lengths must be contiguous rank-one int32 tensors.')
        if cu.device.type != 'cpu' and cu.device != q.device:
            raise ValueError('Cumulative lengths must be on CPU or the Q device.')
        values = cu.cpu().tolist()
        if values[0] != 0 or values[-1] != tokens:
            raise ValueError('Cumulative lengths must start at zero and end at the token count.')
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError('Each sequence must be nonempty and offsets strictly increasing.')
        offsets.append(values)
    return offsets


def select_path(dim, offsets):
    q_offsets, k_offsets = offsets
    lengths = [b - a for a, b in zip(q_offsets, q_offsets[1:])]
    aligned = lengths[0] >= MIN_KV_TILES * KV_TILE and lengths[0] % KV_TILE == 0
    if dim == RESIDENT_DIM and q_offsets == k_offsets and aligned and len(set(lengths)) == 1:
        return PathKind.RESIDENT
    return PathKind.GENERAL


def run(q, k, v, cu_seqlens_q, cu_seqlens_k):
    offsets = validate(q, k, v, cu_seqlens_q, cu_seqlens_k)
    ends_q, ends_k = (values[1:] for values in offsets)
    heads, dim = q.shape[1:]
    scale = dim ** -0.5
    if select_path(dim, offsets) == PathKind.RESIDENT:
        return torch_npu.npu_fused_infer_attention_score(
            q, k, v, num_heads=heads, num_key_value_heads=heads, scale=scale,
            input_layout='TND', actual_seq_lengths=ends_q, actual_seq_lengths_kv=ends_k,
            sparse_mode=0, pre_tokens=FULL_CONTEXT, next_tokens=FULL_CONTEXT)[0]
    # This is a distinct operator, not FIA under the overridden dispatch key.
    # It supports tails and ragged sequences without entering resident UB code.
    return torch_npu.npu_fusion_attention(
        q, k, v, head_num=heads, input_layout='TND', scale=scale, keep_prob=1.0,
        actual_seq_qlen=ends_q, actual_seq_kvlen=ends_k,
        pre_tockens=FULL_CONTEXT, next_tockens=FULL_CONTEXT, sparse_mode=0)[0].contiguous()
