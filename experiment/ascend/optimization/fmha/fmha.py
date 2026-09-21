"""Packed-NHD FMHA for 8 x 2048 tokens, 64 heads, D=128, FP16, Ascend 910B1.

Launch a fresh process with ASCEND_OPP_PATH pointing to this directory's
opp_candidate (optimized) or opp_baseline (official) BEFORE importing torch_npu.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if os.environ.get('ASCEND_OPP_PATH') not in (str(ROOT/'opp_candidate'), str(ROOT/'opp_baseline')):
    raise RuntimeError('Use run_with_opp.sh to select an isolated OPP before Python starts.')

import torch
import torch_npu

ENDS = [2048*i for i in range(1,9)]


def run(q, k, v, cu_seqlens_q, cu_seqlens_k):
    for x in (q,k,v):
        if (x.shape != (16384,64,128) or x.dtype != torch.float16
                or not x.is_contiguous() or x.device != q.device or x.device.type != 'npu'):
            raise ValueError('This specialization requires contiguous FP16 NPU [16384,64,128] Q/K/V.')
    offsets = []
    for cu in (cu_seqlens_q,cu_seqlens_k):
        if cu.shape != (9,) or cu.dtype != torch.int32 or not cu.is_contiguous() or cu.device != q.device:
            raise ValueError('Expected contiguous int32 [9] cu_seqlens on the Q device.')
        values = cu.cpu().tolist()
        if values != [0]+ENDS:
            raise ValueError('This specialization supports exactly eight sequences of 2048 tokens.')
        offsets.append(values[1:])
    return torch_npu.npu_fused_infer_attention_score(
        q,k,v,num_heads=64,num_key_value_heads=64,scale=128**-0.5,input_layout='TND',
        actual_seq_lengths=offsets[0],actual_seq_lengths_kv=offsets[1],
        sparse_mode=0,pre_tokens=2147483647,next_tokens=2147483647)[0]
