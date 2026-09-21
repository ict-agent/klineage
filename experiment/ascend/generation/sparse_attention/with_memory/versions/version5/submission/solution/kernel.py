"""Sparse MLA prefill (DeepSeek-V3.2 supplied indices) as Triton kernels.

Stage 1 compacts the selected KV rows into one contiguous buffer so the
following stage reads regular addresses instead of scattered ones.  Stage 2 is
a flash-attention style reduction over the selected keys with FP32 products of
BF16 operands: the online softmax rescales an FP32 accumulator, and only the
final output is cast back to BF16.
"""

import torch
import triton
import triton.language as tl

#: Fixed YaRN softmax scale from the workload's model constants.
SCALE = 0.1352337788608801
#: Stand-in for -inf; keeps a fully masked key block from producing a NaN.
NEG = -1.0e30
#: Selected rows copied per compaction program.
CB = 32
#: Heads per attention program.
HB = 32
#: Selected keys per attention step.
BN = 128

_WORKSPACE: dict = {}


@triton.jit
def _compact(kv_ptr, idx_ptr, kc_ptr, n_keys, D: tl.constexpr, BN: tl.constexpr):
    token = tl.program_id(0)
    kb = tl.program_id(1) * BN
    offs = tl.arange(0, BN)
    kidx = tl.load(idx_ptr + token * n_keys + kb + offs)
    kmask = kidx >= 0
    src = tl.where(kmask, kidx, 0)[:, None] * D
    dst = (token * n_keys + kb + offs)[:, None]
    o_d = tl.arange(0, D)
    kv = tl.load(kv_ptr + src + o_d[None, :], mask=kmask[:, None], other=0.0)
    tl.store(kc_ptr + dst * D + o_d[None, :], kv)


@triton.jit
def _attn(kc_ptr, q_ptr, idx_ptr, out_ptr, mx_ptr, lse_ptr, n_keys,
          H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
          HB: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr, NEG: tl.constexpr):
    pid = tl.program_id(0)
    n_hb = H // HB
    token = pid // n_hb
    hb = pid % n_hb
    o_h = hb * HB + tl.arange(0, HB)
    o_d = tl.arange(0, D)
    o_v = tl.arange(0, DV)
    q = tl.load(q_ptr + token * H * D + o_h[:, None] * D + o_d[None, :])
    m_i = tl.full([HB], NEG, tl.float32)
    l_i = tl.zeros([HB], tl.float32)
    acc = tl.zeros([HB, DV], tl.float32)
    base = token * n_keys
    # Valid indices only occupy the first min(n_keys, token + 1) slots.
    bound = tl.minimum(n_keys, token + 1)
    for kb in range(0, bound, BN):
        rows = (base + kb + tl.arange(0, BN))[:, None]
        kv = tl.load(kc_ptr + rows * D + o_d[None, :])
        v = tl.load(kc_ptr + rows * D + o_v[None, :])
        kmask = tl.load(idx_ptr + base + kb + tl.arange(0, BN)) >= 0
        s = tl.where(kmask[None, :], tl.dot(q, tl.trans(kv)) * SCALE, NEG)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(kmask[None, :], tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    out = (acc / l_i[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + token * H * DV + o_h[:, None] * DV + o_v[None, :], out)
    tl.store(mx_ptr + token * H + o_h, m_i)
    tl.store(lse_ptr + token * H + o_h, m_i + tl.log(l_i))


def kernel(q, kv, indices, output, max_logits, lse):
    """q [T, H, D] bf16, kv [T, 1, D] bf16, indices [T, 1, K] int32."""
    tokens, heads, width = q.shape
    n_keys = indices.shape[-1]
    value_dim = output.shape[-1]
    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous()
    key = (tokens, heads, width, value_dim, q.device, q.dtype)
    packed = _WORKSPACE.get(key)
    if packed is None:
        packed = torch.empty((tokens, n_keys, width), dtype=q.dtype, device=q.device)
        _WORKSPACE[key] = packed
    _compact[(tokens, n_keys // CB)](kv, indices, packed, n_keys, D=width, BN=CB)
    _attn[(tokens * (heads // HB),)](
        packed, q, indices, output, max_logits, lse, n_keys,
        H=heads, D=width, DV=value_dim, HB=HB, BN=BN, SCALE=SCALE, NEG=NEG,
    )
    return None
