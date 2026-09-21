"""Sparse MLA prefill (DeepSeek-V3.2 supplied indices) as Triton kernels.

Two stages: a compaction kernel copies the selected KV rows into contiguous
buffers, then a flash-attention style kernel reduces them with FP32 products.
Only `output` is cast back to BF16.
"""

import torch
import triton
import triton.language as tl

#: Fixed YaRN softmax scale from the workload's model constants.
SCALE = 0.1352337788608801
#: Stand-in for -inf; keeps fully masked key blocks from producing NaN.
NEG = -1.0e30
#: Rows per compaction program.
CB = 16
#: Heads per attention program and keys per attention step.
HB = 32
BN = 128

_WORKSPACE: dict = {}


@triton.jit
def _compact(kv_ptr, idx_ptr, kc_ptr, vc_ptr, n_keys,
             D: tl.constexpr, DV: tl.constexpr, BN: tl.constexpr):
    token = tl.program_id(0)
    kb = tl.program_id(1) * BN
    offs = tl.arange(0, BN)
    kidx = tl.load(idx_ptr + token * n_keys + kb + offs)
    kmask = kidx >= 0
    src = tl.where(kmask, kidx, 0)[:, None] * D
    dst = (token * n_keys + kb + offs)[:, None]
    o_d = tl.arange(0, D)
    o_v = tl.arange(0, DV)
    kv = tl.load(kv_ptr + src + o_d[None, :], mask=kmask[:, None], other=0.0)
    tl.store(kc_ptr + dst * D + o_d[None, :], kv)
    v = tl.load(kv_ptr + src + o_v[None, :], mask=kmask[:, None], other=0.0)
    tl.store(vc_ptr + dst * DV + o_v[None, :], v)


@triton.jit
def _attn(kc_ptr, vc_ptr, q_ptr, idx_ptr, out_ptr, mx_ptr, lse_ptr, n_keys,
          H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
          HB: tl.constexpr, BN: tl.constexpr,
          SCALE: tl.constexpr, NEG: tl.constexpr):
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
    # Valid indices occupy only the first min(n_keys, token + 1) slots.
    bound = tl.minimum(n_keys, token + 1)
    for kb in range(0, bound, BN):
        rows = (base + kb + tl.arange(0, BN))[:, None]
        kv = tl.load(kc_ptr + rows * D + o_d[None, :])
        v = tl.load(vc_ptr + rows * DV + o_v[None, :])
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
    workspace = _WORKSPACE.get(key)
    if workspace is None:
        workspace = (
            torch.empty((tokens, n_keys, width), dtype=q.dtype, device=q.device),
            torch.empty((tokens, n_keys, value_dim), dtype=q.dtype, device=q.device),
        )
        _WORKSPACE[key] = workspace
    kc, vc = workspace
    _compact[(tokens, n_keys // CB)](
        kv, indices, kc, vc, n_keys, D=width, DV=value_dim, BN=CB
    )
    _attn[(tokens * (heads // HB),)](
        kc, vc, q, indices, output, max_logits, lse, n_keys,
        H=heads, D=width, DV=value_dim, HB=HB, BN=BN, SCALE=SCALE, NEG=NEG,
    )
    return None
