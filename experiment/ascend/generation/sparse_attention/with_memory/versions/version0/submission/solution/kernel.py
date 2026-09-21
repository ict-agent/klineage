"""Sparse MLA prefill (DeepSeek-V3.2 supplied indices) as a Triton kernel.

Scores are FP32 products of BF16 operands; only `output` is cast to BF16.
"""

import torch
import triton
import triton.language as tl

#: Fixed YaRN softmax scale from the workload's model constants.
SCALE = 0.1352337788608801
#: Stand-in for -inf so fully masked key blocks never produce a NaN.
NEG = -1.0e30


@triton.jit
def _sparse_attn(
    q_ptr,
    kv_ptr,
    idx_ptr,
    out_ptr,
    mx_ptr,
    lse_ptr,
    n_tokens,
    n_keys,
    H: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    HB: tl.constexpr,
    BN: tl.constexpr,
    SCALE: tl.constexpr,
    NEG: tl.constexpr,
):
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

    idx_base = idx_ptr + token * n_keys
    for kb in range(0, n_keys, BN):
        kidx = tl.load(idx_base + kb + tl.arange(0, BN))
        kmask = (kidx >= 0) & (kidx < n_tokens)
        row = tl.where(kmask, kidx, 0)[:, None] * D
        k = tl.load(kv_ptr + row + o_d[None, :], mask=kmask[:, None], other=0.0)
        v = tl.load(kv_ptr + row + o_v[None, :], mask=kmask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * SCALE
        s = tl.where(kmask[None, :], s, NEG)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.where(kmask[None, :], tl.exp(s - m_new[:, None]), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    out = (acc / l_i[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + token * H * DV + o_h[:, None] * DV + o_v[None, :], out)
    tl.store(mx_ptr + token * H + o_h, m_i)
    tl.store(lse_ptr + token * H + o_h, m_i + tl.log(l_i))


HB = 16
BN = 32


def kernel(q, kv, indices, output, max_logits, lse):
    """q [T, H, D] bf16, kv [T, 1, D] bf16, indices [T, 1, K] int32."""
    tokens, heads, width = q.shape
    n_keys = indices.shape[-1]
    grid = (tokens * (heads // HB),)
    _sparse_attn[grid](
        q,
        kv,
        indices,
        output,
        max_logits,
        lse,
        tokens,
        n_keys,
        H=heads,
        D=width,
        DV=output.shape[-1],
        HB=HB,
        BN=BN,
        SCALE=SCALE,
        NEG=NEG,
    )
    return None
