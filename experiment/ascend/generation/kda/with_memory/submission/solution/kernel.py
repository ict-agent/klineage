"""Kimi-K3 KDA prefill (B=1, T=4096, H=96, K=V=128) on Ascend 910B1.

Chunked WY / UT-transform form of the bounded-gate delta rule, split in two
triton-ascend kernels:

* ``_kda_prepare`` -- per (head, 16-token chunk): L2-normalises q/k, builds the
  log2-domain gate cumsum and the chunk-local ``Aqk`` / ``A = (I + Akk)^-1``
  matrices plus the transformed operands ``w``, ``u``, ``qg``, ``kg``.
* ``_kda_fwdh`` -- per (head, V-block): walks the chunks in order with the fp32
  state resident in registers, writing the output and the final state.

The per-key gate factor ``2**gk`` can span hundreds of binades inside one chunk
(the Kimi-K3 bound is -5), so the intra-chunk matrices are formed from the
separable per-row factors ``2**gk`` / ``2**-gk`` only for a 16-token chunk,
where both factors stay inside the bf16 exponent range.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: log2(e): carrying the gate in the log2 domain turns decay into ``exp2``.
LOG2E = tl.constexpr(1.4426950408889634)
#: The torch reference's normalisation epsilon (inside the sqrt).
NORM_EPS = tl.constexpr(1e-6)

#: Chunk length and value-block width; both divide the problem exactly.
BT = 16
BV = 64
#: Persistent worker counts (910B1 exposes 24 cube cores).
PREPARE_WORKERS = 24
FWDH_WORKERS = 24


@triton.jit
def _kda_prepare(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, alog_ptr, dtb_ptr, scale_ptr, lb_ptr,
    w_ptr, u_ptr, qg_ptr, kg_ptr, aqk_ptr, gkl_ptr,
    H, NT,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    m_causal = o_t[:, None] >= o_t[None, :]
    m_strict = o_t[:, None] > o_t[None, :]
    m_eye = o_t[:, None] == o_t[None, :]
    i_eye = tl.where(m_eye, 1.0, 0.0)
    scale = tl.load(scale_ptr)
    lb = tl.load(lb_ptr)

    for task in range(pid, H * NT, nprog):
        ih = task // NT
        it = task - ih * NT
        rows = it * BT + o_t
        off_k = rows[:, None] * (H * K) + ih * K + o_k[None, :]
        off_v = rows[:, None] * (H * V) + ih * V + o_v[None, :]

        qf = tl.load(q_ptr + off_k).to(tl.float32)
        kf = tl.load(k_ptr + off_k).to(tl.float32)
        bg = tl.load(g_ptr + off_k).to(tl.float32)
        bv = tl.load(v_ptr + off_v).to(tl.float32)
        bbeta = tl.load(beta_ptr + rows * H + ih)
        alog = tl.load(alog_ptr + ih)
        dtb = tl.load(dtb_ptr + ih * K + o_k)

        qn = qf / tl.sqrt(tl.sum(qf * qf, axis=1) + NORM_EPS)[:, None]
        kn = kf / tl.sqrt(tl.sum(kf * kf, axis=1) + NORM_EPS)[:, None]

        gate = lb * tl.sigmoid(tl.exp(alog) * (bg + dtb[None, :]))
        gk = tl.cumsum(gate * LOG2E, axis=0)
        gkl = tl.sum(tl.where(o_t[:, None] == BT - 1, gk, 0.0), axis=0)

        fq = tl.exp2(gk)
        fk = tl.exp2(-gk)
        bs = tl.sigmoid(bbeta)

        kkf = tl.trans((kn * fk).to(tl.bfloat16))
        Aqk = tl.dot((qn * fq).to(tl.bfloat16), kkf)
        Akk = tl.dot((kn * fq).to(tl.bfloat16), kkf)
        Aqk = tl.where(m_causal, Aqk, 0.0) * scale
        L = tl.where(m_strict, Akk * bs[:, None], 0.0)

        lb_ = L.to(tl.bfloat16)
        l2 = tl.dot(lb_, lb_).to(tl.bfloat16)
        l4 = tl.dot(l2, l2).to(tl.bfloat16)
        l8 = tl.dot(l4, l4).to(tl.bfloat16)
        Ai = i_eye - L
        Ai = Ai + tl.dot(Ai.to(tl.bfloat16), l2)
        Ai = Ai + tl.dot(Ai.to(tl.bfloat16), l4)
        Ai = Ai + tl.dot(Ai.to(tl.bfloat16), l8)
        aib = Ai.to(tl.bfloat16)

        bu = tl.dot(aib, (bs[:, None] * bv).to(tl.bfloat16))
        bw = tl.dot(aib, (bs[:, None] * kn * fq).to(tl.bfloat16))
        bqg = qn * fq * scale
        bkg = kn * tl.exp2(gkl[None, :] - gk)

        tl.store(w_ptr + off_k, bw.to(tl.bfloat16))
        tl.store(u_ptr + off_v, bu.to(tl.bfloat16))
        tl.store(qg_ptr + off_k, bqg.to(tl.bfloat16))
        tl.store(kg_ptr + off_k, bkg.to(tl.bfloat16))
        blk = (it * H + ih) * (BT * BT)
        tl.store(aqk_ptr + blk + o_t[:, None] * BT + o_t[None, :], Aqk.to(tl.bfloat16))
        tl.store(gkl_ptr + (it * H + ih) * K + o_k, gkl)


@triton.jit
def _kda_fwdh(
    w_ptr, u_ptr, qg_ptr, kg_ptr, aqk_ptr, gkl_ptr, h0_ptr, o_ptr, ht_ptr,
    H, NT,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    NV: tl.constexpr = V // BV
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, BV)

    for task in range(pid, H * NV, nprog):
        ih = task // NV
        iv = task - ih * NV
        off_v0 = ih * V * K + (iv * BV + o_v)[:, None] * K + o_k[None, :]
        h = tl.trans(tl.load(h0_ptr + off_v0)).to(tl.float32)

        for it in range(0, NT):
            rows = it * BT + o_t
            off_k = rows[:, None] * (H * K) + ih * K + o_k[None, :]
            off_v = rows[:, None] * (H * V) + ih * V + iv * BV + o_v[None, :]
            blk = (it * H + ih) * (BT * BT)

            bw = tl.load(w_ptr + off_k)
            bqg = tl.load(qg_ptr + off_k)
            bkg = tl.load(kg_ptr + off_k)
            bu = tl.load(u_ptr + off_v).to(tl.float32)
            baqk = tl.load(aqk_ptr + blk + o_t[:, None] * BT + o_t[None, :])
            gkl = tl.load(gkl_ptr + (it * H + ih) * K + o_k)

            hb = h.to(tl.bfloat16)
            vn = bu - tl.dot(bw, hb)
            o = tl.dot(bqg, hb) + tl.dot(baqk, vn.to(tl.bfloat16))
            h = h * tl.exp2(gkl)[:, None] + tl.dot(tl.trans(bkg), vn.to(tl.bfloat16))
            tl.store(o_ptr + off_v, o.to(tl.bfloat16))

        tl.store(ht_ptr + off_v0, tl.trans(h))


_WORKSPACE: dict = {}


def _workspace(T, H, K, V, device):
    key = (T, H, K, V, BT, str(device))
    tensors = _WORKSPACE.get(key)
    if tensors is None:
        ntk = (T // BT) * H
        tensors = (
            torch.empty((T, H, K), dtype=torch.bfloat16, device=device),
            torch.empty((T, H, V), dtype=torch.bfloat16, device=device),
            torch.empty((T, H, K), dtype=torch.bfloat16, device=device),
            torch.empty((T, H, K), dtype=torch.bfloat16, device=device),
            torch.empty((ntk, BT, BT), dtype=torch.bfloat16, device=device),
            torch.empty((ntk, K), dtype=torch.float32, device=device),
        )
        _WORKSPACE[key] = tensors
    return tensors


def kernel(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, initial_state):
    """Entry point: returns ``(output, final_state)``."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    a_log = a_log.contiguous()
    dt_bias = dt_bias.contiguous()
    initial_state = initial_state.contiguous()

    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT

    w, u, qg, kg, aqk, gkl = _workspace(T, H, K, V, q.device)

    _kda_prepare[(PREPARE_WORKERS,)](
        q, k, v, g, beta, a_log, dt_bias, scale, lower_bound,
        w, u, qg, kg, aqk, gkl,
        H, NT,
        K=K, V=V, BT=BT,
    )
    output = torch.empty((B, T, H, V), dtype=v.dtype, device=v.device)
    final_state = torch.empty((B, H, V, K), dtype=torch.float32, device=v.device)
    _kda_fwdh[(FWDH_WORKERS,)](
        w, u, qg, kg, aqk, gkl, initial_state, output, final_state,
        H, NT,
        K=K, V=V, BT=BT, BV=BV,
    )
    return output, final_state
