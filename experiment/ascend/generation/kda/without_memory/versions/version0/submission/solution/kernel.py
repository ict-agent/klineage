"""Chunked KDA prefill for Ascend 910B1 (triton-ascend).

The reference recurrence is, per token t (state S is [value, key], the gate
gates the key axis)::

    S <- S diag(d_t)                    d_t = exp(lb sigmoid(exp(a_log)(g_t + dt_bias)))
    delta <- sig(b_t) (v_t - (S k_t))
    S <- S + delta k_t^T
    o_t <- S q_t

Chunking a block of C tokens with L_j = cumsum(log d) inside the chunk, and
writing the state in the rescaled frame

    S_hat_j = S_j diag(exp(-L_j)),

turns the recurrence into the plain delta rule

    S_hat_j = S_hat_{j-1} + delta_j kbar_j^T,   kbar_j = k_j / exp(L_j)
    delta_j = sig(b_j) (v_j - S_hat_{j-1} kchk_j),   kchk_j = k_j exp(L_j)

so the intra-chunk part is the lower-triangular solve

    (I + diag(sig(b)) tril(Kchk Kbar^T, -1)) delta = sig(b) (v - Kchk S_hat^T)

and the chunk is finished with

    o_j  = S_hat (q_j exp(L_j)) + tril_inc(Qchk Kbar^T) delta
    S   <- S diag(exp(L_{C-1})) + update,  update_t = delta_t (k_t exp(L_{C-1}-L_t))^T

`kbar` is as large as 1/exp(L) but it only ever meets `kchk`/`qchk`, so every
product is a bounded decay ratio exp(L_t - L_i) <= 1: no overflow and no
cancellation.  That caps C at 16 for the worst-case gate (-5 per token), since
exp(-L) must stay representable in fp32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _kda_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, scale_ptr, a_log_ptr, dt_bias_ptr,
    lb_ptr, h0_ptr, o_ptr, ht_ptr,
    T, H: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
):
    pid = tl.program_id(0)
    i_b = pid // H
    i_h = pid % H

    o_i = tl.arange(0, D)
    o_c = tl.arange(0, C)

    scale = tl.load(scale_ptr)
    lb = tl.load(lb_ptr)
    a_log = tl.load(a_log_ptr + i_h)
    dtb = tl.load(dt_bias_ptr + i_h * D + o_i)

    # state is stored (value, key); the register block keeps its transpose so
    # that the state applications are plain matmuls.
    p_h0 = h0_ptr + (i_b * H + i_h) * D * D
    St = tl.trans(tl.load(p_h0 + o_i[:, None] * D + o_i[None, :]))

    base = i_b * T * H * D + i_h * D
    stride_t = H * D
    rows = i_b * T + o_c
    qk = q_ptr + base + rows[:, None] * stride_t + o_i[None, :]
    kk = k_ptr + base + rows[:, None] * stride_t + o_i[None, :]
    vk = v_ptr + base + rows[:, None] * stride_t + o_i[None, :]
    gk = g_ptr + base + rows[:, None] * stride_t + o_i[None, :]
    bk = beta_ptr + i_b * T * H + i_h + o_c * H
    ok = o_ptr + base + rows[:, None] * stride_t + o_i[None, :]

    m_s = o_c[:, None] > o_c[None, :]
    m_i = o_c[:, None] >= o_c[None, :]
    eye = o_c[:, None] == o_c[None, :]
    last = (o_c == C - 1)[:, None]

    for _ in range(0, T // C):
        K = tl.load(kk).to(tl.float32)
        Q = tl.load(qk).to(tl.float32)
        Q = Q * (scale / tl.sqrt(tl.sum(Q * Q, 1) + 1e-6))[:, None]
        K = K / tl.sqrt(tl.sum(K * K, 1) + 1e-6)[:, None]

        gain = tl.sigmoid(tl.load(bk).to(tl.float32))
        ell = lb * tl.sigmoid(tl.exp(a_log) * (tl.load(gk).to(tl.float32) + dtb[None, :]))
        eL = tl.exp(tl.cumsum(ell, axis=0))          # exp(L_j), decreasing
        el_last = tl.sum(tl.where(last, eL, 0.0), 0)  # exp(L_{C-1})

        Kchk = K * eL
        Kbar = K / eL
        Qchk = Q * eL

        KS = tl.dot(Kchk, St)                         # S_hat k_t exp(L_t)
        QS = tl.dot(Qchk, St)
        V = tl.load(vk).to(tl.float32)
        rhs = gain[:, None] * (V - KS)

        A = tl.dot(Kchk, tl.trans(Kbar))              # (kbar_i . kchk_t)
        B = tl.where(m_s, A * gain[:, None], 0.0)
        # (I + B)^-1 by forward substitution, one row at a time.
        N = tl.where(eye, 1.0, 0.0)
        for j in range(1, C):
            rj = o_c == j
            brow = tl.sum(tl.where(rj[:, None], B, 0.0), 0)
            nrow = tl.where(rj, 1.0, 0.0) - tl.sum(brow[:, None] * N, 0)
            N = tl.where(rj[:, None], nrow[None, :], N)
        delta = tl.dot(N, rhs)

        St = St * el_last[:, None] + tl.dot(tl.trans(Kbar * el_last[None, :]), delta)

        Aq = tl.where(m_i, tl.dot(Qchk, tl.trans(Kbar)), 0.0)
        O = QS + tl.dot(Aq, delta)
        tl.store(ok, O.to(tl.bfloat16))

        qk += C * stride_t
        kk += C * stride_t
        vk += C * stride_t
        gk += C * stride_t
        bk += C * H
        ok += C * stride_t

    tl.store(ht_ptr + (i_b * H + i_h) * D * D + o_i[:, None] * D + o_i[None, :],
             tl.trans(St))


def kernel(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, initial_state):
    B, T, H, D = q.shape
    out = torch.empty_like(v)
    final = torch.empty_like(initial_state)
    _kda_kernel[(B * H,)](
        q, k, v, g, beta, scale, a_log, dt_bias, lower_bound,
        initial_state, out, final,
        T, H=H, D=D, C=16,
    )
    return out, final
