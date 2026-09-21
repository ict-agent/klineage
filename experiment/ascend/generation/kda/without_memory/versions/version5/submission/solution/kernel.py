"""Chunked KDA prefill for Ascend 910B1 (triton-ascend).

Per token the reference recurrence is (state S is [value, key]; the gate acts
on the key axis)::

    S <- S diag(d_t)        d_t = exp(lb sigmoid(exp(a_log)(g_t + dt_bias)))
    delta <- sig(b_t) (v_t - S k_t)
    S <- S + delta k_t^T
    o_t <- S q_t

Chunk the tokens into blocks of C and let L_j = cumsum(log d) inside a block
(per key dimension).  Writing kbar_j = k_j / exp(L_j), kchk_j = k_j exp(L_j),
the intra-block recurrence becomes the triangular system

    A     = Kchk Kbar^T                      (kbar_i . kchk_t, i < t)
    B     = tril(A diag(gain), -1)
    delta = (I + B)^-1 gain (V - Kchk S^T)
    O     = Qchk S^T + triu(Qchk Kbar^T) delta
    S    <- S diag(exp(L_{C-1})) + (Kbar exp(L_{C-1}) exp(-L))^T delta

where S is the state at the block start.  Every contraction is a decay ratio
exp(L_t - L_i) <= 1, so nothing overflows, and the factors exp(-L) are only
worth keeping in a *centred* frame: the block frame is shifted by half its
worst-case log decay, which keeps both exp(+-L/2) and its reciprocal normal in
fp32 up to C = 32 (5 * C / 2 <= 88).

(I + B)^-1 is expanded as (I - B)(I + B^2)(I + B^4) ~ sum_{n<8} (-B)^n: the
gate decays by at least exp(-5) per token, so B is banded and B^n is negligible
well before n = 8 - the truncated inverse matches the exact one to fp32.

The value axis is split over NV programs (the state block would otherwise be
three 128x128 fp32 tiles, more than the UB holds).  A, B and the inverse are
recomputed per value block; they are cheap next to the state contractions.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _kda_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, scale_ptr, a_log_ptr, dt_bias_ptr,
    lb_ptr, h0_ptr, o_ptr, ht_ptr,
    T, H: tl.constexpr, D: tl.constexpr, C: tl.constexpr, NV: tl.constexpr,
):
    DV: tl.constexpr = D // NV

    pid = tl.program_id(0)
    i_v = pid % NV
    i_bh = pid // NV
    i_b = i_bh // H
    i_h = i_bh % H

    o_d = tl.arange(0, D)                 # key axis, full width
    o_v = i_v * DV + tl.arange(0, DV)     # value slice owned by this program
    o_c = tl.arange(0, C)

    scale = tl.load(scale_ptr)
    lb = tl.load(lb_ptr)
    a_log = tl.load(a_log_ptr + i_h)
    dtb = tl.load(dt_bias_ptr + i_h * D + o_d)

    # State block in [key, value] layout; h0 is stored [value, key] row major.
    p_h0 = h0_ptr + (i_b * H + i_h) * D * D
    St = tl.trans(tl.load(p_h0 + o_v[:, None] * D + o_d[None, :]))

    base = i_b * T * H * D + i_h * D
    stride_t = H * D
    rows = i_b * T + o_c
    qk = q_ptr + base + rows[:, None] * stride_t + o_d[None, :]
    kk = k_ptr + base + rows[:, None] * stride_t + o_d[None, :]
    gk = g_ptr + base + rows[:, None] * stride_t + o_d[None, :]
    vk = v_ptr + base + rows[:, None] * stride_t + o_v[None, :]
    bk = beta_ptr + i_b * T * H + i_h + o_c * H
    ok = o_ptr + base + rows[:, None] * stride_t + o_v[None, :]

    m_s = o_c[:, None] > o_c[None, :]
    m_i = o_c[:, None] >= o_c[None, :]
    eye = o_c[:, None] == o_c[None, :]
    last = (o_c == C - 1)[:, None]
    first = (o_c == 0)[:, None]

    for _ in range(0, T // C):
        K = tl.load(kk).to(tl.float32)
        Q = tl.load(qk).to(tl.float32)
        Q = Q * (scale / tl.sqrt(tl.sum(Q * Q, 1) + 1e-6))[:, None]
        K = K / tl.sqrt(tl.sum(K * K, 1) + 1e-6)[:, None]

        gain = tl.sigmoid(tl.load(bk).to(tl.float32))
        ell = lb * tl.sigmoid(tl.exp(a_log) * (tl.load(gk).to(tl.float32) + dtb[None, :]))
        Lc = tl.cumsum(ell, axis=0)                   # raw log decay
        l_first = tl.sum(tl.where(first, Lc, 0.0), 0)
        l_last = tl.sum(tl.where(last, Lc, 0.0), 0)
        # Centre the chunk frame on the geometric midpoint of the chunk so that
        # eL and 1/eL stay representable; the shift cancels from every ratio.
        shift = 0.5 * (l_first + l_last)              # geometric midpoint
        eLr = tl.exp(Lc)                              # raw exp(L_j) <= 1
        eL = tl.exp(Lc - shift[None, :])              # centred, range exp(+-5C/2)
        el_last = tl.exp(l_last)                      # raw exp(L_{C-1})
        Efac = tl.exp(l_last[None, :] - Lc)           # exp(L_{C-1} - L_j) <= 1

        Kchk = K * eL
        Kbar = K / eL
        Qchk = Q * eL
        Kr = K * eLr                                  # raw decay, <= 1
        Qr = Q * eLr

        KS = tl.dot(Kr, St)                           # state read, raw decay
        QS = tl.dot(Qr, St)
        V = tl.load(vk).to(tl.float32)
        rhs = gain[:, None] * (V - KS)

        A = tl.dot(Kchk, tl.trans(Kbar))
        B = tl.where(m_s, A * gain[:, None], 0.0)
        # (I+B)^-1 = (I-B)(I+B^2)(I+B^4) + O(B^8); the decay makes B banded.
        B2 = tl.dot(B, B)
        B4 = tl.dot(B2, B2)
        N = tl.where(eye, 1.0, 0.0) - B
        N = N + tl.dot(N, B2)
        N = N + tl.dot(N, B4)
        delta = tl.dot(N, rhs)

        St = St * el_last[:, None] + tl.dot(tl.trans(K * Efac), delta)

        Aq = tl.where(m_i, tl.dot(Qchk, tl.trans(Kbar)), 0.0)
        O = QS + tl.dot(Aq, delta)
        tl.store(ok, O.to(tl.bfloat16))

        qk += C * stride_t
        kk += C * stride_t
        vk += C * stride_t
        gk += C * stride_t
        bk += C * H
        ok += C * stride_t

    tl.store(ht_ptr + (i_b * H + i_h) * D * D + o_v[:, None] * D + o_d[None, :],
             tl.trans(St))


def kernel(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, initial_state):
    B, T, H, D = q.shape
    out = torch.empty_like(v)
    final = torch.empty_like(initial_state)
    NV = 2
    _kda_kernel[(B * H * NV,)](
        q, k, v, g, beta, scale, a_log, dt_bias, lower_bound,
        initial_state, out, final,
        T, H=H, D=D, C=32, NV=2,
    )
    return out, final
