"""kda: Kimi-K3 bounded-gate KDA prefill recurrence in triton-ascend.

Independent per-head FP32 recurrence in [value, key] layout:

    state = state * decay[t]                    # decay over the key axis
    pred  = sum_k state[v, k] * key[t, k]
    delta = gain[t] * (value[t] - pred)
    state = state + delta[:, None] * key[t]
    out[t] = sum_k state[v, k] * query[t, k]

``decay`` and the normalisation of ``q``/``k`` are recomputed in-kernel; the
timed path launches exactly one triton program per (head, value block).
"""

import torch
import triton
import triton.language as tl

_EPS = tl.constexpr(1e-6)
_VP = 2


@triton.jit
def _kda_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, scale_ptr, a_log_ptr, dt_bias_ptr,
    lower_ptr, state_ptr, out_ptr, final_ptr,
    T,
    H: tl.constexpr, D: tl.constexpr, VP: tl.constexpr,
):
    BV: tl.constexpr = D // VP
    pid = tl.program_id(0)
    b = pid // (H * VP)
    rest = pid % (H * VP)
    h = rest // VP
    vb = rest % VP

    rk = tl.arange(0, D)
    rv = vb * BV + tl.arange(0, BV)

    gmul = tl.exp(tl.load(a_log_ptr + h))
    lb = tl.load(lower_ptr)
    sc = tl.load(scale_ptr)
    bias = tl.load(dt_bias_ptr + h * D + rk)

    s_ptrs = state_ptr + (b * H + h) * D * D + rv[:, None] * D + rk[None, :]
    S = tl.load(s_ptrs)

    for t in range(0, T):
        base = ((b * T + t) * H + h) * D
        kf = tl.load(k_ptr + base + rk).to(tl.float32)
        kf = kf / tl.sqrt(tl.sum(kf * kf) + _EPS)
        qf = tl.load(q_ptr + base + rk).to(tl.float32)
        qf = qf / tl.sqrt(tl.sum(qf * qf) + _EPS) * sc
        gf = tl.load(g_ptr + base + rk).to(tl.float32)
        decay = tl.exp(lb * tl.sigmoid(gmul * (gf + bias)))
        gain = tl.sigmoid(tl.load(beta_ptr + t * H + h))
        vv = tl.load(v_ptr + base + rv).to(tl.float32)

        S = S * decay[None, :]
        delta = gain * (vv - tl.sum(S * kf[None, :], axis=1))
        S = S + delta[:, None] * kf[None, :]
        out = tl.sum(S * qf[None, :], axis=1)
        tl.store(out_ptr + base + rv, out.to(tl.bfloat16))

    f_ptrs = final_ptr + (b * H + h) * D * D + rv[:, None] * D + rk[None, :]
    tl.store(f_ptrs, S)


def kernel(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, initial_state):
    batch, tokens, heads, dim = q.shape
    out = torch.empty_like(v)
    final = torch.empty_like(initial_state)
    grid = (batch * heads * _VP,)
    _kda_kernel[grid](
        q, k, v, g, beta, scale, a_log, dt_bias, lower_bound,
        initial_state, out, final, tokens,
        H=heads, D=dim, VP=_VP,
    )
    return out, final

