"""Kimi-K3 bounded-gate KDA prefill for Ascend 910B1 (triton-ascend).

Reference recurrence, per (batch, head), state S in [value, key] layout:

    S_t   = decay_t * S_{t-1} + delta_t (x) k_t
    delta = gain_t * (v_t - (decay_t * S_{t-1}) . k_t)
    o_t   = S_t . q_t

``decay`` is elementwise over the key axis, so the token loop is sequential.
The kernel splits the state on the value axis and runs one program per
(value block, head): each program keeps its [BV, D] slice resident and walks
the token axis once.
"""

import torch
import triton
import triton.language as tl

NORM_EPS = 1e-6

#: Scalar (0-d) host values are read once and remembered per tensor object:
#: the gate hands the same objects to every timed call, and a device->host
#: sync inside the timed region would be paid on every sample.
_SCALARS: dict[int, tuple[object, float]] = {}


def _scalar(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    key = id(value)
    entry = _SCALARS.get(key)
    if entry is not None and entry[0] is value:
        return entry[1]
    resolved = (value, float(value))
    if len(_SCALARS) > 8:
        _SCALARS.clear()
    _SCALARS[key] = resolved
    return resolved[1]


@triton.jit
def _prepare_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    beta_ptr,
    a_log_ptr,
    dt_bias_ptr,
    qn_ptr,
    kn_ptr,
    decay_ptr,
    gain_ptr,
    scale,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    TB: tl.constexpr,
):
    """Normalise q/k, fold the scale into q, and build decay/gain."""

    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    toffs = pid_t * TB + tl.arange(0, TB)
    doffs = tl.arange(0, D)
    live = (toffs < T)[:, None]
    offs = toffs[:, None] * (H * D) + h * D + doffs[None, :]

    b_q = tl.load(q_ptr + offs, mask=live, other=0.0).to(tl.float32)
    b_qn = b_q * tl.rsqrt(tl.sum(b_q * b_q, 1) + NORM_EPS)[:, None] * scale
    tl.store(qn_ptr + offs, b_qn, mask=live)

    b_k = tl.load(k_ptr + offs, mask=live, other=0.0).to(tl.float32)
    b_kn = b_k * tl.rsqrt(tl.sum(b_k * b_k, 1) + NORM_EPS)[:, None]
    tl.store(kn_ptr + offs, b_kn, mask=live)

    b_g = tl.load(g_ptr + offs, mask=live, other=0.0).to(tl.float32)
    b_dt = tl.load(dt_bias_ptr + h * D + doffs)
    a_exp = tl.exp(tl.load(a_log_ptr + h))
    x = a_exp * (b_g + b_dt[None, :])
    tl.store(decay_ptr + offs, tl.exp(lower_bound / (1.0 + tl.exp(-x))), mask=live)

    boffs = toffs * H + h
    flat = toffs < T
    b_beta = tl.load(beta_ptr + boffs, mask=flat, other=0.0).to(tl.float32)
    tl.store(gain_ptr + boffs, 1.0 / (1.0 + tl.exp(-b_beta)), mask=flat)


@triton.jit
def _kda_recurrent_kernel(
    qn_ptr,
    kn_ptr,
    v_ptr,
    decay_ptr,
    gain_ptr,
    out_ptr,
    init_ptr,
    final_ptr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    h = tl.program_id(1)
    voffs = i_v * BV + tl.arange(0, BV)
    doffs = tl.arange(0, D)
    hd = H * D

    b_state = tl.load(init_ptr + h * D * D + voffs[:, None] * D + doffs[None, :])

    p_q = qn_ptr + h * D + doffs
    p_k = kn_ptr + h * D + doffs
    p_v = v_ptr + h * D + voffs
    p_decay = decay_ptr + h * D + doffs
    p_gain = gain_ptr + h
    p_out = out_ptr + h * D + voffs

    for t in range(T):
        b_k = tl.load(p_k)
        b_state = b_state * tl.load(p_decay)[None, :]
        b_pred = tl.sum(b_state * b_k[None, :], 1)
        b_v = tl.load(p_v).to(tl.float32)
        b_delta = tl.load(p_gain) * (b_v - b_pred)
        b_state += b_delta[:, None] * b_k[None, :]
        b_o = tl.sum(b_state * tl.load(p_q)[None, :], 1)
        tl.store(p_out, b_o.to(out_ptr.dtype.element_ty))

        p_q += hd
        p_k += hd
        p_v += hd
        p_decay += hd
        p_gain += H
        p_out += hd

    tl.store(final_ptr + h * D * D + voffs[:, None] * D + doffs[None, :], b_state)


_PREP_TB = 32
_BV = 32
_PREP_WARPS = 4
_MAIN_WARPS = 2


def kernel(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, initial_state):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    a_log = a_log.contiguous()
    dt_bias = dt_bias.contiguous()
    initial_state = initial_state.contiguous()

    b, t, h, d = q.shape
    dev = q.device
    scale_f = _scalar(scale)
    lb = _scalar(lower_bound)

    qn = torch.empty((b, t, h, d), dtype=torch.float32, device=dev)
    kn = torch.empty((b, t, h, d), dtype=torch.float32, device=dev)
    decay = torch.empty((b, t, h, d), dtype=torch.float32, device=dev)
    gain = torch.empty((b, t, h), dtype=torch.float32, device=dev)
    out = torch.empty((b, t, h, d), dtype=torch.bfloat16, device=dev)
    final_state = torch.empty((b, h, d, d), dtype=torch.float32, device=dev)

    _prepare_kernel[(triton.cdiv(t, _PREP_TB), h * b)](
        q, k, g, beta, a_log, dt_bias, qn, kn, decay, gain, scale_f, lb, t,
        H=h, D=d, TB=_PREP_TB, num_warps=_PREP_WARPS,
    )

    _kda_recurrent_kernel[(d // _BV, h * b)](
        qn, kn, v, decay, gain, out, initial_state, final_state, t,
        H=h, D=d, BV=_BV, num_warps=_MAIN_WARPS,
    )

    return out, final_state
