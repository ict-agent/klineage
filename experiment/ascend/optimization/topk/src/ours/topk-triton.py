"""
TopK on Ascend NPU (Triton) - standalone final path
===================================================

This file keeps a generic fallback path and layers the current optimized
dispatch on top:

  - generic final: use round-robin sort plus generic merge+unpack for verified
    tiny-S, small-N 2048, direct final, and high-K medium-S cases.
  - round-robin sort path: sort proposal runs with length min(K, SEG_LEN).
  - tiny-S small-batch path: restore the older 2048-run + merge4 path for
    B<=4, S<=2, K<=512.
  - stream16 small-K merge path: route K <= 2048 through one cascade merge path.
  - batch-local path: merge per-row runs into out_segs, optionally group
    them further, then finish with merge+unpack.

Data format: proposal = [value(f32), index(i32 as f32)] = 8B.
"""

import argparse
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import triton
import triton.language as tl
import torch_npu
import triton.language.extra.cann.extension as al
from triton.backends.ascend.testing import do_bench_npu

DEVICE = "npu"
DEFAULT_NPU_DEVICE = 7

NUM_CORES = 48
CHUNK = 2048          # UB 每路一次最多 proposal 数
# 中间 per-core/group merge 的 chunk。1152 在 batch=48,K=8192 的
# sweep 中比 1528 更稳，且仍低于当前 UB 安全上限。
CORE_LOCAL_CHUNK = 1152
MAX_WAYS = 4
DIRECT_GROUP_MIN_SEGS = 16
DIRECT_GROUP_MAX_SEGS = 128
SMALLK_SORT_MAX_K = 512
LAYERED_SORT_MAX_K = 2048
TINY_S_SMALL_BATCH_MAX_K = 512
STREAM16_SMALLK_MERGE_MAX_K = 2048
STREAM16_SMALLK_GROUP_RUNS = 16
SMALLN_2048_MIN_N = 2049
SMALLN_2048_MAX_N = 8192
BC_DIR = os.getenv("TOPK_BC_DIR", "/home/topk/topk_import_check/custom-topk")
SMALLK_BC_DIR = os.getenv("TOPK_SMALLK_BC_DIR")
if SMALLK_BC_DIR is None:
    _synced_smallk_dir = "/home/topk/topk/custom-topk"
    SMALLK_BC_DIR = (
        _synced_smallk_dir
        if os.path.exists(os.path.join(_synced_smallk_dir, "sort_topk_smallk.bc"))
        else BC_DIR
    )
LAYERED_BC_DIR = os.getenv("TOPK_LAYERED_BC_DIR")
if LAYERED_BC_DIR is None:
    _synced_layered_dir = "/home/topk/topk/custom-topk"
    LAYERED_BC_DIR = (
        _synced_layered_dir
        if os.path.exists(os.path.join(_synced_layered_dir, "sort_topk_layered.bc"))
        else BC_DIR
    )
def bench_total_us(fn, warmup=2, active=5):
    """Profile a callable and sum all kernel rows belonging to one invocation."""
    prof_dir = tempfile.mkdtemp(prefix="topk_profile_")
    try:
        do_bench_npu(fn, warmup=warmup, active=active, prof_dir=prof_dir, keep_res=True)
        kernel_details = None
        for root, _, files in os.walk(prof_dir):
            if "kernel_details.csv" in files:
                kernel_details = os.path.join(root, "kernel_details.csv")
                break
        if kernel_details is None:
            raise RuntimeError(f"kernel_details.csv not found under {prof_dir}")
        df = pd.read_csv(kernel_details)
        total_calls = warmup + active
        if len(df) % total_calls != 0:
            raise RuntimeError(f"kernel row count {len(df)} is not divisible by calls {total_calls}")
        rows_per_call = len(df) // total_calls
        sums = []
        breakdown = {}
        name_col = "Name"
        if name_col not in df.columns:
            for candidate in ("Kernel Name", "Op Name"):
                if candidate in df.columns:
                    name_col = candidate
                    break
        for call_idx in range(warmup, total_calls):
            rows = df.iloc[call_idx * rows_per_call:(call_idx + 1) * rows_per_call]
            sums.append(float(rows["Duration(us)"].sum()))
            if name_col in rows.columns:
                for name, group in rows.groupby(name_col):
                    breakdown[str(name)] = breakdown.get(str(name), 0.0) + float(group["Duration(us)"].sum())
        for name in list(breakdown):
            breakdown[name] /= active
        return sum(sums) / len(sums), rows_per_call, breakdown
    finally:
        shutil.rmtree(prof_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
# custom op 注册
# ════════════════════════════════════════════════════════════════════════════
@al.register_custom_op
class sort_1d_topk_proposals:
    core = al.CORE.VECTOR; pipe = al.PIPE.PIPE_V; mode = al.MODE.SIMD
    def __init__(self, src, tmp_buf, descending, TOPK, index_offset, out=None):
        assert out
        self.symbol = "custom_sort_1d_topk_proposals_float"
        self.bitcode = f"{BC_DIR}/sort_topk.bc"
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class sort_1d_topk_proposals_4x1024:
    core = al.CORE.VECTOR; pipe = al.PIPE.PIPE_V; mode = al.MODE.SIMD
    def __init__(self, src, tmp_buf, descending, TOPK, index_offset, out=None):
        assert out
        self.symbol = "custom_sort_1d_topk_proposals_4x1024_float"
        self.bitcode = f"{SMALLK_BC_DIR}/sort_topk_smallk.bc"
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class sort_1d_topk_proposals_layered_4096:
    core = al.CORE.VECTOR; pipe = al.PIPE.PIPE_V; mode = al.MODE.SIMD
    def __init__(self, src, tmp_buf, descending, TOPK, index_offset, out=None):
        assert out
        self.symbol = "custom_sort_1d_topk_proposals_layered_4096_float"
        self.bitcode = f"{LAYERED_BC_DIR}/sort_topk_layered.bc"
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class vmrgsort4_exhaust_step:
    core = al.CORE.VECTOR; pipe = al.PIPE.PIPE_V; mode = al.MODE.SIMD
    def __init__(self, src_proposals, ways, off0, off1, off2, off3,
                 len0, len1, len2, len3, dst_proposals, consumed_out, out=None):
        assert out
        self.symbol = "custom_vmrgsort4_exhaust_step_float"
        self.bitcode = f"{BC_DIR}/merge-sort-exhaust-1.bc"
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class unpack_topk_float:
    core = al.CORE.VECTOR; pipe = al.PIPE.PIPE_V; mode = al.MODE.SIMD
    def __init__(self, src_proposals, topk, dst_value, dst_index, out=None):
        assert out
        self.symbol = "custom_unpack_topk_float"
        self.bitcode = f"{BC_DIR}/merge-sort-exhaust-1.bc"
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class gm_to_ub_copy_float:
    core = al.CORE.VECTOR; pipe = al.PIPE.PIPE_MTE2; mode = al.MODE.SIMD
    def __init__(self, src, dst_offset, out=None):
        assert out
        self.symbol = "custom_gm_to_ub_copy_float"
        self.bitcode = f"{BC_DIR}/gm-to-ub.bc"
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class gm_to_ub_copy_n_float:
    """Size-aware GM→UB copy: only copies `copy_len` f32 (runtime scalar),
    not the full block_shape. Prevents over-reading when remaining < CHUNK."""
    core = al.CORE.VECTOR; pipe = al.PIPE.PIPE_MTE2; mode = al.MODE.SIMD
    def __init__(self, src, dst_offset, copy_len, out=None):
        assert out
        self.symbol = "custom_gm_to_ub_copy_n_float"
        self.bitcode = f"{BC_DIR}/gm-to-ub-n.bc"
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class ub_to_gm_copy_float:
    core = al.CORE.VECTOR; pipe = al.PIPE.PIPE_MTE3; mode = al.MODE.SIMD
    def __init__(self, src, src_offset, copy_len, dst, out=None):
        assert out
        self.symbol = "custom_ub_to_gm_copy_float"
        self.bitcode = f"{BC_DIR}/ub-to-gm.bc"
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class merge4_static_gm_to_output_float:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, run_len, topk, num_runs, dst_value, dst_index, out=None):
        assert out
        self.symbol = "custom_merge4_static_gm_to_output_float"
        self.bitcode = f"{BC_DIR}/merge4-static-gm-to-output.bc"
        self.extra_buffers = [(tl.float16, 0)]


# ════════════════════════════════════════════════════════════════════════════
# Kernel 2-fused:通用归并树 + unpack。
#   用于替换尾部 merge and unpack 两次 launch。
#   归并仍复用 WorkGM ping-pong 区;完成后在同一 kernel 内读取最终区并
#   分块 unpack 到 Yv/Yi。
# ════════════════════════════════════════════════════════════════════════════
@triton.jit
def generic_merge_unpack_kernel(WorkGM, Yv, Yi,
                                NUM_SEG: tl.constexpr, SEG_LEN: tl.constexpr,
                                ROW_WORDS: tl.constexpr,
                                K: tl.constexpr, CHUNK_C: tl.constexpr,
                                IN_CAP: tl.constexpr, OUT_CAP: tl.constexpr,
                                MAX_STREAM_ITERS: tl.constexpr,
                                MAX_WAYS_C: tl.constexpr,
                                UNPACK_CHUNK: tl.constexpr,
                                NUM_CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    base_p0 = (row * 2 + 0) * region_props
    base_p1 = (row * 2 + 1) * region_props

    in_buf = tl.zeros([IN_CAP], dtype=tl.float32)
    out_buf = tl.zeros([OUT_CAP + 8], dtype=tl.float32)
    cons = tl.zeros([4], dtype=tl.int32)

    cur_n = NUM_SEG
    cur_full = SEG_LEN
    cur_last = SEG_LEN
    src_phase = 0

    while cur_n > 1:
        stride = cur_full
        out_len = cur_full * MAX_WAYS_C
        if out_len > K:
            out_len = K
        out_n = (cur_n + MAX_WAYS_C - 1) // MAX_WAYS_C
        last_ways = cur_n - MAX_WAYS_C * (out_n - 1)
        last_total = (last_ways - 1) * cur_full + cur_last
        nxt_full = out_len
        nxt_last = last_total
        if nxt_last > K:
            nxt_last = K

        src_base = base_p0
        if src_phase == 1:
            src_base = base_p1
        dst_base = base_p1
        if src_phase == 1:
            dst_base = base_p0

        for g in tl.range(0, out_n):
            base_seg = g * MAX_WAYS_C
            ways = MAX_WAYS_C
            if base_seg + ways > cur_n:
                ways = cur_n - base_seg
            is_last_group = (g == out_n - 1)
            g0 = (base_seg + 0) * stride
            g1 = (base_seg + 1) * stride
            g2 = (base_seg + 2) * stride
            g3 = (base_seg + 3) * stride
            sl0 = cur_full
            sl1 = cur_full if ways > 1 else 0
            sl2 = cur_full if ways > 2 else 0
            sl3 = cur_full if ways > 3 else 0
            if is_last_group:
                if ways == 1:
                    sl0 = cur_last
                elif ways == 2:
                    sl1 = cur_last
                elif ways == 3:
                    sl2 = cur_last
                else:
                    sl3 = cur_last
            total = sl0 + sl1 + sl2 + sl3
            grp_cap = total
            if grp_cap > K:
                grp_cap = K
            dst_seg_off = g * out_len

            c0 = 0; c1 = 0; c2 = 0; c3 = 0
            produced = 0
            it = 0
            while (produced < grp_cap) and (it < MAX_STREAM_ITERS):
                it += 1
                r0 = sl0 - c0
                r1 = sl1 - c1 if ways > 1 else 0
                r2 = sl2 - c2 if ways > 2 else 0
                r3 = sl3 - c3 if ways > 3 else 0
                l0 = r0 if r0 < CHUNK_C else CHUNK_C
                l1 = r1 if r1 < CHUNK_C else CHUNK_C
                l2 = r2 if r2 < CHUNK_C else CHUNK_C
                l3 = r3 if r3 < CHUNK_C else CHUNK_C
                rem_ways = 0
                if r0 > 0: rem_ways += 1
                if r1 > 0: rem_ways += 1
                if r2 > 0: rem_ways += 1
                if r3 > 0: rem_ways += 1
                aw = 0
                if l0 > 0: aw += 1
                if l1 > 0: aw += 1
                if l2 > 0: aw += 1
                if l3 > 0: aw += 1

                if rem_ways == 0:
                    produced = grp_cap
                elif rem_ways == 1:
                    soff = src_base + g0 + c0
                    sres = r0
                    if r1 > 0:
                        soff = src_base + g1 + c1; sres = r1
                    if r2 > 0:
                        soff = src_base + g2 + c2; sres = r2
                    if r3 > 0:
                        soff = src_base + g3 + c3; sres = r3
                    take = sres
                    if produced + take > grp_cap:
                        take = grp_cap - produced
                    cpos = 0
                    while cpos < take:
                        clen = take - cpos
                        if clen > CHUNK_C:
                            clen = CHUNK_C
                        cbp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((soff + cpos) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", cbp, 0, clen * 2, out=in_buf)
                        cobp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                                 offsets=((dst_base + dst_seg_off + produced + cpos) * 2,),
                                                 block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("ub_to_gm_copy_float", in_buf, 0, clen * 2, cobp,
                                           out=in_buf)
                        cpos += clen
                    produced += take
                else:
                    if l0 > 0:
                        bp0 = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((src_base + g0 + c0) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", bp0, 0 * CHUNK_C * 2, l0 * 2, out=in_buf)
                    if l1 > 0:
                        bp1 = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((src_base + g1 + c1) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", bp1, 1 * CHUNK_C * 2, l1 * 2, out=in_buf)
                    if l2 > 0:
                        bp2 = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((src_base + g2 + c2) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", bp2, 2 * CHUNK_C * 2, l2 * 2, out=in_buf)
                    if l3 > 0:
                        bp3 = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((src_base + g3 + c3) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", bp3, 3 * CHUNK_C * 2, l3 * 2, out=in_buf)

                    out_buf, cons = al.custom(
                        "vmrgsort4_exhaust_step",
                        in_buf, aw,
                        0 * CHUNK_C, 1 * CHUNK_C, 2 * CHUNK_C, 3 * CHUNK_C,
                        l0, l1, l2, l3,
                        out_buf, cons, out=[out_buf, cons])

                    e0 = al.get_element(cons, (0,))
                    e1 = al.get_element(cons, (1,))
                    e2 = al.get_element(cons, (2,))
                    e3 = al.get_element(cons, (3,))
                    batch = e0 + e1 + e2 + e3

                    take = batch
                    if produced + take > grp_cap:
                        take = grp_cap - produced
                    obp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                            offsets=((dst_base + dst_seg_off + produced) * 2,),
                                            block_shape=(OUT_CAP,), order=(0,))
                    out_buf = al.custom("ub_to_gm_copy_float", out_buf, 0, take * 2, obp,
                                        out=out_buf)
                    c0 += e0; c1 += e1; c2 += e2; c3 += e3
                    produced += batch
                    if batch == 0:
                        produced = grp_cap

        cur_n = out_n
        cur_full = nxt_full
        cur_last = nxt_last
        if src_phase == 0:
            src_phase = 1
        else:
            src_phase = 0

    res_base = base_p0
    if src_phase == 1:
        res_base = base_p1

    dval = tl.zeros([UNPACK_CHUNK], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CHUNK], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CHUNK)
    for c in tl.range(0, NUM_CHUNKS):
        cstart = c * UNPACK_CHUNK
        clen = K - cstart
        if clen > UNPACK_CHUNK:
            clen = UNPACK_CHUNK
        rbp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                offsets=((res_base + cstart) * 2,),
                                block_shape=(UNPACK_CHUNK * 2,), order=(0,))
        in_buf = al.custom("gm_to_ub_copy_n_float", rbp, 0, clen * 2, out=in_buf)
        dval, didx = al.custom("unpack_topk_float", in_buf, clen, dval, didx,
                               out=[dval, didx])
        tl.store(Yv + row * K + cstart + offs, dval, mask=offs < clen)
        tl.store(Yi + row * K + cstart + offs, didx, mask=offs < clen)


@triton.jit
def smallk_stream_merge_unpack_kernel(WorkGM, Yv, Yi,
                                      ROW_WORDS: tl.constexpr,
                                      RUN_LEN: tl.constexpr,
                                      NUM_RUNS_C: tl.constexpr,
                                      K: tl.constexpr,
                                      STREAM_CAP: tl.constexpr,
                                      UNPACK_CAP: tl.constexpr):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    src_base = (row * 2 + 0) * region_props

    buf_a = tl.zeros([STREAM_CAP], dtype=tl.float32)
    buf_b = tl.zeros([STREAM_CAP], dtype=tl.float32)
    cons = tl.zeros([4], dtype=tl.int32)

    first_runs = NUM_RUNS_C
    if first_runs > 4:
        first_runs = 4

    first_bp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                 offsets=(src_base * 2,),
                                 block_shape=(4 * RUN_LEN * 2,),
                                 order=(0,))
    buf_a = al.custom("gm_to_ub_copy_n_float", first_bp, 0,
                      first_runs * RUN_LEN * 2, out=buf_a)

    phase = 0
    if first_runs > 1:
        l0 = RUN_LEN
        l1 = RUN_LEN if first_runs > 1 else 0
        l2 = RUN_LEN if first_runs > 2 else 0
        l3 = RUN_LEN if first_runs > 3 else 0
        buf_b, cons = al.custom(
            "vmrgsort4_exhaust_step",
            buf_a, 4,
            0, RUN_LEN, 2 * RUN_LEN, 3 * RUN_LEN,
            l0, l1, l2, l3,
            buf_b, cons, out=[buf_b, cons])
        phase = 1

    run = first_runs
    while run < NUM_RUNS_C:
        next_runs = NUM_RUNS_C - run
        if next_runs > 3:
            next_runs = 3

        if phase == 0:
            next_bp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((src_base + run * RUN_LEN) * 2,),
                                        block_shape=(3 * RUN_LEN * 2,),
                                        order=(0,))
            buf_a = al.custom("gm_to_ub_copy_n_float", next_bp, RUN_LEN * 2,
                              next_runs * RUN_LEN * 2, out=buf_a)
            l0 = RUN_LEN
            l1 = RUN_LEN if next_runs > 0 else 0
            l2 = RUN_LEN if next_runs > 1 else 0
            l3 = RUN_LEN if next_runs > 2 else 0
            buf_b, cons = al.custom(
                "vmrgsort4_exhaust_step",
                buf_a, 4,
                0, RUN_LEN, 2 * RUN_LEN, 3 * RUN_LEN,
                l0, l1, l2, l3,
                buf_b, cons, out=[buf_b, cons])
            phase = 1
        else:
            next_bp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((src_base + run * RUN_LEN) * 2,),
                                        block_shape=(3 * RUN_LEN * 2,),
                                        order=(0,))
            buf_b = al.custom("gm_to_ub_copy_n_float", next_bp, RUN_LEN * 2,
                              next_runs * RUN_LEN * 2, out=buf_b)
            l0 = RUN_LEN
            l1 = RUN_LEN if next_runs > 0 else 0
            l2 = RUN_LEN if next_runs > 1 else 0
            l3 = RUN_LEN if next_runs > 2 else 0
            buf_a, cons = al.custom(
                "vmrgsort4_exhaust_step",
                buf_b, 4,
                0, RUN_LEN, 2 * RUN_LEN, 3 * RUN_LEN,
                l0, l1, l2, l3,
                buf_a, cons, out=[buf_a, cons])
            phase = 0
        run += next_runs

    dval = tl.zeros([UNPACK_CAP], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CAP], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CAP)
    if phase == 0:
        dval, didx = al.custom("unpack_topk_float", buf_a, K, dval, didx,
                               out=[dval, didx])
    else:
        dval, didx = al.custom("unpack_topk_float", buf_b, K, dval, didx,
                               out=[dval, didx])
    tl.store(Yv + row * K + offs, dval, mask=offs < K)
    tl.store(Yi + row * K + offs, didx, mask=offs < K)


@triton.jit
def smallk_stream_merge4_unpack_kernel(WorkGM, Yv, Yi,
                                       ROW_WORDS: tl.constexpr,
                                       RUN_LEN: tl.constexpr,
                                       NUM_RUNS_C: tl.constexpr,
                                       K: tl.constexpr,
                                       STREAM_CAP: tl.constexpr,
                                       UNPACK_CAP: tl.constexpr):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    src_base = (row * 2 + 0) * region_props

    in_buf = tl.zeros([STREAM_CAP], dtype=tl.float32)
    out_buf = tl.zeros([STREAM_CAP], dtype=tl.float32)
    cons = tl.zeros([4], dtype=tl.int32)

    src_bp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                               offsets=(src_base * 2,),
                               block_shape=(4 * RUN_LEN * 2,),
                               order=(0,))
    in_buf = al.custom("gm_to_ub_copy_n_float", src_bp, 0,
                       NUM_RUNS_C * RUN_LEN * 2, out=in_buf)

    l0 = RUN_LEN
    l1 = RUN_LEN if NUM_RUNS_C > 1 else 0
    l2 = RUN_LEN if NUM_RUNS_C > 2 else 0
    l3 = RUN_LEN if NUM_RUNS_C > 3 else 0
    out_buf, cons = al.custom(
        "vmrgsort4_exhaust_step",
        in_buf, 4,
        0, RUN_LEN, 2 * RUN_LEN, 3 * RUN_LEN,
        l0, l1, l2, l3,
        out_buf, cons, out=[out_buf, cons])

    dval = tl.zeros([UNPACK_CAP], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CAP], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CAP)
    dval, didx = al.custom("unpack_topk_float", out_buf, K, dval, didx,
                           out=[dval, didx])
    tl.store(Yv + row * K + offs, dval, mask=offs < K)
    tl.store(Yi + row * K + offs, didx, mask=offs < K)


@triton.jit
def smallk_core_local_stream_merge_kernel(SortGM, LocalGM,
                                          NUM_SEG_PER_ROW: tl.constexpr,
                                          SORT_ROW_WORDS: tl.constexpr,
                                          LOCAL_ROW_WORDS: tl.constexpr,
                                          RUN_LEN: tl.constexpr,
                                          K: tl.constexpr,
                                          NUM_CORES_C: tl.constexpr,
                                          STREAM_CAP: tl.constexpr):
    row = tl.program_id(0)
    cid = tl.program_id(1)

    sort_region_props = SORT_ROW_WORDS // 2
    local_region_props = LOCAL_ROW_WORDS // 2
    sort_base = (row * 2 + 0) * sort_region_props
    local_base = (row * 2 + 0) * local_region_props + cid * K

    first_seg = (cid * NUM_SEG_PER_ROW) // NUM_CORES_C
    end_seg = ((cid + 1) * NUM_SEG_PER_ROW) // NUM_CORES_C
    local_n = end_seg - first_seg
    src_base = sort_base + first_seg * RUN_LEN

    buf_a = tl.zeros([STREAM_CAP], dtype=tl.float32)
    buf_b = tl.zeros([STREAM_CAP], dtype=tl.float32)
    cons = tl.zeros([4], dtype=tl.int32)

    first_runs = local_n
    if first_runs > 4:
        first_runs = 4

    first_bp = tl.make_block_ptr(base=SortGM, shape=(1 << 30,), strides=(1,),
                                 offsets=(src_base * 2,),
                                 block_shape=(4 * RUN_LEN * 2,),
                                 order=(0,))
    buf_a = al.custom("gm_to_ub_copy_n_float", first_bp, 0,
                      first_runs * RUN_LEN * 2, out=buf_a)

    phase = 0
    if first_runs > 1:
        l0 = RUN_LEN
        l1 = RUN_LEN if first_runs > 1 else 0
        l2 = RUN_LEN if first_runs > 2 else 0
        l3 = RUN_LEN if first_runs > 3 else 0
        buf_b, cons = al.custom(
            "vmrgsort4_exhaust_step",
            buf_a, 4,
            0, RUN_LEN, 2 * RUN_LEN, 3 * RUN_LEN,
            l0, l1, l2, l3,
            buf_b, cons, out=[buf_b, cons])
        phase = 1

    run = first_runs
    while run < local_n:
        next_runs = local_n - run
        if next_runs > 3:
            next_runs = 3

        if phase == 0:
            next_bp = tl.make_block_ptr(base=SortGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((src_base + run * RUN_LEN) * 2,),
                                        block_shape=(3 * RUN_LEN * 2,),
                                        order=(0,))
            buf_a = al.custom("gm_to_ub_copy_n_float", next_bp, RUN_LEN * 2,
                              next_runs * RUN_LEN * 2, out=buf_a)
            l0 = RUN_LEN
            l1 = RUN_LEN if next_runs > 0 else 0
            l2 = RUN_LEN if next_runs > 1 else 0
            l3 = RUN_LEN if next_runs > 2 else 0
            buf_b, cons = al.custom(
                "vmrgsort4_exhaust_step",
                buf_a, 4,
                0, RUN_LEN, 2 * RUN_LEN, 3 * RUN_LEN,
                l0, l1, l2, l3,
                buf_b, cons, out=[buf_b, cons])
            phase = 1
        else:
            next_bp = tl.make_block_ptr(base=SortGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((src_base + run * RUN_LEN) * 2,),
                                        block_shape=(3 * RUN_LEN * 2,),
                                        order=(0,))
            buf_b = al.custom("gm_to_ub_copy_n_float", next_bp, RUN_LEN * 2,
                              next_runs * RUN_LEN * 2, out=buf_b)
            l0 = RUN_LEN
            l1 = RUN_LEN if next_runs > 0 else 0
            l2 = RUN_LEN if next_runs > 1 else 0
            l3 = RUN_LEN if next_runs > 2 else 0
            buf_a, cons = al.custom(
                "vmrgsort4_exhaust_step",
                buf_b, 4,
                0, RUN_LEN, 2 * RUN_LEN, 3 * RUN_LEN,
                l0, l1, l2, l3,
                buf_a, cons, out=[buf_a, cons])
            phase = 0

        run += next_runs

    obp = tl.make_block_ptr(base=LocalGM, shape=(1 << 30,), strides=(1,),
                            offsets=(local_base * 2,),
                            block_shape=(K * 2,),
                            order=(0,))
    if phase == 0:
        buf_a = al.custom("ub_to_gm_copy_float", buf_a, 0, K * 2, obp,
                          out=buf_a)
    else:
        buf_b = al.custom("ub_to_gm_copy_float", buf_b, 0, K * 2, obp,
                          out=buf_b)


@triton.jit
def static_merge4_unpack_kernel(WorkGM, Yv, Yi,
                                ROW_WORDS: tl.constexpr,
                                RUN_LEN: tl.constexpr,
                                NUM_RUNS_C: tl.constexpr,
                                K: tl.constexpr):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    src_base = (row * 2 + 0) * region_props
    token = tl.zeros([1], dtype=tl.float32)

    src_bp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                               offsets=(src_base * 2,),
                               block_shape=(NUM_RUNS_C * RUN_LEN * 2,),
                               order=(0,))
    yv_bp = tl.make_block_ptr(base=Yv, shape=(1 << 30,), strides=(1,),
                              offsets=(row * K,),
                              block_shape=(K,),
                              order=(0,))
    yi_bp = tl.make_block_ptr(base=Yi, shape=(1 << 30,), strides=(1,),
                              offsets=(row * K,),
                              block_shape=(K,),
                              order=(0,))
    token = al.custom("merge4_static_gm_to_output_float",
                      src_bp, RUN_LEN, K, NUM_RUNS_C, yv_bp, yi_bp,
                      out=token)


# ════════════════════════════════════════════════════════════════════════════
# 分步校验:# ════════════════════════════════════════════════════════════════════════════
# 分步校验:把 WorkGM 中间结果拉回 host,用 CPU 参考逐 kernel 核对。
#   定位思路:哪一步先报不一致,bug 就在那个 kernel。
# ════════════════════════════════════════════════════════════════════════════
def _props_to_vi(props):
    """props: np.float32 [.., 2*L] → (value[L], index[L])。"""
    v = props[0::2].copy()
    i = props[1::2].copy().view(np.int32)
    return v, i


def _check_after_sort(x, work_gm, m, n, seg_len, num_seg_per_row,
                      row_words, prop_seg, tail_len):
    """校验 Kernel1:每个段在 WorkGM 区0 内是否 = 该段输入的完整降序。
    尾段(seg == num_seg_per_row-1)真实长度 tail_len,余下应为 -inf padding。"""
    wg = work_gm.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    bad = 0
    for r in range(m):
        region0 = (r * 2 + 0) * row_words           # 区0 f32 起点
        for s in range(num_seg_per_row):
            base = region0 + s * prop_seg
            props = wg[base: base + prop_seg]
            v, idx = _props_to_vi(props)
            is_tail = (s == num_seg_per_row - 1)
            seg_real = tail_len if is_tail else seg_len
            col0 = s * seg_len
            # 参考:该段真实输入降序(尾段只取真实 seg_real 个)
            seg_in = x_np[r, col0:col0 + seg_real]
            ref_v = np.sort(seg_in)[::-1]
            # 真实部分 value 一致
            if not np.allclose(v[:seg_real], ref_v, rtol=1e-4, atol=1e-4):
                bad += 1
                if bad <= 3:
                    print(f"  [sort BAD] row={r} seg={s}: "
                          f"v[:4]={v[:4]} ref[:4]={ref_v[:4]}")
                continue
            # 尾段 padding 部分应为 -inf
            if is_tail and seg_real < seg_len:
                pad = v[seg_real:]
                if not np.all(np.isneginf(pad)):
                    bad += 1
                    if bad <= 3:
                        n_bad = np.sum(~np.isneginf(pad))
                        print(f"  [sort BAD pad] row={r} seg={s}: "
                              f"{n_bad} 个 padding 不是 -inf,样例={pad[:4]}")
                    continue
            # index 还原(只验证真实部分):x[r, idx] == v
            idx_real = idx[:seg_real]
            if idx_real.min() < 0 or idx_real.max() >= n:
                bad += 1
                if bad <= 3:
                    print(f"  [sort BAD idx range] row={r} seg={s}: "
                          f"idx range=[{idx_real.min()},{idx_real.max()}]")
                continue
            gathered = x_np[r, idx_real]
            if not np.allclose(gathered, v[:seg_real], rtol=1e-4, atol=1e-4):
                bad += 1
                if bad <= 3:
                    print(f"  [sort BAD idx] row={r} seg={s}: gather mismatch")
    if bad == 0:
        print(f"  [sort OK] {m}x{num_seg_per_row} 段全部降序且 index 正确"
              f"(尾段 tail_len={tail_len})")
    else:
        print(f"  [sort FAIL] {bad} 段错误")


def _check_after_unpack(x, y_vals, y_idx, m, k):
    """校验 Kernel3:Yv/Yi 是否 = 全局 topk。"""
    x_np = x.detach().cpu().numpy()
    yv = y_vals.detach().cpu().numpy()
    yi = y_idx.detach().cpu().numpy()
    bad = 0
    for r in range(m):
        ref_v = np.sort(x_np[r])[::-1][:k]
        if not np.allclose(yv[r], ref_v, rtol=1e-3, atol=1e-3):
            bad += 1
            print(f"  [unpack BAD] row={r}: yv[:4]={yv[r][:4]} ref[:4]={ref_v[:4]}")
            continue
        if yi[r].min() < 0 or yi[r].max() >= x_np.shape[1]:
            bad += 1
            print(f"  [unpack BAD idx range] row={r}")
            continue
        if not np.allclose(x_np[r, yi[r]], yv[r], rtol=1e-3, atol=1e-3):
            bad += 1
            print(f"  [unpack BAD idx] row={r}: gather mismatch")
    if bad == 0:
        print(f"  [unpack OK] {m} 行 value/index 全局 topk 正确")
    else:
        print(f"  [unpack FAIL] {bad} 行错误")


def parse_bool(text):
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {text!r}")


def parse_cases(text):
    cases = []
    normalized = str(text).replace("\n", ";").replace("，", ",")
    for raw_case in normalized.split(";"):
        item = raw_case.strip()
        if not item:
            continue
        item = item.strip("()[]")
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid case {raw_case!r}; expected m,n,k")
        m, n, k = (int(part) for part in parts)
        if m <= 0 or n <= 0 or k <= 0 or k > n:
            raise ValueError(f"Invalid case {raw_case!r}; require m>0, n>0, 0<k<=n")
        cases.append((m, n, k))
    if not cases:
        raise ValueError("--cases did not contain any valid case")
    return cases


def run_correctness(m, n, k, seg_len=2048, seed=0, debug=False, bench=False,
                    device=DEFAULT_NPU_DEVICE, sorted_output=False,
                    warmup=2, active=5):
    torch.manual_seed(seed)
    torch_npu.npu.set_device(device)
    x = torch.rand((m, n), device=DEVICE, dtype=torch.float32)
    print(f"\n=== RUN  M={m} N={n} K={k} seg_len={seg_len} sorted={sorted_output} ===")
    yv, yi = topk(x, k, seg_len, debug=debug)
    if bench:
        npu_us, npu_rows, npu_detail = bench_total_us(
            lambda: torch.topk(x, k, dim=1, sorted=sorted_output),
            warmup=warmup,
            active=active,
        )
        triton_us, triton_rows, triton_detail = bench_total_us(
            lambda: topk(x, k, seg_len, debug=debug),
            warmup=warmup,
            active=active,
        )
        print(f"BENCH  torch.topk={npu_us:.3f} us  triton_topk={triton_us:.3f} us  "
              f"torch_rows={npu_rows} triton_rows={triton_rows}")
        print("KERNEL_AVG torch " + " ".join(f"{k}={v:.3f}" for k, v in sorted(npu_detail.items())))
        print("KERNEL_AVG triton " + " ".join(f"{k}={v:.3f}" for k, v in sorted(triton_detail.items())))

    tv, _ = torch.topk(x, k, dim=1, sorted=sorted_output)
    tv_sorted = torch.sort(tv, dim=1, descending=True).values
    yv_sorted = torch.sort(yv, dim=1, descending=True).values
    torch.testing.assert_close(yv_sorted, tv_sorted, rtol=1e-3, atol=1e-3)
    gathered = x.gather(1, yi.to(torch.int64))
    torch.testing.assert_close(gathered, yv, rtol=1e-3, atol=1e-3)
    print(f"PASSED  M={m} N={n} K={k} seg_len={seg_len}")


def run_simulator_only(m, n, k, seg_len=2048, seed=0, device=DEFAULT_NPU_DEVICE):
    torch.manual_seed(seed)
    torch_npu.npu.set_device(device)
    x = torch.rand((m, n), device=DEVICE, dtype=torch.float32)
    yv, yi = topk(x, k, seg_len, debug=False)
    torch_npu.npu.synchronize()
    print({
        "mode": "simulator_only",
        "m": m,
        "n": n,
        "k": k,
        "seg_len": seg_len,
        "device": device,
        "values_shape": list(yv.shape),
        "indices_shape": list(yi.shape),
    })



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_len", type=int, default=4096)
    parser.add_argument("--merge_cores", type=int, default=8,
                        help="Ignored by the original merge kernel; accepted for workflow compatibility.")
    parser.add_argument("--partition_iters", type=int, default=8,
                        help="Ignored by the original merge kernel; accepted for workflow compatibility.")
    parser.add_argument("--device", type=int, default=DEFAULT_NPU_DEVICE)
    parser.add_argument("--m", type=int, default=None,
                        help="只运行单个 case 时的 batch size")
    parser.add_argument("--n", type=int, default=None,
                        help="只运行单个 case 时的列长度")
    parser.add_argument("--k", type=int, default=None,
                        help="只运行单个 case 时的 top-k")
    parser.add_argument("--cases", default="",
                        help="串行运行多个 case,格式: 'm,n,k;m,n,k'. 不能和 --m/--n/--k 同时使用")
    parser.add_argument("--debug", action="store_true",
                        help="逐 kernel 校验 WorkGM 中间结果(定位哪一步写坏)")
    parser.add_argument("--bench", action="store_true",
                        help="额外执行 do_bench_npu 性能测量(会触发 profiler 输出)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="do_bench_npu warmup 次数")
    parser.add_argument("--active", type=int, default=5,
                        help="do_bench_npu active 次数")
    parser.add_argument("--sorted_output", default="false",
                        help="torch.topk baseline 的 sorted 参数; Triton 输出仍按当前实现返回")
    parser.add_argument("--simulator-only", action="store_true",
                        help="只执行一次自定义 Triton TopK,用于 msprof op simulator")
    parser.add_argument("--msprof-op-only", default="",
                        help="兼容旧入口:只执行指定 Triton kernel 一次")
    parser.add_argument("--simulator-single-kernel", default="",
                        help="只执行指定 Triton kernel 一次,用于 msprof op simulator 精确抓取")
    args = parser.parse_args()

    if args.cases and any(item is not None for item in (args.m, args.n, args.k)):
        parser.error("--cases 不能和 --m/--n/--k 同时使用")

    single_case_flags = [args.m is not None, args.n is not None, args.k is not None]
    if any(single_case_flags) and not all(single_case_flags):
        parser.error("--m --n --k 必须同时提供,才能只运行单个 case")

    if args.cases:
        if args.simulator_only or args.msprof_op_only or args.simulator_single_kernel:
            parser.error("--cases 只支持 correctness/bench 串行运行,不支持 simulator-only/single-kernel")
        try:
            cases = parse_cases(args.cases)
        except ValueError as exc:
            parser.error(str(exc))
    elif all(single_case_flags):
        single_kernel = (args.simulator_single_kernel or args.msprof_op_only).strip()
        if single_kernel:
            run_msprof_op_only(
                args.m, args.n, args.k,
                kernel_name=single_kernel,
                seg_len=args.seg_len,
                device=args.device,
            )
            return
        if args.simulator_only:
            run_simulator_only(args.m, args.n, args.k, seg_len=args.seg_len, device=args.device)
            return
        cases = [(args.m, args.n, args.k)]
    else:
        parser.error("请通过 --m/--n/--k 或 --cases 指定要运行的 case")

    sorted_output = parse_bool(args.sorted_output)
    for m, n, k in cases:
        run_correctness(m, n, k, seg_len=args.seg_len, debug=args.debug,
                        bench=args.bench, device=args.device,
                        sorted_output=sorted_output,
                        warmup=args.warmup, active=args.active)



# ============================================================================
# Final batch-local standalone implementation
# ============================================================================
# This section inlines the batch scheduling experiment from
# topk-step-two-batch-core-local-merge.py.

@triton.jit
def core_local_nocopy_merge_kernel(SortGM, TmpGM, FinalGM,
                                   NUM_SEG_PER_ROW: tl.constexpr,
                                   SEG_LEN: tl.constexpr,
                                   SORT_RUN_LEN: tl.constexpr,
                                   SORT_ROW_WORDS: tl.constexpr,
                                   TMP_ROW_WORDS: tl.constexpr,
                                   FINAL_ROW_WORDS: tl.constexpr,
                                   CORE_WORK_LEN: tl.constexpr,
                                   CORE_OUT_LEN: tl.constexpr,
                                   K: tl.constexpr,
                                   CHUNK_C: tl.constexpr,
                                   IN_CAP: tl.constexpr,
                                   OUT_CAP: tl.constexpr,
                                   MAX_STREAM_ITERS: tl.constexpr,
                                   MAX_WAYS_C: tl.constexpr,
                                   LOCAL_GROUPS_C: tl.constexpr,
                                   NUM_CORES_C: tl.constexpr,
                                   SKIP_FINAL_COPY: tl.constexpr):
    row = tl.program_id(0)
    cid = tl.program_id(1)

    sort_region_props = SORT_ROW_WORDS // 2
    tmp_region_props = TMP_ROW_WORDS // 2
    final_region_props = FINAL_ROW_WORDS // 2

    sort_base = (row * 2 + 0) * sort_region_props
    tmp_base_p0 = (row * 2 + 0) * tmp_region_props + cid * CORE_WORK_LEN
    tmp_base_p1 = (row * 2 + 1) * tmp_region_props + cid * CORE_WORK_LEN
    final_base_p0 = (row * 2 + 0) * final_region_props + cid * CORE_OUT_LEN

    first_seg = (cid * NUM_SEG_PER_ROW) // NUM_CORES_C
    end_seg = ((cid + 1) * NUM_SEG_PER_ROW) // NUM_CORES_C
    local_n = end_seg - first_seg

    in_buf = tl.zeros([IN_CAP], dtype=tl.float32)
    out_buf = tl.zeros([OUT_CAP + 8], dtype=tl.float32)
    cons = tl.zeros([4], dtype=tl.int32)

    # First merge round reads directly from SortGM and writes TmpGM phase 1.
    # This removes the original full SortGM->TmpGM pre-copy.
    cur_n = local_n
    cur_full = SORT_RUN_LEN
    cur_last = SORT_RUN_LEN
    src_phase = 1

    stride = SORT_RUN_LEN
    out_len = SORT_RUN_LEN * MAX_WAYS_C
    if out_len > K:
        out_len = K
    out_n = (cur_n + MAX_WAYS_C - 1) // MAX_WAYS_C
    last_ways = cur_n - MAX_WAYS_C * (out_n - 1)
    last_total = (last_ways - 1) * SORT_RUN_LEN + SORT_RUN_LEN
    nxt_full = out_len
    nxt_last = last_total
    if nxt_last > K:
        nxt_last = K

    for g in tl.range(0, LOCAL_GROUPS_C):
        if g < out_n:
            base_seg = g * MAX_WAYS_C
            ways = MAX_WAYS_C
            if base_seg + ways > cur_n:
                ways = cur_n - base_seg
            is_last_group = (g == out_n - 1)
            g0 = (base_seg + 0) * stride
            g1 = (base_seg + 1) * stride
            g2 = (base_seg + 2) * stride
            g3 = (base_seg + 3) * stride
            sl0 = SORT_RUN_LEN
            sl1 = SORT_RUN_LEN if ways > 1 else 0
            sl2 = SORT_RUN_LEN if ways > 2 else 0
            sl3 = SORT_RUN_LEN if ways > 3 else 0
            if is_last_group:
                if ways == 1:
                    sl0 = cur_last
                elif ways == 2:
                    sl1 = cur_last
                elif ways == 3:
                    sl2 = cur_last
                else:
                    sl3 = cur_last

            total = sl0 + sl1 + sl2 + sl3
            grp_cap = total
            if grp_cap > K:
                grp_cap = K
            dst_seg_off = g * out_len

            c0 = 0; c1 = 0; c2 = 0; c3 = 0
            produced = 0
            it = 0
            while (produced < grp_cap) and (it < MAX_STREAM_ITERS):
                it += 1
                r0 = sl0 - c0
                r1 = sl1 - c1 if ways > 1 else 0
                r2 = sl2 - c2 if ways > 2 else 0
                r3 = sl3 - c3 if ways > 3 else 0
                l0 = r0 if r0 < CHUNK_C else CHUNK_C
                l1 = r1 if r1 < CHUNK_C else CHUNK_C
                l2 = r2 if r2 < CHUNK_C else CHUNK_C
                l3 = r3 if r3 < CHUNK_C else CHUNK_C
                rem_ways = 0
                if r0 > 0: rem_ways += 1
                if r1 > 0: rem_ways += 1
                if r2 > 0: rem_ways += 1
                if r3 > 0: rem_ways += 1
                aw = 0
                if l0 > 0: aw += 1
                if l1 > 0: aw += 1
                if l2 > 0: aw += 1
                if l3 > 0: aw += 1

                if rem_ways == 0:
                    produced = grp_cap
                elif rem_ways == 1:
                    src_seg_idx = first_seg + base_seg
                    sres = r0
                    csrc = c0
                    if r1 > 0:
                        src_seg_idx = first_seg + base_seg + 1; sres = r1; csrc = c1
                    if r2 > 0:
                        src_seg_idx = first_seg + base_seg + 2; sres = r2; csrc = c2
                    if r3 > 0:
                        src_seg_idx = first_seg + base_seg + 3; sres = r3; csrc = c3
                    take = sres
                    if produced + take > grp_cap:
                        take = grp_cap - produced
                    cpos = 0
                    while cpos < take:
                        clen = take - cpos
                        if clen > CHUNK_C:
                            clen = CHUNK_C
                        sbp = tl.make_block_ptr(base=SortGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((sort_base + src_seg_idx * SORT_RUN_LEN + csrc + cpos) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", sbp, 0, clen * 2, out=in_buf)
                        dbp = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((tmp_base_p1 + dst_seg_off + produced + cpos) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("ub_to_gm_copy_float", in_buf, 0, clen * 2, dbp,
                                           out=in_buf)
                        cpos += clen
                    produced += take
                else:
                    if l0 > 0:
                        bp0 = tl.make_block_ptr(base=SortGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((sort_base + (first_seg + base_seg + 0) * SORT_RUN_LEN + c0) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", bp0, 0 * CHUNK_C * 2, l0 * 2, out=in_buf)
                    if l1 > 0:
                        bp1 = tl.make_block_ptr(base=SortGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((sort_base + (first_seg + base_seg + 1) * SORT_RUN_LEN + c1) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", bp1, 1 * CHUNK_C * 2, l1 * 2, out=in_buf)
                    if l2 > 0:
                        bp2 = tl.make_block_ptr(base=SortGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((sort_base + (first_seg + base_seg + 2) * SORT_RUN_LEN + c2) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", bp2, 2 * CHUNK_C * 2, l2 * 2, out=in_buf)
                    if l3 > 0:
                        bp3 = tl.make_block_ptr(base=SortGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((sort_base + (first_seg + base_seg + 3) * SORT_RUN_LEN + c3) * 2,),
                                                block_shape=(CHUNK_C * 2,), order=(0,))
                        in_buf = al.custom("gm_to_ub_copy_n_float", bp3, 3 * CHUNK_C * 2, l3 * 2, out=in_buf)

                    out_buf, cons = al.custom(
                        "vmrgsort4_exhaust_step",
                        in_buf, aw,
                        0 * CHUNK_C, 1 * CHUNK_C, 2 * CHUNK_C, 3 * CHUNK_C,
                        l0, l1, l2, l3,
                        out_buf, cons, out=[out_buf, cons])

                    e0 = al.get_element(cons, (0,))
                    e1 = al.get_element(cons, (1,))
                    e2 = al.get_element(cons, (2,))
                    e3 = al.get_element(cons, (3,))
                    batch = e0 + e1 + e2 + e3
                    take = batch
                    if produced + take > grp_cap:
                        take = grp_cap - produced
                    obp = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                            offsets=((tmp_base_p1 + dst_seg_off + produced) * 2,),
                                            block_shape=(OUT_CAP,), order=(0,))
                    out_buf = al.custom("ub_to_gm_copy_float", out_buf, 0, take * 2, obp,
                                        out=out_buf)
                    c0 += e0; c1 += e1; c2 += e2; c3 += e3
                    produced += batch
                    if batch == 0:
                        produced = grp_cap

    cur_n = out_n
    cur_full = nxt_full
    cur_last = nxt_last

    while cur_n > 1:
        stride = cur_full
        out_len = cur_full * MAX_WAYS_C
        if out_len > K:
            out_len = K
        out_n = (cur_n + MAX_WAYS_C - 1) // MAX_WAYS_C
        last_ways = cur_n - MAX_WAYS_C * (out_n - 1)
        last_total = (last_ways - 1) * cur_full + cur_last
        nxt_full = out_len
        nxt_last = last_total
        if nxt_last > K:
            nxt_last = K

        src_base = tmp_base_p0
        if src_phase == 1:
            src_base = tmp_base_p1
        dst_base = tmp_base_p1
        if src_phase == 1:
            dst_base = tmp_base_p0

        for g in tl.range(0, LOCAL_GROUPS_C):
            if g < out_n:
                base_seg = g * MAX_WAYS_C
                ways = MAX_WAYS_C
                if base_seg + ways > cur_n:
                    ways = cur_n - base_seg
                is_last_group = (g == out_n - 1)
                g0 = (base_seg + 0) * stride
                g1 = (base_seg + 1) * stride
                g2 = (base_seg + 2) * stride
                g3 = (base_seg + 3) * stride
                sl0 = cur_full
                sl1 = cur_full if ways > 1 else 0
                sl2 = cur_full if ways > 2 else 0
                sl3 = cur_full if ways > 3 else 0
                if is_last_group:
                    if ways == 1:
                        sl0 = cur_last
                    elif ways == 2:
                        sl1 = cur_last
                    elif ways == 3:
                        sl2 = cur_last
                    else:
                        sl3 = cur_last
                total = sl0 + sl1 + sl2 + sl3
                grp_cap = total
                if grp_cap > K:
                    grp_cap = K
                dst_seg_off = g * out_len

                c0 = 0; c1 = 0; c2 = 0; c3 = 0
                produced = 0
                it = 0
                while (produced < grp_cap) and (it < MAX_STREAM_ITERS):
                    it += 1
                    r0 = sl0 - c0
                    r1 = sl1 - c1 if ways > 1 else 0
                    r2 = sl2 - c2 if ways > 2 else 0
                    r3 = sl3 - c3 if ways > 3 else 0
                    l0 = r0 if r0 < CHUNK_C else CHUNK_C
                    l1 = r1 if r1 < CHUNK_C else CHUNK_C
                    l2 = r2 if r2 < CHUNK_C else CHUNK_C
                    l3 = r3 if r3 < CHUNK_C else CHUNK_C
                    rem_ways = 0
                    if r0 > 0: rem_ways += 1
                    if r1 > 0: rem_ways += 1
                    if r2 > 0: rem_ways += 1
                    if r3 > 0: rem_ways += 1
                    aw = 0
                    if l0 > 0: aw += 1
                    if l1 > 0: aw += 1
                    if l2 > 0: aw += 1
                    if l3 > 0: aw += 1

                    if rem_ways == 0:
                        produced = grp_cap
                    elif rem_ways == 1:
                        soff = src_base + g0 + c0
                        sres = r0
                        if r1 > 0:
                            soff = src_base + g1 + c1; sres = r1
                        if r2 > 0:
                            soff = src_base + g2 + c2; sres = r2
                        if r3 > 0:
                            soff = src_base + g3 + c3; sres = r3
                        take = sres
                        if produced + take > grp_cap:
                            take = grp_cap - produced
                        cpos = 0
                        while cpos < take:
                            clen = take - cpos
                            if clen > CHUNK_C:
                                clen = CHUNK_C
                            cbp = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                                    offsets=((soff + cpos) * 2,),
                                                    block_shape=(CHUNK_C * 2,), order=(0,))
                            in_buf = al.custom("gm_to_ub_copy_n_float", cbp, 0, clen * 2, out=in_buf)
                            cobp = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                                     offsets=((dst_base + dst_seg_off + produced + cpos) * 2,),
                                                     block_shape=(CHUNK_C * 2,), order=(0,))
                            in_buf = al.custom("ub_to_gm_copy_float", in_buf, 0, clen * 2, cobp,
                                               out=in_buf)
                            cpos += clen
                        produced += take
                    else:
                        if l0 > 0:
                            bp0 = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                                    offsets=((src_base + g0 + c0) * 2,),
                                                    block_shape=(CHUNK_C * 2,), order=(0,))
                            in_buf = al.custom("gm_to_ub_copy_n_float", bp0, 0 * CHUNK_C * 2, l0 * 2, out=in_buf)
                        if l1 > 0:
                            bp1 = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                                    offsets=((src_base + g1 + c1) * 2,),
                                                    block_shape=(CHUNK_C * 2,), order=(0,))
                            in_buf = al.custom("gm_to_ub_copy_n_float", bp1, 1 * CHUNK_C * 2, l1 * 2, out=in_buf)
                        if l2 > 0:
                            bp2 = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                                    offsets=((src_base + g2 + c2) * 2,),
                                                    block_shape=(CHUNK_C * 2,), order=(0,))
                            in_buf = al.custom("gm_to_ub_copy_n_float", bp2, 2 * CHUNK_C * 2, l2 * 2, out=in_buf)
                        if l3 > 0:
                            bp3 = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                                    offsets=((src_base + g3 + c3) * 2,),
                                                    block_shape=(CHUNK_C * 2,), order=(0,))
                            in_buf = al.custom("gm_to_ub_copy_n_float", bp3, 3 * CHUNK_C * 2, l3 * 2, out=in_buf)

                        out_buf, cons = al.custom(
                            "vmrgsort4_exhaust_step",
                            in_buf, aw,
                            0 * CHUNK_C, 1 * CHUNK_C, 2 * CHUNK_C, 3 * CHUNK_C,
                            l0, l1, l2, l3,
                            out_buf, cons, out=[out_buf, cons])

                        e0 = al.get_element(cons, (0,))
                        e1 = al.get_element(cons, (1,))
                        e2 = al.get_element(cons, (2,))
                        e3 = al.get_element(cons, (3,))
                        batch = e0 + e1 + e2 + e3
                        take = batch
                        if produced + take > grp_cap:
                            take = grp_cap - produced
                        obp = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                                offsets=((dst_base + dst_seg_off + produced) * 2,),
                                                block_shape=(OUT_CAP,), order=(0,))
                        out_buf = al.custom("ub_to_gm_copy_float", out_buf, 0, take * 2, obp,
                                            out=out_buf)
                        c0 += e0; c1 += e1; c2 += e2; c3 += e3
                        produced += batch
                        if batch == 0:
                            produced = grp_cap

        cur_n = out_n
        cur_full = nxt_full
        cur_last = nxt_last
        if src_phase == 0:
            src_phase = 1
        else:
            src_phase = 0

    if not SKIP_FINAL_COPY:
        valid_len = local_n * SORT_RUN_LEN
        if valid_len > K:
            valid_len = K
        if valid_len > CORE_OUT_LEN:
            valid_len = CORE_OUT_LEN

        src_final_base = tmp_base_p0
        if src_phase == 1:
            src_final_base = tmp_base_p1

        copy_pos = 0
        while copy_pos < valid_len:
            clen = valid_len - copy_pos
            clen = valid_len - copy_pos
            if clen > CHUNK_C:
                clen = CHUNK_C
            rbp = tl.make_block_ptr(base=TmpGM, shape=(1 << 30,), strides=(1,),
                                    offsets=((src_final_base + copy_pos) * 2,),
                                    block_shape=(CHUNK_C * 2,), order=(0,))
            in_buf = al.custom("gm_to_ub_copy_n_float", rbp, 0, clen * 2, out=in_buf)
            wbp = tl.make_block_ptr(base=FinalGM, shape=(1 << 30,), strides=(1,),
                                    offsets=((final_base_p0 + copy_pos) * 2,),
                                    block_shape=(CHUNK_C * 2,), order=(0,))
            in_buf = al.custom("ub_to_gm_copy_float", in_buf, 0, clen * 2, wbp, out=in_buf)
            copy_pos += clen

        offs = tl.arange(0, CHUNK_C)
        pad_pos = valid_len
        while pad_pos < CORE_OUT_LEN:
            plen = CORE_OUT_LEN - pad_pos
            if plen > CHUNK_C:
                plen = CHUNK_C
            tl.store(FinalGM + (final_base_p0 + pad_pos + offs) * 2,
                     float("-inf"), mask=offs < plen)
            tl.store(FinalGM + (final_base_p0 + pad_pos + offs) * 2 + 1,
                     0.0, mask=offs < plen)
            pad_pos += plen

SMALLK_SORT_CHUNK = 1024
SMALLK_SORT_WAYS = 4
SORT_IMPL_BASE = 0
SORT_IMPL_SMALLK = 1
SORT_IMPL_LAYERED = 2



@triton.jit
def sort_kernel(X, WorkGM,
                n_total: tl.constexpr, N_COLS: tl.constexpr,
                SEG_LEN: tl.constexpr, NUM_SEG: tl.constexpr,
                NUM_SEG_PER_ROW: tl.constexpr,
                ROW_WORDS: tl.constexpr,
                SEGS_PER_CORE: tl.constexpr,
                NUM_CORES_C: tl.constexpr,
                TMP_SIZE: tl.constexpr,
                SORT_TOPK: tl.constexpr,
                PROP_SEG: tl.constexpr,
                SORT_IMPL: tl.constexpr):
    cid = tl.program_id(0)
    buf = tl.zeros([SEG_LEN], dtype=tl.float32)
    tmp = tl.zeros([TMP_SIZE], dtype=tl.float32)
    seg_props = tl.zeros([PROP_SEG], dtype=tl.float32)
    neg_inf = float("-inf")

    for j in tl.range(0, SEGS_PER_CORE):
        seg = cid + j * NUM_CORES_C
        if seg < NUM_SEG:
            row = seg // NUM_SEG_PER_ROW
            seg_in_row = seg - row * NUM_SEG_PER_ROW
            col_off = seg_in_row * SEG_LEN
            in_off = row * N_COLS + col_off
            seg_real = N_COLS - col_off
            if seg_real > SEG_LEN:
                seg_real = SEG_LEN

            bp = tl.make_block_ptr(base=X, shape=(n_total,), strides=(1,),
                                   offsets=(in_off,),
                                   block_shape=(SEG_LEN,), order=(0,))
            if seg_real < SEG_LEN:
                buf = tl.full([SEG_LEN], neg_inf, dtype=tl.float32)
                buf = al.custom("gm_to_ub_copy_n_float", bp, 0, seg_real,
                                out=buf)
            else:
                buf = al.custom("gm_to_ub_copy_float", bp, 0, out=buf)

            if SORT_IMPL == 2:
                seg_props = al.custom("sort_1d_topk_proposals_layered_4096",
                                      buf, tmp, True, SORT_TOPK, col_off,
                                      out=seg_props)
            elif SORT_IMPL == 1:
                seg_props = al.custom("sort_1d_topk_proposals_4x1024",
                                      buf, tmp, True, SORT_TOPK, col_off,
                                      out=seg_props)
            else:
                seg_props = al.custom("sort_1d_topk_proposals", buf, tmp, True,
                                      SORT_TOPK, col_off, out=seg_props)
            dst_off = (row * 2 + 0) * ROW_WORDS + seg_in_row * PROP_SEG
            obp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                    offsets=(dst_off,),
                                    block_shape=(PROP_SEG,), order=(0,))
            seg_props = al.custom("ub_to_gm_copy_float", seg_props, 0,
                                  PROP_SEG, obp, out=seg_props)


@triton.jit
def final_merge_unpack_kernel(WorkGM, Yv, Yi,
                              ROW_WORDS: tl.constexpr,
                              RUN_LEN: tl.constexpr,
                              RUN_STRIDE: tl.constexpr,
                              NUM_RUNS_C: tl.constexpr,
                              SRC_PHASE: tl.constexpr,
                              K: tl.constexpr,
                              CHUNK_C: tl.constexpr,
                              IN_CAP: tl.constexpr,
                              OUT_CAP: tl.constexpr,
                              MAX_STREAM_ITERS: tl.constexpr,
                              UNPACK_CHUNK: tl.constexpr,
                              NUM_CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    src_base = (row * 2 + SRC_PHASE) * region_props
    dst_phase = 1 - SRC_PHASE
    dst_base = (row * 2 + dst_phase) * region_props

    in_buf = tl.zeros([IN_CAP], dtype=tl.float32)
    out_buf = tl.zeros([OUT_CAP + 8], dtype=tl.float32)
    cons = tl.zeros([4], dtype=tl.int32)

    sl0 = RUN_LEN if NUM_RUNS_C > 0 else 0
    sl1 = RUN_LEN if NUM_RUNS_C > 1 else 0
    sl2 = RUN_LEN if NUM_RUNS_C > 2 else 0
    sl3 = RUN_LEN if NUM_RUNS_C > 3 else 0
    g0 = 0
    g1 = RUN_STRIDE
    g2 = RUN_STRIDE * 2
    g3 = RUN_STRIDE * 3

    grp_cap = RUN_LEN * NUM_RUNS_C
    if grp_cap > K:
        grp_cap = K

    c0 = 0; c1 = 0; c2 = 0; c3 = 0
    produced = 0
    it = 0
    while (produced < grp_cap) and (it < MAX_STREAM_ITERS):
        it += 1
        r0 = sl0 - c0
        r1 = sl1 - c1 if NUM_RUNS_C > 1 else 0
        r2 = sl2 - c2 if NUM_RUNS_C > 2 else 0
        r3 = sl3 - c3 if NUM_RUNS_C > 3 else 0
        l0 = r0 if r0 < CHUNK_C else CHUNK_C
        l1 = r1 if r1 < CHUNK_C else CHUNK_C
        l2 = r2 if r2 < CHUNK_C else CHUNK_C
        l3 = r3 if r3 < CHUNK_C else CHUNK_C

        rem_ways = 0
        if r0 > 0: rem_ways += 1
        if r1 > 0: rem_ways += 1
        if r2 > 0: rem_ways += 1
        if r3 > 0: rem_ways += 1
        aw = 0
        if l0 > 0: aw += 1
        if l1 > 0: aw += 1
        if l2 > 0: aw += 1
        if l3 > 0: aw += 1

        if rem_ways == 0:
            produced = grp_cap
        elif rem_ways == 1:
            soff = src_base + g0 + c0
            sres = r0
            if r1 > 0:
                soff = src_base + g1 + c1; sres = r1
            if r2 > 0:
                soff = src_base + g2 + c2; sres = r2
            if r3 > 0:
                soff = src_base + g3 + c3; sres = r3
            take = sres
            if produced + take > grp_cap:
                take = grp_cap - produced
            cpos = 0
            while cpos < take:
                clen = take - cpos
                if clen > CHUNK_C:
                    clen = CHUNK_C
                rbp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((soff + cpos) * 2,),
                                        block_shape=(CHUNK_C * 2,), order=(0,))
                in_buf = al.custom("gm_to_ub_copy_n_float", rbp, 0, clen * 2, out=in_buf)
                wbp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((dst_base + produced + cpos) * 2,),
                                        block_shape=(CHUNK_C * 2,), order=(0,))
                in_buf = al.custom("ub_to_gm_copy_float", in_buf, 0, clen * 2, wbp,
                                   out=in_buf)
                cpos += clen
            produced += take
        else:
            if l0 > 0:
                bp0 = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((src_base + g0 + c0) * 2,),
                                        block_shape=(CHUNK_C * 2,), order=(0,))
                in_buf = al.custom("gm_to_ub_copy_n_float", bp0, 0 * CHUNK_C * 2, l0 * 2, out=in_buf)
            if l1 > 0:
                bp1 = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((src_base + g1 + c1) * 2,),
                                        block_shape=(CHUNK_C * 2,), order=(0,))
                in_buf = al.custom("gm_to_ub_copy_n_float", bp1, 1 * CHUNK_C * 2, l1 * 2, out=in_buf)
            if l2 > 0:
                bp2 = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((src_base + g2 + c2) * 2,),
                                        block_shape=(CHUNK_C * 2,), order=(0,))
                in_buf = al.custom("gm_to_ub_copy_n_float", bp2, 2 * CHUNK_C * 2, l2 * 2, out=in_buf)
            if l3 > 0:
                bp3 = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                        offsets=((src_base + g3 + c3) * 2,),
                                        block_shape=(CHUNK_C * 2,), order=(0,))
                in_buf = al.custom("gm_to_ub_copy_n_float", bp3, 3 * CHUNK_C * 2, l3 * 2, out=in_buf)

            out_buf, cons = al.custom(
                "vmrgsort4_exhaust_step",
                in_buf, aw,
                0 * CHUNK_C, 1 * CHUNK_C, 2 * CHUNK_C, 3 * CHUNK_C,
                l0, l1, l2, l3,
                out_buf, cons, out=[out_buf, cons])

            e0 = al.get_element(cons, (0,))
            e1 = al.get_element(cons, (1,))
            e2 = al.get_element(cons, (2,))
            e3 = al.get_element(cons, (3,))
            batch = e0 + e1 + e2 + e3
            take = batch
            if produced + take > grp_cap:
                take = grp_cap - produced
            obp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                    offsets=((dst_base + produced) * 2,),
                                    block_shape=(OUT_CAP,), order=(0,))
            out_buf = al.custom("ub_to_gm_copy_float", out_buf, 0, take * 2, obp,
                                out=out_buf)
            c0 += e0; c1 += e1; c2 += e2; c3 += e3
            produced += batch
            if batch == 0:
                produced = grp_cap

    dval = tl.zeros([UNPACK_CHUNK], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CHUNK], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CHUNK)
    for c in tl.range(0, NUM_CHUNKS):
        cstart = c * UNPACK_CHUNK
        clen = K - cstart
        if clen > UNPACK_CHUNK:
            clen = UNPACK_CHUNK
        rbp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                offsets=((dst_base + cstart) * 2,),
                                block_shape=(UNPACK_CHUNK * 2,), order=(0,))
        in_buf = al.custom("gm_to_ub_copy_n_float", rbp, 0, clen * 2, out=in_buf)
        dval, didx = al.custom("unpack_topk_float", in_buf, clen, dval, didx,
                               out=[dval, didx])
        tl.store(Yv + row * K + cstart + offs, dval, mask=offs < clen)
        tl.store(Yi + row * K + cstart + offs, didx, mask=offs < clen)


@triton.jit
def unpack_kernel(WorkGM, Yv, Yi,
                  ROW_WORDS: tl.constexpr,
                  SRC_PHASE: tl.constexpr,
                  K: tl.constexpr,
                  UNPACK_CHUNK: tl.constexpr,
                  NUM_CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    region_props = ROW_WORDS // 2
    res_base = (row * 2 + SRC_PHASE) * region_props

    sbuf = tl.zeros([UNPACK_CHUNK * 2], dtype=tl.float32)
    dval = tl.zeros([UNPACK_CHUNK], dtype=tl.float32)
    didx = tl.zeros([UNPACK_CHUNK], dtype=tl.int32)
    offs = tl.arange(0, UNPACK_CHUNK)

    for c in tl.range(0, NUM_CHUNKS):
        cstart = c * UNPACK_CHUNK
        clen = K - cstart
        if clen > UNPACK_CHUNK:
            clen = UNPACK_CHUNK
        rbp = tl.make_block_ptr(base=WorkGM, shape=(1 << 30,), strides=(1,),
                                offsets=((res_base + cstart) * 2,),
                                block_shape=(UNPACK_CHUNK * 2,), order=(0,))
        sbuf = al.custom("gm_to_ub_copy_n_float", rbp, 0, clen * 2, out=sbuf)
        dval, didx = al.custom("unpack_topk_float", sbuf, clen, dval, didx,
                               out=[dval, didx])
        tl.store(Yv + row * K + cstart + offs, dval, mask=offs < clen)
        tl.store(Yi + row * K + cstart + offs, didx, mask=offs < clen)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _final_phase(num_runs: int) -> int:
    phase = 0
    cur = num_runs
    while cur > 1:
        cur = _ceil_div(cur, MAX_WAYS)
        phase ^= 1
    return phase


def _prefer_base_medium_path(batch: int, num_seg_per_row: int, k: int,
                             seg_len: int) -> bool:
    if num_seg_per_row < DIRECT_GROUP_MIN_SEGS:
        return _sort_impl(min(k, seg_len), seg_len) == SORT_IMPL_BASE
    if num_seg_per_row > DIRECT_GROUP_MAX_SEGS:
        return False
    if k <= seg_len:
        return False

    total_cols = num_seg_per_row * seg_len
    k_ratio = k / total_cols if total_cols > 0 else 1.0
    group_runs = _ceil_div(num_seg_per_row, MAX_WAYS)
    final_runs = _ceil_div(group_runs, MAX_WAYS) if group_runs > 8 else group_runs

    # For large-k medium-S cases, the verified base path is better when its
    # group tree naturally ends at a 4-way final merge.  The batch-local path
    # would otherwise spend most time re-merging long runs in core-local.
    return k_ratio >= 0.45 and batch >= NUM_CORES // 2 and final_runs == MAX_WAYS


def _out_segs(batch: int, num_seg_per_row: int, k: int, seg_len: int) -> int:
    max_runs = min(NUM_CORES, num_seg_per_row)
    if max_runs <= 1:
        return max_runs
    if batch == 2:
        return min(16, max_runs)
    if batch <= 8 and num_seg_per_row <= DIRECT_GROUP_MAX_SEGS and k <= seg_len:
        return min(MAX_WAYS, max_runs)
    if batch == 16 and num_seg_per_row == 16 and k == seg_len:
        return min(MAX_WAYS, max_runs)

    def score(runs: int):
        programs = batch * runs
        waves = _ceil_div(programs, NUM_CORES)
        idle_tail = (-programs) % NUM_CORES
        return (waves, idle_tail, runs)

    return min(range(1, max_runs + 1), key=score)


def _use_smallk_sort(k: int, seg_len: int) -> bool:
    return (
        seg_len == SMALLK_SORT_CHUNK * SMALLK_SORT_WAYS
        and 0 < k <= SMALLK_SORT_MAX_K
    )


def _use_layered_sort(k: int, seg_len: int) -> bool:
    if seg_len != 4096 or k <= 0 or k > LAYERED_SORT_MAX_K:
        return False
    # The 4x1024 small-k path is still better around K=512, and the original
    # base path remains better around K=1024 in the measured large-S cases.
    # Keep layered for early-stop tiny K and the 2048 fixed-tree path.
    return k <= 128 or k >= 2048


def _sort_impl(k: int, seg_len: int) -> int:
    if _use_layered_sort(k, seg_len):
        return SORT_IMPL_LAYERED
    if _use_smallk_sort(k, seg_len):
        return SORT_IMPL_SMALLK
    return SORT_IMPL_BASE


def _sort_impl_for_round_robin(k: int, seg_len: int,
                               segs_per_core: int) -> int:
    sort_impl = _sort_impl(k, seg_len)
    allow_layered_multi = k <= 128
    if sort_impl == SORT_IMPL_LAYERED and segs_per_core > 2 and not allow_layered_multi:
        # The layered 4096 sort is right on the UB boundary.  When the
        # round-robin kernel unrolls three or more segment iterations per
        # program, BiSheng allocates enough extra local buffer to overflow UB.
        return SORT_IMPL_BASE
    return sort_impl


def _sort_tmp_size(seg_len: int, sort_run_len: int, sort_impl: int) -> int:
    if sort_impl == SORT_IMPL_BASE:
        return seg_len * 4
    if sort_impl == SORT_IMPL_LAYERED:
        props_a_f32 = seg_len * 2
        props_b_f32 = seg_len * 2
        group_buf_f32 = 4 * 512 * 2
        final_out_f32 = 2 * group_buf_f32 + 8
        return props_a_f32 + props_b_f32 + 2 * group_buf_f32 + final_out_f32
    candidates_f32 = SMALLK_SORT_WAYS * sort_run_len * 2
    chunk_tmp_f32 = SMALLK_SORT_CHUNK * 4
    merge_out_f32 = SMALLK_SORT_WAYS * sort_run_len * 2
    return candidates_f32 + chunk_tmp_f32 + merge_out_f32


def _use_stream16_smallk_final(k: int, run_len: int, num_runs: int) -> bool:
    return (
        0 < k <= STREAM16_SMALLK_MERGE_MAX_K
        and run_len == k
        and num_runs > 1
        and (num_runs > STREAM16_SMALLK_GROUP_RUNS or k <= TINY_S_SMALL_BATCH_MAX_K)
    )


def _use_strided_local_final_merge(out_segs: int, num_seg_per_row: int,
                                   k: int, core_out_len: int,
                                   sort_run_len: int) -> bool:
    # Low-risk subset: every local run yields the same logical output length.
    # Exact segment divisibility is sufficient.  For ragged splits, it is also
    # safe when the shortest local run still has >=K candidates, because every
    # local result is truncated to K before the final merge.
    min_input_segs_per_out_seg = num_seg_per_row // out_segs if out_segs > 0 else 0
    full_length_ragged = min_input_segs_per_out_seg * sort_run_len >= k
    return (
        1 < out_segs <= MAX_WAYS
        and core_out_len == k
        and (num_seg_per_row % out_segs == 0 or full_length_ragged)
    )


def _use_single_run_tmp_unpack(out_segs: int, k: int,
                               core_out_len: int) -> bool:
    return out_segs == 1 and core_out_len == k


def _use_smalln_2048_path(batch: int, n: int, k: int, seg_len: int) -> bool:
    new_segments = _ceil_div(n, 2048)
    return (
        seg_len == 4096
        and SMALLN_2048_MIN_N <= n <= SMALLN_2048_MAX_N
        and new_segments <= MAX_WAYS
        and batch * new_segments <= NUM_CORES
    )


def _use_tiny_s_small_batch_path(batch: int, n: int, k: int,
                                 seg_len: int) -> bool:
    num_seg_per_row = _ceil_div(n, seg_len)
    small_segments = _ceil_div(n, 2048)
    return (
        seg_len == 4096
        and 0 < batch <= 4
        and 0 < k <= TINY_S_SMALL_BATCH_MAX_K
        and num_seg_per_row <= 2
        and small_segments <= MAX_WAYS
        and batch * small_segments <= NUM_CORES
    )


def _new_topk_outputs(x: torch.Tensor, batch: int, k: int):
    y_vals = torch.empty((batch, k), device=x.device, dtype=torch.float32)
    y_idx = torch.empty((batch, k), device=x.device, dtype=torch.int32)
    return y_vals, y_idx


def _finish_debug(x, y_vals, y_idx, batch: int, k: int, debug: bool):
    if debug:
        _check_after_unpack(x, y_vals, y_idx, batch, k)
    return y_vals, y_idx


def _build_sorted_runs_round_robin(x: torch.Tensor, k: int, *,
                                   seg_len: int,
                                   debug: bool = False):
    m, n = x.shape
    n_total = m * n
    num_seg_per_row = _ceil_div(n, seg_len)
    tail_len = n - (num_seg_per_row - 1) * seg_len
    num_seg = m * num_seg_per_row
    sort_run_len = k if k < seg_len else seg_len
    prop_seg = sort_run_len * 2
    sort_row_words = num_seg_per_row * prop_seg
    sort_words = m * 2 * sort_row_words
    sort_gm = torch.empty((sort_words,), device=x.device, dtype=torch.float32)
    x_flat = x.contiguous().view(-1)

    sort_grid = min(num_seg, NUM_CORES)
    segs_per_core = _ceil_div(num_seg, sort_grid)
    sort_impl = _sort_impl_for_round_robin(sort_run_len, seg_len, segs_per_core)
    tmp_size = _sort_tmp_size(seg_len, sort_run_len, sort_impl)
    sort_kernel[(sort_grid,)](
        x_flat, sort_gm,
        n_total=n_total, N_COLS=n, SEG_LEN=seg_len, NUM_SEG=num_seg,
        NUM_SEG_PER_ROW=num_seg_per_row, ROW_WORDS=sort_row_words,
        SEGS_PER_CORE=segs_per_core, NUM_CORES_C=sort_grid,
        TMP_SIZE=tmp_size, SORT_TOPK=sort_run_len, PROP_SEG=prop_seg,
        SORT_IMPL=sort_impl,
        multibuffer=False)

    if debug and sort_run_len == seg_len:
        _check_after_sort(x, sort_gm, m, n, seg_len, num_seg_per_row,
                          sort_row_words, prop_seg, tail_len)
    return sort_gm, sort_row_words, sort_run_len, num_seg_per_row


def _launch_merge_unpack(src_gm, y_vals, y_idx, *, batch: int,
                         num_runs: int, run_len: int, row_words: int,
                         k: int, chunk: int, unpack_chunk: int):
    num_chunks = _ceil_div(k, unpack_chunk)
    in_cap = MAX_WAYS * chunk * 2
    out_cap = MAX_WAYS * chunk * 2
    max_stream = (num_runs * run_len) // chunk + 8
    generic_merge_unpack_kernel[(batch,)](
        src_gm, y_vals, y_idx,
        NUM_SEG=num_runs, SEG_LEN=run_len, ROW_WORDS=row_words,
        K=k, CHUNK_C=chunk, IN_CAP=in_cap, OUT_CAP=out_cap,
        MAX_STREAM_ITERS=max_stream, MAX_WAYS_C=MAX_WAYS,
        UNPACK_CHUNK=unpack_chunk, NUM_CHUNKS=num_chunks,
        multibuffer=False,
        enable_select_analysis=False,
    )


def _launch_final_merge_unpack(src_gm, y_vals, y_idx, *, batch: int,
                               num_runs: int, run_len: int, row_words: int,
                               k: int, chunk: int, unpack_chunk: int,
                               run_stride: int | None = None,
                               src_phase: int = 0):
    if num_runs <= MAX_WAYS:
        if run_stride is None:
            run_stride = run_len
        num_chunks = _ceil_div(k, unpack_chunk)
        in_cap = MAX_WAYS * chunk * 2
        out_cap = MAX_WAYS * chunk * 2
        max_stream = (num_runs * run_len) // chunk + 8
        final_merge_unpack_kernel[(batch,)](
            src_gm, y_vals, y_idx,
            ROW_WORDS=row_words,
            RUN_LEN=run_len,
            RUN_STRIDE=run_stride,
            NUM_RUNS_C=num_runs,
            SRC_PHASE=src_phase,
            K=k, CHUNK_C=chunk,
            IN_CAP=in_cap, OUT_CAP=out_cap,
            MAX_STREAM_ITERS=max_stream,
            UNPACK_CHUNK=unpack_chunk, NUM_CHUNKS=num_chunks,
            multibuffer=False,
            enable_select_analysis=False,
        )
        return

    _launch_merge_unpack(
        src_gm, y_vals, y_idx,
        batch=batch, num_runs=num_runs, run_len=run_len,
        row_words=row_words, k=k, chunk=chunk, unpack_chunk=unpack_chunk)


def _finish_with_final_merge_unpack(x: torch.Tensor, src_gm, *, k: int,
                                    num_runs: int, run_len: int,
                                    row_words: int,
                                    run_stride: int | None = None,
                                    src_phase: int = 0,
                                    y_vals=None, y_idx=None,
                                    debug: bool = False):
    m = x.shape[0]
    if y_vals is None or y_idx is None:
        y_vals, y_idx = _new_topk_outputs(x, m, k)
    _launch_final_merge_unpack(
        src_gm, y_vals, y_idx,
        batch=m, num_runs=num_runs, run_len=run_len,
        row_words=row_words, k=k, chunk=CHUNK, unpack_chunk=min(k, CHUNK),
        run_stride=run_stride, src_phase=src_phase,
    )
    return _finish_debug(x, y_vals, y_idx, m, k, debug)


def _topk_sort_then_final_merge(x: torch.Tensor, k: int, *,
                                seg_len: int,
                                debug: bool = False):
    sort_gm, sort_row_words, sort_run_len, num_seg_per_row = _build_sorted_runs_round_robin(
        x, k, seg_len=seg_len, debug=debug)
    return _finish_with_final_merge_unpack(
        x, sort_gm, k=k, num_runs=num_seg_per_row,
        run_len=sort_run_len, row_words=sort_row_words, debug=debug)


def _topk_tiny_s_small_batch_path(x: torch.Tensor, k: int,
                                  debug: bool = False):
    m, n = x.shape
    small_seg_len = 2048
    num_seg_per_row = _ceil_div(n, small_seg_len)
    num_seg = m * num_seg_per_row
    sort_run_len = k if k < small_seg_len else small_seg_len
    prop_seg = sort_run_len * 2
    n_total = m * n
    x_flat = x.contiguous().view(-1)

    sort_row_words = num_seg_per_row * prop_seg
    sort_words = m * 2 * sort_row_words
    sort_gm = torch.empty((sort_words,), device=x.device, dtype=torch.float32)

    sort_impl = SORT_IMPL_BASE
    tmp_size = _sort_tmp_size(small_seg_len, sort_run_len, sort_impl)
    sort_kernel[(num_seg,)](
        x_flat, sort_gm,
        n_total=n_total, N_COLS=n, SEG_LEN=small_seg_len, NUM_SEG=num_seg,
        NUM_SEG_PER_ROW=num_seg_per_row, ROW_WORDS=sort_row_words,
        SEGS_PER_CORE=1, NUM_CORES_C=num_seg,
        TMP_SIZE=tmp_size, SORT_TOPK=sort_run_len, PROP_SEG=prop_seg,
        SORT_IMPL=sort_impl,
        multibuffer=False,
    )

    y_vals, y_idx = _new_topk_outputs(x, m, k)
    static_merge4_unpack_kernel[(m,)](
        sort_gm, y_vals, y_idx,
        ROW_WORDS=sort_row_words,
        RUN_LEN=sort_run_len,
        NUM_RUNS_C=num_seg_per_row,
        K=k,
        multibuffer=False,
        enable_select_analysis=False,
    )
    return _finish_debug(x, y_vals, y_idx, m, k, debug)


def _run_stream16_smallk_merge_tree(src_gm, y_vals, y_idx, *, batch: int,
                                    num_runs: int, run_len: int,
                                    row_words: int, k: int):
    out_segs = _out_segs(batch, num_runs, k, run_len)
    if num_runs <= MAX_WAYS:
        stream_cap = triton.next_power_of_2(4 * run_len * 2)
        unpack_cap = triton.next_power_of_2(k)
        smallk_stream_merge4_unpack_kernel[(batch,)](
            src_gm, y_vals, y_idx,
            ROW_WORDS=row_words,
            RUN_LEN=run_len,
            NUM_RUNS_C=num_runs,
            K=k,
            STREAM_CAP=stream_cap,
            UNPACK_CAP=unpack_cap,
            multibuffer=False,
            enable_select_analysis=False,
        )
        return

    if batch <= 24 and num_runs > STREAM16_SMALLK_GROUP_RUNS and out_segs > 1:
        local_row_words = out_segs * k * 2
        local_words = batch * 2 * local_row_words
        local_gm = torch.empty((local_words,), device=y_vals.device,
                               dtype=torch.float32)
        stream_cap = triton.next_power_of_2(4 * run_len * 2)
        smallk_core_local_stream_merge_kernel[(batch, out_segs)](
            src_gm, local_gm,
            NUM_SEG_PER_ROW=num_runs,
            SORT_ROW_WORDS=row_words,
            LOCAL_ROW_WORDS=local_row_words,
            RUN_LEN=run_len,
            K=k,
            NUM_CORES_C=out_segs,
            STREAM_CAP=stream_cap,
            multibuffer=False,
            enable_select_analysis=False,
        )
        final_stream_cap = triton.next_power_of_2(4 * k * 2)
        unpack_cap = triton.next_power_of_2(k)
        if out_segs <= MAX_WAYS:
            smallk_stream_merge4_unpack_kernel[(batch,)](
                local_gm, y_vals, y_idx,
                ROW_WORDS=local_row_words,
                RUN_LEN=k,
                NUM_RUNS_C=out_segs,
                K=k,
                STREAM_CAP=final_stream_cap,
                UNPACK_CAP=unpack_cap,
                multibuffer=False,
                enable_select_analysis=False,
            )
        else:
            smallk_stream_merge_unpack_kernel[(batch,)](
                local_gm, y_vals, y_idx,
                ROW_WORDS=local_row_words,
                RUN_LEN=k,
                NUM_RUNS_C=out_segs,
                K=k,
                STREAM_CAP=final_stream_cap,
                UNPACK_CAP=unpack_cap,
                multibuffer=False,
                enable_select_analysis=False,
            )
        return

    stream_cap = triton.next_power_of_2(4 * run_len * 2)
    unpack_cap = triton.next_power_of_2(k)
    smallk_stream_merge_unpack_kernel[(batch,)](
        src_gm, y_vals, y_idx,
        ROW_WORDS=row_words,
        RUN_LEN=run_len,
        NUM_RUNS_C=num_runs,
        K=k,
        STREAM_CAP=stream_cap,
        UNPACK_CAP=unpack_cap,
        multibuffer=False,
        enable_select_analysis=False,
    )


def topk(x: torch.Tensor, k: int, seg_len: int = 4096, debug: bool = False):
    assert x.ndim == 2 and x.dtype == torch.float32
    m, n = x.shape
    num_seg_per_row = (n + seg_len - 1) // seg_len
    sort_run_len = k if k < seg_len else seg_len
    dispatch_sort_impl = _sort_impl(sort_run_len, seg_len)
    prefer_base_tiny = (
        m != 2
        and num_seg_per_row <= MAX_WAYS
        and dispatch_sort_impl == SORT_IMPL_BASE
    )
    use_tiny_s_small_batch = _use_tiny_s_small_batch_path(m, n, k, seg_len)

    force_stream16_smallk = _use_stream16_smallk_final(
        k, sort_run_len, num_seg_per_row) and not use_tiny_s_small_batch

    if (
        not force_stream16_smallk
        and use_tiny_s_small_batch
    ):
        return _topk_tiny_s_small_batch_path(x, k, debug=debug)

    if (
        not force_stream16_smallk
        and _use_smalln_2048_path(m, n, k, seg_len)
        and (k > TINY_S_SMALL_BATCH_MAX_K or not prefer_base_tiny)
    ):
        return _topk_sort_then_final_merge(
            x, k, seg_len=2048, debug=debug)

    # Keep verified tiny-S paths: merge16 has too much fixed overhead for S<=4.
    if (
        not force_stream16_smallk
        and
        prefer_base_tiny
    ):
        return _topk_sort_then_final_merge(
            x, k, seg_len=seg_len, debug=debug)

    # Keep verified medium paths outside the configured batch-local target.
    if (
        not force_stream16_smallk
        and
        m != 2
        and num_seg_per_row > 16
        and _prefer_base_medium_path(m, num_seg_per_row, k, seg_len)
    ):
        return _topk_sort_then_final_merge(
            x, k, seg_len=seg_len, debug=debug)

    sort_gm, sort_row_words, sort_run_len, num_seg_per_row = _build_sorted_runs_round_robin(
        x, k, seg_len=seg_len, debug=debug)

    y_vals, y_idx = _new_topk_outputs(x, m, k)

    if force_stream16_smallk:
        _run_stream16_smallk_merge_tree(
            sort_gm, y_vals, y_idx,
            batch=m, num_runs=num_seg_per_row, run_len=sort_run_len,
            row_words=sort_row_words, k=k,
        )
        return _finish_debug(x, y_vals, y_idx, m, k, debug)

    out_segs = _out_segs(m, num_seg_per_row, k, seg_len)
    if out_segs == num_seg_per_row:
        return _finish_with_final_merge_unpack(
            x, sort_gm, k=k, num_runs=num_seg_per_row,
            run_len=sort_run_len, row_words=sort_row_words,
            y_vals=y_vals, y_idx=y_idx, debug=debug)

    input_segs_per_out_seg = _ceil_div(num_seg_per_row, out_segs)
    core_work_len = input_segs_per_out_seg * sort_run_len
    core_out_len = min(core_work_len, k)
    use_strided_final = _use_strided_local_final_merge(
        out_segs, num_seg_per_row, k, core_out_len, sort_run_len)
    use_single_run_tmp_unpack = _use_single_run_tmp_unpack(
        out_segs, k, core_out_len)

    tmp_row_words = out_segs * core_work_len * 2
    tmp_words = m * 2 * tmp_row_words
    tmp_gm = torch.empty((tmp_words,), device=x.device, dtype=torch.float32)

    final_row_words = out_segs * core_out_len * 2
    final_words = m * 2 * final_row_words
    final_gm = torch.empty((final_words,), device=x.device, dtype=torch.float32)

    local_chunk = CORE_LOCAL_CHUNK
    local_in_cap = MAX_WAYS * local_chunk * 2
    local_out_cap = MAX_WAYS * local_chunk * 2
    local_max_stream = (core_work_len // local_chunk) + 8

    core_local_kernel = core_local_nocopy_merge_kernel
    core_local_kwargs = {
        "LOCAL_GROUPS_C": _ceil_div(input_segs_per_out_seg, MAX_WAYS),
    }

    core_local_kernel[(m, out_segs)](
        sort_gm, tmp_gm, final_gm,
        NUM_SEG_PER_ROW=num_seg_per_row, SEG_LEN=seg_len,
        SORT_RUN_LEN=sort_run_len,
        SORT_ROW_WORDS=sort_row_words,
        TMP_ROW_WORDS=tmp_row_words,
        FINAL_ROW_WORDS=final_row_words,
        CORE_WORK_LEN=core_work_len,
        CORE_OUT_LEN=core_out_len,
        K=k, CHUNK_C=local_chunk,
        IN_CAP=local_in_cap, OUT_CAP=local_out_cap,
        MAX_STREAM_ITERS=local_max_stream,
        MAX_WAYS_C=MAX_WAYS,
        NUM_CORES_C=out_segs,
        SKIP_FINAL_COPY=use_strided_final or use_single_run_tmp_unpack,
        **core_local_kwargs,
        multibuffer=False,
        enable_select_analysis=False,
    )

    unpack_chunk = min(k, CHUNK)
    num_chunks = (k + unpack_chunk - 1) // unpack_chunk

    if out_segs <= MAX_WAYS:
        if use_strided_final:
            final_src_phase = _final_phase(input_segs_per_out_seg)
            return _finish_with_final_merge_unpack(
                x, tmp_gm, k=k, num_runs=out_segs,
                run_len=core_out_len, row_words=tmp_row_words,
                run_stride=core_work_len, src_phase=final_src_phase,
                y_vals=y_vals, y_idx=y_idx, debug=debug)

        if out_segs == 1:
            if use_single_run_tmp_unpack:
                final_src_phase = _final_phase(input_segs_per_out_seg)
                unpack_kernel[(m,)](
                    tmp_gm, y_vals, y_idx,
                    ROW_WORDS=tmp_row_words, SRC_PHASE=final_src_phase, K=k,
                    UNPACK_CHUNK=unpack_chunk, NUM_CHUNKS=num_chunks,
                    multibuffer=False,
                    enable_select_analysis=False)
                return _finish_debug(x, y_vals, y_idx, m, k, debug)

        return _finish_with_final_merge_unpack(
            x, final_gm, k=k, num_runs=out_segs,
            run_len=core_out_len, row_words=final_row_words,
            y_vals=y_vals, y_idx=y_idx, debug=debug)

    _launch_merge_unpack(
        final_gm, y_vals, y_idx,
        batch=m, num_runs=out_segs, run_len=core_out_len,
        row_words=final_row_words, k=k, chunk=CHUNK,
        unpack_chunk=unpack_chunk,
    )
    return _finish_debug(x, y_vals, y_idx, m, k, debug)


def run_msprof_op_only(m, n, k, kernel_name: str, seg_len=4096, seed=0,
                       device=DEFAULT_NPU_DEVICE):
    if kernel_name not in {
        "core_local_nocopy_merge_kernel",
        "generic_merge_unpack_kernel",
        "core_local_merge_nocopy_kernel",
        "merge_unpack_kernel",
    }:
        raise ValueError(f"Unsupported kernel_name for final 28-case path: {kernel_name}")

    torch.manual_seed(seed)
    torch_npu.npu.set_device(device)
    x = torch.rand((m, n), device=DEVICE, dtype=torch.float32)
    assert x.ndim == 2 and x.dtype == torch.float32
    assert k <= n

    num_seg_per_row = _ceil_div(n, seg_len)
    dispatch_sort_run_len = k if k < seg_len else seg_len
    dispatch_sort_impl = _sort_impl(dispatch_sort_run_len, seg_len)
    if num_seg_per_row < DIRECT_GROUP_MIN_SEGS and dispatch_sort_impl == SORT_IMPL_BASE:
        raise ValueError(
            f"kernel {kernel_name!r} is only reachable on this batch-local path "
            f"for num_seg_per_row >= {DIRECT_GROUP_MIN_SEGS}"
        )
    if _prefer_base_medium_path(m, num_seg_per_row, k, seg_len):
        raise ValueError(
            f"kernel {kernel_name!r} is not reachable because this shape/k uses "
            "the verified base medium-segment path"
        )

    num_seg = m * num_seg_per_row
    sort_run_len = k if k < seg_len else seg_len
    prop_seg = sort_run_len * 2
    x_flat = x.contiguous().view(-1)
    sort_row_words = num_seg_per_row * sort_run_len * 2
    sort_words = m * 2 * sort_row_words
    sort_gm = torch.empty((sort_words,), device=x.device, dtype=torch.float32)

    segs_per_core = _ceil_div(num_seg, NUM_CORES)
    sort_impl = _sort_impl_for_round_robin(sort_run_len, seg_len, segs_per_core)
    tmp_size = _sort_tmp_size(seg_len, sort_run_len, sort_impl)
    sort_kernel[(NUM_CORES,)](
        x_flat, sort_gm,
        n_total=m * n, N_COLS=n, SEG_LEN=seg_len, NUM_SEG=num_seg,
        NUM_SEG_PER_ROW=num_seg_per_row, ROW_WORDS=sort_row_words,
        SEGS_PER_CORE=segs_per_core, NUM_CORES_C=NUM_CORES,
        TMP_SIZE=tmp_size, SORT_TOPK=sort_run_len, PROP_SEG=prop_seg,
        SORT_IMPL=sort_impl,
        multibuffer=False)
    torch_npu.npu.synchronize()

    kernel_name_aliases = {
        "core_local_merge_nocopy_kernel": "core_local_nocopy_merge_kernel",
        "merge_unpack_kernel": "generic_merge_unpack_kernel",
    }
    canonical_kernel_name = kernel_name_aliases.get(kernel_name, kernel_name)

    if canonical_kernel_name == "generic_merge_unpack_kernel":
        y_vals = torch.empty((m, k), device=x.device, dtype=torch.float32)
        y_idx = torch.empty((m, k), device=x.device, dtype=torch.int32)
        final_chunk = min(k, CHUNK)
        final_in_cap = MAX_WAYS * final_chunk * 2
        final_out_cap = MAX_WAYS * final_chunk * 2
        final_max_stream = (num_seg_per_row * sort_run_len) // final_chunk + 8
        unpack_chunk = min(k, CHUNK)
        num_chunks = _ceil_div(k, unpack_chunk)
        generic_merge_unpack_kernel[(m,)](
            sort_gm, y_vals, y_idx,
            NUM_SEG=num_seg_per_row,
            SEG_LEN=sort_run_len,
            ROW_WORDS=sort_row_words,
            K=k, CHUNK_C=final_chunk,
            IN_CAP=final_in_cap, OUT_CAP=final_out_cap,
            MAX_STREAM_ITERS=final_max_stream,
            MAX_WAYS_C=MAX_WAYS,
            UNPACK_CHUNK=unpack_chunk, NUM_CHUNKS=num_chunks,
            multibuffer=False,
            enable_select_analysis=False,
        )
        torch_npu.npu.synchronize()
        print({
            "mode": "msprof_op_only",
            "kernel_name": canonical_kernel_name,
            "m": m,
            "n": n,
            "k": k,
            "seg_len": seg_len,
            "sort_run_len": sort_run_len,
            "num_seg_per_row": num_seg_per_row,
            "programs": m,
            "row_words": sort_row_words,
        })
        return

    out_segs = _out_segs(m, num_seg_per_row, k, seg_len)
    input_segs_per_out_seg = _ceil_div(num_seg_per_row, out_segs)
    core_work_len = input_segs_per_out_seg * sort_run_len
    core_out_len = min(core_work_len, k)

    tmp_row_words = out_segs * core_work_len * 2
    tmp_words = m * 2 * tmp_row_words
    tmp_gm = torch.empty((tmp_words,), device=x.device, dtype=torch.float32)

    final_row_words = out_segs * core_out_len * 2
    final_words = m * 2 * final_row_words
    final_gm = torch.empty((final_words,), device=x.device, dtype=torch.float32)

    local_chunk = CORE_LOCAL_CHUNK
    local_in_cap = MAX_WAYS * local_chunk * 2
    local_out_cap = MAX_WAYS * local_chunk * 2
    local_max_stream = (core_work_len // local_chunk) + 8

    core_local_kernel = core_local_nocopy_merge_kernel
    core_local_kernel[(m, out_segs)](
        sort_gm, tmp_gm, final_gm,
        NUM_SEG_PER_ROW=num_seg_per_row, SEG_LEN=seg_len,
        SORT_RUN_LEN=sort_run_len,
        SORT_ROW_WORDS=sort_row_words,
        TMP_ROW_WORDS=tmp_row_words,
        FINAL_ROW_WORDS=final_row_words,
        CORE_WORK_LEN=core_work_len,
        CORE_OUT_LEN=core_out_len,
        K=k, CHUNK_C=local_chunk,
        IN_CAP=local_in_cap, OUT_CAP=local_out_cap,
        MAX_STREAM_ITERS=local_max_stream,
        MAX_WAYS_C=MAX_WAYS,
        LOCAL_GROUPS_C=_ceil_div(input_segs_per_out_seg, MAX_WAYS),
        NUM_CORES_C=out_segs,
        SKIP_FINAL_COPY=False,
        multibuffer=False,
        enable_select_analysis=False,
    )
    torch_npu.npu.synchronize()

    print({
        "mode": "msprof_op_only",
        "kernel_name": canonical_kernel_name,
        "m": m,
        "n": n,
        "k": k,
        "seg_len": seg_len,
        "sort_run_len": sort_run_len,
        "num_seg_per_row": num_seg_per_row,
        "out_segs": out_segs,
        "programs": m * out_segs,
        "input_segs_per_out_seg": input_segs_per_out_seg,
        "core_work_len": core_work_len,
        "core_out_len": core_out_len,
    })



if __name__ == "__main__":
    main()
