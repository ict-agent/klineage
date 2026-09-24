# GDN on Ascend: B1 T4096 HQ16 HV48 D128, chunk 64

## 1. Operator

- Compute: chunked gated delta rule with grouped value heads (GVA), as defined in
  `problems/definitions/gdn.json`.
- Inputs: `q`/`k` BF16 `[B, T, HQ=16, D=128]` (L2-normalized), `v` BF16
  `[B, T, HV=48, D]`, `g`/`beta` FP32 `[B, T, HV]` (log gates, update strength),
  `initial_state` FP32 `[B, HV, D, D]`.
- Outputs: `output` FP32 `[B, T, HV, D]`, `final_state` FP32 `[B, HV, D, D]`.
- Test shape: `B1 T4096 HQ16 HV48 D128`, `chunk_size=64`; Q/K heads repeated x3 to
  match the value heads, as the workload reference does.
- Correctness: `torch.allclose` against the FLA chunk oracle embedded in
  `problems/definitions/gdn.json`, gate `rtol = atol = 1e-2` (the harness gate).

## 2. Baseline

**vLLM-Ascend Triton chain** (`baseline/fla`, vendored verbatim from
`vllm_ascend/ops/triton/fla`; `chunk_delta_h.py` and `wy_fast.py` are byte-identical
to `vllm-ascend@5cb98caaa`): seven stages — local cumsum, causal KKT, triangular
solve, WY recompute, Stage 6 state recursion, Stage 7 output — one Triton kernel
per stage, no fusion.

## 3. Results

910B1, device 6, container `zcj-vllm-v0.22.1rc1`
(`quay.io/ascend/vllm-ascend:v0.22.1rc1`, triton-ascend 3.2.0, torch_npu 2.10.0),
2026-09-24. Raw log and JSON: `results/`.

| Implementation | Correct | Latency | vs. Baseline |
|---|---|---|---|
| vLLM-Ascend chain | yes (0 violations) | 4764.41 us | 1.00x |
| ours (fused Stage 6/7) | yes (0 violations) | 3393.44 us | **1.404x** |

- Latency = sum of device kernel durations per iteration, `do_bench_npu` with
  warmup 3 / active 10, median of the active samples after trimming one minimum
  and one maximum. Host launch gaps are excluded; both chains are measured the
  same way.
- Ten active samples, baseline: 4792.3 / 4744.0 / 4766.6 / 4748.8 / 4786.4 /
  4773.7 / 4771.7 / 4761.0 / 4744.1 / 4762.3 us.
- Ten active samples, ours: 3393.4 / 3378.7 / 3373.2 / 3374.5 / 3393.7 / 3416.9 /
  3405.8 / 3379.1 / 3393.5 / 3402.5 us.

### 3.1 Correctness

`|want|` is the mean absolute value of the oracle output; `viol` counts elements
where `|actual - expected| > atol + rtol * |expected|`.

| actual | expected | tensor | max abs err | \|want\| | viol |
|---|---|---|---|---|---|
| ours | vLLM chain | output | 1.953e-03 | 0.020 | 0 |
| ours | vLLM chain | final_state | 1.171e-02 | 0.214 | 0 |
| vLLM chain | FLA oracle | output | 1.426e-03 | 0.020 | 0 |
| vLLM chain | FLA oracle | final_state | 1.048e-02 | 0.214 | 0 |
| ours | FLA oracle | output | 2.004e-03 | 0.020 | 0 |
| ours | FLA oracle | final_state | 1.026e-02 | 0.214 | 0 |

The margin is thin on both sides: this workload runs the whole recurrence in
BF16, and the FP32 oracle sits about 1e-2 away from any BF16-clean pipeline. The
FP32 staging note in section 5 is what moves ours from 3 violating elements to 0.

### 3.2 Where the speedup comes from (mean us per iteration)

| Kernel | vLLM chain | ours |
|---|---|---|
| Stage 6 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 1662.40 | — |
| Stage 7 `chunk_fwd_kernel_o` | 737.83 | — |
| fused Stage 6/7 `_fused_stage6_7_kernel` | — | 1055.22 |

Stage 6+7: 2400.23 us → 1055.22 us = **2.27x**; the remaining stages are shared
by both chains and cap the whole-op speedup at 1.404x.

## 4. Reproducing

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh   # adjust to your install path
bash run_benchmark.sh [device_id]                # default device 0
```

The script runs `bench_gdn.py`, which stages the workload inputs exactly as the
harness does (`klineage.artifact.problem.trace_inputs`: safetensors for
`q`/`k`/`g`/`beta`, `torch.randn` seed 0 on CPU for `v`/`initial_state`), checks
both chains against the FLA oracle, and times them. `--problem-root` points at a
checkout root that contains `problems/`; `--skip-reference` drops the slow FP32
oracle.

Requirements: an Ascend NPU host with `torch_npu`, triton-ascend (FlagTree), and
`vllm` + `vllm_ascend` importable (`vllm.triton_utils` for the kernels,
`vllm_ascend.ops.triton.triton_utils` for `extract_slice`/`insert_slice` in
`solve_tril`). The vLLM-Ascend kernels only compile through their varlen branch,
so the benchmark passes `cu_seqlens = [0, T]` and prebuilt chunk metadata.
Build matters for the fused kernel: the triton 3.5.1 build in the local
`vllm0.23.0-zcj` container reports `ub overflow, requires 1579008 bits while
1572864 bits available`, while the 3.2.0 build used above compiles it.

## 5. Implementation

`ours/fused.py` + `ours/stage6.py`: our Triton-ascend kernel set. `chunk_h_o_fused`
replaces the Stage 6 + Stage 7 pair with one launch: the chunk state recursion,
the `K^T @ v_new` update and the output read stay in registers on the same AI
core, so the BF16 `h` round trip between two kernels disappears. Stages 1-5 are
the vLLM kernels, unchanged and shared with the baseline (the fused kernel takes
`w`, `u` and `g_cumsum` from them).

Copied from `gdn/workspace/optimized/{fused,stage6}.py` with one deliberate
deviation: the per-chunk `K^T @ v_new` staging buffer is FP32 (`kv_workspace` in
`chunk_h_o_fused_with_intermediates`). In BF16 it rounds the update term every
chunk and pushes 3 of 786432 `final_state` elements past the 1e-2 gate; FP32
staging measures identical (1055.22 us either way) and clears the gate.

## 6. Layout

```text
optimization/
├── README.md                  # this file
├── bench_gdn.py               # correctness + latency for both chains
├── run_benchmark.sh           # one-shot reproduction
├── baseline/fla/              # vLLM-Ascend Triton chain (vendored, unmodified)
├── ours/                      # our fused Stage 6/7 implementation
└── results/                   # raw log and JSON of the run in section 3
```
