# kda — AI-generated Ascend implementation

One Codex session per setting, same prompt, same gate, same budget; the
only difference is the expert material the agent may read, under
`with_memory/work/expert/` versus nothing.

The deliverable is a curve, not a single number: every accepted version
has a gated latency, so a session traces its performance against
wall-clock time and spent tokens. The baseline is a flat line — the
torch reference never improves. Read the curves for two things: how far
above that line a setting ended up, and how early in time/tokens it got
there.

## Problem

`kda` (kda_prefill) — Kimi-K3 bounded-gate KDA prefill, independent FP32 recurrence.

Axes: BATCH=1, TOKENS=4096, HEADS=96, HEAD_DIM=128.

| Tensor | Direction | Shape | Dtype |
| --- | --- | --- | --- |
| `q` | input | [1, 4096, 96, 128] | bfloat16 |
| `k` | input | [1, 4096, 96, 128] | bfloat16 |
| `v` | input | [1, 4096, 96, 128] | bfloat16 |
| `g` | input | [1, 4096, 96, 128] | bfloat16 |
| `beta` | input | [1, 4096, 96] | float32 |
| `scale` | input | scalar | float32 |
| `a_log` | input | [96] | float32 |
| `dt_bias` | input | [96, 128] | float32 |
| `lower_bound` | input | scalar | float32 |
| `initial_state` | input | [1, 96, 128, 128] | float32 |
| `output` | output | [1, 4096, 96, 128] | bfloat16 |
| `final_state` | output | [1, 96, 128, 128] | float32 |

Both settings are graded against the definition's torch reference
running on the NPU: output shape, dtype and device, then
`torch.allclose(rtol=atol=0.01)` (`NUMERICAL_TOLERANCE`,
`src/klineage/constants.py`).

## Setup

- Language: `python` (`submission/config.toml`), shared by both
  settings; the task leaves the choice open.
- Devices: without_memory on NPU 2; with_memory on NPU 3; each setting
  scores against the baseline measured on its own device.

## Measurement protocol

- Gate: `scripts/ascend/eval.py`, one call per candidate version,
  evaluated on 910B1 inside the `vllm0.23.0-zcj` container (torch-npu,
  CANN 9.1.0).
- Timer: NPU stream events (`npu-events`) around one operator call:
  10 warmup + 50 timed iterations, 3 trials, median of trial medians.
  Compilation and input preparation sit outside the timed interval.
- Baseline: the problem definition's torch reference on the NPU, same
  timer and policy, measured once per device and cached in
  `baseline.json`; both settings of a kernel score against the baseline
  of the device they ran on.
- Speedup: `baseline_ms / latency_ms`.
- A version counts only when the gate passes it: output shape, dtype and
  device as the definition asks, plus its correctness oracle.
  Every evaluated version is frozen under `versions/version<N>/`.

## Results

| Kernel | Setting | Correct | Latency (us) | vs. Baseline |
| --- | --- | --- | --- | --- |
| KDA | Without Expert Knowledge | ✓ | 22312.0 us | 74.72× |
| KDA | With Expert Knowledge | ✓ | 23075.3 us | 66.44× |

- Baselines, the flat lines the curves are read against: Without Expert
  Knowledge 1667 ms; With Expert Knowledge 1533 ms.

## Trajectory

Milestones of each session, clocked from its first API response, so both
axes of the curve are readable. `first` is the gate call that first
passed: the curve leaves the flat baseline line there. `peak` is the
first gated call within 5% of that setting's best; from there on,
session time stopped buying performance. `best` is the final artifact's
own measurement.

| Setting | First passing gate | Peak (≤1.05× best) | Best gate | Gate calls |
| --- | --- | --- | --- | --- |
| Without Expert Knowledge | 36.6 min · 11.50 M · 43.07 ms (38.70×) | 75.9 min · 19.58 M · 23.36 ms (71.35×) | 95.8 min · 25.46 M · 22.31 ms (74.72×) | 10 (1 rejected) |
| With Expert Knowledge | 5.1 min · 2.45 M · 38.12 ms (40.22×) | 75.9 min · 15.79 M · 23.29 ms (65.84×) | 102.1 min · 20.30 M · 23.08 ms (66.44×) | 35 (7 rejected) |

## Artifacts

```text
kda/
|- README.md               this file
|- plots/                  csv per setting, table.md, latency curve
|- without_memory/         bare prompt
`- with_memory/            prompt plus the expert pack (see expert/source.json)
```

Each setting directory holds `events.jsonl` (one `api_response` record
per model call, one `evaluate` record per gate call, both timestamped),
`trace.jsonl` (the Codex session), `baseline.json`, the frozen
`versions/version<N>/` bundles and the submitted `submission/`.

## Regenerating

```sh
scripts/ascend/collect.py --kernel kda   # rebuild this tree from ~/klineage-runs
scripts/ascend/plot.py --root <run root> --x tokens --y latency
```
