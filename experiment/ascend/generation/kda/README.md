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

- Language: `ascendc` (`submission/solution/chunk_kda_fwd/` CANN project), shared
  by both settings; the task leaves the choice open.
- Devices: without_memory on NPU 6; with_memory on NPU 7. Each baseline is
  measured on the card its setting runs on (1759.14 ms vs 1867.39 ms, ~6%
  apart), so the latency column is the one that compares across settings.

## Measurement protocol

- Gate: `scripts/ascend/eval.py`, one call per candidate version, evaluated
  on 910b3 in a private `vllm-ascend:v0.23.0` container (torch-npu, CANN 9.1.0).
- Timer: NPU stream events (`npu-events`) around one operator call:
  1 warmup + 3 timed iterations, median reported (`_NPU_POLICY`,
  `src/klineage/harness/timing.py`). Compilation and input preparation sit
  outside the timed interval.
- Baseline: the problem definition's torch reference on the NPU, same
  timer and policy, measured once per card and cached in `baseline.json`
  under the NPU id.
- Speedup: `baseline_ms / latency_ms`.
- A version counts only when the gate passes it: output shape, dtype and
  device as the definition asks, plus its correctness oracle.
  Every evaluated version is frozen under `versions/version<N>/`.

## Results

| Kernel | Setting | Correct | Latency | vs. Baseline |
| --- | --- | --- | --- | --- |
| KDA | Without Expert Knowledge | ✓ | 44.89 ms | 39.19× |
| KDA | With Expert Knowledge | ✓ | 31.56 ms | 59.16× |

Both sessions ran their two-hour budget (119.8 / 120.1 min) and stopped
themselves, not on a timeout. Gate calls / published versions: 39 / 33
without, 25 / 20 with.

The expert run converged earlier, not just lower: its first real gate
(42.53 ms, 20 min in) already beat the bare run's final best (44.89 ms),
and it stayed ahead for the rest of the budget. The pack is CANN
`ops-transformer`'s AscendC `chunk_kda_fwd` (`with_memory/expert/source.json`).

## Artifacts

```text
kda/
|- README.md               this file
|- plots/                  csv, table.md, running-best speedup PNG and PDF
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
