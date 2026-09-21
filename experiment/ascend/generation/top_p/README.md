# top_p — AI-generated Ascend implementation

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

`top_p` (top_p_renorm_probs)

Axes: B=32, V=128256.

| Tensor | Direction | Shape | Dtype |
| --- | --- | --- | --- |
| `probs` | input | [32, 128256] | float32 |
| `top_p` | input | [32] | float32 |
| `renorm_probs` | output | [32, 128256] | float32 |

Both settings are graded against the definition's torch reference
running on the NPU: output shape, dtype and device, then
`torch.allclose(rtol=atol=0.01)` (`NUMERICAL_TOLERANCE`,
`src/klineage/constants.py`). This definition also defines
`check_outputs`, which the gate runs on top of that; its assertions are
in the definition's reference field.

## Setup

- Language: `ascendc` (`submission/config.toml`), shared by both
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
| TOP_P | Without Expert Knowledge | ✓ | 342.8 us | 13.21× |
| TOP_P | With Expert Knowledge | ✓ | 248.1 us | 18.51× |

- Baselines, the flat lines the curves are read against: Without Expert
  Knowledge 4.527 ms; With Expert Knowledge 4.592 ms.

## Trajectory

Milestones of each session, clocked from its first API response, so both
axes of the curve are readable. `first` is the gate call that first
passed: the curve leaves the flat baseline line there. `peak` is the
first gated call within 5% of that setting's best; from there on,
session time stopped buying performance. `best` is the final artifact's
own measurement.

| Setting | First passing gate | Peak (≤1.05× best) | Best gate | Gate calls |
| --- | --- | --- | --- | --- |
| Without Expert Knowledge | 21.6 min · 6.59 M · 0.6951 ms (6.51×) | 83.5 min · 16.26 M · 0.3474 ms (13.03×) | 100.3 min · 18.89 M · 0.3428 ms (13.21×) | 8 (1 rejected) |
| With Expert Knowledge | 29.6 min · 7.48 M · 0.5004 ms (9.18×) | 78.0 min · 17.05 M · 0.2541 ms (18.07×) | 87.8 min · 19.61 M · 0.2481 ms (18.51×) | 14 (2 rejected) |

## Artifacts

```text
top_p/
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
scripts/ascend/collect.py --kernel top_p   # rebuild this tree from ~/klineage-runs
scripts/ascend/plot.py --root <run root> --x tokens --y latency
```
