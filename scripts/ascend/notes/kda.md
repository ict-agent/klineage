## Problem

`experiment/kda/problems/definitions/kda.json`: Kimi-K3 bounded-gate KDA prefill.
One workload only, `experiment/kda/problems/workloads/kda.jsonl`:

```text
BATCH=1  TOKENS=4096  HEADS=96  HEAD_DIM=128
q,k,v,g   : bf16   beta,a_log,dt_bias,scale,lower_bound,initial_state : fp32
output    : bf16 [1,4096,96,128]   final_state : fp32 [1,96,128,128] (value,key)
```

Correctness oracle: the `reference` field of the definition (torch, per-token
recurrence, fp32) executed on the NPU. The gate requires identical output
shape/dtype/device and `torch.allclose(rtol=atol=1e-2)`; the reference defines
no extra `check_outputs`. That constant, `NUMERICAL_TOLERANCE = 1e-2`
(`src/klineage/constants.py`), is upstream's and cannot be overridden
(`parse_config` rejects any other value), so both settings are graded equally.

## Second reference point

The task's baseline column uses the definition's torch reference, which is a
4096-step eager loop (1.58-1.63 s on the NPU). It is not a community-grade
implementation, so every number is also reported against a real Ascend kernel
measured by us with the same gate protocol on device 4:

| Implementation | Latency | vs. torch reference |
| --- | --- | --- |
| vllm-ascend triton `chunk_kda` (chunk 64, qk-l2norm in kernel, `kda.py`) | 106.02 ms | 15.0x |
| without_memory (best) | 43.97 ms | 37.0x |
| with_memory (final) | 81.85 ms | 19.4x |

Speedup against that community implementation: without_memory 2.41x,
with_memory 1.30x. Tool: `scripts/ascend/community_baseline.py`.

## Accuracy margin (independent check, seed 0, same oracle)

| Setting | output max abs err | final_state max abs err | verdict at 1e-2 |
| --- | --- | --- | --- |
| without_memory | 2.4e-4 | 1.2e-7 | pass, large margin |
| with_memory | 4.9e-4 | 4.3e-3 | pass, 2.3x under atol |

## Deviations and caveats

- Language is pinned to triton-ascend for both settings; the task leaves the
  implementation language open. The pin keeps the two settings comparable and
  the gate rejects torch-only bundles.
- Setting A also had the upstream AscendC bundle skill readable at
  `src/klineage/skills/ascendc/SKILL.md` (only the `.agents/skills` symlink had
  been removed). The task itself cites that file as the artifact-spec sample,
  so it counts as "artifact implementation spec"; it is pruned from future units.
- Setting A listed the sibling setting's directory once (`ls -la
  ~/klineage-runs/kda/{without_memory,with_memory}`); no file contents of the
  other setting were read.
- Upstream ships no `inputs/kda.safetensors`, so the workload here generates the
  eight data inputs with `{"type": "random"}` (seeded `torch.randn` of the
  definition's shape/dtype) and keeps `scale`/`lower_bound` in a 176-byte
  safetensors file. Reference and shapes are unchanged.
- Each unit's first gate call measured the baseline and, in the harness version
  those units froze, deleted the mirrored run directory before the snapshot
  fetch: the first evaluated version of each setting has no snapshot under
  `versions/`. `with_memory/versions/version0` was reconstructed from the final
  submission, which is byte-identical to what that single call evaluated
  (`submission/solution/kernel.py` mtime 15:36:12, gate at 15:36:28).
