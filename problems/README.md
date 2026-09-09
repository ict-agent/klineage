# Paper workloads

Five problems in [FlashInfer Trace](https://bench.flashinfer.ai/docs/flashinfer-trace/definition)
format, using AKO4X's dataset layout:

```text
problems/
  definitions/{name}.json   # Contract and executable reference.run
  workloads/{name}.jsonl    # One Trace record with axes and input sources
  inputs/{fmha,gdn}.safetensors   # Structured input samples
```

| Name | Inputs | Default workload |
| --- | --- | --- |
| gemm | `x[M,K]`, `weight[N,K]`; `x @ weight.T` | M=N=K=4096, BF16 |
| conv2d | NHWC activation, flattened HWCF filter | N8 C64 H56 W56 F128, FP16; K3, stride1, padding1 |
| fmha | Packed NHD Q/K/V, int32 sequence offsets | B8 H64 S2048 D128, FP16; noncausal |
| gdn | BTHD Q/K/V, log gates, beta, initial state | B1 T4096 Hq16 Hv48 D128; chunk64 |
| topk | Row-major values; unsorted largest K | B64 S4096 K512, FP32; int64 output indices |

GDN's Q/K/V are BF16; gates, beta, initial state and both outputs are FP32.
The reference repeats Q/K heads to match value heads and retains FLA's chunk
oracle and MIT attribution.

Workloads use `random` for Gaussian inputs. FMHA offsets and GDN's normalized
Q/K, log gates and beta retain the original CUDA seed-0 samples in safetensors
(about 35 MB total). Changing the evaluator seed changes only random inputs;
it does not regenerate these fixed samples. Paths are relative to `problems/`.
The JSONL line is a Trace record; `ProblemSpec.workload` uses its inner `workload`.

```python
from pathlib import Path
from klineage.harness import inspect_problem

problem = inspect_problem(
    Path("problems/definitions/gemm.json"),
    Path("agent-workspace/inspect-gemm"),
)
```

Pass the same definition path as `problem` to `Init` or `workflow`.
The harness requires exactly one workload per definition.

GEMM, Conv2d, FMHA and Top-K preserve the paper references from
`ict-agent/fgw_workspace/experiments/_archive/trajectory_transfer`.
GDN preserves the mathematical references adapted from
[FLA](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py)
under its [MIT license](FLA_LICENSE).
Top-K retains the paper's sequence length 4096.
