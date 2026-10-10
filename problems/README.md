# Paper workloads

Five problems use [FlashInfer Trace](https://bench.flashinfer.ai/docs/flashinfer-trace/definition)
definitions and workload records. Each problem owns one directory:

```text
problems/
  {name}/
    definition.json        # Contract and executable reference.run
    workloads.jsonl        # Required file; zero or more Trace records
    inputs/                # Optional fixed input files
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
it does not regenerate these fixed samples. Input paths are relative to their
own problem directory, such as `problems/fmha/`.
Each JSONL line is a Trace record. `Problem.workloads` exposes the ordered
tuple of inner `workload` mappings; every case has a unique nonempty `uuid`.
The five bundled examples currently have one case each. Additional records can
bind other shapes or inputs without duplicating their definition. `workloads.jsonl`
must exist; an empty file leaves a valid contract with no evaluation cases.
Generation requires at least one record.

Pass the whole directory to generation:

```bash
klineage-generation problems/gemm --workdir agent-workspace/gemm
```

By default, generation detects the local backend and starts evaluation. Add
`--server ENDPOINT` to use an existing evaluator and its registered target.
An existing `target.json` is preserved and must match the selected backend.

Inspect a problem with the library on a host with the matching backend runtime:

```python
from pathlib import Path
from klineage.artifact import load_problem
from klineage.backend import get_backend

problem = load_problem(
    Path("problems/gemm"),
    backend=get_backend("cuda"),
)
print(problem.name, len(problem.workloads))
```

For the [evaluation service](../docs/guides/evaluation.md), set
`KLINEAGE_PROBLEM=problems` to register the collection, or pass
`--problem problems/gemm` to override it with one problem. IDs come from each
Problem's name, not its directory name. Agents discover them with
`Client().info()["problems"]` and select `Client().problem(problem_id)`.
Pass the same problem directory to `Init` or `init_memory`, using the matching registered
problem ID.
Evaluation runs every registered workload in a separate worker under one shared
deadline and retains results by UUID. All cases must pass; a definition without
cases cannot pass evaluation. Multiple cases have no aggregate latency or
cross-shape average. Use `problem.values("inputs", workload=case)` or
`problem.values("outputs", workload=case)` for ordered dictionaries with concrete shapes;
without a selection, zero or multiple cases leave variable dimensions symbolic.

GEMM, Conv2d, FMHA and Top-K preserve the paper references from
`ict-agent/fgw_workspace/experiments/_archive/trajectory_transfer`.
GDN preserves the mathematical references adapted from
[FLA](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py)
under its [MIT license](FLA_LICENSE).
Top-K retains the paper's sequence length 4096.
