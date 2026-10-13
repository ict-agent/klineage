# GEMM workload

GEMM uses a [FlashInfer Trace](https://bench.flashinfer.ai/docs/flashinfer-trace/definition)
definition and workload records:

```text
problems/
  gemm/
    definition.json        # Contract and executable reference.run
    workloads.jsonl        # Required file; zero or more Trace records
    inputs/                # Optional fixed input files
```

| Name | Inputs | Default workload |
| --- | --- | --- |
| gemm | `x[M,K]`, `weight[N,K]`; `x @ weight.T` | M=N=K=4096, BF16 |

The workload uses `random` for Gaussian inputs.
Each JSONL line is a Trace record. `Problem.workloads` exposes the ordered
tuple of inner `workload` mappings; every case has a unique nonempty `uuid`.
GEMM currently has one case. Additional records can
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
Problem's name, not its directory name. With `KLINEAGE_SERVER` configured, agents
discover them with `klineage.info()["problems"]` and select `klineage.problem(problem_id)`.
Pass the same problem directory to `Init` or `init_memory`, using the matching registered
problem ID.
Evaluation runs every registered workload in a separate worker under one shared
deadline and retains results by UUID. All cases must pass; a definition without
cases cannot pass evaluation. Multiple cases have no aggregate latency or
cross-shape average. Use `problem.values("inputs", workload=case)` or
`problem.values("outputs", workload=case)` for ordered dictionaries with concrete shapes;
without a selection, zero or multiple cases leave variable dimensions symbolic.

GEMM preserves the paper reference from
`ict-agent/fgw_workspace/experiments/_archive/trajectory_transfer`.
