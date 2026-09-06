# Baseline experiments

Five default workloads, three methods, two models. Every job receives 43,200
seconds, including generation, compilation and validation. The runner reserves
five minutes for independent final validation. It completes Astra before Luna
and runs one job at a time on the visible GPU.
Each isolated workspace receives the repository's [AGENTS.md](../AGENTS.md).

| Directory | Model | Reasoning |
| --- | --- | --- |
| `*/astra/<problem>/` | `gpt-6-astra` | `xhigh` |
| `*/luna/<problem>/` | `gpt-5.6-luna` | `max` |

Methods are `KDA`, `codex` and `AKO4ALL`. KDA uses its pinned upstream minimal
workflow, KernelWiki and profiling skill; Humanize is optional in that revision.
AKO4ALL follows its upstream skill using our custom evaluator. Direct Codex
receives the same workload and evaluator without either framework.

Upstreams:

- [KDA](https://github.com/mit-han-lab/kernel-design-agents/tree/9f5ee5d5cd623c2d69c4ff71c8a5155edcb8c43b)
- [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL/tree/8d3065a7588c124182fa0fe0b3e589879641e1b7)

Install CUDA-enabled PyTorch, Triton, FlashInfer and CUPTI Python >=13, authenticate
Codex and Git, then clone the pinned upstreams into `agent-workspace/upstream/`.
KDA requires its pinned submodules (`git submodule update --init --recursive`).

```bash
python baseline/run.py
# One job:
python baseline/run.py --job KDA astra gemm
# Resume a failed job after inspecting its saved error:
python baseline/run.py --resume
```

Each finished job is committed and pushed to `origin`. Model-capacity failures
retry after 60 seconds with the same session, model and deadline. Other failures
save and publish their evidence, then stop the queue for inspection. `--resume`
continues failed jobs within their original deadlines. A running lock prevents
concurrent suite processes; restarting never silently resets a budget.

Each job saves:

- `prompt.txt`, `status.json`: contract, model, budget, upstream and outcome.
- `traces/`, `rollout.jsonl.gz`: CLI events and the complete Codex session.
- `kernel/`, `evaluation.json`: selected sources and independent final verdict.
- `workspace.tar.gz`: candidate sources, measurements, notes and reports.

Build caches and binary profiler reports are excluded; retain profiler summaries
in the workspace. `validated` means the numerical and timing gates passed;
source review remains necessary before accepting the experimental result.

The evaluator uses unchanged default shapes, seeds 17/43/101, and an additional
input mutation check after timing. Timing uses FlashInfer CUPTI, 10 warmups,
50 repeats, three trials and cold L2. Final evaluations also time the reference.
A shared process lock prevents concurrent evaluator GPU work.
All outputs must preserve reference shapes and dtypes. Top-K instead returns
unique int32 indices whose selected values match exactly, regardless of order.

| Problem | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| GEMM | .02 | .02 |
| Conv2d | .01 | .01 |
| FMHA | .01 | .005 |
| GDN | .01 | .001 |

Tolerances are experiment choices, fixed across methods and models. Timings
measure GPU activity with reused inputs and cold L2, excluding compilation and
allocation; they are not end-to-end application latency. No GPU clock locking
is assumed. Compare each final candidate with its contemporaneous reference.

The [GEMM contiguous-output comparison](KDA/astra/gemm/contiguous/README.md)
tests direct contiguous output and the original kernel followed by a timed
`.contiguous()` copy. Reproduce it with `baseline/compare_gemm_layout.py`.
