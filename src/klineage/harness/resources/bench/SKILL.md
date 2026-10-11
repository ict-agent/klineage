---
name: bench
description: Evaluate Kernel artifacts, compare implementations, and inspect performance with klineage.evaluate.
---

# Bench

## Evaluate a candidate

```python
from pathlib import Path
import klineage

reply = klineage.evaluate(Path("current_kernel"))
print(reply.accepted)
print(reply.diagnostics)
for workload_id, result in reply.validation.workloads.items():
    print(workload_id, result.latency_ms, result.reference_latency_ms)
```

`evaluate` accepts a Kernel artifact or its directory. It freezes the sources,
builds the kernel, checks correctness and measures every workload. It uses the
configured evaluator when present, otherwise the local backend. A configured
evaluator's failure is reported without switching to local execution.

`reply` contains the measured `kernel`, `validation`, `accepted`, `diagnostics`
and `job_id`. Check `accepted` before retaining a candidate. Read diagnostics for
compile failures, incorrect outputs and timing failures. Completed workloads
keep their results when another workload fails.

Each result, including a failed measurement, is saved under `versions/<N>/` in
the current work directory. Its `metadata.json` keeps per-workload measurements
and diagnostics. Keep candidates in `current_kernel/` and supporting files in
`workspace/`. Submit again after source edits; saved versions stay unchanged.

## Read the problem and target

Use `klineage.problem()` for the evaluation Problem and `klineage.info()` for
backend, device and toolchain information. When several problems are available,
pass the assigned ID to `klineage.problem(problem_id)`. Kernel metadata retains
the original problem directory reference; evaluate selects the matching task.

Preserve the definition, every workload and all fixed inputs. A Problem without
workloads cannot be evaluated. Query its layout and the backend's kernel layout
through the Python interfaces described in `workspace/AGENTS.md`.

## Compare implementations

```python
baseline = klineage.evaluate(input_kernel)
reply = klineage.evaluate(Path("current_kernel"), reference=baseline)
```

A reference may be a Kernel, directory or earlier Evaluation. It supplies the
performance baseline; the Problem supplies the correctness oracle. The paired
measurement uses the same backend and timing policy for both implementations.
Compare individual workloads; multi-workload summaries have no combined latency.

For Apply, keep the input as the reference throughout the round. A changed
implementation must improve at least one workload and pass every workload's
performance gate. Init compares against its fixed adapter in `workspace/expert/`.
Generation and Decompose evaluate without a performance reference.

To finish with an earlier accepted result:

```python
reply = klineage.result(best_reply)
assert reply.accepted, reply.diagnostics
```

This appends the measured Kernel as the latest version without evaluating it
again. The action returns the largest numbered version; a failed latest version
must be corrected or replaced by an accepted result.

## Inspect performance

Get a correct, measured implementation before spending the remaining budget on
profiling. With available devices and tools, use `ncu` or the relevant HIP/Ascend
profiler to investigate a specific question. Put scripts, binaries and reports
in `workspace/`. `Kernel.load_kernel()` can build a callable for local diagnostics.

Run diagnostics and evaluation sequentially on a device. If a profiler is absent
or lacks counter permissions, retain the error and use other available evidence.
Submit changes through `klineage.evaluate`; its measurements determine acceptance.
`profile_passed` in validation reports the timing gate.
