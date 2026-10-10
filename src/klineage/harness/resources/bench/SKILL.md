---
name: bench
description: Submit kernel sources to KLineage's evaluation service; inspect correctness, timing, and fixed-baseline comparisons on CUDA, Hygon HIP, or AscendC.
---

# Bench

## Service entry

The evaluation service supplies official correctness and timing evidence.
Read `Client().info()["problems"]` and set `problem_id` to the assigned task.
Submit each candidate through that task's client:

```python
from pathlib import Path
from klineage.server import Client

client = Client(problem_id=problem_id)
info = client.info()
problem = client.problem()
reply = client.evaluate(Path("kernel"), Path.cwd())
assert reply.accepted, reply.diagnostics
```

`Client()` reads `KLINEAGE_SERVER`; `KLINEAGE_PROBLEM` is a server-side directory
setting. An unbound client can choose explicitly per request; only a single
registered problem is selected automatically. Keep the service, its environment and registered
problem fixed. Do not start a server, modify its configuration, change the oracle,
inputs, tolerance or timing policy. Use `client.info()` for the server's device
and toolchain; local device detection describes only the agent's supplied runtime.

Author `kernel/metadata.json`, `kernel/submission/config.toml` and complete sources
under `kernel/submission/solution/`. Follow the backend skill. Preserve name and
compile_flags when editing an input; clear stale validation. Pass the directory
as a `Path`; the client reads text without compiling. The server reconstructs the
Kernel from the registered problem and checks the backend spec before building.
The problem remains in its original directory, linked under problems/<problem_id>.
Submission does not copy input binaries.
`problem.workloads` contains every registered case, identified by a unique UUID.
Keep the entire definition and workload set fixed; never reduce it to the first
case or clone a separate problem for each shape. Definition-only artifacts are
valid, but cannot pass evaluation until the controller supplies workloads.

`client.evaluate(kernel, work, reference=None)` accepts a Kernel artifact or directory.
Each request freezes the source text and compile flags. Re-submit after source edits.
The client does not substitute a local result if the service fails.

## Performance inspection

Submit an initial implementation before optional local performance diagnosis.
Retain an accepted service result before spending the remaining budget on tuning.

With explicitly supplied devices, runtimes and tools, write diagnostic scripts
and invoke vendor tools directly, such as CUDA `ncu` or the platform's HIP/CANN
profiler. Local builds and exploratory checks may support that investigation.
KLineage provides no dedicated profiling API or prescribed metric set.

For local builds, prefer `Kernel.load_kernel(Path("kernel"))` from
`klineage.artifact`; it applies the backend's build settings and returns a callable
with the problem's input ABI. Reuse it across diagnostic calls. The controller
mounts the supplied problem directory read-only. Fixed inputs outside that directory
may need explicit mounts; otherwise generate diagnostic inputs with matching shapes,
dtypes and constraints.
Choose a case explicitly with `problem.values(role, workload=case)` when preparing
diagnostic inputs. Without a selected case, zero or multiple workloads retain
symbolic axes. Direct tensor adapters bind those axes from inputs; output-only
axes without bindings raise an error.

Keep scripts, commands, tool versions and reports in the workspace. Select
observations that answer the current performance question. Diagnostic timings
are not acceptance timings and cannot fill validation or replace service results.
Submit the final source files with `Client.evaluate` after inspection.

Do not run diagnostics and service benchmarks concurrently on the same device.
Wait for submitted evaluation jobs to finish before using that device directly;
stop diagnostic processes before submitting the next measurement.
GPU access alone does not establish performance-counter permission. If a tool
reports missing permissions, retain the error and continue with available evidence;
do not alter host driver permissions.

## Comparisons

Omit `reference` for standalone correctness and timing. Decompose measures its
input and deoptimized candidate separately, without a relative speed floor.
Apply compares every trial against the fixed input when it has sources.

`reference` accepts a Kernel, directory, accepted job ID, or Evaluation. Each
workload's paired measurement uses one worker, backend and policy. Workloads run
in separate workers under one shared deadline. The oracle remains the registered
problem reference; the reference Kernel is the performance baseline. Each case's
gate requires baseline_ms / candidate_ms >= 0.99 for its reduced latency and in
every trial. All cases must pass. For a changed output, Apply's prompt additionally
requires at least one workload's ratio to exceed 1; inspect each case independently
without averaging shapes. The host checks matching service results and the per-case
gates; the stricter improvement objective remains an agent responsibility.

For Init, author an original-expert adapter under `expert/metadata.json` and
`expert/submission/{config.toml,solution/}`. Its host entry calls the fixed expert,
preserves its schedule, and uses the caller stream. It may use controller-registered
repository headers; the candidate must be standalone. Only the controller registers
include paths. Preserve the upstream release macros, optimization settings,
assertion mode and architecture in metadata.json compile_flags.

```python
expert_reply = client.evaluate(Path("expert"), Path.cwd())
assert expert_reply.accepted, expert_reply.diagnostics
reply = client.evaluate(Path("kernel"), Path.cwd(), reference=expert_reply.job_id)
assert reply.accepted, reply.diagnostics
```

Keep the chosen expert instance and adapter fixed throughout the comparison.
Use the service's compiler diagnostics to confirm its build settings. Successful
compilation alone does not prove the intended specialization or pipeline.

## Evidence and handoff

`Evaluation` returns `job_id`, `kernel`, `validation`, `accepted`, and `diagnostics`.
Its `ValidationResult` contains `compile_passed`, `correctness_passed`,
`profile_passed` and a `workloads` map of individual results keyed by UUID. Each
case retains its own `latency_ms` and `reference_latency_ms`; top-level latency
fields repeat a sole case and are `None` for multiple cases. There is no combined
cross-shape latency. Acceptance requires all three gates in every case.
Check `reply.accepted`: a job-level error can reject the job even when
`reply.validation.accepted` is true, for example when results arrive after the
total deadline. Retained measurements do not erase that error.
`profile_passed` describes valid timing and the performance gate;
it does not mean NCU ran. Failed or missing measurements are not successful results.

```python
for workload_id, result in reply.validation.workloads.items():
    print(workload_id, result.accepted, result.latency_ms, result.reference_latency_ms)
```

Completed cases remain in the report when another fails. A case that exhausts
the shared deadline, or cannot start before it, remains a failed entry.
Request preparation, fixed-input hashing and sandbox setup consume the same
budget. An accepted job must finish within it; later cases do not reset the clock.

Each saved Kernel's `metadata.json` contains `validation.workloads` and
`evaluation = {job_id, error, diagnostics}`. Read `reply.validation` and
`reply.diagnostics` or these fields for compiler failures, mismatch counts, error
indices and timing evidence. The client links `problems/<problem_id>` to the
original problem directory. Read diagnostics before changing code; failed trials
remain as saved versions.
Server errors that occur before a job exists raise `ServerError`.

Each `evaluate` or `result` call with `work` automatically appends the complete
returned Kernel under `kernels/0`, `kernels/1`, ... including failed results.
Do not save Kernels manually or edit these versions. The controller checks the
largest numbered version against its service record. A failed latest version
does not fall back to an earlier accepted one.

To finish with an earlier best trial:

```python
reply = client.result(best_job_id, work=Path.cwd())
assert reply.accepted, reply.diagnostics
```

This appends that measured Kernel as a new latest version with its diagnostics in metadata.json.
`kernel/` is independent trial source space; no source restoration or cleanup is required.
Evaluating an unchanged terminal input also saves its complete Kernel automatically,
with the same name and sources and fresh validation. Emit no SkillCard for that result.

CUDA timing uses CUPTI 13+ device activity spans, including kernels, copies,
memsets and intervening gaps; cold L2 is enabled by default. Hygon uses HIP events
and Ascend uses NPU events on the caller stream; their defaults disable cold L2.
Compare only within the same backend and policy. The numerical gate uses
`rtol=1e-2`, `atol=1e-2`, plus shape, dtype and problem-specific invariant checks.
Ascend supports ND, NCHW, NHWC and NCDHW base tensor formats; incompatible physical
formats must be addressed in the registered problem before starting a campaign.

Finish after the action's checks pass and its artifacts agree. Repeat measurements
only after relevant edits, failures or new evidence. Do not tune a deoptimized
kernel to force a slowdown or replace unavailable profiler evidence with guesses.
