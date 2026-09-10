---
name: bench
description: Measure KLineage kernel correctness and performance on CUDA, Hygon HIP, or AscendC; compare candidates with an original expert or baseline and inspect available diagnostics.
---

# Bench

## Entry point

Use the existing Kernel evaluator for every correctness and latency check:

```python
from pathlib import Path
from klineage.artifact.kernel import load_kernel
from klineage.harness import evaluate

work = Path.cwd()
candidate = load_kernel(work)
result = evaluate(candidate, work)
assert result.accepted, result.to_dict()
```

`evaluate(kernel, work, *, reference=None, include_paths=())` builds frozen sources
in a worker, then calls `CallableKernelEvaluator` with the problem's backend timer.
`Kernel.problem` supplies the workload and correctness oracle. The existing
checker handles output contracts, dtype tolerances, and Top-K selection semantics.
Ascend's tensor ABI accepts ND, NCHW, NHWC, and NCDHW base formats. Convert blocked,
reordered, or unknown formats to ND before defining the workload; logical strides
alone do not describe their physical storage.
Inspect recorded requests for the selected timer and policy:

- CUDA requires CUPTI 13 or newer. `FlashInferCuptiTimer` measures the span from
  the first correlated device activity's start to the last activity's end per call.
  Activities include kernels, copies, and memset. Gaps within the span count;
  overlapping activities count once. Cold L2 is enabled by default.
- Hygon uses HIP events; Ascend uses NPU events. These measure the current stream
  interval around the callable, including dispatch gaps. Cold L2 is disabled;
  requesting it on these backends is unsupported.

Compare measurements only within the same backend and policy.
Do not construct another runtime, checker, or timer in the action.

Reconstruct Kernel from final sources after edits. Building or editing disk files
does not update an already serialized source map. Preserve the complete problem
and referenced input files. Path arguments are `pathlib.Path` objects.

## Comparisons

Omit `reference` for standalone correctness and timing. Decompose evaluates input
and deoptimized kernels separately; it imposes no relative speed floor.

Use `evaluate(candidate, work, reference=baseline)` when the action requires a
paired performance gate. The oracle remains the problem reference. The supplied
Kernel reference is the timing baseline. Both use the same worker, backend, and timing policy;
the gate requires baseline_ms / candidate_ms >= 0.99 overall and in every trial.

For Init, resolve `backend = klineage.backend.get_backend(candidate.problem.language,
candidate.problem.platform)`. Freeze an original expert adapter under
`expert/<backend.raw_source>`, implementing `backend.raw_abi`. Construct its Kernel
using the candidate's problem and `{backend.raw_source: adapter.read_text()}`.
CUDA uses a CUDA stream, HIP a HIP stream, and Ascend an ACL stream.
Pass required repository headers through `include_paths`.
First evaluate that expert alone, then evaluate the candidate with
`reference=expert`. The adapter must call the fixed expert on the supplied stream.

## Results and evidence

`ValidationResult` contains `compile_passed`, `correctness_passed`, `profile_passed`,
`latency_ms`, and `reference_latency_ms`. `accepted` requires all three gates.
`profile_passed` means timing and any requested performance gate passed; it does
not mean NCU ran. Missing measurements and failed gates are failures.

Each call writes evaluations/evaluate-<id>/ with eval-0001-request.json,
eval-0001-result.json, stdout/stderr logs, and eval-0001-process.json.
The request freezes sources, problem, and policy. Candidate samples appear in the
stderr `Timing evidence:` record. Paired samples, trial ratios, and rejection
reasons appear in `Performance gate:`. Invalid timing evidence fails evaluation,
including standalone runs.
Read these records before changing code; preserve failed observations.

Finish when the action's objective and planned evidence-guided trials are complete,
required checks pass, and final artifacts agree. Repeat checks only after relevant
edits, failed checks, or new evidence. Do not tune a deoptimized kernel
to force a slowdown, change workload/tolerances, or replace the evaluator.
Self-checks remain required when the action disables external verification;
leave changed Kernel.validation unset. External verification runs only when enabled.

`inspect_problem(problem_path, work)` resolves an input ProblemSpec.
For CUDA NCU diagnostics:

```python
from klineage.harness.profiling import ProfileOptions
from klineage.tools import profile

diagnostics = profile(candidate, work, options=ProfileOptions())
```

Hardware-counter profiling is unsupported on Hygon and Ascend; report that
capability error without substituting event timings for bottleneck metrics.
Neither replaces `evaluate` for correctness or latency.

Adapted from [AKO4X bench](https://github.com/TongmingLAIC/AKO4X/tree/c8fd2777d5387fa10563c704a5beeddba53f0411/templates/skills/bench),
MIT; see [LICENSE](LICENSE). KLineage evaluates one supplied workload through Python
interfaces; AKO4X scripts, Modal, filters, labels, and scoring are not available here.
