# KLineage

KLineage extracts reusable GPU optimization skills from validated kernel
lineages, then retrieves and applies them to a target kernel.

```text
repository → init → raw expert → decompose → lineage + SkillCards
                                                     │
new problem / current kernel → apply ← retrieve path ─┘
                                 │
                           code_gen → verify → replan
```

Code generation occurs during initialization, backward and forward
decomposition trials, and skill application. `code_gen` also exposes skill
materialization without validation.

## Setup

Requires Python 3.12, uv, Codex, Docker Compose, NVIDIA Container Toolkit,
and an NVIDIA GPU. CUDA timing requires CUPTI 13 or newer.

```bash
uv sync
codex login
```

Actions own their Codex sandbox and CUDA evaluator. Callers supply problems,
kernels, and skills.

## Workflow

```python
from pathlib import Path

from klineage.action import decompose, init
from klineage.memory import save_lineage

expert = init(
    "problems/gemm/reference.py",
    "https://github.com/NVIDIA/cutlass.git",
    Path("path/to/executable_expert.cu").resolve(),
)
lineage = decompose(expert)
save_lineage(lineage, "agent-workspace/lineage.json")
```

Apply the saved lineage to another problem file:

Set `OPERATOR = "gemm"` in that file to retain the operator identity across
different case names; otherwise it defaults to the problem name.

```python
from klineage.action import apply
from klineage.memory import load_lineage

lineage = load_lineage("agent-workspace/lineage.json")
new_problem = "path/to/new_gemm.py"
optimized = apply(new_problem, (lineage,))
```

- `init(problem, repo, expert_kernel)` snapshots inputs, discovers the tensor
  ABI, and accepts raw CUDA only after 99% performance and mechanism fidelity
  checks, followed by independent performance confirmation.
- `decompose(expert)` validates simpler predecessors and independently
  re-derived forward edits, then lifts transitions into SkillCards. A shared
  MMA few-shot defines semantic granularity; necessary fragment, layout, and
  thread-mapping changes belong together. Tile/vector-width tuning remains
  carrier choices and evidence, not repeated high-level skills.
- `retrieve_paths(lineages, current_or_context, problem=..., abi=...)` returns
  ranked plans with `lineage`, `entry_index`, and `skills`. It filters operator,
  hardware, and ABI compatibility, then follows applicable lineage subpaths.
- `apply(problem_path, lineages)` retrieves before generating a baseline.
  `apply(current, lineages)` starts from an existing kernel. Both generate,
  verify, and replan from the resulting state.
- `apply(current, skills)` retains explicit-card application; `retrieve(skills,
  context)` retains flat scope/prior-action filtering for existing callers.
- `code_gen(current, skills)` returns an unvalidated candidate.

Path retrieval checks `requires / provides / conflicts` against verified
`{locus, name}` features. History alone does not establish a mechanism. Plans
preserve source transitions, skip already-present features at the same locus,
and rank by operator similarity, shape proximity, and measured source gains.
Source gains guide search; they do not predict target speedup. Legacy cards
without feature contracts remain available through flat retrieval.

`Lineage.termination` distinguishes `complete`, `step_limit`, `rejection_limit`,
and `unknown`; `reason` records the stopping evidence. Limits do not establish
a naive kernel. Empty retrieval does not establish optimality.

`save_lineage` / `load_lineage` persist the complete lineage;
`save_memory` / `load_memory` persist a flat SkillCard pool.

## Admission

`SkillAdmission.OFF` is the default. Extracted cards are hypotheses usable
within a lineage. Enable admission to require held-out evidence:

```python
from klineage.memory import SkillAdmission, retrieve

lineage = decompose(
    expert,
    roundtrip_cases=(held_out_kernel,),
    effect_verifier=effect_verifier,
    skill_admission=SkillAdmission.ON,
)
cards = retrieve(
    lineage.skills, held_out_kernel.context, skill_admission=SkillAdmission.ON,
)
current = apply(held_out_kernel, cards, skill_admission=SkillAdmission.ON)
```

A card is admitted after a trial in a context absent from its source lineage
passes all evaluator gates and reproduces its declared effect according to
`effect_verifier`. Each held-out kernel uses its own trusted run and evaluator.

## Contracts and validation

A problem file defines `make_inputs()` and `torch_ref()`. A repository may be
a local directory or Git URL. The expert must be an executable `.cu` wrapper,
identified by a path inside the repository or an absolute local path.
Headers and uninstantiated templates provide context, not executable experts.

Every kernel carries a `ProblemSpec`, `KernelABI`, and `TargetContext`.
Generation receives the contract and source once; full validation evidence
stays in the saved lineage and evaluator logs.

Raw CUDA uses the `klineage_launch` ABI in `klineage.contract`. Source bundles
use `python-callable-v1`: import the declared module, call its zero-argument
loader outside timing, then invoke the returned callable with ABI inputs in
order. A single output is returned directly; multiple outputs use a tuple.
Python supplies build and binding glue; CUDA targets require kernel code.
CUDA bundles build in the loader, validate ABI metadata, and launch on the
input device's current stream. Every call recomputes its outputs.

Correctness uses the problem reference. Timing uses FlashInfer's CUPTI activity
timer; loading, compilation, and input creation occur outside the timed region.
Performance fidelity requires matching timing policies and
`expert_ms / candidate_ms >= 0.99` overall and in every trial. Init repeats the
check on the frozen candidate in a fresh worker before acceptance. This gate
covers only the declared case, expert instance, hardware, and timing policy.

Mechanism fidelity separately audits computation, tiling, pipeline, layout,
and scheduling against the instantiated expert and library definitions. Source
citations use bounded line ranges; the verifier restores exact quotes and
hashes locally. This is a semantic audit,
not a formal proof; changed, unknown, or unsupported findings fail the gate.

Runner failures retry the frozen candidate's audit up to three times, without
regenerating its code. Init atomically saves `checkpoint-NN.json` (expert,
candidate, selection evidence) and `fidelity-NN.json` before auditing and after
each result. Pending checkpoints remain unaccepted until independent confirmation.
Exhausted retries raise `ValidationGateError` with the candidate attached.
Checkpoints preserve evidence; there is no public cross-process resume API.

## Isolation

Runs live under `agent-workspace/runs/`. Each action has separate writable
`work/`, committed `artifacts/`, and evaluation logs. Problem and repository
snapshots are read-only. Later actions verify contracts and artifact contents
before resuming a run.

Codex commands and fresh CUDA evaluator containers have no network access.
Only declared inputs are mounted. Source bundles reject symbolic links and
unsafe paths, and are limited to 1024 files and 16 MiB.

## Tests

```bash
uv run python -m unittest discover -s tests -q
```

PyTorch-dependent tests also run in the CUDA image with GPU access disabled.
