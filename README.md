# KLineage

KLineage extracts reusable GPU optimization skills from validated kernel
lineages, then retrieves and applies them to a target kernel.

```text
repository → init → expert → decompose → naive + SkillCards
                                          │         │
                                          │      retrieve
                                          │         │
                                          └── apply ←┘
                                                │
                                         validated kernel
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

from klineage.action import apply, decompose, init
from klineage.memory import retrieve, save_lineage

expert = init(
    "problems/gemm/reference.py",
    "https://github.com/NVIDIA/cutlass.git",
    Path("experiments/cutlass_gemm/sm80_bf16_gemm.cu").resolve(),
)
lineage = decompose(expert)
save_lineage(lineage, "agent-workspace/lineage.json")

current = lineage.naive_kernel
pending = lineage.skills
while cards := retrieve(pending, current.context):
    current = apply(current, cards)
    applied = {card.skill_id for card in cards}
    pending = tuple(card for card in pending if card.skill_id not in applied)
```

- `init(problem, repo, expert_kernel)` snapshots inputs, discovers the tensor
  ABI, and generates raw CUDA reaching at least 95% of the expert's performance.
- `decompose(expert)` validates simpler predecessors and independently
  re-derived forward edits, then lifts accepted transitions into SkillCards.
- `retrieve(skills, target)` filters case, language, platform, and prior-action
  prerequisites. It preserves memory order and deduplicates skill IDs. Scope
  dimensions match exact values or `"*"`; semantic preconditions remain for
  the generator to check.
- `apply(current, skills)` generates once and enforces the compile,
  correctness, and profiling gates against the current kernel.
- `code_gen(current, skills)` returns an unvalidated candidate.

Remove successfully applied IDs from the pending pool before retrieving
again. One action category can apply at several code locations. An empty
retrieval means no remaining card currently matches; it does not prove the
kernel is fully optimized.

`save_lineage` / `load_lineage` persist the complete lineage;
`save_memory` / `load_memory` persist a flat SkillCard pool.

## Admission

`SkillAdmission.OFF` is the default. Extracted cards are hypotheses usable
within a lineage. Enable admission to require held-out evidence:

```python
from klineage.memory import SkillAdmission

lineage = decompose(
    expert,
    roundtrip_cases=(held_out_kernel,),
    effect_verifier=effect_verifier,
    skill_admission=SkillAdmission.ON,
)
cards = retrieve(
    lineage.skills, current.context, skill_admission=SkillAdmission.ON,
)
current = apply(current, cards, skill_admission=SkillAdmission.ON)
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

Correctness uses the problem reference. Timing uses FlashInfer's CUPTI activity
timer; loading, compilation, and input creation occur outside the timed region.

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
The existing CUTLASS initialization example is
`experiments/cutlass_gemm/run_init.py`.
