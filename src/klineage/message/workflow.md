# Workflow

The Profile/Retrieve loop requires CUDA counter capture. `init_memory` runs only
Init and Decompose and supports CUDA, Hygon HIP, and AscendC.

[Shared message contract](README.md). Workflow orchestrates actions; it is not an
Action subclass and does not emit a separate LLM message.

## Input

`workflow(problem, repo, expert_kernel, ...)` accepts Init's problem/repository
inputs and a fresh `workdir`. `max_decompose_step` is positive; `max_apply_step`
is nonnegative. Both default to 15. `enable_verifier` defaults to `True`; each action
must pass external verification before handoff. Disabling it skips that verifier;
action self-checks and deterministic Decompose handoff checks still apply.

## Directory layout

```text
<workdir>/
  init/
    kernel.json
    submission/
  decompose/<i>/
    kernel.json                # One step's predecessor
    SKILL.md                   # One independently reusable optimization
    submission/                # Present when this step changes code
  profile/0/
    kernel.json                # Profile of the final decomposition state
    evaluations/ncu-<id>/
  retrieve/<j>/
    SKILL.md                   # One selected card, absent for an empty selection
  apply/<j>/
    kernel.json                # Result of that one selected skill
    submission/
  profile/<j+1>/
    kernel.json                # Fresh profile after Apply
    evaluations/ncu-<id>/
```

Every action directory also carries the runner records described in the shared
contract. Loop indices start at zero; only executed stages exist.

## Handoffs

| Sender | Receiver | Message |
| --- | --- | --- |
| Init | Decompose 0 | `init/` |
| Decompose i | Decompose i+1 | `decompose/i/`; read `kernel.json` |
| Final Decompose | Profile 0 | Final decomposition directory |
| All Decompose steps | Each Retrieve | Array of emitted SKILL.md file paths |
| Profile j | Retrieve j | `profile/j/`; read kernel and referenced NCU evidence |
| Retrieve j + Profile j | Apply j | Selected SKILL.md path plus current kernel directory |
| Apply j | Profile j+1 | `apply/j/` |

Decompose stops when an accepted invocation preserves the naive input and emits
no SKILL.md, or at its run limit. Count successful removals by emitted cards, not
retries or the unchanged terminal check. Each invocation is a fresh Codex run.
A removal that reaches naive still emits a card; a subsequent invocation confirms
that no optimization remains. Exhausting the limit never establishes naive status.
Progress and remaining-mechanism estimates stay in the existing runner responses.

Before accepting each Decompose result, the workflow requires kernel.json and the
unchanged problem. A changed kernel requires SKILL.md; an unchanged kernel must
have no card and preserve its name. These checks also run with verification
disabled. They establish artifact consistency, not naive status or correctness.

Retrieval excludes all previously applied skill IDs. An accepted selection without SKILL.md stops the apply loop;
otherwise one Apply and one Profile run before the next retrieval. The apply limit
counts applied skills, not verification attempts. A stage failure stops the workflow.

## Result

The final artifact is the latest `profile/<n>/kernel.json`, with its complete source_files; NCU evidence
resides beside it under evaluations/. `workflow()` loads and returns that Kernel. With zero
applies it returns `profile/0/kernel.json`. It does not write a root-level kernel,
merge skill histories, or choose a historical best candidate.

## Memory initialization

`init_memory(problem, repo, expert_kernel, max_decompose_step=15,
enable_verifier=False, *, workdir=None, timeout=3600, max_retries=3, memory_dir=...)`
runs Init and Decompose with the same handoffs and stop rules. `memory_dir` is
required; `workdir` must be fresh and defaults to a generated directory.
External verification is disabled by default; action self-checks and the handoff
checks above still run.

Each successful removal saves its card to
`memory_dir/<skill-id>-<timestamp>-<unique-suffix>/SKILL.md`; directory names are
sanitized. Existing cards and repeated skill IDs are preserved. Later failures
propagate after keeping cards from successful steps. The function returns a tuple
of saved paths in removal order; a naive input returns an empty tuple.
