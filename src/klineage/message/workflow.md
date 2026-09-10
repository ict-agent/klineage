# Workflow

`klineage-workflow` runs Init, Decompose, then repeated Apply. Memory selection with
eligible candidates requires CUDA counter capture. `klineage-init-memory` runs
only Init and Decompose and supports CUDA, Hygon HIP, and AscendC.
`klineage-optimize` runs repeated Apply from an existing kernel, with optional memory.

[Shared message contract](README.md). Workflow orchestrates actions; it is not an
Action subclass and does not emit a separate LLM message.

## Input

Run `uv run klineage-workflow` or `uv run klineage-init-memory` with:

| Option | Meaning / default |
| --- | --- |
| `--problem` | Required problem definition |
| `--repo` | Required expert repository |
| `--expert-kernel` | Required expert path or locator within the repository |
| `--max-decompose-step` | Positive Decompose invocation limit; 15 |
| `--workdir` | Fresh directory; generated when omitted |
| `--timeout` | Per-action timeout in seconds; 3600 |
| `--max-retries` | Retries after the initial attempt; 3 |
| `--verifier` / `--no-verifier` | External verification; enabled for workflow, disabled for memory initialization |

Workflow additionally accepts `--max-apply-step`, a nonnegative Apply invocation limit
defaulting to 15. Memory initialization requires `--memory-dir`.
Disabling verification preserves action self-checks and deterministic handoff checks.

Module entry points are `python -m klineage.cli.workflow` and
`python -m klineage.cli.init_memory`. The respective modules also expose callable
`workflow` and `init_memory` functions.

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
  memory/<i>/
    SKILL.md                   # Collected Decompose card
  apply/<j>/
    kernel.json                # Result of that one selected skill
    SKILL.md                   # Selected unchanged card, absent for empty selection
    submission/
    evaluations/ncu-<id>/
    .agents/skills/memory       # Mounted memory directory
```

Every action directory also carries the runner records described in the shared
contract. Loop indices start at zero; only executed stages exist.

## Handoffs

| Sender | Receiver | Message |
| --- | --- | --- |
| Init | Decompose 0 | `init/` |
| Decompose i | Decompose i+1 | `decompose/i/`; read `kernel.json` |
| Final Decompose | Apply 0 | Final decomposition directory |
| All Decompose steps | Each Apply | Cards collected in `<workdir>/memory/`, mounted into each Apply |
| Apply j | Apply j+1 | Current kernel directory, same memory, prior selected IDs excluded |

Decompose stops when an accepted invocation preserves the naive input and emits
no SKILL.md, or at its invocation limit. An unchanged terminal check consumes one
invocation; retries stay within it. Count successful removals separately by emitted
cards. Each invocation is a fresh Codex run.
A removal that reaches naive still emits a card; a subsequent invocation confirms
that no optimization remains. Exhausting the limit never establishes naive status.
Progress and remaining-mechanism estimates stay in the existing runner responses.

Before accepting each Decompose result, the workflow requires kernel.json and the
unchanged problem and ordered input/output ABI. A changed kernel requires SKILL.md;
an unchanged kernel must preserve its name and have no SKILL.md or submission/.
These checks also run with verification
disabled. They establish artifact consistency, not naive status or correctness.

Workflow always uses the collected memory directory, even when empty. Apply calls
`klineage.agent_tools.retrieve` to filter declared scope and excluded IDs.
If candidates exist, it calls `profile` for fresh counters, checks textual
prerequisites and existing mechanisms, then selects and applies top-1.
Both are ordinary Python calls.

An unchanged kernel ends the apply loop only with its name and validation preserved
and no SKILL.md or submission/. A changed kernel requires an unchanged card from
memory whose ID was not excluded. Handoffs preserve the problem and ordered ABI,
including when verification is disabled. A missing or unsupported profile is an
error, not an empty selection. The limit counts Apply invocations, including the
terminal check; retries stay within an invocation. Stage failures stop the workflow.

## Result

`klineage-workflow` writes the latest `Kernel.to_dict()` as JSON to stdout,
including an unchanged terminal Apply result when present. With zero applies,
it returns the final Decompose kernel. The Python function returns that Kernel.
Profiling and measurement evidence stays in the Apply workdir's evaluations/.
No root-level kernel or historical-best selection is produced.

## Memory initialization

`klineage-init-memory` runs Init and Decompose with the same handoffs and stop
rules. External verification is disabled by default; action self-checks and the
handoff checks above still run.

Each successful removal saves its card to
`memory_dir/<skill-id>-<timestamp>-<unique-suffix>/SKILL.md`; directory names are
sanitized. Existing cards and repeated skill IDs are preserved. Later failures
propagate after keeping cards from successful steps. Stdout contains a JSON array
of saved path strings in removal order; a naive input produces `[]`. The Python
function returns those paths as a tuple.

## Optimization

Run `uv run klineage-optimize` with required `--start_kernel` and `--workdir`.
Start with a `kernel.json` file or its directory, containing the complete problem
and source bundle. Preserve referenced workload files. The workdir must not exist.

Omit `--memory_dir`, or pass a blank value, for baseline optimization. Each Apply
independently chooses an optimization and emits no SKILL.md. Changed kernels continue
the loop; an evidence-backed unchanged result stops it. Both preserve the problem and
ordered ABI; terminal results also preserve name/validation and have no submission/.
Counter profiling is optional; changed kernels require paired self-checks.

With `--memory_dir`, use the same memory selection and handoffs as workflow. Memory
contains SKILL.md files and is mounted at each Apply workdir's `.agents/skills/memory`;
treat it as read-only. The workdir must be outside memory. An existing empty memory
directory produces an unchanged terminal result; it does not enable baseline mode.
`--start-kernel` and `--memory-dir` are equivalent flag spellings.

`--max-apply-step` limits invocations, defaults to 15, and includes terminal checks.
Verification defaults to on;
`--verifier` / `--no-verifier`, `--timeout`, and `--max-retries` behave as above.
Stdout contains the latest kernel as JSON; zero steps preserves the input.
The module entry point is `python -m klineage.cli.optimize`.
