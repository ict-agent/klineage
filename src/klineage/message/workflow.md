# Workflow

`klineage-workflow` runs Init, Decompose, then Apply optimization rounds.
`klineage-init-memory` runs only Init and Decompose.
`klineage-optimize` runs Apply rounds from an existing kernel, with optional memory.
All support CUDA, Hygon HIP, and AscendC; optional counter profiling requires CUDA/NCU.

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

Workflow additionally accepts `--max-apply-step`, a nonnegative optimization round limit
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
    kernel.json                # Best validated kernel from this round
    submission/
    evaluations/               # Trial measurements and optional profiles
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
| Apply j | Apply j+1 | Current kernel directory and the same optional memory |

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

Workflow supplies the collected memory directory, even when empty. Within a round,
Apply may combine techniques, adapt relevant recipes, and iterate on measured
results. It checks prerequisites against each current implementation. Memory is
guidance; an empty directory or no applicable recipe does not end the round.
Prior recipe use is not automatically excluded from later rounds.

An unchanged kernel ends the apply loop only with its name and validation preserved
and no SKILL.md or submission/. Changed kernels continue the loop without a card.
Handoffs preserve the problem and ordered ABI, including when verification is
disabled. Each round self-checks with its fixed input as the paired reference and
returns its best validated kernel. Counter profiling is optional. The limit counts
rounds, including an unchanged round; retries stay within a round. Failures propagate.

## Result

`klineage-workflow` writes the latest `Kernel.to_dict()` as JSON to stdout,
including an unchanged terminal Apply result when present. With zero applies,
it returns the final Decompose kernel. The Python function returns that Kernel.
Profiling and measurement evidence stays in the Apply workdir's evaluations/.
No root-level kernel is produced; selection among trials happens inside each round.

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

Omit `--memory_dir`, or pass a blank value, for independent baseline rounds.
Apply emits no SKILL.md in either mode. Changed kernels continue
the loop; an evidence-backed unchanged result stops it. Both preserve the problem and
ordered ABI; terminal results also preserve name/validation and have no submission/.
Counter profiling is optional; changed kernels require paired self-checks.

With `--memory_dir`, use the same memory guidance and handoffs as workflow. Memory
contains SKILL.md files and is mounted at each Apply workdir's `.agents/skills/memory`;
treat it as read-only. The workdir must be outside memory. An existing empty directory
is mounted but supplies no recipes; optimization continues from source analysis.
`--start-kernel` and `--memory-dir` are equivalent flag spellings.

`--max-apply-step` limits rounds, defaults to 15, and includes unchanged rounds.
Verification defaults to on;
`--verifier` / `--no-verifier`, `--timeout`, and `--max-retries` behave as above.
Stdout contains the latest kernel as JSON; zero steps preserves the input.
The module entry point is `python -m klineage.cli.optimize`.
