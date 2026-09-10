# Decompose

[Shared message contract](README.md). One invocation removes one optimization.

## Input

| Parameter | Format |
| --- | --- |
| `input_kernel` | Absolute kernel directory or inline Kernel |

The workflow starts from Init and passes each simplified kernel to the next session.
Sessions do not inherit conversations. Source is authoritative; prior runner responses
can explain previous choices.

## Output

```text
<workdir>/
  kernel.json                  # Complete simplified kernel, or unchanged naive input
  submission/                  # Present for a removal
  SKILL.md                     # One forward optimization; present for a removal
  evaluations/                 # Self-checks and independent reconstruction evidence
```

A removal always emits the complete bundle, kernel.json, and exactly one SKILL.md,
even when its output is naive. An already naive input emits the unchanged kernel
with no SKILL.md or submission/. Remove stale local proposals on retries. Missing
SKILL.md is a valid stop signal for an unchanged naive input after self-checks.
When enabled, the external verifier must also confirm this case. Without it,
the stop is self-reported, not independently verified. A changed kernel without
a card, or an unchanged kernel with a card, is an error. Both workflow CLIs reject
these cases, missing kernel.json, and changed problems even with the
external verifier disabled.

The final response records the removed optimization and locus, remaining
mechanisms and dependencies, a rough remaining-step estimate, and a possible next
removal, or a source-grounded explanation of naive status. Reinspect each output;
estimates may change. Retries and unchanged checks do not add successful removals.
Budget exhaustion does not prove that a kernel is naive.

## SkillCard format

Use `klineage.memory.save_skill` and `klineage.memory.load_skill` for SKILL.md.
YAML frontmatter contains exactly skill_id, intent, preconditions,
and scope. The independent body
contains `# Overview`, `# Precondition`, `# Scope`, and `# Code Change Snippet`, with
`## Before` and `## After` code blocks showing the forward optimization.
SkillCard.to_dict() carries the four metadata fields plus body for inline handoffs.
Keep SKILL.md outside submission/ and Kernel.source_files.

| Field | Format |
| --- | --- |
| `skill_id` | Nonempty stable identifier |
| `intent` | One brief sentence describing the removed optimization technique |
| `preconditions` | Nonempty string array of requirements before applying the optimization |
| `scope` | `cases`, `languages`, `platforms`: nonempty string arrays |
| `body` | Independent Markdown instructions and code; outside YAML frontmatter |

Preconditions cover data types; layout, contiguity, and data-to-parallel-unit
mapping; storage hierarchy placement; pipeline stages, buffers, and ordering; and
required hardware features of the actual backend. State concrete
minimum requirements and explain their dependencies. Put example configuration
in Overview / Example configuration: fixed dimensions, stage counts, launch sizes,
and retained mechanisms
are not prerequisites without a reason the transformation needs them.
Keep metadata and prose consistent. Changes introduced
by the optimization belong in Overview and After, not prerequisites.
Apply this rule to both YAML and the entire Precondition section. Prose expands
the same dependency set; it must not add preserved instance settings. Each
requirement needs an operation or hardware rule in the removed technique that
depends on it. Keep all five categories explicit in YAML and prose: Data types,
Layout, Storage, Pipeline, and Hardware. If a category has no additional requirement,
say so and explain why. Pipeline covers producer/consumer ordering, visibility, and
safe buffer reuse, including preserved synchronization; fixed stage counts require
a dependency. Record this semantic audit in evaluations/audit.md; serialization
roundtrip alone is insufficient.

Body explains how to restore the removed optimization from the deoptimized kernel
alone, including necessary implementation details. Intent contains no recipe or code.
When enabled, the verifier checks input and predecessor independently, reconstructs using only
the predecessor, card, problem, and output directory, and compares reconstruction
against input with the harness. It attaches predecessor validation to kernel.json.
SKILL.md stays unchanged; measurements and source citations stay in evaluations/.

## Handoff

Pass this directory to the next Decompose or Apply. Collect emitted SKILL.md files
in a memory directory for Apply. Consume artifacts after the action succeeds and, when enabled,
its verifier succeeds. Disabled verification does not disable required self-checks.
