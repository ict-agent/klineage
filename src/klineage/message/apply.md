# Apply

[Shared message contract](README.md). Apply one optimization, independently or from memory.

## Input

| Parameter | Format |
| --- | --- |
| `current_kernel` | Absolute directory containing kernel.json, or inline Kernel |
| `memory` | Optional directory; mounted absolute path in the prompt when supplied |
| `exclude_skills` | Skill IDs excluded from memory selection |

With `memory=None`, an empty string, or whitespace, Apply runs a baseline step:
choose one optimization from source and target hardware, implement it, and emit no
SKILL.md. The prompt omits memory and exclusions; memory instructions below do not
apply. Counter profiling is optional. If no proposed optimization passes the checks,
preserve the exact input without SKILL.md or submission/ and explain actual attempts
and source/measurement evidence. Tool failures or exhausted budgets alone do not
justify stopping.

Pass a memory directory for selection. CodexRunner mounts it at
`.agents/skills/memory` in each workdir; keep the
mount and source memory read-only. Enumerate and read its SKILL.md files explicitly:
KLineage SkillCard metadata need not appear in the native skill index. Packaged
bench/backend skills are runtime instructions, not optimization candidates.

Call `klineage.agent_tools.retrieve` with the mounted directory to filter
scope/exclusions.
`klineage.agent_tools.profile` captures counters for ranking.
The agent checks prerequisites and existing mechanisms against source, then selects
top-1 by measured bottlenecks. Counter capture currently supports CUDA only.
Missing or unsupported profiling is an error, not an empty selection. An existing
empty directory remains in memory mode and produces an unchanged terminal result.

## Output

```text
<workdir>/
  kernel.json                  # Candidate with complete problem and source_files
  submission/                  # Complete candidate bundle
  SKILL.md                     # Selected unchanged card; memory applications only
  evaluations/                 # Self-check, profiling, and verification evidence
```

Both modes build the complete changed bundle and self-check correctness, the intended
effect, preserved mechanisms, and paired performance against current_kernel using
`klineage.harness.evaluate`. Self-checks remain required with external verification
disabled. Generation leaves new validation unset; the optional verifier independently
checks the same result and attaches measured validation. Upstream artifacts stay unchanged.

In memory mode, no applicable card produces the exact unchanged kernel, including
its validation, without SKILL.md or submission/. Explain every rejection and remove
stale local outputs. In either mode, an unchanged terminal result retains its original
validation; do not remeasure or attach new validation.

## Handoff

In memory mode, pass the candidate directory to the next Apply with memory and prior selected IDs
excluded. It profiles that current kernel before ranking further candidates.
An unchanged terminal result ends the loop in either mode.
