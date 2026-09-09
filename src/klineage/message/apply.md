# Apply

[Shared message contract](README.md). Apply one supplied or selected SkillCard.

## Input

| Parameter | Format |
| --- | --- |
| `current_kernel` | Absolute directory containing kernel.json, or inline Kernel |
| `skill` | Absolute SKILL.md path, inline SkillCard, or null for selection |
| `memory` | Absolute memory directory, or array of SKILL.md paths/inline cards |
| `exclude_skills` | Skill IDs excluded from memory selection |

With a supplied skill, apply that exact card. Otherwise use the documented Python
functions in AGENTS.md: `klineage.agent_tools.retrieve` filters scope/exclusions;
`klineage.agent_tools.profile` captures actual counters for ranking candidates.
The agent checks prerequisites and existing mechanisms against source, then selects
top-1 by measured bottlenecks. Counter capture currently supports CUDA only.
Missing or unsupported profiling is an error, not an empty selection.

## Output

```text
<workdir>/
  kernel.json                  # Candidate with complete problem and source_files
  submission/                  # Complete candidate bundle
  SKILL.md                     # Selected unchanged card, in selection mode
  evaluations/                 # Profiling and external verification evidence
```

Generation clears validation. The verifier checks correctness, the one intended
effect, preserved mechanisms, and paired performance against current_kernel, then
attaches measured validation to kernel.json. Upstream kernel and card stay unchanged.
In selection mode, no applicable card produces the exact unchanged kernel without
SKILL.md or submission/. Explain every rejection. Stale local outputs must be removed
before emitting this terminal result.

## Handoff

Pass the candidate directory to the next Apply with memory and prior selected IDs
excluded. It profiles that current kernel before ranking further candidates.
An unchanged result without a selected card ends the workflow.
