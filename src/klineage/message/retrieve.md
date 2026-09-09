# Retrieve

Ranking requires a supported counter profile, currently CUDA NCU. HIP and Ascend
event timings alone do not establish hardware bottlenecks.

[Shared message contract](README.md). Select one applicable SkillCard.

## Input

| Parameter | Format |
| --- | --- |
| `current_kernel` | Profile directory containing kernel.json, or inline Kernel |
| `skills` | Array of absolute SKILL.md paths or inline SkillCards |
| `exclude_skills` | Already-applied skill_id strings |

Load files with klineage.memory.load_skill. klineage.memory.retrieve filters scope
and excluded IDs. The agent checks intent, preconditions, and body against current
code: types, layout/contiguity and parallel ownership, storage placement, pipeline
stages/ordering, hardware features, conflicts, and optimizations already present.
Rank eligible cards using matching NCU captures under the current kernel directory
(this workdir for inline input). Missing or stale profiling is an error.

## Output

```text
<workdir>/
  SKILL.md                     # Exact selected card
```

Choose top-1. If no card applies, emit no SKILL.md and explain rejections in the
final response. Remove any stale local selection on retries. The verifier checks
both selection and absence against supplied cards, current source and measurements.
No kernel or profile is produced.

## Handoff

Pass the selected SKILL.md path and current_kernel to Apply. The workflow stops on
an accepted empty selection; otherwise it applies the card, profiles, and retrieves
again with that skill_id excluded.
