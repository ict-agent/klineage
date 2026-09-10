# Apply

[Shared message contract](README.md). One kernel optimization round may combine
multiple techniques and implementation trials to produce the best validated kernel.

## Input

| Parameter | Format |
| --- | --- |
| `current_kernel` | Absolute directory containing kernel.json, or inline Kernel |
| `memory` | Optional directory; mounted absolute path in the prompt when supplied |
| `exclude_skills` | Skill IDs excluded from memory guidance |

With `memory=None`, an empty string, or whitespace, the round runs independently.
The prompt omits memory and exclusions, and no memory is mounted.

When supplied, memory is mounted at `.agents/skills/memory`. Read its SKILL.md files
explicitly; custom metadata need not appear in the native skill index. Keep the
mount and source memory read-only. Packaged bench/backend skills provide runtime
instructions separately.

Use relevant recipes as guidance. Combine or adapt techniques after checking
prerequisites against the current implementation, including changes from earlier
trials. An empty directory or no applicable recipe does not end optimization;
continue from source analysis. Do not copy or emit a selected SKILL.md.

Counter diagnostics from `klineage.tools.profile` are optional and require
CUDA/NCU. Correctness and paired performance checks remain required on every backend.

## Output

```text
<workdir>/
  kernel.json                  # Best validated result of this round
  submission/                  # Complete source bundle, when changed
  evaluations/                 # Trial measurements and source audit
```

Keep the round input fixed as the reference for
`klineage.harness.evaluate(candidate, workdir, reference=current_kernel)`.
Evaluate proposed changes, retain the best accepted implementation, and restore
its complete sources before finishing. Acceptance alone is not evidence of a speedup;
compare actual candidate and reference latencies. Record changes, measurements,
and recipes used in evaluations/ and the final response. Do not emit SKILL.md.

Self-checks remain required with external verification disabled. Generation leaves
new validation unset; the optional verifier independently checks the final result
and attaches measured validation. Preserve the full problem, ordered ABI, and
upstream artifacts.

If the round finds no validated improvement, preserve the exact input, including
validation, with no submission/ or SKILL.md. Explain the trials and evidence;
unavailable evaluation tools do not establish a successful result.

## Handoff

Pass the resulting kernel directory and optional memory to the next round.
Each round checks applicability against its current code; prior use alone does
not exclude a recipe. An unchanged result ends the CLI loop. `--max-apply-step`
limits rounds, not the number of optimization techniques within a round.
