# CodeGen

[Shared message contract](README.md).

## Input

| Template parameter | Format |
| --- | --- |
| `current_kernel` | Inline `Kernel.to_dict()` |
| `skill` | One inline `SkillCard.to_dict()`; see [Decompose](decompose.md) |

These inputs are objects, not upstream directory paths. Reconstruct them with
`Kernel.from_dict` and `SkillCard.from_dict`. Retain problem input files and evidence.

## Output

```text
<workdir>/
  kernel.json                  # Current Kernel after exactly one skill
  submission/                  # Complete candidate sources
  evaluations/                 # Verifier evidence, when enabled
```

Native bundles use the skill matching problem.language: [cuda](../skills/cuda/SKILL.md),
[hip](../skills/hip/SKILL.md), or [ascendc](../skills/ascendc/SKILL.md).
Python retains its config.toml language, configured callable, entry_point, and
destination_passing_style. Implementation files stay under solution/.

Preserve problem. Replace source_files with the complete bundle
and clear validation. Verification attaches the measured ValidationResult.

Final response:

```json
{"done": true}
```

## Handoff

Pass `workdir` to Profile or another kernel-consuming action.
CodeGen is a standalone generation action; the current workflow invokes Apply directly.
