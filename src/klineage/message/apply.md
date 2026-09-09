# Apply

[Shared message contract](README.md). Apply one supplied SkillCard.

## Input

| Parameter | Format |
| --- | --- |
| `current_kernel` | Absolute directory containing kernel.json, or inline Kernel |
| `skill` | Absolute SKILL.md file path, or inline SkillCard |

Load the existing Kernel and the card; no baseline is inferred from another artifact.
Read the card's scope, preconditions, intent, and body against actual code. The caller
owns selection; Apply neither repeats retrieval nor applies additional skills.

## Output

```text
<workdir>/
  kernel.json                  # Candidate with complete problem and source_files
  submission/                  # Complete candidate bundle
  evaluations/                 # External verification evidence
```

Generation clears validation. The verifier checks correctness, the one intended
effect, preserved mechanisms, and paired performance against current_kernel, then
attaches measured validation to kernel.json. Upstream kernel and card stay unchanged.

## Handoff

Pass the candidate directory to Profile. The next retrieval uses that fresh profile
and excludes the applied skill_id.
