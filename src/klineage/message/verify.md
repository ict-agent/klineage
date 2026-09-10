# Verify

[Shared message contract](README.md).

## Input

`Verify(prompt, workdir)` receives a verification prompt and the producer's workdir.
Action.verify supplies the verification criteria and the exact generation record
at `.klineage/codex-runs/<ActionClass>-<id>/`. Read its `prompt.txt` for task
parameters, `final_message.txt` for the proposed result, and `trace.jsonl` for evidence.
Input formats are defined by the producer's contract; the verifier does not
receive a second copy. Inspect the actual artifacts before accepting claims.
A manual check before generation reads the latest non-Verify record; absent records fail.

## Output

```text
<producer-workdir>/
  .klineage/codex-runs/Verify-<id>/
    prompt.txt
    trace.jsonl
    final_message.txt          # Exactly true or false
    stderr.log
  evaluations/                # Evidence, for verification that performs checks
```

The final response must be the lowercase token `true` or `false`, without JSON
wrapping, explanation or Markdown. Surrounding whitespace is stripped.
Other text raises `StructuredOutputError`. Missing evidence or failed checks mean
`false`. Verify disables its own verifier to prevent recursive verification.

| Producer | Permitted artifact updates |
| --- | --- |
| Init | Original adapter under `expert/`; measurements in `evaluations/`; validation in `kernel.json` |
| Apply | Audit round trials and final selection; write evaluation evidence and changed-kernel validation |
| Decompose | Reconstruction/evidence under `evaluations/`; predecessor validation in `kernel.json` |

Measurement checks also create build caches under `build/`. Preserve candidate
source_files, problem, SKILL.md and upstream files.

## Handoff

`Verify.run()` returns no payload; `passed` holds the boolean. The `verify(...)`
wrapper returns that boolean to the producer. A false result rejects the attempt;
the producer retries within its budget or raises on exhaustion. A true result permits handoff of
the producer's existing artifact directory, including permitted evidence updates.
