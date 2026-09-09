# Action

[Shared message contract](README.md).

## Input

`Action(prompt, workdir, enable_verifier=True, timeout=3600, max_retries=3)`.
`prompt` is a nonempty string; `workdir` is the action's working directory.
Each subclass renders its task parameters with Jinja. Paths and inline objects
are quoted as data; missing template parameters fail before execution.

Use a fresh workdir for each logical action. Every generation or verification
attempt receives a unique `.klineage/codex-runs/<ActionClass>-<id>/` directory.

## Output

```text
<workdir>/.klineage/codex-runs/<ActionClass>-<id>/
  prompt.txt
  trace.jsonl
  final_message.txt
  stderr.log
```

`run()` accepts no arguments and returns no payload. `response` contains the final
model text. Subclasses write the artifacts listed in their message contracts.
The caller already owns `workdir` and passes it downstream after successful completion.

When enabled, `verify(verify_prompt)` checks that workdir using the exact generation
record selected by `run_id`. Subclasses supply their verification criteria.
Failure retries the complete generation/verification routine up to `max_retries`
times after the initial attempt. Retry prompts include the previous failure and
direct the agent to repair artifacts in the same workdir; old traces remain.
Exhaustion raises. Interrupts propagate immediately. With verification disabled,
generation runs once and errors propagate. Failed artifacts are not a handoff.
