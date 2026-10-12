# Runtime API

`AgentRuntime` selects the Python environment, tools and devices available to
an agent. `CodexRunner` and `ClaudeCodeRunner` run sessions against Responses
and Anthropic Messages endpoints respectively. See
[Optional device diagnostics](../guides/evaluation.md#optional-device-diagnostics)
for configuration examples.

## Run an agent

Direct AgentRuntime instances have no device nodes by default. Local generation
supplies backend resources automatically. Accepted results come from the
configured evaluation service.

Codex sessions reuse `CODEX_HOME`. Claude Code sessions use
`workspace/.klineage/claude-home/` and require a host API key or auth token.
Run records live under `workspace/.klineage/agent-runs/`. See
[Model access](../guides/evaluation.md#model-access) for credential visibility.

Both runners use the same artifact and attempt-record layout:

```text
current_kernel/
versions/<N>/
problems/
workspace/.klineage/agent-runs/<attempt>/
    prompt.txt
    trace.jsonl
    stats.jsonl
    stderr.log
    final_message.txt
```

`trace.jsonl` preserves the CLI's native events: Codex JSONL or Claude Code
`stream-json`. `stats.jsonl` uses KLineage's common model/evaluation event schema.
Failed attempts retain these files too. CLI session storage is separate from
artifact and record layouts. Action records are collected only from `agent-runs/`.

The runner writes `workspace/AGENTS.md` from `prompts/templates/agents.j4`.
It introduces the Python APIs; agents use `help()` to read signatures and docstrings.

Each attempt records `stats.jsonl` beside `trace.jsonl`. `api_response` events
track model usage and request duration; `evaluate` events track job IDs,
acceptance and per-workload measurements. Failed calls retain an error record.

```python
from pathlib import Path
from klineage.harness import ClaudeCodeRunner, create_runner, session_id

# Select from KLINEAGE_RUNNER and its provider environment.
runner = create_runner(Path("agent-workspace/task"), timeout=600)

# Explicit selection and a host credential variable.
runner = ClaudeCodeRunner(
    Path("agent-workspace/claude"), model="provider-model-id",
    api_base_url="https://gateway.example/anthropic",
    api_key_env="MODEL_API_KEY", auth_mode="bearer", timeout=600,
)
reply = runner("Read workspace/AGENTS.md and inspect the available interfaces.")
```

`create_runner(..., runner="claude-code")` overrides `KLINEAGE_RUNNER`.
To resume explicitly, assign `runner.resume_id = session_id(reply.trace_path)`
and invoke the runner again. Actions do this automatically on retry.

::: klineage.harness.create_runner

::: klineage.harness.RunnerKind

::: klineage.harness.AgentRuntime

::: klineage.harness.CodexRunner
    options:
      members: [mount_memory, __call__]

::: klineage.harness.ClaudeCodeRunner
    options:
      members: [mount_memory, __call__]

::: klineage.harness.RunResult

::: klineage.harness.RunnerError

::: klineage.harness.session_id

## Evaluation runtime hooks

These interfaces support service workers and configured developer environments.
`EvaluationRuntime` evaluates all workloads in separate workers under one deadline.
`CallableKernelEvaluator` evaluates one selected case in the calling process.
See [Per-workload results](../guides/evaluation.md#per-workload-results) for the output.

::: klineage.harness.ProblemRuntime

::: klineage.harness.CallInputs

::: klineage.harness.CallableKernelEvaluator
    options:
      members: [evaluate]

::: klineage.harness.eval.EvaluationRuntime
    options:
      members: [evaluate, info]

Use `klineage.evaluate()` for local and remote evaluation. Problems use
`definition.json` plus `workloads.jsonl`; Python-file inspection is unsupported.

Event logging and model usage collection are available through
`klineage.harness.logging`.
