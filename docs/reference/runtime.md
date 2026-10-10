# Runtime API

`AgentRuntime` selects the Python environment, tools and devices available to
an agent. `CodexRunner` runs its session. See
[Optional device diagnostics](../guides/evaluation.md#optional-device-diagnostics)
for configuration examples.

## Run an agent

Direct AgentRuntime instances have no device nodes by default. Local generation
supplies backend resources automatically. Accepted results come from the
configured evaluation service.

::: klineage.harness.AgentRuntime

::: klineage.harness.CodexRunner
    options:
      members: [mount_memory, __call__]

::: klineage.harness.CodexRunResult

::: klineage.harness.CodexRunnerError

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

::: klineage.harness.evaluate

::: klineage.harness.inspect_problem

Event logging and model usage collection are available through
`klineage.harness.logging`.
