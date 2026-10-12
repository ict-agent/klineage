# Action API

Use `generate` to create a kernel from a problem directory. Use `Generation`,
`Init`, `Decompose` and `Apply` to control individual actions. See
[Action Primitive](../concepts/actions.md) for a complete composition example.

## Generate from a problem

`generate` returns a measured Kernel, using a local evaluator or a supplied
service. It invokes `Generation` with the loaded Problem. The
[generation CLI](../guides/cli.md#generate-a-kernel) exposes the same workflow.

::: klineage.generation.generate

`Generation(..., runner=runner)` accepts a preconfigured runner, avoiding a
second CLI initialization. Its timeout and model settings come from that runner.

## Compose actions

Each successful `run()` returns an Artifact containing the kernel and any skill
card or records. The convenience functions construct and run one action.
`KLINEAGE_RUNNER` selects Codex or Claude Code for every action; see
[model configuration](../getting-started.md#configure-model-access).

::: klineage.action.Action
    options:
      members: [run, resume_from]

::: klineage.action.Init

::: klineage.action.Generation

::: klineage.action.Decompose

::: klineage.action.Apply

::: klineage.action.init

::: klineage.action.decompose

::: klineage.action.apply

::: klineage.action.same_implementation
