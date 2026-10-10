# Action API

Use `generate` to create a kernel from a problem directory. Use `Init`,
`Decompose` and `Apply` when you want to control each round yourself. See
[Action Primitive](../concepts/actions.md) for a complete composition example.

## Generate from a problem

`generate` returns a measured Kernel, using a local evaluator or a supplied
service. The [generation CLI](../guides/cli.md#generate-a-kernel) exposes the same
workflow.

::: klineage.generation.generate

## Compose actions

Each successful `run()` returns an Artifact containing the kernel and any skill
card or records. The convenience functions construct and run one action.

::: klineage.action.Action
    options:
      members: [run, resume_from]

::: klineage.action.Init

::: klineage.action.Decompose

::: klineage.action.Apply

::: klineage.action.init

::: klineage.action.decompose

::: klineage.action.apply

::: klineage.action.same_implementation
