# Evaluation service API

Use `klineage.evaluate` to measure kernels locally or through a configured service.
Use Service to embed
an evaluator in a Python application. The [Evaluation guide](../guides/evaluation.md)
covers server startup, problem registration and device access.

## Submit and retrieve

Evaluation returns the measured Kernel and saves a numbered snapshot. Check
`Evaluation.accepted` for the complete job status, including errors.

::: klineage.evaluate

::: klineage.info

::: klineage.problem

::: klineage.result

Client remains available for explicit service connections and controller code.

::: klineage.server.Client
    options:
      members: [health, info, problem, evaluate, result, verify]

::: klineage.server.Evaluation

::: klineage.server.ServerError

## Host a service

A Service owns a backend, its registered problems and evaluation records.
`load_problems` accepts problem directories or collections of them.

::: klineage.server.Service
    options:
      members: [problem, info, evaluate, result, start]

::: klineage.server.load_problems

::: klineage.server.RunningServer
    options:
      members: [start, close]

::: klineage.server.ClientGateway
    options:
      members: [start, close]

## Configure workers

WorkerRuntime runs evaluations in the configured device environment. Use the
server CLI options or supply `Service.runtime_factory` when embedding.
A custom factory accepts `timeout=remaining`; its runtime's `evaluate` method
must accept the recalculated per-call timeout as well.

::: klineage.server.worker.WorkerRuntime
    options:
      members: false
