# KLineage

Klineage is an agentic harness tailored for device kernel programming across heterogeneous accelerators.
KLineage generates accelerator kernels from high-level program descriptions and extracts
reusable optimization techniques from expert implementations.

## Execution model

<figure class="diagram" tabindex="0" aria-label="Generation or Apply submits a kernel artifact for evaluation and receives performance feedback. Apply can use procedural memory.">
  <img src="assets/diagrams/execution.svg"
       alt="Generation or Apply submits a kernel artifact for evaluation and receives performance feedback. Apply can use procedural memory."
       width="800" height="392" loading="lazy">
</figure>

The flow starts with a **problem artifact**, which defines the computation and
the workloads it must handle. [Artifacts](concepts/artifacts.md) give problems,
generated kernels and reusable skills a common representation that connects the stages.

KLineage organizes Codex and Claude Code sessions as [Actions](concepts/actions.md).
[Model configuration](getting-started.md#configure-model-access) selects a
Responses-compatible or Anthropic Messages-compatible endpoint.
**Generation** creates a kernel from the problem. **Apply** refines a kernel,
drawing on procedural memory when useful. **Init** adapts expert implementations,
while **Decompose** extracts
their optimizations into reusable skills. Those skills populate the procedural
memory that carries optimization knowledge across tasks.

The kernel artifact goes to the **evaluation service**, which checks correctness
against the problem and measures performance. A [backend](concepts/backends.md)
provides execution and timing for the target hardware. The service returns
performance feedback to the action, guiding the next refinement of the kernel.

## Start here

| Content | Read |
| --- | --- |
| Install and run the project | [Getting started](getting-started.md) |
| Artifact organization | [Artifact](concepts/artifacts.md) |
| Action primitive | [Action Primitive](concepts/actions.md) |
| Accelerator Backend | [Accelerator Backend](concepts/backends.md) |
| Deploy evaluation service | [Evaluation guide](guides/evaluation.md) |
| Add a new artifact format or backend | [Extension guide](guides/extending.md) |
| API Documentation | [API reference](reference/artifacts.md) |
