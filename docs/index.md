# KLineage

Klineage is an agentic harness tailored for device kernel programming across heterogeneous accelerators.
KLineage generates accelerator kernels from high-level program discriptions and extracts
reusable optimization techniques from expert implementations.

## Execution model

<figure class="diagram" tabindex="0" aria-label="Apply queries procedural memory, submits a kernel artifact to the evaluation service, and receives performance feedback.">
  <img src="assets/diagrams/execution.svg"
       alt="Apply queries procedural memory, submits a kernel artifact to the evaluation service, and receives performance feedback."
       width="800" height="392" loading="lazy">
</figure>

The flow starts with a **problem artifact**, which defines the computation and
the workloads it must handle. [Artifacts](concepts/artifacts.md) give problems,
generated kernels and reusable skills a common representation that connects the stages.

Klineage organize agent sessions in [Actions](concepts/actions.md). Actions perform transformations on these artifacts. In the diagram,
**Apply action** generates or improves a kernel, drawing on procedural memory when
useful. **Init** adapts expert implementations, while **Decompose** extracts
their optimizations into reusable skills. Those skills populate the procedural
memory that carries optimization knowledge across tasks.

The kernel artifact goes to the **evaluation service**, which checks correctness
against the problem and measures performance. A [backend](concepts/backends.md)
provides execution and timing for the target hardware. The service returns
performance feedback to Apply action, guiding the next refinement of the kernel.

## Start here

| Content | Read |
| --- | --- |
| Install and run the project | [Getting started](getting-started.md) |
| Artifact organization | [Artifact](concepts/artifacts.md) |
| Action primitive | [Action Primitive](concepts/actions.md) |
| Accelerator Backend | [Accelerator Backend](concepts/backends.md) |
| Deploy evaluation service | [Evaluation guide](guides/evaluation.md) |
| Add an new artifact format or backend | [Extension guide](guides/extending.md) |
| API Documentation | [API reference](reference/artifacts.md) |
