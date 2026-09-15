# Init

[Shared message contract](README.md).

## Input

| Template parameter | Format |
| --- | --- |
| `problem` | Absolute FlashInfer Trace definition JSON or Python reference path |
| `repository` | Local repository path or Git URL |
| `expert_kernel` | Expert source path or name; may identify a template header |

For a Trace problem, the input dataset is:

```text
<dataset>/
  definitions/gemm.json   # Operator, axes, tensors, reference.run
  workloads/gemm.jsonl    # One Trace record binding axes and input sources
  inputs/*.safetensors         # Files referenced by workload inputs, if any
```

Definition and workload paths share the same relative name beneath their roots;
workload files use `.jsonl`. Tensor paths are relative to the dataset root.
A Python problem instead defines `make_inputs` and `torch_ref`.
`make_inputs` may accept `seed` and `device` keyword parameters; inspection supplies
the selected backend device. Without a device parameter, inputs must already be
allocated on that target.

## Output

```text
<workdir>/
  kernel.json                  # Expert-equivalent standalone Kernel
  submission/                  # Native backend bundle with Python binding
  repository/                  # Staged expert repository
  evaluations/                 # Inspection, self-checks, and external gate evidence
  expert/                      # Adapter calling the original expert instance
```

Inspection supplies the full `problem`, including executable reference code and
absolute paths for file-backed workload inputs. Retain those input files.
Resolve the target from supplied problem metadata or the installed runtime:
CUDA/NVIDIA uses cuda, Hygon uses hip, and Ascend uses ascendc. Preserve the resolved
language and platform throughout reproduction and all downstream actions.
Repository location is provided by the Init task.
Retries reuse an intact staged copy; incomplete staging needs a fresh destination.
Init measures the bundle against the original expert through the harness and
judges mechanism fidelity from source. Self-check evidence stays in `evaluations/`.
The expert retains its upstream release build settings; required native arguments
are persisted in `compile_flags`. Record the actual build command and version.
Generation leaves validation unset. The separate external gate adds measured
validation to `kernel.json`; source comparisons remain in evaluations/.

The final response identifies the fixed expert specialization and host call,
reports paired latency_ms/reference_latency_ms, and cites self-check evidence.
Both self-checks and the external gate use this instance as their reference.

## Handoff

Pass `workdir` to Decompose or Apply; each reads `workdir/kernel.json`.
