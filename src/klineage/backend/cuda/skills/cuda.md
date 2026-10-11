# CUDA bundle

## Layout and build

Read the server's hardware information with `Client().info()` from `klineage.server`.
Its backend information reports SM, thread, register, shared-memory and L2 capacities.
Local device queries describe only the explicitly supplied diagnostic device;
check it against the service target before interpreting results.

```text
<action workdir>/
  problems/<problem_id>     # Symlink to the original problem directory
  current_kernel/           # Trial source space
    metadata.json           # Trial problem reference and build flags
    submission/
      config.toml
      solution/
        kernel.cu           # Host entry, binding and device implementation
  versions/
    0/                      # Complete returned Kernel
      metadata.json         # validation.workloads + evaluation diagnostics
      submission/
    1/                      # Next result, even if evaluation failed
  workspace/                # Scripts, diagnostics, builds and run records
```

Only config.toml and implementation text under solution/ belong in submission/.
Keep metadata at the Kernel root. Put skills, repository copies, diagnostic scripts,
binaries, caches and run records under workspace/.
Use normalized relative paths without symlinks, `..`, backslashes, or NUL bytes.
Keep the bundle below 16 MiB and 1024 files.
Use one exact layout from `CudaKernel.kernel_spec()` (`from klineage.backend.cuda import CudaKernel`). Additional source
or header files must be declared in the Kernel class's Node layout.

Copy [config.toml](assets/add_one/config.toml) and adapt:

| Table | Field | Value |
| --- | --- | --- |
| solution | name | Candidate name |
| solution | definition | Exact `problem.definition["name"]` |
| solution | author | `klineage` |
| build | language | `cuda` |
| build | entry_point | `kernel.cu::kernel`, relative to solution/ |
| build | destination_passing_style | `true` |

The service builds the configured host entry. Its native callable
receives `(*inputs, *outputs)`. Define one
`PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)` and export the configured symbol
with `module.def("kernel", &kernel)`. PyTorch supplies pybind11 headers and tensor
type conversion. No Python loader or extra build system is needed.
For local builds, use `Kernel.load_kernel(Path("current_kernel"))` from
`klineage.artifact` to keep compiler settings aligned with the service.
PyTorch 2.14 requires C++20. Let its extension loader select the language standard;
an explicit `-std=c++17` overrides that default and fails to compile its headers.
Call the loaded Kernel as `outputs = kernel(*inputs)`, using tensors matching a
registered workload. It allocates destination tensors for the native host entry;
only that entry receives the trailing output arguments.
Do not add AKO4X benchmark settings to config.toml.
The controller configures the service's target architecture. Use client.info()
before choosing architecture-specific instructions; do not override its environment.

## Tensor ABI

Implement a void host wrapper with `const torch::Tensor&` arguments: inputs first,
then preallocated outputs, each in problem.definition insertion order. Read all
cases from `problem.workloads`; each case's `axes` binds the variable dimensions
in `problem.definition["axes"]`. Implement every registered shape, preserving
dtypes, layouts, declared strides and semantics. Never treat the first case as
the whole contract. Write every output in place on every call.
Include `<torch/extension.h>` and export the wrapper through pybind11 as above.

`problem.values(role, workload=case)` describes one bound ABI. Without a selected
case, zero or multiple workloads retain symbolic variable axes; a sole case binds
automatically. Direct loaded tensor adapters resolve input axes from tensors and
reject inconsistent dimensions or unbound output-only axes. Evaluation workers
supply explicit case bindings. Kernels keep one directory reference to the complete problem.

The evaluator supports tensor inputs/outputs, including rank-zero tensors.
A Python scalar declared with `shape=null` is unsupported. Report that mismatch;
do not silently coerce or remove the argument.

Check device, device ID, dtype, shape, and declared strides before touching memory.
Use `tensor.data_ptr<T>()`, which already includes the storage offset; do not add
that offset again. Use `c10::cuda::CUDAGuard(tensor.device())` and obtain the caller
stream with `c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream()`.
Pass it to every launch and asynchronous copy. Propagate launch failures with
`TORCH_CHECK` and CUDA error messages. PyTorch supplies tensor metadata and binding;
the computation remains raw CUDA.

Keep computation in raw CUDA C/C++. Expand expert abstractions while retaining the
requested mechanisms. Do not include CUTLASS/CuTe implementations, call expert
libraries, delegate computation to frameworks, or cache output values. The original
expert adapter used for measurement may depend on its repository; the candidate may not.

## Evaluation

Use `Client().evaluate(Path("current_kernel"), Path.cwd())` for service-side correctness and
CUPTI latency across all registered workloads. Read individual outcomes from
`reply.validation.workloads[uuid]`; every case must pass. Multi-case top-level
latencies are `None`, so do not average shapes or use them as a scalar speedup.
A definition with no workloads cannot pass evaluation.
Follow the action's comparison responsibilities; see
`workspace/.agents/skills/bench/SKILL.md` for the entrypoint and evidence contract.

Do not synchronize or read tensor values on the CPU inside the callable. Build
and prepare launch resources outside the measured path. Preserve the caller stream.

## Binding example

The [two-file example](assets/add_one/config.toml) computes `y = x + 1` for contiguous
float32 CUDA tensors. Adapt its signature, checks, and computation to the target.
Read these assets as templates; write the candidate into current_kernel/submission/.
Submit it through the client for official validation. An authorized diagnostic
runtime may also build it for local inspection.

## Submit and evaluate

With authored current_kernel/submission/ files, set `reference` to the action's fixed
baseline, or None for a standalone/Decompose measurement. Select `problem_id`
from `Client().info()["problems"]` for the assigned task:

```python
from pathlib import Path
from klineage.serialization import to_json
from klineage.server import Client

client = Client(problem_id=problem_id)
problem_spec = client.problem()
problem_dir = (Path("problems") / problem_id).resolve(strict=True)
root = Path("current_kernel")
metadata = {
    "name": problem_spec.name,
    "problem": {"path": str(problem_dir), **problem_spec.target()},
    "validation": None,
    "compile_flags": [],
}
(root / "metadata.json").write_text(to_json(metadata))
reply = client.evaluate(root, Path.cwd(), reference=reference)
assert reply.accepted, reply.diagnostics
```

Edit the source files and metadata.json in the directory, clearing stale validation.
Re-submit after changes. Client appends every complete result under versions/0,
versions/1, ... including failures. Each version's metadata.json stores validation
and evaluation diagnostics; read it or reply.validation/reply.diagnostics.
The original problem directory is linked under problems/<problem_id>; input binaries are not copied. Do not save or edit returned Kernel versions.
Action uses the largest numbered version, without falling back after failure.
Use client.result(job_id, work=Path.cwd()) to append an earlier accepted job
as the latest version. current_kernel/ remains independent trial source space.

## Local checks

With registered workload inputs already on the device, load and call the Kernel:

```python
from pathlib import Path
from klineage.artifact import Kernel

kernel = Kernel.load_kernel(build_root=Path("current_kernel"))
outputs = kernel(*inputs)
```

The returned Kernel is directly callable and has no `.kernel` entry attribute.
Pass only input tensors; destination-passing outputs are allocated internally.
One output is returned as a tensor; multiple outputs are returned as a tuple.
Use the registered workload's shapes and dtypes for these checks.

## Performance inspection

If the controller supplies a CUDA device, compatible runtime and Nsight Compute,
write a diagnostic script and invoke `ncu` directly. Choose its options for the
question under investigation; KLineage has no profiling helper or fixed metric set.
Save reports in the workspace. Stop diagnostic work before service evaluation on
the same device, and use only the service result for validation and acceptance latency.

Device nodes and driver libraries must be accessible, but mounting a GPU does not
grant counter access. `ERR_NVGPUCTRPERM` means the driver denied that access.
Linux permissions depend on the installed driver's policy and capabilities;
newer drivers may also use profiling capability nodes under `/dev/nvidia-caps`.
Do not infer permission from CUDA availability or modify host driver settings.
See NVIDIA's [counter-permission guide](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
and [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
