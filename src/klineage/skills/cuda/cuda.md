# CUDA bundle

## Layout and build

```text
<action workdir>/
  kernel.json
  submission/               # Generated source bundle
    config.toml
    solution/
      kernel.cu             # pybind11 host export and device implementation
      *.cuh                 # Optional implementation headers
  evaluations/
  build/
```

Only config.toml and implementation text under solution/ belong in the bundle.
Keep metadata, skills, repository copies, binaries, caches, and measurements outside.
Use normalized relative paths without symlinks, `..`, backslashes, or NUL bytes.
Keep the bundle below 16 MiB and 1024 files.

Copy [config.toml](assets/add_one/config.toml) and adapt:

| Table | Field | Value |
| --- | --- | --- |
| solution | name | Candidate name |
| solution | definition | Exact problem.definition.name |
| solution | author | `klineage` |
| build | language | `cuda` |
| build | entry_point | `kernel.cu::kernel`, relative to solution/ |
| build | destination_passing_style | `true` |

The evaluator compiles native sources with `torch.utils.cpp_extension.load` before
timing. It calls `module.kernel(*inputs, *outputs)`. Define one
`PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)` and export the configured symbol
with `module.def("kernel", &kernel)`. PyTorch supplies pybind11 headers and tensor
type conversion. No Python loader or extra build system is needed.
Do not add AKO4X benchmark settings to config.toml.
For architecture-specific instructions, configure the process's
`TORCH_CUDA_ARCH_LIST` for the actual target, such as `9.0a` on Hopper.

## Tensor ABI

Implement a void host wrapper with `const torch::Tensor&` arguments: inputs first,
then preallocated outputs, each in problem.definition insertion order. Resolve
symbolic shapes from definition.axes and workload.axes; preserve dtypes, layouts,
declared strides, and semantics. Write every output in place on every call.
Include `<torch/extension.h>` and export the wrapper through pybind11 as above.

The local evaluator supports tensor inputs/outputs, including rank-zero tensors.
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

Use `klineage.harness.evaluate(candidate, Path.cwd())` for correctness and CUPTI
latency. Follow the action's comparison and measurement responsibilities; see
[the bench skill](../bench/SKILL.md) for the entrypoint and evidence contract.

Do not synchronize or read tensor values on the CPU inside the callable. Build
and prepare launch resources outside the measured path. Preserve the caller stream.

## One-shot pybind11 example

The [two-file example](assets/add_one/config.toml) computes `y = x + 1` for contiguous
float32 CUDA tensors. Adapt its signature, checks, and computation to the target.
From an action workdir, this command builds the packaged source and calls it:

```bash
python - .agents/skills/cuda/assets/add_one <<'PY'
import hashlib
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

source = Path(sys.argv[1]).resolve() / "solution/kernel.cu"
name = "add_one_" + hashlib.sha256(source.read_bytes()).hexdigest()[:20]
module = load(name=name, sources=[str(source)], with_cuda=True,
              extra_cflags=["-O3"], extra_cuda_cflags=["-O3"])
x = torch.randn(1027, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
module.kernel(x, y)
torch.testing.assert_close(y, x + 1)
print("pybind11 add-one passed")
PY
```

This checks the binding; use the `bench` skill for performance evidence.
Read skill assets as templates; write the candidate into submission/.

## Save the candidate

With the inspected `problem`:

```python
from pathlib import Path

from klineage.harness.artifacts import read_source_tree, save_kernel
from klineage.kernel import Kernel

source_files = read_source_tree(Path.cwd() / "submission", "candidate")
candidate = Kernel.from_sources(
    name=problem.name,
    problem=problem,
    source_files=source_files,
    build_root=Path.cwd() / "build",
)
save_kernel(candidate, Path.cwd())
```

For an existing Kernel, preserve its other fields and replace source_files with the
complete map and validation with None. Call build() before direct execution.
The subprocess harness builds from that map in its worker. The embedded problem
contains the oracle and workload; retain files referenced by workload inputs.
Follow the action's handoff contract for kernel.json and any additional artifacts.
