---
name: hip
description: Write or inspect KLineage native HIP bundles for Hygon DCUs, including tensor bindings, caller streams, and DTK compilation.
---

# Hygon HIP

The service uses Hygon's DTK compiler and HIP-enabled PyTorch. Tensors use `device="cuda"`
and `torch.cuda`; this API spelling does not make the target NVIDIA CUDA.
Use `client.info()` from `klineage.server.Client` for the target; inspect supplied headers.
Do not assume NVIDIA warp size,
instructions, memory limits, or architecture flags.

## Bundle

Write `submission/config.toml` and implementation text under `submission/solution/`:

```toml
[solution]
name = "candidate"
definition = "<problem definition name>"
author = "klineage"

[build]
language = "hip"
entry_point = "kernel.hip::kernel"
destination_passing_style = true
```

The service compiles with its HIP toolchain and loads the pybind11 export.
The controller configures `HIP_HOME` or `ROCM_HOME`; report missing toolchains.
Select architecture flags from the installed DCU; do not copy CUDA flags.
The default layout has `config.toml` and `solution/kernel.hip`. Keep binaries,
caches, expert repositories and measurements outside submission/.
Follow an exact layout from `.klineage/specs/kernel.hip.json`; additional source
or header files must be declared in the Kernel class's Node layout.

## Binding and execution

- Export one `PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)` and
  `module.def("kernel", &kernel)` matching entry_point.
- Accept ordered input tensors followed by preallocated output tensors; write all
  outputs on every call. Derive types, shapes, and strides from Problem.
- Check device, device index, dtype, shape, and strides. `data_ptr<T>()` already
  includes the tensor's storage offset.
- Use `c10::DeviceGuard` from `<c10/core/DeviceGuard.h>` for the tensor device.
  Obtain `c10::hip::getCurrentHIPStream(device_index).stream()` from
  `<c10/hip/HIPStream.h>` and pass it to every launch and asynchronous copy.
- Use `<hip/hip_runtime.h>` and propagate HIP launch errors. Keep computation in
  explicit HIP; preserve required wavefront participation, LDS layout, and fences.
- Keep compilation, resource preparation, host tensor reads, and synchronization
  outside the measured callable. Do not cache output values or call expert libraries
  from the candidate. An original-expert adapter may depend on its repository.

Use `client.evaluate(Path("kernel"), Path.cwd())`; the client automatically appends
the complete result under `kernels/0`, `kernels/1`, ... including failures.
Finish with an accepted latest version; use `client.result(job_id, work=Path.cwd())`
to append an earlier accepted job as the latest version.
With an explicitly supplied DTK runtime and DCU, diagnostic scripts may build and
inspect the kernel using vendor tools. Their timings do not replace service
validation; avoid concurrent diagnostics and service benchmarks on the same device. Read
[bench](../bench/SKILL.md) for correctness, HIP event timing, and evidence.

Reference: [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).

The config.toml entry_point is a host callable with all launch parameters defined
in source. Compute dynamic launch dimensions from the inputs and acquire the
current stream on each call. The caller supplies only problem inputs.
