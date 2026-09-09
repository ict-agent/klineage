---
name: hip
description: Write or inspect KLineage native HIP bundles for Hygon DCUs, including tensor bindings, caller streams, and DTK compilation.
---

# Hygon HIP

Use Hygon's DTK compiler and HIP-enabled PyTorch. Tensors use `device="cuda"`
and `torch.cuda`; this API spelling does not make the target NVIDIA CUDA.
Inspect the installed target and headers. Do not assume NVIDIA warp size,
instructions, memory limits, or architecture flags.

## Bundle

Write `submission/config.toml` and implementation text under `submission/solution/`:

```toml
[solution]
name = "candidate"
definition = "<problem.definition.name>"
author = "klineage"

[build]
language = "hip"
entry_point = "kernel.hip::kernel"
destination_passing_style = true
```

The harness compiles with the installed HIP toolchain and loads the pybind11 export.
Set `HIP_HOME` or `ROCM_HOME` to the actual DTK installation when discovery needs it.
Select architecture flags from the installed DCU; do not copy CUDA flags.
Include every required source/header in the bundle; keep binaries, caches, expert
repositories, measurements, and kernel.json outside it. Follow
`.klineage/message/README.md` for persistence.

## Binding and execution

- Export one `PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)` and
  `module.def("kernel", &kernel)` matching entry_point.
- Accept ordered input tensors followed by preallocated output tensors; write all
  outputs on every call. Derive types, shapes, and strides from ProblemSpec.
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

Use `klineage.harness.evaluate(candidate, Path.cwd())`; read
[bench](../bench/SKILL.md) for correctness, HIP event timing, and evidence.

Reference: [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).
