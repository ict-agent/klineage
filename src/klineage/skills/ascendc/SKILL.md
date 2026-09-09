---
name: ascendc
description: Write or inspect KLineage AscendC bundles for Ascend NPUs, including CANN compilation, torch_npu bindings, and caller-stream ordering.
---

# AscendC

Use CANN with its AscendC compiler and a compatible PyTorch/torch_npu installation.
Tensors use `device="npu"`. Inspect the actual SoC's execution units, memory levels,
alignment, supported types, and synchronization rules; CUDA thread and shared-memory
assumptions do not describe an Ascend kernel.

## Bundle

Write `submission/config.toml` and implementation text under `submission/solution/`:

```toml
[solution]
name = "candidate"
definition = "<problem.definition.name>"
author = "klineage"

[build]
language = "ascendc"
entry_point = "kernel.asc::kernel"
destination_passing_style = true
```

The harness compiles native `.asc`/`.cpp` sources with `bisheng -x asc` and loads
the pybind11 export. Set `ASCEND_HOME_PATH` to the installed toolkit and
`ASCEND_ARCH` to its supported `dav-...` architecture for the actual SoC.
Do not guess a compiler architecture from a marketing model name. Older CANN
installations lacking this compiler mode need a supported toolkit.
Keep required headers and implementation sources in solution/; keep binaries,
caches, expert repositories, measurements, and kernel.json outside the bundle.
Follow `.klineage/message/README.md` for persistence.

## Binding and execution

- Export one `PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)` and
  `module.def("kernel", &kernel)` matching entry_point.
- Accept ordered input tensors followed by preallocated outputs. Derive types,
  shapes, and strides from ProblemSpec, and write every output on every call.
- Check device, device index, dtype, shape, and strides. `data_ptr<T>()` includes
  the tensor's storage offset.
- The tensor ABI accepts base formats ND, NCHW, NHWC, and NCDHW. Blocked formats,
  reordered NDHWC, and unknown formats are rejected because physical addressing
  differs from logical strides. Cast such inputs to ND before defining the workload.
- Use a `c10::DeviceGuard` from `<c10/core/DeviceGuard.h>` for the tensor device.
  Include `<torch_npu/csrc/core/npu/NPUStream.h>` and obtain
  `c10_npu::getCurrentNPUStream().stream(true)`. This drains queued host tasks before
  direct native launch. Pass the returned ACL stream to the kernel launch;
  bypassing that queue with `stream(false)` can reorder producer work.
- Use the installed AscendC kernel API and its launch syntax
  `kernel<<<block_dim, nullptr, stream>>>(...)`. Preserve required GM/UB/L1/L0
  transfers, Cube/vector ordering, events, and buffer lifetimes.
- Expand expert abstractions into native AscendC. Hardware-required execution and
  data transfers remain necessary in a naive implementation. Keep compilation,
  host tensor reads, and device synchronization outside the measured callable.
- The candidate must not call expert libraries or cache outputs. Only the original
  expert adapter may depend on the supplied repository.

Use `klineage.harness.evaluate(candidate, Path.cwd())`; read
[bench](../bench/SKILL.md) for correctness, NPU event timing, and evidence.

References: [Ascend native extension example](https://github.com/Ascend/op-plugin/blob/master/examples/cpp_extension_asc/README.md),
[TorchNPU storage formats](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/framework/FormatHelper.cpp).
