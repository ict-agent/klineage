---
name: ascendc
description: Write or inspect KLineage AscendC bundles for Ascend NPUs, including CANN compilation, torch_npu bindings, caller-stream ordering, and msprof profiling.
---

# AscendC

The service uses CANN and compatible PyTorch/torch_npu. Tensors use `device="npu"`.
Use `klineage.info()` for its SoC and toolchain.
Generation agents are device-free by default: the evaluation worker owns NPU
access. Use `klineage.info()` and `klineage.evaluate()`; local `npu-smi` and
`torch.npu` checks do not describe the evaluator's device availability.
Inspect the actual SoC's execution units, memory levels,
alignment, supported types, and synchronization rules; CUDA thread and shared-memory
assumptions do not describe an Ascend kernel.

## Bundle

Write `submission/config.toml` and implementation text under `submission/solution/`:

```toml
[solution]
name = "candidate"
definition = "<problem definition name>"
author = "klineage"

[build]
language = "ascendc"
entry_point = "kernel.cpp::kernel"
destination_passing_style = true
```

The service compiles native `.asc`/`.cpp` sources with `bisheng -x asc` and loads
the pybind11 export. The controller sets `ASCEND_HOME_PATH` and `ASCEND_ARCH`
for the installed toolkit and the SoC's supported `dav-...` architecture.
Do not guess a compiler architecture from a marketing model name. Older CANN
installations lacking this compiler mode need a supported toolkit.
The default layout has `config.toml` and `solution/kernel.cpp`. Keep binaries,
caches, expert repositories and measurements outside submission/.
Follow an exact layout from `AscendKernel.kernel_spec()` (`from klineage.backend.ascend import AscendKernel`); other filenames
or headers must be declared in the Kernel class's Node layout.

## Binding and execution

- Export one `PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)` and
  `module.def("kernel", &kernel)` matching entry_point.
- Accept ordered input tensors followed by preallocated outputs. Derive types,
  shapes, and strides from Problem, and write every output on every call.
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

Use `klineage.evaluate(Path("current_kernel"))`; evaluation automatically appends
the complete result under `versions/0`, `versions/1`, ... including failures.
Finish with an accepted latest version; use `klineage.result(best_reply)`
to append an earlier accepted job as the latest version.
Read [bench](../bench/SKILL.md) for correctness, NPU event timing, and evidence.

## Profile with msprof

After correctness passes, use `msprof op` to investigate kernel bottlenecks and
`msprof` for application timelines. Check `msprof --help` and `msprof op --help`
for the installed CANN version and supported metrics.

Run profiling in a controller-provided NPU-enabled diagnostic environment while
the evaluator is idle; exit that process before the next evaluation. Default
agents have no NPU access, and the evaluation API does not collect msprof reports.
If profiling is unavailable, retain the error and continue with evaluation timings.

Write `workspace/profile_kernel.py` to build/load a fixed candidate and replay
the problem's shapes, dtypes, layouts and inputs on the caller's stream. Prepare
inputs and warm up before the selected capture. The script must launch the
kernel itself; `klineage.evaluate()` executes in a separate worker.
Replace `gemm_device` with the actual device-kernel name prefix:

```bash
kernel_name=gemm_device
profile_script=workspace/profile_kernel.py
profile_python=$(command -v python)
mkdir -p workspace/msprof
msprof op --kernel-name="$kernel_name" --launch-count=1 \
  --aic-metrics=PipeUtilization --output=workspace/msprof/op \
  "$profile_python" "$profile_script"
msprof --application="$profile_python $profile_script" --output=workspace/msprof/timeline
```

Filter by kernel name so input generation and reference operations are excluded.
Inspect Cube/Vector utilization, memory-transfer activity, cache traffic and
synchronization stalls. Use supported `Memory`, `L2Cache` or
`ResourceConflictRatio` metrics to test a specific hypothesis. Read exported
reports or open them in MindStudio Insight. Retain the candidate version,
workload, command and report paths under `workspace/`.

Profiling adds overhead. Validate each resulting change with an unprofiled
`klineage.evaluate()` run; its timings determine acceptance.

References: [Ascend native extension example](https://github.com/Ascend/op-plugin/blob/master/examples/cpp_extension_asc/README.md),
[TorchNPU storage formats](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/framework/FormatHelper.cpp),
[msprof op](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0082.html),
[msprof](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/profiling/atlasprofiling_16_0011.html).

The config.toml entry_point is a host callable with all launch parameters defined
in source. Compute dynamic launch dimensions from the inputs and acquire the
current stream on each call. The caller supplies only problem inputs.
