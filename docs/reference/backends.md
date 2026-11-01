# Backend and timing API

A Backend owns the kernel layout and build process, reports device information
and measures loaded kernels. See
[Accelerator Backend](../concepts/backends.md) for the design and
[Extension guide](../guides/extending.md#add-a-device-implementation) to add a device.

## Measure a kernel

Inputs must already be on the target device. `bench` returns latency samples
and timing evidence; use the evaluation service to check correctness as well.

::: klineage.backend.Backend

::: klineage.backend.BackendKind

::: klineage.backend.TimingPolicy

::: klineage.backend.TimingResult

::: klineage.backend.bench

## Selection and registration

`get_backend` selects a registered implementation. `detect_backend` also checks
device availability. Register external backends before loading their kernels.

::: klineage.backend.get_backend

::: klineage.backend.detect_backend

::: klineage.backend.register_backend

::: klineage.backend.backends

::: klineage.backend.get_timing_backend

::: klineage.backend.platform_backend

## Built-in implementations

CUDA, HIP and Ascend share `Kernel.load_kernel()` and define their own layouts,
builders and timing providers.

::: klineage.backend.cuda.CudaBackend

::: klineage.backend.hip.HipBackend

::: klineage.backend.ascend.AscendBackend
