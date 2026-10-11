# Accelerator Backend

A backend connects KLineage to an accelerator. Its Kernel class defines how an
implementation is organized and built. Its Backend class measures that kernel
and reports the device and runtime information needed to run it.

## Design

The two classes divide the work into four interfaces:

| Interface | Purpose |
| --- | --- |
| `Kernel.kernel_spec()` | Describe the kernel's file layout |
| `Kernel.build_kernel()` | Build the implementation and return a callable |
| `Backend.bench()` | Measure a loaded kernel |
| `Backend.info()` | Report device, architecture and runtime information |

`Kernel.load_kernel()` reads a directory, checks its layout and calls the
backend's builder. The entry function in the source owns its launch parameters
and bindings. Callers use the resulting Kernel like an ordinary Python function.

The evaluation service uses these interfaces to build, check and measure each
workload. It keeps the results separate, so kernels can be compared across shapes
without combining their latencies into one average.

<figure class="diagram" tabindex="0" aria-label="Submit sources to the service for an isolated worker to build, check and time the kernel.">
  <img src="../assets/diagrams/backend-evaluation.svg"
       alt="Submit sources to the service for an isolated worker to build, check and time the kernel."
       width="800" height="392" loading="lazy">
</figure>

## Select a backend

`get_backend(language, platform)` selects a registered backend without accessing
a device. `detect_backend()` probes the installed runtimes and returns an
available backend. Set `KLINEAGE_BACKEND` to select a platform explicitly; device
visibility remains controlled by the vendor runtime.

| Device family | Python package | Source language | Timing |
| --- | --- | --- | --- |
| NVIDIA CUDA | `klineage.backend.cuda` | `cuda` | FlashInfer with CUPTI |
| Hygon HIP | `klineage.backend.hip` | `hip` | Current-stream device events |
| Ascend | `klineage.backend.ascend` | `ascendc` | Current-stream device events |

Each backend needs its matching Python dependencies, driver and toolchain.
Use separate evaluation services for different runtime environments. A controller
can connect to any of them without installing their device libraries locally.
See [Getting started](../getting-started.md) for installation and
[Evaluation guide](../guides/evaluation.md) for server setup.

## CUDA implementation

`CudaKernel` defines the CUDA file layout and compilation path.
`CudaBackend` supplies device information and delegates measurement to
`klineage.backend.cuda.timing`.

<figure class="diagram" tabindex="0" aria-label="Read the CUDA entry, compile the sources, and prepare the callable.">
  <img src="../assets/diagrams/cuda-build.svg"
       alt="Read the CUDA entry, compile the sources, and prepare the callable."
       width="800" height="384" loading="lazy">
</figure>

The native layout contains `metadata.json`, `submission/config.toml` and
`submission/solution/kernel.cu`. The configuration selects a host function;
PyTorch's extension loader and `nvcc` compile the source. The host function
exports its binding and chooses the launch geometry and current CUDA stream.
See [CUDA Kernel Artifact](artifacts.md#cuda-kernel-artifact) for the configuration.

With destination-passing style, KLineage allocates outputs and passes them after
the inputs. The public Kernel callable returns those outputs. A return-style
entry allocates and returns its own outputs. CUDA also supports Python host
entries through a separate file layout.

`CudaBackend.info()` reports the device architecture, compilation target, compiler
and runtime versions. `TORCH_CUDA_ARCH_LIST` can override the compilation target
without changing the reported device architecture.

CUDA timing uses FlashInfer's CUPTI benchmark and requires CUPTI 13 or newer.
The default policy uses 10 warmups, 50 samples per trial, 3 trials and cold L2.
The result retains the samples and reports the median of trial medians.
HIP and Ascend use device events and do not support cold-L2 measurement.

## Build and time a kernel locally

In a configured device environment, load an authored kernel directory and pass
inputs already allocated on that device:

```python
from pathlib import Path

from klineage.artifact import Kernel
from klineage.backend import bench

kernel = Kernel.load_kernel(build_root=Path("current_kernel"), include_paths=())
timing = bench(kernel, args=inputs)
print(timing.median_ms)
```

Here `inputs` is the tuple of tensors required by the problem. `bench` detects
the backend and measures the callable; it does not check correctness. Use
`Client.evaluate()` for the complete evaluation used by actions.

An agent with device and tool access can also inspect performance with vendor
tools such as `ncu`. Finish this work before benchmarking on the same device.
See [Optional device diagnostics](../guides/evaluation.md#optional-device-diagnostics)
for sandbox configuration.

## Add a backend

Each backend provides `kernel.py` and `timing.py`. The first defines its Kernel,
Backend and compilation helpers; the second implements measurement. Other modules
can be added as needed, and `skills/SKILL.md` teaches the agent how to author kernels
for that platform.

Implement the four interfaces above, then register the Backend for selection.
Artifact layouts remain owned by their Kernel classes. Follow the
[Extension guide](../guides/extending.md#add-a-device-implementation) for the module
layout, timing contract and registration steps. The
[Backend and timing API](../reference/backends.md) provides the full signatures.
