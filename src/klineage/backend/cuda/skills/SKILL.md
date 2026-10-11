---
name: cuda
description: Write or inspect KLineage CUDA bundles callable from Python through pybind11. Use for expert-kernel reproduction, code generation, decomposition, skill application, and ABI or CUDA-stream debugging.
---

# CUDA

Read [cuda.md](cuda.md) before producing or auditing a CUDA bundle. It defines
the directory layout, build configuration, tensor ABI, and runtime constraints.

1. Read the supplied Problem and resolve its ordered tensor signature.
2. Implement the requested mechanisms in raw CUDA C/C++.
3. Export the host wrapper through pybind11 using the caller's device and stream.
4. Write the complete bundle in current_kernel/submission/; clear stale validation.
5. Use klineage.evaluate and the `bench` skill; evaluation automatically saves each returned Kernel.

Start from [the add-one bundle](assets/add_one/config.toml) and its
[CUDA source](assets/add_one/solution/kernel.cu). Follow cuda.md to submit sources,
evaluate on the service, and select an accepted latest saved version.
With an explicitly supplied CUDA runtime and device, write diagnostic scripts
and invoke `ncu` directly. Keep diagnostic results separate from service validation
and do not overlap profiling with service benchmarks on that device.

The config.toml entry_point is a host callable with all launch parameters defined
in source. Compute dynamic launch dimensions from the inputs and acquire the
current stream on each call. The caller supplies only problem inputs.
