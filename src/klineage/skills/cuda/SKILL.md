---
name: cuda
description: Write or inspect KLineage CUDA bundles callable from Python through pybind11. Use for expert-kernel reproduction, code generation, decomposition, skill application, and ABI or CUDA-stream debugging.
---

# CUDA

Read [cuda.md](cuda.md) before producing or auditing a CUDA bundle. It defines
the directory layout, build configuration, tensor ABI, and runtime constraints.

1. Read the supplied ProblemSpec and resolve its ordered tensor signature.
2. Implement the requested mechanisms in raw CUDA C/C++.
3. Export the host wrapper through pybind11 using the caller's device and stream.
4. Save the complete bundle text in Kernel.source_files; clear changed validation.
5. Use the `bench` skill for the action's required self-checks, including when
   external verification is disabled.

Start from [the add-one bundle](assets/add_one/config.toml) and its
[CUDA source](assets/add_one/solution/kernel.cu). The one-shot command in cuda.md
compiles these exact files and calls the export from Python.

