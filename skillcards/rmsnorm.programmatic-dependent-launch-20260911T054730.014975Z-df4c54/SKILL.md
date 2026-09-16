---
skill_id: rmsnorm.programmatic-dependent-launch
intent: Enable programmatic dependent launch to overlap dependent CUDA kernel scheduling.
preconditions:
- 'Data types: no additional requirement; launch dependency control neither interprets
  tensor representations nor changes arithmetic.'
- 'Layout: no additional requirement; grid dependency control does not depend on tensor
  strides, alignment, or intra-grid element ownership.'
- 'Storage: any values shared across launches must already be device-addressable and
  remain valid through their last access; dependency control cannot make private storage
  accessible or extend buffer lifetimes.'
- 'Pipeline: dependencies must follow the same CUDA stream, with identifiable dependent
  accesses and completion points reachable by every producer CTA without downstream
  progress; these permit visibility waits and collective launch release without cycles.
  Buffer reuse must follow the last access, since launch release alone does not complete
  memory operations.'
- 'Hardware: compute capability 9.0 or newer and a CUDA toolchain/runtime supporting
  griddepcontrol and cudaLaunchAttributeProgrammaticStreamSerialization; these implement
  early launch and dependency waits. No additional per-CTA storage capacity is required.'
scope:
  cases:
  - rmsnorm
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore programmatic dependent launch (PDL) in `solution/fused_norm.cuh`.
The launch attribute permits early grid scheduling; the device wait orders
consumer accesses, and the release makes a producer CTA eligible to unblock
subsequent PDL launches. Concurrency is opportunistic. The CUDA dependency rules
are documented in the [NVIDIA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

Apply these three edits together:

1. Add `constexpr int kPdlEnabled = 1;` beside `kWeightBias`. In `launch()`,
   create the attribute shown below after the second copy succeeds; attach it
   to the existing zero-initialized `cudaLaunchConfig_t` before launching.
   Keep the attribute alive through `cudaLaunchKernelEx`.
2. In `fused_norm()`, insert the guarded `griddepcontrol.wait` immediately after
   `float sum_sq = 0.0f;`, before either input is read.
3. Insert the guarded `griddepcontrol.launch_dependents` after the complete
   output-pass loop, immediately before `fused_norm()` returns. Every thread
   follows this path; every CTA therefore signals release.

A release is not a memory-visibility fence. A later PDL consumer must wait for
its predecessors before accessing their data. Keep ordinary stream ordering
for the two device copies. Do not recycle buffers at the release instruction;
retain their lifetime through the last access. Preserve the existing CTA and
warp barriers: inter-grid dependency control does not replace shared-memory
publication or protection against scratch reuse.

The wait stays before all tensor work and release stays after all output
stores. This replay exposes entry/exit scheduling opportunities without moving
computation. Keep `config.toml`, the pybind11 ABI, device guard, caller stream,
error checks, grid/block dimensions, dynamic shared memory, and launch API.
There are no ownership, layout, or bounds changes.

## Example configuration

The supplied workload has 8192 rows of 7168 elements. Each CTA owns one row;
`block=(32,28)` supplies 896 threads, each owning eight consecutive columns at
`(round * threads + thread) * kVector`, with `thread=lane+warp*32`.
The current workload uses one round. Retain the `column < width` guards;
its width and row strides are divisible by the eight-element group size.
Activation rows have a 14336-byte stride, weights are contiguous, and the ABI
retains its 16-byte alignment checks. These settings are not PDL prerequisites.

Keep BF16 inputs, weights, and outputs; FP32 addition and squared-sum
accumulation; epsilon `1.0e-6f`; zero weight bias; and BF16 round-to-nearest
stores. The output pass recomputes unrounded FP32 sums before overwriting the
residual. Preserve the two-level butterfly reduction and
`rsqrt.approx.ftz.f32`, including their arithmetic order.

Retain 28 shared reduction floats and 896 exchange floats: 3696 dynamic
shared-memory bytes per CTA. Each shared exchange keeps both `__syncwarp()`
calls, and both `__syncthreads()` calls remain. Retain the two 117440512-byte
asynchronous device copies and the in-place device calculation on their
output buffers. Keep `cudaFuncSetAttribute` and `cudaLaunchKernelEx` unchanged.
The target is `nvidia-sm90a-cuda13`; the device instructions retain their
`__CUDA_ARCH__ >= 900` guards.

# Precondition

- Data types: no additional requirement. PDL orders launches and memory
  dependencies without inspecting tensor bits or altering arithmetic.
- Layout: no additional requirement. Neither the launch attribute nor the
  grid dependency instructions use tensor strides, alignment, or element
  ownership within a grid.
- Storage: values crossing a launch dependency must already be accessible
  to the consuming device kernel and stay valid through their last access.
  Dependency control provides ordering; it cannot expose another CTA's
  private storage or extend an allocation's lifetime.
- Pipeline: dependencies run forward within the same CUDA stream. Dependent
  accesses must have identifiable ordering points, and every producer CTA
  must be able to reach completion without requiring downstream progress.
  These properties allow a wait before dependent reads and release across
  all producer CTAs without a cycle. Buffer reuse must remain after the last
  access: release only enables scheduling and does not establish producer
  completion or memory visibility by itself.
- Hardware: compute capability 9.0 or newer, with a CUDA toolchain/runtime
  implementing `griddepcontrol` and
  `cudaLaunchAttributeProgrammaticStreamSerialization`. Earlier targets
  cannot perform this early-launch/dependency-wait mechanism. No additional
  per-CTA storage capacity is needed; scheduling overlap is not guaranteed.

# Scope

- Cases: rmsnorm
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These excerpts replace only launch configuration and the two device boundary
sites. Comments marking retained code are placeholders, not replacement bodies.

## Before

```cuda
// launch(): after both device copies succeed.
cudaLaunchConfig_t config{};
config.gridDim = dim3(kTokens);
config.blockDim = dim3(kWarp, kWarps);
config.dynamicSmemBytes = kSharedBytes;
config.stream = stream;
// Keep cudaFuncSetAttribute, cudaLaunchKernelEx, and error checks.

// fused_norm(): before the first input pass.
float sum_sq = 0.0f;
// Keep the complete input pass, reduction, and output pass.
// The function returns immediately after the output-pass loop.
```

## After

```cuda
// Namespace constant beside kWeightBias.
constexpr int kPdlEnabled = 1;

// launch(): after both device copies succeed.
cudaLaunchAttribute attribute{};
attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute.val.programmaticStreamSerializationAllowed = kPdlEnabled;
cudaLaunchConfig_t config{};
config.gridDim = dim3(kTokens);
config.blockDim = dim3(kWarp, kWarps);
config.dynamicSmemBytes = kSharedBytes;
config.stream = stream;
config.attrs = &attribute;
config.numAttrs = 1;
// Keep cudaFuncSetAttribute, cudaLaunchKernelEx, and error checks.

// fused_norm(): before the first input pass.
float sum_sq = 0.0f;
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;");
#endif
// Keep the complete input pass, reduction, and output pass.
// Immediately after the output-pass loop, before returning:
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;");
#endif
```
