---
skill_id: fmha.output-conversion-store-fusion
intent: Fuse output conversion with final stores to eliminate intermediate memory
  traffic and a kernel launch.
preconditions:
- 'Data types: local conversion must preserve the original rounding and stored bit
  representation; eliminating the intermediate store/reload must not change values.'
- 'Layout: each output-store worker must be able to obtain the source elements for
  its conversions using known ownership and indices; otherwise direct fusion needs
  a redistribution.'
- 'Storage: converted scratch must serve only the final output stores, and their workers
  must have access to the unconverted source state; removing scratch would otherwise
  discard live data.'
- 'Pipeline: source producers must complete and make their writes visible before the
  fused store launch, and source storage must remain live until its reads finish;
  the removed conversion launch must have no other required effects.'
- 'Hardware: no additional feature or fixed capacity is required; fusion uses ordinary
  per-thread execution without new collective instructions or explicitly reserved
  on-chip storage.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse output conversion into the final store kernel. Convert each normalized value
in the thread that stores it, retaining rounded pairs locally. Remove the global
packed intermediate and its producer launch.

In `solution/attention.cuh`, delete `attention_pack`. Change `epilogue` to accept
`float (&out)[kPvRegs]` instead of the packed array. Immediately after computing
`length`, create the local packed array and fill it with the existing `half_pair`.
Retain the remaining LSE and scalar output stores verbatim. In
`attention_epilogue`, load `state.out`, retain the LSE loads, and pass `out` to
`epilogue`.

In `solution/ops.cuh`, remove only `EpilogueState::packed`. Keep `out`, `lse`,
`denominator`, and `maximum` in their existing order. In `solution/kernel.cu`,
remove the `attention_pack` launch and its error check. The existing
`sizeof(EpilogueState)` allocation updates automatically; no separate allocation
or pointer changes are needed. Rebuild the complete bundle because every kernel
uses the state structure's stride.

Check the rebuilt binary: conversion belongs to `attention_epilogue`, and no
`attention_pack` launch remains. Validate with `klineage.harness.evaluate` using
the complete unchanged problem, oracle, tolerances, seed, and timing policy.

## Ownership and ordering

Thread `tid` in CTA `blockIdx.x` owns state entry
`size_t(blockIdx.x) * kThreads + tid`. Packed entry `i` contains the rounded bits
of output components `2*i` and `2*i+1`, low half first. For output stores, with
`lane=tid%kWarpSize` and `warp=tid/kWarpSize`, pair `i` maps to:

- Row: `work.y*kM + warp*2*kCoreRows + lane/kLanesPerRow + (i%2)*kCoreRows`.
- Column: `(i/2)*kInputPanelCols + (lane%kLanesPerRow)*kPairElems`.
- Address: `output + (start+row)*kRow + work.z*kDim + column`.

Keep this mapping, the invalid-batch return, and the `row >= length` store guard.
Preserve the existing leader-only LSE stores. Input and output remain contiguous
packed NHD, with row stride `kHeads*kDim` half elements.

Keep `attention_normalize` immediately before `attention_epilogue` on the caller
stream. Its completion publishes normalized FP32 state; earlier launches publish
LSE. The fused kernel reads this state and converts before each output store.
No new barrier is needed: the conversion and store have the same thread owner.
The invocation's state tensor stays alive through the final launch, and allocator
reuse remains ordered on that stream. All earlier synchronization remains intact.

Keep `half_pair`, its two `scalar_half` calls, and `store_half` unchanged. Their
round-to-nearest FP32-to-FP16 conversion, low/high bit order, and separate scalar
stores preserve numerical behavior. Do not fuse normalization into this kernel or
change any QK/PV arithmetic, softmax reduction grouping, or probability rounding.

## Example configuration

Replay uses CUDA on `nvidia-sm90a-cuda13`, eight sequences, 16,384 packed tokens,
64 heads, and head dimension 128. Q/K/V/output are FP16; offsets are int32;
normalized state and row statistics are FP32. Tiles remain 128 query rows by
176 keys, with 256 threads, two 128-thread groups, and 64 output components per
thread. Each thread converts 32 pairs. Keep the 88-component score fragments,
44 probability pairs, 94 key slabs, 135 query-tile bound, and 8,640-CTA launch.

`EpilogueState` shrinks from 408 to 280 bytes by removing 128 packed bytes per
thread. Its allocation shrinks from 902,430,720 to 619,315,200 bytes. Keep the
zero dynamic-shared-memory launch configuration and `__launch_bounds__(kThreads,1)`.
The launch sequence ends `attention_values` (descending key tiles),
`attention_normalize`, `attention_epilogue` after fusion.

Retain global score/probability storage reuse, scalar QK/PV FMAs, separate maxima,
softmax, probability-conversion and normalization launches, global reduction
scratch, fragment ownership, ragged bounds, and caller-stream allocation lifetime.
Keep the native ABI, SM90 target, and compiler flags `-O3 -std=c++17 --use_fast_math
--resource-usage -lineinfo -DNDEBUG`. No compiler-control change is needed: separate
CUDA launches in the preceding state enforce the conversion/store boundary.

# Precondition

- Data types: the local conversion must reproduce the original rounding and
  stored representation. The intermediate write/read must add no numerical
  transformation; otherwise removing it changes results. The technique itself
  imposes no particular source or destination dtype.
- Layout: output-store workers must be able to obtain their conversion inputs
  from known ownership and indices. Otherwise a redistribution is required before
  this direct fusion can supply the correct values. No extra contiguity or
  alignment is required beyond the retained accesses.
- Storage: converted scratch must have no consumers except the final stores.
  Their workers must be able to read the unconverted source state. Removing the
  array would otherwise destroy live data or leave conversions without operands.
- Pipeline: all source producers must finish and publish their writes before the
  fused launch. Source buffers must stay live until its reads complete, preventing
  premature reuse. The removed launch must have no other required effects; those
  could not disappear with the intermediate. No particular stage count is needed.
- Hardware: no additional feature or fixed capacity is required. Fusion uses
  ordinary per-thread execution without new collective instructions or explicitly
  reserved on-chip storage. The compiler manages temporary registers and spills;
  no new shared-memory capacity condition follows from this change.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Fragments below replace the named locations. The helper's unchanged LSE and output
store body follows its shown prefix. Delete the complete `attention_pack`
definition and `EpilogueState::packed` member as described above.

## Before

```cuda
// EpilogueState contains this intermediate member after out.
uint32_t packed[kPvRegs / kPairElems];

// epilogue signature; its body stores the supplied rounded pairs.
__device__ __forceinline__ void epilogue(
    const Params& p, int4 work,
    const uint32_t (&packed)[kPvRegs / kPairElems], float (&lse)[2]);

// attention_epilogue, after the unchanged work lookup and batch guard.
const auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + threadIdx.x];
uint32_t packed[kPvRegs / kPairElems];
float lse[2];
#pragma unroll
for (int i = 0; i < kPvRegs / kPairElems; ++i) packed[i] = state.packed[i];
#pragma unroll
for (int r = 0; r < 2; ++r) lse[r] = state.lse[r];
epilogue(p, work, packed, lse);

// Host launch tail.
check(cudaLaunchKernelEx(&config, attention_normalize, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_pack, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_epilogue, p));
check(cudaGetLastError());
```

## After

```cuda
// epilogue signature; retain its existing body and add the conversion below.
__device__ __forceinline__ void epilogue(
    const Params& p, int4 work, float (&out)[kPvRegs], float (&lse)[2]);

// Inside epilogue, immediately after computing length.
uint32_t packed[kPvRegs / kPairElems];
#pragma unroll
for (int i = 0; i < kPvRegs / kPairElems; ++i)
    packed[i] = half_pair(out[i * kPairElems], out[i * kPairElems + 1]);
// Existing LSE and guarded scalar output stores follow unchanged.

// attention_epilogue, after the unchanged work lookup and batch guard.
const auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + threadIdx.x];
float out[kPvRegs], lse[2];
#pragma unroll
for (int i = 0; i < kPvRegs; ++i) out[i] = state.out[i];
#pragma unroll
for (int r = 0; r < 2; ++r) lse[r] = state.lse[r];
epilogue(p, work, out, lse);

// Host launch tail; remove attention_pack and its error check.
check(cudaLaunchKernelEx(&config, attention_normalize, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_epilogue, p));
check(cudaGetLastError());
```
