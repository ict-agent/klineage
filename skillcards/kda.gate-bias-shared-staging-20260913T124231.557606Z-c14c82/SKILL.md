---
skill_id: kda.gate-bias-shared-staging
intent: Stage gate bias in shared memory for reuse across token rows.
preconditions:
- 'Data types: no additional dtype restriction; shared storage and loads must preserve
  bias representation because staging must not change arithmetic or rounding.'
- 'Layout: CTA consumers repeatedly address the same bias elements, and cooperative
  writer indices must match consumer head and column indices; mismatches select another
  bias. No additional contiguity or alignment is needed beyond the scalar accesses.'
- 'Storage: bias is readable in global memory and remains unchanged throughout its
  staged uses; otherwise the shared snapshot becomes stale.'
- 'Pipeline: all CTA threads can reach a barrier after cooperative producers finish
  and before consumers read; preserve consumer completion before storage reuse. Missing
  participation or ordering permits incomplete or overwritten reads.'
- 'Hardware: CTA shared memory and a CTA barrier are required; the staged layout plus
  static shared storage must fit the device per-block shared-memory limit or the launch
  cannot run.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore gate-bias staging in `solution/native.cuh` and `solution/prepare.cuh`.
Each preparation CTA processes one head and token chunk. Copy that head's bias
row once from global memory into shared memory, then let each gate-column
thread reload its shared element for every token row.

In `PrepareShared`, insert `bias[kDim]` immediately before `gt[kDim]`, using the
combined aligned declaration below. Update the size assertion. At the end of
`load_prepare`, after the beta-copy loop, add the cooperative bias copy. In the
gate-prefix loop inside `prepare`, replace only the bias load as shown. Remove
the now-unused `ld_global_scalar` helper from `native.cuh`; retain `ld_scalar`
and all its call sites. Both helpers enforce repeated loads against compiler
hoisting. Keep the unrolled row loop and its per-row reload comment: this change
restores shared staging, not register caching.

Global bias uses `head*kDim+column`; shared bias uses `column`. The cooperative
loop covers each column once. Existing CTA barriers after `load_prepare`
publish all bias writes before gate consumers run. Keep every barrier and the
`tid < kDim` guard. Bias has dedicated storage and no subsequent writer;
consumption finishes before CTA retirement. No buffer rotation or new barrier
is needed. Retain full-chunk bounds handling and beta boundary zero fill.

The launch already passes `sizeof(PrepareShared)` to both the dynamic shared
memory attribute and `prepare`. Those calls automatically use the larger
layout. Keep grid, block, host ABI, caller device, and caller stream unchanged.
Preserve the gate-prefix reduction order, sigmoid, multiplication order, gate
gain calculation, and downstream conversions. Only bias placement changes.

Use the existing Kernel evaluator with the unchanged problem for correctness
and latency. Inspect generated code for one cooperative global bias load per
column and repeated shared bias loads in the gate loop. Keep measurements
outside this card.

## Example configuration

Preserve batch 1, 4096 tokens, 96 heads, dimension 128, and 16 rows per chunk.
Preparation uses `grid=(256,96)`, 256 threads, and
`__launch_bounds__(kPrepareThreads,8)`. Threads 0 through 127 copy and consume
one bias column each. Bias is FP32 in contiguous `dt_bias[96,128]`; gate input
is BF16. Each consumer reloads its bias for all 16 rows while computing
`s.g[r*128+tid]` and `s.gt[tid]`.

Use `alignas(128) float bias[kDim], gt[kDim];`. Dynamic preparation storage
increases from 41856 to 42368 bytes; the independent transpose scratch remains
1024 bytes. The generated preparation function reports 2048 static shared bytes. All later fields follow the C++ layout automatically.
Retain recurrence storage and launches, workspace allocations, complete compile
flags, and config. Retain normalization grouping, scalar fragment transfers and
conversions, shared staging of other operands, BF16 tensor-core products with
FP32 accumulation, and block inversion. These instance settings and independent
mechanisms are not prerequisites of bias staging.

# Precondition

- Data types: no additional dtype restriction. Shared storage and scalar loads
  must preserve the bias representation so the same arithmetic receives the
  same values; staging must not introduce conversion or different rounding.
- Layout: consumers within a CTA must reuse bias elements for staging to save
  repeated global accesses. Cooperative writers and consumers must agree on
  head and column addresses; disagreement supplies another bias. No additional
  contiguity or alignment is required beyond the existing scalar accesses.
- Storage: global bias must be readable and remain unchanged throughout the
  staged uses. A modification during this interval invalidates the shared
  snapshot and changes the values observed by consumers.
- Pipeline: all CTA threads must reach the barrier that follows completed
  cooperative writes and precedes reads. Missing participation or visibility
  permits incomplete reads. Consumers must finish before storage reuse to
  prevent overwritten reads; retain that ordering even with dedicated storage.
- Hardware: the device needs CTA shared memory and a CTA barrier. The complete
  staged layout, including alignment padding, plus static shared storage must
  fit the per-block shared-memory limit; otherwise the launch cannot execute.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are the changed declarations and load sites, not contiguous function
bodies. Keep surrounding code unchanged. Remove the unused global-load helper
after replacing its only call.

## Before

```cuda
// solution/native.cuh: PrepareShared gate-bias position and size assertion.
alignas(128) float gt[kDim];
// ...remaining PrepareShared fields...
static_assert(sizeof(PrepareShared) == 41856);

// solution/prepare.cuh: load_prepare ends after its beta-copy loop.

// prepare: inside the existing gate-prefix row loop.
float bias = ld_global_scalar(args.bias+head*kDim+tid);
float g = gain * (bf_float(s.gate[r*kDim+tid])+bias);
```

## After

```cuda
// solution/native.cuh: restore dedicated bias storage before gt.
alignas(128) float bias[kDim], gt[kDim];
// ...remaining PrepareShared fields...
static_assert(sizeof(PrepareShared) == 42368);

// solution/prepare.cuh: append after load_prepare's beta-copy loop.
for (int i = tid; i < kDim; i += kPrepareThreads)
    s.bias[i] = args.bias[head * kDim + i];

// prepare: existing CTA barriers publish the copy before this row loop.
float bias = ld_scalar(s.bias+tid);
float g = gain * (bf_float(s.gate[r*kDim+tid])+bias);
```
