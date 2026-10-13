---
skill_id: kda.gate-bias-register-reuse
intent: Cache invariant gate bias in a thread register across token rows.
preconditions:
- 'Data types: no additional dtype restriction; caching must preserve the bias representation
  and leave arithmetic order and rounding unchanged.'
- 'Layout: each participating thread repeatedly addresses the same bias element; otherwise
  one cached value cannot serve all of its iterations. No additional contiguity or
  alignment is required beyond the existing scalar load.'
- 'Storage: the bias is readable in shared memory at the hoist point and stays logically
  unchanged across the replaced reads; otherwise the cached value becomes stale.'
- 'Pipeline: bias producers must finish and make their writes visible before the hoisted
  load; retain ordering that completes consumers before shared storage reuse. No new
  collective participation is required for thread-local caching.'
- 'Hardware: registers must hold one bias scalar per participating thread through
  its consuming loop; spilling can erase the shared-load reduction benefit. No additional
  special instruction is required.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

In `solution/prepare.cuh`, hoist the bias load from the gate-prefix loop in
`prepare` into the enclosing `if (tid < kDim)` block. Each participating thread
owns one key dimension and reuses its bias across token rows. Replace the
per-row `ld_scalar(s.bias+tid)` with one ordinary `s.bias[tid]` load, as shown
below. Remove only the per-row reload comment; keep `ld_scalar` and its other
call sites unchanged.

`load_prepare` fills `s.bias` from the head's global bias row. Existing CTA
barriers publish it before the gate-prefix loop. Keep these barriers and the
barrier after the loop. Bias storage is not overwritten during this phase;
shared storage remains live until the CTA finishes. No layout, launch, scratch
allocation, ownership, or synchronization change is needed. Retain the
`tid < kDim` guard and existing full-chunk bounds handling.

The optimization changes only bias loading. Preserve the increasing-row FP32
prefix sum, sigmoid expression, multiplication order, gate gain computation,
and every downstream BF16 conversion. Keep the loop unrolled. The deoptimized
load uses the existing volatile shared-load helper to prevent compiler hoisting;
the forward change removes that restriction only at this call site.

Check correctness and latency with the existing Kernel evaluator under the
unchanged problem. Inspect generated code to confirm one bias shared load per
participating thread feeds all loop iterations without bias spills. Keep
measurements outside this card.

## Example configuration

Preserve batch 1, 4096 tokens, 96 heads, dimension 128, and 16 rows per chunk.
`prepare` launches `grid=(256,96)` with 256 threads; threads 0 through 127
process one key column each. The loop consumes `s.gate[r*128+tid]` and writes
`s.g[r*128+tid]`, then writes `s.gt[tid]`. Bias is FP32 in `s.bias[128]`, copied
from contiguous `dt_bias[head*128+tid]`; gate input is BF16. Cache one FP32 value
across all 16 iterations. Retain `gain = expf(a_log[head])` and all constants.

Keep the existing `__launch_bounds__(kPrepareThreads,8)`, 42368-byte
`PrepareShared`, separate transpose scratch, complete compile flags, config,
pybind11 ABI, caller device, and caller stream. Retain normalization grouping,
shared staging, scalar fragment transfers and conversions, BF16 tensor-core
products with FP32 accumulation, block inversion, and separate recurrence
launches. These settings and retained mechanisms are not prerequisites of bias
caching.

# Precondition

- Data types: no additional dtype restriction. Holding an unchanged bias in a
  register must preserve its representation; keep arithmetic order and rounding
  unchanged so the transformation changes only the loading path.
- Layout: a thread's bias address must remain invariant over the reused
  iterations. A changing address requires different cached values. Existing
  scalar-load addressing suffices; no new contiguity or alignment rule applies.
- Storage: bias must already be readable in shared memory at the hoist point
  and remain logically unchanged across the original reads. Otherwise the
  earlier load can be invalid or yield stale data.
- Pipeline: producer completion and visibility must precede the hoisted load.
  Preserve consumer completion before shared storage reuse so caching does not
  cross a buffer lifetime. Each thread caches its own value; no additional
  collective participation is needed.
- Hardware: one bias scalar per participating thread must remain in registers
  through the loop. Register spilling can replace the saved shared loads with
  other memory traffic. No additional special instruction is needed.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace this complete guarded block inside `prepare`; all surrounding code
remains unchanged.

## Before

```cuda
if (tid < kDim) {
    float sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < kChunk; ++r) {
        // Reload each row's bias; volatile loads prevent register reuse.
        float bias = ld_scalar(s.bias+tid);
        float g = gain * (bf_float(s.gate[r*kDim+tid])+bias);
        sum += kGateScale * sigmoid(g);
        s.g[r*kDim+tid] = sum;
    }
    s.gt[tid] = sum;
}
```

## After

```cuda
if (tid < kDim) {
    float bias = s.bias[tid], sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < kChunk; ++r) {
        float g = gain * (bf_float(s.gate[r*kDim+tid])+bias);
        sum += kGateScale * sigmoid(g);
        s.g[r*kDim+tid] = sum;
    }
    s.gt[tid] = sum;
}
```
