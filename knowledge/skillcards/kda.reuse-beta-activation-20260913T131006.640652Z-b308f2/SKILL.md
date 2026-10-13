---
skill_id: kda.reuse-beta-activation
intent: Reuse beta activation factors in thread-local registers.
preconditions:
- 'Data types: cached evaluation must preserve each consumer''s factor representation
  and rounding; reuse requires identical deterministic activation results, not a particular
  dtype.'
- 'Layout: each thread must identify consumers with the same beta index; otherwise
  sharing a factor changes their values. No additional contiguity or alignment is
  required for scalar loads.'
- 'Storage: no particular memory level is required; beta logits must be readable by
  their consumer thread without required per-read side effects, because caching removes
  repeated accesses.'
- 'Pipeline: producers must complete and publish logits before caching, and logits
  must remain unchanged through their consumers. Refresh factors for each input tile
  and finish reads before storage reuse to avoid stale or overwritten values; no new
  collective participation is required.'
- 'Hardware: no specialized feature is required; the existing live state plus cached
  factors must fit the thread and block register budgets for reuse to remain in registers
  rather than spill.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

In `solution/recurrence.cuh`, cache beta activations in `recur_tile` before
forming `residual`. Evaluate each distinct row factor once per thread, then
reuse it across that thread's residual elements.

Replace the block shown below, immediately after `Reg inv = load_a...` and
before `u = Acc{}; mma(u,inv,transpose(residual));`. The existing `group`,
`val`, `u`, `in`, `BF16`, `Pair`, `quantize`, `sigmoid`, and `bf_float` supply
the shared context. Add the two cached factors at this location and remove
the per-element activation loop. Replace only its `ld_scalar` calls with
ordinary beta reads; retain the helper and its other callers. Those volatile
loads enforce recomputation in the deoptimized code and must leave this block
to restore reuse.

Each lane owns residual elements from two rows. Select the cached factor by
fragment parity; both elements of a pair and both column fragments for that
row use the same factor. Factor registers remain private to their thread.
The shared beta layout, writers, ownership, ABI, and launch geometry stay as
configured. No new synchronization is needed: preserve the CTA barrier after
`load_input` and all subsequent warp and CTA barriers. Logits remain immutable
during `recur_tile`; the post-compute barrier precedes subsequent storage use.
Compute fresh factors on every call, after the current tile has been published.

Preserve `BF16(sigmoid(bf_float(logit)))`, including the existing fast-math
activation and BF16 rounding. Keep the BF16 subtraction followed by BF16
multiplication, `quantize` boundaries, matrix accumulation order, and state
conversion unchanged. Do not substitute a different sigmoid expression or
combine arithmetic across rounding boundaries.

## Example configuration

The supplied workload has batch 1, 4096 tokens, 96 heads, and head dimension
128. Chunks contain 16 tokens. Preparation launches grid `(256,96)` with 256
threads; each of 256 ordered recurrence launches uses grid `(1,96)` with 256
threads on the caller stream. Retain these launches and the existing compiler
flags, including `--use_fast_math` and the SM90a target.

Each recurrence warp owns 16 value columns. With `lane = threadIdx.x % 32`
and `group = lane / 4`, fragment `j` owns row `group + (j % 2) * 8`.
Four fragments each hold two BF16 elements: eight activations become two,
each reused four times. `in.beta` contains 32 contiguous BF16 slots with
element stride `sizeof(BF16)`; the consumed indices are 0 through 15.
Keep `load_input`'s allocation-end zero fill and the head-major global beta
conversion. These fixed chunks require no new tail handling or early returns.

Retain dynamic shared allocations of 41856 bytes for preparation and 124672
bytes for recurrence, including its 18048-byte input structure; retain the
1024-byte transpose scratch. Preserve row-major matrices, scalar fragment
transfers, tensor-core MMA, register fragments, shared correction staging,
chunked recurrence, and BF16 global state handoffs. Their dimensions and
hardware features are example settings, not requirements of factor reuse.

Check correctness and latency with `klineage.harness.evaluate` under the
unchanged problem and policy. Inspect compiled recurrence code to verify
factor reuse; for this configuration, beta activation sites decrease from
eight to two while state-decay evaluations remain unchanged.

# Precondition

- Data types: all consumers sharing a cache entry must require the same
  deterministic activation result, representation, and rounding. Otherwise
  one cached result cannot replace their separate evaluations. The technique
  does not require a particular dtype.
- Layout: identify repeated beta indices within each thread's consumers.
  Incorrect grouping substitutes another row's factor. Scalar reads add no
  contiguity or alignment requirement beyond the existing valid accesses.
- Storage: no particular memory level is required. Logits must be readable
  by their consumer thread without required per-read side effects. Removing
  side-effectful reads would change semantics; the deoptimized volatile
  helper only inhibits compiler reuse of ordinary immutable data.
- Pipeline: finish producers and establish reader visibility before caching.
  Keep logits unchanged through their consumers, refresh factors for each
  input tile, and complete reads before reusing input storage. These rules
  prevent premature, stale, or overwritten reads. Thread-private reuse adds
  no collective participation or barrier requirement.
- Hardware: no specialized feature is needed. Existing live values plus
  distinct cached factors must fit the per-thread and per-block register
  limits for register reuse without spills; factor storage scales with the
  distinct-factor count times its register representation size.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// Publish this warp's corrected values for subsequent products.
Reg residual = quantize(u);
#pragma unroll
for (int j = 0; j < 4; ++j) {
    Pair v{val.x[j]}, r{residual.x[j]}, result;
    constexpr int kPairElems = sizeof(uint32_t) / sizeof(BF16);
    constexpr int kHalfChunk = kChunk / 2;
    #pragma unroll
    for (int e = 0; e < kPairElems; ++e) {
        // Reload each logit so activation factors cannot be reused.
        BF16 beta = BF16(sigmoid(bf_float(ld_scalar(in.beta+group+(j%2)*kHalfChunk))));
        result.h[e] = (v.h[e]-r.h[e])*beta;
    }
    residual.x[j] = result.u;
}
```

## After

```cuda
constexpr int kHalfChunk = kChunk / 2;
const BF16 beta0 = BF16(sigmoid(bf_float(in.beta[group])));
const BF16 beta1 = BF16(sigmoid(bf_float(in.beta[group+kHalfChunk])));

// Reuse each row's rounded factor across its residual elements.
Reg residual = quantize(u);
#pragma unroll
for (int j = 0; j < 4; ++j) {
    Pair v{val.x[j]}, r{residual.x[j]}, result;
    const BF16 beta = j%2 == 0 ? beta0 : beta1;
    result.h[0] = (v.h[0]-r.h[0])*beta;
    result.h[1] = (v.h[1]-r.h[1])*beta;
    residual.x[j] = result.u;
}
```
