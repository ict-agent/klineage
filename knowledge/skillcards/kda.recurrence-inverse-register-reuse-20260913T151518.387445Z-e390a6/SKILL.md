---
skill_id: kda.recurrence-inverse-register-reuse
intent: Reuse invariant inverse-matrix fragments in thread registers across correction
  products.
preconditions:
- 'Data types: no additional dtype restriction; retain loaded bits and consumer arithmetic
  because caching only removes duplicate loads.'
- 'Layout: each thread repeatedly requests the same addressed fragment with the same
  element order, and those addresses are valid at the first load; no additional contiguity
  or alignment beyond the existing loads is needed.'
- 'Storage: the reused operand is repeatedly loaded from global memory and every cached
  use belongs to the loading thread; a private copy cannot supply another thread without
  communication.'
- 'Pipeline: operand production must complete and become visible before the first
  load, and the operand must remain unchanged until the last cached use; otherwise
  reuse can observe an earlier value. No additional participation or synchronization
  is required for thread-private reuse.'
- 'Hardware: allocatable thread registers must cover the fragment plus overlapping
  live values; otherwise spills can defeat register reuse.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Retain each thread's inverse-matrix fragment in registers across the correction
products in `solution/recurrence.cuh::recur_tile`. This eliminates repeated global
loads of an invariant operand without reusing the correction computation itself.

The initial declaration already loads the fragment before residual construction:

```cuda
Reg inv = load_global_frag<FragView::Normal, kChunk>(args.inv+matrix_base,0,0,lane);
```

Keep that declaration. In the `m` loop, delete the assignment that reloads `inv`
and its explanatory comment, as shown below. Both correction-product sites then
use the original `inv`. Keep the loop body after the shown prefix unchanged.
Retain `load_global_frag` and its volatile scalar loader: removing their compiler
controls would affect unrelated loads. No launch, allocation, layout, or barrier
changes are needed.

`matrix_base = (head*kTiles + tile)*kMatrixElems` selects the inverse belonging
to this head and chunk. Each thread keeps its existing fragment ownership and
element order; no sharing or remapping is introduced. The preparation launch
publishes this global workspace before recurrence launches on the same caller
stream. Workspace contents remain unchanged throughout their consumption, and
the same stream orders the next invocation's reuse.

Keep all correction recomputations, FP32 accumulators, BF16 rounding points,
transpositions, stores, and warp/CTA barriers in their existing order. The shared
transpose scratch still requires publication before reads and reader completion
before reuse. The correction buffer retains its existing warp synchronization
and per-warp ownership. Caching `inv` adds no shared buffer or new collective.

The initial load already executes on every consuming thread, so moving no loads
outside their existing execution domain preserves bounds handling. All matrices
in this workload contain complete chunks; retain the host shape checks.

## Example configuration

Preserve batch 1, 4096 tokens, 96 heads, dimension 128, and `kChunk=16`.
Preparation uses grid `(256,96)`, 256 threads, and 41856 dynamic shared bytes.
Each of 256 recurrence launches uses grid `(1,96)`, 256 threads, and 124672 dynamic
shared bytes. Existing static shared scratch, compile flags, ABI, caller stream,
and buffer allocations remain unchanged.

The inverse workspace holds contiguous row-major BF16 matrices with row stride
`kChunk*sizeof(BF16)` bytes. Each of eight warps owns a 16-column value tile.
Each lane's `Reg` contains four 32-bit words, packing eight BF16 elements.
For word `j` and half `e`, its inverse coordinates are
`(lane/4 + (j%2)*8, 2*(lane%4) + (j/2)*8 + e)`.
The first correction product and eight state-product corrections use that same
fragment: caching replaces nine fragment loads with one per thread per chunk.

Retain BF16 `mma.sync.aligned.m16n8k16` products with FP32 accumulation, scalar
round-to-nearest BF16 conversions, shared transpose exchange, shared recurrent
state, global BF16 carry, and repeated residual-to-correction products. Those
mechanisms and this workload's sizes are replay settings, not caching prerequisites.
Preserve the complete problem, oracle, tolerances, and timing policy. Validate
through `klineage.harness.evaluate`; inspect generated code to confirm later
correction products reuse the initial fragment instead of reloading it or spilling
its extended live range. Store measurements outside this card.

# Precondition

- Data types: no additional dtype restriction. Retain the exact operand bits and
  consumer arithmetic; eliminating duplicate loads introduces no conversion or
  reduction change.
- Layout: each loading thread must request the same addressed fragment in the
  same element order at every cached use. Otherwise its saved words represent
  different operands. Addresses must be valid at the first load. Caching adds no
  contiguity or alignment requirement beyond the existing loads.
- Storage: the operand is repeatedly read from global memory, and each saved
  fragment supplies only its loading thread. A private register copy cannot
  serve another thread without an additional exchange mechanism.
- Pipeline: the producer must complete and make the operand visible before the
  first load. The operand must remain unchanged through the last cached use,
  including protection from buffer reuse; otherwise caching can retain stale
  data. Thread-private reuse adds no participation or synchronization requirement.
- Hardware: the allocator must have register capacity for the fragment and all
  overlapping live values, symbolically `R_fragment + R_other_live <= R_available`
  under the launch's allocation limits. Otherwise spilling can defeat register
  reuse. No special load or matrix instruction is required by caching itself.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The initial `Reg inv` declaration above remains in both versions. Replace only
this prefix of the `m` loop; its omitted remainder and closing brace are unchanged.

## Before

```cuda
#pragma unroll
for (int m = 0; m < kDim/kChunk; ++m) {
    Reg kr = load_global_frag<FragView::Transposed>(args.kr+tile_base,0,m*kChunk,lane);

    // Reload the inverse for each product; volatile loads prevent reuse.
    inv = load_global_frag<FragView::Normal, kChunk>(args.inv+matrix_base,0,0,lane);

    // Recompute the rounded correction for every state-product consumer.
    u = Acc{};
    mma(u,inv,transpose(residual));
    store_c<kChunk>(s.correction,quantize(u),0,value,lane);
    __syncwarp(kAllLanes);
```

## After

```cuda
#pragma unroll
for (int m = 0; m < kDim/kChunk; ++m) {
    Reg kr = load_global_frag<FragView::Transposed>(args.kr+tile_base,0,m*kChunk,lane);

    // Recompute the rounded correction for every state-product consumer.
    u = Acc{};
    mma(u,inv,transpose(residual));
    store_c<kChunk>(s.correction,quantize(u),0,value,lane);
    __syncwarp(kAllLanes);
```
