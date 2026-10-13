---
skill_id: kda.partial-readout-register-retention
intent: Retain the partial readout in registers across correction computation.
preconditions:
- 'Data types: no additional dtype restriction; retain the stored representation and
  producer-side rounding so arithmetic is unchanged.'
- 'Layout: each fragment element has the same producer and consumer thread and slot
  interpretation; private forwarding otherwise selects the wrong data. No additional
  contiguity or alignment is needed when deleting transfers.'
- 'Storage: the shared intermediate serves only its later same-thread consumer, with
  no other required observer of the write; deleting another reader''s publication
  would lose data. No additional shared capacity is needed.'
- 'Pipeline: producer and consumer execute in order in one kernel invocation with
  an unchanged shared temporary, consumed before buffer overwrite; preserve shared
  publication, visibility, and reuse barriers for other data to prevent incomplete
  or overwritten reads.'
- 'Hardware: the retained fragment and other live state must fit per-thread and per-block
  register allocation, including allocation granularity; otherwise spills defeat residency
  or the launch exceeds limits. Retention needs no special matrix or copy instruction.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Retain the rounded partial readout in each thread's registers across correction
computation, eliminating its intermediate shared-memory store and reload.

In `solution/recurrence.cuh`, edit only `recur_tile`:

1. Immediately after the query/state product loop, replace
   `store_c<kChunk>(out,quantize(o),0,value,lane);` with
   `Reg out_bf = quantize(o);`.
2. After `Reg prod = quantize(o);`, remove
   `Reg out_bf = load_a<kChunk>(out,0,value,lane);`.
3. Update the two staging comments. Keep the intervening correction computation,
   final `add_pair` loop, and final `store_c` unchanged.

The deleted transfers use `st_scalar`/`ld_scalar` with volatile shared PTX.
Remove these call sites only; retain the helpers and their compiler controls
for every other consumer. No compile-flag changes are needed.

`out` aliases `s.out`, a row-major `[kChunk,kDim]` array. Warp `w` owns the
value columns beginning at `value=w*kChunk`. For lane `l`, fragment `j`, and
half `e`, both the old store and reload address
`r=l/4+(j%2)*8`, `c=value+(l%4)*2+(j/2)*8+e`.
Thus each lane can retain exactly the bits it previously reloaded. The final
shared output is still needed by the cooperative global store; do not remove it
or shrink `RecurShared`.

The partial readout has no cross-thread reader. Same-thread instruction order
already orders its store, reload, and final overwrite, so neither edit changes
barriers. Preserve the correction publication `__syncwarp(kAllLanes)`, the
transpose helper's scratch barriers, and the CTA barrier before `store_output`.
These protect separate shared consumers and scratch reuse. Preserve kernel
launch order on the caller stream, which orders chunk-state handoffs.

Keep `quantize(o)` at the original producer: retaining FP32 accumulators until
the final addition would change rounding. Retain product traversal, BF16
conversions, scalar BF16 `add_pair`, and all output semantics. Launch dimensions,
shared allocation, indexing, and bounds handling do not change.

Inspect compiled recurrence code to confirm the intermediate shared store/load
is absent and the fragment survives in registers without local spills. Validate
with the existing evaluator and the unchanged problem and timing policy.

## Example configuration

This replay uses batch 1, 4096 tokens, 96 heads, head dimension 128, and chunks
of 16 tokens. There are 256 recurrence launches, each with grid `(1,96)` and
256 threads. Eight warps each compute a 16-by-16 readout tile. Each lane owns
eight BF16 elements packed into four 32-bit `Reg` words; retention extends
these four words' lifetime across the correction products.

Keep BF16 operands/intermediates, FP32 MMA accumulation, and round-to-nearest
BF16 quantization. Retain the two `mma.sync.aligned.m16n8k16` operations per
16-by-16 product step, eight reduction steps, separate state-product loops,
scalar volatile fragment transfers, shared correction staging, and shared
transpose scratch. These are surrounding mechanisms, not requirements of
register retention.

Keep `sizeof(RecurShared)=124672`, including the 4096-byte output array, and
the existing static transpose scratch. Preserve preparation, buffer allocation,
the tensor ABI, caller stream, `config.toml`, compile flags, SM90 target, and
all numerical requirements in the problem.

# Precondition

- Data types: no additional dtype restriction. Retained register bits must
  represent exactly the previously stored value, including any producer-side
  rounding; moving quantization to the consumer would change arithmetic.
- Layout: each consumed fragment element must have the same producing and
  consuming thread and register-slot interpretation. Otherwise direct private
  retention substitutes the wrong element or requires a separate exchange.
  No additional memory contiguity or alignment is required because the removed
  transfers, rather than a new memory instruction, determine the old mapping.
- Storage: the intermediate is stored in shared memory solely for the later
  same-thread consumer; its write has no other required observer. Removing a
  publication used by another thread would leave that reader without data.
  No additional shared-memory capacity is required.
- Pipeline: producer and consumer execute in order within one kernel invocation,
  and the shared temporary is unchanged between them. Its consumer finishes
  before the buffer is overwritten. Any shared publication, visibility barrier, or
  buffer-reuse barrier serving other data must remain; deleting those can expose
  incomplete or overwritten values even though this intermediate is private.
- Hardware: per-thread and per-block register allocation must accommodate the
  retained fragment plus the other simultaneously live state, accounting for
  allocation granularity. Otherwise spilling defeats register residency or the
  chosen launch exceeds register limits. No special matrix or copy instruction
  is required by retention itself.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are two excerpts from `recur_tile`; the intervening block stays in place.
The final addition and output store below are shared context, unchanged.

## Before

```cuda
// Immediately after the query/state product loop.
store_c<kChunk>(out,quantize(o),0,value,lane);

// ... existing value, residual, inverse, and correction computation ...

mma(o,mqk,correction);
Reg prod = quantize(o);
Reg out_bf = load_a<kChunk>(out,0,value,lane);
#pragma unroll
for (int j = 0; j < 4; ++j) {
    Pair a{out_bf.x[j]}, b{prod.x[j]}, c;
    c.u = add_pair(a.u,b.u);
    out_bf.x[j] = c.u;
}
store_c<kChunk>(out,out_bf,0,value,lane);
```

## After

```cuda
// Retain the rounded readout across correction computation.
Reg out_bf = quantize(o);

// ... existing value, residual, inverse, and correction computation ...

mma(o,mqk,correction);
Reg prod = quantize(o);
#pragma unroll
for (int j = 0; j < 4; ++j) {
    Pair a{out_bf.x[j]}, b{prod.x[j]}, c;
    c.u = add_pair(a.u,b.u);
    out_bf.x[j] = c.u;
}
store_c<kChunk>(out,out_bf,0,value,lane);
```
