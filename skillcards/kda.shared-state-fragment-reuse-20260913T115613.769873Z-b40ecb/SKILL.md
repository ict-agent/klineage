---
skill_id: kda.shared-state-fragment-reuse
intent: Reuse state fragments across independent matrix products through loop fusion.
preconditions:
- 'Data types: no specific dtype is required by register reuse; both consumers must
  accept identical operand bits, and each independent reduction must retain its arithmetic
  and rounding order.'
- 'Layout: corresponding loads must yield the same operand fragment for the same thread,
  with a register representation accepted by both consumers; otherwise direct forwarding
  supplies the wrong elements.'
- 'Storage: no particular source memory level is required; both consumers must reload
  the same unchanged operand so one retained register value can replace both reads.'
- 'Pipeline: state producers must finish and publish their writes before either product
  reads; both products must finish before state mutation or buffer reuse, and neither
  reduction may depend on the other intermediate result.'
- 'Hardware: no additional feature or shared-memory capacity is required; ordinary
  registers retain the existing fragment between its two compute instructions.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse the key-state and query-state reduction loops in
`solution/recurrence.cuh::recur_tile`. Load each state fragment once and pass the
same `Reg sb` to both `mma` calls. Replace only the two loops between `Acc u, o;`
and `Reg out_bf = quantize(o);` with the After snippet.

Each warp retains its value tile. For each increasing K step, load `ka`, `qa`,
and `sb`, then update `u` followed by `o`. Keep every accumulator's original MMA
sequence and subsequent quantization. The two products are independent until
both complete, so their interleaving changes no reduction grouping or rounding.
Do not retain operands across K steps.

Keep `load_a`, `load_b`, fragment layouts, volatile scalar load helpers, launch
parameters, and all barriers unchanged. The compiler cannot remove the duplicate
loads in Before because `load_b` reaches `ld.volatile.shared.b16`; forwarding
`sb` explicitly removes that second load without weakening the helper.

State initialization is published by the existing CTA barrier. Later chunks
read the previous chunk's completed state after CTA synchronization. State
updates follow both products; the chunk barriers finish readers before storage
reuse. Fusion introduces no new communication or synchronization. Existing full
tiles require no new bounds handling; preserve all other load/store guards and
beta zero fill. Keep the caller stream, tensor ABI, complete problem, compiler
flags, and numerical checks unchanged.

Evaluate the supplied workload with `klineage.harness.evaluate`. Inspect generated
code to confirm one state-fragment load per K step and both MMA chains.

## Example configuration

The workload has batch 1, 4096 tokens, 96 heads, and dimension 128. Chunks contain
16 tokens; each head processes 256 chunks. Recurrence launches `dim3(1,kHeads)`
with 256 threads: eight 32-lane warps, each owning 16 value rows. Each product has
eight K steps. Preparation remains `dim3(kTiles,kHeads)`, 256 threads.

`kd` and `qd` are BF16 `[kChunk,kDim]` matrices; state is BF16 `[kDim,kDim]` in
value-key order. Shared addressing remains
`offset<Rows>(r,c) = r*8 + (c&7) + (c/8)*Rows*8`.
`load_a` and `load_b` retain their existing per-lane fragment mappings. Each
`Reg` contains four packed 32-bit words; each `Acc` contains eight FP32 values.
The helper retains two `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
instructions per product step, followed by the existing BF16 rounding points.
These instruction shapes and dtypes belong to the retained computation.

Preserve the single `InputShared` buffer, shared correction matrix, scalar
fragment transfers, shared-memory lane exchanges, chunk recurrence, and
triangular preparation. `PrepareShared`, `InputShared`, and `RecurShared` remain
42368, 18048, and 124672 bytes; static transpose scratch and launch bounds also
remain unchanged. No workspace, register-budget setting, or allocation changes.

# Precondition

- Data types: no specific dtype is required to reuse a register operand. Both
  consumers must accept the same bits; keep each independent reduction's
  operations and rounding order to avoid numerical changes.
- Layout: corresponding loads must produce the same thread's fragment, in a
  representation both consumers already accept. A different lane ownership or
  representation would require redistribution instead of direct register reuse.
- Storage: no particular source memory level is required. Both products must
  reload the same operand, unchanged between corresponding uses; otherwise
  the second consumer would receive a stale register value.
- Pipeline: producers must complete and make state visible before reads. Neither
  reduction may depend on the other's intermediate result, and both must finish
  before state mutation or buffer reuse. These dependencies permit interleaving
  while preventing incomplete reads and premature overwrites.
- Hardware: no additional feature or shared-memory capacity is required.
  Ordinary per-thread registers hold the already loaded fragment through both
  compute instructions; existing load and arithmetic support is sufficient.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both snippets use the existing `in`, `s`, `value`, `lane`, and zero-initialized
`Acc u, o`. Replace the complete two-loop block; leave surrounding code intact.

## Before

```cuda
// Compute each product separately, reloading its state fragments.
#pragma unroll
for (int k = 0; k < kDim/kChunk; ++k) {
    Reg ka = load_a<kChunk>(in.kd,0,k*kChunk,lane);
    Reg sb = load_b<kDim>(s.state,value,k*kChunk,lane);
    mma(u,ka,sb);
}

#pragma unroll
for (int k = 0; k < kDim/kChunk; ++k) {
    Reg qa = load_a<kChunk>(in.qd,0,k*kChunk,lane);
    Reg sb = load_b<kDim>(s.state,value,k*kChunk,lane);
    mma(o,qa,sb);
}
```

## After

```cuda
// Interleave k@state and q@state using only current-step operands.
#pragma unroll
for (int k = 0; k < kDim/kChunk; ++k) {
    Reg ka = load_a<kChunk>(in.kd,0,k*kChunk,lane);
    Reg qa = load_a<kChunk>(in.qd,0,k*kChunk,lane);
    Reg sb = load_b<kDim>(s.state,value,k*kChunk,lane);
    mma(u,ka,sb);
    mma(o,qa,sb);
}
```
