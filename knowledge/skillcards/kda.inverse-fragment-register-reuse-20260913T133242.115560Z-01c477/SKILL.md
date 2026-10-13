---
skill_id: kda.inverse-fragment-register-reuse
intent: Reuse the diagonal inverse fragment in registers across block-merge consumers.
preconditions:
- 'Data types: No additional dtype restriction; retain loaded bits without conversion
  so every consumer receives the same representation.'
- 'Layout: Consumers must request the same per-thread elements in the same order;
  changed ownership or indexing would make the cached fragment incorrect. No additional
  alignment or contiguity is needed because the first load is unchanged.'
- 'Storage: Repeated reads must access an already produced, stable operand; caching
  cannot supply intervening updates. No particular source memory level is intrinsic
  to retaining these bits.'
- 'Pipeline: Producer writes must be visible before the first load, which must precede
  every consumer. The operand must remain unchanged through its last use, and consumers
  must finish before storage reuse; otherwise caching can expose stale or uninitialized
  data.'
- 'Hardware: Ordinary per-thread registers suffice. The retained fragment and overlapping
  live values must fit the allocatable register budget to keep the cache in registers
  rather than spill it to local memory.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Retain the diagonal inverse fragment across its three block-merge consumers in
`solution/prepare.cuh`, function `invert()`. Delete the two assignments that reload
`p` after its declaration, plus their reload comment. Keep the initial
`Reg p = load_a<kChunk, kChunk>(s.inv,0,0,tid);` and all consumer expressions.
The first product reads `p`, the second reads `transpose(p)`, and the final
elementwise addition reads `p.x[j]`. `transpose` takes its argument by value;
none of these consumers modifies `p` or `s.inv`.

The first warp owns this operation. `s.inv` contains the two diagonal inverse
blocks in contiguous row-major storage. For lane `l`, word `j`, and half `e`,
the fragment owns row `l/4 + (j%2)*8` and column
`(l%4)*2 + (j/2)*8 + e`. Retain this mapping for all consumers. Load the fragment
once after the existing producer `__syncwarp()`. Keep the barriers inside
`transpose()` for its separate scratch buffer, and keep the final `store_c()`
after the last use. The backing inverse is unchanged throughout the merge.
No launch, allocation, layout, barrier, or bounds change is needed.

Retain `load_a` and its volatile scalar loads in `solution/native.cuh`; they
remain necessary elsewhere. No compiler-flag change is required. Removing only
the two reload sites keeps the initial fragment live. Preserve every BF16
conversion, scalar addition, FP32 product accumulation, sign change, and their
order. This transformation changes data reuse only.

Inspect generated code for one inverse-fragment load group per executed path
instead of three; the compiler may duplicate control-flow paths. Check that
retained values feed all three consumers without spills. Use the
registered evaluator with the unchanged problem for correctness and timing.

## Example configuration

The workload is BATCH=1, TOKENS=4096, HEADS=96, HEAD_DIM=128, with 16-token
chunks and 256 chunks per head. Keep both preparation and recurrence launches
at 256 threads. Preparation uses `grid=(256,96)`; recurrence launches
`grid=(1,96)` once per chunk on the caller stream. Only preparation's first
32-thread warp enters `invert()`; its fragment coordinates are within the
complete 16-by-16 inverse matrix, so no edge predicate changes are needed.

Each fragment is four packed 32-bit words holding eight BF16 elements.
Inverse rows have a 32-byte stride. The three loads each read eight scalar
shared-memory elements per lane. The forward change retains the first fragment
for the remaining two consumers. Keep the two 8-by-8 FP32 substitutions,
BF16 block merge, `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`,
scalar conversion/addition helpers, and shared-memory fragment transpose.
These arithmetic and fragment settings belong to this replay, not to register
caching in general.

Keep `PrepareShared` at 41856 bytes, `InputShared` at 18048 bytes, and
`RecurShared` at 124672 bytes, including reserved slots. Keep the inverse-row
and transpose scratch arrays. Preserve `__launch_bounds__(kPrepareThreads,8)`,
all compile flags, host ABI, caller-stream ordering, normalization grouping,
gate evaluation, global workspace, and chunk-state handoff. The scale,
gate bound, workload, oracle, and numerical tolerances remain unchanged.

# Precondition

- Data types: No additional dtype restriction. Preserve the loaded representation
  exactly; conversion while caching could change the values received by later
  consumers. The arithmetic remains untouched.
- Layout: Every consumer must require the same elements, in the same order,
  owned by the same thread. Otherwise its register fragment cannot substitute
  for the later load. No additional alignment or contiguity is required because
  the initial load and its indexing remain unchanged.
- Storage: The repeated reads must address an already produced, stable operand.
  Caching cannot reflect intervening writes. The source memory level imposes no
  additional restriction on retaining the bits.
- Pipeline: Producer completion and visibility must precede the initial load,
  and that load must execute before every consumer. Keep the operand unchanged
  through its last use, and complete consumers before storage reuse. These
  dependencies prevent stale values, uninitialized registers, and overwritten
  inputs. Retain synchronization that establishes those dependencies.
- Hardware: Ordinary per-thread registers suffice; no special instruction is
  introduced. Require `R_fragment + R_other_live <= R_allocatable` over the
  extended lifetime to retain the cache in registers. Exceeding that budget can
  replace eliminated source loads with local-memory spills.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
    Reg p = load_a<kChunk, kChunk>(s.inv,0,0,tid);
    Reg m = load_a<kChunk>(s.ki,0,0,tid);
    Acc dc;
    mma(dc,p,transpose(m));
    #pragma unroll
    for (int j = 0; j < 8; ++j) dc.x[j] *= -1.0f;
    Acc result;
    // Reload the diagonal inverse for each consumer; scalar loads are volatile.
    p = load_a<kChunk, kChunk>(s.inv,0,0,tid);
    mma(result,quantize(dc),transpose(p));
    Reg o = quantize(result);
    p = load_a<kChunk, kChunk>(s.inv,0,0,tid);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        Pair a{p.x[j]}, b{o.x[j]}, c;
        c.u = add_pair(a.u,b.u);
        o.x[j] = c.u;
    }
    store_c<kChunk, kChunk>(s.inv,o,0,0,tid);
```

## After

```cuda
    Reg p = load_a<kChunk, kChunk>(s.inv,0,0,tid);
    Reg m = load_a<kChunk>(s.ki,0,0,tid);
    Acc dc;
    mma(dc,p,transpose(m));
    #pragma unroll
    for (int j = 0; j < 8; ++j) dc.x[j] *= -1.0f;
    Acc result;
    mma(result,quantize(dc),transpose(p));
    Reg o = quantize(result);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        Pair a{p.x[j]}, b{o.x[j]}, c;
        c.u = add_pair(a.u,b.u);
        o.x[j] = c.u;
    }
    store_c<kChunk, kChunk>(s.inv,o,0,0,tid);
```
