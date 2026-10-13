---
skill_id: kda.residual-register-retention
intent: Retain residual fragments in registers across correction products.
preconditions:
- 'Data types: no additional dtype restriction; retain the stored fragment''s bits
  and all preceding arithmetic and rounding, since only its storage lifetime changes.'
- 'Layout: each consumer lane must reuse exactly the fragment produced by that lane;
  eliminating its shared reload cannot supply values owned by another lane.'
- 'Storage: the shared intermediate must have no readers beyond its producing lane''s
  reloads; otherwise removing its stores would leave those readers without data.'
- 'Pipeline: production must precede every use, the fragment must remain unchanged
  through its final consumer, and all readers must finish before storage reuse. Preserve
  ordering for any subsequent cross-lane exchange.'
- 'Hardware: available registers must hold the retained fragment plus other live values
  within per-thread and per-block allocation limits; spills would defeat register
  residency.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Keep each lane's rounded residual fragment in registers across the inverse
products in `solution/recurrence.cuh::recur_tile`. This removes repeated shared
loads of an unchanged intermediate.

The deoptimized code computes `residual` once, then writes it through
`store_c<kChunk>(s.input.v,residual,0,value,lane)`. Each inverse product reloads
it with `load_a<kChunk>(s.input.v,0,value,lane)` before `transpose`.

Delete that store and replace both occurrences of
`transpose(load_a<kChunk>(s.input.v,0,value,lane))` with `transpose(residual)`.
One occurrence precedes the state-update loop; the other is inside it. Retain
the existing `Reg residual` declaration and its complete elementwise construction.
Its unchanged value then remains live through every inverse product.

`s.input.v` uses row-major `[token, value]` indexing. Warp `w` owns value columns
`w*kChunk .. (w+1)*kChunk-1`. For fragment word `j` and packed element `e`, lane
`l` stores and reloads
`r=l/4+(j%2)*8`, `c=value+(l%4)*2+(j/2)*8+e`.
These matching writer/reader coordinates establish lane ownership. The later
`transpose` still exchanges fragments through its own shared scratch.

No launch, bounds, allocation, or synchronization change is needed. Leave
`InputShared::v` allocated as reserved storage and update its enclosing comment
in `solution/native.cuh` accordingly. Preserve every existing warp/block barrier,
including both barriers around each `transpose` scratch exchange. Every lane
publishes before scratch reads and finishes reading before scratch reuse.

Preserve residual subtraction, beta activation, BF16 rounding, fragment order,
all MMA accumulation, and subsequent correction rounding. Do not recompute the
residual inside the state-update loop: the state changes there. The retained
fragment represents the state before those updates.

Check generated code for removal of the residual stores/reloads and for spills
that could defeat register retention. Validate with `klineage.harness.evaluate`
on the preserved problem, workload, oracle, and numerical/timing policy.

## Example configuration

This bundle uses B=1, 4096 tokens, 96 heads, dimension 128, and chunks of 16.
Both kernels launch 256 threads. Preparation uses grid `(256,96)`; 256 ordered
recurrence launches each use grid `(1,96)` on the caller stream.

Eight warps each own one 16-column value tile. `Reg` contains four 32-bit words,
each packing two BF16 elements. Each lane retains eight residual elements for
nine inverse products: one readout correction and eight state corrections.
All lanes participate; the supplied dimensions have no partial chunks or tiles.

The reserved `s.input.v` buffer has 2048 BF16 elements, 4096 bytes, beginning at
byte 32768 of `RecurShared`. The residual mapping covers it once without overlap.
Keep `sizeof(RecurShared)==124672`, `sizeof(InputShared)==18048`, and
`sizeof(PrepareShared)==41856`; the launch already requests these dynamic sizes.

Retain BF16 inputs/intermediates/output, FP32 accumulators and final-state ABI,
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, shared state/correction/output
staging, scalar volatile fragment transfers, and the shared-scratch transpose.
Retain inverse reloads and correction recomputation inside the key-block loop.
The kernel's compile flags, chunk ordering, normalization, gate preparation,
global workspace, and state handoff remain unchanged.

# Precondition

- Data types: no additional dtype restriction. Register retention moves no
  arithmetic across a rounding point; preserve the fragment's stored bits and
  the arithmetic that produced them. Removing a conversion would change this
  transformation into a numerical change.
- Layout: every eliminated shared reload must reproduce the same lane's own
  fragment with the same element order. Otherwise that lane's registers do not
  contain the values the reload supplied. Later exchanges may remain unchanged.
- Storage: the intermediate's shared stores must serve only these lane-local
  reloads. Any other reader would lose its producer when the stores disappear.
  Unrelated shared buffers remain available to their existing consumers.
- Pipeline: finish production before consumption; retain the unchanged fragment
  until its final use and finish every reader before reusing its storage.
  Preserve publication, visibility, and read-completion ordering in subsequent
  cross-lane exchanges. These are lifetime dependencies, not a stage-count rule.
- Hardware: the retained fragment and other live values must fit available
  registers under both thread and block allocation limits. Otherwise spilling
  reintroduces memory traffic and defeats the intended residency.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These replace the statements immediately after residual construction and the
inverse-product statements inside the existing key-block loop, respectively.
All surrounding code remains unchanged. `residual`, `u`, `inv`, `s`, `value`,
and `lane` already exist in `recur_tile`.

## Before

```cuda
// Immediately after the existing residual construction.
store_c<kChunk>(s.input.v,residual,0,value,lane);
u = Acc{};
mma(u,inv,transpose(load_a<kChunk>(s.input.v,0,value,lane)));
```

```cuda
// Inside the existing state-update loop, after its inverse reload.
u = Acc{};
mma(u,inv,transpose(load_a<kChunk>(s.input.v,0,value,lane)));
```

## After

```cuda
// Immediately after the existing residual construction.
u = Acc{};
mma(u,inv,transpose(residual));
```

```cuda
// Inside the existing state-update loop, after its inverse reload.
u = Acc{};
mma(u,inv,transpose(residual));
```
