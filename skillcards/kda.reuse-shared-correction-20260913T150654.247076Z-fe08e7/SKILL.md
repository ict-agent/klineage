---
skill_id: kda.reuse-shared-correction
intent: Reuse an invariant correction matrix across recurrence products.
preconditions:
- 'Data types: repeated correction evaluations must produce identical stored bits
  under the required rounding; reuse needs no particular dtype but cannot change those
  bits.'
- 'Layout: producers and repeated consumers must address the same logical correction
  elements, with ownership excluding unrelated writers; existing valid accesses need
  no new contiguity or alignment.'
- 'Storage: the correction is already materialized in shared memory visible to its
  consumers, and its allocation remains available through the last read; otherwise
  omitted writes cannot supply the value.'
- 'Pipeline: correction operands remain invariant across consumers; the existing publication
  barrier completes writes and establishes visibility for converged participants,
  readers finish before each overwrite, and only identical recomputations may write
  the tile through its final consumer, without unrelated buffer reuse.'
- 'Hardware: no additional features or capacity are required because reuse retains
  the existing shared allocation, accesses, and warp barrier while eliminating repeated
  computation.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reuse the rounded correction matrix across the readout and state-update
products in `solution/recurrence.cuh::recur_tile`. Its inverse and residual
operands are invariant throughout these consumers. Compute the matrix once,
publish it in `s.correction`, and retain it until every consumer has read it.

The deoptimized function first forms the correction before loading `mqk`, then
repeats the same calculation inside every iteration of the state-update `m`
loop. Remove only the repeated block between the `kr` load and the accumulator
reset preceding `load_correction`. Keep the first producer, including its
`__syncwarp(kAllLanes)`, before the readout consumer. Keep every
`load_correction`, the state-product accumulator reset, and subsequent state
arithmetic and stores. No helper, allocation, launch, or compiler-flag changes
are needed.

Each warp owns one value-column tile of the shared correction matrix. The
existing `store_c` writer and `load_correction` reader provide its normal and
transposed fragment views. Their indices remain unchanged. The publication
barrier completes the producer's writes before other lanes read them. The
optimized region never overwrites this tile; the surrounding recurrence CTA
finishes its reads before its shared storage can be reused. Different warps
write disjoint columns.

Preserve the inverse and residual values, zero-initialized FP32 accumulator,
MMA sequence, and BF16 round-to-nearest quantization. This transformation only
eliminates repeated evaluations of that same stored result. Keep the preceding
normalization, beta rounding, gate evaluation, and subsequent state arithmetic.
Existing volatile MMA, conversion, and shared-store instructions make the
deoptimized repetitions observable to the compiler; retain those helpers.

After replay, inspect generated code for one correction product per warp and
no correction production in the state-update loop. Check correctness and timing
with `klineage.harness.evaluate` using the preserved problem and policy.

## Example configuration

The supplied problem has batch 1, 4096 tokens, 96 heads, and head dimension 128.
`kChunk=16`, `kTiles=256`, and `kDim=128`. Both kernels use 256 threads;
recurrence has eight compute warps. Preparation launches `(kTiles,kHeads)`
CTAs, then each chunk launches `(1,kHeads)` on the caller stream. Keep this
ordering and the BF16 global carry between chunk launches.

The shared correction is row-major `[kChunk,kDim]`, with row stride
`kDim*sizeof(BF16)`. Warp `w` owns columns `[w*kChunk,(w+1)*kChunk)` and all
chunk rows: a 16-by-16 tile, 512 bytes. The full correction allocation is
4096 bytes inside the unchanged 124672-byte `RecurShared`. Keep its alignment,
other shared fields, and the existing transpose scratch.

For lane `l`, fragment `j`, and pair element `e`, `store_c` writes row
`l/4+(j%2)*8`, column `value+(l%4)*2+(j/2)*8+e`.
`load_correction` reads row `(j%2)*8+(l%4)*2+e`, column
`value+(j/2)*8+l/4`. Preserve these existing adapters and `value=warp*kChunk`.
All lanes participate; exact tile divisibility makes these accesses in bounds.
Keep existing bounds checks elsewhere.

One warp computes a correction for one readout and eight state products.
Forward application reduces nine correction evaluations to one per warp.
Each evaluation uses the retained two
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instructions, FP32
accumulation, shared fragment transpose, scalar BF16 conversions, and scalar
shared stores. Retain the SM90a target, all compile flags, ABI checks, input
layouts, workload, oracle, and numerical tolerances. None of these instance
dimensions or MMA choices is a new prerequisite for invariant-value reuse.

# Precondition

- Data types: each repeated correction evaluation must produce the same stored
  bits under the required rounding. Reading one saved evaluation is then
  equivalent to reading each recomputation. No particular dtype is required;
  changing representation or rounding during reuse would break that equivalence.
- Layout: the producer and every repeated consumer must address the same
  logical correction elements, and ownership must exclude unrelated writers.
  Otherwise a saved element could be misindexed or replaced. The existing
  accesses remain valid without additional contiguity or alignment constraints.
- Storage: the preceding implementation already materializes the correction
  in shared memory accessible to its consumers. That allocation must remain
  available through the final read; eliminating later writes cannot recover an
  unavailable or overwritten value.
- Pipeline: correction operands must remain invariant across all consumers.
  The existing publication barrier must follow completed producer writes,
  establish reader visibility, and be reached by its required participants.
  Readers must finish before each overwrite. Through the final consumer, only
  the identical recomputations may write this tile; unrelated writes or buffer
  reuse would invalidate the saved value. These rules prevent stale, incomplete,
  or replaced values; no particular stage count is required.
- Hardware: no additional feature or capacity is needed. The transformation
  retains the existing shared allocation, accesses, and warp barrier and removes
  repeated computation; it introduces no new instruction or storage demand.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These contiguous snippets start at the `kr` load inside `recur_tile`'s state
update loop. Keep the rest of the loop unchanged. In both versions, retain the
existing correction producer and its publication barrier before the `mqk` load.

## Before

```cuda
Reg kr = load_global_frag<FragView::Transposed>(args.kr+tile_base,0,m*kChunk,lane);

// Recompute the rounded correction for every state-product consumer.
u = Acc{};
mma(u,inv,transpose(residual));
store_c<kChunk>(s.correction,quantize(u),0,value,lane);
__syncwarp(kAllLanes);

u = Acc{};
Reg correction = load_correction(s,value,lane);
mma(u,kr,correction);
```

## After

```cuda
Reg kr = load_global_frag<FragView::Transposed>(args.kr+tile_base,0,m*kChunk,lane);
u = Acc{};
Reg correction = load_correction(s,value,lane);
mma(u,kr,correction);
```
