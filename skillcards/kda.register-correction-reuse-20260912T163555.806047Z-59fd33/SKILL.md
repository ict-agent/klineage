---
skill_id: kda.register-correction-reuse
intent: Reuse corrected-value fragments across matrix products in warp registers.
preconditions:
- 'Data types: bit-preserving 16-bit intermediates for movmatrix.b16; retain producer
  conversions and arithmetic order to preserve rounding.'
- 'Layout: producer and all consumers share a full warp with matching packed 8-by-8
  fragment ownership; shared writer and transposed reader describe the same tile,
  or register forwarding selects wrong values. No added global contiguity or alignment
  is needed for register operands.'
- 'Storage: warp-exclusive shared correction scratch has no external readers or observable
  stores, and all uses share one invocation; otherwise removing the scratch loses
  values or exceeds register lifetime.'
- 'Pipeline: producers publish before readers, readers finish before reuse, and every
  lane executes the register transpose in producer-to-consumer order within one invocation.
  Preserve independent buffer synchronization to maintain visibility and safe reuse.'
- 'Hardware: movmatrix.sync.aligned.m8n8.trans.b16 support and a live-register budget
  covering E*S/(W*sizeof(uint32_t)) correction words per lane plus other live values,
  for E correction elements per W-lane warp and S bytes per element; insufficient
  capacity causes spills. No extra shared memory is needed.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Keep the corrected-value tile in warp registers between its inverse product,
output product, and recurrent-state updates. This removes a shared-memory store,
warp publication barrier, and repeated transposed matrix loads.

Apply these edits to `solution/recurrence.cuh::recur_tile`:

1. Declare `Reg ubf[2]` after `beta1`. Replace the correction `store_c` with
   `ubf[i] = quantize(u[i])`. Preserve the preceding inverse MMA and every rounding.
2. Remove only the `__syncwarp(kAllLanes)` immediately after this loop.
3. Declare `Reg ub[2]` after `mqk`. Replace the output loop's `load_correction`
   with `ub[i] = transpose(ubf[i])`; pass `ub[i]` to its MMA.
4. Remove the state-update loop's `load_correction`; pass `ub[bi]` to its MMA
   for every `m`. Keep `ub` live and unmodified through the final state update.
5. Delete `load_correction`. In `solution/native.cuh::RecurShared`, delete
   `correction` and its comment. Set the size assertion to `160768`.

`store_c` currently writes a token-by-value matrix in eight-column slabs:
`offset<Rows>(r,c) = r*8 + (c%8) + (c/8)*Rows*8`.
`load_correction` loads the same four submatrices with transposition.
The existing `transpose(Reg)` performs the equivalent bit permutation using one
`movmatrix.sync.aligned.m8n8.trans.b16` per register word. Use it unchanged.

Within each 8-by-8 submatrix, lane `l` initially owns row `l/4` and columns
`2*(l%4)` and `2*(l%4)+1`. `Reg.x[0..3]` orders the submatrices as
top-left, bottom-left, top-right, bottom-right. Transpose each word without
reordering these four words; this reproduces the existing B-operand layout.

Each compute warp produces and consumes its own value columns. No other warp
reads its correction. The replacement changes only intermediate storage and its
register permutation; all matrix operations, reduction order, BF16 quantization,
residual arithmetic, and output/state conversions stay unchanged.

Keep the input-stage completion wait, `compute_sync()`, async fences, output
publication, and input/output reuse barriers. The removed warp barrier serves
only the deleted correction stores. Registers are produced before consumption
in the same invocation; the next invocation creates a fresh correction.

Rebuild from the edited source bundle with the preserved compile flags. Check
correctness and latency using `klineage.harness.evaluate` on the supplied problem;
keep measurements outside this card.

## Example configuration

Preserve batch 1, 4096 tokens, 96 heads, head dimension 128, and chunks of 16.
There are 256 chunks per head. BF16 operands/corrections and FP32 accumulators
retain all existing rounding points, fast activations, scale, gate bound, and
normalization epsilon. Keep the complete problem, workload, oracle, and compiler
flags unchanged.

The recurrence has 192 threads: four compute warps, one load warp, and one store
warp. Compute warp `w` owns the 16-by-32 correction slice starting at value column
`w*32`, split into two 16-by-16 fragments. All indices are in bounds for this
fixed configuration; add no predicates or early returns. Each lane retains eight
32-bit correction words across the consumer loops.

Keep three input stages and two output stages, persistent traversal of all chunks
within one CTA per head, MMA fragment adapters, K-step prefetching, and separate
FP32 state-conversion scratch. The removed correction allocation is 4096 bytes;
`RecurShared` shrinks from 164864 to 160768 bytes. Preparation and input-stage
storage remain 42368 and 18048 bytes respectively.

No host edit is needed: `solution/native.cu::launch` already sets the dynamic
shared-memory attribute and allocation from `sizeof(RecurShared)`. Preserve the
recurrence grid `(1,kHeads)`, block `kRecurThreads`, preparation grid
`(kTiles,kHeads)`, block `kPrepareThreads=256`, and caller stream. Retain all tensor
maps, transfer groups, transaction counts, and ABI checks.

# Precondition

- Data types: the intermediate has a bit-preserving 16-bit representation;
  `movmatrix...b16` transposes 16-bit elements. Preserve the producer's conversion
  and subsequent arithmetic order so eliminating stores does not change rounding.
  BF16 arithmetic itself is not required by this bit permutation.
- Layout: each correction's producer and all consumers belong to the same full
  warp, with known packed 8-by-8 fragment ownership matching the register transpose.
  Otherwise local registers cannot supply every reader or the permutation selects
  the wrong elements. The shared writer and transposed reader must describe the
  same logical tile. No global contiguity or additional alignment is required;
  the replacement accesses registers.
- Storage: the intermediate currently passes through warp-exclusive shared
  scratch, with no other readers or observable use of those stores. Eliminating
  externally consumed scratch would lose data. All uses must fit within the
  producer warp's invocation, where register values can remain live.
- Pipeline: shared producers complete and publish their values before readers,
  and readers finish before scratch reuse. Producer and consumers execute in
  order within one invocation, with every lane participating in the collective
  transpose. These conditions let register dependencies replace the scratch
  publication barrier without breaking visibility, collective participation, or
  lifetime. Independent input/output buffer synchronization must remain.
- Hardware: the target supports `movmatrix.sync.aligned.m8n8.trans.b16` and has
  sufficient registers for the retained corrections alongside other live values.
  For `E` correction elements per warp of `W` lanes and element size `S` bytes,
  the correction needs `E*S/(W*sizeof(uint32_t))` register words per lane under
  this even mapping. Exceeding the available live-register budget causes spills
  and defeats register residency. The forward change needs no extra shared memory.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These excerpts are the changed sites, not a replacement function. Keep omitted
arithmetic and loops. The numbered edits above specify declaration placement and
scratch/helper deletion.

## Before

```cuda
// native.cuh: RecurShared member, then its external size assertion.
alignas(128) BF16 correction[kTileElems];
static_assert(sizeof(RecurShared) == 164864);

// recurrence.cuh: helper used by both consumers.
__device__ __forceinline__ Reg load_correction(const RecurShared& s, int value, int lane) {
    return ld_trans(s.correction + offset<kChunk>(lane % 16, value + lane / 16 * 8));
}

// recur_tile: end of the inverse-product loop, followed by publication.
u[i] = Acc{};
mma(u[i],inv,transpose(residual));
store_c<kChunk>(s.correction,quantize(u[i]),0,value+i*kChunk,lane);
// After that loop:
__syncwarp(kAllLanes);

// Output-product loop.
Reg correction = load_correction(s,value+i*kChunk,lane);
o[i] = Acc{};
mma(o[i],mqk,correction);

// Inner state-update loop, repeated for each m.
u[bi] = Acc{};
Reg correction = load_correction(s,value+bi*kChunk,lane);
mma(u[bi],kr,correction);
```

## After

```cuda
// native.cuh: delete the correction member; update the external assertion.
static_assert(sizeof(RecurShared) == 160768);

// recurrence.cuh: delete load_correction. In recur_tile, after beta1:
Reg ubf[2];

// End of the inverse-product loop; delete its following publication barrier.
u[i] = Acc{};
mma(u[i],inv,transpose(residual));
ubf[i] = quantize(u[i]);

// After mqk, before the output-product loop:
Reg ub[2];

// Output-product loop: transpose once and retain the B operands.
ub[i] = transpose(ubf[i]);
o[i] = Acc{};
mma(o[i],mqk,ub[i]);

// Inner state-update loop: reuse the same operands for every m.
u[bi] = Acc{};
mma(u[bi],kr,ub[bi]);
```
