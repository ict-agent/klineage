---
skill_id: fmha.dynamic-persistent-cta-scheduling
intent: Reuse a fixed CTA pool through atomic allocation of attention tiles.
preconditions:
- 'Data types: no tensor dtype restriction; scheduling preserves each tile''s arithmetic.
  Tile counts and all allocated indices must fit the integer representation, or counter/index
  overflow can duplicate or omit work.'
- 'Layout: a finite index mapping must identify independent, nonoverlapping output
  tiles, so arbitrary CTA ownership preserves results. No additional tensor contiguity
  or alignment is needed because operand addressing is unchanged.'
- 'Storage: writable grid-visible metadata storage must be available and isolated
  from concurrent resets, so every CTA can share one work allocator. Inputs and tile
  metadata must remain valid until all workers finish.'
- 'Pipeline: metadata initialization must precede acquisition; every lane named by
  the allocation broadcast must participate. Work publication must follow prior consumption,
  and readers must observe publication before computing. Tile buffers may be reused
  only after their readers finish, with matching barrier phases across successive
  tiles.'
- 'Hardware: device-wide integer atomic fetch-add and CUDA warp broadcast are needed
  for unique allocation and agreement within the producer warp. Storage must accommodate
  one naturally aligned counter word; no additional per-CTA shared memory is required
  by this allocator.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Use a fixed pool of CTAs that repeatedly claim attention tiles from a global
counter. This reuses CTA setup and allows workers to claim further tiles as they
finish. The preceding kernel launches one CTA per logical tile, plus bounded
excess CTAs, and each CTA executes at most once.

## Replay

Edit these bundle-relative locations:

1. In `solution/scheduler.cuh`, replace `kMetadata` and `kMaxQTiles` with
   `kCounterOffset = 3 * kBatch` and `kMetadata = kCounterOffset + 1`.
   In `prepare`, immediately after `griddepcontrol.launch_dependents`, have lane
   zero reset the new counter. The existing allocation uses `kMetadata`, so it
   grows automatically. Keep the sequence-order metadata and `tile()` unchanged.
2. In `solution/kernel.cu`, add `int sms = 0;` to `Workspace`. In `ensure`, after
   querying and checking `prop`, assign `sms = prop.multiProcessorCount`.
   Set `config.gridDim = dim3(scratch.sms)`. This chooses a CTA pool equal to the
   SM count; it does not pin individual CTAs to SMs. Keep the block size, shared
   allocation, caller stream, tensor ABI, and launch attributes.
3. Change the outer `if (work.w < kBatch)` to `while (work.w < kBatch)` in both
   `producer` in `solution/scheduler.cuh` and `consumer` in
   `solution/attention.cuh`. Keep `pos` and `iteration` outside these loops.
4. In the producer tail, immediately before the final `load_v`, let lane zero
   claim `atomicAdd(counter, 1) + gridDim.x`. After incrementing `pos` and
   `iteration`, broadcast that index and call `tile(index, p.metadata)` in place
   of the unconditional terminal sentinel. Keep the following work handoff and
   the final output/K/V drain unchanged. Update the direct-mapping comment in
   `prepare` to describe persistent assignment.

The initial claim remains `tile(blockIdx.x, p.metadata)`. With pool size `P`,
these claims cover indices `[0, P)`; atomic claims begin at `P`, so ownership
never overlaps. `tile()` maps an index to sequence, head, and query block in
that order. Each CTA owns all head-dimension outputs for its query block.
The producer warp publishes that descriptor through `s.work`; both consumer
warp groups use the same descriptor.

For `T` logical tiles, `tile(index)` returns `work.w == kBatch` whenever
`index >= T`. Preserve this termination test and the existing partial-K score
mask and partial-Q output/LSE guards. Do not change the sequence-offset mapping.
In this signed-int implementation, require `T + P - 1 <= INT_MAX`: every real
tile makes one later claim, including claims that terminate workers.

## Ordering

Retain `prepare` and attention launch ordering, including
`griddepcontrol.wait` before producer metadata reads. Counter reset must complete
before any claim; an atomic allocation alone does not publish metadata.

Retain `WorkEmpty` before overwriting `s.work`, `WorkFull` before consumers read
it, and the consumer's `next(s)` before the previous tile's epilogue. The mailbox
already carries an initial tile and a terminal sentinel in the preceding code;
the optimization allows it to carry further tiles between those events.

Retain Q completion/`QueryEmpty`, K/V full/empty barriers, WGMMA completion,
`o_empty`, and the producer's final drain. They protect input visibility and
buffer reuse across iterations. Keep phase counters advancing across tiles;
resetting them inside either loop would desynchronize reused barriers.
The counter allocates ownership only; it does not replace these barriers.

Each tile still traverses K blocks in descending order. Preserve online-softmax
rescaling, reduction grouping, FP16 probability conversion, FP32 accumulation,
and final round-to-nearest FP16 packing. Global tile order does not change a
tile's arithmetic. Evaluate with the unchanged problem and harness policy.

## Example configuration

Preserve this specialization: packed contiguous NHD tensors
`[16384, 64, 128]`, FP16 Q/K/V/output, eight sequences, int32 offsets, query tiles
of 128 rows, and K/V tiles of 176 rows. The global row stride is 8192 half
elements. The direct launch has `kMaxQTiles = 135`, hence 8640 CTAs; the existing
mapping rejects indices beyond the actual tile count. Replay replaces this
launch with the device SM count.

Keep 384 threads per CTA: one 128-thread producer group with its first warp
active, plus two 128-thread consumer groups. Work barriers count 288 threads;
math barriers count 256. Keep the 24/240 producer/consumer register budgets,
88 score registers, 64 output registers, and 44 packed probability registers.

Retain the two K/V stages, one Q tile, TMA transfers with 128-byte swizzling,
cache hints, tensor-map prefetch, and WGMMA `m64n176k16` / `m64n128k16` operations.
`Shared` remains 213248 bytes: Q uses 32768 bytes, staged K/V use 180224 bytes,
and synchronization fields, work descriptor, and padding occupy the remainder.
Global metadata grows from 24 to 25 int words. These are replay settings;
persistent allocation itself requires neither these tensor shapes nor TMA/MMA.

Preserve direct packed output stores, scale constants `kScale` and `kLog2Scale`,
the SM90 check, and compilation flags `-O3 -std=c++17 --use_fast_math
--resource-usage -lineinfo -DNDEBUG`. Keep the complete problem, oracle, and
workload unchanged.

# Precondition

- Data types: no tensor dtype restriction is introduced: work assignment changes
  neither operand representation nor within-tile arithmetic. Tile counts and
  allocated indices must fit their integer representation; overflow could
  revisit completed work or terminate early.
- Layout: the logical index mapping must cover independent, nonoverlapping
  output tiles. Arbitrary execution order would otherwise race writes or alter
  inter-tile dependencies. Tensor contiguity and alignment need no additional
  restriction because the optimization leaves operand addresses unchanged.
- Storage: writable metadata accessible to all CTAs must be available and
  protected from concurrent resets. A shared allocator cannot assign unique
  work if workers see different counters or another invocation resets it.
  Inputs and tile metadata must remain valid until all workers finish.
- Pipeline: initialization must finish before acquisition, and every lane in
  the broadcast mask must participate to receive a consistent index. The prior
  work descriptor must be consumed before replacement; publication must be
  visible before compute. All tile-buffer readers must complete before reuse,
  and barrier phases must match across successive tiles. These dependencies
  prevent stale descriptors, partial inputs, and premature overwrites; no fixed
  stage count is required.
- Hardware: device-wide integer atomic fetch-add supplies unique indices, and
  CUDA warp broadcast shares each claim with its producer lanes. Available
  storage must include one naturally aligned counter word for a legal atomic
  access. The allocator adds no per-CTA shared-memory requirement beyond the
  existing tile implementation.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are separate replacement sites, not a complete function. The outer-guard
excerpts show only the keyword change; retain their bodies except for the shown
producer tail. Existing surrounding declarations supply all identifiers.

## Before

```cuda
// scheduler.cuh: metadata and direct-launch bound.
constexpr int kMetadata = 3 * kBatch;
constexpr int kMaxQTiles = (kTokens + kM - 1) / kM + kBatch - 1;

// prepare: no counter reset follows this instruction.
asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

// producer and consumer: outer guard in each function.
if (work.w < kBatch) { /* existing tile body */ }

// producer: tail after the K/V traversal.
if (lane == 0) load_v(p, s, pos, kstart, work.z);
++pos;
++iteration;
work = make_int4(0, 0, 0, kBatch);
sync(Named::WorkEmpty, kActiveThreads);
if (lane == 0) s.work = work;
arrive(Named::WorkFull, kActiveThreads);

// kernel.cu: Workspace member, device query, and launch site.
int device = -1;
// In ensure:
check(cudaGetDeviceProperties(&prop, current));
TORCH_CHECK(prop.major == 9, "SM90 required");
// In kernel:
config.gridDim = dim3(kMaxQTiles * kHeads);
```

## After

```cuda
// scheduler.cuh: extend the existing grid-visible metadata allocation.
constexpr int kCounterOffset = 3 * kBatch;
constexpr int kMetadata = kCounterOffset + 1;

// prepare: reset once per invocation, before the lane bounds return.
asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
if (lane == 0) metadata[kCounterOffset] = 0;

// producer and consumer: outer loop in each function.
while (work.w < kBatch) { /* existing tile body */ }

// producer: claim the next tile, then retain the existing handoff.
int index = 0;
if (lane == 0)
    index = atomicAdd(p.metadata + kCounterOffset, 1) + gridDim.x;
if (lane == 0) load_v(p, s, pos, kstart, work.z);
++pos;
++iteration;
index = __shfl_sync(0xffffffff, index, 0);
work = tile(index, p.metadata);
sync(Named::WorkEmpty, kActiveThreads);
if (lane == 0) s.work = work;
arrive(Named::WorkFull, kActiveThreads);

// kernel.cu: cache the pool size and launch it on the caller stream.
int device = -1;
int sms = 0;
// In ensure:
check(cudaGetDeviceProperties(&prop, current));
TORCH_CHECK(prop.major == 9, "SM90 required");
sms = prop.multiProcessorCount;
// In kernel:
config.gridDim = dim3(scratch.sms);
```
