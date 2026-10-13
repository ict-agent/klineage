---
skill_id: cuda.shared-statistic-vector-transfer
intent: Pack adjacent shared softmax statistics into vector loads and stores.
preconditions:
- 'Data types: vector transfers preserve both field representations without conversion;
  no floating-point arithmetic property is required because only bits move.'
- 'Layout: each thread accesses both adjacent 32-bit fields at an 8-byte-aligned pair
  base, within allocated bounds; vector accesses require this alignment and stores
  must respect pair ownership.'
- 'Storage: both fields already occupy the same CTA shared allocation and are visible
  there; shared vector instructions cannot access split or inaccessible storage.'
- 'Pipeline: producer completion and visibility precede reads, and all readers finish
  before overwrite or reuse; preserve the synchronization providing this ordering
  because vector transfers are not barriers or atomic snapshots.'
- 'Hardware: aligned 64-bit shared loads/stores are supported; no additional capacity
  is required because vectorization preserves the allocation.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Pack each shared softmax-statistic pair into one vector load or store. In
`solution/hopper.cuh`, replace only `load_stats` and `store_stats` with the After
implementations. Remove their volatile qualifiers: those qualifiers deliberately
keep the Before field accesses scalar. Keep every call site in
`solution/attention.cuh` unchanged. No launch or allocation changes are needed.

`softmax` publishes and reads `Shared::maximum`; `consume<0>` reads the peer
maximum; `reduce_sum` publishes and reads `Shared::sum`. Each pair holds the two
rows owned by a lane. The vector transfer copies their bits together; it performs
no arithmetic, conversion, or atomic update. Retain the existing FP32 reduction
order, BF16 rounding, masking, and output semantics.

## Ownership and ordering

`maximum[lane / 4]` is shared by four adjacent lanes. Only `lane % 4 == 0`
stores that pair. Keep the warp synchronization before this store. `Max0` makes
group 0's pair visible to group 1; `Max1` makes group 1's replacement visible to
group 0. Keep both arrive/sync pairs. `Prob0` orders group 0's consumption before
group 1 can advance, and the next `Max0` exchange orders the next iteration's
maximum handoff. Together with the local warp synchronization, these dependencies
prevent a pair from being overwritten during its readers' accesses.

`sum[threadIdx.x / 4]` has one writer per four lanes. Preserve the 256-thread
`Barrier::Sum` synchronization before loading
`sum[(threadIdx.x / 4) ^ 32]`; this storage has no later writes in the invocation.
All accesses remain within their existing arrays, so add no bounds predicates.
Retain the tensor ABI, caller stream, and all other synchronization.

## Example configuration

This replay uses CUDA on `nvidia-sm90a-cuda13`, with BF16 Q/KV/output, FP32
statistics and accumulators, and int32 indices. The workload is 8192 tokens,
128 heads, QK width 576, value width 512, and 2048 supplied indices per token.
Preserve softmax scale `0.1352337788608801f`, the existing exp2/log conversion,
invalid-index handling, probability BF16 conversion, and final BF16 rounding.

`Shared` retains `float2 maximum[32]` and `float2 sum[64]`: each entry contains
two adjacent FP32 fields, occupies 8 bytes, and has 8-byte alignment. The pair
index is unchanged. `row_index(row, lane)` maps the two fields to
`(lane / 32) * 16 + row * 8 + (lane % 32) / 4`.

Keep 384 threads per CTA: two 128-thread consumers and one 128-thread producer.
Launch 16384 CTAs, one per token and 64-head slice, with a single-CTA cluster and
231376 dynamic shared-memory bytes. Retain both KV buffers, the four transfer
groups, 16-byte cp.async copies, resident query storage, probability-buffer aliasing,
unswizzled operand layouts, online softmax, register tiles, and all WGMMA scheduling.
Keep all compile flags, including the SM90a build target and fast-math setting.
These settings describe this replay; the vector transfer does not require WGMMA
or this pipeline configuration.

After applying the helpers, use the existing evaluator for correctness and timing.
Inspect generated code at these helpers: scalar shared accesses should become
64-bit shared loads/stores. Keep measurements and inspection evidence outside this
card.

# Precondition

- Data types: a vector transfer must preserve both fields' representations without
  conversion. No floating-point arithmetic property is required: the transfer
  moves bits, while the existing arithmetic and rounding remain unchanged.
- Layout: each pair consists of two adjacent 32-bit fields with an 8-byte-aligned
  base, and each accessing thread needs or owns both fields. The 64-bit shared
  access requires this alignment and adjacency; a store must not replace a field
  owned by another writer. Pair indexing must remain inside allocated storage.
- Storage: both fields already reside in the same CTA's shared memory and are
  visible to their consumers there. The shared vector instruction cannot address
  fields split across storage spaces or inaccessible CTA allocations.
- Pipeline: both producer stores must complete and become visible before pair
  consumption; all readers must finish before overwrite or reuse. Preserve the
  synchronization establishing these dependencies. Vector loads/stores provide
  neither a producer/consumer barrier nor an atomic snapshot of concurrent writes.
- Hardware: the target supports aligned 64-bit shared loads/stores; without them,
  the pair cannot use the intended instruction. No additional capacity is required
  because the transformation preserves every field, address, and allocation.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace these two complete helper definitions in `solution/hopper.cuh`.
The existing `float2` arrays and all callers are shared context and remain unchanged.

## Before

```cuda
__device__ __forceinline__ void load_stats(
    const volatile float2* src, float (&dst)[2]) {
    // Volatile fields keep the shared transfers scalar after compilation.
    dst[0] = src->x;
    dst[1] = src->y;
}

__device__ __forceinline__ void store_stats(
    volatile float2* dst, const float (&src)[2]) {
    // Each field retains its own shared-memory store.
    dst->x = src[0];
    dst->y = src[1];
}
```

## After

```cuda
__device__ __forceinline__ void load_stats(
    const float2* src, float (&dst)[2]) {
    // Fetch both row statistics with one shared vector load.
    const float2 pair = *src;
    dst[0] = pair.x;
    dst[1] = pair.y;
}

__device__ __forceinline__ void store_stats(
    float2* dst, const float (&src)[2]) {
    // Publish both row statistics with one shared vector store.
    *dst = make_float2(src[0], src[1]);
}
```
