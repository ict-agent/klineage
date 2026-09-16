---
skill_id: sparse-mla.reuse-gather-indices
intent: Reuse sparse row indices and offsets across KV column tiles.
preconditions:
- 'Data types: cached indices, invalid markers, validity decisions, and derived offsets
  must retain their original integer meaning without overflow or truncation; otherwise
  reuse could select another row. KV element dtype imposes no additional requirement
  because this caches integer metadata only.'
- 'Layout: a thread repeatedly accesses the same index slot across column tiles, with
  a known ownership mapping and invariant KV row stride; otherwise one cached offset
  cannot serve those accesses. No additional contiguity or alignment is required.'
- 'Storage: indices are readable in global memory and remain unchanged, including
  through aliases, throughout their reuse interval; otherwise a retained index can
  become stale.'
- 'Pipeline: index producers complete before preloading, and cached metadata lives
  through its last transfer or mask use. Preserve publication before shared consumers
  and reader completion before buffer reuse to prevent incomplete or overwritten operands;
  no new barrier participation or stage count is required.'
- 'Hardware: no additional feature or fixed capacity requirement; reuse uses ordinary
  thread-private integer state and adds no shared allocation. Register pressure can
  affect speed but does not invalidate metadata reuse.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reuse each sparse row selection across its KV column tiles. In
`solution/attention.cuh`, replace `copy_tiles` and `load_kv` with the After
functions. Preload signed row indices once per `load_kv` call, compute element
offsets and validity, and pass those per-thread arrays to all four transfer
groups. Reuse the same validity for shared masks.

Remove `load_index` from `solution/hopper.cuh`. Its volatile global load forces
repeated reads in Before; use ordinary `rows[offset]` loads in the preload.
Retain `row_pointer` and its compiler control, scalar `copy_kv`, transfer order,
and loop directives. This removes repeated metadata reads and calculations;
it does not change KV transfers.

Each loading thread retains its own metadata. Keep the existing row/column
ownership and `kv_index` destinations. Invalid indices still suppress KV reads,
zero-fill their shared elements, and publish false masks. Keep signed 64-bit
multiplication before applying the runtime element stride.

Keep `shared_fence()` after transfers, the consumer stage barrier before reads,
and the ending stage barrier after WGMMA readers complete. These preserve
visibility and safe buffer reuse. No launch, shared allocation, ABI, or caller
stream changes are needed. Preserve all arithmetic, scaling, reduction order,
and rounding. Check with the existing evaluator; inspect generated index loads
to confirm they occur once per cached selection, outside column-tile transfers.

## Example configuration

The supplied workload has 8192 tokens, 128 heads, QK width 576, value width 512,
and 2048 selections per token. Indices are int32; cached offsets are int64 in
BF16 elements. Validity is `index >= 0 && index < kTokens`.

Each CTA has 256 threads and owns one token and 64 heads. Grid size is 16384;
cluster size is one CTA. Group 0's 128 threads load Q and KV; both groups compute.
Eight loading threads share a selected row, each copying eight BF16 columns.
There are 16 such row groups and four rows per loading thread in each of two
KV buffers. Cache eight offsets and eight validity predicates per thread, then
reuse each across nine column tiles. Only the first lane of each copy group
publishes masks.

Retain the transfer ranges `(0,0,4)`, `(1,4,9)`, `(0,4,9)`, `(1,0,4)`, where
entries mean `(Buffer,Begin,End)`. Retain 64-square tiles, scalar 16-bit KV loads
and stores, and 16-byte logical copies. Q reloads for each key pair. The shared
dynamic allocation remains 231296 bytes, including probability/K0 RoPE storage
aliasing. Retain the compiler flags and register controls; caching extends the
lifetime of eight offsets and predicates, so register pressure can affect benefit.

Keep unswizzled WGMMA layouts, BF16 QK/PV operands, FP32 accumulators, online
softmax, shared peer reductions, recomputed rescaling/normalization, and BF16
round-to-nearest conversions. Keep scale `0.1352337788608801f`, output stores,
max-logit/LSE calculations, all barriers, and compiler flags. These are retained
settings, not prerequisites of metadata reuse.

# Precondition

- Data types: cached indices, invalid markers, validity decisions, and derived
  offsets must retain their original integer meaning without overflow or
  truncation. Otherwise a cached value could redirect a KV load. This caches
  integer metadata only; it adds no KV dtype or floating-point requirement.
- Layout: each thread must reuse the same index slot across column tiles, with
  known ownership and an invariant KV row stride. Otherwise one retained offset
  would address different intended rows. No additional alignment or contiguity
  is needed for this reuse.
- Storage: the indices must be readable in global memory and remain unchanged,
  including through aliases, until their final cached use. Concurrent updates
  would make preloaded values stale.
- Pipeline: the index producer must finish before preloading; cached metadata
  must remain live through the final transfer or mask use. Preserve publication
  before shared consumers and reader completion before storage reuse; these
  prevent reads of incomplete or overwritten operands. Reuse adds no barrier
  participation or fixed stage count.
- Hardware: no additional feature or fixed capacity is needed. Ordinary
  thread-private integer state holds the cached metadata; shared allocation is
  unchanged. Register pressure can change performance but does not invalidate
  metadata reuse.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets replace both functions together. Existing constants, helpers, and
callers remain available. Remove the now-unused `load_index` helper after applying
After; its volatile load is only the control that enforces Before's repeated reads.

## Before

```cuda
template<int Buffer, int Begin, int End>
__device__ __forceinline__ void copy_tiles(
    Shared& sm, const Params& params, int lane, int token, int block) {
    const int group = lane / kCopyGroup;
    const int column = (lane % kCopyGroup) * kVectorElems;
    Bf16* dst = sm.kv[Buffer] + kv_index(group, column);
    const Bf16* src = params.kv + column;
    const int* rows = params.indices + token * kTopk;
#pragma unroll
    for (int row = 0; row < kCopyRows; ++row) {
        const int offset = (block + Buffer) * kTile + row * kCopyGroups + group;
#pragma unroll
        for (int tile = Begin; tile < End; ++tile) {
            const int index = load_index(rows + offset);
            const bool valid = index >= 0 && index < kTokens;
            const Bf16* row_src = row_pointer(src, int64_t(index) * params.kv_stride);
            copy_kv(row_src + tile * kTile,
                    dst + tile * kTileElems + row * kCopyGroups * kCoreElems,
                    valid ? 16 : 0);
        }
    }
}

__device__ __forceinline__ void load_kv(
    Shared& sm, const Params& params, int lane, int token, int block) {
    const int group = lane / kCopyGroup;
    const int* rows = params.indices + token * kTopk;

    // Retain the four transfer groups and their operand layouts.
    copy_tiles<0, 0, kHalfTiles>(sm, params, lane, token, block);
    copy_tiles<1, kHalfTiles, kKeyTiles>(sm, params, lane, token, block);
    copy_tiles<0, kHalfTiles, kKeyTiles>(sm, params, lane, token, block);
    copy_tiles<1, 0, kHalfTiles>(sm, params, lane, token, block);

    if (lane % kCopyGroup == 0) {
#pragma unroll
        for (int buffer = 0; buffer < 2; ++buffer) {
#pragma unroll
            for (int row = 0; row < kCopyRows; ++row) {
                const int offset = (block + buffer) * kTile + row * kCopyGroups + group;
                const int index = load_index(rows + offset);
                sm.valid[buffer][row * kCopyGroups + group] = index >= 0 && index < kTokens;
            }
        }
    }
    // Publish KV stores before the consumers' stage barrier.
    shared_fence();
}
```

## After

```cuda
template<int Buffer, int Begin, int End>
__device__ __forceinline__ void copy_tiles(
    Shared& sm, const Params& params, int lane, const int64_t (&indices)[2][kCopyRows],
    const bool (&valid)[2][kCopyRows]) {
    const int group = lane / kCopyGroup;
    const int column = (lane % kCopyGroup) * kVectorElems;
    Bf16* dst = sm.kv[Buffer] + kv_index(group, column);
    const Bf16* src = params.kv + column;
#pragma unroll
    for (int row = 0; row < kCopyRows; ++row) {
        const Bf16* row_src = row_pointer(src, indices[Buffer][row]);
#pragma unroll
        for (int tile = Begin; tile < End; ++tile)
            copy_kv(row_src + tile * kTile,
                    dst + tile * kTileElems + row * kCopyGroups * kCoreElems,
                    valid[Buffer][row] ? 16 : 0);
    }
}

__device__ __forceinline__ void load_kv(
    Shared& sm, const Params& params, int lane, int token, int block) {
    const int group = lane / kCopyGroup;
    const int* rows = params.indices + token * kTopk;
    int64_t indices[2][kCopyRows];
    bool valid[2][kCopyRows];
#pragma unroll
    for (int buffer = 0; buffer < 2; ++buffer) {
#pragma unroll
        for (int row = 0; row < kCopyRows; ++row) {
            const int offset = (block + buffer) * kTile + row * kCopyGroups + group;
            const int index = rows[offset];
            indices[buffer][row] = int64_t(index) * params.kv_stride;
            valid[buffer][row] = index >= 0 && index < kTokens;
        }
    }

    // Retain the four transfer groups and their operand layouts.
    copy_tiles<0, 0, kHalfTiles>(sm, params, lane, indices, valid);
    copy_tiles<1, kHalfTiles, kKeyTiles>(sm, params, lane, indices, valid);
    copy_tiles<0, kHalfTiles, kKeyTiles>(sm, params, lane, indices, valid);
    copy_tiles<1, 0, kHalfTiles>(sm, params, lane, indices, valid);

    if (lane % kCopyGroup == 0) {
#pragma unroll
        for (int buffer = 0; buffer < 2; ++buffer) {
#pragma unroll
            for (int row = 0; row < kCopyRows; ++row)
                sm.valid[buffer][row * kCopyGroups + group] = valid[buffer][row];
        }
    }
    // Publish KV stores before the consumers' stage barrier.
    shared_fence();
}
```
