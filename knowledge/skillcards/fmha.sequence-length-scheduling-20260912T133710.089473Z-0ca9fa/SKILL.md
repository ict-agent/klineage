---
skill_id: fmha.sequence-length-scheduling
intent: Schedule longer KV sequences first through stable batch sorting.
preconditions:
- 'Data types: no additional attention dtype or rounding requirement; only task metadata
  is permuted. Integer cost calculations must be representable, and padding keys must
  compare below valid costs, or sorting can lose valid tasks.'
- 'Layout: cumulative Q/K offsets have known indexing, and each batch has an independent
  tile range and original identity. Tile counts, split fields, and identities are
  available as associated metadata, and consumers resolve tensor addresses through
  original identities; otherwise a permutation cannot preserve task ownership.'
- 'Storage: Q/K offsets are device-readable and dispatch metadata is device-writable
  and visible to the scheduling CTAs. These arrays supply sort keys and publish the
  permutation without moving tensor data.'
- 'Pipeline: batch execution order is unconstrained, and all preparation threads can
  synchronize scratch writes, reads, and reuse without early exits. Offset producers
  finish before preparation; metadata publication finishes before dispatch, and dispatch
  readers finish before metadata reuse. These dependencies prevent partial scratch
  reads and stale or overwritten descriptors.'
- 'Hardware: CUDA warp shuffles, CTA barriers, and CTA shared memory with capacity
  for P * sizeof(descriptor), where P is the padded sorting width. Shuffles obtain
  adjacent offsets; barriers and scratch support each cooperative merge pass.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Schedule batches by descending KV tile count using a stable shared-memory
merge-path sort. Longer KV sequences yield more work per query tile; issuing
them first can reduce the tail of a persistent launch. Equal costs retain input
order, so equal-length batches need not benefit.

Replace only `prepare()` in `solution/scheduler.cuh` with the After snippet.
Retain its signature, metadata offsets, counter initialization, and launch.
The sort moves descriptors, never Q/K/V or output elements.

Each preparation lane starts with one descriptor:
`(ceil(k_length / kN), ceil(q_length / kM), split, original_batch)`.
The fields are `int4.x/y/z/w`; split remains one. Pad unused lanes with
`INT_MIN` costs and zero query tiles. Real sequence lengths are positive, so
padding sorts last. All lanes, including padding, participate until the sort
finishes. Guard offset loads with `lane <= kBatch`; guard final metadata stores
with `lane < kBatch`.

Each merge pass combines two adjacent descending runs. Its first barrier protects
the previous pass's shared reads before overwriting `merge`; its second publishes
all descriptors before binary-search and selection reads. Strict `>` comparisons
choose the left run on ties, preserving stable order. Move all descriptor fields
together, then publish split, query-tile count, and original batch identity in
schedule-rank order.

`tile()` continues to decode each linear work index using query-tile prefixes
and heads. `producer()`, `consumer()`, and `epilogue()` already resolve
`p.metadata[kBatchOffset + work.w]` before accessing original sequence offsets.
Preserve these lookups: `work.w` is a schedule rank after this transformation.
Every original query tile and head must still execute exactly once.

Keep the `griddepcontrol.launch_dependents` instruction in `prepare()`, the
programmatic serialization launch attribute in `solution/kernel.cu`, and the
`griddepcontrol.wait` preceding metadata reads in `producer()`. Early dependent
launch does not publish unfinished metadata. Retain caller-stream ordering and
finish all scheduling reads before a later invocation overwrites the workspace.
The merge scratch is local to preparation and expires with that kernel.

Leave attention arithmetic, masking, reductions, FP32 accumulation, FP16
round-to-nearest conversions, and output addressing unchanged. Reordering
independent batches does not reorder any batch's arithmetic. Keep the complete
problem, oracle, tolerances, seed, and timing policy. Validate with
`klineage.harness.evaluate`; keep evidence outside this card.

## Example configuration

The supplied problem has 16384 packed tokens, 64 heads, head dimension 128,
eight sequences, and nine cumulative offsets. Q/K/V/output are contiguous
FP16 `[TOKENS, HEADS, HEAD_DIM]`; offsets are contiguous int32 arrays.
The token stride is `kRow = kHeads * kDim` half elements (16384 bytes here).

Preserve `prepare<<<1, 32, 0, stream>>>`. Its full-warp shuffles use mask
`0xffffffff`; eight descriptors and their terminal offsets fit in the warp.
The snippet sorts 32 padded slots through widths 1, 2, 4, 8, 16, allocating
`int4 merge[32]` (512 bytes). The load scheme requires the terminal offset's
lane to exist; a larger batch count needs adapted offset loading and sorting.
Retain the no-unroll merge-loop hint. `INT_MIN` is the padding sentinel.

Metadata uses offsets `0`, `kBatch`, `2*kBatch`, and `3*kBatch` for splits,
query-tile counts, original batch identities, and the atomic counter;
`kMetadata = 3*kBatch + 1`. Keep the counter reset and all split values unchanged.
The attention grid has one CTA per SM, each with 384 threads and `sizeof(Shared)`
dynamic shared bytes. One 32-thread producer warp and 256 math threads participate
in its work handoff; warp-group size is 128. Preserve the 24/240 producer/consumer
register budgets and all named barriers and transaction phases.

Retain query tiles of 128 rows, KV tiles of 176 rows, two K/V stages, reverse KV
traversal, 64-column TMA panels, TMA's 128-byte swizzle and cache hints, and the
QK/PV WGMMA operations. Retain online softmax, register-held probabilities,
linear shared output layout, STSM output staging, vector output stores, and
V/output storage reuse. These mechanisms do not depend on sequence sorting.

Keep the CUDA pybind11 destination-passing ABI and caller device/stream. Preserve
SM90a compilation and flags `-O3 -std=c++17 --use_fast_math --resource-usage
-lineinfo -DNDEBUG`. No host launch or attention-kernel change is needed.

# Precondition

- Data types: no additional attention dtype or rounding requirement. Sorting
  compares metadata and moves identities, never arithmetic operands. Integer
  cost calculations must not overflow; padding must compare below valid keys,
  or real descriptors can be displaced from the published range.
- Layout: cumulative Q/K offsets have known indexing. Each batch owns an
  independent tile range and an original identity. Tile counts, split fields,
  and identities are available as associated metadata. Consumers recover original
  identities before tensor addressing. Without those associations and lookups,
  a permutation would assign work or outputs to the wrong sequence.
- Storage: preparation can read device Q/K offsets and write device dispatch
  metadata that scheduling CTAs can observe. The offsets supply cost/count keys;
  the metadata publishes their associated identities. Tensor data need not move.
- Pipeline: batches can execute in any order, and all preparation threads can
  synchronize scratch writes, reads, and reuse without early exits. This permits
  publishing each pass before reads and completing readers before overwrite.
  Offset producers finish before preparation; metadata publication finishes before
  dispatch, and dispatch readers finish before later metadata reuse. Otherwise
  readers can observe partial keys or descriptors. No particular attention stage
  count is required.
- Hardware: CUDA warp shuffles obtain adjacent offsets. CTA barriers and shared
  memory support cooperative merging, with capacity at least
  `P * sizeof(descriptor)` for padded sorting width `P`. Without that storage or
  synchronization, the merge passes cannot safely exchange descriptors.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both snippets are complete replacements for `prepare()` only. Existing constants,
CUDA types, metadata consumers, and host launch supply the shared context.

## Before

```cuda
__global__ void prepare(const int* qo, const int* ko, int* metadata) {
    const int lane = threadIdx.x;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (lane == 0) metadata[kCounterOffset] = 0;

    int q = lane <= kBatch ? qo[lane] : 0;
    const int qnext = __shfl_down_sync(0xffffffff, q, 1);
    if (lane >= kBatch) return;

    // Keep sequence order; the persistent scheduler still assigns each tile.
    metadata[kSplitOffset + lane] = 1;
    metadata[kBlocksOffset + lane] = (qnext - q + kM - 1) / kM;
    metadata[kBatchOffset + lane] = lane;
}
```

## After

```cuda
__global__ void prepare(const int* qo, const int* ko, int* metadata) {
    __shared__ int4 merge[32];
    const int lane = threadIdx.x;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (lane == 0) metadata[kCounterOffset] = 0;

    int q = lane <= kBatch ? qo[lane] : 0;
    int k = lane <= kBatch ? ko[lane] : 0;
    const int qnext = __shfl_down_sync(0xffffffff, q, 1);
    const int knext = __shfl_down_sync(0xffffffff, k, 1);
    int4 item = make_int4(lane < kBatch ? (knext - k + kN - 1) / kN : INT_MIN,
                         lane < kBatch ? (qnext - q + kM - 1) / kM : 0, 1, lane);

    // Sort descending by KV tile count; retain input order for ties.
    #pragma unroll 1
    for (int width = 1; width < 32; width *= 2) {
        __syncthreads();
        merge[lane] = item;
        __syncthreads();
        const int start = lane & -(2 * width);
        const int diag = lane & (2 * width - 1);
        int lo = max(0, diag - width), hi = min(diag, width);
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            int left = merge[start + mid].x;
            int right = merge[start + width + diag - 1 - mid].x;
            if (right > left) hi = mid;
            else lo = mid + 1;
        }
        int a = lo, b = diag - lo;
        if (b < width && (a >= width || merge[start + width + b].x > merge[start + a].x))
            item = merge[start + width + b];
        else item = merge[start + a];
    }
    if (lane >= kBatch) return;
    metadata[kSplitOffset + lane] = item.z;
    metadata[kBlocksOffset + lane] = item.y;
    metadata[kBatchOffset + lane] = item.w;
}
```
