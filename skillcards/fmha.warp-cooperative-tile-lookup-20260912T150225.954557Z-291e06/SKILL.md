---
skill_id: fmha.warp-cooperative-tile-lookup
intent: Decode sequence tiles with a warp-cooperative prefix scan and ballot.
preconditions:
- 'Data types: nonnegative integer counts and indices, a positive head count, and
  an integer representation that holds every prefix sum and head-scaled boundary exactly;
  overflow would corrupt range comparisons. Attention-value dtypes impose no additional
  requirement because this lookup only processes metadata.'
- 'Layout: an ordered count table defines consecutive sequence ranges, with one lookup
  index shared by the warp; the sequence count must fit one 32-lane warp so each entry
  has a lane. These conditions make the prefix boundaries and ballot rank identify
  the same range in every lane.'
- 'Storage: every participating lane can read the same metadata table; inaccessible
  or inconsistent counts would produce incorrect prefix boundaries. No additional
  shared-memory allocation is required.'
- 'Pipeline: metadata production must finish and become visible before lookup, and
  all readers must finish before metadata reuse. All lanes named by the full-warp
  mask must reach each shuffle and ballot together; missing participants invalidate
  these collectives.'
- 'Hardware: CUDA warp shuffle, ballot, and population-count operations with 32-lane
  warps; the single-warp scan and ballot rank require these operations. No additional
  shared-memory capacity is needed.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace the serial metadata search in `solution/scheduler.cuh::tile()` with a
warp-cooperative inclusive scan and ballot lookup. Replace only this function
with the After snippet. Existing `kWarpSize` and `kFullWarpMask` constants are
available through the header chain. No ABI, launch, or allocation changes are needed.

Let `b[j] = metadata[kBlocksOffset + j]`, `H = kHeads`, and
`P[j] = sum(b[0:j+1])`. Sequence `j` owns the half-open tile range
`[H*P[j-1], H*P[j])`, using zero for `P[-1]`. Within a sequence,
query tiles vary fastest and heads vary next. Return
`int4(sequence_start, query_tile, head, sequence)` exactly as before.

Each producer lane loads one count, or zero when its lane exceeds the table.
Shuffle-up steps form inclusive prefixes. The last lane supplies the total.
For a valid index, ballot the completed ranges and population-count the result
to select the sequence. Broadcast its count and preceding prefix, then retain
the existing quotient/remainder decoding. This distributes metadata loads and
prefix work across lanes instead of repeating a serial search in each lane.

The padded lanes contain the final prefix and cannot enter the ballot for a
valid index. Return `make_int4(index, 0, 0, kBatch)` before selection when the
index reaches the total. Because all lanes receive the same index, this return
is uniform. Nonnegative counts give monotone boundaries; zero-count ranges are
skipped, so the selected count is positive before division.

`producer()` calls `tile(blockIdx.x, p.metadata)` from its whole first warp.
Lane zero retains ownership of `s.work`. Preserve `WorkFull` publication and
`WorkEmpty` before overwriting that slot. `prepare` completes on the caller
stream before attention reads metadata; keep that order and do not overwrite
metadata until readers finish. Scan intermediates stay in registers, so this
change introduces no shared buffer or barrier. Preserve all TMA and WGMMA
completion and buffer-reuse handshakes.

Only integer dispatch changes. Preserve attention accumulation order, score
masking, online softmax, FP16 rounding, output bounds, and the complete problem.

## Example configuration

The supplied workload has 16,384 packed tokens, 64 heads, head dimension 128,
eight sequences, and nine cumulative offsets. Q/K/V/output are contiguous FP16
NHD tensors; offsets and dispatch metadata are signed 32-bit integers. Metadata
has 24 entries: split counts, query-block counts, and sequence IDs, with
`kBlocksOffset = 8` and `kBatchOffset = 16`.

Retain query tiles of 128 rows, KV tiles of 176 rows, two input stages, and
eight-column unswizzled input panels. Retain the 384-thread CTA: its first warp
produces work, and 256 math threads form two WGMMA groups. `prepare` launches
one 32-thread block. Attention launches `kMaxQTiles * kHeads = 8640` CTAs,
with `kMaxQTiles = 135`, `sizeof(Shared)` dynamic shared memory, and the caller
stream. The replay scan uses shuffle distances 1, 2, 4, 8, and 16, with the
existing unroll hint; these are one complete warp scan, not separate changes.

Retain TMA staging, staggered K/V production, WGMMA QK/PV instructions,
register fragments, shared-memory softmax exchanges, scalar half conversions,
scalar output stores, and direct CTA ordering. Keep the supplied SM90a build
target and compile flags. These compute and pipeline settings are replay
context, not prerequisites of metadata lookup.

# Precondition

- Data types: counts and indices are nonnegative integers, and the head count
  is positive. The integer representation must hold every prefix sum and
  head-scaled boundary exactly; overflow would change comparisons and selected
  ranges. No additional attention dtype constraint applies because
  the transformation never reads or computes attention values.
- Layout: ordered counts describe consecutive sequence ranges, and every lane
  receives the same lookup index. The table must fit a 32-lane warp for the
  one-entry-per-lane scan. These properties make the ballot's rank select the
  same sequence in every lane; a larger table needs a different scan design.
- Storage: participating lanes must read the same accessible metadata table.
  Different snapshots or inaccessible entries corrupt the common prefix
  boundaries. No additional shared-memory allocation is required.
- Pipeline: the metadata producer must complete and make counts visible before
  lookup. Readers must finish before those counts are overwritten. Every lane
  named by the full-warp mask must reach each shuffle and ballot together;
  otherwise the collective result is invalid. No particular input stage count
  follows from these ordering requirements.
- Hardware: CUDA must provide shuffle, ballot, and population count for
  32-lane warps. The prefix scan uses shuffle communication and the range
  selection uses ballot rank. No additional shared-memory capacity is needed.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
__device__ __forceinline__ int4 tile(int index, const int* metadata) {
    int start = 0;

    // Each producer lane scans sequence ranges without exchanging metadata.
    for (int batch = 0; batch < kBatch; ++batch) {
        const int blocks = metadata[kBlocksOffset + batch];
        const int end = start + blocks * kHeads;
        if (index >= end) {
            start = end;
            continue;
        }

        const int within = index - start;
        const int head = within / blocks;
        return make_int4(start, within - head * blocks, head, batch);
    }

    return make_int4(index, 0, 0, kBatch);
}
```

## After

```cuda
__device__ __forceinline__ int4 tile(int index, const int* metadata) {
    const int lane = threadIdx.x % kWarpSize;
    int blocks = lane < kBatch ? metadata[kBlocksOffset + lane] : 0;
    int prefix = blocks;

    // Scan sequence counts once across the producer warp.
    #pragma unroll
    for (int delta = 1; delta < kWarpSize; delta *= 2) {
        const int prev = __shfl_up_sync(kFullWarpMask, prefix, delta);
        if (lane >= delta) prefix += prev;
    }

    const int total = __shfl_sync(kFullWarpMask, prefix, kWarpSize - 1) * kHeads;
    if (index >= total) return make_int4(index, 0, 0, kBatch);

    // Count completed sequence ranges, then broadcast the selected range.
    const int batch = __popc(__ballot_sync(kFullWarpMask, prefix * kHeads <= index));
    blocks = __shfl_sync(kFullWarpMask, blocks, batch);
    const int before = __shfl_sync(kFullWarpMask, prefix, max(0, batch - 1));
    const int start = batch == 0 ? 0 : before * kHeads;
    const int within = index - start;
    const int head = within / blocks;
    return make_int4(start, within - head * blocks, head, batch);
}
```
