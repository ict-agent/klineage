---
skill_id: sparse-mla.query-tile-hoisting
intent: Hoist invariant query-tile staging out of the sparse-key loop.
preconditions:
- 'Data types: No additional dtype requirement; retaining copied bits must preserve
  the existing representation and arithmetic, since only load frequency changes.'
- 'Layout: A CTA repeatedly consumes the same unchanged query tile with invariant
  source addresses and consumer indexing; the cooperative loader must cover that tile
  before the first read. No additional contiguity or alignment is required beyond
  the existing loader and readers.'
- 'Storage: The tile is repeatedly copied from global memory to CTA-shared memory,
  and its shared region can remain allocated without conflicting writes throughout
  all uses; otherwise later consumers would read stale or overwritten data.'
- 'Pipeline: All loader writes must complete and become visible in the readers'' memory
  domain before first use, with every CTA thread reaching the initial barrier. All
  readers must finish before the tile storage is reused; otherwise hoisting permits
  incomplete reads or premature overwrites.'
- 'Hardware: CTA-shared memory and a CTA barrier are required; query-tile bytes plus
  other simultaneously live shared allocations must fit the configured per-CTA limit
  and device capacity. No new instruction feature is required.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Keep a query tile in shared memory across the sparse-key loop instead of loading
it for every pair of key tiles. Edit only `solution/attention.cuh`.

Move `if constexpr (Group == 0) load_query(sm, params, lane, head, token);`
from the beginning of `consume<Group>`'s `block` loop to `attention`, immediately
after `lane` is computed. There, use the runtime condition `if (group == 0)`.
Follow that call with `__syncthreads()` before either consumer branch or return.
The existing loader includes `shared_fence()`; retain it to publish generic
shared stores to WGMMA's asynchronous proxy. The new CTA barrier completes all
query writes before either consumer starts. Do not add a query load elsewhere.

Retain both `Stage` barriers per loop iteration, all WGMMA waits, and every KV,
probability, and statistics barrier. They still order KV production and safe
buffer reuse. The query region has no other writer: it remains intact until all
QK reads finish. P0 aliases K0's final tile, not the query region. The loader
stays owned by group 0; both consumer groups read the same query tile.

Keep `load_query`, `q_index`, and `desc_q` unchanged. For lane `lane`, the loader
visits `i = lane + n*kWarpgroup`, reads the contiguous global query tile, and
writes `(col/kTile)*kTileElems + q_index(row, col%kTile)`, where
`row=i/kWidth` and `col=i%kWidth`. No layout, allocation, host ABI, stream, or
launch change is needed. Change the shared-region comment to
`Q stays resident; P0 reuses K0's RoPE tile.` Describe the loop's first barrier as
waiting for KV and masks, and its final barrier as protecting KV overwrites.

Preserve query bounds and the supplied-index masks. This workload contains only
complete query tiles; the loader's existing loop bounds cover them. For partial
tiles in another workload, preserve the loader's guarded loads and zero filling
when moving it. Preserve all arithmetic, reduction ordering, probability BF16
rounding, output BF16 rounding, and maximum/LSE semantics.

Inspect generated device code: query loads must precede the sparse-key loop and
must not recur on its backedge. Use the existing Kernel evaluator with the
unchanged problem for correctness and latency; keep evidence outside this card.

## Example configuration

The supplied workload is `TOKENS=8192`, `HEADS=128`, `QK_DIM=576`, `VALUE_DIM=512`,
and `TOPK=2048`. Inputs Q/KV and probabilities use BF16; indices use int32;
accumulators and maximum/LSE outputs use FP32. Output uses BF16. Preserve
`kScale=0.1352337788608801f`, base-two online softmax, and the existing operation
order and compile flags, including fast math.

Each CTA owns one token and 64 heads. Its 384 threads form three 128-thread groups:
groups 0 and 1 consume; group 2 loads KV. Group 0's query loader covers a
`64*576`-element tile with global row stride `576*sizeof(Bf16)` and unswizzled
shared cores. Q occupies 73728 bytes within the unchanged 231296-byte `Shared`.
The launch uses 16384 CTAs, one CTA per cluster, and `kThreads` threads per CTA.
Preserve `__launch_bounds__(kThreads, 1, 1)` and the caller's CUDA stream.

There are 32 key tiles, traversed two at a time over 16 loop iterations. Hoisting
reduces query staging from 16 loads per CTA to one. Keep the two KV buffers,
scalar synchronous KV transfers, unswizzled operand descriptors, register output
accumulators, shared probability exchange, P0/K0 storage alias, scalar statistics
transfers, shared peer reductions, and repeated normalization/rescaling
calculations. Retain SM90a WGMMA QK `m64n64k16` and PV `m64n256k16`, their fences,
commits and waits, and the original register/compiler settings. These retained
compute mechanisms are not prerequisites for query-load hoisting.

# Precondition

- Data types: No additional dtype requirement. Keeping the same copied bits and
  unchanged arithmetic preserves representation and numerical behavior; hoisting
  does not convert operands or change reductions.
- Layout: The CTA's repeated uses must address the same unchanged source tile
  with the same consumer indexing. The cooperative loader must cover every consumed element
  before first use; otherwise later iterations reuse the wrong or missing data.
  There is no extra contiguity or alignment requirement beyond those already
  imposed by the loader and readers.
- Storage: The preceding implementation repeatedly stages the tile from global
  to CTA-shared memory. Its shared region must be available for the entire reuse
  interval without conflicting writes; removing reloads cannot repair an
  intervening overwrite or a changed source value.
- Pipeline: Loader writes must complete and be visible in the readers' memory
  domain before first use. Every CTA thread must reach the initial barrier,
  preventing incomplete publication or deadlock. All readers must complete before
  any reuse of that region, preventing overwrite of an operand still in use.
- Hardware: CTA-shared memory and a CTA barrier provide storage and publication.
  `query_tile_bytes + other_live_shared_bytes` must fit both the configured
  per-CTA limit and device capacity. Hoisting adds no instruction feature.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are two insertion locations in `solution/attention.cuh`; preserve omitted
loop bodies and dispatch branches. `load_query` and all constants already exist.

## Before

```cuda
// Beginning of consume<Group>'s existing loop.
#pragma unroll 1
for (int block = 0; block < kBlocks; block += 2) {
    if constexpr (Group == 0) load_query(sm, params, lane, head, token);

    // Publish Q, KV, and masks before either consumer reads them.
    sync<Barrier::Stage, kThreads>();
    // Existing consumer body and final Stage barrier follow.
}

// In attention, immediately before the existing consumer dispatch.
const int group = group_index();
const int lane = threadIdx.x % kWarpgroup;

if (group == 0) {
    consume<0>(sm, params, lane, head, token);
    return;
}
```

## After

```cuda
// Beginning of consume<Group>'s existing loop.
#pragma unroll 1
for (int block = 0; block < kBlocks; block += 2) {
    // Consume only after all synchronous KV and mask stores finish.
    sync<Barrier::Stage, kThreads>();
    // Existing consumer body and final Stage barrier follow.
}

// In attention, immediately before the existing consumer dispatch.
const int group = group_index();
const int lane = threadIdx.x % kWarpgroup;

if (group == 0) load_query(sm, params, lane, head, token);
__syncthreads();

if (group == 0) {
    consume<0>(sm, params, lane, head, token);
    return;
}
```
