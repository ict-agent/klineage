---
skill_id: sparse-mla.dedicated-kv-producer
intent: Assign KV gathering to a dedicated producer warpgroup.
preconditions:
- 'Data types: no additional dtype or arithmetic requirement; assigning copies to
  another warp preserves payload bits and leaves consumer arithmetic intact.'
- 'Layout: copy ownership must be expressible through role-local lanes in whole warps;
  relocating the copy role must neither omit nor duplicate destinations or split required
  warp collectives.'
- 'Storage: operands and masks are already staged in CTA-visible storage, and copy
  inputs do not depend on consumer-private arithmetic state; a separate producer must
  access the same inputs and publish to the same consumers.'
- 'Pipeline: every participating role must rendezvous after producer completion and
  visibility in each reader''s memory domain, then after all readers finish before
  buffer reuse; matching iteration order and counted-barrier membership prevent stale
  reads, overwrites, and deadlock.'
- 'Hardware: independently scheduled warps, CTA-shared storage, and counted barriers;
  consumer plus producer threads must fit the CTA limit, and actual rounded register
  allocation and shared storage must fit one resident CTA.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Assign KV gathering to a dedicated producer warpgroup. This separates copy
instructions and temporary state from consumer arithmetic and lets KV gathering
run alongside query loading. Keep the existing load/compute rendezvous; this
recipe does not introduce a copy/compute pipeline.

All edits are in `solution/attention.cuh`:

1. Set `kThreads = 3 * kWarpgroup`. Keep `kConsumerThreads` unchanged.
2. In `consume<Group>`, remove only the `load_kv` call and its comment. Group 0
   still calls `load_query` before the first stage barrier of each iteration.
3. Keep `load_kv` and its declaration unchanged. Add the `produce` wrapper below
   its definition, before `attention`. It loops over the same key pairs and calls
   both stage barriers after each load.
4. In `attention`, keep the group-0 branch. Give group 1 its own returning branch;
   dispatch group 2 to `produce`. Keep role-local `lane` calculation unchanged.

`solution/kernel.cu` already derives the block size from `kThreads`; no host edit
is needed. The launch bounds and both consumers' `Stage` barriers also use that
constant. They now include the producer. Keep all other barriers' participation
unchanged: consumer exchanges include only consumers, and local barriers include
only their owning consumer group. The producer exits after the last reuse barrier
and never joins final sum reduction or output stores.

The producer inherits the former loader's lane-to-address mapping exactly. Only
the physical thread range changes. Keep `copy_tiles`, `copy_kv`, `row_pointer`,
all shared layouts/descriptors, and the order of the four transfer groups intact.
Preserve signed index bounds checks, zero-filled invalid rows, and validity masks;
an invalid global row must never be dereferenced. Do not change query or output
ownership, indexing, allocation, grid mapping, ABI, or caller stream.

`load_kv` finishes scalar stores and calls `shared_fence`. Group 0 independently
publishes Q. The first stage rendezvous makes Q, KV, and masks ready for readers.
Existing proxy fences remain necessary for asynchronous readers. The second
rendezvous follows every consumer's `wait<0>()`, preventing the next load from
overwriting Q, KV, or aliased probability storage while a reader is active. Both
roles traverse identical key-pair iterations. No extra buffers or stages are added.

Keep FP32 accumulation and reduction grouping, BF16 rounding points, masking,
softmax scale, and max-logit/LSE semantics unchanged. Preserve the complete problem
and compile flags. Validate with the existing Kernel evaluator and inspect the
compiled launch limit, stage participation, and dedicated producer dispatch.

## Example configuration

This replay uses CUDA SM90 with CUDA 13, `kWarpgroup=128`, two consumer groups,
and one new producer group. The CTA grows from 256 to 384 threads: consumer 0
uses threads 0–127, consumer 1 uses 128–255, and producer uses 256–383. Its lane
is `threadIdx.x % kWarpgroup`. `Stage` uses 384 participants; Max0, Max1, Prob0,
Prob1, and Sum retain 256; Local0 and Local1 retain 128. Keep barrier IDs unchanged.

The workload has 8192 tokens, 128 heads, QK width 576, value width 512, and 2048
selected indices. Keep contiguous BF16 `q[8192,128,576]` and `kv[8192,1,576]`,
INT32 indices, BF16 output, FP32 statistics, scale `0.1352337788608801f`, and all
existing arithmetic. There are 16384 CTAs, one per token and 64-head slice, with
one CTA per cluster and `__launch_bounds__(kThreads, 1, 1)`.

Keep 231296 dynamic shared bytes: Q, two KV buffers, the separate probability
tile, validity masks, statistics, and reduction scratch. Group-0 probabilities
continue to alias KV0's final key tile. There are 32 key tiles, traversed as 16
pairs. Keep nine 64-column QK tiles, four tiles per value half, 32 score registers,
128 output registers per consumer lane, and two logical rows per lane.

The copy mapping uses groups of eight lanes, 16 groups, four rows per lane, and
eight BF16 elements per chunk. For role-local lane `l`, row iteration `r`, and
column tile `t`, its logical KV row is `r*kCopyGroups + l/kCopyGroup`; its first
column is `t*kTile + (l%kCopyGroup)*kVectorElems`. Selected-row offsets remain
signed 64-bit products of the index and runtime `kv_stride=576`. Shared indexing
remains `kv_index` plus the existing tile/row offsets.

Retain scalar volatile KV transfers, unswizzled core layouts, WGMMA QK/PV,
shared-memory peer reductions, scalar statistics transfers, per-element
normalization and exponential recomputation, online softmax, direct output
stores, query reloads per key pair, and all existing compiler controls.

# Precondition

- Data types: no additional dtype or arithmetic requirement. Moving the copy
  role changes its owner, not the bits copied or the consumer's arithmetic.
- Layout: copy ownership must use role-local lanes in whole warps. The relocated
  mapping must cover each destination without omissions or duplicates; using new
  absolute thread indices without adapting addresses would misaddress data.
  Whole-warp roles must not split required warp collectives, which need uniform
  participation. No particular absolute thread range is required.
- Storage: staged operands and masks must already be visible within the CTA,
  and their transfer inputs must not depend on consumer-private arithmetic state.
  A relocated producer cannot directly supply another thread's private registers
  or derive unavailable inputs; it must publish into the existing shared storage.
- Pipeline: all roles must follow matching iteration and barrier order. Producer
  completion and visibility in every reader's memory domain must precede reads;
  all readers must complete before storage reuse. Counted barriers must name
  exactly their participants. These dependencies prevent stale data, overwrite
  races, and deadlock; no particular stage count is required.
- Hardware: independently scheduled warps, CTA-shared storage, and counted
  barriers must support the separated roles. Consumer plus producer threads must
  fit the CTA limit; actual rounded CTA register allocation and shared storage
  must fit one resident CTA. Otherwise the expanded launch cannot execute.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are focused replacement sites in `solution/attention.cuh`. Keep the rest
of `consume` and `attention` unchanged. The new wrapper calls the existing
`load_kv`; its implementation is already present in the deoptimized bundle.

## Before

```cuda
constexpr int kThreads = 2 * kWarpgroup;

// consume<Group>: start of each key-pair iteration.
if constexpr (Group == 0) {
    load_query(sm, params, lane, head, token);
    // The first consumer also loads KV; no dedicated producer is launched.
    load_kv(sm, params, lane, token, block);
}
sync<Barrier::Stage, kThreads>();

// attention: dispatch tail.
if (group == 0) {
    consume<0>(sm, params, lane, head, token);
    return;
}
consume<1>(sm, params, lane, head, token);
```

## After

```cuda
constexpr int kThreads = 3 * kWarpgroup;

// consume<Group>: start of each key-pair iteration.
if constexpr (Group == 0) load_query(sm, params, lane, head, token);
sync<Barrier::Stage, kThreads>();

// Add after load_kv, before attention.
__device__ __forceinline__ void produce(
    Shared& sm, const Params& params, int lane, int token) {
#pragma unroll 1
    for (int block = 0; block < kBlocks; block += 2) {
        load_kv(sm, params, lane, token, block);
        // Publish operands, then wait until every reader finishes.
        sync<Barrier::Stage, kThreads>();
        sync<Barrier::Stage, kThreads>();
    }
}

// attention: dispatch tail.
if (group == 0) {
    consume<0>(sm, params, lane, head, token);
    return;
}
if (group == 1) {
    consume<1>(sm, params, lane, head, token);
    return;
}
produce(sm, params, lane, token);
```
