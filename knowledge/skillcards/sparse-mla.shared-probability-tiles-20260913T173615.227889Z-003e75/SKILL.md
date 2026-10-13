---
skill_id: sparse-mla.shared-probability-tiles
intent: Keep reused probability tiles in CTA shared memory.
preconditions:
- 'Data types: no particular arithmetic dtype is required; moving each stored element
  without changing its representation or conversion points preserves the values consumed
  by PV.'
- 'Layout: every tile''s writers and readers belong to one CTA, with known element
  indexing and unique or ordered writes; shared indexing must preserve that ownership
  to supply the same probability to each reader.'
- 'Storage: the preceding implementation uses per-CTA global scratch for intermediate
  probabilities, with no external or cross-CTA consumers; CTA-local storage cannot
  satisfy such external reads.'
- 'Pipeline: participating threads can complete and publish tile writes before reads,
  and finish all reads before overwrite, using correctly participating CTA synchronization;
  otherwise readers race producers or buffer reuse.'
- 'Hardware: CTA shared memory and synchronization are available; existing live shared
  state plus all simultaneously live probability tiles and alignment padding must
  fit the configured per-CTA shared-memory limit, or the allocation and launch cannot
  support this placement.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Move probability intermediates from CTA-private global scratch into `Shared`.
`save_prob` writes each tile once per publication; `pv_smem` reuses its elements
across output columns and both consumer groups. Shared placement changes these
probability transfers from global to shared memory. KV remains in shared memory.

Apply these edits to the supplied bundle:

1. In `solution/attention.cuh`, insert `Bf16 prob[kTileElems]` and
   `Bf16 prob0[kTileElems]` immediately after `Shared::kv`. Include their sizes in
   `kSharedBytes`, preserving the existing size assertion and other fields.
2. In `consume`, replace the two global tile pointers with `sm.prob` and `sm.prob0`
   as shown below. Keep the local pointer names and every use unchanged.
   `prob0` holds group 0 probabilities; `prob` holds group 1 probabilities.
   Keep `save_prob`, `pv_smem`, `prob_index`, lane ownership, and all barriers.
3. Remove `Params::probabilities` and `kProbBuffers`. In
   `solution/kernel.cu::kernel`, remove the `torch::empty` probability allocation
   and its final pointer argument in the `Params` initializer; `kWidth` becomes
   the final argument. Preserve the six tensor arguments and configured export.
4. Keep the launch grid, threads, cluster, device guard, and caller stream.
   Both the dynamic shared-memory attribute and launch size already use
   `sizeof(Shared)` and therefore increase automatically. Keep compiler flags.

Each CTA owns its complete probability tiles. `save_prob` maps each lane's
register pairs into the existing unswizzled core layout. `pv_smem` reads the same
layout through `prob_index`; only the probability pointer's memory space changes.
Its value pointer still addresses shared KV. No new copy loop, permutation,
cross-CTA exchange, or output staging is needed.

Retain `Local0`/`Local1` publication before local PV. Group 0 completes local PV
and the `Max1` rendezvous before rewriting its tile with rescaled probabilities.
Retain `Prob0` publication before group 1 reads that rewrite, and `Prob1`
publication before group 0 reads group 1's tile. The final `Stage` barrier finishes
all readers before the next iteration overwrites either tile. Keep `Max0`, `Max1`,
`Sum`, and warp exchanges unchanged; their statistics ordering is independent.
The barriers already order shared and global memory for their participants.

Retain all score masks and KV zero filling for indices outside `[0,kTokens)`.
Every probability slot is written before consumption; no scratch initialization
or new tile bounds check is needed. Keep reduction order, exponential arguments,
probability rounding, normalization, and output conversion unchanged. Check the
restored bundle with the existing evaluator and unchanged problem. Inspect the
compiled `save_prob` stores and `pv_smem` probability loads to confirm shared
addressing; timing alone does not establish placement.

## Example configuration

Preserve `TOKENS=8192`, `HEADS=128`, `QK_DIM=576`, `VALUE_DIM=512`, and `TOPK=2048`.
Inputs are contiguous BF16 Q/KV and int32 indices. Outputs are BF16 values and
FP32 maxima/LSE. Keep scale `0.1352337788608801f`, FP32 scalar FMA accumulations,
online softmax, BF16 probability conversion before PV, and final BF16 rounding.
Group 0 traverses QK width tiles `0..8`; group 1 traverses `4..8,0..3`.

The grid has 16384 CTAs: one token and 64-head slice per CTA. Each CTA has 256
threads in two 128-thread consumer groups and a one-CTA cluster. The 32 selected-key
blocks are traversed two at a time. Retain register tiling, both shared KV buffers,
scalar volatile KV transfers and index reloads, shared statistic exchanges,
per-element rescaling/normalization, and direct global output stores.

The two probability tiles each contain `64*64` BF16 elements. Their layout is
`prob_index(r,c) = (r/8)*8*64 + (c/8)*64 + (r%8)*8 + c%8`.
`row_index(row,lane) = (lane/32)*16 + row*8 + (lane%32)/4`;
register pair `i,i+1` owns columns `8*(i/4)+(lane%4)*2` and its successor.
Keep this writer/reader mapping and the existing `prob_index` helper unchanged.

Before replay, scratch shape is `[16384,2,4096]`, totaling 268435456 bytes. Each
CTA's first tile is `prob`, followed by `prob0`; the caller-stream allocation lives
through launch and its allocator orders later reuse on that stream. Replay
removes that allocation. Shared bytes increase from 149376 to 165760 per CTA.
The added tiles remain distinct from one another and from KV; retain group 1's
reuse of its own tile for local and peer PV. The `Local0`/`Local1` barriers use
128 participants; the intergroup named barriers and `Stage` use 256.
These dimensions, counts, and layouts are replay settings, not technique limits.

# Precondition

- Data types: no particular arithmetic dtype is required. Preserve each stored
  element's representation and all conversion points, so relocating the tile
  supplies exactly the same operand values to PV.
- Layout: all writers and readers of a tile must be in one CTA, with known
  element indexing and unique or ordered writes. Shared memory is CTA-local;
  mismatched indexing or ownership would expose another element or a write race.
  No additional vector alignment or global row-contiguity constraint is imposed
  by this placement change.
- Storage: probabilities currently occupy per-CTA global scratch and have no
  external or cross-CTA consumers. Moving them into CTA-local storage would make
  them unavailable to any such consumer.
- Pipeline: tile producers must complete and publish writes before consumers
  read, and every consumer must finish before the tile is overwritten. The
  participating threads must reach the corresponding CTA synchronization with
  correct participant counts; otherwise publication or reuse races occur.
- Hardware: CTA shared memory and synchronization must be available. Let
  `S_live` be the existing live shared state and tile `j` contain `E_j` elements
  of stored type `T_j`. Require
  `S_live + sum_j(E_j*sizeof(T_j)) + padding <= configured_CTA_shared_limit`
  for simultaneously live tiles. Exceeding this capacity prevents the shared
  allocation and launch; it does not require a particular example tile count.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The fragments identify the changed regions; retain surrounding fields and code.
Remove the host allocation and final `Params` initializer argument as directed
above. No `save_prob` or `pv_smem` call-site changes are needed.

## Before

```cuda
// attention.cuh: shared storage contains no probability tiles.
constexpr int kProbBuffers = 2;
// Shared begins with: Bf16 kv[2][kTile * kWidth];
constexpr int kSharedBytes = sizeof(Shared::kv) + sizeof(Shared::valid) +
                             sizeof(Shared::maximum) + sizeof(Shared::sum) +
                             sizeof(Shared::reduction);
static_assert(sizeof(Shared) == kSharedBytes);

// Params: final field, following kv_stride.
Bf16* probabilities;

// consume: CTA-private global storage, before the block loop.
Bf16* prob = params.probabilities + blockIdx.x * kProbBuffers * kTileElems;
Bf16* prob0 = prob + kTileElems;

// kernel.cu::kernel: allocation before the Params initializer.
const auto probabilities = torch::empty(
    {kTokens * (kHeads / kTile), kProbBuffers, kTileElems}, q.options());
// Params initializer's final argument:
// reinterpret_cast<Bf16*>(probabilities.data_ptr<at::BFloat16>())
```

## After

```cuda
// attention.cuh: insert directly after Shared::kv.
Bf16 prob[kTileElems];
Bf16 prob0[kTileElems];

// Account for both new fields; keep the remaining shared state intact.
constexpr int kSharedBytes = sizeof(Shared::kv) + sizeof(Shared::valid) +
                             sizeof(Shared::maximum) + sizeof(Shared::sum) +
                             sizeof(Shared::reduction) + sizeof(Shared::prob) +
                             sizeof(Shared::prob0);
static_assert(sizeof(Shared) == kSharedBytes);

// consume: use the same aliases and all existing reads/writes/barriers.
Bf16* prob = sm.prob;
Bf16* prob0 = sm.prob0;

// Remove kProbBuffers, Params::probabilities, the torch::empty allocation,
// and its final Params initializer argument. Keep sizeof(Shared) at launch.
```
