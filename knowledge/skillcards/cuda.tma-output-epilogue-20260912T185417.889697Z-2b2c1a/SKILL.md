---
skill_id: cuda.tma-output-epilogue
intent: Gather final values in shared memory for bulk asynchronous output stores.
preconditions:
- 'Data types: final values must retain their representation through staging; the
  tensor-map element format must match the stored bits. No additional arithmetic precision
  is required because bulk stores only move data.'
- 'Layout: each transfer has known, nonoverlapping ownership within one CTA and a
  dense tensor box or contiguous byte range. Tensor destinations need a contiguous
  inner dimension, 16-byte-aligned global bases and outer byte strides, and inner
  box extents in multiples of 16 bytes; tensor scratch needs 128-byte alignment. Linear
  bulk-copy endpoints and sizes need 16-byte alignment. These are copy/descriptor
  validity rules.'
- 'Storage: completed output values currently reside in CTA threads'' registers with
  global destinations; CTA-local scratch can be reserved or safely reclaimed to gather
  each transfer. A shared-to-global copy cannot gather registers or another CTA''s
  private storage directly.'
- 'Pipeline: value producers and old users of reclaimed scratch must finish before
  staging; every staging writer must reach visibility fences and its transfer barrier.
  Sources must remain unchanged until copy reads complete, and output consumers must
  follow copy completion. Otherwise bulk copies can observe incomplete or overwritten
  data.'
- 'Hardware: SM90-or-later TMA tensor stores, linear bulk stores, tensor-map encoding,
  proxy fences, and CTA barriers are required. Shared capacity must satisfy S_available
  >= max_t bytes(live_regions(t)), counting compute and epilogue allocations, alignment
  padding, and pending transfer sources; aliased regions count once.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Gather each CTA's final values in shared memory and issue bulk stores from
one elected thread per output tile. This replaces distributed global stores
for both the attention output and its maximum/LSE statistics. Preserve all
arithmetic; staging copies the already rounded output bits.

Apply the Before/After replacements in `solution/attention.cuh`. The snippets
contain complete `save_output` and `store_output` functions and the tail of
`consume`; retain the surrounding `consume` body and its closing brace.
`save_output` changes its row stride from the global output width to the
shared tile width. Its thread-to-element mapping stays unchanged.

## Wiring and storage

- In `Params`, remove `Bf16* output`; remove its matching initializer in
  `solution/kernel.cu`. Append `CUtensorMap output` after `Maps.query`.
- In the existing `Maps maps` initializer, retain the query map and append
  `tensor_map(output.data_ptr<at::BFloat16>(), kValue,
  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_SWIZZLE_NONE)`.
  Reuse the existing encoder: output map dimensions are `(kValue,kHeads,kTokens)`,
  byte strides `(kValue*sizeof(Bf16),kValue*kHeads*sizeof(Bf16))`, box
  `(kTile,kTile,1)`, unit element strides, and no interleave or OOB fill.
- Add `prefetch(&maps.output)` next to the query prefetch in `attention`.
- Append `Store0, Store1` after `Sum` in `hopper::Barrier`; their IDs become
  13 and 14. Add the three supplied store helpers inside `namespace hopper`.
- Add `float final_max[kTile]` and `float final_lse[kTile]` after `Shared.sum`
  and before its barriers. Set `kSharedBytes = 230864`. Reuse `q_o` for the
  output tiles only after all Q readers finish. The launch already uses
  `sizeof(Shared)` for both its opt-in attribute and dynamic allocation.

## Ownership and ordering

CTA `b` owns token `b/(kHeads/kTile)` and heads starting at
`(b%(kHeads/kTile))*kTile`. For lane `lane = threadIdx.x % kWarpgroup`,
row `r` is `row_index(r,lane)` and register pair starting at `i` owns columns
`8*(i/4)+(lane%4)*2` and the next column. Group `g` and tile `t` own output
columns `(g*kHalfTiles+t)*kTile + column`. Global output is contiguous
`[token,head,value]`; shared tile `j` starts at `q_o+j*kTileElems` and is
row-major. Every output bit has one writer. No output swizzle is introduced.

Keep the final WGMMA waits and `reduce_sum` barrier: both consumers must
finish QK/PV before reusing Q storage. Every output writer executes
`shared_fence()` before its group's barrier. One elected lane in the first
warp of each consumer group then issues each tensor store. Group 1 stages
statistics with `lane%4==0`; those writers fence before the group barrier,
then lane 0 issues both linear bulk stores.

Retain the supplied terminal `store_commit()` calls. Each output tile and
statistics array occupies a distinct source region, written once and never
reused afterward in this CTA; there is no in-kernel output reader. Keep the
caller stream so downstream work follows kernel completion. If adapting this
epilogue to reuse a source, first have its issuing thread commit and execute
`cp.async.bulk.wait_group.read 0`, then synchronize all reusing writers;
an in-kernel global reader needs full copy completion instead.

Keep all tensor checks and the exact launch domain: this workload has full
tiles, so the epilogue needs no tail predicate. Preserve invalid-index masking,
zero-fill, softmax scale, reduction order, normalization, infinity branches,
and `__float2bfloat16_rn` placement. Do not change compiler flags or
`solution/mma.cuh`. Validate the complete problem with the registered evaluator;
inspect generated code for tensor/linear bulk stores and their commits.

## Example configuration

Preserve tokens=8192, heads=128, QK width=576, value width=512, and top-k=2048.
Inputs and output use BF16; scores, accumulators, maximum, and LSE use FP32.
Scale remains `0.1352337788608801f`. Each CTA covers 64 heads for one token;
the grid has 16384 CTAs, 384 threads per CTA, and cluster `(1,1,1)`.
Two 128-thread consumer groups each own 256 value columns; one 128-thread
producer group loads KV. Keep consumer/producer register budgets 216/72,
32 score registers, 128 output registers per consumer thread, two row slices,
four output tiles per consumer, and all existing unroll directives.

Each 64x64 output tile is 8192 bytes, with shared row stride 128 bytes and
global row stride 1024 bytes. All eight tiles occupy 65536 bytes of the dead
73728-byte Q region. The two 64-element statistics arrays add 512 bytes,
raising `sizeof(Shared)` from 230352 to 230864 bytes. Retain the dynamic shared
base used by the existing TMA query load; tile offsets preserve its alignment.
Keep two KV buffers, four transfer groups, mbarrier phase rotation, overlapped
QK/PV, query TMA, KV `cp.async`, input/probability swizzles, WGMMA operand
adapters, online softmax, warp reductions, and the current cache policies.
These are replay settings, not additional prerequisites for bulk output stores.

# Precondition

- Data types: staging must preserve final representations, and the tensor map
  must describe the bits being stored. No additional arithmetic requirement
  applies: neither tensor nor linear bulk copies perform the reductions or
  rounding. Linear statistics copies have no element-type requirement.
- Layout: a transfer's values have known, nonoverlapping ownership inside one
  CTA and form a dense tensor box or contiguous byte range. The tensor path
  needs a contiguous inner dimension, global base and outer byte strides
  divisible by 16, inner box bytes divisible by 16, and a 128-byte-aligned
  shared source. Linear copies require 16-byte-aligned endpoints and byte
  counts divisible by 16. These constraints make descriptors and copy
  addresses legal; ownership allows complete gathering without write races.
- Storage: final values currently live in CTA threads' registers with global
  destinations. Suitable CTA-local scratch must be reservable or reclaimable:
  shared-to-global instructions cannot read registers or another CTA's private
  storage. The optimization introduces staging; shared residency of the final
  values is not a prerequisite.
- Pipeline: producers and prior scratch readers must complete before staging.
  All staging writers must reach their proxy fence and transfer barrier so
  the elected issuer observes complete data. Source storage must remain
  unchanged until asynchronous reads finish, and consumers must follow copy
  completion. These dependencies prevent uninitialized reads and premature
  reuse; no particular input stage count is required.
- Hardware: the target must provide SM90-or-later TMA tensor stores, linear
  bulk stores, tensor-map encoding, proxy fences, and CTA barriers. Available
  shared capacity must satisfy `S_available >= max_t bytes(live_regions(t))`,
  counting compute/epilogue allocations, padding, and pending transfer sources
  with aliased regions counted once. Otherwise
  this gathering and transfer path cannot be issued or stored legally.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// solution/attention.cuh: save_output
__device__ __forceinline__ void save_output(const Bf16* p, Bf16* dest, int lane) {
    constexpr int kLanesPerRow = 4;
    constexpr int kPairElems = 2;
    constexpr int kColStep = kLanesPerRow * kPairElems;
    constexpr int kRegsPerStep = kRowsPerThread * kPairElems;

    // Write each lane's elements directly to their global row.
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
        const int real_row = row_index(row, lane);
#pragma unroll
        for (int i = row * kPairElems; i < kScoreRegs; i += kRegsPerStep) {
            const int col = kColStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
            dest[real_row * kValue + col] = p[i];
            dest[real_row * kValue + col + 1] = p[i + 1];
        }
    }
}

// solution/attention.cuh: store_output
template<int Group>
__device__ __forceinline__ void store_output(
    const Params& params, int lane, int head, int token,
    float (&o)[kOutputRegs], float (&l)[2]) {
    float scale[2];
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row)
        scale[row] = l[row] == 0.0f ? 0.0f : 1.0f / l[row];

    wait<0>();
#pragma unroll
    for (int tile = 0; tile < kHalfTiles; ++tile) {
        Bf16 converted[kScoreRegs];
#pragma unroll
        for (int i = 0; i < kScoreRegs; ++i)
            converted[i] = __float2bfloat16_rn(o[tile * kScoreRegs + i] * scale[i % 4 >= 2]);
        const int global_tile = Group * kHalfTiles + tile;
        Bf16* dest = params.output + (token * kHeads + head) * kValue + global_tile * kTile;
        save_output(converted, dest, lane);
    }
}

// solution/attention.cuh: tail of consume, inside its function body
    reduce_sum(sm, lane, l);
    store_output<Group>(params, lane, head, token, o, l);
    if constexpr (Group == 1) {
        if (lane % 4 == 0) {
#pragma unroll
            for (int row = 0; row < kRowsPerThread; ++row) {
                const int real_row = row_index(row, lane);
                const int offset = token * kHeads + head + real_row;
                params.maximum[offset] = l[row] == 0.0f ? -INFINITY : m[row] * CUDART_LN2_F;
                params.lse[offset] = l[row] == 0.0f ? INFINITY : logf(l[row]) + m[row] * CUDART_LN2_F;
            }
        }
    }
```

## After

Add these hardware helpers to `solution/hopper.cuh`:

```cuda
__device__ __forceinline__ void store_tile(
    const CUtensorMap* map, const Bf16* src, int col, int head, int token) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
                 :: "l"(map), "r"(shared_addr(src)), "r"(col), "r"(head), "r"(token) : "memory");
}

__device__ __forceinline__ void store_stats(const float* src, float* dst) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(dst), "r"(shared_addr(src)), "r"(kTile * int(sizeof(float))) : "memory");
}

__device__ __forceinline__ void store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
```

Replace the corresponding epilogue fragments after applying the wiring above:

```cuda
// solution/attention.cuh: save_output
__device__ __forceinline__ void save_output(const Bf16* p, Bf16* dest, int lane) {
    constexpr int kLanesPerRow = 4;
    constexpr int kPairElems = 2;
    constexpr int kColStep = kLanesPerRow * kPairElems;
    constexpr int kRegsPerStep = kRowsPerThread * kPairElems;

    // Store each lane's elements at the row-major coordinates.
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
        const int real_row = row_index(row, lane);
#pragma unroll
        for (int i = row * kPairElems; i < kScoreRegs; i += kRegsPerStep) {
            const int col = kColStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
            dest[real_row * kTile + col] = p[i];
            dest[real_row * kTile + col + 1] = p[i + 1];
        }
    }
}

// solution/attention.cuh: store_output
template<int Group>
__device__ __forceinline__ void store_output(
    Shared& sm, const Maps& maps, int lane, int warp, int head, int token,
    float (&o)[kOutputRegs], float (&l)[2]) {
    float scale[2];
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row)
        scale[row] = l[row] == 0.0f ? 0.0f : 1.0f / l[row];

    const bool writer = warp % 4 == 0 && elect();
    wait<0>();
#pragma unroll
    for (int tile = 0; tile < kHalfTiles; ++tile) {
        Bf16 converted[kScoreRegs];
#pragma unroll
        for (int i = 0; i < kScoreRegs; ++i)
            converted[i] = __float2bfloat16_rn(o[tile * kScoreRegs + i] * scale[i % 4 >= 2]);
        save_output(converted, sm.q_o + (Group * kHalfTiles + tile) * kTileElems, lane);
        shared_fence();
        sync<Group == 0 ? Barrier::Store0 : Barrier::Store1, kWarpgroup>();
        if (writer) {
            const int global_tile = Group * kHalfTiles + tile;
            store_tile(&maps.output, sm.q_o + global_tile * kTileElems,
                       global_tile * kTile, head, token);
        }
    }
    store_commit();
}

// solution/attention.cuh: tail of consume, inside its function body
    reduce_sum(sm, lane, l);
    store_output<Group>(sm, maps, lane, warp, head, token, o, l);
    if constexpr (Group == 1) {
        if (lane % 4 == 0) {
#pragma unroll
            for (int row = 0; row < kRowsPerThread; ++row) {
                const int real_row = row_index(row, lane);
                sm.final_max[real_row] = l[row] == 0.0f ? -INFINITY : m[row] * CUDART_LN2_F;
                sm.final_lse[real_row] = l[row] == 0.0f ? INFINITY : logf(l[row]) + m[row] * CUDART_LN2_F;
            }
            shared_fence();
        }
        sync<Barrier::Store1, kWarpgroup>();
        if (lane == 0) {
            const int offset = token * kHeads + head;
            store_stats(sm.final_max, params.maximum + offset);
            store_stats(sm.final_lse, params.lse + offset);
            store_commit();
        }
    }
```
