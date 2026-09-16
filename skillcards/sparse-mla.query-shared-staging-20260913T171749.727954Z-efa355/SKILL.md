---
skill_id: sparse-mla.query-shared-staging
intent: Reuse query operands within a CTA through cooperative shared-memory staging.
preconditions:
- 'Data types: no specific dtype is required; staging must copy the query representation
  unchanged and retain its conversion and accumulation order to preserve numerical
  behavior.'
- 'Layout: query strides and each CTA reader''s row/column ownership must be known,
  with within-CTA reuse to amortize the copy; the cooperative writer and shared reader
  must address the same element, using legal element alignment.'
- 'Storage: query values are available in global memory and remain stable while the
  CTA uses them; consumers of each staged copy must share the CTA-local storage and
  synchronization scope used by this recipe.'
- 'Pipeline: input production must finish before copying, all CTA participants must
  reach the publish barrier before query reads, and readers must finish before buffer
  reuse; otherwise a consumer can read incomplete or overwritten data.'
- 'Hardware: CUDA shared memory, element loads/stores, and CTA barriers are required;
  existing shared storage plus the query tile and alignment padding must fit the per-CTA
  shared-memory limit.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Stage each CTA's query tile in shared memory so its QK consumers reuse one
cooperative global-memory copy. Change only the Q loading path; preserve scalar
QK/PV arithmetic, register ownership, KV staging, and softmax.

In `solution/attention.cuh`, prepend `Bf16 q_o[kTile * kWidth]` to `Shared`
and change `kSharedBytes` to `230272 + kConsumerThreads * sizeof(float)`.
Retain the `sizeof(Shared)` assertion. Add `load_query` before `row_index` and
add `q_index` beside `prob_index` in `solution/hopper.cuh`, as shown below.

The query source is contiguous `[token, head, qk]`; its row stride is
`kWidth * sizeof(Bf16)` bytes. Producer lane `lane` copies flattened offsets
`lane + n * kWarpgroup`. For source row `r`, column `c`, the shared offset is
`(c / kTile) * kTileElems + q_index(r, c % kTile)`.
The unswizzled core mapping is bijective; `qk_tile` uses the same mapping to
read its row and reduction coordinate. It adds no vector or asynchronous copy.

In `consume<Group>`, call `load_query(sm, params, lane, head, token)` immediately
before `load_kv` inside the existing `Group == 0` branch, on every key-pair
iteration. Keep the following `sync<Barrier::Stage, kThreads>()` outside that
branch: both consumer groups must wait until all query writes are visible.
Keep the end-of-iteration Stage barrier so no copy overwrites an active reader.
The caller stream orders input production before launch. No barrier is removed,
added, or moved; no early return is introduced.

In `qk_tile`, replace the global query base and query load with the shared
expressions below. Remove `params`, `head`, and `token` from the signatures and
calls of `qk_tile`, `qk_left`, `qk_right`, and `qk_peer`; each then takes only
`Shared& sm, float (&p)[kScoreRegs]`. Keep their template arguments and call
order unchanged. Update the two Stage-barrier comments to include Q.

`solution/kernel.cu` already derives both the shared-memory opt-in and launch
allocation from `sizeof(Shared)`, so neither host code nor launch geometry needs
editing. Keep the tensor checks, destination-passing ABI, and caller stream.
All query rows and columns in this workload are in bounds. For a partial tile,
guard producer loads and final outputs while keeping every barrier participant;
the current replay needs no new bounds predicates. Sparse-KV masks are unchanged.

## Example configuration

Preserve tokens=8192, heads=128, QK width=576, value width=512, and TOPK=2048.
Each of 16384 CTAs covers one token and 64 heads, using 256 threads in two
128-thread consumer groups. Group zero copies the complete Q tile. Each group
has its existing two-row-per-lane mapping; four lanes partition a row's scores.

Keep `kTile=64`, `kCoreRows=8`, `kCoreBytes=16`, and BF16 elements. The core
contains eight columns; `q_index` has no XOR swizzle. Q occupies 73728 bytes;
dynamic shared allocation grows from 157568 to 231296 bytes. Both builds also
use 1024 bytes of compiler-generated static shared storage. Preserve the two KV
buffers, 16 key-pair iterations, KV/probability storage alias, shared reductions,
and all named barriers. These are replay settings, not general staging requirements.

Preserve BF16 query/KV values, FP32 `fmaf` accumulators, BF16 probability rounding,
and final BF16 output rounding. Group zero visits QK tiles 0 through 8; group one
visits 4 through 8, then 0 through 3. Keep increasing `k` within each tile, scale
`0.1352337788608801f`, rescaling, normalization, maximum, and LSE calculations.
Retain all compiler flags, including fast math and register-usage settings.

Inspect generated code to confirm cooperative global-to-shared query writes and
shared query reads. Use the existing evaluator with the unchanged problem for
correctness and timing; keep results outside this card.

# Precondition

- Data types: no specific dtype is required. The copy must preserve query bits;
  its readers must retain the original conversion and accumulation order.
  Changing either would turn a storage optimization into a numerical change.
- Layout: query strides and each CTA reader's row/column ownership must be known,
  with reuse within that CTA to amortize the copy. Cooperative writers and consumers must map each
  query element to the same shared address. Element alignment must permit the
  loads and stores; otherwise the accesses are invalid. No vector alignment is
  additionally required by the scalar copy.
- Storage: queries must already be available in global memory and remain stable
  while the CTA uses them. Consumers of each staged copy must share this recipe's
  CTA-local storage and synchronization scope so its barriers cover every reader.
- Pipeline: input production must complete before the copy. Every CTA participant
  must reach the publish barrier before query reads, and all readers must finish
  before buffer reuse. These dependencies prevent incomplete reads and overwrite
  races; no fixed stage count is required.
- Hardware: CUDA shared memory, element loads/stores, and CTA barriers must be
  available. Capacity must satisfy `bytes(existing static and dynamic storage) +
  tile_heads * query_width * sizeof(query_element) + alignment_padding <=
  per_CTA_shared_limit`; exceeding it prevents the launch.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These focused excerpts share the existing constants, `Params`, `row_index`,
and QK loops. Retain omitted fields and arithmetic. Apply the signature/call
edits described above to all three QK wrappers.

## Before

```cuda
// attention.cuh: shared allocation; Shared starts with kv.
constexpr int kSharedBytes = 156544 + kConsumerThreads * sizeof(float);
struct alignas(16) Shared {
    Bf16 kv[2][kTile * kWidth];
    // Remaining fields unchanged.
};

// consume<Group>: beginning of each key-pair iteration.
if constexpr (Group == 0) {
    load_kv(sm, params, lane, token, block);
}
sync<Barrier::Stage, kThreads>();

// qk_tile: base outside the K loop; load inside its row loop.
const Bf16* q = params.q + (token * kHeads + head) * kWidth + Tile * kTile;
// ... existing K and row loops ...
const float a = __bfloat162float(q[row_index(row, lane) * kWidth + k]);
```

## After

```cuda
// hopper.cuh: writer and reader share this unswizzled core mapping.
__device__ __forceinline__ int q_index(int row, int col) {
    return (row / kCoreRows) * kCoreRows * kTile +
           (col / kCoreElems) * kCoreTileElems +
           (row % kCoreRows) * kCoreElems + col % kCoreElems;
}

// attention.cuh: prepend Q; retain every other Shared field.
constexpr int kSharedBytes = 230272 + kConsumerThreads * sizeof(float);
struct alignas(16) Shared {
    Bf16 q_o[kTile * kWidth];
    Bf16 kv[2][kTile * kWidth];
    // Remaining fields unchanged.
};

__device__ __forceinline__ void load_query(
    Shared& sm, const Params& params, int lane, int head, int token) {
    const Bf16* src = params.q + (token * kHeads + head) * kWidth;

    // Copy each query element once for reuse by both consumer groups.
    for (int i = lane; i < kTile * kWidth; i += kWarpgroup) {
        const int row = i / kWidth;
        const int col = i % kWidth;
        sm.q_o[(col / kTile) * kTileElems + q_index(row, col % kTile)] = src[i];
    }
}

// consume<Group>: preserve both Stage barriers and all participants.
if constexpr (Group == 0) {
    load_query(sm, params, lane, head, token);
    load_kv(sm, params, lane, token, block);
}
sync<Barrier::Stage, kThreads>();

// qk_tile: base outside the K loop; load inside its row loop.
const Bf16* q = sm.q_o + Tile * kTileElems;
// ... existing K and row loops ...
const float a = __bfloat162float(q[q_index(row_index(row, lane), k)]);
```
