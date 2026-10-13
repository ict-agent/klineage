---
skill_id: cuda.tma-resident-query-loading
intent: Load resident query tiles with the tensor memory accelerator.
preconditions:
- 'Data types: the tensor-map element format must preserve the source bits; the transfer
  introduces no arithmetic or rounding requirement.'
- 'Layout: source tiles need tensor-map-encodable affine strides and contiguous innermost
  elements. For the retained 128-byte swizzle, source alignment is 128 bytes, outer
  byte strides are multiples of 16, and destination alignment is at least 128 bytes.
  Tensor-map swizzle phase and consumer indexing must agree; otherwise consumers read
  different elements.'
- 'Storage: queries already reside in device global memory and feed a CTA-local shared
  buffer. Both allocations must remain valid through transfer completion; this instruction
  cannot replace a transfer from another memory space.'
- 'Pipeline: source production must precede the copy on the caller stream. Every consumer
  must be able to wait for publication before reading; shared storage must not be
  overwritten until all consumers finish. These dependencies prevent incomplete reads
  and reuse races.'
- 'Hardware: tensor-map loads and transaction-counted shared mbarriers require SM90
  or later and a compatible CUDA toolchain/driver. CTA shared capacity must cover
  the resident query, other live storage, and one aligned 8-byte completion barrier.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace the cooperative query copy with TMA loads into the existing resident
query buffer. One elected thread launches the transfers; both consumer
warpgroups wait on their completion. This also lets KV production begin while
the query transfer is outstanding. Preserve query reuse and its shared swizzle.

Apply the following edits within `solution/`:

1. In `attention.cuh`, remove `load_query` and its call before the startup
   `__syncthreads()`. Keep that barrier for initialization of all mbarriers.
   Replace `Params::q` with a separate `Maps` structure holding `CUtensorMap query`.
   Add `query` before `Shared::free0`. The existing shared-size assertion remains
   valid: the added barrier occupies the former trailing alignment allowance.
2. Add `Mbar::expect` and `load_q` from After to `hopper.cuh`. The former counts
   all query bytes against one barrier phase; the latter copies one tensor tile.
3. Add `__grid_constant__ const Maps maps` after `params` in `attention`.
   Initialize `sm.query.init(1)` in its existing `warp == 0 && elect()` branch,
   before `init_fence()`. Keep the existing CTA initialization barrier.
4. Change `consume` arguments to
   `(Shared& sm, const Params& params, const Maps& maps, int lane, int warp,
   int head, int token)` and pass these from both calls in `attention`.
   Insert the After query-issue block at the start of `consume`.
   Insert `sm.query.wait(0)` after the local accumulator declarations and before
   `int phase = 0`. Every consumer must wait, including Group 1.
5. In `kernel.cu`, add the encoder helpers from After inside the anonymous
   namespace, after the existing `check_cuda`. Remove the query pointer from
   the `Params` initializer; retain its other fields in their current order.
   Build `Maps` from the query's current data pointer on every invocation and
   append `maps` to `cudaLaunchKernelEx`. Preserve the tensor ABI, device guard,
   caller stream, launch geometry, shared-memory opt-in, and launch checks.

The tensor map describes global `[token, head, column]` storage with column
innermost. Tile coordinate `(tile*kTile, head, token)` lands at
`sm.q_o + tile*kTileElems`. TMA's 128-byte swizzle must reproduce the existing
`swizzle(row, col)` layout consumed by `desc_k`; do not change that descriptor.
The query occupies offset zero within the workspace, whose compiled shared base
is 1024 bytes; tile starts advance by whole swizzle periods. This gives the zero
phase used by the existing XOR.
If placement changes, account for `(shared_address / 128) % 8` in both layouts;
zero-phase indexing requires a 1024-byte boundary.

Initialize and publish the query barrier before issue. The elected issuer
submits every tile and calls `expect` once with their total byte count.
The completion wait makes all query bytes available before QK.
The removed loader's generic-store proxy fence is unnecessary on this TMA path;
retain every probability proxy fence, WGMMA fence/wait, and KV barrier.
Query storage stays read-only until the CTA finishes, so it needs no reuse phase.

All query tiles are in bounds in this configuration. Keep the exact tensor-map
extents and unit element strides; no query padding is consumed. Retain the
independent sparse-index validation, invalid-KV zero-fill, and score masking.
TMA copies query bits without conversion. Keep all QK/PV instruction order,
FP32 softmax/reduction arithmetic, scaling, and BF16 round-to-nearest conversions.

Check that each source element retains its shared destination and that every
consumer waits before QK. After replay, inspect generated code for tensor loads
and use `klineage.harness.evaluate` with the unchanged problem for correctness
and timing. Keep measurements outside this card.

Descriptor encoding and swizzle alignment follow the
[CUDA tensor-map API](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
and [CUDA swizzle rules](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#the-swizzle-modes).

## Example configuration

Preserve `TOKENS=8192`, `HEADS=128`, `QK_DIM=576`, `VALUE_DIM=512`, and `TOPK=2048`.
Queries/KV/output are BF16, indices are int32, and maximum/LSE are FP32.
Global query byte strides are `(147456,1152,2)`; the map sizes are
`(576,128,8192)` with outer byte strides `(1152,147456)`.
Use nine `64x64x1` TMA boxes, each 8192 bytes, totaling 73728 query bytes.
Use `BFLOAT16`, `INTERLEAVE_NONE`, `SWIZZLE_128B`, `L2_PROMOTION_NONE`, and
`FLOAT_OOB_FILL_NONE`; add no descriptor prefetch or cache policy.

Launch 16384 CTAs, 384 threads each, cluster `(1,1,1)`, on the caller stream.
Group 0 and Group 1 each have 128 consumer threads; Group 2 retains KV production.
Warp 0's elected thread issues all query tiles. Consumer output ownership remains
`row_index`, with two rows per thread, 32 score registers and 128 output registers.
Keep both KV buffers, their four transfer groups, QK/PV overlap, online softmax,
unswizzled probability cores, shared reduction exchanges, and direct output stores.
Retain SM90a WGMMA and all compile flags, including fast math and register-usage
level 10. These are retained computation settings, not TMA prerequisites.

`Shared` contains 231368 payload bytes before replay and 231376 afterward;
its 16-byte struct alignment makes `sizeof(Shared)` 231376 in both cases.
Keep `kSharedBytes`, `Shared` order except for the restored query barrier,
and the existing 16-byte workspace declaration. Shared Q starts at address 1024
in the compiled layout. Preserve the fixed YaRN scale `0.1352337788608801f`,
initial maximum, log-base conversion, and all existing numerical grouping.

# Precondition

- Data types: the selected tensor-map element format must reproduce the source
  representation exactly. This is a bit transfer, so BF16 and a particular
  accumulation precision are not technique requirements. A format mismatch
  changes what the consumer reads.
- Layout: each source tile must have affine strides encodable by TMA and a
  contiguous inner dimension. With the retained 128-byte swizzle, source
  addresses require 128-byte alignment, outer byte strides must be multiples
  of 16, and shared destinations require at least 128-byte alignment. The
  descriptor's swizzle phase must match consumer indexing, or element ownership
  changes. These are address/encoding constraints, not fixed tensor extents.
- Storage: the existing copy must read device global memory into CTA-local
  shared storage. Source and destination allocations must outlive completion;
  the selected instruction addresses these memory spaces and cannot operate
  on expired storage.
- Pipeline: source production must finish before the transfer through caller-
  stream ordering. Every consumer must wait for publication before its first
  read. Writers cannot reuse the shared allocation until all readers finish.
  Otherwise readers can observe partial or overwritten data. No fixed stage
  count is required by this transfer.
- Hardware: SM90-or-later tensor-map loads, transaction-counted shared
  mbarriers, and a supporting CUDA toolchain/driver are required to encode and
  execute the instructions. Available CTA shared memory must accommodate the
  resident query plus retained live storage plus an aligned 8-byte barrier,
  including allocation padding. WGMMA is a retained consumer, not a prerequisite
  of moving these bits.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

`attention.cuh` currently loads Q before the startup CTA barrier:

```cuda
__device__ __forceinline__ void load_query(
    Shared& sm, const Params& params, int lane, int head, int token) {
    const Bf16* src = params.q + (token * kHeads + head) * kWidth;

    // Preserve the tiled swizzle expected by QK's shared descriptors.
    for (int i = lane; i < kTile * kWidth; i += kWarpgroup) {
        const int row = i / kWidth;
        const int col = i % kWidth;
        sm.q_o[(col / kTile) * kTileElems + swizzle(row, col % kTile)] = src[i];
    }

    // Publish generic stores to WGMMA's asynchronous proxy.
    shared_fence();
}

// In attention, after mbarrier initialization:
if (group == 0) load_query(sm, params, lane, head, token);
__syncthreads();
```

## After

These focused additions use existing `Bf16`, `Mbar`, `shared_addr`, dimensions,
and CUDA headers. Integrate the signatures, field changes, and call sites listed
in Overview; leave the rest of the computation unchanged.

```cuda
// attention.cuh: replace Params::q with separate map storage.
struct Maps {
    CUtensorMap query;
};

// Shared: add query before the existing barrier fields.
Mbar query, free0[2], ready0[2], free1[2], ready1[2], mask;

// hopper.cuh: add this method inside Mbar.
__device__ __forceinline__ void expect(int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(shared_addr(this)), "r"(bytes) : "memory");
}

// hopper.cuh: add beside copy_kv.
__device__ __forceinline__ void load_q(
    const CUtensorMap* map, Bf16* dst, Mbar& bar, int col, int head, int token) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%3, %4, %5}], [%2];"
                 :: "r"(shared_addr(dst)), "l"(map), "r"(shared_addr(&bar)),
                    "r"(col), "r"(head), "r"(token) : "memory");
}

// attention: initialize query in the existing elected initializer.
sm.query.init(1);  // Before init_fence() and the startup __syncthreads().

// consume: issue once at function entry, after startup __syncthreads().
if constexpr (Group == 0) {
    if (warp == 0 && elect()) {
#pragma unroll
        for (int tile = 0; tile < kKeyTiles; ++tile)
            load_q(&maps.query, sm.q_o + tile * kTileElems,
                   sm.query, tile * kTile, head, token);
        sm.query.expect(kTile * kWidth * sizeof(Bf16));
    }
}

// consume: after accumulator declarations, before int phase = 0.
sm.query.wait(0);

// kernel.cu: host descriptor helpers, after check_cuda.
constexpr unsigned int kTensorMapVersion = 12000;
using EncodeMap = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeMap map_encoder() {
    static const auto encode = [] {
        void* function = nullptr;
        cudaDriverEntryPointQueryResult query;
        check_cuda(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function,
                   kTensorMapVersion, cudaEnableDefault, &query));
        TORCH_CHECK(query == cudaDriverEntryPointSuccess, "Tensor map encoder unavailable");
        return reinterpret_cast<EncodeMap>(function);
    }();
    return encode;
}

CUtensorMap tensor_map(void* address, int width, CUtensorMapL2promotion promotion,
                       CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    const cuuint64_t sizes[3] = {cuuint64_t(width), kHeads, kTokens};
    const cuuint64_t strides[2] = {width * sizeof(Bf16), width * kHeads * sizeof(Bf16)};
    const cuuint32_t box[3] = {kTile, kTile, 1};
    const cuuint32_t element_strides[3] = {1, 1, 1};
    const auto status = map_encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, address,
        sizes, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status == CUDA_SUCCESS, "Tensor map encoding failed: ", int(status));
    return map;
}

// kernel: retain Params fields after removing its query pointer.
const Maps maps{tensor_map(q.data_ptr<at::BFloat16>(), kWidth,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_SWIZZLE_128B)};
check_cuda(cudaLaunchKernelEx(&config, attention, params, maps));
```
