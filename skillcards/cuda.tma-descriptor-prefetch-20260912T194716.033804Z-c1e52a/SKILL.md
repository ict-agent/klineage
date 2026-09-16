---
skill_id: sparse-mla.tma-descriptor-prefetch
intent: Prefetch tensor-map descriptors to hide their first-use latency.
preconditions:
- 'Data types: no additional operand representation or arithmetic requirement; descriptor
  prefetch does not read or convert tensor elements.'
- 'Layout: the prefetch address must identify the descriptor later used by the tensor
  transfer; a different address cannot warm that descriptor. No added tensor contiguity,
  alignment, or thread-to-element mapping requirement.'
- 'Storage: an existing encoded tensor map must be accessible in parameter or constant
  memory; these are the descriptor spaces supported by this instruction.'
- 'Pipeline: descriptor construction and publication must precede prefetch, and the
  descriptor must remain valid through its users. An earlier issue point must precede
  tensor-map demand to hide latency. Prefetch adds no collective participation, data-visibility
  guarantee, or buffer-reuse completion; existing transfer completion and reader-before-reuse
  ordering remain necessary.'
- 'Hardware: SM90 or newer with assembler support for prefetch.tensormap; older targets
  cannot issue it. No additional shared-memory capacity is needed because this hint
  allocates no software buffer.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Prefetch the query tensor-map descriptor before its first TMA use, overlapping
descriptor access with barrier setup. NVIDIA documents the descriptor spaces and
target support in the [PTX prefetch reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu).

In `solution/hopper.cuh`, insert the `prefetch` helper immediately before
`load_q`, inside `namespace hopper`. In `solution/attention.cuh::attention`,
call it with `&maps.query` immediately before `sm.query.init(1)`, inside the
existing `if (warp == 0 && elect())` block. This elects one issuer in CTA warp
zero. Keep `__grid_constant__ const Maps maps`: the helper must address the
launch descriptor rather than a thread-local copy.

Keep the host ABI, caller stream, map encoding, grid, threads, tensor layouts,
bounds handling, and all copy/compute instructions unchanged. Preserve volatile
inline assembly and its memory clobber. No new synchronization or compiler flags
are needed. Barrier initialization, `init_fence()`, and `__syncthreads()` still
precede query loads. Both consumer groups still wait on `sm.query` before reading
Q. Prefetch supplies no transfer-completion signal. Q remains resident until CTA
completion; retain all existing KV/probability completion and reuse barriers.

Preserve the complete problem and arithmetic, including reduction order,
softmax scale, BF16 round-to-nearest conversions, masks, and output statistics.
Check correctness and latency with `klineage.harness.evaluate`. Inspect generated
PTX/SASS for `prefetch.tensormap`/`UTMACCTL.PF` before the query loads; retain
`UTMALDG.3D`, the query eviction-policy operand, and host map promotion encoding.

## Example configuration

The supplied sparse MLA workload uses 8192 tokens, 128 heads, QK width 576,
value width 512, and 2048 selected indices. Q/KV/output are BF16; accumulators,
maxima, and log-sum-exp use FP32; indices are INT32. Preserve scale
`0.1352337788608801f` and the existing intermediate BF16 rounding.

Keep the 16384-CTA grid, 384 threads per CTA, and single-CTA clusters. Each CTA
owns one token and 64 heads; two 128-thread consumer groups share Q and one
128-thread producer group gathers KV. Dynamic shared storage remains 231376 bytes,
including the resident Q tile, two KV buffers, probability storage, and barriers.

The query map has extents `(576,128,8192)`, byte strides `(1152,147456)`,
box `(64,64,1)`, unit element strides, no interleave, 128-byte swizzle,
`CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and the existing OOB setting. Group zero's
elected lane issues nine query TMA tiles, with 73728 expected transaction bytes.
Retain `kEvictFirst = 0x12f0000000000000ULL` on query TMA, KV's 16-byte
`cp.async` transfers with evict-last and 256-byte prefetch hints, split transfer
barriers, QK/PV overlap, WGMMA, shared reductions, and direct global output stores.
Keep all inherited build flags, including SM90a, fast math, and register-usage
settings. These are replay settings, not descriptor-prefetch prerequisites.

# Precondition

- Data types: no additional tensor dtype or numerical constraint. The instruction
  accesses a descriptor, so it introduces no element conversion or arithmetic.
- Layout: prefetch must address the same descriptor that the later transfer uses;
  otherwise it warms the wrong location. No additional operand contiguity,
  alignment, or element ownership is required because tensor indexing is unchanged.
- Storage: an encoded tensor map already resides in accessible parameter or
  constant memory. This instruction targets those descriptor spaces; the recipe
  does not relocate tensor data or create another descriptor.
- Pipeline: the descriptor producer must finish and publish it before prefetch;
  its storage must remain valid through subsequent users. An issue point before
  demand provides the opportunity to hide access latency. No new collective
  participation is required. Prefetch does not publish tensor data or establish
  completion: consumers must retain transfer-completion waits, and buffers must
  remain protected until readers finish. It adds no buffer or stage-count condition.
- Hardware: SM90 or newer and an assembler accepting `prefetch.tensormap` are
  required to emit the instruction. No extra shared-memory capacity is required;
  the hint uses hardware caching without allocating a software buffer.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

`solution/hopper.cuh` has `load_q` but no descriptor-prefetch helper.
The insertion point in `solution/attention.cuh::attention` is:

```cuda
const int lane = threadIdx.x % kWarpgroup;
if (warp == 0 && elect()) {
    sm.query.init(1);
```

The remainder of this existing block and function stays unchanged.

## After

Insert before `load_q` in `solution/hopper.cuh`:

```cuda
__device__ __forceinline__ void prefetch(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}
```

Replace the preceding `attention` fragment with:

```cuda
const int lane = threadIdx.x % kWarpgroup;
if (warp == 0 && elect()) {
    prefetch(&maps.query);
    sm.query.init(1);
```
