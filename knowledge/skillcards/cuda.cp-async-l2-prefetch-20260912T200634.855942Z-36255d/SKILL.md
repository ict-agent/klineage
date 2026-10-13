---
skill_id: cuda.cp-async-l2-prefetch
intent: Prefetch nearby KV data into L2 during asynchronous copies.
preconditions:
- 'Data types: no additional requirement; the cache hint neither interprets copied
  bits nor changes arithmetic or rounding.'
- 'Layout: no additional alignment or thread-ownership requirement beyond legal existing
  copies; nearby source accesses provide the spatial locality needed for prefetch
  benefit.'
- 'Storage: existing global-to-shared cp.async transfers with readable source bytes
  for active copies; the L2 prefetch qualifier applies to global source addresses.'
- 'Pipeline: input producers finish before copying, consumers observe copy completion,
  and all readers finish before shared slots are reused; the hint supplies no synchronization
  or additional participation requirement.'
- 'Hardware: cp.async L2 prefetch-size support (PTX ISA 7.4 or newer and sm_80 or
  newer); unsupported targets cannot encode the hint. No additional software-managed
  storage is allocated or fixed L2 residency required.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Add `.L2::256B` to the inline `cp.async.cg.shared.global` instruction in
`solution/hopper.cuh::copy_kv`. This requests additional source data in L2 for
nearby KV accesses. It leaves each copy's destination byte count unchanged.
The hint is advisory; its benefit depends on locality and cache contention.
See NVIDIA's [PTX cp.async specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).

Replace only the instruction mnemonic shown below. Keep the operands, `.cg`,
inline-assembly constraints, helper signature, and callers unchanged.
No launch, allocation, layout, ownership, or synchronization changes are needed.
Keep the independent query tensor-map L2 promotion and its transfer unchanged.

`solution/attention.cuh::copy_tiles` calls this helper for each producer-owned
KV segment. Preserve its `valid ? 16 : 0` source-size argument: invalid indices
zero-fill the same shared segment and remain masked out of attention.
Do not enlarge logical loads or require extra array padding for this hint.

Preserve the complete problem and numerical behavior: the same BF16 operands,
FP32 score/output accumulation and reduction order, softmax scale, probability
rounding, final BF16 conversion, FP32 maximum, and FP32 log-sum-exp.
Retain the six-tensor destination-passing ABI, device guard, caller stream,
all checks, and all compile flags.

Rebuild from the final source bundle. Use `klineage.harness.evaluate` for
correctness and latency with the supplied problem and policy. Inspect the
compiled KV-copy instructions to confirm the hint is encoded; timing alone
cannot establish that. Keep measurements outside this card.

## Example configuration

The supplied CUDA SM90 workload has 8192 tokens, 128 heads, QK width 576,
value width 512, and 2048 selected indices per token. Each CTA owns one token
and 64 heads. Retain the 16384-CTA grid, 384 threads, single-CTA cluster,
231376 shared-memory bytes, and `__launch_bounds__(kThreads, 1, 1)`.
Keep `TORCH_CUDA_ARCH_LIST=9.0a` and the saved compiler options, including
`--use_fast_math` and `--register-usage-level=10`.

Producer threads 256 through 383 form sixteen groups of eight. Within each
group, lane `j` copies eight BF16 elements (16 bytes) starting at column `8*j`.
Each group handles four selected rows in each 64-row KV buffer. Preserve
16-byte copy alignment. Global row stride is `params.kv_stride*sizeof(Bf16)`
(1152 bytes here); source element offset is
`index*params.kv_stride + tile*kTile + column`.

Retain both `64*576` KV buffers, resident query storage, the 128-byte Q/K/V
shared swizzle, unswizzled probability cores, and the probability tile that
reuses the first KV buffer's last tile. Preserve `copy_tiles` indexing and
`row_pointer`'s compiler barrier.

Retain the four producer transfer groups, their `free` waits and `ready`
`cp_arrive()` calls, and all consumer completion waits. The producer waits
until consumers release each region before overwriting it. Consumers wait
for the corresponding ready phase before reading. WGMMA completion precedes
the region's free arrival; probability publication fences and named barriers
continue to protect the aliased storage.

Keep the two consumer warpgroups, QK/PV overlap, BF16 WGMMA with FP32
accumulators, 32 score and 128 output accumulator elements per thread,
online softmax, shared peer reductions, and direct global output stores.
These are retained mechanisms, not prerequisites of the cache hint.

# Precondition

- Data types: no additional requirement. Prefetching neither interprets the
  transferred representation nor changes arithmetic or rounding.
- Layout: no additional alignment or thread-ownership requirement beyond
  legal existing copies. Nearby source accesses must provide spatial locality
  for the extra cached data to be useful; otherwise the hint may waste traffic.
  The transformation does not require a new contiguous layout or lane mapping.
- Storage: global-to-shared `cp.async` transfers already exist, with readable
  source bytes for active copies. The qualifier applies to global source
  addresses; it cannot add L2 prefetching to a shared-only exchange.
- Pipeline: input producers finish before copying, consumers observe copy
  completion, and all readers finish before shared slots are reused. The hint
  supplies no synchronization, so these dependencies still prevent stale
  inputs, incomplete reads, and overwrites. It adds no participation rule.
- Hardware: the compiler and GPU support `cp.async` with L2 prefetch size:
  PTX ISA 7.4 or newer and `sm_80` or newer. Otherwise this instruction cannot
  be encoded. No additional software-managed storage is allocated and no
  fixed L2 residency is required; eviction can reduce benefit without
  changing results.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace `copy_kv` in `solution/hopper.cuh`. Its existing `Bf16` alias,
`shared_addr` helper, and all call sites remain unchanged.

## Before

```cuda
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_addr(dst)), "l"(src), "r"(bytes));
}
```

## After

```cuda
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;"
                 :: "r"(shared_addr(dst)), "l"(src), "r"(bytes));
}
```
