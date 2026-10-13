---
skill_id: cuda.cp-async-kv-l2-evict-last
intent: Favor retention of reused KV cache lines with an L2 evict-last copy policy.
preconditions:
- 'Data types: no additional constraint; an eviction hint changes neither copied bits
  nor arithmetic.'
- 'Layout: no additional constraint beyond legal existing cp.async transfers; adding
  a policy leaves addresses, alignment, sizes, and thread ownership unchanged.'
- 'Storage: existing global-to-shared cp.async accesses use L2; source reuse under
  cache contention motivates higher retention priority. No fixed working-set fit is
  required because the policy is advisory.'
- 'Pipeline: preserve source readiness, copy-completion publication before consumers
  read, and consumer completion before shared-buffer reuse; cache hints provide none
  of these ordering guarantees.'
- 'Hardware: PTX 7.4 or newer and sm_80 or newer support createpolicy and cp.async
  L2 cache hints; no additional shared-memory allocation or fixed cache-capacity requirement.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Give KV global-to-shared copies an L2 evict-last policy, favoring retention of
reused KV lines. Apply the policy to every KV copy, including both buffers and
both column partitions. This is an advisory eviction preference, not reserved
cache storage or a guaranteed speedup.

In `solution/hopper.cuh`, add `evict_last()` immediately before `copy_kv()` and
replace `copy_kv()` with the After version. The helper creates a fractional
policy with fraction `1.0`; the copy consumes its 64-bit operand with
`.L2::cache_hint`. Retain `.cg`, `.L2::256B`, transfer size, and zero filling.
The separate 256-byte prefetch hint is not part of this transformation.

In `solution/attention.cuh`:

1. Append `uint64_t policy` to `copy_tiles()` after its `valid` parameter.
2. Append `policy` to its `copy_kv()` call, after `valid[Buffer][row] ? 16 : 0`.
3. In `produce()`, insert `const uint64_t policy = evict_last();` immediately
   before `int phase = 1;`, outside the block loop.
4. Append `policy` after `valid` at all four `copy_tiles` calls, retaining their
   order: `<0, 0, kHalfTiles>`, `<1, kHalfTiles, kKeyTiles>`,
   `<0, kHalfTiles, kKeyTiles>`, `<1, 0, kHalfTiles>`.

Keep the host ABI, caller stream, launch, buffer allocation, and all addresses
unchanged. The producer owns the policy value and passes it through the copy
helpers; consumers need no policy state. Retain every `free*.wait`,
`ready*.cp_arrive`, consumer `ready*.wait`, WGMMA completion wait, and `free*.arrive`.
The cache hint changes neither producer completion nor consumer visibility nor
when either buffer can be overwritten. Keep mask publication and phase toggles.

Bounds behavior is unchanged: a KV index outside `[0, kTokens)` supplies source
size zero to the existing 16-byte copy, and the existing score mask excludes it.
Preserve input representations, FP32 accumulations and reductions, BF16 rounding,
softmax scale, and output semantics. Do not change compiler flags.

The [NVIDIA PTX specification](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)
defines cache hints independently of memory consistency. Inspect compiled KV
copies for the restored eviction preference; preserve the input problem when
checking correctness and latency through the registered evaluator.

## Example configuration

This replay uses 8192 tokens, 128 heads, query/key width 576, value width 512,
and 2048 selected indices per token. Q/KV and output are BF16; maxima and LSE
are FP32. The softmax scale is `0.1352337788608801f`. Keep the existing WGMMA
accumulation order, intermediate BF16 round-to-nearest conversion, and reduction
order.

Each of 16384 CTAs owns one token and a 64-head slice, with a one-CTA cluster
and 384 threads: two 128-thread consumers and one 128-thread producer.
The producer partitions its lanes into groups of eight. A lane copies eight
BF16 elements (16 bytes) per instruction from four selected rows per buffer;
the row stride is 576 BF16 elements. Preserve `row_pointer()` and its compiler
barrier, all unrolling, and existing register settings.

Keep 64-by-64 tiles, nine Q/K tiles, two KV buffers, four transfer groups,
32 selected-key blocks traversed two at a time, and 231376 shared-memory bytes.
Retain Q/KV 128-byte swizzles and descriptors, unswizzled probability cores,
resident TMA-loaded Q, Q's 128-byte L2 promotion, `.L2::256B` KV prefetch,
probability storage reuse, WGMMA, online softmax, shared-memory peer reductions,
and direct output stores. No launch, layout, allocation, or synchronization
changes accompany the policy.

# Precondition

- Data types: no additional constraint. Eviction priority affects which cache
  lines tend to remain resident, not the representation copied or the arithmetic.
- Layout: no additional constraint beyond already legal `cp.async` transfers.
  The transformation changes no pointer, size, alignment, or thread ownership;
  it introduces no new contiguity or lane-mapping rule.
- Storage: existing `cp.async` accesses copy global sources through L2 into
  shared memory. The L2 hint applies to that global access. Reuse under cache
  contention supplies the reason to favor retention; without reuse, retention
  has no reuse benefit. The advisory policy does not require the entire source
  or working set to fit L2.
- Pipeline: sources must already be ready, copies must finish and publish their
  results before consumers read, and consumers must finish before the producer
  reuses shared storage. Retain the existing mechanisms enforcing these rules;
  eviction hints cannot replace completion or visibility synchronization.
- Hardware: `createpolicy` and `cp.async` L2 cache hints require PTX 7.4 or newer
  and `sm_80` or newer. The change adds no shared buffer and imposes no fixed
  cache-capacity requirement; unsupported instruction support prevents assembly.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These snippets replace `copy_kv()` in `solution/hopper.cuh`; After also inserts
`evict_last()` immediately before it. Apply the four argument-plumbing edits
listed in Overview to `solution/attention.cuh`. Keep the existing `Bf16` alias,
`shared_addr()`, copy loops, and all intervening barriers.

## Before

```cuda
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;"
                 :: "r"(shared_addr(dst)), "l"(src), "r"(bytes));
}
```

## After

```cuda
__device__ __forceinline__ uint64_t evict_last() {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes, uint64_t policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2, %3;"
                 :: "r"(shared_addr(dst)), "l"(src), "r"(bytes), "l"(policy));
}
```
