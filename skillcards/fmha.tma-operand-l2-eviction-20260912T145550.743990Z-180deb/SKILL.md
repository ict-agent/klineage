---
skill_id: fmha.tma-operand-l2-eviction
intent: Prioritize reused operands in L2 with operand-specific TMA eviction policies.
preconditions:
- 'Data types: no additional operand representation or arithmetic requirement; eviction
  priority does not transform copied bits or arithmetic.'
- 'Layout: no additional contiguity, alignment, or thread mapping requirement beyond
  the existing legal tensor copies; the policy leaves addresses and ownership unchanged.'
- 'Storage: existing TMA loads read global operands through L2 into shared memory,
  with distinguishable reuse across loads; otherwise operand-specific retention has
  no reuse to favor.'
- 'Pipeline: no additional synchronization requirement; source and descriptor publication,
  copy completion before reads, and reader completion before buffer reuse must already
  be ordered because cache hints provide none of these guarantees.'
- 'Hardware: TMA support for .L2::cache_hint and valid 64-bit cache policies is required
  to express eviction priority; no added shared-memory capacity or whole-working-set
  L2 fit is required because retention is advisory.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Favor reused K/V cache lines over Q lines by attaching operand-specific L2
eviction policies to existing TMA loads. Each query-tile CTA reloads its head's
K/V sequence; its Q tile has shorter reuse. This motivates evict-last for K/V
and evict-first for Q. The policy is advisory and changes no memory-consistency
guarantees. See [NVIDIA's PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor).

In `solution/ops.cuh`, add the two policy constants in `namespace native`,
parameterize `load` by `uint64_t Hint`, and append `.L2::cache_hint` plus its
64-bit operand to the copy instruction. Use `kEvictLast` in `load_k` and `load_v`.
In `solution/scheduler.cuh::producer`, use `kEvictFirst` for the Q load.
The After block supplies every changed expression and the complete load helper;
retain surrounding caller code.

Producer lane zero still issues every copy. Addresses, tensor maps, shared
panels, consumer ownership, grid, block, shared allocation, ABI, and caller
stream stay unchanged. Keep tensor-map L2 promotion independently configured.
Preserve transaction-byte accounting, full/empty barrier phases, reader
visibility, and completion before stage reuse. Hints add no synchronization.

Retain tensor-map bounds handling, the final KV tile's score mask, and guarded
output stores. Keep all arithmetic, reduction order, FP16 rounding, workload,
oracle, and numerical tolerances unchanged.

## Example configuration

This replay uses CUDA on `nvidia-sm90a-cuda13`, FP16 packed-NHD Q/K/V/output
`[16384,64,128]`, and int32 cumulative offsets `[9]`. Preserve noncausal attention,
no dropout, and scale `1/sqrt(128)`.

Keep `kM=128`, `kN=176`, `kDim=128`, two K/V stages, and eight-column unswizzled
input panels. Global row/head byte strides are `kHeads*kDim*sizeof(__half)`
and `kDim*sizeof(__half)`. A copied element `(row,col)` occupies shared index
`(col/kInputPanelCols)*Rows*kInputPanelCols + row*kInputPanelCols
+ col%kInputPanelCols`. Each tile expects `Rows*kDim*sizeof(__half)` bytes.

Keep `grid=kMaxQTiles*kHeads`, `block=kThreads=384`, and dynamic shared memory
`sizeof(Shared)`. Threads 0–31 run the producer, lane zero copies, threads
128–383 form two math warpgroups, and the other threads exit after initialization.
Each CTA owns one query tile and head. Retain WGMMA QK/PV, register fragments,
online softmax, asynchronous copy/compute overlap, shared-memory peer exchange,
scalar half conversions/stores, and direct CTA scheduling.

Keep `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, existing descriptor encoding,
`__launch_bounds__`, unroll directives, and compile flags
`-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.
The policy constants below are the SM90 encodings identified by
[NVIDIA's cache-hint enumeration](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp).
They require no library include or runtime dependency.

# Precondition

- Data types: no additional representation or arithmetic constraint. The policy
  selects eviction priority without changing payload bits, conversion, or arithmetic.
- Layout: no additional contiguity, alignment, or thread mapping constraint beyond
  already legal tensor copies. Existing addresses and ownership are untouched;
  adding a policy does not impose another layout rule.
- Storage: the preceding implementation loads global operands through L2 into
  shared memory. Operand loads have distinguishable reuse; without reuse to favor,
  selective cache retention offers no reuse benefit.
- Pipeline: no additional synchronization requirement. Sources and descriptors
  must already be published before copies, completed copies visible before reads,
  and readers finished before buffer reuse. Cache hints establish none of these
  orderings, so existing copy synchronization remains necessary.
- Hardware: the target must support TMA's `.L2::cache_hint` operand and valid
  64-bit policy encodings, or it cannot express the selected priorities. No added
  shared-memory capacity or full working-set fit in L2 is required: priorities
  are advisory and do not reserve cache storage.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The final three expressions in each block occur inside existing callers, whose
arguments, conditions, and waits remain in place.

## Before

```cuda
// solution/ops.cuh
template<int Rows>
__device__ __forceinline__ void load(const CUtensorMap& map, __half* dst,
                                    uint64_t* full, int token, int head) {
    expect(full, Rows * kDim * sizeof(__half));
    #pragma unroll
    for (int col = 0; col < kDim; col += kInputPanelCols) {
        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4, 0}], [%5];"
                     :: "r"(shared_addr(dst + col * Rows)), "l"(&map), "r"(col),
                        "r"(token), "r"(head), "r"(shared_addr(full)) : "memory");
    }
}

// load_k and load_v, after their existing empty-stage waits:
load<kN>(p.kmap, s.k + stage * kN * kDim, &s.k_full[stage], token, head);
load<kN>(p.vmap, s.v + stage * kN * kDim, &s.v_full[stage], token, head);

// solution/scheduler.cuh::producer, after QueryEmpty synchronization:
if (lane == 0) load<kM>(p.qmap, s.q, &s.q_full, qstart, work.z);
```

## After

```cuda
// solution/ops.cuh, namespace native
constexpr uint64_t kEvictFirst = 0x12f0000000000000ULL;
constexpr uint64_t kEvictLast = 0x14f0000000000000ULL;

template<int Rows, uint64_t Hint>
__device__ __forceinline__ void load(const CUtensorMap& map, __half* dst,
                                    uint64_t* full, int token, int head) {
    expect(full, Rows * kDim * sizeof(__half));
    #pragma unroll
    for (int col = 0; col < kDim; col += kInputPanelCols) {
        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                     "[%0], [%1, {%2, %3, %4, 0}], [%5], %6;"
                     :: "r"(shared_addr(dst + col * Rows)), "l"(&map), "r"(col),
                        "r"(token), "r"(head), "r"(shared_addr(full)), "l"(Hint) : "memory");
    }
}

// load_k and load_v, after their existing empty-stage waits:
load<kN, kEvictLast>(p.kmap, s.k + stage * kN * kDim, &s.k_full[stage], token, head);
load<kN, kEvictLast>(p.vmap, s.v + stage * kN * kDim, &s.v_full[stage], token, head);

// solution/scheduler.cuh::producer, after QueryEmpty synchronization:
if (lane == 0) load<kM, kEvictFirst>(p.qmap, s.q, &s.q_full, qstart, work.z);
```
