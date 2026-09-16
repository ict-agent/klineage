---
skill_id: cuda.tma-query-evict-first
intent: Reduce query cache retention with TMA L2 evict-first hints.
preconditions:
- 'Data types: no additional representation or arithmetic requirement; the eviction
  hint changes cache priority without interpreting or converting transferred values.'
- 'Layout: no additional contiguity, alignment, or thread-ownership requirement beyond
  an already legal tensor transfer; its addresses, descriptor, and destination mapping
  stay unchanged.'
- 'Storage: target global-source TMA traffic that passes through L2 and has little
  future L2 reuse relative to competing data; otherwise lowering its retention priority
  has no cache-reuse rationale.'
- 'Pipeline: no additional ordering or participation requirement; existing producer
  completion, reader visibility, and completion before buffer reuse must remain, because
  an eviction hint provides no synchronization.'
- 'Hardware: TMA hardware and a compiler supporting the bulk tensor L2 cache-hint
  operand and the SM90 eviction-policy encoding; otherwise the instruction or policy
  is unsupported. No additional shared-memory capacity is needed.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Give query TMA loads L2 evict-first priority. Each query tile is loaded once
and retained in CTA shared memory, so later attention iterations reuse shared
data. Lowering these global lines' retention priority can leave L2 space for
reused KV lines. This is a cache hint, not an eviction or speed guarantee.

In `solution/hopper.cuh`, add the namespace constant `kEvictFirst` below
`kWaitTicks`. Replace `load_q` with the After function: append
`.L2::cache_hint`, add the final `%6` operand, and bind the constant with the
64-bit `"l"` constraint. The snippets contain the entire change; no launch,
call-site, tensor-map, allocation, or compiler-flag changes are needed.

The encoding is NVIDIA's SM90 `EVICT_FIRST` value.
[NVIDIA definition](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp).
The hint preserves memory-consistency semantics.
[PTX instruction contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor).

## Example configuration

Preserve these replay settings and mechanisms:

- `q[8192,128,576]` and `kv[8192,1,576]` contain contiguous BF16 values;
  `indices[8192,1,2048]` contains INT32 selected rows. Outputs are BF16
  `output[8192,128,512]` and FP32 `max_logits` and `lse` of shape `[8192,128]`.
- `kernel.cu::kernel` retains destination-passing pybind11 ABI, device checks,
  the caller stream, and tensor-map sizes `(576,128,8192)`, byte strides
  `(1152,147456)`, box `(64,64,1)`, and unit element strides. Keep its 128-byte
  swizzle, 128-byte L2 promotion, and existing out-of-bounds fill setting.
- `attention` launches 16384 CTAs, 384 threads per CTA, a `(1,1,1)` cluster,
  and 231376 dynamic shared-memory bytes. Its launch bounds remain
  `(kThreads,1,1)`. CTA `b` owns token `b/2` and heads `(b%2)*64` through
  `(b%2)*64+63`.
- In `attention.cuh::consume<0>`, one elected thread of warp zero issues nine
  `load_q` calls for columns `tile*64`, writing
  `sm.q_o + tile*4096`. Each swizzled tile contains 64 heads by 64 query
  elements. Both 128-thread consumers read this resident query storage;
  the third warpgroup produces KV tiles. Thread/fragment ownership stays intact.
- Keep `sm.query.init(1)`, barrier initialization fencing and CTA rendezvous,
  `sm.query.expect(73728)`, and both consumers' `sm.query.wait(0)`.
  Input and descriptor production precede the caller-stream launch; transaction
  completion makes the entire query visible before consumption. `sm.q_o` is
  never overwritten during this CTA. Retain all KV/probability completion and
  reuse barriers, the two KV buffers, and the existing QK/PV overlap schedule.
- Keep QK `m64n64k16` and PV `m64n256k16` BF16 WGMMA with FP32 accumulators,
  existing reduction grouping, online softmax, BF16 probability conversion,
  final BF16 round-to-nearest conversion, and scale `0.1352337788608801f`.
  Invalid selected rows still zero-fill KV and mask scores; query tiles are
  in bounds for this fixed workload. Preserve all workload inputs, the FP32
  mathematical oracle, tolerances, seed, and timing policy.
- Keep KV `evict_last()`, its `copy_kv` cache hint and 256-byte fetch hint,
  shared-memory layouts, scalar output stores, and shared reduction exchanges.
  Preserve the existing SM90a build target and every serialized compile flag,
  including fast math, register-usage settings, and source line information.

Use the existing evaluator for correctness and latency on the complete problem.
Inspect generated TMA instructions to confirm they carry the restored policy;
timing alone does not establish that the hint survived compilation. Keep
measurement records outside this card.

# Precondition

- Data types: no additional representation or arithmetic requirement. The hint
  sets cache priority; it neither interprets values nor changes their conversion
  or accumulation. BF16 is this replay's configuration, not a hint prerequisite.
- Layout: no additional contiguity, alignment, or thread-ownership requirement
  beyond an already legal tensor transfer. The existing descriptor, addresses,
  and destination mapping remain valid because the change only adds a policy
  operand. The hint introduces no fragment or shared-layout constraint.
- Storage: the targeted global-source TMA traffic passes through L2. Its lines
  have little future L2 reuse relative to competing data, providing the reason
  to lower retention priority. Otherwise the priority change has no cache-reuse
  rationale. The existing shared destination needs no additional allocation.
- Pipeline: no additional ordering or participation requirement. Preserve
  producer completion and visibility before readers consume transferred data,
  and reader completion before storage is reused. The hint supplies none of
  these guarantees; it leaves existing synchronization responsible for them.
- Hardware: TMA hardware and a compiler must support the bulk tensor
  `.L2::cache_hint` operand and the SM90 eviction-policy encoding used here;
  otherwise the hinted instruction or policy cannot be used. No additional
  shared-memory capacity is needed because neither buffers nor transfers grow.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both functions belong to namespace `hopper` in `solution/hopper.cuh`.
The After constant belongs beside `kWaitTicks`; all other helpers stay intact.

## Before

```cuda
__device__ __forceinline__ void load_q(
    const CUtensorMap* map, Bf16* dst, Mbar& bar, int col, int head, int token) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%3, %4, %5}], [%2];"
                 :: "r"(shared_addr(dst)), "l"(map), "r"(shared_addr(&bar)),
                    "r"(col), "r"(head), "r"(token) : "memory");
}
```

## After

```cuda
constexpr uint64_t kEvictFirst = 0x12f0000000000000ULL;

__device__ __forceinline__ void load_q(
    const CUtensorMap* map, Bf16* dst, Mbar& bar, int col, int head, int token) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%3, %4, %5}], [%2], %6;"
                 :: "r"(shared_addr(dst)), "l"(map), "r"(shared_addr(&bar)),
                    "r"(col), "r"(head), "r"(token), "l"(kEvictFirst) : "memory");
}
```
