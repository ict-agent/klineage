---
skill_id: topk.resident-bulk-copy-pipeline
intent: Pipeline bulk asynchronous global-to-shared copies ahead of resident-chunk
  computation.
preconditions:
- 'Data types: no additional operand dtype or arithmetic constraint; byte copying
  must preserve the representation consumed by the unchanged computation.'
- 'Layout: each copy covers corresponding contiguous global and CTA-shared intervals
  with known consumer offsets; both addresses must be 16-byte aligned and the in-bounds
  byte count a multiple of 16, as required by the bulk-copy instruction.'
- 'Storage: input bytes already reside in readable global memory and destinations
  in the executing CTA''s shared memory; the selected copy direction cannot target
  another CTA or replace missing resident storage.'
- 'Pipeline: input producers must finish and publish before copying, every consumer
  must observe copy completion before reading, and copies and readers must finish
  before storage overwrite or release; the CTA can coordinate these dependencies without
  divergent barrier participation.'
- 'Hardware: SM90 or newer with an assembler supporting PTX 8.6 CTA-destination bulk
  copies and mbarriers; available CTA shared memory must cover existing resident storage
  plus one 8-byte-aligned, 8-byte completion object per independently outstanding
  copy, including padding, and byte counts must fit the instruction and transaction
  counters.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Pipeline global-to-shared bulk copies ahead of radix histogram computation.
One elected thread issues all resident-chunk transfers; each CTA thread waits
only for the chunk it next consumes. This permits later copies to overlap the
first histogram and avoids routing copied keys through thread registers.

In `solution/native.cuh`, replace the cooperative `native::copy` with the
`init`, `copy`, and `wait_copy` helpers below, inside `namespace native`.
Retain `address`, election, cluster synchronization, remote operations, and scan
helpers. Remove the obsolete decomposition comments in both edited headers.

In `solution/topk.cuh`, restore `constexpr int kStages = 8;` immediately before
`kChunks`, insert `uint64_t barriers[kStages];` between `Shared::scan` and
`Shared::edges`, and change the `sizeof(Shared)` assertion from 8576 to 8640.
The new field is naturally aligned for mbarriers; preceding state offsets stay
unchanged. Replace the region from `reset(sm)` through the first chunk-histogram
loop in `select` with After. No host ABI or launch changes are needed.

## Ownership and ordering

CTA rank `r` owns chunk `c` beginning at
`offsets[c] = head + (r + c*kBlocks)*kChunkItems`, stored at
`keys + c*kChunkItems`. Keep these offsets, counts, and consumer indexing.
Before optimization, thread `t` copies elements `t + j*blockDim.x` and all
threads rendezvous before reading. After optimization, the existing
`leader_thread` issues each complete interval; histogram ownership stays striped.

Threads initialize distinct barriers before the existing CTA barrier publishes
them. Each barrier expects one arrival and the exact transfer byte count.
Every reader executes `wait_copy` before accessing that chunk; successful waits
make completed copy writes visible. The initial `native::arrive()` and all
later cluster barriers still order DSMEM reductions, independently of copying.

Each destination and barrier is used once per invocation. Both resident chunks
remain immutable through later radix passes and `emit`; all copies complete
before readers finish or the CTA releases shared storage. There is no buffer
rotation or parity toggle: every wait uses initial parity zero. Keep input
production ordered before the kernel on the caller stream.

Retain the scalar head/tail peel and its bounds checks. Only the bounded aligned
interior enters bulk copies; never round a transfer beyond `counts[chunk]`.
Keep exact original key bits, `ordered`, histogram arithmetic, split decisions,
and value/index pairing. Output order and choice among tied indices remain
unspecified. This transformation adds no conversion, accumulation, or rounding.

## Example configuration

Preserve contiguous FP32 input `[1,131072]`, K=2048, FP32 values and int64
indices `[1,2048]`; row strides are 131072, 2048, and 2048 elements. The complete
problem and oracle remain unchanged.

Keep `kThreads=512`, `kBlocks=16`, 32-lane warps, `kItems=8`, 11-bit radix
histograms with 2048 buckets, three possible passes, early termination, shared
atomics, warp scans, DSMEM merges, and descending CTA output prefixes. Retain
the overlap of nonleader scans with cluster reductions and the register preload
loops in `histogram` and `emit`.

Keep 16 KiB chunks (4096 FP32 elements), two resident chunks per CTA,
`kAlign=128`, and the 128-byte head/tail peel. The aligned interior sizes are
positive here, including a shortened final chunk for a misaligned input base.
The existing later-pass contiguous read uses `counts[0] + counts[1]`; the first
resident chunk is full, so retain this instance's two-chunk mapping.

Restore eight barrier slots, of which two are used. Static shared storage rises
from 8576 to 8640 bytes; dynamic storage stays 32896 bytes, for 41536 total
bytes per CTA after replay. Keep the 25% carveout request and its 64 KiB Hopper
partition, `__launch_bounds__(512,1)`, grid/cluster `(1,16,1)`, block `(512,1,1)`,
nonportable cluster permission, dynamic-memory attribute, and caller CUDA stream.
Build for `TORCH_CUDA_ARCH_LIST=9.0a` on the supplied CUDA 13 target.

Use the existing Kernel evaluator with the complete problem for correctness and
latency. Keep evidence outside this card. Instruction alignment and target rules
are specified by the [NVIDIA PTX bulk-copy documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk).

# Precondition

- Data types: no additional operand dtype or arithmetic constraint. The move
  copies bytes; their representation must match what unchanged consumers expect.
  FP32 is this instance's computation type, not a requirement for bulk copying.
- Layout: global and CTA-shared intervals are contiguous, with known offsets
  mapping each consumer to its original element. Both addresses must be aligned
  to 16 bytes, and each bounded byte count must be divisible by 16; otherwise the
  bulk-copy instruction has undefined behavior. Unaligned edges need a separate
  path, as already provided here.
- Storage: the preceding implementation reads global memory into allocated
  shared storage in its own CTA. That placement is required by the chosen
  global-to-CTA-shared instruction; remote destinations or unavailable resident
  storage cannot be substituted.
- Pipeline: producer writes are complete and visible before copying. Each
  reader must observe transfer completion before loading shared values. Copies
  and readers finish before any overwrite or release of their storage. The CTA
  must coordinate these dependencies with consistent barrier participation;
  otherwise readers can see incomplete data or synchronization can deadlock.
- Hardware: SM90 or newer and an assembler supporting the PTX 8.6
  CTA-destination bulk-copy form and mbarriers. CTA shared capacity must be at
  least existing resident storage plus `n*sizeof(uint64_t)` and alignment
  padding for `n` independently outstanding copies. Each completion object
  needs 8-byte alignment and occupies 8 bytes. Insufficient capacity prevents
  distinct live completion state; copy sizes must also fit the instruction and
  transaction counters. Fixed launch dimensions and barrier-slot counts are
  example settings.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are replacements inside the named namespaces/functions, not complete
files. Apply the constant, `Shared` member, and size-assertion edits described
above; leave the following edge handling and all later computation intact.

## Before

```cuda
// solution/native.cuh, namespace native
__device__ __forceinline__ void copy(float* dst, const float* src,
    unsigned bytes) {
  // Publish each thread's striped stores before any thread consumes this chunk.
  const unsigned count = bytes / sizeof(float);
  for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
    dst[i] = src[i];
  __syncthreads();
}

// solution/topk.cuh, select: after state initialization
reset(sm);
native::arrive();
__syncthreads();

// Keep resident reuse, but finish each cooperative copy before its histogram.
#pragma unroll 1
for (int chunk = 0; chunk < kChunks; ++chunk) {
  native::copy(keys + chunk * kChunkItems, input + offsets[chunk],
      counts[chunk] * sizeof(float));
  histogram<Pass::first>(sm, keys + chunk * kChunkItems, counts[chunk],
      0, 0, start_bit(0), kBuckets - 1);
}
```

## After

```cuda
// solution/native.cuh, namespace native
__device__ __forceinline__ void init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared.b64 [%0], 1;" : : "r"(address(bar)) : "memory");
}

__device__ __forceinline__ void copy(float* dst, const float* src,
    unsigned bytes, uint64_t* bar) {
  const unsigned b = address(bar);
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      : : "r"(address(dst)), "l"(src), "r"(bytes), "r"(b) : "memory");
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
      : : "r"(b), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wait_copy(uint64_t* bar) {
  const unsigned b = address(bar);
  // Each resident slot is filled once, so every stage starts at parity zero.
  asm volatile("{ .reg .pred p; wait_loop: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; "
      "@!p bra wait_loop; }" : : "r"(b) : "memory");
}

// solution/topk.cuh, select: after state initialization
reset(sm);
native::arrive();
if (tid < kStages) native::init(sm.barriers + tid);
__syncthreads();

// Two live stages of the eight-stage resident-load pipeline; no overflow.
if (leader_thread) {
  #pragma unroll 1
  for (int chunk = 0; chunk < kChunks; ++chunk)
    native::copy(keys + chunk * kChunkItems, input + offsets[chunk],
        counts[chunk] * sizeof(float), sm.barriers + chunk);
}
#pragma unroll 1
for (int chunk = 0; chunk < kChunks; ++chunk) {
  native::wait_copy(sm.barriers + chunk);
  histogram<Pass::first>(sm, keys + chunk * kChunkItems, counts[chunk],
      0, 0, start_bit(0), kBuckets - 1);
}
```
