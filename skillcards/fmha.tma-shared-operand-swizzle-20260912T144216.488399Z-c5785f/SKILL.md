---
skill_id: fmha.tma-shared-operand-swizzle
intent: Spread shared-memory operand accesses across banks with TMA swizzling.
preconditions:
- 'Data types: no additional arithmetic or rounding restriction; the swizzle moves
  intact 16-byte groups and must preserve their bit representations.'
- 'Layout: known TMA-compatible global strides and contiguous 16-byte groups must
  permit 128-byte panels; all shared readers must admit the matching mapping, including
  descriptor address units and swizzle phase, or they select wrong elements.'
- 'Storage: operands are already staged from global memory into CTA-local shared tiles
  whose producer and consumer layouts can change together; externally fixed shared
  layouts would prevent the permutation.'
- 'Pipeline: producer completion and visibility must precede reads, and all readers
  must complete before reuse; preserve barrier participation and completion accounting
  to avoid incomplete or overwritten tiles.'
- 'Hardware: TMA and shared-operand descriptors must support matching 128-byte swizzling;
  live tiles plus control storage must fit CTA shared memory, since the permutation
  uses that storage without adding capacity.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Apply 128-byte swizzling to Q/K/V shared tiles so their operand accesses spread
across shared-memory banks. Retile the TMA writer and update every WGMMA descriptor
together. This changes addresses, not arithmetic or register-fragment ownership.

In `solution/ops.cuh`, widen `kInputPanelCols` from 8 to 64 half elements and set
the descriptor swizzle field. In `solution/kernel.cu::encode`, select
`CU_TENSOR_MAP_SWIZZLE_128B`. Its box already uses `kInputPanelCols`.
`load` already advances by that constant and writes to `dst + col * Rows`;
retain its instruction, cache hints, and total transaction byte count.
This yields two transfers per tile instead of sixteen.

In `solution/attention.cuh`, replace only the operand setup/traversal in
`query_key` and `prob_value` with the After fragments. Keep their fences,
commits, waits, and arithmetic calls. Q/K are K-major; V is MN-major.
Changing only the tensor map or only the descriptors misaddresses operands.

## Layout and ordering

For tile height `R`, element column `c`, row `r`, and panel width `P`, the linear
half index is `(c / P) * R * P + r * P + c % P`. Before uses `P=8` without
permutation. After uses `P=64`: within each 1024-byte pattern, XOR the
16-byte group index with the low three row bits. TMA performs this permutation;
descriptors perform its matching read mapping. Do not add a software transpose.

All descriptor offsets below use 16-byte units. Before, Q/K use
`leading=R, stride=8`; V uses `leading=8, stride=kN`. After, Q/K use
`leading=1, stride=64`; V uses `leading=kN*8, stride=64`.
The Q warpgroup base and both K traversals must change with panel width.
V advances by 128 descriptor units per reduction step after replay.
These settings follow the [PTX shared-operand layout rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout).

Retain the 1024-aligned dynamic shared base and all tile/stage placements:
each After panel and warpgroup origin has zero swizzle base offset here.
Other origins require encoding their matching swizzle phase.

The producer waits for empty storage before issuing copies. Its full barrier
accounts for all tile bytes; consumers wait for completion and visibility before
reading. Each consumer group finishes its WGMMA reads before its leader signals
K/V empty; the producer waits for both arrivals. Preserve query-lifetime, output-completion, and work handoffs,
barrier phases, and every participant. No launch or synchronization change is needed.

Keep global bounds behavior and sequence masks: TMA handles tensor bounds,
softmax masks the final partial KV tile, and epilogue guards query rows.
Preserve descending KV traversal, WGMMA accumulation order, online-softmax
arithmetic, scalar round-to-nearest half conversion, and scalar output stores.
The linear output `offset` helper is unrelated; leave it unchanged.

After rebuilding the frozen source bundle, check correctness and latency with
`klineage.harness.evaluate` using the complete supplied problem. Require acceptance
and preserve its oracle, tolerances, seed, and timing policy.

## Example configuration

Replay this packed-NHD FMHA instance with `TOKENS=16384`, `HEADS=64`,
`HEAD_DIM=128`, eight sequences, and the supplied cumulative offsets.
Inputs/output are FP16; offsets are int32. Retain FP32 accumulation, noncausal
attention, no dropout, and scale `1/sqrt(128)` with the existing constants.
The global half index is `token*kRow + head*kDim + column`, with
`kRow=kHeads*kDim`; byte strides remain `(16384,256,2)`.

Retain `kM=128`, `kN=176`, `kStages=2`, `kMmaK=16`, and the existing
WGMMA `m64n176k16` QK and `m64n128k16` PV instructions. Keep 88 score,
64 output, and 44 packed-probability registers per consumer thread.
One producer warp and two consumer warpgroups remain in a 384-thread CTA;
threads 32 through 127 remain inactive after initialization. Each consumer
warpgroup owns 64 query rows. Producer lane zero copies tiles; consumers retain
all output-fragment ownership.

Keep the `prepare` launch and caller stream. Attention launches
`kMaxQTiles*kHeads=8640` CTAs with dynamic `sizeof(Shared)` storage.
Shared operand arrays occupy 212992 bytes: one Q tile and two K/V stages;
retain the entire structure, including control and reduction storage.
Retain direct CTA mapping, TMA prefetch/cache policies, staggered K/V production,
consumer overlap, shared reduction exchanges, ABI, build flags, and SM90a target.
These are replay settings, not prerequisites of the permutation.

# Precondition

- Data types: no additional arithmetic or rounding restriction. Moving intact
  16-byte groups preserves stored bits; changing conversions would be a separate
  numerical transformation.
- Layout: known TMA-compatible global strides and contiguous 16-byte groups must
  permit 128-byte panels. Every shared reader must admit the corresponding
  mapping, including descriptor units and swizzle phase; mismatches read another
  logical element. These are address constraints, not fixed tensor dimensions.
- Storage: operands already pass from global memory into CTA-local shared tiles.
  Their producer and consumer layouts must be jointly changeable; an externally
  fixed shared layout would prevent applying the permutation consistently.
- Pipeline: copy completion and reader visibility must precede consumption;
  every reader must finish before overwrite. Preserve barrier participation and
  transaction accounting so no tile appears ready early or is reused while live.
  No particular stage count is required by swizzling.
- Hardware: TMA and shared-operand descriptors must support the same 128-byte
  swizzle, or they cannot produce and consume this layout. Available CTA shared
  memory must cover `sum(live_tile_elements * element_bytes) + control_storage`.
  The permutation adds no storage capacity requirement.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The fragments below replace the named constants, descriptor function, encoder
call, and operand blocks. All other function code stays in place. In `encode`,
retain `shape`, `strides`, `box`, `step`, and result checking. In `load`, retain
the loop using `kInputPanelCols` and its full-tile byte expectation.

## Before

```cuda
// ops.cuh: panel constant and descriptor.
constexpr int kInputPanelCols = 8;

__device__ __forceinline__ uint64_t descriptor(const __half* p, int leading, int stride) {
    constexpr int kAddressShift = 4;
    constexpr int kLeadingShift = 16;
    constexpr int kStrideShift = 32;

    return (uint64_t(stride) << kStrideShift) | (uint64_t(leading) << kLeadingShift)
           | (shared_addr(p) >> kAddressShift);
}

// kernel.cu::encode: shape, strides, box, and step are existing locals.
auto result = encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, const_cast<void*>(ptr),
                        shape, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

// attention.cuh::query_key: operand setup and loop.
uint64_t a = descriptor(s.q + wg * kMmaRows * kInputPanelCols, kM, kCoreRows);
uint64_t b = descriptor(s.k + stage * kN * kDim, kN, kCoreRows);
#pragma unroll
for (int i = 0; i < kDim / kMmaK; ++i) {
    uint64_t ai = a + i * (kMmaK / kInputPanelCols) * kM;
    uint64_t bi = b + i * (kMmaK / kInputPanelCols) * kN;
    qk(score, ai, bi, i == 0 ? Accum::Clear : Accum::Add);
}

// attention.cuh::prob_value: V descriptor and loop.
uint64_t b = descriptor(s.v + stage * kN * kDim, kCoreRows, kN);
#pragma unroll
for (int i = 0; i < kN / kMmaK; ++i) pv(out, prob + i * 4, b + i * kMmaK);
```

## After

```cuda
// ops.cuh: replace the panel constant and add shared layout constants.
constexpr int kDescBytes = 16;
constexpr int kSwizzleBytes = 128;
constexpr int kInputPanelCols = kSwizzleBytes / sizeof(__half);
constexpr int kPanelUnits = kSwizzleBytes / kDescBytes;

__device__ __forceinline__ uint64_t descriptor(const __half* p, int leading, int stride) {
    constexpr int kAddressShift = 4;
    constexpr int kLeadingShift = 16;
    constexpr int kStrideShift = 32;
    constexpr int kSwizzleShift = 62;
    constexpr uint64_t kSwizzleMode = 1;

    // Tile origins retain zero swizzle phase in this replay.
    return (kSwizzleMode << kSwizzleShift) | (uint64_t(stride) << kStrideShift)
           | (uint64_t(leading) << kLeadingShift) | (shared_addr(p) >> kAddressShift);
}

// kernel.cu::encode: box[0] follows the widened kInputPanelCols.
auto result = encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, const_cast<void*>(ptr),
                        shape, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

// attention.cuh::query_key: follow the swizzled K-major panels.
constexpr int kUnusedLeading = 1;
constexpr int kKStepsPerPanel = kInputPanelCols / kMmaK;
constexpr int kMmaUnits = kMmaK * sizeof(__half) / kDescBytes;
uint64_t a = descriptor(s.q + wg * kMmaRows * kInputPanelCols,
                        kUnusedLeading, kCoreRows * kPanelUnits);
uint64_t b = descriptor(s.k + stage * kN * kDim,
                        kUnusedLeading, kCoreRows * kPanelUnits);
#pragma unroll
for (int i = 0; i < kDim / kMmaK; ++i) {
    uint64_t ai = a + (i % kKStepsPerPanel) * kMmaUnits
                 + (i / kKStepsPerPanel) * (kM * kPanelUnits);
    uint64_t bi = b + (i % kKStepsPerPanel) * kMmaUnits
                 + (i / kKStepsPerPanel) * (kN * kPanelUnits);
    qk(score, ai, bi, i == 0 ? Accum::Clear : Accum::Add);
}

// attention.cuh::prob_value: match the MN-major V panel strides.
uint64_t b = descriptor(s.v + stage * kN * kDim,
                        kN * kPanelUnits, kCoreRows * kPanelUnits);
#pragma unroll
for (int i = 0; i < kN / kMmaK; ++i)
    pv(out, prob + i * 4, b + i * kMmaK * kPanelUnits);
```
