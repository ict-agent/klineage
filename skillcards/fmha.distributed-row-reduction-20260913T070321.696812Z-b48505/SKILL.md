---
skill_id: fmha.distributed-row-reduction
intent: Distribute softmax row reductions across participating lanes with XOR exchanges.
preconditions:
- 'Data types: the scalar combine operations must be commutative under the required
  numerical equivalence; XOR peers reverse operands, so retain the prescribed reduction
  grouping, rounding, and exceptional-value behavior.'
- 'Layout: each power-of-two, group-aligned set of contiguous lanes lies within one
  warp and contributes one partial for the same row; XOR partners must remain in that
  group, with distinct naturally aligned scratch slots.'
- 'Storage: CTA-private global scratch already provides one scalar slot per participating
  thread, readable by every group peer; independent CTAs and concurrent invocations
  must not alias these slots.'
- 'Pipeline: every lane named by each warp barrier reaches reductions in the same
  order; publish partials before peer reads, finish reads before slot reuse, and keep
  scratch alive until the caller-stream kernel completes.'
- 'Hardware: CUDA scalar global loads/stores and warp synchronization with memory
  ordering; no additional capacity is needed beyond the existing participating_threads
  * sizeof(partial) scratch and structure-alignment padding.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Distribute softmax row reductions across their participating lanes. Replace the
leader's gather/combine/broadcast with two XOR exchange rounds. Each lane combines
its own partial with a peer, then combines the two-lane results. Each lane performs
two combines and obtains the final result directly, eliminating the leader broadcast.

In `solution/ops.cuh`, replace `Reduce` and the complete `reduce` template with
`Peer` and `exchange` from After. In `solution/attention.cuh`, replace the
`reduce<Reduce::Maximum>` call in `softmax` and the `reduce<Reduce::Sum>` call in
`consumer` with the corresponding After statements. Apply these replacements to
both first-tile and subsequent-tile softmax through the existing template.

Each thread still publishes to `s.reduction[threadIdx.x]`. Exchange first with
`threadIdx.x ^ 2`, then `threadIdx.x ^ 1`; the aligned four-lane group owns one
query row for each row iteration. Keep the scratch global, CTA-private, volatile,
and invocation-local. Each exchange has a warp barrier after publication and
another after reading, before the next exchange or row reuses the slots. The
leader-only implementation's three barriers become two barriers per exchange.
Keep the separate CTA barriers around probability-tile consumption and reuse.

Keep the launch, tensor ABI, current device, caller stream, allocations, and
ownership unchanged. Retain uniform returns for excess CTAs, invalid-key
`-INFINITY` masking, and guarded query output stores. Padded query lanes continue
to participate in reductions. No shared-memory allocation or shuffle is added.

For row partials `a,b,c,d`, preserve `(a+c)+(b+d)` for the sum and the matching
max tree. XOR partners may reverse operands at a node; this retains the numerical
result for the existing nonnegative finite sums and max values. Keep local sum
order, descending key-tile traversal, increasing-dimension QK FMAs, increasing-key
PV FMAs, FP32 accumulation, and round-to-nearest FP16 conversion unchanged. Do
not replace the sum by a left fold or change scale/exp/log arithmetic.

Keep the existing compiler flags and scalar conversion/store controls. Inspect
generated code: reduction exchanges and combines must run in every participating
lane; leader-predicated gathers and the separate result broadcast should disappear.
Use the existing evaluator with the unchanged problem for numerical and timing
checks; retain evidence outside this card.

## Example configuration

The supplied FMHA workload uses eight 2048-token sequences, 16384 packed tokens,
64 heads, and head dimension 128. Q/K/V/output are contiguous FP16 NHD tensors;
the token stride is `kRow = kHeads * kDim = 8192` half elements. Offsets are int32.

Preserve `kM=128`, `kN=176`, `kThreads=kMathThreads=256`, `kWarpSize=32`, and
`kGroupSize=128`. Eight warps process one query tile; each thread owns two query
rows. Each aligned four-lane group covers one row per iteration, with 44 scores
and 32 output components per lane per row (`kQkRegs=88`, `kPvRegs=64`,
`kProbRegs=44`). Preserve all fragment/panel index formulas.

Keep `Reduction` aligned to 128 bytes with 256 FP32 slots: 1024 bytes per CTA.
The unchanged bound `kMaxQTiles=135` launches 8640 CTAs and reserves 8847360
reduction bytes. Dynamic shared memory stays zero; retain
`__launch_bounds__(kThreads, 1)`. Register allocation remains compiler-selected.
The probability tile remains FP16 in global memory, indexed by
`(col / kInputPanelCols) * kM * kInputPanelCols + row * kInputPanelCols +
col % kInputPanelCols`, with `kInputPanelCols=8`. It occupies 22528 elements,
45056 bytes per CTA, and 389283840 allocated bytes over the launch bound.

Keep online softmax, first-tile specialization, final reciprocal normalization,
the scalar QK/PV loops, `scalar_half`'s noinline conversion, scalar output stores,
and offset-derived scheduling. Preserve `kLog2Scale=0.127517432f`,
`kScale=0.0883883461f`, and the full-warp mask `0xffffffffu`. The default sequence
uses 12 descending key tiles, with 112 valid keys in its last tile. Retain the
ragged bounds logic for the full supplied contract. Build for the existing SM90
platform with `-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.

# Precondition

- Data types: XOR exchange reverses operand order between partners. The scalar
  combine must be commutative under the required numerical equivalence, with the
  prescribed grouping, rounding, and exceptional-value behavior preserved.
  No particular operand or output dtype is intrinsic to this scheduling change;
  a different partial representation needs matching operations and slot types.
- Layout: each contiguous, group-aligned, power-of-two lane group must fit within
  one warp and supply partials for the same logical row at each reduction.
  Otherwise XOR addresses select unrelated values or cross the synchronization
  domain. Slots must be distinct and naturally aligned for the scalar access type.
- Storage: the preceding implementation must already expose CTA-private global
  scalar slots to all group peers, one slot per participating thread. Slot aliasing
  across CTAs or concurrent invocations corrupts exchanges; retain disjoint storage.
- Pipeline: all lanes named by each warp barrier must reach reductions in the
  same order. Publication must complete before peer reads, and reads must complete
  before the next write reuses a slot. These dependencies prevent stale and
  overwritten partials. Keep scratch valid through completion on the caller stream.
- Hardware: scalar CUDA global memory accesses and warp synchronization must
  provide the visibility and ordering required by exchanges. No additional capacity
  is needed beyond the existing `participating_threads * sizeof(partial)` scratch
  and structure-alignment padding. No shared-memory or matrix feature is required.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both blocks show the enum/helper in `solution/ops.cuh`, followed by the two
replacement call sites in `solution/attention.cuh`. Existing `Reduction`,
`kLanesPerRow`, and `kFullWarpMask` declarations stay in place.

## Before

```cuda
enum class Reduce { Maximum, Sum };

template<Reduce Op>
__device__ __forceinline__ float reduce(float value, Reduction& s) {
    const int tid = threadIdx.x;
    const int leader = tid - tid % kLanesPerRow;
    volatile float* slots = s.reduction;

    // Publish row partials before their leader gathers them.
    slots[tid] = value;
    __syncwarp(kFullWarpMask);

    // One lane retains the original alternate-then-adjacent grouping.
    if (tid == leader) {
        const float a = slots[leader];
        const float b = slots[leader + 1];
        const float c = slots[leader + 2];
        const float d = slots[leader + 3];
        if constexpr (Op == Reduce::Maximum) {
            slots[leader] = fmaxf(fmaxf(a, c), fmaxf(b, d));
        } else {
            slots[leader] = (a + c) + (b + d);
        }
    }
    __syncwarp(kFullWarpMask);
    const float result = slots[leader];

    // Finish the broadcast before another reduction reuses these slots.
    __syncwarp(kFullWarpMask);
    return result;
}

// softmax: after the lane-local maximum loop.
m = reduce<Reduce::Maximum>(m, s);

// consumer: before final reciprocal normalization.
sum[r] = reduce<Reduce::Sum>(sum[r], s);
```

## After

```cuda
enum class Peer : int { Adjacent = 1, Alternate = 2 };

__device__ __forceinline__ float exchange(float value, Peer peer, Reduction& s) {
    const int tid = threadIdx.x;
    volatile float* slots = s.reduction;

    // Publish each lane's value before its peer reads it.
    slots[tid] = value;
    __syncwarp(kFullWarpMask);
    const float other = slots[tid ^ int(peer)];

    // Complete all reads before the next exchange reuses these slots.
    __syncwarp(kFullWarpMask);
    return other;
}

// softmax: after the lane-local maximum loop.
m = fmaxf(m, exchange(m, Peer::Alternate, s));
m = fmaxf(m, exchange(m, Peer::Adjacent, s));

// consumer: before final reciprocal normalization.
sum[r] += exchange(sum[r], Peer::Alternate, s);
sum[r] += exchange(sum[r], Peer::Adjacent, s);
```
