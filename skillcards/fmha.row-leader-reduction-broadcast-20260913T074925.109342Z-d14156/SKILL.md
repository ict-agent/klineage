---
skill_id: fmha.row-leader-reduction-broadcast
intent: Compute each row reduction once in its leader lane and broadcast the result.
preconditions:
- 'Data types: all consumers must require the same reduction representation, operand
  grouping, and exceptional-value behavior; a shared result cannot replace distinct
  numerical results.'
- 'Layout: each row group must have known lane membership and a unique leader within
  one warp, with addressable partial slots and the same result required by every member;
  otherwise gathering or broadcasting selects the wrong row.'
- 'Storage: per-invocation partial slots must be writable and visible to their row
  group, with a leader slot reusable for the result; leader publication cannot work
  through private or conflicting storage.'
- 'Pipeline: all lanes named by the warp mask must reach the reduction and permit
  synchronization between production, consumption, and reuse; producers must complete
  and publish before dependent reads, and all readers must finish before overwrites
  to prevent incomplete or stale results.'
- 'Hardware: warp synchronization must order accesses to the scratch memory space;
  existing scratch must hold at least one partial per participating lane, including
  each leader result slot.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

In `solution/ops.cuh`, `reduce<Reduce Op>` currently repeats the same row
reduction in every row lane. Restore computation sharing: only the first lane
of each row group gathers the partials, reduces them, and writes its result to
its own slot. After a warp barrier, every row lane reads that slot.

Replace only the function body after `volatile float* slots = s.reduction;`
with After below. Keep `tid`, `leader = tid - tid % kLanesPerRow`, the template,
and the signature. The snippets use this existing context:

```cuda
template<Reduce Op>
__device__ __forceinline__ float reduce(float value, Reduction& s) {
    const int tid = threadIdx.x;
    const int leader = tid - tid % kLanesPerRow;
    volatile float* slots = s.reduction;
    // Replace the remaining body with the selected snippet.
}
```

Each CTA owns one `Reduction` in global scratch. Each thread publishes one
partial at `slots[tid]`; contiguous row groups address their leader's slot.
Keep volatile accesses. They preserve publication and reads through compilation.
No allocation, layout, launch, caller-stream, or bounds changes are needed.
The existing CTA-uniform invalid-tile return stays before any reduction; all
threads of valid CTAs participate, including lanes for padded query rows.

Keep the first warp barrier after partial publication. Add a barrier after
leader computation and before result reads. Keep the final barrier after all
reads and before another invocation can overwrite slots. The leader's write
replaces only a partial it has already consumed; other lanes no longer gather
partials. Probability-tile CTA barriers remain unchanged.

Preserve `(a + c) + (b + d)` and `fmaxf(fmaxf(a, c), fmaxf(b, d))` exactly.
Do not change softmax scaling, accumulation, rounding, or exceptional-value
handling. The forward change eliminates repeated reduction arithmetic and
partial gathers while introducing a leader-result store and broadcast load.

Use the existing evaluator on the unchanged problem. Inspect generated code
for a leader predicate around gathers, arithmetic, and the result store,
followed by an all-lane broadcast load. Retain the source warp barriers;
the compiler may lower them without a distinct barrier instruction.

## Example configuration

Retain packed contiguous FP16 Q/K/V/output `[16384,64,128]`, int32 offsets `[9]`,
and eight sequences of 2048 tokens. Row strides are 8192 half elements
(16384 bytes); this reduction operates on FP32 partials.

Retain `kM=128`, `kN=176`, `kThreads=kMathThreads=256`, `kGroupSize=128`,
`kWarpSize=32`, `kLanesPerRow=4`, and `kFullWarpMask=0xffffffffu`. Each warp
has eight four-lane row groups. Each thread owns two row fragments, 88 score
register entries, and 64 output entries; calls reuse the same partial slot
for each row. The reduction has no additional buffer stage.

Keep `Reduction` aligned to 128 bytes, with 256 floats (1024 bytes) per CTA;
its slots have a four-byte stride. The grid reserves `kMaxQTiles*kHeads=8640`
CTAs and reduction scratch occupies 8,847,360 bytes. Keep the separate
probability buffer: 22,528 half elements per CTA, 389,283,840 bytes reserved.
Launch remains 256 threads per CTA, zero dynamic shared memory, and
`__launch_bounds__(kThreads,1)` on the caller stream.

Retain scalar QK/PV FMA loops, descending key-tile traversal, online softmax,
probability reuse, FP16 rounding through `scalar_half`, separate half stores,
and per-component guarded reciprocal normalization. Keep `kLog2Scale=0.127517432f`
and `kScale=0.0883883461f`, the SM90a build, and compiler flags `-O3`,
`-std=c++17`, `--use_fast_math`, `--resource-usage`, `-lineinfo`, `-DNDEBUG`.
No compiler-control change is required for this transformation.

# Precondition

- Data types: consumers must require the same representation, operand grouping,
  and exceptional-value behavior. Sharing one computed result would otherwise
  replace numerically distinct results. No particular input dtype is required;
  the reduction's representation and arithmetic semantics must be preserved.
- Layout: row groups have known lane membership and a unique leader within one
  warp. Their partial slots are addressable, and every member needs the same
  result. Incorrect membership, addressing, or leader selection mixes rows;
  differing desired results cannot share one broadcast.
- Storage: per-invocation partial slots are writable and visible to the row
  group, with a leader slot reusable for the result. Private storage prevents
  publication; overlapping concurrent invocations can corrupt the broadcast.
- Pipeline: every lane named by the synchronization mask reaches the reduction
  and permits synchronization between production, consumption, and reuse.
  Producers complete and publish before dependent reads; all readers finish
  before overwrites. The existing invocation must allow these ordering points
  so result sharing preserves its producer/consumer contract. Missing one
  exposes incomplete or stale values.
- Hardware: warp synchronization orders accesses to the scratch memory space.
  Without that memory ordering, barriers cannot publish the leader result.
  Existing capacity must cover at least one partial per participating lane,
  including each leader's result slot; the optimization needs no extra buffer.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
    // Publish row partials before every lane gathers them.
    slots[tid] = value;
    __syncwarp(kFullWarpMask);

    // Repeat the same grouped reduction independently in every row lane.
    const float a = slots[leader];
    const float b = slots[leader + 1];
    const float c = slots[leader + 2];
    const float d = slots[leader + 3];
    float result;
    if constexpr (Op == Reduce::Maximum) {
        result = fmaxf(fmaxf(a, c), fmaxf(b, d));
    } else {
        result = (a + c) + (b + d);
    }

    // Finish every gather before another reduction reuses these slots.
    __syncwarp(kFullWarpMask);
    return result;
```

## After

```cuda
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
```
