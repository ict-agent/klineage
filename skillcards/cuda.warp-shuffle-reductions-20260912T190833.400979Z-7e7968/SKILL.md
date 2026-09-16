---
skill_id: cuda.warp-shuffle-reductions
intent: Exchange reduction partials through warp shuffles instead of shared memory.
preconditions:
- 'Data types: partials must transfer losslessly through a supported shuffle type
  or register-word decomposition; preserve the reduction tree and arithmetic operations
  to avoid changing rounding.'
- 'Layout: each reduction peer must occupy a known lane in the same warp, with XOR
  partners inside the participating group; shuffles cannot read another warp.'
- 'Storage: partials are already produced in thread registers, and the shared scratch
  serves only their peer exchange; removing scratch must not discard state needed
  by another consumer.'
- 'Pipeline: every named lane must execute each exchange with the same participation
  mask and a ready source value; existing scratch writes must be visible before reads
  and reads complete before reuse; remove scratch barriers only when they order that
  exchange alone, since shuffles do not fence other memory.'
- 'Hardware: the target and compiler must support synchronized warp shuffles for the
  chosen representation, with registers for each live partial and received peer; otherwise
  the register exchange cannot execute.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace the shared-memory peer exchanges in `solution/attention.cuh::softmax`
and `reduce_sum` with `__shfl_xor_sync`. Each lane receives the same partner's
partial directly in a register. Preserve every `fmaxf`, addition, operand order,
and XOR step; this changes communication only.

Apply these edits to the supplied bundle:

1. In `softmax`, replace both `reduce_peer(sm.reduction, cur, mask)` calls with
   `__shfl_xor_sync(0xffffffff, cur, mask)`, retaining masks 1 then 2.
2. In `reduce_sum`, make the same replacement for the four calls on `l[0]`
   and `l[1]`, retaining each row's masks 1 then 2 and row order.
3. Remove `hopper::reduce_peer` from `solution/hopper.cuh`. Its volatile scratch
   accesses force the deoptimized shared-memory path; remove both internal
   `__syncwarp` calls with that helper. Keep all other synchronization.
4. Remove `Shared::reduction` and `kConsumerThreads`; restore
   `constexpr int kSharedBytes = 230352;`. The trailing scratch removal leaves
   every earlier shared member's offset unchanged.

`solution/kernel.cu` already uses `sizeof(Shared)` for both shared-memory opt-in
and launch allocation, so it needs no edit. Preserve the ABI, caller device and
stream, launch mapping, input checks, padding mask, and arithmetic conversions.
The source indices `threadIdx.x ^ mask` and warp-local `lane ^ mask` select the
same thread because these masks never change the warp-number bits.

The removed scratch barriers publish each exchange and finish its reads before
reuse. Once the exchange is register-only, no scratch remains to publish or
reuse. Keep `softmax`'s existing `__syncwarp` before the maximum store, the
maximum/probability/sum named barriers, async-copy readiness/free barriers,
proxy fences, and WGMMA fences/commit/wait ordering. Those protect other data.

Rebuild from the edited complete bundle with the preserved compiler flags.
Check the supplied problem with the registered evaluator. Inspect generated
code: the reduction sites should use shuffle butterfly instructions instead of
the helper's shared stores and loads. The existing index
broadcast shuffles in `warp_index` and `group_index` remain unchanged.

## Example configuration

This sparse MLA prefill uses 8192 tokens, 128 heads, QK width 576, value width
512, and 2048 selected indices. Q/KV and output are BF16; score, maximum, sum,
and output accumulators are FP32. Preserve the softmax scale
`0.1352337788608801f`, its log2 conversion, BF16 round-to-nearest conversions,
masked `-INFINITY`, and all reduction grouping.

The launch has 16384 CTAs of 384 threads and a one-CTA cluster. Each CTA owns
one token and 64 heads. Two 128-thread consumer groups own threads 0–255;
the remaining group produces KV tiles. Each consumer lane holds two row
partials. Four adjacent lanes reduce one row using XOR masks 1 then 2,
with participation mask `0xffffffff` and default warp width 32.
`row_index(row, lane)` remains
`(lane / 32) * 16 + row * 8 + (lane % 32) / 4`.

Before optimization, `Shared::reduction` holds 256 FP32 slots indexed by CTA
thread number. Its 1024 bytes raise `sizeof(Shared)` from 230352 to 231376.
Each warp uses disjoint slots; rows and XOR steps reuse its slots sequentially.
After optimization, those slots and their helper are absent. No launch count
or ownership changes are needed, and no new tail handling is introduced:
all consumer warps execute these reductions fully, including masked scores.

Retain resident Q, two KV buffers, four producer transfer groups, 64-by-64
operand tiles, 128-byte shared swizzles, TMA Q loads, vector `cp.async` KV
loads and cache policy, online softmax, WGMMA QK/PV, register-fed local PV,
shared peer probabilities, and direct global output stores. Preserve the
32 score registers and 128 output registers per consumer thread, existing
launch bounds, and the supplied SM90a compilation settings.

# Precondition

- Data types: exchange the exact partial representation, using a supported
  shuffle type or a lossless register-word decomposition. Preserve arithmetic
  operations and tree order because changing floating-point grouping can change
  rounding. BF16 inputs are an example setting, not a shuffle requirement.
- Layout: know each partial's owner and keep every requested XOR partner in
  the same warp and participating group. A shuffle cannot fetch a different
  warp's register. Global contiguity and alignment add no requirement because
  this change acts only on already-produced partials.
- Storage: the partials already exist in registers; shared scratch is solely
  their communication mailbox. Scratch with other consumers or persistent state
  cannot be deleted by this recipe. No new shared-memory capacity is required.
- Pipeline: source values must be ready, and all lanes named by the mask must
  execute each exchange with that mask. The preceding scratch writes must be
  visible before peer reads, and reads must finish before scratch reuse.
  Removing the scratch removes those hazards only; its barriers must have no
  unrelated ordering duty. Preserve ordering for all other memory because a
  shuffle supplies no memory fence.
- Hardware: synchronized shuffle support must cover the chosen representation,
  and registers must hold the live partial and received peer. Without these,
  the replacement register exchange is unavailable. Tensor-core instructions
  and input buffering are retained example features, not prerequisites here.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

The declarations and function excerpts below come from the named locations;
intervening code is retained.

```cuda
// solution/attention.cuh: shared storage declarations.
constexpr int kConsumerThreads = 2 * kWarpgroup;
constexpr int kSharedBytes = 230352 + kConsumerThreads * sizeof(float);
// Last member of Shared:
float reduction[kConsumerThreads];

// solution/hopper.cuh: shared exchange helper.
__device__ __forceinline__ float reduce_peer(
    volatile float* scratch, float value, int lane_mask) {
    constexpr unsigned int kActiveWarp = 0xffffffff;
    const int thread = threadIdx.x;

    scratch[thread] = value;
    __syncwarp(kActiveWarp);
    const float peer = scratch[thread ^ lane_mask];
    __syncwarp(kActiveWarp);
    return peer;
}

// solution/attention.cuh: softmax, inside each row's reduction.
cur = fmaxf(cur, reduce_peer(sm.reduction, cur, 1));
cur = fmaxf(cur, reduce_peer(sm.reduction, cur, 2));

// solution/attention.cuh: reduce_sum, before the existing shared sum store.
l[0] += reduce_peer(sm.reduction, l[0], 1);
l[0] += reduce_peer(sm.reduction, l[0], 2);
l[1] += reduce_peer(sm.reduction, l[1], 1);
l[1] += reduce_peer(sm.reduction, l[1], 2);
```

## After

Delete the helper, `kConsumerThreads`, and `Shared::reduction`; retain all
surrounding code and synchronization.

```cuda
// solution/attention.cuh: shared storage size without reduction scratch.
constexpr int kSharedBytes = 230352;

// softmax: preserve the same maximum tree.
cur = fmaxf(cur, __shfl_xor_sync(0xffffffff, cur, 1));
cur = fmaxf(cur, __shfl_xor_sync(0xffffffff, cur, 2));

// reduce_sum: preserve the same addition tree and row order.
l[0] += __shfl_xor_sync(0xffffffff, l[0], 1);
l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
l[1] += __shfl_xor_sync(0xffffffff, l[1], 1);
l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);
```
