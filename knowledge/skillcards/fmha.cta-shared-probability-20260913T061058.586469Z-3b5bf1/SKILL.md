---
skill_id: fmha.cta-shared-probability
intent: Place probability intermediates in CTA shared memory for repeated local reads.
preconditions:
- 'Data types: no additional arithmetic type requirement; changing backing storage
  must preserve each stored value''s representation and existing rounding.'
- 'Layout: producer and consumer indices must describe the same bounded tile, with
  race-free element ownership and natural element alignment; otherwise shared addressing
  supplies the wrong values. No additional global contiguity is required.'
- 'Storage: the intermediate currently occupies global scratch and is needed only
  within its producing CTA; shared storage cannot supply external consumers or retain
  values after that CTA finishes.'
- 'Pipeline: every participating CTA thread must reach barriers after tile writes
  and after tile reads, before storage reuse; these establish visibility and prevent
  incomplete reads or overwrites.'
- 'Hardware: CUDA CTA shared memory and CTA barriers must be available, and the aligned
  shared allocation for retained fields plus the tile footprint at its element size
  must fit the device''s configured per-CTA limit.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Place the rounded probability tile in CTA shared memory to replace its global
stores and repeated global reads during probability-times-value accumulation.
Change only the tile's backing storage; retain its K-major panel indices,
writer/reader ownership, rounding, arithmetic, and barriers.

Apply these edits to the existing bundle:

1. In `solution/ops.cuh`, add `__half p[kM * kN]` after `Shared::reduction`.
   Remove `kProbTileElems` and the `Params::prob` pointer.
2. In `solution/attention.cuh`, change `prob_value`'s second parameter from
   `const __half* prob_tile` to `const Shared& s`; replace its indexed probability
   load with `s.p[panel<kM>(row, key)]`. Change `store_prob`'s second parameter
   from `__half* prob_tile` to `Shared& s`; replace its two `prob_tile` stores
   with `s.p` stores at the same indices. Update its comment to name shared storage.
3. In `consumer`, remove the per-CTA `prob_tile` pointer and its comment.
   Pass `s` to `store_prob` and `prob_value`. Retain both `__syncthreads()` calls:
   the first publishes every producer's writes; the second completes all readers
   before the next key tile overwrites P. Retain the separate warp exchange barriers.
4. In `solution/kernel.cu::kernel`, remove the local `torch::empty` allocation,
   its comment, and `p.prob` assignment. Keep the existing `Workspace::ensure`
   attribute and `config.dynamicSmemBytes`, both expressed using `sizeof(Shared)`;
   they automatically request the enlarged allocation. Keep the grid, block,
   caller stream, tensor ABI, and LSE workspace unchanged.

Each CTA owns one Q tile/head and its entire P tile. Each lane writes its rounded
QK fragment through `store_prob`; all output-column owners read P through
`prob_value`. For logical `(row, key)`, both use
`(key / kInputPanelCols) * kM * kInputPanelCols + row * kInputPanelCols
+ key % kInputPanelCols`. No swizzle or new ownership mapping is introduced.

Keep invalid-key masking, zero-filled Q/K/V loads, and guarded output stores.
Every P slot is written before consumption, including masked keys; no shared
initialization or new boundary return is needed. The existing invalid-work return
is uniform across the CTA and precedes all tile accesses.

Preserve increasing-dimension QK `fmaf`, increasing-key PV `fmaf`, descending key
tiles, online maximum/sum updates, scalar round-to-nearest FP16 conversions, and
scalar output stores. Moving already rounded bits changes none of these operations.
Use the existing evaluator with the unchanged problem and compiler flags to check
correctness and latency. Inspect generated probability accesses for shared loads
and stores; leave measurements outside this card.

## Example configuration

This replay uses CUDA on `nvidia-sm90a-cuda13`, packed FP16 Q/K/V/output
`[16384,64,128]`, INT32 offsets `[9]`, and eight sequences. Q/K/V row stride is
`kHeads * kDim = 8192` half elements. P is FP16; accumulators and reductions are
FP32. Preserve `kM=128`, `kN=176`, `kInputPanelCols=8`, and `kDim=128`.

The block has 256 threads, eight 32-thread warps, and two 128-thread math groups.
A lane owns two Q rows and two columns per panel, with 88 QK, 64 PV, and 44 packed
probability registers in the source arrays. These are retained fragment settings,
not prerequisites of shared placement. Preserve `__launch_bounds__(kThreads, 1)`.
There are no MMA instructions or asynchronous operand-copy stages.

The grid has `kMaxQTiles * kHeads = 8640` CTAs, where `kMaxQTiles=135` covers
ragged Q lengths. The preceding allocation reserves 22528 half elements per CTA,
389283840 bytes total, independently per invocation on the caller stream. Its
storage is released through the stream-aware allocator; forward replay removes it.
`Shared` grows from 1024 to 46080 bytes, retaining its 128-byte structure alignment
and the kernel's 1024-byte dynamic-storage alignment. P contributes 45056 bytes;
the 256-float exchange array remains separate. One P tile is reused serially.

Keep `-O3`, `-std=c++17`, `--use_fast_math`, `--resource-usage`, `-lineinfo`, and
`-DNDEBUG`. No compiler-control change is needed for this memory-space change.
Retain the workload, oracle, numerical tolerances, seed, and timing policy.

# Precondition

- Data types: no additional arithmetic type requirement. This optimization moves
  stored values, so it must preserve their representation and existing rounding;
  it does not require a particular floating-point format or change arithmetic.
- Layout: producer and consumer indices must address the same bounded tile with
  race-free element ownership and natural element alignment. Shared indexing
  otherwise selects wrong values or creates invalid accesses. No additional
  global contiguity is required; adapt the addressing to the known tile footprint.
- Storage: the intermediate is in global scratch and has no consumers outside
  its producing CTA or after that CTA finishes. CTA shared visibility and lifetime
  cannot replace global storage for such consumers.
- Pipeline: every participating CTA thread reaches the post-write and post-read
  barriers before reuse. The first establishes producer completion and reader
  visibility; the second prevents overwriting data still being consumed. The
  number of tiles or pipeline stages is not an additional requirement.
- Hardware: the target supplies CUDA CTA shared memory and CTA barriers. The
  aligned allocation of retained shared fields plus the indexed tile footprint
  times its element size must fit the configured per-CTA shared-memory limit;
  otherwise the launch cannot provide the required storage.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The following are paired fragments in the named existing scopes, not a standalone
translation unit. All unshown code remains unchanged.

## Before

```cuda
// solution/ops.cuh: constant, Shared, and member of Params.
constexpr int kProbTileElems = kM * kN;
struct alignas(128) Shared {
    float reduction[kMathThreads];
};
// Params member:
__half* prob;

// solution/kernel.cu::kernel: allocation and Params assignment.
const auto prob = torch::empty(
    {int64_t(kMaxQTiles) * kHeads * kProbTileElems}, q.options());
p.prob = reinterpret_cast<__half*>(prob.data_ptr<at::Half>());

// solution/attention.cuh::store_prob: second parameter is __half* prob_tile.
prob_tile[index] = __ushort_as_half(uint16_t(prob[i]));
prob_tile[index + 1] = __ushort_as_half(uint16_t(prob[i] >> kHalfBits));

// prob_value: second parameter is const __half* prob_tile.
out[i] = fmaf(__half2float(prob_tile[panel<kM>(row, key)]), value, out[i]);

// consumer: pointer is initialized after the uniform invalid-work return.
__half* const prob_tile = p.prob + size_t(blockIdx.x) * kProbTileElems;
// Inside the existing key-tile loop, after convert(score, prob):
store_prob(prob, prob_tile);
__syncthreads();
prob_value(out, prob_tile, p.v + base, valid, wg);
__syncthreads();
```

## After

```cuda
// solution/ops.cuh: remove kProbTileElems and Params::prob.
struct alignas(128) Shared {
    float reduction[kMathThreads];
    __half p[kM * kN];
};

// solution/kernel.cu::kernel: remove the prob allocation and p.prob assignment.
// Existing sizeof(Shared) expressions configure the larger shared allocation.

// solution/attention.cuh::store_prob: second parameter becomes Shared& s.
s.p[index] = __ushort_as_half(uint16_t(prob[i]));
s.p[index + 1] = __ushort_as_half(uint16_t(prob[i] >> kHalfBits));

// prob_value: second parameter becomes const Shared& s.
out[i] = fmaf(__half2float(s.p[panel<kM>(row, key)]), value, out[i]);

// consumer: remove the prob_tile pointer and retain the existing loop/barriers.
store_prob(prob, s);
__syncthreads();
prob_value(out, s, p.v + base, valid, wg);
__syncthreads();
```
