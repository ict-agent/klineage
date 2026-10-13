---
skill_id: fmha.output-epilogue-fusion
intent: Fuse the output epilogue into its producer to eliminate intermediate memory
  traffic and a kernel launch.
preconditions:
- 'Data types: no particular dtype is required; deleting the temporary roundtrip must
  preserve the producer representation and the epilogue''s arithmetic and rounding,
  or fusion changes numerical results.'
- 'Layout: each epilogue thread consumes only its matching producer thread''s results,
  with the same tile identity and output ownership; otherwise a direct call cannot
  supply the correct values. No extra contiguity or alignment is required.'
- 'Storage: the intermediate is private scratch with no other readers or visible side
  effects, and its values are available at producer completion; otherwise removing
  materialization loses required data.'
- 'Pipeline: the separate consumer currently follows producer completion and visibility,
  and scratch survives its last read. The epilogue needs only its own thread''s completed
  results, and its stores cannot overwrite unfinished producer inputs; otherwise removing
  the kernel boundary introduces a race.'
- 'Hardware: no additional feature or shared-memory capacity is required; fusion uses
  the existing device instructions and per-thread storage.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse the output epilogue into the attention producer. Keep normalized outputs
and log-sum-exp values in their owning threads, then call the existing
`epilogue` device function directly. This removes one launch and the global
write/read roundtrip through `EpilogueState`.

Apply these edits to the supplied bundle:

1. In `solution/attention.cuh`, replace `consumer`'s materialization after
   `normalize(out, denominator)` with `epilogue(p, work, out, sum)`.
   Delete the entire `attention_epilogue` global function. Keep the existing
   `epilogue`, `normalize`, and `attention` functions unchanged.
2. In `solution/ops.cuh`, delete `EpilogueState` and the `Params::epilogue` field.
   Keep the other fields in their existing relative order.
3. In `solution/kernel.cu`, delete the `epilogue_state` allocation, the assignment
   to `p.epilogue`, and the second launch with its error check. Keep the first
   launch, configuration, device guard, caller stream, and other allocations.

The split version maps scratch as
`state[blockIdx.x * kThreads + threadIdx.x]`, with each entry containing
`out[kPvRegs]` followed by `lse[2]`. Both launches use `tile(blockIdx.x, q_offsets)`.
Fusion passes those same arrays directly, without changing output addresses.
For warp index `w`, lane `l`, and output component `i`, the owning query row is
`work.y*kM + w*kFragmentRows + l/kLanesPerRow
+ ((i % kFragmentSize)/kPairElems)*kCoreRows`; its column is
`(i/kFragmentSize)*kInputPanelCols + (l%kLanesPerRow)*kPairElems + i%kPairElems`.
The existing epilogue retains this mapping and its packed-NHD output indexing.

The split launches are ordered on one caller stream; the temporary stays alive
through the second launch. Fusion replaces that boundary with same-thread
program order. No new barrier is needed: the epilogue reads only local results
and writes disjoint outputs. Keep probability publication/reuse CTA barriers and
reduction publication/reuse warp barriers unchanged. The temporary has no other
consumer, so delete it completely rather than retaining an unused allocation.

Preserve the uniform invalid-CTA return in `consumer`, the epilogue's query-row
bounds checks, and the producer's key-tail mask. Invalid CTAs never access state
in either version. Preserve all scalar FMA ordering, reduction grouping,
normalization, and the existing scalar round-to-nearest FP16 conversions and
16-bit stores. The materialized FP32 values undergo no conversion; fusion must
not introduce an earlier FP16 rounding or recompute the normalization.

## Example configuration

Replay this bundle with FP16 Q/K/V/output, FP32 accumulators and temporary
values, and int32 offsets. The workload has 16,384 tokens, 64 heads, head
dimension 128, and eight sequences. Packed-NHD token stride is
`kRow = kHeads*kDim = 8192` half elements; the head stride is 128 halves.
Output is separately allocated from Q/K/V and offsets.

Keep query/key tiles `kM=128`, `kN=176`, 256 threads, 32-lane warps, two
128-thread groups, and four lanes per row. Each lane owns two rows, 88 scores,
44 packed probability pairs, and 64 output components. Retain the existing
fragment constants, K-major probability layout, scalar QK/PV loops, descending
key-tile traversal, online softmax, and independent per-lane row reductions.
Keep `kLog2Scale=0.127517432f`, `kScale=0.0883883461f`, and per-component
`rcp.approx.ftz.f32` normalization.

Both split launches use `grid=(kMaxQTiles*kHeads)`, `block=(kThreads)`, and zero
dynamic shared memory, with `kMaxQTiles=135`: 8,640 CTAs. Keep the producer's
`__launch_bounds__(kThreads, 1)` and `__grid_constant__ Params` parameter.
The temporary has 2,211,840 entries of 264 bytes, totaling 583,925,760 bytes;
fusion deletes that allocation. Keep global probability scratch
(389,283,840 bytes), reduction scratch (8,847,360 bytes), and LSE storage
(4,194,304 bytes), with their existing lifetimes. No shared-memory staging,
MMA, asynchronous-copy pipeline, swizzle, or persistent scheduler is introduced.

Retain CUDA SM90a compilation, the SM90 host check, pybind11 destination-passing
ABI, and flags `-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo
-DNDEBUG`. No compiler-control change is needed: separate host launches cannot
be fused by device compilation. Inspect generated entry points and global
state traffic when checking the split implementation. Use the existing Kernel
evaluator with the supplied complete problem; keep its numerical and timing
policy unchanged and its evidence outside this card.

# Precondition

- Data types: no particular dtype is required. The removed scratch roundtrip
  must preserve the producer representation, and the same epilogue arithmetic
  and rounding must consume it. A lossy scratch conversion would need explicit
  preservation; bypassing it would change results.
- Layout: every epilogue thread uses only its matching producer thread's results.
  Tile identity and output ownership must agree, or a direct device call would
  consume the wrong values. No additional physical contiguity or alignment is
  needed because fusion removes memory accesses without remapping outputs.
- Storage: the intermediate must be private scratch with no other reader or
  externally visible side effect. Its values must still be available at producer
  completion. Otherwise eliminating its writes would discard required data.
- Pipeline: the preceding implementation completes and exposes producer writes
  before consumer reads, and retains scratch through the last read. Replacing
  this boundary is valid only when the epilogue depends on its own thread's
  completed results and its stores cannot overwrite inputs still being read by
  other producers. Same-thread order then supplies readiness; no cross-thread
  completion or scratch-reuse boundary remains at the epilogue. Otherwise fusion
  creates a race.
- Hardware: no additional feature or shared-memory capacity is required. The
  same device instructions use existing per-thread storage after fusion; no
  new hardware primitive or capacity relationship is introduced.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets replace the indicated blocks. All surrounding computation and the
existing `epilogue` device function remain unchanged. Delete
`attention_epilogue` in full; it only reloads state and calls that function.

## Before

```cuda
// solution/ops.cuh: scratch type and field in Params.
struct EpilogueState {
    float out[kPvRegs];
    float lse[2];
};
// Within Params, between reduction and lse:
    EpilogueState* epilogue;

// solution/attention.cuh: end of consumer.
normalize(out, denominator);
auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + tid];
#pragma unroll
for (int i = 0; i < kPvRegs; ++i) state.out[i] = out[i];
#pragma unroll
for (int r = 0; r < 2; ++r) state.lse[r] = sum[r];

// solution/attention.cuh: separate output entry point.
__global__ __launch_bounds__(kThreads, 1) void attention_epilogue(
    const __grid_constant__ Params p) {
    const int4 work = tile(blockIdx.x, p.q_offsets);
    if (work.w >= kBatch) return;

    const auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + threadIdx.x];
    float out[kPvRegs], lse[2];
    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) out[i] = state.out[i];
    #pragma unroll
    for (int r = 0; r < 2; ++r) lse[r] = state.lse[r];
    epilogue(p, work, out, lse);
}

// solution/kernel.cu: allocation and later Params assignment.
const auto epilogue_state = torch::empty(
    {int64_t(kMaxQTiles) * kHeads * kThreads * sizeof(EpilogueState) / sizeof(float)},
    q.options().dtype(torch::kFloat32));
p.epilogue = reinterpret_cast<EpilogueState*>(epilogue_state.data_ptr<float>());

// solution/kernel.cu: launches using the same configuration and stream.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_epilogue, p));
check(cudaGetLastError());
```

## After

```cuda
// solution/ops.cuh: delete EpilogueState and Params::epilogue.
// solution/attention.cuh: end of consumer; delete attention_epilogue.
normalize(out, denominator);
epilogue(p, work, out, sum);

// solution/kernel.cu: delete epilogue_state and p.epilogue assignment.
// Keep only the producer launch and its existing error check.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());
```
