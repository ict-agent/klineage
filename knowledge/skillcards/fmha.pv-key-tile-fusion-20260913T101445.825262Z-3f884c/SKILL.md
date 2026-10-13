---
skill_id: fmha.pv-key-tile-fusion
intent: Fuse PV key-tile launches to retain output accumulators across tiles.
preconditions:
- 'Data types: eliminating partial-result stores must preserve accumulator representation
  and required rounding; retain each output''s reduction order and final normalization
  so fusion does not change numerical behavior.'
- 'Layout: consecutive tile launches must assign each partial output to the same logical
  thread, with no dependency on another thread''s partial output; otherwise thread-local
  continuation cannot replace the launch boundary.'
- 'Storage: all operand tiles and row statistics must already be available in device-visible
  storage, and intermediate output states must have no external readers; fusion removes
  their between-tile publication.'
- 'Pipeline: operand producers must finish before PV reads; dependencies between tiles
  must be internal to each output owner. Final consumers must follow PV completion,
  and operand/output storage must remain live without conflicting writes until its
  last reader finishes.'
- 'Hardware: no additional specialized feature is required; ordinary CUDA thread-local
  state and ordered launches suffice. The fused kernel must satisfy the target''s
  launch and per-thread resource limits.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse the descending PV key-tile traversal into one CUDA launch. Keep each
thread's output accumulators private across tiles, removing intermediate global
loads/stores and repeated launches. Scores and probabilities remain materialized.

In `solution/kernel.cu`, replace the descending `attention_values` launch loop
with one launch on the existing caller stream. In `solution/attention.cuh`, remove
that kernel's `n` argument and `n > last` return. Initialize `out` once, traverse
`n = last ... 0` inside the kernel, normalize after the loop, and store once to
the existing epilogue state. The snippets replace this launch region and the
complete `attention_values` function. No other source or compiler change is needed.

Retain the CTA/thread mapping and launch configuration. Thread `tid` of CTA `b`
owns `p.epilogue[b * kThreads + tid].out`. Component `i` maps to query row
`wg*kMmaRows + warp*kFragmentRows + lane/kLanesPerRow
+ ((i%kFragmentSize)/kPairElems)*kCoreRows` and output column
`(i/kFragmentSize)*kInputPanelCols + (lane%kLanesPerRow)*kPairElems
+ i%kPairElems`, relative to the CTA's query tile and head. Here
`wg=tid/kGroupSize`, `warp=(tid%kGroupSize)/kWarpSize`, and `lane=tid%kWarpSize`.
Keep the row-major probability tile and packed-NHD V addressing unchanged.

The softmax launch completes probability and denominator production before PV
starts. Preserve its CTA barrier before probability writes reuse score storage.
PV reads immutable probabilities, V, and denominators; each thread updates only
its own outputs. No inter-tile barrier is needed inside the fused loop. Keep the
epilogue launch after PV and retain all scratch allocations through that launch.

Keep the invalid-CTA return and derive `last` separately for each sequence. The
fused loop visits only that sequence's valid tiles; retain `valid=length-n*kN`
and the existing padded-key behavior in `prob_value`. Preserve descending tile
order, increasing-key scalar FMA order within each tile, FP32 accumulators, and
normalization exactly once after tile zero. The removed FP32 stores introduce
no extra rounding. Retain probability/output half rounding and all numerical
checks in the supplied problem.

## Example configuration

Preserve CUDA SM90a compilation, the bundle configuration, ABI, and compile flags.
The workload has 16,384 tokens, 64 heads, dimension 128, and eight sequences;
Q/K/V/output are contiguous FP16 packed-NHD arrays with element strides
`(8192,128,1)`. Sequence offsets remain device-resident int32 inputs.

The query/key tile is `kM=128` by `kN=176`. Keep 256 threads per CTA, 32 threads
per warp, `kGroupSize=128`, grid `kMaxQTiles*kHeads=8640`, launch bounds
`(kThreads,1)`, and zero dynamic shared memory. Ownership constants remain
`kMmaRows=64`, `kFragmentRows=16`, `kCoreRows=8`, `kFragmentSize=4`,
`kLanesPerRow=4`, `kPairElems=2`, and `kInputPanelCols=8`.

Each thread retains `kPvRegs=64` FP32 output components and two row denominators.
Probability tiles contain `128*176` FP16 elements. Keep all 94 score slabs,
score/probability storage reuse, scalar QK/PV loops, the full-row softmax maximum,
global-scratch row reductions and their warp barriers, scalar half conversions,
and the separate score, softmax, and epilogue launches. Keep `kLog2Scale`, `kScale`,
`score_offset`, `row_sum`, and `normalize` unchanged.

The deoptimized host submits 94 PV launches; CTAs skip tiles beyond their own
`last`. Fusion replaces those with one launch whose tile count remains per-CTA.
No allocation size or layout changes. Check correctness and timing through the
prescribed Kernel evaluator; inspect launch structure to confirm fusion.

# Precondition

- Data types: eliminating partial-result stores must preserve accumulator
  representation and required rounding. Preserve each output's reduction order
  and normalize at the same logical point; otherwise removing memory boundaries
  can change the arithmetic result.
- Layout: consecutive tiles must have the same logical thread owner for every
  partial output. That continuation must not read another thread's partial
  output; such a dependency would require synchronization lost with the launch
  boundary. No extra contiguity or vector alignment is required: operand
  addressing and ownership stay unchanged.
- Storage: operand tiles and row statistics must already be device-visible.
  No external consumer may require intermediate output publication between
  tiles, since the fused kernel retains that state privately until completion.
- Pipeline: operand producers finish before PV consumption. Between-tile
  dependencies remain within each output owner; final consumers follow PV
  completion. Keep every operand/output allocation live and prevent conflicting
  writes until its last reader finishes. These conditions replace the removed
  launch boundaries without incomplete reads or premature buffer reuse.
- Hardware: no additional specialized feature is required. Ordinary CUDA
  thread-local state and ordered launches implement this transformation. The
  compiled fused kernel must satisfy launch and per-thread resource limits;
  no fixed register count or shared-memory capacity is intrinsic to fusion.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

`solution/kernel.cu`, between the softmax and epilogue launches:

```cuda
// Publish partial outputs between descending key-tile launches.
for (int n = kMaxKeyTiles - 1; n >= 0; --n) {
    check(cudaLaunchKernelEx(&config, attention_values, p, n));
    check(cudaGetLastError());
}
```

`solution/attention.cuh`:

```cuda
__global__ __launch_bounds__(kThreads, 1) void attention_values(
    const __grid_constant__ Params p, const int n) {
    const int tid = threadIdx.x;
    const int wg = tid / kGroupSize;
    const int4 work = tile(blockIdx.x, p.q_offsets);
    if (work.w >= kBatch) return;

    const int kstart = p.k_offsets[work.w];
    const int length = p.k_offsets[work.w + 1] - kstart;
    const int last = (length + kN - 1) / kN - 1;
    if (n > last) return;

    auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + tid];
    float out[kPvRegs], denominator[2];

    // Resume each output from the preceding tile's caller-stream launch.
    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) out[i] = n == last ? 0.f : state.out[i];

    const int base = (kstart + n * kN) * kRow + work.z * kDim;
    const int valid = length - n * kN;
    const auto& prob = prob_state(p, n);
    prob_value(out, prob.prob, p.v + base, valid, wg);

    // Normalize only after the final key tile has contributed.
    if (n == 0) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) denominator[r] = state.denominator[r];
        normalize(out, denominator);
    }

    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) state.out[i] = out[i];
}
```

## After

`solution/kernel.cu`, at the same location:

```cuda
check(cudaLaunchKernelEx(&config, attention_values, p));
check(cudaGetLastError());
```

`solution/attention.cuh`:

```cuda
__global__ __launch_bounds__(kThreads, 1) void attention_values(
    const __grid_constant__ Params p) {
    const int tid = threadIdx.x;
    const int wg = tid / kGroupSize;
    const int4 work = tile(blockIdx.x, p.q_offsets);
    if (work.w >= kBatch) return;

    const int kstart = p.k_offsets[work.w];
    const int length = p.k_offsets[work.w + 1] - kstart;
    const int last = (length + kN - 1) / kN - 1;
    float out[kPvRegs], denominator[2];
    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) out[i] = 0.f;

    // Caller-stream ordering publishes every tile before this traversal.
    #pragma unroll 1
    for (int n = last; n >= 0; --n) {
        const int base = (kstart + n * kN) * kRow + work.z * kDim;
        const int valid = length - n * kN;
        const auto& state = prob_state(p, n);
        prob_value(out, state.prob, p.v + base, valid, wg);
    }

    auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + tid];
    #pragma unroll
    for (int r = 0; r < 2; ++r) denominator[r] = state.denominator[r];
    normalize(out, denominator);

    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) state.out[i] = out[i];
}
```
