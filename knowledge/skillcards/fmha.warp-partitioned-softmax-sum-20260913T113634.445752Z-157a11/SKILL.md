---
skill_id: fmha.warp-partitioned-softmax-sum
intent: Distribute softmax denominator scans across row lanes and exchange their partial
  sums.
preconditions:
- 'Data types: lane partials must preserve the accumulator representation, each partition''s
  addition order, and the final reduction tree; changing these can change rounding.'
- 'Layout: known score strides and ownership must partition each row among lanes within
  one warp; matching row identities and traversal order are needed to combine the
  correct partials.'
- 'Storage: row partitions must already be readable by their owner lanes from global
  memory and remain intact through summation; otherwise distributing the reads changes
  their inputs.'
- 'Pipeline: score production must finish before summation; every lane named in the
  warp mask must be able to participate in common publication and reuse barriers in
  the same order. Readers must be able to finish before scratch reuse, and summation
  must finish before score storage is overwritten.'
- 'Hardware: CUDA warp synchronization must order global-memory exchange, with device-memory
  headroom for B records of T accumulator slots, including record padding, where B
  is the launched CTA count and T its thread count.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Distribute each softmax denominator's partition scans across its row lanes, then
exchange partial sums through global scratch. This replaces every lane's redundant
scan of all partitions with one partition scan per lane. Every lane still receives
the same complete denominator for its existing epilogue state.

In `solution/attention.cuh::attention_sum`, replace the block beginning at
`constexpr int kRowsPerLane` and ending immediately before
`float denominator[kRowsPerLane]` with the After summation block. Keep `tid`, `work`,
`length`, `last`, and `maxima` above it and all denominator/LSE stores below it.
Retain `row_sum`, including its no-inline boundary, serial column traversal, and
floating-point additions. Retain descending key-tile traversal and the merge tree
`(partial[0] + partial[2]) + (partial[1] + partial[3])`; a sequential full-row sum
would change rounding.

Add the scratch type, Params field, and sum helper in `solution/ops.cuh`, then the
allocation and pointer assignment in `solution/kernel.cu::kernel`, as shown below.
Allocate a separate record per launched CTA and one accumulator slot per thread.
Each lane publishes its partial, synchronizes its warp, reads its row group's
slots, and synchronizes again before the next row reuses the record. Volatile
accesses preserve the inter-thread global-memory loads and stores. Scratch needs
no initialization because each slot is written before it is read. Keep the tensor
alive for the invocation and allocate it on the guarded caller device/stream.
The simplified `consumer(p)` signature stays unchanged; it does not use this scratch.

The preceding exponential launch publishes scores before summation. The following
probability-conversion launch may overwrite scores only after summation completes.
Keep every launch on the caller stream. Keep the probability conversion's CTA
barrier before its score-buffer reuse; it serves a separate dependency.
The sum kernel's invalid-work return is CTA-uniform. All lanes of valid CTAs,
including lanes assigned padded query rows, must participate in both warp barriers.
Retain masking of invalid key columns before exponentiation, so their stored
exponentials contribute zero. Add no early per-row return in the sum kernel.

## Example configuration

Replay this bundle's `fmha` specialization: FP16 Q/K/V and output, FP32 scores,
exponentials, sums, and scalar FMA accumulators; packed NHD `[16384,64,128]`,
with element strides `(8192,128,1)` and int32 cumulative offsets of length 9.
The supplied workload has eight sequences of 2048 tokens; retain ragged-offset
indexing and the upper-bound launch grid.

Keep `kM=128`, `kN=176`, `kThreads=kMathThreads=256`, `kWarpSize=32`,
`kLanesPerRow=4`, `kQkRegs=88`, and two rows per lane. A group beginning at
`leader=tid-tid%kLanesPerRow` partitions the same two rows. For lane `tid`, fragment
entry `i` belongs to row `(i % kFragmentSize) / kPairElems` and key column
`(tid % kLanesPerRow)*kPairElems + (i/kFragmentSize)*kInputPanelCols + i%kPairElems`.
A score fragment starts at `p.scores[n][blockIdx.x*kThreads+tid]`.

Keep `kMaxQTiles=135`, `kMaxKeyTiles=94`, grid `kMaxQTiles*kHeads=8640`, block
`kThreads=256`, and zero dynamic shared memory. The replay scratch record has
128-byte alignment and 1024 bytes; all records use 8,847,360 bytes. Alignment and
counts are replay settings, not intrinsic to distributing a reduction.

Retain scalar QK/PV arithmetic, full-row maximum computation, separate score,
maximum, exponential, sum, conversion, per-key-tile PV, normalization, packing,
and output launches, existing score/probability storage reuse, scalar half
rounding/stores, scales, and caller ABI. Preserve all compile flags, including
`--use_fast_math`, and the existing SM90 target/host check. This transformation
requires no compiler-control changes outside the shown volatile exchange.

For application checks, use the existing Kernel evaluator with the complete
problem and unchanged numerical/timing policy. Inspect generated code for owner
partition scans and global partial-sum exchange; do not infer either from latency.

# Precondition

- Data types: lane partials must keep the accumulator representation, each
  partition's addition order, and the final reduction tree. Distributed computation
  is valid only if moving an addition to another lane preserves its rounding;
  operand and output dtypes impose no additional restriction on the exchange.
- Layout: known score strides and ownership partition each row among lanes within
  one warp. Those lanes must agree on row identities and traversal order, or the
  gather combines unrelated partials. No additional source alignment or contiguity
  is needed beyond valid typed accesses to the existing partitions.
- Storage: each owner's row partition is already readable from global memory and
  remains intact through summation. Without that visibility and lifetime, the
  assigned lane cannot reproduce the values previously scanned by every lane.
- Pipeline: score production finishes before summation. Every lane named in the
  warp mask can participate in common publication and reuse barriers in matching
  order, allowing readers to see completed writes and finish before scratch is
  overwritten. Summation finishes
  before score storage is overwritten. These ordering rules do not require a
  particular number of pipeline stages.
- Hardware: CUDA warp synchronization must order the global-memory exchange.
  Device memory must have headroom for `B * record_bytes(T, sizeof(accumulator))`,
  where each record contains T slots plus any alignment padding and B is the
  launched CTA count. Otherwise the exchange cannot synchronize or allocate its
  scratch. No shared-memory or matrix-instruction capability is required by this
  transformation.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

`attention_sum` repeats all owner scans locally. There is no reduction scratch
field, helper, or host allocation.

```cuda
    constexpr int kRowsPerLane = 2;
    const int leader = tid - tid % kLanesPerRow;
    float partial[kLanesPerRow][kRowsPerLane];
    float score[kQkRegs], maximum[kRowsPerLane], sum[kRowsPerLane];
    #pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
        maximum[r] = maxima.maximum[r];
    }

    // Each lane scans every row partition without exchanging partial sums.
    #pragma unroll
    for (int owner = 0; owner < kLanesPerRow; ++owner) {
        #pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) partial[owner][r] = 0.f;

        #pragma unroll 1
        for (int n = last; n >= 0; --n) {
            const auto& scores = p.scores[n][size_t(blockIdx.x) * kThreads + leader + owner];
            #pragma unroll
            for (int i = 0; i < kQkRegs; ++i) score[i] = scores.value[i];
            #pragma unroll
            for (int row = 0; row < kRowsPerLane; ++row) {
                partial[owner][row] = row_sum(score, partial[owner][row], row);
            }
        }
    }

    // Preserve the original four-partition addition grouping locally.
    #pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
        sum[r] = (partial[0][r] + partial[2][r])
                 + (partial[1][r] + partial[3][r]);
    }
```

## After

Add the support declarations and helper at the indicated locations:

```cuda
// solution/ops.cuh: add before EpilogueState.
constexpr unsigned kFullWarpMask = 0xffffffffu;
struct alignas(128) Reduction {
    float reduction[kMathThreads];
};

// Add inside Params, after output.
Reduction* reduction;

// Add after prob_state.
__device__ __forceinline__ float reduce_sum(float value, Reduction& s) {
    const int tid = threadIdx.x;
    const int leader = tid - tid % kLanesPerRow;
    volatile float* slots = s.reduction;

    // Publish every partial before any lane reads its row group.
    slots[tid] = value;
    __syncwarp(kFullWarpMask);
    const float a = slots[leader];
    const float b = slots[leader + 1];
    const float c = slots[leader + 2];
    const float d = slots[leader + 3];
    const float result = (a + c) + (b + d);

    // Finish readers before the next row overwrites these slots.
    __syncwarp(kFullWarpMask);
    return result;
}
```

Add the host allocation and Params assignment:

```cuda
// solution/kernel.cu, kernel(): after scratch.ensure(...).
const auto reduction = torch::empty(
    {int64_t(kMaxQTiles) * kHeads * sizeof(Reduction) / sizeof(float)},
    q.options().dtype(torch::kFloat32));

// After Params p{} and before any launch.
p.reduction = reinterpret_cast<Reduction*>(reduction.data_ptr<float>());
```

Replace only the Before summation block; keep its surrounding kernel code:

```cuda
    constexpr int kRowsPerLane = 2;
    auto& s = p.reduction[blockIdx.x];
    float score[kQkRegs], maximum[kRowsPerLane], sum[kRowsPerLane];
    #pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
        maximum[r] = maxima.maximum[r];
        sum[r] = 0.f;
    }

    // Each lane visits only its own partition, retaining its addition order.
    #pragma unroll 1
    for (int n = last; n >= 0; --n) {
        const auto& scores = p.scores[n][size_t(blockIdx.x) * kThreads + tid];
        #pragma unroll
        for (int i = 0; i < kQkRegs; ++i) score[i] = scores.value[i];
        #pragma unroll
        for (int row = 0; row < kRowsPerLane; ++row) {
            sum[row] = row_sum(score, sum[row], row);
        }
    }

    // Replicate the grouped total into every row lane for downstream consumers.
    #pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
        sum[r] = reduce_sum(sum[r], s);
    }
```
