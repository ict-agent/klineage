---
skill_id: fmha.qk-register-tiling
intent: Interleave per-thread QK dot products in a register tile to reuse operands.
preconditions:
- 'Data types: independent dot products permit interleaving while preserving each
  accumulator type, operand conversion, ordered FMA sequence, and rounding; changing
  those operations can change results.'
- 'Layout: each thread owns multiple independent scores with known Q/K addresses over
  a common reduction index, and some scores share operands; this permits private accumulation
  and operand reuse. Contiguity and vector alignment add no requirement for scalar
  loads.'
- 'Storage: the complete score set is thread-private until consumed; register accumulators
  cannot directly supply another thread. Q/K remain readable in their existing storage
  throughout score production.'
- 'Pipeline: Q/K producers complete and make their values visible before reads; operands
  remain unchanged until all dot products finish. Consumers must not require individual
  scores before the whole set completes, and buffers cannot be reused earlier, because
  interleaving changes score completion order.'
- 'Hardware: CUDA registers must accommodate the chosen score tile plus other live
  state within per-thread and per-CTA limits; otherwise spills defeat register residency.
  No tensor-core or warp-exchange feature is required.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore QK accumulator register tiling in `solution/attention.cuh::query_key`.
Replace its body as shown below. Initialize all scores, make the reduction
coordinate `d` the outer loop, and fully unroll the score loop `i`. Independent
scores then remain live in registers together; the compiler can reuse Q/K values
shared by those scores and schedule independent FMAs. Remove the serialized
loop's scalar accumulator and dynamic score store. Its `#pragma unroll 1` on `i`
must become `#pragma unroll`; retain `#pragma unroll 1` on `d`.

Each thread keeps its existing `(row, col)` ownership. Q/K addresses remain
`row * kRow + d` and `col * kRow + d`; scores retain the same array indices for
softmax. Keep invalid-row/key loads zero-filled and the caller's invalid-key
negative-infinity mask. Each score still starts at zero and accumulates with
`fmaf` in increasing `d`, using the same half-to-float conversions. Interleaving
independent scores changes no individual reduction or rounding sequence.

No launch, allocation, layout, or synchronization changes are needed. Keep
caller-stream input visibility and finish QK before softmax consumes the score
array. Retain shared-memory softmax exchanges and both CTA barriers surrounding
global probability consumption and reuse. They are outside this transformation.

Inspect compiled device code for multiple live score accumulators and reused
operand loads. Evaluate the unchanged problem with the existing harness; preserve
its oracle, workload, seed, tolerances, and timing policy. Keep evidence outside
this card.

## Example configuration

Preserve packed contiguous FP16 Q/K/V/output `[16384, 64, 128]`, FP32 scores and
accumulators, eight sequences, and int32 offsets `[9]`. Packed row stride is
`kRow = kHeads * kDim = 8192` half elements (16384 bytes). QK uses `kDim = 128`,
`kM = 128`, `kN = 176`, and `kQkRegs = 88` scores per thread. The map assigns two
rows and 44 columns to each thread: two 128-thread groups, 32-thread warps,
`kMmaRows = 64`, `kFragmentRows = 16`, `kCoreRows = 8`,
`kLanesPerRow = 4`, `kFragmentSize = 4`, `kPairElems = 2`, and
`kInputPanelCols = 8`.

Keep `__launch_bounds__(kThreads, 1)`, the 256-thread launch, `grid.x = kMaxQTiles * kHeads = 8640`, dynamic shared
memory `sizeof(Shared) = 1024`, and the ragged sequence tile scheduler. Each CTA
keeps its 22528-half global probability tile (45056 bytes), with total allocation
`kMaxQTiles * kHeads * kProbTileElems` half elements. Retain 64 PV accumulators
per thread, reverse key-tile traversal, increasing-key scalar PV FMAs, online
softmax, FP16 probability rounding, scalar conversion calls, scalar output
stores, and the LSE workspace. Preserve scale constants `kLog2Scale = 0.127517432f`
and `kScale = 0.0883883461f`.

Keep the SM90 target and compile flags `-O3 -std=c++17 --use_fast_math
--resource-usage -lineinfo -DNDEBUG`. The host ABI, caller stream, bounds handling,
and remaining files stay unchanged. These dimensions and retained mechanisms
specify this replay; they are not prerequisites of register tiling.

# Precondition

- Data types: independent dot products must allow interleaving with the same
  accumulator type, operand conversions, ordered FMA sequence, and rounding.
  Altering those operations can change numerical results. Register tiling itself
  adds no fixed operand dtype requirement.
- Layout: one thread owns multiple independent scores with known Q/K address
  mappings over a common reduction coordinate. Some scores share operands, which
  enables load reuse; ownership enables private accumulators. Scalar loads require
  no additional contiguity or vector alignment.
- Storage: scores remain private to their owning thread until consumed, because
  registers cannot directly serve another thread. Q/K stay readable in their
  existing storage throughout score production; no memory-space move is required.
- Pipeline: Q/K production completes with reader visibility before score reads.
  Q/K remain unchanged until all dot products finish. Consumers must accept the
  whole score set after completion, without depending on earlier per-score
  completion. Backing buffers cannot be reused before all reads finish.
  Interleaving otherwise changes observed operands or exposes unfinished scores.
- Hardware: the chosen accumulator tile plus other live state must fit CUDA's
  per-thread and per-CTA register limits for register residency. Spills defeat
  that property. Tensor-core and warp-exchange features add no requirement.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
__device__ __forceinline__ void query_key(float (&score)[kQkRegs], const __half* q,
                                        const __half* k, int qvalid, int kvalid, int wg) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = (threadIdx.x % kGroupSize) / kWarpSize;
    // Finish one dot product at a time, preserving each score's FMA order.
    #pragma unroll 1
    for (int i = 0; i < kQkRegs; ++i) {
        const int row = wg * kMmaRows + warp * kFragmentRows
                        + lane / kLanesPerRow + (i % kFragmentSize) / kPairElems * kCoreRows;
        const int col = (i / kFragmentSize) * kInputPanelCols
                        + (lane % kLanesPerRow) * kPairElems + i % kPairElems;
        float acc = 0.f;

        #pragma unroll 1
        for (int d = 0; d < kDim; ++d) {
            const float qvalue = row < qvalid ? __half2float(q[row * kRow + d]) : 0.f;
            const float kvalue = col < kvalid ? __half2float(k[col * kRow + d]) : 0.f;
            acc = fmaf(qvalue, kvalue, acc);
        }
        score[i] = acc;
    }
}
```

## After

```cuda
__device__ __forceinline__ void query_key(float (&score)[kQkRegs], const __half* q,
                                        const __half* k, int qvalid, int kvalid, int wg) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = (threadIdx.x % kGroupSize) / kWarpSize;
    #pragma unroll
    for (int i = 0; i < kQkRegs; ++i) score[i] = 0.f;

    // Keep each lane's output tile; read Q/K directly from global memory.
    #pragma unroll 1
    for (int d = 0; d < kDim; ++d) {
        #pragma unroll
        for (int i = 0; i < kQkRegs; ++i) {
            const int row = wg * kMmaRows + warp * kFragmentRows
                            + lane / kLanesPerRow + (i % kFragmentSize) / kPairElems * kCoreRows;
            const int col = (i / kFragmentSize) * kInputPanelCols
                            + (lane % kLanesPerRow) * kPairElems + i % kPairElems;
            const float qvalue = row < qvalid ? __half2float(q[row * kRow + d]) : 0.f;
            const float kvalue = col < kvalid ? __half2float(k[col * kRow + d]) : 0.f;
            score[i] = fmaf(qvalue, kvalue, score[i]);
        }
    }
}
```
