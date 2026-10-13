---
skill_id: fmha.pv-register-tiling
intent: Interleave independent probability–value accumulations in a per-thread register
  tile.
preconditions:
- 'Data types: output accumulations must be independent, and loop interchange must
  preserve each output''s operand conversions, accumulator representation, and ordered
  arithmetic; otherwise rounding or dependencies can change results. No particular
  operand dtype is intrinsic to register tiling.'
- 'Layout: each thread owns multiple independent outputs with known operand indices
  and a common reduction traversal; outputs sharing a probability row permit reuse
  of that probability across columns. No additional contiguity or alignment is required
  because accesses remain scalar.'
- 'Storage: probabilities and values are readable and stable during the reduction,
  and output state is thread-private and cannot alias those operands; interleaving
  updates would otherwise change later reads.'
- 'Pipeline: probability producers finish and publish their writes before consumption,
  every reader finishes before tile reuse, and output initialization or rescaling
  precedes accumulation; loop interchange must not reorder synchronization or participation.'
- 'Hardware: the register file must accommodate the simultaneous output accumulators
  and other live state within per-thread and per-CTA allocation limits for resident
  register tiling; insufficient capacity causes spills. No tensor instructions or
  extra shared storage are required.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore probability–value register tiling in `solution/attention.cuh::prob_value`.
Replace that function with After below; all referenced constants and `panel` already
exist in the bundle. Move the increasing-key loop outside the output-entry loop,
fully unroll the latter, and update `out[i]` directly. This keeps independent
accumulators live together, exposes instruction parallelism, and lets each row's
probability load feed multiple output columns. The serial version instead finishes
one output before starting another.

Keep the key loop's `#pragma unroll 1`. Replace the serial output loop's
`#pragma unroll 1` with `#pragma unroll` after interchange; this makes output indices
constant and enables register allocation. Remove the scalar `acc` initialization
and final writeback. These compiler controls support the same tiling change.

Ownership, global addresses, launch dimensions, shared storage, and synchronization
stay unchanged. Each thread owns the same `out` entries. Read probability `(row,key)`
through `panel<kM>` and value `(key,col)` through `key*kRow+col`. Preserve the `valid`
guard and zero-valued invalid V loads. Keep all padded key iterations: each output
still performs the same increasing-key FMA sequence, starting from its existing,
possibly rescaled accumulator. Do not change probability rounding or tile traversal.

`consumer` publishes `store_prob` writes with `__syncthreads()` before `prob_value`;
its following barrier completes all readers before reusing that CTA's global tile.
Keep both barriers and the uniform invalid-CTA return. There are no barriers inside
the interchanged loops. Q, K, and V retain caller-stream visibility and lifetime.

Inspect generated code at the PV loop: the serial version should have one dependent
accumulator and an outer output loop; the optimized version should have independent
accumulator registers inside one key loop. Use the existing evaluator for correctness
and timing; keep its workload and policy unchanged.

## Example configuration

Preserve this replay's packed contiguous NHD FP16 Q/K/V and output, FP32 scalar
accumulation, and int32 offsets. The workload has 16384 tokens, 64 heads, head
dimension 128, and eight sequences. `kRow = kHeads*kDim = 8192` half elements,
so the global token-row stride is 16384 bytes; head stride is 128 half elements.

A CTA owns one query tile and head. Retain `kM=128`, `kN=176`, `kPvRegs=64`,
`kQkRegs=88`, `kProbRegs=44`, 256 threads, 32 lanes per warp, and two 128-thread
groups. `wg=threadIdx.x/128`; the function uses the warp index within that group.
Each lane owns two rows and 32 columns per row. Retain `kMmaRows=64`,
`kFragmentRows=16`, `kFragmentSize=4`, `kCoreRows=8`, `kLanesPerRow=4`, and
`kPairElems=2`. These names describe retained ownership; PV uses scalar arithmetic.

Probability storage is a CTA-private global tile of `kM*kN=22528` half elements
(45056 bytes). Retain eight-column panels with
`panel<Rows>(row,col)=(col/8)*Rows*8+row*8+col%8`.
`store_prob` uses the same mapping; no writer changes are needed. Preserve the
per-invocation allocation of `kMaxQTiles*kHeads*kProbTileElems` half elements,
389283840 bytes, and its caller-stream lifetime.

Keep the launch at `kMaxQTiles*kHeads=8640` CTAs, 256 threads per CTA,
`__launch_bounds__(kThreads,1)`, and `sizeof(Shared)=1024` dynamic shared bytes
for reductions. Register allocation remains compiler-managed; add no register cap.
Keep SM90a compilation and flags `-O3 -std=c++17 --use_fast_math --resource-usage
-lineinfo -DNDEBUG`. Preserve the pybind11 destination-passing ABI, CUDA device
guard, caller stream, and config.toml.

Retain serial QK dots, online softmax, descending key tiles, increasing keys within
each tile, FP32 `fmaf`, scale constants `kLog2Scale=0.127517432f` and
`kScale=0.0883883461f`, FP16 probability rounding, scalar conversion helpers,
scalar output stores, shared-memory peer exchange, and ordinary CTA scheduling.

# Precondition

- Data types: each output has an independent accumulation. Interleaving outputs
  must retain each one's conversions, accumulator representation, and arithmetic
  sequence; cross-output dependencies or changed rounding invalidate the result.
  Register tiling itself imposes no particular operand dtype.
- Layout: known per-thread ownership must include multiple independent outputs
  over a common reduction traversal. Sharing a probability row across columns
  supplies the operand reuse. Scalar accesses impose no new alignment or contiguity
  requirement; different layouts require adapting the indices.
- Storage: probability and value inputs must remain readable and unchanged while
  thread-private output state is updated. Output state must not alias these inputs,
  because interleaving updates could otherwise alter subsequent operand reads.
- Pipeline: producers must finish and make probabilities visible before consumers
  start. Readers must finish before the tile is overwritten. Initialization or
  rescaling must finish before accumulation. Interchange cannot cross synchronization
  or change its participants; those ordering dependencies prevent stale operands,
  premature reuse, and wrongly initialized accumulators. No stage count is required.
- Hardware: simultaneous accumulator storage plus other live state must fit the
  register allocation limits per thread and CTA to retain the register tile.
  Symbolically, the live tile needs `outputs_per_thread*sizeof(accumulator)` bytes,
  plus other live registers and the target's allocation rounding. Exceeding the
  limits forces spills. No tensor instructions or additional shared capacity are
  needed for this scalar loop interchange.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace only `prob_value`; its signature and existing constants provide the shared
context. Before completes one output at a time; After interleaves all owned outputs
at each key without changing any output's reduction order.

## Before

```cuda
__device__ __forceinline__ void prob_value(float (&out)[kPvRegs], const __half* prob_tile,
                                         const __half* v, int valid, int wg) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = (threadIdx.x % kGroupSize) / kWarpSize;

    // Finish one output at a time, preserving its increasing-key FMA order.
    #pragma unroll 1
    for (int i = 0; i < kPvRegs; ++i) {
        const int row = wg * kMmaRows + warp * kFragmentRows
                        + lane / kLanesPerRow + (i % kFragmentSize) / kPairElems * kCoreRows;
        const int col = (i / kFragmentSize) * kInputPanelCols
                        + (lane % kLanesPerRow) * kPairElems + i % kPairElems;
        float acc = out[i];

        #pragma unroll 1
        for (int key = 0; key < kN; ++key) {
            const float value = key < valid ? __half2float(v[key * kRow + col]) : 0.f;
            acc = fmaf(__half2float(prob_tile[panel<kM>(row, key)]), value, acc);
        }
        out[i] = acc;
    }
}
```

## After

```cuda
__device__ __forceinline__ void prob_value(float (&out)[kPvRegs], const __half* prob_tile,
                                         const __half* v, int valid, int wg) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = (threadIdx.x % kGroupSize) / kWarpSize;

    // Accumulate the rounded probability tile in increasing key order.
    #pragma unroll 1
    for (int key = 0; key < kN; ++key) {
        #pragma unroll
        for (int i = 0; i < kPvRegs; ++i) {
            const int row = wg * kMmaRows + warp * kFragmentRows
                            + lane / kLanesPerRow + (i % kFragmentSize) / kPairElems * kCoreRows;
            const int col = (i / kFragmentSize) * kInputPanelCols
                            + (lane % kLanesPerRow) * kPairElems + i % kPairElems;
            const float value = key < valid ? __half2float(v[key * kRow + col]) : 0.f;
            out[i] = fmaf(__half2float(prob_tile[panel<kM>(row, key)]), value, out[i]);
        }
    }
}
```
