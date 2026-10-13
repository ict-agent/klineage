---
skill_id: fmha.exponential-row-sum-fusion
intent: Fuse exponential production and row summation to reuse local values.
preconditions:
- 'Data types: preserve the materialized values and required reduction arithmetic
  order; reproduce any intermediate rounding locally so fusion keeps the same summation
  inputs. No fixed dtype is intrinsic.'
- 'Layout: producer threads can traverse the same row elements as their summation
  counterparts in the required order, enabling correct local reuse. No additional
  alignment or contiguity is required.'
- 'Storage: reduction operands are available locally during production, and row-statistic
  destinations and reduction scratch are accessible there; otherwise moved work lacks
  inputs or storage. Preserve materialization needed by other consumers.'
- 'Pipeline: producer inputs are complete and visible; reduction participants reach
  publication and reuse barriers; later consumers wait for results before reading
  or overwriting buffers; the removed launch has no remaining work or required external
  ordering effects. These prevent races and lost behavior; no fixed stage count is
  required.'
- 'Hardware: no additional feature or fixed capacity requirement; ordinary thread-local
  execution and existing reduction synchronization suffice without a new shared buffer
  or specialized resource.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse exponential production and row summation in `solution/attention.cuh`.
Accumulate each thread's row partials while its exponential fragment is available,
eliminating `attention_sum` and its global reloads. Keep the exponential stores:
`attention_prob` still needs them for conversion.

## Replay

1. Add `float (&sum)[2]` to `softmax` after `maximum`. After its inner exponential
   loop, inside the row loop, call `sum[row] = row_sum(score, sum[row], row);`.
   Retain the separate, noinline `row_sum` traversal and its exact addition order.
2. In `consumer`, add `sum[2]` beside `score` and `maximum`. Initialize `sum[r]`
   to zero in the existing maxima-load loop. Change the call to
   `softmax(score, maximum, sum)`. Keep all score loads, tail masking, descending
   key-tile traversal, and stores. Update the store comment to describe conversion
   as its remaining consumer.
3. Move the complete tail of `attention_sum`, beginning with
   `float denominator[2];` and ending with the `state.lse`/`state.denominator`
   stores, unchanged to `consumer`, immediately after its key-tile loop.
   `consumer` already receives the CTA's `Reduction& s`; retain that parameter.
   Delete the entire `attention_sum` entry after moving its tail.
4. In `solution/kernel.cu`, remove only the `attention_sum` launch and its following
   `check(cudaGetLastError())`. Keep `attention` immediately before `attention_prob`.
   No launch dimensions, allocation, structure, ABI, or compiler-flag changes.

Every thread keeps its original score fragment and two row partials. Element `i`
belongs to local row `(i % kFragmentSize) / kPairElems`; `row_sum` visits those
indices in its existing order. The final four-lane reduction keeps
`(a + c) + (b + d)`. The fused kernel reads maxima published by
`attention_maximum`; its caller-stream completion publishes exponentials and row
statistics before conversion and normalization. Preserve both `__syncwarp` calls
in `reduce`: the first publishes partials, the second prevents scratch reuse
before all gathers finish. CTA-uniform invalid-work returns remain before them.

`attention_prob` retains its `__syncthreads()` before overwriting consumed score
storage with FP16 probabilities. No new barrier is needed for thread-local
exponential-to-sum reuse. Buffer lifetimes and caller-stream allocation reuse
remain unchanged. Keep invalid key scores at `-INFINITY`, their zero exponentials,
and the finite-maximum guard. Preserve FP32 arithmetic, scale constants, `exp2f`,
`score_offset`, summation grouping, FP16 conversion, and output semantics.

## Example configuration

Replay the existing packed-NHD workload: 16384 tokens, 64 heads, head dimension
128, eight sequences, and nine offsets. Q/K/V/output are contiguous FP16 with
half-element strides `(8192, 128, 1)`; offsets are int32. The supplied sequences
have 2048 tokens each; retain offset-based ragged bounds.

Keep `kM=128`, `kN=176`, `kThreads=256`, `kGroupSize=128`, `kWarpSize=32`,
`kLanesPerRow=4`, `kPairElems=2`, `kFragmentSize=4`, `kCoreRows=8`, and
`kInputPanelCols=8`. Each lane owns 88 FP32 scores, 44 entries in each of two rows.
A row's four lanes cover 176 keys. Keep 64 PV components per thread and all
existing fragment mappings. `kMaxQTiles=135`, `kMaxKeyTiles=94`; launch 8640 CTAs
with 256 threads, zero dynamic shared memory, and `__launch_bounds__(kThreads, 1)`.
No explicit register budget changes.

Score storage remains 94 slabs of 778567680 bytes each, addressed as
`p.scores[n][size_t(blockIdx.x) * kThreads + tid]`. Each CTA occupies 90112 bytes
of FP32 score storage; its 45056-byte row-major half probability tile subsequently
reuses that region. `Reduction` remains 1024 bytes per CTA. `EpilogueState` remains
408 bytes per thread. No scratch size changes.

Keep `kLog2Scale=0.127517432f`, `kScale=0.0883883461f`, decreasing tile order and
increasing within-row addition order. Retain scalar QK/PV FMA loops, full-row
maxima, separate probability conversion, per-tile PV launches, normalization,
output packing, and scalar stores. The target remains SM90, compiled for
compute_90a/sm_90a with `-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo
-DNDEBUG`. These instance settings are unchanged by fusion.

# Precondition

- Data types: eliminating the intermediate read must preserve the values seen by
  summation and its required arithmetic order. Any materialization rounding must
  be reproduced locally; otherwise fusion changes the reduction inputs. No fixed
  dtype is intrinsic to this fusion.
- Layout: each producer thread can identify and traverse the same row elements
  assigned to its summation counterpart in the required order. Otherwise direct
  local reuse sums the wrong values or requires redistribution. No additional
  alignment or contiguity requirement: existing accesses remain valid.
- Storage: reduction operands are available locally during production, and the
  producer can access the row-statistic destinations and reduction scratch.
  Otherwise the moved work lacks inputs or storage. Other consumers of the
  materialized exponentials still require their stores.
- Pipeline: producer inputs must be complete and visible before use; all reduction
  participants must reach its publication and reuse barriers. Later consumers
  must observe completed exponentials and row statistics before reading or
  overwriting their buffers. The removed launch must have no remaining work or
  required external ordering effects. Without these conditions fusion introduces
  races or removes required behavior. No fixed stage count is required.
- Hardware: no additional feature or fixed capacity requirement. Fusion uses
  ordinary thread-local execution and the existing reduction synchronization;
  it reserves no new shared-memory buffer or specialized instruction resource.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The excerpts show the changed dataflow. Keep omitted bounds, loads, exponential
arithmetic, and stores unchanged. Move the complete row-statistic tail as directed
above; the excerpt only abbreviates its unchanged publication stores.

## Before

```cuda
// softmax: exponential production only.
__device__ __forceinline__ void softmax(
    float (&score)[kQkRegs], const float (&maximum)[2]) {
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        // Existing finite-maximum guard and exponential loop.
    }
}

// consumer: inside its descending key-tile loop.
softmax(score, maximum);
#pragma unroll
for (int i = 0; i < kQkRegs; ++i) scores.value[i] = score[i];

// attention_sum: separate kernel, sum initialized to zero for both rows.
#pragma unroll 1
for (int n = last; n >= 0; --n) {
    const auto& scores = p.scores[n][size_t(blockIdx.x) * kThreads + tid];
    #pragma unroll
    for (int i = 0; i < kQkRegs; ++i) score[i] = scores.value[i];
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        sum[row] = row_sum(score, sum[row], row);
    }
}
// Existing denominator, reduction, LSE, and state-publication tail.

// Host launch sequence.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_sum, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_prob, p));
check(cudaGetLastError());
```

## After

```cuda
// softmax: retain the existing exponential loop, then sum its local row.
__device__ __forceinline__ void softmax(
    float (&score)[kQkRegs], const float (&maximum)[2], float (&sum)[2]) {
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        // Existing finite-maximum guard and exponential loop.
        sum[row] = row_sum(score, sum[row], row);
    }
}

// consumer: declarations and maxima initialization.
float score[kQkRegs], maximum[2], sum[2];
const auto& maxima = p.epilogue[size_t(blockIdx.x) * kThreads + tid];
#pragma unroll
for (int r = 0; r < 2; ++r) {
    maximum[r] = maxima.maximum[r];
    sum[r] = 0.f;
}

// Inside consumer's unchanged descending key-tile loop, after masking.
softmax(score, maximum, sum);
#pragma unroll
for (int i = 0; i < kQkRegs; ++i) scores.value[i] = score[i];

// After that loop: move attention_sum's complete tail here unchanged.
float denominator[2];
#pragma unroll
for (int r = 0; r < 2; ++r) {
    sum[r] = reduce<Reduce::Sum>(sum[r], s);
    denominator[r] = sum[r];
    sum[r] = maximum[r] * kScale + __logf(sum[r]);
}
// Existing state.lse and state.denominator publication stores.

// Delete attention_sum and its launch; retain the caller stream.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_prob, p));
check(cudaGetLastError());
```
