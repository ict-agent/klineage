---
skill_id: fmha.softmax-probability-conversion-fusion
intent: Fuse softmax with probability conversion to eliminate intermediate global
  traffic and a kernel launch.
preconditions:
- 'Data types: eliminating the intermediate store/load must preserve its representation
  and the conversion rounding point; otherwise fusion can change values or reductions.'
- 'Layout: producer and converter must have compatible CTA/thread ownership, and each
  CTA must own every source byte its output scatter can overwrite; otherwise a CTA
  barrier cannot protect readers.'
- 'Storage: the producer-consumer intermediate is materialized in device global memory,
  with no remaining consumer requiring that materialization; otherwise removing its
  store/load would discard needed data.'
- 'Pipeline: upstream production must be visible before use; all CTA readers must
  finish before aliased writes, every CTA thread must reach that barrier, downstream
  readers must follow conversion, and scratch must remain live until readers finish;
  these orders prevent stale reads and premature reuse.'
- 'Hardware: CUDA kernel launches and a CTA barrier with global-memory ordering are
  required; the combined block must fit device launch and resource limits to replace
  the launch boundary safely.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse probability conversion and scatter into the softmax producer. Pass its
exponentials directly to `convert`, eliminating their global store/load and the
`attention_prob` launch. Keep the separate exponential and `row_sum` traversals.

In `solution/attention.cuh`, update `consumer`:

1. Add `uint32_t prob[kProbRegs];` after the score/statistics arrays.
2. Make its per-tile `scores` reference const again; retain the existing fragment
   load, bounds masking, and `softmax` call.
3. Replace the FP32 exponential writeback with `convert`, `__syncthreads`, and
   `store_prob`, as below. Conversion follows `softmax`, including its FP32 sum.
4. Delete the entire `attention_prob` kernel.

In `solution/kernel.cu`, delete the `attention_prob` launch and its immediately
following error check. Keep every other launch and allocation unchanged.

`p.scores[n][blockIdx.x*kThreads+tid]` stores each thread's fragment.
`prob_state(p,n)` aliases the beginning of that CTA's score region as a row-major
probability tile. Preserve `convert` and `store_prob`: they round each scalar once
and scatter the packed halves using the existing lane ownership. The barrier
must precede the scatter because its writes can overlap other threads' raw
score fragments. Every thread in an active CTA follows the same tile loop.
Neither scores nor probabilities cross CTA boundaries.

Keep the prior maximum launch ordered before softmax on the caller stream.
It finishes all full-row score reads before any overwrite. The fused kernel
finishes before the separate PV launches read probabilities and denominators.
Each key tile has distinct storage; no extra barrier is needed between tiles.
Keep scratch alive through the final caller-stream launch.

Preserve ragged sequence lookup, uniform inactive-CTA returns, per-score key
masking, reduction grouping, scalar conversion helpers, and output guards.
Do not move half rounding before the FP32 sum or change FMA order, scales,
exponentials, normalization, or compiler flags.

Check that `attention_prob` is absent and `attention` contains the conversion
and half stores without the intermediate FP32 writeback. Validate the supplied
problem with `klineage.harness.evaluate`; retain its oracle and timing policy.
Keep measurements outside this card.

## Example configuration

This replay uses packed contiguous FP16 Q/K/V/output `[16384,64,128]`, int32
sequence offsets `[9]`, FP32 scores/statistics/accumulators, and scalar
round-to-nearest FP16 conversion. Input strides are `(8192,128,1)` in elements.
The supplied workload has eight sequences of 2048 tokens; retain ragged bounds.

Preserve `kM=128`, `kN=176`, `kThreads=256`, warp size 32, and group size 128.
Each thread owns 88 score values, 44 packed probability words, two row
statistics, and 64 output components. Four lanes share each logical row;
`store_prob` retains its two-row, two-half-pair fragment mapping.

The grid remains `kMaxQTiles*kHeads = 135*64` blocks with no dynamic shared memory
and `__launch_bounds__(kThreads,1)`. Each of the 94 key-tile slabs retains
`135*64*256*sizeof(ScoreState)` bytes. Per CTA, 90112 bytes of score storage
contain the 45056-byte probability tile after its scores are consumed.
No scratch allocation or layout changes are needed.

Retain scalar QK/PV arithmetic, descending key-tile traversal, separate maxima
and PV/output launches, global reduction scratch, and scalar output stores.
Keep CUDA SM90 compilation with `-O3`, `-std=c++17`, `--use_fast_math`,
`--resource-usage`, `-lineinfo`, and `-DNDEBUG`.

# Precondition

- Data types: the removed store/load must not introduce a required rounding or
  representation change. Conversion must keep its original rounding point,
  after every computation that requires the wider intermediate. Otherwise the
  fused consumer or row reduction can receive different values. No particular
  operand dtype is intrinsically required for this fusion.
- Layout: producer and converter must support the same CTA/thread ownership,
  allowing direct fragment handoff. Each CTA must contain every reader of the
  source region its scatter overwrites; a CTA barrier cannot protect another
  CTA's reads. No additional contiguity or alignment beyond the retained accesses
  is needed because their addresses and ownership stay unchanged.
- Storage: a device-global intermediate currently connects the two launches.
  No remaining consumer may require that materialization; otherwise removing
  its store/load would discard needed data.
- Pipeline: upstream production must complete and become visible before use.
  Every active CTA thread must finish its source reads and reach the barrier
  before any aliased output write. Downstream readers must follow conversion,
  and scratch must remain live until those readers finish. These dependencies
  prevent incomplete reads and early storage reuse when a launch boundary is
  removed; no fixed stage count is required.
- Hardware: CUDA launches and a CTA barrier that orders global-memory accesses
  are needed for this replacement. The combined block's thread count and
  compiler-assigned resources must satisfy the device's launch limits. Fusion
  adds no tensor-core, asynchronous-copy, or shared-memory requirement.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The loop tails below share the existing fragment load and bounds masking.
`score`, `maximum`, `sum`, `scores`, `p`, and `n` retain their current meanings.
The After version additionally declares `uint32_t prob[kProbRegs];` beside
`consumer`'s local arrays. Remove `attention_prob` after moving its conversion
and scatter into `consumer`; retain the existing helper definitions.

## Before

```cuda
// consumer: materialize exponentials after the FP32 row sum.
softmax(score, maximum, sum);
#pragma unroll
for (int i = 0; i < kQkRegs; ++i) scores.value[i] = score[i];

// attention_prob: reload the same thread's fragment in a later launch.
const auto& scores = p.scores[n][size_t(blockIdx.x) * kThreads + tid];
#pragma unroll
for (int i = 0; i < kQkRegs; ++i) score[i] = scores.value[i];
convert(score, prob);
__syncthreads();
auto& state = prob_state(p, n);
store_prob(prob, state.prob);

// kernel: separate ordered launches, followed by the existing PV loop.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_prob, p));
check(cudaGetLastError());
```

## After

```cuda
// consumer: convert its live fragment after the unchanged FP32 row sum.
softmax(score, maximum, sum);
convert(score, prob);

// Every raw-score reader finishes before the alias receives rounded halves.
__syncthreads();
auto& state = prob_state(p, n);
store_prob(prob, state.prob);

// kernel: fused launch, followed by the unchanged PV loop.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());
```
