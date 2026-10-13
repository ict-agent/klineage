---
skill_id: fmha.qk-softmax-fusion
intent: Fuse QK score production with softmax consumption to eliminate global score
  materialization.
preconditions:
- 'Data types: locally produced scores must preserve the representation, arithmetic
  order, and rounding observed after materialization; removing the store/load must
  not change softmax inputs.'
- 'Layout: producer and consumer score ownership must coincide within a thread, with
  known input strides and bounds; otherwise direct local consumption would use another
  thread''s score.'
- 'Storage: scores are internal global-memory intermediates with no other readers
  or observable uses, and their input operands remain accessible to the consumer;
  otherwise deleting materialization loses required data.'
- 'Pipeline: operand producers must finish and make inputs visible before fused computation;
  each score must be computed before consumption, inputs must remain stable, and readers
  must finish before local or retained scratch reuse. No intervening cross-CTA dependency
  may require the removed launch boundary.'
- 'Hardware: the existing CUDA device must support the retained arithmetic, and per-thread
  score state plus other live state must fit legal kernel resource limits, allowing
  compiler spills; fusion requires no new instruction family or shared-memory allocation.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse QK score production into the online softmax/PV consumer to remove the global
score intermediate and its producer launch. Keep the existing scalar `query_key`
implementation and its per-score FMA order.

In `solution/attention.cuh`, replace the score-loading block in `consumer`'s
descending key-tile loop with the `query_key` call shown below. Insert it after
`base` and `valid` are computed, before the existing last-tile mask. All call
arguments already exist in `consumer`; retain `qstart` and `qvalid` there.
Delete the entire `attention_scores` kernel. Every consumer thread now computes
its own score fragment immediately before masking and softmax.

In `solution/kernel.cu`, delete the `score_state` array and its allocation
loop (including the `p.scores[n]` assignments), the `<array>` include, and the
`attention_scores` launch with its following error check. Retain `Params p{};`.
In `solution/ops.cuh`, delete `kMaxKeyTiles`, `ScoreState`, and `Params::scores`.
These objects serve only the removed intermediate. Keep all other fields,
allocations, checks, functions, compile flags, and source files.

The producer and consumer use the same CTA work mapping from `tile`, the same
thread ownership, and the same descending key-tile traversal. Before fusion,
`p.scores[n]` selects one global slab per key tile. Each slab is laid out as
`[query_tile_and_head][thread][fragment_element]`, with the final axis contiguous.
Its element address is `((blockIdx.x * kThreads + tid) * kQkRegs + i)`; use wide
address arithmetic. After fusion, `score[i]` is thread-local and needs
no inter-thread transfer. Preserve Q/K packed-NHD indexing and both `query_key`
zero-fill predicates. Keep the last-tile invalid-key mask in `consumer` unchanged.

Caller-stream order originally makes all producer stores visible before the
consumer launch. Fusion replaces that edge with each thread's program order:
complete QK before masking and softmax, and finish score conversion before the
next key tile overwrites `score`. Retain both warp synchronization points in
`reduce` and both CTA barriers surrounding probability consumption and reuse.
Keep `attention` followed by `attention_epilogue` on the caller stream, including
scratch lifetimes and launch-error checks. No new barrier is needed because
score ownership stays within each thread.

Keep the FP32 score arithmetic and rounding. The removed global FP32 roundtrip
must not introduce a conversion or change FMA grouping. Retain descending key
tiles, increasing dimension/key scalar FMA loops, online maximum/sum updates,
FP16 probability rounding, normalization, output rounding, and all existing
compiler controls. Do not substitute MMA or a different softmax algorithm.

Use the Kernel evaluator on the unchanged problem for correctness and timing.
Inspect generated code to confirm that QK executes in `attention` and the
separate score producer and materialization are absent.

## Example configuration

Preserve CUDA on `nvidia-sm90a-cuda13`, packed contiguous FP16 Q/K/V/output
`[16384,64,128]`, strides `(8192,128,1)` in elements, and int32 offsets `[9]`.
The workload contains eight sequences of 2048 tokens; retain device-side ragged
bounds and the packed-token upper bounds rather than specializing to that length.
Q/K/V row addresses are `token * kRow + head * kDim + component`.

Keep `kM=128`, `kN=176`, `kDim=128`, `kThreads=256`, `kGroupSize=128`, and
`kWarpSize=32`. Launch both remaining kernels with `grid=(8640,1,1)`,
`block=(256,1,1)`, zero dynamic shared memory, and the original launch bounds.
Each CTA owns one query tile and head; no SM assignment is fixed.
The scheduler skips excess CTAs. Keep the two retained math groups and the
existing global probability, reduction, epilogue, and LSE scratch.

Each thread owns 88 QK scores and 64 PV outputs. For fragment element `i`,
`lane=tid%32`, `warp=(tid%128)/32`, and `wg=tid/128`:

```cuda
row = wg * 64 + warp * 16 + lane / 4 + (i % 4) / 2 * 8;
col = (i / 4) * 8 + (lane % 4) * 2 + i % 2;
```

These are preserved fragment coordinates, not MMA requirements. Retain the
K-major probability panel mapping, grouped row reductions, scalar conversions
and output stores, redundant per-score offset multiplication, and per-component
reciprocal calculation. Keep `kLog2Scale=0.127517432f`, `kScale=0.0883883461f`,
and compile flags `-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.

The deoptimized scratch uses `kMaxQTiles=135`, `kMaxKeyTiles=94`, and
`sizeof(ScoreState)=88*sizeof(float)`: 94 slabs of 778,567,680 bytes (742.5 MiB),
73,185,361,920 bytes total. Slabs permit caching-allocator reuse without requiring
one enormous contiguous block. Only valid query CTAs
and key tiles are written and read. Removing this allocation requires no host
read of sequence offsets. These sizes describe this replay, not fusion prerequisites.

# Precondition

- Data types: the local score must equal the value the consumer previously read
  after materialization. Preserve its representation and required arithmetic
  order and rounding; otherwise softmax sees different inputs. No particular
  operand dtype is intrinsic to fusion.
- Layout: the producer and consumer must assign each score to the same thread,
  with known input strides and bounds. Direct local consumption has no exchange
  to correct mismatched ownership. No extra contiguity or alignment is needed
  beyond the existing input accesses.
- Storage: the materialized global scores must be internal, without other
  readers or observable uses; deletion would otherwise remove required results.
  The original Q/K operands must remain accessible where scores are recomputed.
- Pipeline: operand producers must complete and make their values visible before
  fused computation, and inputs must remain stable through their reads. Compute
  each score before consuming it, finish consumers before local state reuse,
  and preserve the existing completion/visibility barriers before retained
  scratch reads and reuse. An intervening cross-CTA dependency cannot rely on
  the deleted launch boundary, since thread program order cannot replace it.
  No particular stage count is required.
- Hardware: the device already supports the retained CUDA arithmetic. The
  combined per-thread live state must fit legal launch resource limits, allowing
  compiler spills; otherwise the fused kernel cannot launch. Fusion adds no
  instruction-family or shared-memory requirement.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace only this block inside `consumer` in `solution/attention.cuh`.
Keep the surrounding loop, last-tile mask, and softmax/PV sequence. Remove the
producer, allocation, field, and type listed in Overview; the launch snippet
shows the corresponding ordering change.

## Before

```cuda
// The preceding launch publishes this thread's complete score fragment.
const auto& scores = p.scores[n][size_t(blockIdx.x) * kThreads + tid];
#pragma unroll
for (int i = 0; i < kQkRegs; ++i) score[i] = scores.value[i];
```

```cuda
check(cudaLaunchKernelEx(&config, attention_scores, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_epilogue, p));
check(cudaGetLastError());
```

## After

```cuda
// Compute this thread's scores directly before masking and softmax.
query_key(score, p.q + qstart * kRow + work.z * kDim, p.k + base,
          qvalid, valid, wg);
```

```cuda
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_epilogue, p));
check(cudaGetLastError());
```
