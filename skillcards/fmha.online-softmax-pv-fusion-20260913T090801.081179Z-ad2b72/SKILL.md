---
skill_id: fmha.online-softmax-pv-fusion
intent: Fuse online softmax with PV accumulation to consume probability tiles immediately.
preconditions:
- 'Data types: eliminating intermediate stores must preserve probability rounding,
  scale and denominator values, and accumulator operation order; otherwise fusion
  changes the numerical result. No particular operand dtype is intrinsic to fusion.'
- 'Layout: each probability producer and all its PV consumers must belong to one CTA
  with known element ownership and matching tile traversal; CTA barriers cannot publish
  data to another CTA. Fusion adds no input-contiguity or alignment requirement.'
- 'Storage: probabilities, scales, and denominators are internal global intermediates
  with no remaining external reader; scores and values remain accessible to the fused
  computation. Removing externally observed intermediates would lose required data.'
- 'Pipeline: input producers must complete and publish before reads, and inputs must
  remain stable through consumption. Every participating CTA thread must reach probability
  publication and reader-completion barriers; all readers must finish before buffer
  reuse. No dependency may require grid-wide completion at the deleted launch boundary.'
- 'Hardware: CUDA CTA barriers and sufficient device memory for one probability tile
  per launched CTA alongside retained allocations; the fused live thread state must
  permit a legal launch, with compiler spills allowed. Insufficient capacity or barrier
  support makes this implementation invalid.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse the online-softmax producer and its PV consumer in
`solution/attention.cuh`. Compute a tile's softmax update, rescale the running
output, round probabilities, publish one reusable probability tile, and consume
it immediately. This removes the `attention_values` launch and materialized
per-key-tile probabilities, scales, and final denominators. Keep QK and the output
epilogue as separate launches.

## Replay edits

1. In `solution/ops.cuh`, remove `SoftmaxState`, its capacity assertion, and
   `prob_state`. Remove only `denominator` from `EpilogueState`. Add
   `__half* prob;` after `Params::output`. Retain `ScoreState`, `Params::scores`,
   `kMaxKeyTiles`, reductions, and all arithmetic helpers.
2. In `solution/kernel.cu::kernel`, allocate the probability tensor and set its
   pointer as shown below. Keep it alive through every launch. Remove only the
   `attention_values` launch and its following `cudaGetLastError()` check.
   Retain the same caller stream, grid, block, and launch attributes. The launch
   order becomes `attention_scores`, `attention`, `attention_epilogue`.
   Existing `sizeof(EpilogueState)` allocation arithmetic picks up its smaller
   record automatically; retain the per-key-tile score allocations.
3. In `consumer`, add `const int wg = tid / kGroupSize;` and
   `__half* const prob_tile = p.prob + size_t(blockIdx.x) * kProbTileElems;`
   after the uniform invalid-work return. Add `float out[kPvRegs];` and initialize
   every element to `0.f`, with `#pragma unroll`, before the key-tile loop.
   At each loop iteration, define
   `base = (kstart + n * kN) * kRow + work.z * kDim` and
   `valid = length - n * kN`, both `const int`.
4. Keep the score loads and last-key mask. Replace the materialization block
   beginning at `softmax(...)` with the fused block below. The former barrier
   protecting aliased score storage disappears because scores are no longer
   overwritten. The two shown CTA barriers publish probabilities and prevent
   their reuse before every PV reader finishes.
5. Keep the final denominator reduction and LSE calculation. Replace the final
   statistics-store block with the shown normalization and result stores.
   Delete the entire `attention_values` device function. Retain `attention`,
   `attention_scores`, `attention_epilogue`, and every math helper unchanged.

Each CTA owns its probability tile; `store_prob` writes the same K-major panel
layout consumed by `prob_value`:
`panel<kM>(row,key) = (key / kInputPanelCols) * kM * kInputPanelCols
+ row * kInputPanelCols + key % kInputPanelCols`.
There is no new thread mapping. Per-thread scale and denominator values stay
local; probability exchange remains in global memory. No shared-memory staging,
vectorization, MMA, scheduler change, or QK fusion accompanies this change.

Maintain the uniform `work.w >= kBatch` return before barriers. All remaining
threads, including padded query rows, participate. Keep the last-key score mask,
PV's invalid-key zero fill, and epilogue query-row store guard. Keep decreasing
`n = last..0`, increasing-key scalar PV `fmaf`, the two-row online maximum/sum
recurrence, reduction grouping, FP16 probability rounding before PV, per-tile
output rescaling, and normalization. In particular, do not replace these online
probabilities with a once-normalized full attention matrix.

Use the existing kernel evaluator on the unchanged problem. Inspect generated
code for a fused PV loop in `attention` and no `attention_values` entry; preserve
the scalar compiler controls already in the bundle.

## Example configuration

This replay uses CUDA on `nvidia-sm90a-cuda13`, packed contiguous FP16
`q/k/v/output[16384,64,128]` with element strides `(8192,128,1)`, int32 offsets
for eight sequences, FP32 score/softmax/PV arithmetic, and FP16 rounded
probabilities. Preserve `kLog2Scale = 0.127517432f` and `kScale = 0.0883883461f`.

Keep query/key tiles `kM = 128`, `kN = 176`; 256 threads per CTA; two 128-thread
groups; 32-thread warps; four lanes per row; two rows per lane; eight-column
probability panels; and 88 score, 44 packed probability, and 64 output entries
per thread. `kMaxQTiles = 135` and `kMaxKeyTiles = 94` retain the conservative
ragged-sequence bounds. Grid is `kMaxQTiles * kHeads = 8640`, block is `kThreads`,
dynamic shared memory is zero, and launch bounds remain `(kThreads, 1)`.

Before fusion, a CTA's 90,112-byte score region in each slab is reused as a
47,104-byte `SoftmaxState`: 45,056 probability bytes followed by 2,048 scale
bytes. The producer barrier protects all score reads before this overwrite.
The PV launch reads these immutable records and the denominator fields published
by the preceding launch. Removing this materialization also removes that storage
alias; keep the 94 score slabs and their total 73,185,361,920 bytes unchanged.

After fusion, allocate `8640 * 22528` half elements: 389,283,840 bytes, one
45,056-byte reusable probability tile per CTA. `EpilogueState` returns from
272 to 264 bytes per thread. Keep the global reduction scratch, separate QK and
epilogue launches, scalar conversions/stores, repeated reciprocal and score-offset
compiler controls, and all existing numerical helpers. Preserve compile flags
`-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG` and SM90a
compilation. None of these fixed sizes or retained instruction choices defines
the general fusion prerequisites.

# Precondition

- Data types: the values formerly reloaded must match the locally retained
  probabilities, scales, and denominators. Preserve the probability conversion
  and accumulator operation order; skipping rounding or reassociating the online
  recurrence changes results. Fusion itself requires no particular operand dtype.
- Layout: known ownership and matching tile traversal must place each producer
  and all its PV consumers in the same CTA. Otherwise the proposed CTA barriers
  cannot supply a consumer's data. Fusion imposes no additional input contiguity
  or alignment requirement.
- Storage: the global probabilities, scales, and denominators must be internal
  intermediates with no other remaining reader, and the scores and values must
  stay accessible. Eliminating observable records or unavailable inputs would
  break the operator.
- Pipeline: prior producers must publish inputs before reads; those inputs remain
  stable until consumed. Every participating thread reaches probability
  publication and reader-completion barriers, including boundary threads.
  Writers complete before PV reads, and all readers complete before tile reuse.
  A dependency requiring grid-wide producer completion at the deleted launch
  boundary cannot be satisfied by CTA barriers. These are ordering constraints, not stage counts.
- Hardware: CUDA CTA barriers must support the exchange. Available device memory
  must cover `launched_CTAs * probability_tile_elements * sizeof(probability)`
  alongside retained allocations. The fused live thread state must fit legal
  launch resource limits; compiler spills are permitted. No new arithmetic
  instruction family is needed, but insufficient memory or unsupported barriers
  prevents this implementation.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

These excerpts are from `consumer`; the surrounding score loads, key mask,
denominator reduction, and LSE calculation remain unchanged.

```cuda
softmax(score, maximum, sum, scale, s);
convert(score, prob);

// Finish score reads before writing the aliased materialized tile.
__syncthreads();
auto& state = prob_state(p, n);
store_prob(prob, state.prob);
#pragma unroll
for (int r = 0; r < 2; ++r) state.scale[tid][r] = scale[r];
```

```cuda
// After final denominator reduction and LSE calculation.
auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + tid];
#pragma unroll
for (int r = 0; r < 2; ++r) {
    state.lse[r] = sum[r];
    state.denominator[r] = denominator[r];
}
```

The separate `attention_values` loop reloads each tile and scale, calls
`rescale` and `prob_value`, then reloads denominators, normalizes, and stores
`state.out`. Delete that function and its launch after moving those operations.

## After

Add this invocation-local allocation in `kernel` before reduction allocation;
set `p.prob` after `Params p{}` and before launch:

```cuda
const auto prob = torch::empty(
    {int64_t(kMaxQTiles) * kHeads * kProbTileElems}, q.options());

// After Params construction and output pointer setup.
p.prob = reinterpret_cast<__half*>(prob.data_ptr<at::Half>());
```

With `wg`, `prob_tile`, initialized `out`, `base`, and `valid` added as specified
above, replace the first Before block:

```cuda
softmax(score, maximum, sum, scale, s);
rescale(out, scale);
convert(score, prob);
store_prob(prob, prob_tile);
__syncthreads();
prob_value(out, prob_tile, p.v + base, valid, wg);

// Finish all readers before the next tile overwrites probabilities.
__syncthreads();
```

Replace the second Before block:

```cuda
normalize(out, denominator);

// Publish normalized outputs and LSE for the existing epilogue launch.
auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + tid];
#pragma unroll
for (int i = 0; i < kPvRegs; ++i) state.out[i] = out[i];
#pragma unroll
for (int r = 0; r < 2; ++r) state.lse[r] = sum[r];
```
