---
skill_id: kda.chunk-decay-materialization
intent: Materialize chunk gate exponentials once for reuse across consumers.
preconditions:
- 'Data types: repeated exponentials must use identical argument bits and arithmetic
  semantics; existing scalar slots must hold their results without extra rounding,
  or reuse changes numerical behavior.'
- 'Layout: consumers must identify the same chunk/channel argument through known indices,
  with one designated writer per valid slot; otherwise values can be confused or overwritten.
  No extra contiguity or alignment beyond legal scalar accesses is required.'
- 'Storage: chunk sums already occupy shared-memory and global-workspace slots accessible
  to their consumers; all later users of those slots must permit replacing the sum
  with its exponential, since the in-place cache discards the sum.'
- 'Pipeline: sum production must finish before conversion, conversion must be visible
  before reads or export, and exported data must be ready before downstream consumption.
  Slots must remain unchanged until all readers finish; otherwise reuse observes incomplete
  or overwritten values.'
- 'Hardware: no additional device feature is required beyond existing scalar exponentiation,
  CTA barriers, and shared/global storage. Each reused storage region must hold N
  live results at their result element size; insufficient capacity prevents materialization.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Compute each chunk/channel gate exponential once and reuse it in decay
preparation and recurrent state updates. The deoptimized kernel carries the raw
log2 gate sum in `gt` and evaluates `exp2_fast` independently at both consumers.

Apply these three edits together:

1. In `solution/prepare.cuh`, `prepare`, replace the comment between the two
   barriers immediately after `s.gt[tid] = sum` with the guarded conversion shown
   below. Thread `tid < kDim` owns channel `tid`. Both barriers remain collective.
2. In the same function's `kr` assignment, remove the consumer's `exp2_fast`.
   The existing scalar gather now reads the materialized exponential into `rgt`.
3. In `solution/recurrence.cuh`, `recur_tile`, remove `exp2_fast` from both
   `decay0` and `decay1` assignments. Keep their indices and all subsequent
   multiplication, addition, and BF16 rounding unchanged.

`store_prepare` exports `s.gt[i]` to `args.gt[(head*kTiles+tile)*kDim+i]`.
`load_input` copies the same slot to `in.gt[i]`. These copies need no edits:
the existing FP32 path now transports exponentials instead of log2 sums.
The separate `s.g` matrix continues to hold raw prefix sums.

Each preparation CTA owns one `(tile, head)`. Its consumer mapping is
`r=m*8+warp`, `c=n*64+(lane/4)*8+(lane%4)*2+j`.
Every row reuses the same channel's chunk decay. In recurrence, warp `w` owns
value columns starting at `w*kChunk`; `group=lane/4` selects key channels
`m*kChunk+group` and `m*kChunk+group+8`. Value warps reuse those same key decays.

Retain the barrier before conversion and the barrier before gathers. Keep
`prepare` and `recurrence` launches on the caller stream so workspace publication
finishes before recurrence loads. Keep the recurrence barrier after `load_input`
and the barriers completing computation/output before the next chunk overwrites
`InputShared`. No launch, allocation, or ownership change is needed.

Use the existing `exp2_fast`, which evaluates `ex2.approx.ftz.f32`. Preserve gate
prefix addition order, FP32 exponential results, and the later BF16 conversion
in `kr`. Retain the two separate BF16 multiplications in `rk * ie * BF16(rgt)`;
combining exponent arguments would change rounding. Recurrence still multiplies
the BF16 state converted to FP32 by an FP32 decay before the existing update.
Preserve the full problem, oracle, workload, tolerances, and compiler flags.

## Example configuration

Replay uses batch 1, 4096 tokens, 96 heads, dimension 128, 16-token chunks,
and 256 chunks per head. Both kernels use 256 threads and 32-thread warps.
Preparation launches `dim3(kTiles,kHeads)`; recurrence launches `dim3(1,kHeads)`
and processes chunks sequentially with eight value warps.

`gt` has 128 contiguous FP32 elements per chunk/head, a 512-byte row.
Inputs remain contiguous BTHD; final state remains value-key ordered.
Every tile is full. Preserve the existing beta boundary zero fill and ABI guards.
For adapted dimensions, guard every owner/gather index without bypassing barriers.

Retain dynamic shared-memory sizes: `PrepareShared=42368`, `InputShared=18048`,
and `RecurShared=124672` bytes; preserve all fields and offsets, including the
existing transpose scratch allocation. Workspace sizing remains
`kHeads*(kTiles+1)*kWorkspaceTile+128` bytes, with `kWorkspaceTile=13824`.

Retain BF16 operands/intermediates, FP32 accumulators, `m16n8k16` MMA and fragment
adapters, scalar shared loads/stores, separate recurrence products, chunked
triangular inversion, shared staging, and normalization reduction grouping.
Keep SM90/CUDA13 compilation, `-O3`, `--use_fast_math`, launch bounds, and the
existing register-usage setting. These are replay settings, not cache prerequisites.

Validate with the existing Kernel evaluator on the supplied workload. Inspect
generated code to confirm that the shared `gt` store receives an exponential
and the recurrence update no longer executes exponentials for its decays.
Keep evidence outside this card.

# Precondition

- Data types: repeated calls must receive identical bits under the same
  exponential semantics. Existing slots must represent the result without an
  extra rounding step; otherwise moving evaluation changes what consumers use.
  The technique does not depend on the dtype of unrelated matrix operands.
- Layout: known chunk/channel indices must map every reuse to the same argument,
  and each valid slot needs one writer. Ambiguous indexing or multiple writers
  can select another channel's decay. Ordinary scalar alignment suffices;
  additional vector alignment or contiguity is unnecessary.
- Storage: the preceding implementation already provides shared and global slots
  for chunk sums and access paths to their users. All later users of these slots
  must accept the exponential representation, because overwriting the slot
  destroys its raw sum. A raw sum needed elsewhere requires separate storage.
- Pipeline: finish sum production before conversion, publish conversion before
  local reads or export, and finish export before downstream consumption.
  Keep slots immutable through the last read and finish all reads before buffer
  reuse. Without these dependencies, a consumer can read an unfinished sum,
  stale result, or another chunk's value. No particular stage count is required.
- Hardware: no additional device feature is introduced. Existing scalar
  exponentiation, CTA barriers, and shared/global scalar storage suffice.
  In each storage region, capacity must satisfy
  `available_bytes >= N_live_results * sizeof(result)`; the cache cannot
  materialize all live values otherwise. Reusing eligible slots adds no storage.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The three excerpts are separate replacement sites; surrounding loops, arguments,
gathers, copies, and barriers remain as supplied.

## Before

```cuda
// prepare: immediately after the guarded gate-prefix accumulation.
__syncthreads();
// Keep chunk sums in log2 form; consumers evaluate their own decay.
__syncthreads();

// prepare: inside the existing decay-preparation loop.
kr.h[j] = rk[t][j] * ie * BF16(exp2_fast(rgt[t][j]));

// recur_tile: inside the existing key-block update loop.
float decay0 = exp2_fast(in.gt[m*kChunk+group]);
float decay1 = exp2_fast(in.gt[m*kChunk+group+8]);
```

## After

```cuda
// prepare: publish one exponential per chunk/channel for all consumers.
__syncthreads();
if (tid < kDim) s.gt[tid] = exp2_fast(s.gt[tid]);
__syncthreads();

// prepare: gathered rgt now holds the shared exponential.
kr.h[j] = rk[t][j] * ie * BF16(rgt[t][j]);

// recur_tile: the existing workspace copies supply the cached decay.
float decay0 = in.gt[m*kChunk+group];
float decay1 = in.gt[m*kChunk+group+8];
```
