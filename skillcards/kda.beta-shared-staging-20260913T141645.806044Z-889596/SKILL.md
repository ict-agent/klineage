---
skill_id: kda.beta-shared-staging
intent: Reuse beta logits within each CTA through shared-memory staging.
preconditions:
- 'Data types: staging must reproduce each direct load''s conversion and rounding
  before activation; otherwise consumers receive different numerical operands. No
  particular dtype is intrinsic to shared reuse.'
- 'Layout: multiple consumers within a CTA address the same logits through a known
  index map; cooperative writers must cover every consumed index, with valid source
  bounds and naturally aligned scalar accesses. Contiguous global rows are unnecessary
  for a gather.'
- 'Storage: logits are globally readable and unchanged during the CTA''s uses; a shared
  snapshot cannot reproduce intervening updates or communication across CTAs.'
- 'Pipeline: input production must complete before staging; every consumer must observe
  completed cooperative writes, and all reads must finish before buffer reuse. CTA-wide
  barriers require uniform participation.'
- 'Hardware: CUDA shared memory and CTA barriers are required; shared capacity must
  cover existing live storage plus the staged logit count times its stored element
  size, including alignment padding.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Stage beta logits once per CTA, then reuse the rounded values across its
matrix-element consumers. Apply this in `solution/prepare.cuh` and
`solution/recurrence.cuh`; shared declarations are in `solution/native.cuh`.
Only the beta loading path changes. Keep sigmoid evaluation at each consumer.

Rename `PrepareShared::beta_reserved` and `InputShared::beta_reserved` to
`beta`, removing their reservation comments. Add the cooperative loops below
at the end of `load_prepare` and `load_input`. Each thread owns indices
`threadIdx.x + n * block_thread_count`. Index `i` represents a logical
head-major token offset, gathered from token-major global logits. Store the
converted value in contiguous shared `beta[i]`.

In `prepare`, replace `load_beta(args.beta,tile,head,r)` with `s.beta[r]`.
In `recur_tile`, replace the direct load with
`ld_scalar(in.beta+group+(j%2)*kHalfChunk)` inside the existing element loop.
Retain this volatile shared load so activation-factor reuse stays absent.
Restore `recur_tile` to its three-argument signature and call, then remove
`load_beta` from `native.cuh`. Keep `ld_global_scalar`: gate preparation still
uses it. No compiler-flag change is needed. The direct helper's volatile global
load prevents compiler elimination of repeated loads before this optimization.

The existing `__syncthreads()` after each loader publishes the cooperative
writes before any consumer. Preserve every barrier and warp synchronization.
Beta storage is written once per CTA and remains unchanged until CTA completion;
no rotation or intermediate reuse is introduced. Input production and consecutive
recurrence launches remain ordered on the caller stream.

Retain the gather formula and zero fill in the snippets. In this workload,
consumers use only rows below `kChunk`; staging also fills the remaining reserved
slots. The logical head rollover in those extra slots is intentional. Never
read global memory when `src >= kTokens*kHeads`. Preserve FP32-to-BF16 rounding
before `bf_float` and sigmoid, including the recurrence's additional BF16
rounding of the sigmoid result. Keep all other arithmetic and its order.

## Example configuration

Replay with batch 1, 4096 tokens, 96 heads, dimension 128, chunk size 16, and
256 chunks. Global beta is contiguous FP32 `[1,4096,96]`; its token stride is
`kHeads*sizeof(float)` bytes. Stage 32 BF16 logits, with 128-byte member
alignment, in each loader. The reserved slots already provide this storage.
`PrepareShared`, `InputShared`, and `RecurShared` stay 41856, 18048, and
124672 bytes. Preserve their other fields and all static shared scratch.

Preparation uses `grid=(kTiles,kHeads)`, 256 threads, and its existing
`__launch_bounds__(kPrepareThreads,8)`. Recurrence uses `grid=(1,kHeads)`,
256 threads, and one launch per chunk. Keep dynamic shared bytes equal to the
respective shared-structure size. A preparation thread owns matrix element
`r=tid/8%kChunk`, `c=tid%8+(tid/(kChunk*8))*8`; only `r>c` consumes beta.
Each recurrence warp owns 16 value columns. Its lane group is `lane/4`, and
fragment `j` consumes row `group+(j%2)*(kChunk/2)` for both pair elements.

Retain BF16 matrix operands, FP32 accumulators, `mma.sync` products and fragment
adapters, scalar shared transfers, shared operand/state reuse, normalization
reduction grouping, triangular inversion, repeated gate-prefix calculation,
per-element sigmoid/decay evaluation, and BF16 carry between launches. Keep the
CUDA tensor ABI, SM90 target, existing compile flags, bounds checks, and
caller stream. These settings describe this replay, not general staging
prerequisites. Check correctness and latency using the existing evaluator and
unchanged problem; inspect generated loads to confirm the transfer path.

# Precondition

- Data types: the staged representation must reproduce the direct load's
  conversion and rounding before activation. Moving a conversion across sigmoid
  changes its input and is invalid. Shared reuse itself imposes no particular
  dtype; use a representation with the required consumer values.
- Layout: consumers within a CTA must reuse logits with a known mapping from
  source addresses to consumer indices. Cooperative ownership must initialize
  every consumed slot; missing slots yield uninitialized reads. Source bounds
  must be handled, and scalar addresses must satisfy their types' natural
  alignment. Global contiguity is unnecessary because loading can gather.
- Storage: the source logits must be globally visible and stable during the
  CTA's uses. Staging takes one snapshot, so it cannot reproduce intermediate
  source updates. Consumers in different CTAs require separate staging.
- Pipeline: source production must finish before the cooperative loads;
  writes must complete and become visible before consumers read. All readers
  must finish before overwriting the buffer. All CTA threads must reach each
  CTA-wide barrier, otherwise synchronization can fail. No fixed stage count
  is required.
- Hardware: CUDA shared memory and CTA barriers implement the shared snapshot
  and its visibility. Total capacity must cover existing live storage plus
  `staged_count*sizeof(staged_element)` and alignment padding, counting any
  reserved storage only once. Exceeding the device limit prevents the launch.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

The two shared structures reserve unused slots:

```cuda
alignas(128) BF16 beta_reserved[kBetaElems];
```

`load_prepare` and `load_input` do not load beta. The common direct helper and
its two consumer sites are:

```cuda
__device__ __forceinline__ BF16 load_beta(
    const float* beta, int tile, int head, int row) {
    const int src = head * kTokens + tile * kChunk + row;
    if (src >= kTokens * kHeads) return BF16(0);

    const int gathered = (src % kTokens) * kHeads + src / kTokens;
    return BF16(ld_global_scalar(beta + gathered));
}

// prepare: preserve the triangular mask and arithmetic.
s.l[r*kChunk+c] = r <= c ? 0.0f
    : s.l[r*kChunk+c]*sigmoid(bf_float(load_beta(args.beta,tile,head,r)));

// recur_tile: inside the existing j/e loops.
BF16 beta = BF16(sigmoid(bf_float(
    load_beta(beta_logits,tile,head,group+(j%2)*kHalfChunk))));
result.h[e] = (v.h[e]-r.h[e])*beta;
```

The direct recurrence interface is:

```cuda
__device__ __forceinline__ void recur_tile(
    RecurShared& s, InputShared& in, BF16* out,
    const float* beta_logits, int tile, int head);

if (warp < kComputeWarps)
    recur_tile(s,s.input,s.out,args.beta,tile,head);
```

## After

Rename each reserved member without changing its size or alignment:

```cuda
alignas(128) BF16 beta[kBetaElems];
```

Append these loops inside the respective loaders. Both already define `tid`,
`token=tile*kChunk`, and have `head` and `args` parameters.

```cuda
// End of load_prepare: one shared logit serves several matrix elements.
for (int i = tid; i < kBetaElems; i += kPrepareThreads) {
    const int src = head * kTokens + token + i;
    const int gathered = (src % kTokens) * kHeads + src / kTokens;
    s.beta[i] = src < kTokens * kHeads ? BF16(args.beta[gathered]) : BF16(0);
}

// End of load_input: share logits across value columns and warps.
for (int i = tid; i < kBetaElems; i += kRecurThreads) {
    const int src = head * kTokens + token + i;
    const int gathered = (src % kTokens) * kHeads + src / kTokens;
    in.beta[i] = src < kTokens * kHeads ? BF16(args.beta[gathered]) : BF16(0);
}
```

Keep each loader's existing following CTA barrier. Replace the consumer sites:

```cuda
// prepare: the row index and triangular mask stay unchanged.
s.l[r*kChunk+c] = r <= c ? 0.0f
    : s.l[r*kChunk+c]*sigmoid(bf_float(s.beta[r]));

// recur_tile: retain per-element activation and BF16 arithmetic.
BF16 beta = BF16(sigmoid(bf_float(
    ld_scalar(in.beta+group+(j%2)*kHalfChunk))));
result.h[e] = (v.h[e]-r.h[e])*beta;
```

Remove the now-unused `load_beta` helper. Keep the recurrence body and restore
its interface and call:

```cuda
__device__ __forceinline__ void recur_tile(
    RecurShared& s, InputShared& in, BF16* out);

if (warp < kComputeWarps)
    recur_tile(s,s.input,s.out);
```
