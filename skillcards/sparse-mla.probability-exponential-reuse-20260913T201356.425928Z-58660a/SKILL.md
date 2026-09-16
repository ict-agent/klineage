---
skill_id: sparse-mla.probability-exponential-reuse
intent: Retain per-logit exponentials for reuse across probability publications.
preconditions:
- 'Data types: both uses require the same exponential with identical argument bits
  and arithmetic; retain its original result precision so reuse adds no rounding.'
- 'Layout: each repeated use belongs to the same thread and score element as its producer;
  no additional contiguity, alignment, or warp mapping is needed for private-array
  reuse.'
- 'Storage: per-thread storage can retain the first result, and the original logits
  in the reused array have no remaining use except recomputing that result; otherwise
  overwriting them destroys live data.'
- 'Pipeline: the first evaluation precedes reuse in the same thread, and its slot
  remains unmodified until the last reader; thread-private visibility requires no
  cross-thread publication.'
- 'Hardware: no additional feature or capacity is required; existing scalar instructions
  and private-array storage already support the computation and retained value.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Retain the first per-logit exponential in `p[i]` and reuse it when consumer group 0 publishes probabilities at the peer's updated maximum. This removes the repeated exponential evaluation in `consume<0>` while preserving the original rescaling operation.

Edit only `solution/attention.cuh`. In `softmax`, replace the first Before block with its After block: both groups leave FP32 exponentials in `p`, and BF16 probabilities plus the FP32 sum still derive from those same values. In group 0's `consume` branch after `sync<Barrier::Max1>()`, replace the second block as shown. Remove `local_max` and its assignments; remove the two exponential recomputations, then multiply cached `p` values by the unchanged `rescale_exp(delta[row])`.

Each thread owns the same score-array elements across both publications. The original logits become dead after their first exponential calculation. The scalar FMA loops, score mask, local PV, peer-maximum wait, probability stores, and every barrier stay unchanged. Group 1 retains its existing behavior. No layout, allocation, ABI, launch, or synchronization change is needed.

Preserve `BF16(exp2(logit*scale_log2-local_max) * exp2(local_max-peer_max))` with the original FP32 intermediate and BF16 round-to-nearest conversion. Do not combine the exponentials, replace the original local maximum with the peer maximum inside the first exponential, or add an intermediate BF16 rounding. Invalid scores remain negative infinity and yield zero weight.

The deoptimized recomputation uses the existing `rescale_exp` helper because its volatile argument forces another exponential instruction. Remove only these new calls when applying the optimization; retain the helper and all existing delta-rescaling calls. Inspect generated code for the removed repeated exponentials and evaluate the unchanged problem through `klineage.harness.evaluate`.

## Example configuration

The fixed workload is `TOKENS=8192`, `HEADS=128`, `QK_DIM=576`, `VALUE_DIM=512`, `TOPK=2048`. Queries/KV/output are BF16, indices int32, and scores, weights, sums, maxima, and output accumulators FP32. Preserve scale `0.1352337788608801f`, `kScaleLog2`, fast math, all compile flags, the oracle, and tolerances.

Keep 64-wide tiles, 32 score values and 128 output accumulators per lane, two heads per lane, and 256 threads split into two groups of 128. `p` uses the existing four-lane row mapping: each lane owns adjacent pairs, advances by four array entries, and processes two rows. Group 0 uses its first exponential for local BF16 probabilities and later reuses it for the rescaled publication. Group 1 needs no new cache operation.

Retain 32 selected-row blocks processed in pairs, QK group 0's feature order 0–8, group 1's order 4–8 then 0–3, and increasing selected-row order in PV. Keep two global probability buffers, two gathered KV buffers, their layouts/copy order, scalar volatile transfers, and online softmax/consumer scheduling. Per CTA, probability scratch is 16384 bytes, KV scratch 147456 bytes, and shared storage 1920 bytes. The grid remains 16384 CTAs with `(256,1,1)` blocks, `(1,1,1)` clusters, launch bounds `(256,1,1)`, and caller-stream execution. No host or helper-header edits are needed.

# Precondition

- Data types: the repeated expression must require exactly the same exponential result, with identical argument bits and arithmetic. Keep its original precision in the cache; extra rounding would change the later product.
- Layout: producer and later consumer must be the same thread accessing the same score element. A private array cannot supply another thread's value. Reusing that array adds no alignment, contiguity, or warp-size requirement.
- Storage: private storage must retain the first result until reuse. For this in-place replacement, original logits must be dead apart from recomputation; overwriting an otherwise live logit would be incorrect.
- Pipeline: same-thread execution must finish the first evaluation before its later read, and the slot must not be overwritten before that read completes. Thread-private visibility needs no cross-thread publication or new barrier.
- Hardware: no additional feature or storage capacity is needed. Existing scalar instructions and private-array storage already provide the computation and space for the retained result.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

Inside `softmax`'s innermost score loop:

```cuda
const float first = exp2f(p[i] * kScaleLog2 - next[row]);
const float second = exp2f(p[i + 1] * kScaleLog2 - next[row]);

// Group 0 retains logits and recomputes weights for its next publication.
if constexpr (Group == 1) {
    p[i] = first;
    p[i + 1] = second;
}
s[i] = __float2bfloat16_rn(first);
s[i + 1] = __float2bfloat16_rn(second);
sum += first + second;
```

Inside group 0's `consume`, immediately after `sync<Barrier::Max1>()`:

```cuda
float next[2], delta[2], local_max[2];
load_stats(sm.maximum + lane / 4, next);
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
    local_max[row] = m[row];
    delta[row] = m[row] - next[row];
    m[row] = next[row];
}
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
    for (int i = row * 2; i < kScoreRegs; i += 4) {
        // Volatile exponential arguments prevent compiler reuse.
        const float first = rescale_exp(p[i] * kScaleLog2 - local_max[row]);
        const float second = rescale_exp(p[i + 1] * kScaleLog2 - local_max[row]);
        s[i] = __float2bfloat16_rn(first * rescale_exp(delta[row]));
        s[i + 1] = __float2bfloat16_rn(second * rescale_exp(delta[row]));
    }
}
```

## After

Replace the first block with in-place caching:

```cuda
p[i] = exp2f(p[i] * kScaleLog2 - next[row]);
p[i + 1] = exp2f(p[i + 1] * kScaleLog2 - next[row]);
s[i] = __float2bfloat16_rn(p[i]);
s[i + 1] = __float2bfloat16_rn(p[i + 1]);
sum += p[i] + p[i + 1];
```

Replace the second block with reuse of the cached FP32 weights:

```cuda
float next[2], delta[2];
load_stats(sm.maximum + lane / 4, next);
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
    delta[row] = m[row] - next[row];
    m[row] = next[row];
}
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
    for (int i = row * 2; i < kScoreRegs; i += 4) {
        s[i] = __float2bfloat16_rn(p[i] * rescale_exp(delta[row]));
        s[i + 1] = __float2bfloat16_rn(p[i + 1] * rescale_exp(delta[row]));
    }
}
```
