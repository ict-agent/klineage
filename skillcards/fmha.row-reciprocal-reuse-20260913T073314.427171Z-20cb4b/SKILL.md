---
skill_id: fmha.row-reciprocal-reuse
intent: Reuse each softmax row reciprocal across output components.
preconditions:
- 'Data types: consumers sharing a denominator must accept the same reciprocal approximation,
  zero/NaN handling, and subsequent multiplication; reuse must preserve required rounding.'
- 'Layout: each thread must identify multiple output components using an identical
  row denominator; this association determines which reciprocal may be shared. No
  additional contiguity or alignment is required.'
- 'Storage: completed row denominators must be readable by their normalizing thread;
  existing output and denominator storage can remain in place because reuse needs
  no cross-thread exchange.'
- 'Pipeline: denominator producers must complete and publish their results before
  normalization, and denominator values must remain stable until all consumers finish;
  preserve existing visibility and buffer-reuse ordering to avoid incomplete or overwritten
  values.'
- 'Hardware: no additional feature or capacity is required; the transformation reuses
  existing scalar reciprocal arithmetic and replaces each live denominator with its
  reciprocal without adding a buffer.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reuse one reciprocal of each completed softmax row sum across that thread's
output components. This removes repeated reciprocal evaluation from final
attention normalization while retaining each output multiplication.

In `solution/attention.cuh`, delete `normalize`. At the end of `consumer`,
replace the local `denominator` array, final reduction loop, and
`normalize(out, denominator)` call with the After snippet. Retain the following
`epilogue(p, work, out, sum)` call. The existing `scale[2]` array and `rescale`
helper already provide the needed reciprocal storage and component mapping.
Do not change the earlier `rescale(out, scale)` inside the key-tile loop.

Output component `i` uses row slot `(i % kFragmentSize) / kPairElems`, equivalent
here to `(i % 4) / 2`. Each thread owns components from two rows and reuses each
row's reciprocal across its components. Keep output ownership, probability
layout, packed-NHD addresses, grid, block, scratch allocations, and caller
stream unchanged. This transformation requires no launch or barrier changes.

Reduce each row sum before computing its reciprocal. Preserve `reduce`'s
alternate-then-adjacent FP32 addition tree and its publication, broadcast, and
reuse barriers. Preserve both CTA barriers around probability consumption;
their producer completion, reader visibility, and safe reuse remain required.
Only the final normalization changes. Leave descending key-tile traversal,
increasing-dimension QK FMA order, increasing-key PV FMA order, masking, online
rescaling, and FP16 probability/output rounding unchanged.

Keep the zero/NaN denominator guard and multiplication by zero for those cases;
do not replace it with an unconditional zero output. Preserve the logarithmic
sum used by the LSE epilogue. The Before helper explicitly issues
`rcp.approx.ftz.f32` once per component to prevent compiler common-expression
elimination. Remove that helper with this optimization. With the retained
`--use_fast_math` build, `1.f / sum[r]` uses the same approximate reciprocal and
flush-to-zero behavior. Do not substitute precise division or change arithmetic
flags. Final FP16 conversion remains round-to-nearest-even.

Rebuild the complete bundle and use `klineage.harness.evaluate` with its unchanged
problem. Inspect generated device code to check that final normalization now
evaluates one reciprocal per row slot and reuses it for the component
multiplications. Distinguish reciprocals used by unrelated scheduler arithmetic.
Keep check evidence outside this card.

## Example configuration

Preserve FP16 Q/K/V/output, FP32 arithmetic and sums, and int32 offsets. The
workload has 16384 tokens, 64 heads, head dimension 128, and nine offsets for
eight sequences. Packed-NHD element strides are `(8192, 128, 1)`; each token
row occupies 16384 bytes. Keep ragged sequence lookup, invalid-Q guards,
invalid-key zero loads, final-key score masking, and guarded output stores.

Tiles cover 128 query rows and 176 keys. Each CTA has 256 threads, grouped into
32-thread warps and two 128-thread math groups. Four lanes cooperate per row;
fragment size is four, pair size two, core row span eight, fragment row span
16, and input panel width eight. Each thread owns 88 score values, 64 output
components, 44 packed probability pairs, and two row-state slots. Thus final
normalization changes from 64 component reciprocals to two row reciprocals per
thread. Keep scalar sequential QK/PV loops and fused online softmax.

Keep the K-major probability index
`(key / kInputPanelCols) * kM * kInputPanelCols + row * kInputPanelCols + key % kInputPanelCols`.
The grid is `kMaxQTiles * kHeads = 8640` CTAs, where
`kMaxQTiles = ceil(kTokens / kM) + kBatch - 1 = 135`.
Each CTA retains 22528 FP16 probability elements and 256 FP32 reduction slots
in global memory. Invocation allocations total 389283840 probability bytes
and 8847360 reduction bytes. The existing LSE workspace is 4194304 bytes.
Keep zero shared-memory allocation and `__launch_bounds__(kThreads, 1)`.

Keep `kLog2Scale = 0.127517432f`, `kScale = 0.0883883461f`, scalar-half
conversion/store compiler controls, and the SM90 device check. Preserve
`-O3`, `-std=c++17`, `--use_fast_math`, `--resource-usage`, `-lineinfo`, and
`-DNDEBUG`, with the existing SM90a target. These settings describe this replay,
not general prerequisites for reciprocal reuse.

# Precondition

- Data types: all consumers of a shared denominator must permit the same
  reciprocal approximation, zero/NaN guard, and later multiplication. Reuse
  cannot replace distinct rounding rules or exceptional-value policies.
- Layout: the thread must know which output components have the identical row
  denominator. Sharing across different denominators produces incorrect
  normalization. No additional contiguity or alignment is needed because
  addresses and ownership stay unchanged.
- Storage: completed denominators must be available to the normalizing thread.
  Existing output and denominator storage remains sufficient; the operation
  needs no new exchange between threads. An inaccessible denominator could not
  supply the shared reciprocal.
- Pipeline: producers must finish and make denominators visible before use,
  and denominator values must stay fixed through their last consumer.
  Preserve existing visibility and buffer-reuse ordering; otherwise a
  reciprocal can describe incomplete or overwritten state.
- Hardware: no additional feature or capacity is required. Existing scalar
  reciprocal arithmetic suffices, and a reciprocal replaces its live
  denominator without increasing buffer capacity.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
__device__ __forceinline__ void normalize(float (&out)[kPvRegs], const float (&sum)[2]) {
    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) {
        const float denominator = sum[(i % kFragmentSize) / kPairElems];
        float inverse = 0.f;

        // Volatile PTX prevents reuse of a reciprocal across output components.
        if (denominator != 0.f && !isnan(denominator)) {
            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                         : "=f"(inverse) : "f"(denominator));
        }
        out[i] *= inverse;
    }
}

// Tail of consumer, after the key-tile loop.
    float denominator[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        sum[r] = reduce<Reduce::Sum>(sum[r], s);
        denominator[r] = sum[r];
        sum[r] = maximum[r] * kScale + __logf(sum[r]);
    }
    normalize(out, denominator);
```

## After

Delete `normalize`; replace the final normalization block in `consumer`:

```cuda
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        sum[r] = reduce<Reduce::Sum>(sum[r], s);
        scale[r] = sum[r] == 0.f || isnan(sum[r]) ? 0.f : 1.f / sum[r];
        sum[r] = maximum[r] * kScale + __logf(sum[r]);
    }
    rescale(out, scale);
```
