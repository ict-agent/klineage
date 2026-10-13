---
skill_id: fmha.first-tile-softmax-specialization
intent: Specialize the first online-softmax tile to skip neutral-state rescaling.
preconditions:
- 'Data types: running arithmetic represents zero and negative infinity; each first-tile
  row maximum and its exponentials are finite, so neutral-state updates can be omitted
  without changing required rounding or exceptional-value behavior.'
- 'Layout: no additional contiguity, alignment, or ownership requirement; the specialization
  keeps every score index and row-to-thread mapping unchanged.'
- 'Storage: running maxima start at negative infinity and sums and output accumulators
  at zero, with no prior contributions; otherwise skipping their first rescaling discards
  live state. No memory relocation is required.'
- 'Pipeline: identify exactly the first processed tile and finish its scores before
  normalization; participating lanes must choose the same specialization and preserve
  reduction publication, reader visibility, and scratch-reuse barriers, or collective
  reads can observe incomplete or overwritten values.'
- 'Hardware: no additional feature or capacity requirement; specialization removes
  scalar work using ordinary device control flow and retains existing scratch allocations.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Specialize the first processed key tile in `solution/attention.cuh::softmax`
and `consumer`. Its running sum and output have no earlier contributions.
Initialize the row maximum from its scores, assign the first exponential to
the sum, and skip the old-state exponential, sum multiplication, and output
rescaling. Subsequent tiles retain the general update.

Add `enum class Step { First, Next };` and template `softmax` on `Step Mode`.
Replace its body as shown below. Remove only the maximum/sum initialization
loop from `consumer`; retain zero initialization of `out`. Immediately after
the existing final-key-tile bounds mask, dispatch `First` when `n == last`.
Call `rescale(out, scale)` only on the `Next` path. The loop traverses key tiles
backwards, so the first processed tile is the highest tile index.

Keep the mask inside its existing `n == last` branch. After the new dispatch,
retain `convert`, `store_prob`, the CTA barrier, `prob_value`, and the CTA reuse
barrier. Keep final sum reduction, normalization, LSE, and output stores intact.
No host ABI, launch, allocation, address, or synchronization change is needed.

The first finite row maximum makes the general initial scale zero. Multiplying
zero sums/output by that scale and adding the first nonnegative exponential
produce the same values as the specialized initialization. Preserve maximum
and sum reduction grouping, score evaluation order, scalar FMA order, and FP16
round-to-nearest conversion. Fully masked first rows or nonfinite maxima need
separate treatment before using this shortcut.

Validate with the existing Kernel evaluator on the complete problem. Inspect
generated code to confirm the first tile skips old-state exponentials and output
multiplications while later tiles retain them. Keep evidence outside this card.

## Example configuration

Replay settings are packed contiguous FP16 Q/K/V/output `[16384,64,128]`, eight
sequences, and int32 offsets. The token stride is `kRow = 64*128 = 8192` half
elements. Scores, running state, and accumulators use FP32. Tile dimensions
are `kM=128`, `kN=176`, and `kDim=128`; retain `kScale=0.0883883461f` and
`kLog2Scale=0.127517432f`.

Each CTA uses 256 threads, divided into two groups of 128. Each four-lane row
group owns two query rows per lane: warp-local row `lane/4 + r*8`, plus
`warp*16`. Each lane stores 88 scores, 64 output components, and 44 probability
pairs. Keep both scalar dot-product loops sequential and their existing
`#pragma unroll 1` controls. Keep the pointwise loops' unroll hints.

Keep the CTA-owned global probability tile of `128*176` half elements in the
existing eight-column panel layout, the 256-float global reduction region, and
all warp/CTA barriers. Probability consumers finish before tile reuse; reduction
readers finish before slot reuse. Scratch lifetimes extend through completion
on the caller stream.

Retain the linear offset-derived scheduler, grid `kMaxQTiles*kHeads` with
`kMaxQTiles=135`, block `kThreads=256`, launch bounds `(kThreads,1)`, and zero
dynamic shared memory. Keep scalar half-conversion calls, scalar output-store
assembly, reciprocal reuse in final normalization, and the fused attention
kernel. The bundle retains its SM90 check and compiler flags `-O3`, `-std=c++17`,
`--use_fast_math`, `--resource-usage`, `-lineinfo`, and `-DNDEBUG`.

# Precondition

- Data types: running arithmetic must represent zero and negative infinity.
  Each first-tile row maximum and its exponentials must be finite: then the
  general initial rescaling maps zero to zero, and adding the first exponential
  to zero agrees with assignment. Preserve required rounding and exceptional
  behavior; otherwise skipping those operations may change results.
- Layout: no additional contiguity, alignment, or ownership requirement.
  Specialization changes scalar initialization and dispatch, leaving all score
  addresses and row ownership intact.
- Storage: maxima begin at negative infinity, sums and output accumulators at
  zero, with no previous contributions. Skipping first rescaling is invalid for
  live accumulated state. Existing placement suffices; no memory move is needed.
- Pipeline: select exactly the first processed tile, after its scores are ready.
  Lanes participating in a reduction must take the same specialization. Preserve
  publication before reads, reader visibility, and completion before scratch
  reuse; divergent collectives or missing ordering can expose incomplete or
  overwritten data. These dependencies impose no particular stage count.
- Hardware: no additional feature or capacity requirement. Ordinary device
  control flow and the existing scalar operations implement specialization;
  scratch allocations remain unchanged.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The first fragment replaces `softmax`; the second shows the affected portions
of `consumer`. Omitted QK, masking, conversion, PV, and epilogue code stays in
place. The snippets use constants and helpers already present in the bundle.

## Before

```cuda
__device__ __forceinline__ void softmax(float (&score)[kQkRegs], float (&maximum)[2],
                                      float (&sum)[2], float (&scale)[2], Reduction& s) {
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        float m = maximum[row];
        #pragma unroll
        for (int col = 0; col < kQkRegs / 2; ++col) m = fmaxf(m, score[(col / 2) * 4 + row * 2 + col % 2]);
        m = reduce<Reduce::Maximum>(m, s);
        scale[row] = exp2f((maximum[row] - m) * kLog2Scale);
        maximum[row] = m;
        sum[row] *= scale[row];
        float scaled = (m == -INFINITY ? 0.f : m) * kLog2Scale;
        #pragma unroll
        for (int col = 0; col < kQkRegs / 2; ++col) {
            int i = (col / 2) * 4 + row * 2 + col % 2;
            score[i] = exp2f(score[i] * kLog2Scale - scaled);
            sum[row] += score[i];
        }
    }
}

// consumer: before the key-tile loop; out's zero initialization follows.
#pragma unroll
for (int r = 0; r < 2; ++r) {
    maximum[r] = -INFINITY;
    sum[r] = 0.f;
}

// consumer: inside the loop, after QK and the existing bounds mask.
softmax(score, maximum, sum, scale, s);
rescale(out, scale);
```

## After

```cuda
enum class Step { First, Next };
template<Step Mode>
__device__ __forceinline__ void softmax(float (&score)[kQkRegs], float (&maximum)[2],
                                      float (&sum)[2], float (&scale)[2], Reduction& s) {
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        float m = Mode == Step::First ? score[row * 2] : maximum[row];
        #pragma unroll
        for (int col = 0; col < kQkRegs / 2; ++col) m = fmaxf(m, score[(col / 2) * 4 + row * 2 + col % 2]);
        m = reduce<Reduce::Maximum>(m, s);
        scale[row] = Mode == Step::First ? 1.f : exp2f((maximum[row] - m) * kLog2Scale);
        maximum[row] = m;
        if constexpr (Mode == Step::Next) sum[row] *= scale[row];
        float scaled = (m == -INFINITY ? 0.f : m) * kLog2Scale;
        #pragma unroll
        for (int col = 0; col < kQkRegs / 2; ++col) {
            int i = (col / 2) * 4 + row * 2 + col % 2;
            score[i] = exp2f(score[i] * kLog2Scale - scaled);
            if (Mode == Step::First && col == 0) sum[row] = score[i];
            else sum[row] += score[i];
        }
    }
}

// consumer: remove maximum/sum initialization; retain out initialization.

// consumer: inside the loop, after QK and the unchanged bounds mask.
if (n == last) {
    softmax<Step::First>(score, maximum, sum, scale, s);
} else {
    softmax<Step::Next>(score, maximum, sum, scale, s);
    rescale(out, scale);
}
```
