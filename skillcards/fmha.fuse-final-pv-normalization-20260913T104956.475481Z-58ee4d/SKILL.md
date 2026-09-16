---
skill_id: fmha.fuse-final-pv-normalization
intent: Fuse output normalization into the final probability-value accumulation launch.
preconditions:
- 'Data types: bypassing the intermediate store/load must preserve accumulator values;
  normalization must retain its arithmetic and rounding, since fusion cannot discard
  a required conversion.'
- 'Layout: each final PV producer must own the complete output components and address
  their denominators; otherwise local normalization would use incomplete sums or the
  wrong rows. No additional contiguity or alignment is needed.'
- 'Storage: final accumulators remain available before their global publication, denominators
  are readable by the producer, and no other consumer needs the intervening unnormalized
  state; fusion removes that state transition.'
- 'Pipeline: denominators must be published before the final PV launch, normalization
  must follow all key contributions, and consumers and scratch reuse must wait for
  the normalized publication; these dependencies prevent incomplete, stale, or overwritten
  reads.'
- 'Hardware: no additional feature or capacity requirement; fusion uses the existing
  CUDA arithmetic and adds no shared-memory allocation or collective instruction.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

In `solution/attention.cuh`, move `attention_normalize`'s denominator loads and
`normalize(out, denominator)` into `attention_values`, immediately after
`prob_value` and before publishing `state.out`. Execute them only for `n == 0`,
the last launch in the descending key-tile traversal. Delete `attention_normalize`.
In `solution/kernel.cu`, remove its launch and associated error check.

This saves one launch and the extra global read/write of completed PV outputs.
Retain `normalize` unchanged, including its per-component volatile
`rcp.approx.ftz.f32`, zero/NaN denominator guard, and subsequent multiplication.
Do not move normalization into the key loop or combine it with an FMA.

The final producer already owns each output accumulator. Its `state` is
`p.epilogue[size_t(blockIdx.x) * kThreads + tid]`; the denominator for component
`i` is `state.denominator[(i % kFragmentSize) / kPairElems]`. Earlier PV launches
continue to publish unnormalized partials. The final one publishes normalized
values for the separate output epilogue. No buffer, layout, launch dimension,
thread ownership, or synchronization change is needed.

Keep the existing `work.w >= kBatch` and `n > last` returns. Valid nonempty
sequences always execute the final `n == 0` tile. Keep Q/output bounds checks,
invalid-key zero filling, and probability masking. The softmax launch publishes
denominators before every PV launch on the caller stream; that stream also orders
partial-output reads, final publication, and epilogue reads. Keep scratch tensors
alive through their readers and retain the conversion kernel's barrier before
score storage is reused for probabilities.

## Example configuration

Preserve the supplied packed-NHD FP16 workload: 16384 tokens, 64 heads, head
width 128, and eight sequences. Input/output row stride is `kHeads * kDim`
half elements. Arithmetic and scratch accumulators are FP32; the extra
FP32 store/load being removed performs no precision conversion. Keep FP16
probability and output rounding, row-sum grouping, increasing inner FMA order,
and descending key-tile order.

Keep `kM = 128`, `kN = 176`, `kThreads = 256`, `kMaxQTiles = 135`, and
`kMaxKeyTiles = 94`. The launch grid has `kMaxQTiles * kHeads` CTAs with zero
dynamic shared memory. Each thread owns `kPvRegs = 64` output components and
two row denominators. For component `i`, its local output coordinates are:

```cuda
const int row = (tid / kWarpSize) * kFragmentRows
                + (tid % kWarpSize) / kLanesPerRow
                + (i % kFragmentSize) / kPairElems * kCoreRows;
const int col = (i / kFragmentSize) * kInputPanelCols
                + (tid % kLanesPerRow) * kPairElems + i % kPairElems;
```

The query tile starts at `q_offsets[work.w] + work.y * kM`; `work.z` selects
the head. Preserve constants `kWarpSize = 32`, `kFragmentRows = 16`,
`kLanesPerRow = 4`, `kFragmentSize = 4`, `kPairElems = 2`, `kCoreRows = 8`,
and `kInputPanelCols = 8` for this ownership mapping.

Keep scalar QK/PV computation, materialized scores, separate maximum, softmax,
probability conversion, and output launches; global reduction scratch; aliased
score/probability storage; and the existing SM90 ABI, caller stream, compiler
flags, and launch bounds. Validate through `klineage.harness.evaluate` with the
unchanged problem and policy; keep measurements outside this card.

# Precondition

- Data types: eliminating the intermediate store/load must leave each accumulator
  unchanged. Preserve normalization arithmetic and rounding. A storage conversion
  that changes values would need to be reproduced explicitly inside the producer.
- Layout: the final producer must own complete output components and locate each
  component's denominator. Otherwise normalization would act on partial sums or
  another row. Fusion adds no contiguity or alignment requirement.
- Storage: completed accumulators must still be available before global publication,
  and the producer must be able to read denominators. No other consumer may require
  the unnormalized state between these launches, since fusion removes it.
- Pipeline: denominator production must complete and become visible before the
  final PV launch. All key contributions must precede normalization; normalized
  publication must precede downstream reads. Scratch must remain live and must not
  be reused until its readers finish. Without these orderings, fusion can read
  stale denominators, normalize partial sums, or expose overwritten outputs.
- Hardware: no additional feature or capacity requirement. The transformation uses
  the existing CUDA arithmetic and adds neither shared storage nor collective
  instructions; it only moves an existing computation into its producer.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

The first fragment is the tail of `attention_values`; the second is the complete
standalone normalization kernel. The last fragment is the host launch tail.

```cuda
    prob_value(out, prob.prob, p.v + base, valid, wg);

    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) state.out[i] = out[i];
}

__global__ __launch_bounds__(kThreads, 1) void attention_normalize(
    const __grid_constant__ Params p) {
    const int4 work = tile(blockIdx.x, p.q_offsets);
    if (work.w >= kBatch) return;

    // The final PV launch publishes unnormalized outputs on the caller stream.
    auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + threadIdx.x];
    float out[kPvRegs], denominator[2];
    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) out[i] = state.out[i];
    #pragma unroll
    for (int r = 0; r < 2; ++r) denominator[r] = state.denominator[r];
    normalize(out, denominator);

    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) state.out[i] = out[i];
}
```

```cuda
for (int n = kMaxKeyTiles - 1; n >= 0; --n) {
    check(cudaLaunchKernelEx(&config, attention_values, p, n));
    check(cudaGetLastError());
}
check(cudaLaunchKernelEx(&config, attention_normalize, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention_epilogue, p));
check(cudaGetLastError());
```

## After

Replace the `attention_values` tail with this fragment and delete the entire
`attention_normalize` definition. The local `denominator` array supplies the
unchanged helper's array-reference parameter.

```cuda
    prob_value(out, prob.prob, p.v + base, valid, wg);

    // Normalize only after the final key tile has contributed.
    if (n == 0) {
        float denominator[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) denominator[r] = state.denominator[r];
        normalize(out, denominator);
    }

    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) state.out[i] = out[i];
}
```

```cuda
for (int n = kMaxKeyTiles - 1; n >= 0; --n) {
    check(cudaLaunchKernelEx(&config, attention_values, p, n));
    check(cudaGetLastError());
}
check(cudaLaunchKernelEx(&config, attention_epilogue, p));
check(cudaGetLastError());
```
