---
skill_id: fmha.shared-reduction-scratch
intent: Place temporary softmax peer-exchange scratch in CTA shared memory.
preconditions:
- 'Data types: no additional dtype requirement; scratch stores and loads must preserve
  each exchanged value''s representation without changing reduction arithmetic.'
- 'Layout: producer slots and peer indices are known, distinct live producers have
  distinct slots, and every peer belongs to the same CTA; shared indexing must preserve
  this ownership.'
- 'Storage: scratch currently resides in global memory and carries only temporary,
  CTA-private exchange values; other CTAs or the host must not require its contents.'
- 'Pipeline: producers complete writes before peers read, and all reads complete before
  slot reuse; existing synchronization must order shared-memory accesses for every
  participating producer and reader.'
- 'Hardware: CUDA shared memory and suitable memory-ordering barriers are available;
  per-CTA capacity must cover the scratch layout, including padding, plus other simultaneously
  live shared allocations, with alignment sufficient for the scratch layout type.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Move the softmax exchange workspace from per-invocation global memory to
per-CTA dynamic shared memory. Each thread still writes its own slot and reads
the slot selected by `tid ^ int(peer)`. Preserve the reduction tree and every
arithmetic operation; only the communication storage changes.

Apply these edits to the supplied bundle:

1. In `solution/ops.cuh`, remove `Reduction* reduction` from `Params`.
   Retain `Reduction`, its alignment, and `exchange` unchanged.
2. In `solution/attention.cuh::attention`, replace the global scratch reference
   with the shared declaration and cast shown below. Keep `consumer(p, s)`.
3. In `solution/kernel.cu::kernel`, remove the local `reduction` tensor allocation
   and `p.reduction` assignment. Set `config.dynamicSmemBytes = sizeof(Reduction)`.
4. In `Workspace::ensure`, after the SM check and before `device = current`, add
   the `cudaFuncSetAttribute` call below. Retain the device guard, caller stream,
   grid, block, probability allocation, and launch error handling.

No scratch initialization is needed: every live producer writes before its peer
reads. Keep both `__syncwarp(kFullWarpMask)` calls in `exchange`: the first orders
publication, the second prevents overwrite before peer reads finish. Keep the
two CTA barriers around probability consumption and reuse. Excess CTAs return
uniformly in `consumer` before any exchange; valid CTAs retain complete warps.
Keep query/key bounds, masked scores, output guards, and offset scheduling.

Do not change scalar FMA order, descending key-tile traversal, online softmax,
FP32 reduction state, FP16 rounding, or scalar output stores. This transfer of
scratch placement adds no conversion and leaves exchanged bits unchanged.
Use the existing evaluator with the unchanged problem for correctness and timing;
inspect generated exchange accesses for shared loads/stores. Keep evidence outside
this card.

## Example configuration

Preserve packed contiguous FP16 Q/K/V/output `[16384,64,128]`, int32 offsets `[9]`,
and eight sequences. The row stride is `kRow = kHeads * kDim = 8192` half elements.
The tile is `kM=128`, `kN=176`, `kDim=128`; grid size is
`kMaxQTiles*kHeads = 135*64 = 8640`, with `kThreads=kMathThreads=256`.
Each thread retains 88 score and 64 output values; four lanes cooperate per row.
Warp size is 32, with XOR peers 2 then 1 and the existing full-warp mask.

`Reduction` remains `alignas(128)` with `float reduction[kMathThreads]`.
The current global allocation reserves `grid_ctas*sizeof(Reduction)` bytes,
1024 bytes per CTA, and is isolated per invocation. The optimized launch requests
1024 dynamic shared bytes per CTA with the replay declaration aligned to 1024.
These are replay settings; capacity and alignment must be recomputed for another
scratch layout. Preserve the global FP16 probability tile, its panel mapping,
scalar QK/PV arithmetic, and the host LSE workspace. Retain the SM90 target and
all compiler flags: `-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.
No compiler-control change is needed for this storage transformation.

# Precondition

- Data types: no additional dtype requirement. This optimization relocates
  scratch values without computing on them. Stores and loads must preserve their
  representation; adding conversion could change the reduction result.
- Layout: producer slots and peer indices must be known, with distinct slots for
  distinct live producers and all peers in one CTA. The same slot mapping then
  supplies each reader's intended value; aliasing slots would race, and a peer in
  another CTA cannot address this CTA's ordinary shared memory.
- Storage: the existing global scratch must contain only temporary CTA-private
  exchange values. Host or other-CTA consumers would lose access after relocation.
- Pipeline: producer writes must finish and become visible before peer reads;
  all reads must finish before those slots are overwritten. Existing synchronization
  must cover every sharing producer and reader and order shared-memory accesses.
  Otherwise relocation permits incomplete publication or premature reuse.
- Hardware: CUDA shared memory and barriers with the required ordering must exist.
  Available per-CTA shared capacity must cover `sizeof(scratch_layout)` plus other
  simultaneously live shared allocations, including layout padding; storage must
  meet the scratch layout type's alignment. Insufficient capacity prevents the launch, and
  insufficient alignment makes typed accesses invalid.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets show the changed storage path. Keep `Reduction`, `exchange`, and
all callers intact. In `Params`, delete only the `Reduction* reduction` field.
The host excerpts belong at their existing allocation and launch locations.

## Before

```cuda
// solution/kernel.cu::kernel: allocation and Params binding.
const auto reduction = torch::empty(
    {int64_t(kMaxQTiles) * kHeads * sizeof(Reduction) / sizeof(float)},
    q.options().dtype(torch::kFloat32));
p.reduction = reinterpret_cast<Reduction*>(reduction.data_ptr<float>());
// Existing launch configuration:
config.dynamicSmemBytes = 0;

// solution/attention.cuh: complete device entry.
__global__ __launch_bounds__(kThreads, 1)
void attention(const __grid_constant__ Params p) {
    auto& s = p.reduction[blockIdx.x];
    consumer(p, s);
}
```

## After

```cuda
// solution/kernel.cu::Workspace::ensure: before device = current.
check(cudaFuncSetAttribute(attention,
    cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Reduction)));

// solution/kernel.cu::kernel: remove the reduction allocation and binding.
config.dynamicSmemBytes = sizeof(Reduction);

// solution/attention.cuh: complete device entry.
__global__ __launch_bounds__(kThreads, 1)
void attention(const __grid_constant__ Params p) {
    extern __shared__ __align__(1024) char storage[];
    auto& s = *reinterpret_cast<Reduction*>(storage);
    consumer(p, s);
}
```
