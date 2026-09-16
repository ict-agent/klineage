---
skill_id: cuda.shared-memory-lifetime-aliasing
intent: Reuse CTA shared-memory storage across nonoverlapping access lifetimes.
preconditions:
- 'Data types: No additional dtype or numerical requirement; reusing allocation bytes
  changes neither a live value''s representation nor its arithmetic.'
- 'Layout: Candidate views have known extents, internal offsets, and access alignment
  requirements; these determine whether a common allocation preserves legal addresses
  and indexing.'
- 'Storage: Candidate buffers occupy separate regions of the same CTA shared-memory
  address space; storage live throughout their phases remains outside those regions
  because alias writes would destroy it. Another CTA''s buffers cannot share this
  allocation.'
- 'Pipeline: Producers complete and their writes become visible before consumption;
  all readers and asynchronous transfers touching an address finish before its next
  view overwrites that address. Participants must reach the required ordering boundaries
  to prevent overwrite races.'
- 'Hardware: No additional hardware feature beyond CTA shared memory and its existing
  ordering primitives. Each overlay must accommodate its largest view at every view''s
  required alignment, and the full aligned allocation must fit the configured per-CTA
  shared-memory limit.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reuse CTA shared memory across nonoverlapping access lifetimes by replacing separate
fields with unions in `solution/native.cuh`. Apply all four overlays shown below:
`{q,k,g}` with `{kd,qd,ki,l,inv,mqk}`, `gate` with `kr`, `bias` with `gt`, and
`{input,out}` with `fp32`. Keep the persistent recurrent `state`, `beta`, and
barrier objects outside their respective overlays. This reduces shared allocation
without changing calculations or moving live values between memory levels.

Replace only `PrepareShared`, `RecurShared`, and their size assertions using the
After snippet. Keep `InputShared` in its existing position and retain all member
names, array extents, and alignments. The `Pair` union represents packed values;
it is unrelated to allocation reuse and remains unchanged.

`solution/native.cu::launch` already uses `sizeof(PrepareShared)` and
`sizeof(RecurShared)` for both the dynamic-memory attributes and launch arguments.
Those expressions automatically adopt the smaller allocations. Preserve both
launch grids, thread counts, caller stream, tensor maps, and compile flags.

## Lifetime boundaries

- In `solution/prepare.cuh::prepare`, keep the initial `wait_bar`, proxy fence,
  and CTA barriers before consuming TMA input. Each gate thread captures its
  own `bias[tid]` in a register before writing the aliased `gt[tid]`; no other
  thread reads that bias element. The barrier after the gate loop completes
  all `gate` reads before any `kr` write.
- Keep the complete `rq/rk/rg/rgt` register gather and its following
  `__syncthreads()` before writing decayed matrices. This ends all shared
  `q/k/g` reads before `kd/qd/ki/l/inv/mqk` can overwrite those bytes. The
  values still needed for decay remain in thread-owned registers.
- In `solution/recurrence.cuh::recurrence`, retain the state-load completion
  wait and proxy fence, `state_in(s)`, and both following CTA barriers.
  All `fp32` readers finish before the load warp starts writing `input`.
- Retain all input/output full/empty barriers and phase rotation. Before the
  final `state_out(s)`, the load warp drains input consumers, the store warp
  waits for each output transfer to finish reading shared memory, and the
  CTA joins at `__syncthreads()`. Thus neither pipeline reads nor pending
  shared-memory transfers overlap the new `fp32` writes. Keep the final
  proxy fence and CTA barrier before the final-state TMA store.

No synchronization is removed or added. Preserve the field-relative indexing,
thread ownership, and existing bounds handling; allocation aliasing adds no
padding elements to the operator. The comments before the prepare register gather
and final-state conversion may describe the restored reuse.

## Example configuration

This instance has batch 1, 4096 tokens, 96 heads, head dimension 128, and chunks
of 16 tokens. Preparation launches `(256,96)` CTAs with 256 threads. Recurrence
launches `(1,96)` CTAs with 192 threads: four compute warps, one load warp, and
one store warp. Keep three input stages and two output stages, including all
transaction counts, barrier arrival counts, and phase initialization.

Keep BF16 inputs, intermediates, and recurrent shared state; FP32 gates and MMA
accumulators; and the FP32 state conversion buffer. Preserve every BF16 rounding
point, the ordered MMA/substitution operations, and the existing normalization
and gate arithmetic. The complete problem oracle and numerical requirements
remain authoritative.

The preparation input rows use `row*kDim+col`; matrix views use the retained
8-column slab mapping `row*8 + (col&7) + (col/8)*Rows*8`. State conversion uses
the same unswizzled slab mapping. Buffer bases remain 128-byte aligned, and
barriers keep their existing alignment. Preserve TMA, warp MMA, matrix
load/store/transpose instructions, register prefetching, packed arithmetic,
register tiling, and stream-local global workspace allocation.

The size assertions change `PrepareShared` from 40192 to 21248 bytes and
`RecurShared` from 160768 to 98432 bytes. `InputShared` stays 18048 bytes.
Keep the SM90 target, `kPrepareThreads` launch bound with minimum-block hint 8,
the recurrence launch bound, and every existing build flag, including register
usage level 10. These are replay settings, not prerequisites of lifetime reuse.

# Precondition

- Data types: No additional dtype or numerical requirement. A union changes
  where a view lives, without converting its values or changing arithmetic.
  Different representations may occupy the same bytes at different times.
- Layout: The candidate views' extents, internal offsets, and access alignments
  must be known. They determine the allocation extent and base alignment;
  violating either makes retained indexing out of bounds or accesses misaligned.
- Storage: The candidate buffers already occupy separate regions in one CTA's
  shared-memory address space. Data or synchronization storage needed throughout
  their phases stays outside these regions; overwriting it would destroy live
  state. These declarations cannot overlay another CTA's allocation.
- Pipeline: Each producer must complete and make its writes visible before
  consumers read. Every consumer and asynchronous transfer touching an address must finish
  before the next view overwrites that address. Participating threads must
  reach the required ordering boundaries. The condition applies per overlapping
  address: same-thread program order suffices when no other thread accesses it. Otherwise aliasing introduces incomplete
  reads or overwrites. Keeping a value in a private register ends its shared
  read lifetime only after that register load has completed.
- Hardware: No additional feature beyond existing CTA shared memory and its
  ordering primitives is required. For overlay set `i`, let `B_ij` be each
  view's byte extent and `A_i` the strongest required alignment. Reserve at least
  `align_up(max_j(B_ij), A_i)` bytes for that set. Nonoverlaid storage, all overlay
  sets, and layout padding must fit the configured per-CTA shared-memory limit;
  otherwise indexing exceeds storage or the kernel cannot launch.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are the affected declarations in `solution/native.cuh`. Existing constants,
`BF16`, and the unchanged `InputShared` definition supply shared context.
All device functions and launch expressions remain as described above.

## Before

```cuda
struct alignas(128) PrepareShared {
    // Give each phase separate storage; retain its layout and access order.
    BF16 q[kTileElems], k[kTileElems];
    float g[kTileElems];
    BF16 kd[kTileElems], qd[kTileElems], ki[kTileElems];
    float l[kMatrixElems];
    BF16 inv[kMatrixElems], mqk[kMatrixElems];

    alignas(128) BF16 beta[kBetaElems];
    alignas(128) BF16 gate[kTileElems], kr[kTileElems];
    alignas(128) float bias[kDim], gt[kDim];
    alignas(16) uint64_t ready;
};

// Keep the existing InputShared declaration here.

struct alignas(128) RecurShared {
    BF16 state[kStateElems];
    // Keep conversion scratch separate from the input/output pipeline.
    InputShared input[kInputStages];
    BF16 out[kOutputStages][kTileElems];
    alignas(128) float fp32[kStateElems];

    uint64_t load_full[kInputStages], load_empty[kInputStages];
    uint64_t out_full[kOutputStages], out_empty[kOutputStages];
    alignas(16) uint64_t state_ready;
};
static_assert(sizeof(PrepareShared) == 40192);
static_assert(sizeof(InputShared) == 18048);
static_assert(sizeof(RecurShared) == 160768);
```

## After

```cuda
struct alignas(128) PrepareShared {
    union {
        struct { BF16 q[kTileElems], k[kTileElems]; float g[kTileElems]; };
        struct {
            BF16 kd[kTileElems], qd[kTileElems], ki[kTileElems];
            float l[kMatrixElems];
            BF16 inv[kMatrixElems], mqk[kMatrixElems];
        };
    };
    alignas(128) BF16 beta[kBetaElems];
    union { alignas(128) BF16 gate[kTileElems], kr[kTileElems]; };
    union { alignas(128) float bias[kDim], gt[kDim]; };
    alignas(16) uint64_t ready;
};

// Keep the existing InputShared declaration here.

struct alignas(128) RecurShared {
    BF16 state[kStateElems];
    union {
        struct { InputShared input[kInputStages]; BF16 out[kOutputStages][kTileElems]; };
        alignas(128) float fp32[kStateElems];
    };
    uint64_t load_full[kInputStages], load_empty[kInputStages];
    uint64_t out_full[kOutputStages], out_empty[kOutputStages];
    alignas(16) uint64_t state_ready;
};
static_assert(sizeof(PrepareShared) == 21248);
static_assert(sizeof(InputShared) == 18048);
static_assert(sizeof(RecurShared) == 98432);
```
