---
skill_id: kda.warp-specialized-tma-pipeline
intent: Overlap chunk transfers and recurrence with a warp-specialized TMA pipeline.
preconditions:
- 'Data types: copies must preserve a tensor-map-supported representation; no particular
  arithmetic dtype is required because transfers do not compute.'
- 'Layout: innermost-contiguous global tensors with descriptor-encodable extents and
  byte strides, 16-byte-aligned bases, outer byte strides divisible by 16, and 16-byte-multiple
  inner box widths; otherwise tiled TMA cannot describe the copies. Each chunk''s
  consumers and output producers must share its CTA, with known shared indexing, so
  local staging supplies the right elements.'
- 'Storage: chunk inputs already reside in global memory, remain stable while read,
  and feed CTA-shared scratch; outputs go to global memory without overwriting unread
  inputs. Otherwise prefetched data or delayed stores can change the computation.'
- 'Pipeline: future chunk inputs must be ready independently of recurrent state; preserve
  sequential state updates. All designated participants must reach their handshakes,
  publish completed producers before consumers, and finish readers before scratch
  reuse; violations cause stale reads, overwrites, or deadlock.'
- 'Hardware: tiled TMA, transaction-counted mbarriers, async-proxy fences, and CTA
  barriers must be supported. Shared capacity must satisfy S_persistent + N_in*S_in
  + N_out*S_out + S_barriers + S_padding <= S_available; TMA stage addresses need
  128-byte alignment and mbarriers 8-byte alignment. Unsupported instructions, misalignment,
  or excess capacity invalidate execution.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Overlap recurrence input transfers, computation, and output transfers using
dedicated warp leaders, TMA, and rotating shared buffers. Replace the serialized
chunk loop in `solution/recurrence.cuh::recurrence`; leave `recur_tile`,
`state_in`, `state_out`, and `load_correction` unchanged.

The deoptimized kernel copies each chunk cooperatively into one shared input buffer,
computes, copies its output cooperatively, then reuses both buffers. The forward
change replaces those copies and CTA barriers with the pipeline below.

## Storage and descriptors

In `solution/native.cuh`, add the constants shown under After after `kPrepareTx`,
and restore the indicated members.
Keep `state`, `fp32`, `correction`, and `state_ready` in their existing order.
Place the four new barrier arrays immediately before `state_ready`. Update the
`RecurShared` size assertion to `164864`; retain the other assertions.

In `solution/recurrence.cuh`, replace `RecurArgs` with:

```cuda
struct RecurMaps {
    CUtensorMap v, beta, kd, qd, kr, gt, inv, mqk, state_in, state_out, out;
};
```

Delete `load_input` and `store_output`. Change `recurrence`'s parameter type to
`RecurMaps`, retain `const __grid_constant__`, name it `maps`, and change its two
state descriptor accesses from `args.state_in/out` to `maps.state_in/out`.

In `solution/native.cu::launch`, declare `RecurMaps recur`. Delete the nine raw
pointer assignments to `recur`. Insert these entries before the existing
`recur.state_in` request; keep both state requests and every preparation request:

```cuda
{&recur.v,inputs[2],MapKind::Input}, {&recur.beta,b->beta,MapKind::Beta},
{&recur.kd,kd,MapKind::Workspace}, {&recur.qd,qd,MapKind::Workspace},
{&recur.kr,kr,MapKind::Workspace}, {&recur.gt,gt,MapKind::Gate},
{&recur.inv,inv,MapKind::Matrix}, {&recur.mqk,mqk,MapKind::Matrix},
{&recur.out,outputs[0],MapKind::Input},
```

Use the existing `encode_map`, `load1`, `load2`, `load_slabs`, `store_slabs`,
`commit_store`, and `wait_store` unchanged. Descriptors and their global storage
remain valid throughout the launch. The host already derives shared-memory
attributes and launch allocation from `sizeof(RecurShared)`.

Shared matrices keep eight-column slabs:
`offset<Rows>(r,c) = r*8 + (c%8) + (c/8)*Rows*8`.
Each `load_slabs<Rows,Cols>` transfers one eight-column box at a time to
`dst + c*Rows`; each store applies the inverse mapping. No swizzle is introduced.
Global `v` and output use BTHD indexing
`((token+row)*kHeads+head)*kDim+col` for the retained single batch.
Workspace matrices are row-major per `ws=head*kTiles+tile`; beta is head-major;
gate totals are contiguous per workspace tile. Descriptor row strides follow
these extents and element sizes. TMA moves their bits without conversion.

## Ordering and ownership

Insert the initialization block under After immediately after `head` is set,
before the retained initial-state load. Replace only the serialized chunk loop
with the three role loops. Add the five helper functions under After to
`solution/native.cuh` after `wait_bar`.

One elected lane of the load warp fills input stages. Compute threads wait for
all input transaction bytes and an available output stage, then run the unchanged
`recur_tile`. A compute-only barrier joins their state updates. Each compute
thread fences its shared writes into the async proxy before publishing output.
One elected store lane waits for every producer, issues output TMA stores, and
waits until their shared reads finish before releasing that output stage.

`load_full` counts one producer plus transaction bytes; `load_empty` counts all
compute threads. `out_full` counts all compute threads; `out_empty` counts one
store leader. Full-buffer waits start at phase zero. Empty-buffer waits start at
phase one, allowing the initial write. Toggle parity only when wrapping the
corresponding ring. Count transferred payload, excluding shared struct padding.

Keep the loader's final drain and the CTA barrier after all three role loops.
No CTA barrier may appear inside a role loop. Preserve the final-state conversion,
its proxy fence and CTA barriers, and the final state TMA store. Initial/final
state transfers and all preparation transfers retain their original ordering.
Compute still advances recurrent state in increasing chunk order; only copies
overlap. Correction scratch remains warp-exclusive and retains its publication
barrier inside `recur_tile`.

The transfer rules follow NVIDIA's
[tiled-copy alignment documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
and [tensor-map encoding contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html).

## Example configuration

Preserve batch 1, 4096 tokens, 96 heads, dimension 128, chunk size 16, and 256
chunks per head. Preparation uses grid `(kTiles,kHeads)` and 256 threads;
recurrence uses grid `(1,kHeads)` and 192 threads. Four compute warps own 32 value
columns each, starting at `warp*32`; warps 4 and 5 supply the load and store leaders.
The default single-CTA cluster makes the barrier helper's cluster rank zero local.
Preserve the caller device, stream, tensor ABI, workspace allocation, and flags.

Restore three input stages and two output stages. Input payload is 17984 bytes;
`InputShared` occupies 18048 bytes after padding. Each output stage is 4096 bytes.
`RecurShared` grows from 124672 to 164864 bytes. `PrepareShared` remains 42368 bytes.
Keep the compute named-barrier identifier 8. These are replay settings, not general
requirements for the technique.

All token tiles are full. The beta descriptor copies 32 entries although compute
reads only the first 16; its final out-of-range entries retain zero fill. Keep
descriptor OOB settings; add no divergent compute predicates or early returns.

Retain BF16 operands/intermediates/output, FP32 accumulators and external state,
all BF16 rounding points, scalar pair additions, fast activations, normalization
epsilon, scale, and gate bound. Preserve MMA and its fragment adapters, per-warp
register tiles, K-step prefetch, shared correction materialization, preparation
TMA, separate state-conversion scratch, and persistent chunk traversal per head.
No computation or reduction order changes. Rebuild the final bundle with its
preserved compile flags and evaluate against the unchanged problem using
`klineage.harness.evaluate`; keep measurements outside this card.

# Precondition

- Data types: copied elements must have a representation supported by the tensor
  descriptor, and transfers must preserve that representation. TMA cannot encode
  arbitrary unsupported element formats. No particular arithmetic dtype is
  required: the transformation moves bits and leaves arithmetic and conversions
  at their existing sites.
- Layout: global tensors must have contiguous innermost elements and extents and
  byte strides encodable by tiled tensor maps. Bases must be 16-byte aligned;
  outer byte strides and inner box widths in bytes must be multiples of 16.
  These are descriptor/copy rules, not workload dimensions. Each chunk's readers
  and output writers must share its CTA, with known shared indexing, or local
  stages cannot supply the correct consumers. Other layouts require a different
  transfer mapping.
- Storage: inputs already exist in global memory and feed CTA-shared scratch;
  output destinations are global. Inputs must remain stable while read, and
  output writes must not overwrite unread input. Prefetch and deferred output
  would otherwise observe or cause different values.
- Pipeline: future input chunks must be ready independently of recurrent state,
  or ahead-of-compute loads fetch incomplete values. State updates remain ordered.
  Every designated handshake participant must arrive, or a waiter deadlocks.
  Producers must complete and publish before consumers read; all readers must
  finish before storage reuse. Those dependencies prevent incomplete data and
  overwrites, including async output readers. Stage counts are not prerequisites.
- Hardware: the device must support tiled TMA, transaction-counted mbarriers,
  async-proxy fences, and CTA barriers. Unsupported operations cannot implement
  this pipeline. Shared capacity must satisfy
  `S_persistent + N_in*S_in + N_out*S_out + S_barriers + S_padding <= S_available`,
  where `N` counts live stages and each `S` is its corresponding byte allocation.
  TMA stage addresses require
  128-byte alignment; barrier addresses require 8-byte alignment. Insufficient
  capacity or misalignment makes the allocation or accesses invalid. Retained
  matrix computation imposes no additional prerequisite for this transfer change.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are replacement sites, not complete files. Apply the descriptor and helper
deletions above; leave omitted arithmetic, state transfers, and launch code intact.

## Before

```cuda
// native.cuh: RecurShared members, then its external size assertion.
InputShared input;
BF16 out[kTileElems];
static_assert(sizeof(RecurShared) == 124672);

// recurrence.cuh: the complete serialized chunk loop.
for (int t = 0; t < kTiles; ++t) {
    load_input(s.input,args,t,head);
    __syncthreads();
    if (warp < kComputeWarps) recur_tile(s,s.input,s.out);
    __syncthreads();
    store_output(s.out,args,t,head);
    __syncthreads();
}
```

## After

```cuda
// native.cuh: namespace constants.
constexpr int kInputStages = 3;
constexpr int kOutputStages = 2;
constexpr int kComputeBarrier = 8;
constexpr int kRecurTx = 4 * kTileBytes + kBetaElems * sizeof(BF16)
    + kGateBytes + 2 * kMatrixBytes;

// RecurShared: replace input/out; insert barriers before state_ready.
InputShared input[kInputStages];
BF16 out[kOutputStages][kTileElems];
// Keep fp32 and correction here, unchanged.
uint64_t load_full[kInputStages], load_empty[kInputStages];
uint64_t out_full[kOutputStages], out_empty[kOutputStages];
// Keep state_ready; replace the external size assertion:
static_assert(sizeof(RecurShared) == 164864);

// native.cuh: add after wait_bar; reuse shared_addr and wait_bar.
__device__ __forceinline__ void acquire_bar(uint64_t* bar, unsigned phase) {
    unsigned done;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done) : "r"(shared_addr(bar)), "r"(phase) : "memory");
    if (!done) wait_bar(bar, phase);
}
__device__ __forceinline__ void consume_bar(uint64_t* bar, unsigned phase) {
    unsigned done;
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
      : "=r"(done) : "r"(shared_addr(bar)), "r"(phase) : "memory");
    if (!done) wait_bar(bar, phase);
}
__device__ __forceinline__ void arrive_bar(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
      :: "r"(shared_addr(bar)) : "memory");
}
__device__ __forceinline__ void release_bar(uint64_t* bar) {
    asm volatile("{ .reg .b32 remote; mapa.shared::cluster.u32 remote, %0, 0; mbarrier.arrive.shared::cluster.b64 _, [remote]; }"
      :: "r"(shared_addr(bar)) : "memory");
}
__device__ __forceinline__ void compute_sync() {
    asm volatile("bar.sync %0, %1;"
      :: "n"(kComputeBarrier), "n"(kComputeThreads) : "memory");
}

// recurrence: insert after head, before the retained initial-state load.
if (warp == 0 && leader) {
    #pragma unroll
    for (int stage = 0; stage < kInputStages; ++stage) {
        init_bar(s.load_full+stage,1);
        init_bar(s.load_empty+stage,kComputeThreads);
    }
}
init_fence();
__syncthreads();
if (warp == 0 && leader) {
    #pragma unroll
    for (int stage = 0; stage < kOutputStages; ++stage) {
        init_bar(s.out_full+stage,kComputeThreads);
        init_bar(s.out_empty+stage,1);
    }
}
init_fence();
__syncthreads();

// After initial-state loading/conversion: replace the serialized loop.
if (warp == kLoadWarp && leader) {
    int stage = 0, phase = 1;
    for (int t = 0; t < kTiles; ++t) {
        acquire_bar(s.load_empty+stage,phase);
        auto& in = s.input[stage];
        auto* bar = s.load_full+stage;
        expect_bar(bar,kRecurTx);
        int ws = head*kTiles+t;
        load_slabs<kChunk,kDim>(maps.v,in.v,bar,t*kChunk,head);
        load1(maps.beta,in.beta,bar,head*kTokens+t*kChunk);
        load_slabs<kChunk,kDim>(maps.kd,in.kd,bar,0,ws);
        load_slabs<kChunk,kDim>(maps.qd,in.qd,bar,0,ws);
        load_slabs<kChunk,kDim>(maps.kr,in.kr,bar,0,ws);
        load2(maps.gt,in.gt,bar,0,ws);
        load_slabs<kChunk,kChunk>(maps.inv,in.inv,bar,0,ws);
        load_slabs<kChunk,kChunk>(maps.mqk,in.mqk,bar,0,ws);
        if (++stage == kInputStages) { stage = 0; phase ^= 1; }
    }
    #pragma unroll
    for (int j = 0; j < kInputStages; ++j) {
        wait_bar(s.load_empty+stage,phase);
        if (++stage == kInputStages) { stage = 0; phase ^= 1; }
    }
}

if (warp < kComputeWarps) {
    int load_stage = 0, load_phase = 0, out_stage = 0, out_phase = 1;
    for (int t = 0; t < kTiles; ++t) {
        acquire_bar(s.out_empty+out_stage,out_phase);
        wait_bar(s.load_full+load_stage,load_phase);
        recur_tile(s,s.input[load_stage],s.out[out_stage]);
        compute_sync();
        async_fence();
        arrive_bar(s.out_full+out_stage);
        release_bar(s.load_empty+load_stage);
        if (++load_stage == kInputStages) { load_stage = 0; load_phase ^= 1; }
        if (++out_stage == kOutputStages) { out_stage = 0; out_phase ^= 1; }
    }
}

if (warp == kStoreWarp && leader) {
    int stage = 0, phase = 0;
    for (int t = 0; t < kTiles; ++t) {
        consume_bar(s.out_full+stage,phase);
        store_slabs<kChunk,kDim>(maps.out,s.out[stage],t*kChunk,head);
        wait_store();
        release_bar(s.out_empty+stage);
        if (++stage == kOutputStages) { stage = 0; phase ^= 1; }
    }
}
// Keep the following CTA barrier and final-state conversion/store unchanged.
```
