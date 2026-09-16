---
skill_id: kda.recurrence-temporal-fusion
intent: Fuse recurrence chunks to retain state in shared memory.
preconditions:
- 'Data types: intermediate state handoffs must preserve representation without required
  conversion or rounding; deleting them must leave the arithmetic unchanged.'
- 'Layout: successive chunks must use consistent shared-state indexing and permit
  exclusive ownership of each sequence by one CTA; otherwise local state cannot supply
  every dependent update.'
- 'Storage: chunks already stage state in CTA shared memory and export it only for
  the next chunk; no external consumer may require those intermediate global snapshots.'
- 'Pipeline: prepared inputs must be complete before recurrence, and chunks must depend
  only on their own preceding state. All CTA threads must reach barriers that publish
  loads and updates and finish reads before buffer reuse; fusion cannot replace a
  required cross-CTA synchronization.'
- 'Hardware: no additional feature or capacity beyond the existing legal CTA is required;
  fusion reuses its shared allocation and barriers without cross-CTA cooperation.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse sequential recurrence chunks into one CUDA CTA per head. Keep the updated
state in `RecurShared::state` between chunks, eliminating intermediate global
state transfers and repeated recurrence launches.

In `solution/recurrence.cuh`, replace the complete `recurrence` definition with
the After definition below. Remove `RecurArgs::carry`, `load_carry`, and
`store_carry`. Keep every other helper and `RecurShared` unchanged. Move initial
state loading/conversion before the chunk loop and final conversion/export after
it. Keep `recur_tile` and its arithmetic unchanged.

In `solution/native.cu`, replace the host chunk-launch loop with the single
launch below. Remove `Buffers::carry`, its `if (!b.carry)` allocation block,
and `recur.carry = b->carry`. Retain beta/workspace allocation, device/stream
ownership, preparation launches, dynamic shared-memory attributes, error
handling, and compile flags. No public ABI change is needed.

Each CTA owns one head throughout the loop. Each compute warp keeps its existing
value tile; every state element retains its current shared index. The preceding
global carry uses `[head,value,key]`, indexed as
`head*kStateElems + value*kDim + key`; its bit-preserving copies disappear.
There is no new shared allocation, permutation, or thread mapping within a chunk.

The caller stream completes preparation before recurrence. Within the CTA,
retain barriers after initial state conversion, after input loading, after
computation, and after output copying. They publish updated state and prevent
reuse of input/output storage while readers remain. The final conversion stays
after the final chunk. Other heads need no intermediate global barrier because
their state and outputs are disjoint.

Traverse chunks in increasing order and keep existing per-product accumulation,
BF16 rounding, normalization grouping, and gate arithmetic. Removing carry copies
does not remove a numerical conversion. Keep existing input/output indexing and
beta boundary zero filling. The fixed token count contains complete chunks;
the host ABI still rejects unsupported shapes. This recipe introduces no tail
policy or weaker numerical requirement.

Check the resulting bundle with `klineage.harness.evaluate(kernel, Path.cwd())`
using its unchanged problem and compile flags. Inspect the compiled host launch
path for a single recurrence dispatch and the device body for its chunk loop.

## Example configuration

Preserve batch 1, 4096 tokens, 96 heads, head dimension 128, chunk size 16, and
256 chunks. The recurrence grid remains `(1,96)`, with 256 threads (eight warps)
per CTA. Each warp owns 16 value columns. Replace 256 recurrence launches with
one, following the unchanged beta and preparation launches on the caller stream.
The preparation grid remains `(256,96)` with 256 threads per CTA.

Retain row-major BF16 shared state `[128,128]`, BF16 chunk intermediates/output,
FP32 accumulators, and the initial FP32-to-BF16 and final BF16-to-FP32 state
conversions. Preserve `mma.sync.aligned.m16n8k16`, scalar fragment transfers,
shared transpose scratch, scalar BF16 arithmetic/conversions, normalization
exchanges, triangular inversion, and synchronous chunk staging.

Keep `sizeof(RecurShared) == 124672`, `sizeof(InputShared) == 18048`,
`sizeof(PrepareShared) == 42368`, the 1024-byte transpose scratch array, existing
alignment, launch bounds, workspace sizes, and compilation flags unchanged.
The removed global carry allocation is
`kHeads*kStateElems*sizeof(BF16)` (3145728 bytes here); the resident state already
occupies `kStateElems*sizeof(BF16)` (32768 bytes) inside `RecurShared`.
These sizes and retained tensor-core features describe this replay, not additional
requirements of temporal fusion.

# Precondition

- Data types: state handoffs must preserve the carried representation without
  required conversion or rounding. Otherwise removing a handoff changes the
  operator's arithmetic. Fusion itself does not require a particular dtype;
  retain all conversions inside chunk computation and at initial/final boundaries.
- Layout: consecutive chunks must agree on shared-state indices and allow one
  CTA to own a sequence exclusively. Without stable ownership, an update could
  need state owned by another CTA. Physical strides need only retain the existing
  mapping; temporal fusion adds no alignment or contiguity requirement.
- Storage: each chunk already stages its state in CTA shared memory, exporting it
  solely for the next chunk. That storage can remain live for the sequence.
  An external reader requiring an intermediate global snapshot would make its
  removal incorrect.
- Pipeline: input preparation must finish before recurrence starts. Chunk
  dependencies must be confined to the same sequence's preceding state.
  Participating threads must reach barriers publishing copied inputs and updated
  state, and finishing reads before input/output storage is reused. Those barriers
  replace the relevant visibility and completion effects of chunk boundaries;
  they cannot satisfy a dependency requiring a cross-CTA barrier.
- Hardware: no additional hardware feature or capacity is required beyond the
  existing legal CTA. The same shared allocation and barriers remain in use for
  longer; fusion neither adds simultaneous buffers nor depends on another CTA
  being resident or making progress.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace the complete device definition and host launch shown below. The Overview
lists the carry-only fields, allocation, assignment, and helpers to delete.
All names and retained helpers already exist in the deoptimized bundle.

## Before

```cuda
__global__ __launch_bounds__(kRecurThreads) void recurrence(
    const __grid_constant__ RecurArgs args, int tile) {
    extern __shared__ __align__(128) unsigned char raw[];
    auto& s = *reinterpret_cast<RecurShared*>(raw);
    int tid = threadIdx.x, warp = tid/kWarp;
    int head = blockIdx.y;

    if (tile == 0) {
        load_state(s,args,head);
        __syncthreads();
        state_in(s);
    } else load_carry(s,args,head);
    __syncthreads();
    __syncthreads();

    // Each launch copies, computes, and publishes one chunk.
    load_input(s.input,args,tile,head);
    __syncthreads();
    if (warp < kComputeWarps) recur_tile(s,s.input,s.out);
    __syncthreads();
    store_output(s.out,args,tile,head);
    __syncthreads();

    // Finish shared-state updates before export or the final conversion.
    __syncthreads();
    if (tile == kTiles - 1) {
        state_out(s);
        __syncthreads();
        store_state(s,args,head);
    } else store_carry(s,args,head);
    __syncthreads();
}

// solution/native.cu, inside launch():
for (int tile = 0; tile < kTiles; ++tile)
    recurrence<<<dim3(1,kHeads),kRecurThreads,sizeof(RecurShared),stream>>>(recur,tile);
```

## After

```cuda
__global__ __launch_bounds__(kRecurThreads) void recurrence(
    const __grid_constant__ RecurArgs args) {
    extern __shared__ __align__(128) unsigned char raw[];
    auto& s = *reinterpret_cast<RecurShared*>(raw);
    int tid = threadIdx.x, warp = tid/kWarp;
    int head = blockIdx.y;

    load_state(s,args,head);
    __syncthreads();
    state_in(s);
    __syncthreads();
    __syncthreads();

    // Retain this head's state while processing chunks in order.
    for (int tile = 0; tile < kTiles; ++tile) {
        load_input(s.input,args,tile,head);
        __syncthreads();
        if (warp < kComputeWarps) recur_tile(s,s.input,s.out);
        __syncthreads();
        store_output(s.out,args,tile,head);
        __syncthreads();
    }

    // Convert only after the last update, preserving existing rounding.
    __syncthreads();
    state_out(s);
    __syncthreads();
    store_state(s,args,head);
    __syncthreads();
}

// solution/native.cu, inside launch():
recurrence<<<dim3(1,kHeads),kRecurThreads,sizeof(RecurShared),stream>>>(recur);
```
