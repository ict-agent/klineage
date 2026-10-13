---
skill_id: kda.recurrence-input-shared-staging
intent: Reuse recurrence inputs across value warps through cooperative shared-memory
  staging.
preconditions:
- 'Data types: staging must preserve each operand''s representation; changing bits
  or arithmetic order would change the consumers'' results. No particular arithmetic
  dtype is required by the copy technique.'
- 'Layout: known global strides and consumer coordinates must permit an in-bounds
  cooperative copy and matching shared reads; within-CTA consumers reuse operands.
  Scalar copies require only natural element alignment, not contiguous rows across
  heads.'
- 'Storage: operands remain globally visible and unchanged until their last consumer
  read; otherwise a shared snapshot can differ from direct loads. This staging and
  barrier protocol serves consumers within one CTA.'
- 'Pipeline: global producers finish before copying; all CTA threads reach a barrier
  after their writes and before consumption, and all readers finish before storage
  reuse. Otherwise readers can see incomplete or overwritten operands.'
- 'Hardware: CUDA CTA shared memory and block barriers must be available; the sum
  of staged operand bytes, other live shared storage, and padding must fit the CTA
  allocation limit.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Stage the recurrence's read-only inputs once per CTA, then gather fragments from
shared memory. This reuses `kd`, `qd`, `kr`, `inv`, `mqk`, and gate totals across
value warps; `v` travels through the same cooperative input phase.

Edit `solution/recurrence.cuh`. Replace `load_global_frag` with `load_input` below.
In `recurrence`, call `load_input(s.input,args,tile,head)` immediately before the
existing barrier preceding `recur_tile`. Every thread participates in copying.
Keep that barrier and every other barrier, warp synchronization, launch, and
arithmetic operation.

In `recur_tile`, replace the four address declarations `ws`, `tile_base`,
`matrix_base`, and `value_base` with `InputShared& in = s.input;`. Keep its current
signature and call, including access to beta through `args.beta`.
Replace these seven expressions at their existing consumption sites:

| Before | After |
| --- | --- |
| `load_global_frag<FragView::Normal>(args.kd+tile_base,0,k*kChunk,lane)` | `load_a<kChunk>(in.kd,0,k*kChunk,lane)` |
| `load_global_frag<FragView::Normal>(args.qd+tile_base,0,k*kChunk,lane)` | `load_a<kChunk>(in.qd,0,k*kChunk,lane)` |
| `load_global_frag<FragView::Normal, kHeads*kDim>(args.v+value_base,0,value,lane)` | `load_a<kChunk>(in.v,0,value,lane)` |
| `load_global_frag<FragView::Normal, kChunk>(args.inv+matrix_base,0,0,lane)` | `load_a<kChunk, kChunk>(in.inv,0,0,lane)` |
| `load_global_frag<FragView::Normal, kChunk>(args.mqk+matrix_base,0,0,lane)` | `load_a<kChunk, kChunk>(in.mqk,0,0,lane)` |
| `load_global_frag<FragView::Transposed>(args.kr+tile_base,0,m*kChunk,lane)` | `load_t<kChunk>(in.kr,m*kChunk,0,lane)` |
| `ld_global_scalar(args.gt+ws*kDim+m*kChunk+group+(j%2)*kHalfChunk)` | `ld_scalar(in.gt+m*kChunk+group+(j%2)*kHalfChunk)` |

`load_t` exchanges the row and column arguments internally. The table preserves
the existing transposed fragment's element ownership. `load_a` preserves the
normal fragment's ownership. Retain their scalar volatile shared loads and keep
`ld_global_scalar` for its other callers. No compiler flag changes are needed.
In `solution/native.cuh`, update the comment above `RecurShared::input` to describe
its restored role; keep the declarations and allocations unchanged.

The global workspace uses head/chunk-major matrices. With `ws=head*kTiles+tile`,
the BF16 tile base is `ws*kTileElems`, the BF16 square-matrix base is
`ws*kMatrixElems`, and the FP32 gate base is `ws*kDim`. Global values use BTHD:
`((tile*kChunk+row)*kHeads+head)*kDim+col`. Shared tiles are row-major with row
stride `kDim`; shared square matrices use row stride `kChunk`. Each thread copies
linear indices `threadIdx.x + n*kRecurThreads`. These writers populate exactly
the coordinates gathered by the existing shared fragment helpers.

The preparation launch finishes before recurrence on the caller stream. Input
copies finish at the CTA barrier before any consumer runs. The shared input
remains read-only through that chunk; existing completion barriers precede
return, and subsequent chunks execute in separate launches. Preserve stream
ordering for workspace producers and state handoffs.

## Example configuration

Preserve BATCH=1, TOKENS=4096, HEADS=96, HEAD_DIM=128, chunk=16, and 256 chunks.
The preparation grid is `(256,96)` with 256 threads; each recurrence launch uses
grid `(1,96)` with 256 threads and one head/chunk per CTA. Eight warps each own
16 value columns, `value=warp*kChunk`. Keep all fixed-shape bounds and loops:
each tile is complete, so these snippets need no partial-tile predicates.
An adaptation with partial tiles must guard global copies, initialize invalid
shared entries, preserve valid arithmetic, and keep all threads at barriers.

The shared input holds four 16x128 BF16 tiles, two 16x16 BF16 matrices, and 128
FP32 gate totals: 17,920 live bytes. Its existing reserved beta region and padding
bring `sizeof(InputShared)` to 18,048. Retain `sizeof(RecurShared)=124,672`, its
offsets and dynamic allocation, and existing transpose scratch. Compilation
reserves 2,048 static shared bytes for recurrence. Preparation retains its
41,856-byte dynamic allocation and 3,072 static shared bytes. Restore no beta
staging or additional allocation.

Retain the BF16 operand bits, FP32 gates and accumulators, BF16 rounding points,
scalar conversions/additions, tensor-core `m16n8k16` operations, product/reduction
order, shared state/correction/output staging, and shared fragment transpose.
Keep normalization, gate-prefix preparation, inversion, beta gathering,
exponential evaluation, compile flags, ABI checks, caller stream, and FP32
final-state export. The move adds no arithmetic or conversion. Preserve the
complete problem, oracle, workload, and numerical requirements.

After editing, rebuild from the complete source bundle. Check correctness and
latency with `klineage.harness.evaluate`; inspect generated recurrence code for
global-to-shared copies before its barrier and shared reads at these consumers.
Keep measurements outside this card.

# Precondition

- Data types: copies must preserve operand representation without introducing
  conversions or changing arithmetic order; otherwise consumers receive different
  values. No specific arithmetic dtype is intrinsic to staging bits.
- Layout: known strides and consumer coordinates must support in-bounds
  cooperative writes and matching shared reads. Within-CTA operand reuse supplies
  the reuse benefit. Scalar copies need natural element alignment; global rows
  need not be adjacent across heads because source strides are explicit.
- Storage: source operands must remain globally visible and unchanged until
  their last consumer read; otherwise the shared snapshot can differ from direct
  loads. This staging and barrier protocol serves consumers within one CTA.
- Pipeline: source producers must complete before copying, and all CTA threads
  must reach the copy-to-consumption barrier. It publishes every writer's data to
  every reader. All readers must finish before reuse; otherwise a later producer
  can overwrite live data. No particular stage count follows from these rules.
- Hardware: CTA shared memory and block barriers are needed for common storage
  and visibility. `sum(elements(operand)*sizeof(operand_element)) +
  other_live_shared_bytes + padding` must fit the per-CTA allocation limit;
  otherwise the simultaneous staging cannot be allocated.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These excerpts show the producer phase and one consumer. Apply the table above
to all seven consumers. Shared types and fragment helpers already exist in the
deoptimized bundle.

## Before

```cuda
// recurrence: no shared input producer.
__syncthreads();
if (warp < kComputeWarps) recur_tile(s,args,s.out,tile,head);
__syncthreads();

// recur_tile: each value warp reloads the same global key tile.
Reg ka = load_global_frag<FragView::Normal>(args.kd+tile_base,0,k*kChunk,lane);
Reg sb = load_b<kDim>(s.state,value,k*kChunk,lane);
mma(u,ka,sb);
```

## After

```cuda
__device__ __forceinline__ void load_input(
    InputShared& in, const RecurArgs& args, int tile, int head) {
    const int tid = threadIdx.x;
    const int token = tile * kChunk;
    const int ws = head * kTiles + tile;

    // Publish contiguous shared rows for the fragment consumers.
    for (int i = tid; i < kTileElems; i += kRecurThreads) {
        const int row = i / kDim, col = i % kDim;
        const int dst = offset<kChunk>(row, col);
        const int src = ((token + row) * kHeads + head) * kDim + col;
        in.v[dst] = args.v[src];
        in.kd[dst] = args.kd[ws * kTileElems + i];
        in.qd[dst] = args.qd[ws * kTileElems + i];
        in.kr[dst] = args.kr[ws * kTileElems + i];
    }

    for (int i = tid; i < kMatrixElems; i += kRecurThreads) {
        const int dst = offset<kChunk, kChunk>(i / kChunk, i % kChunk);
        in.inv[dst] = args.inv[ws * kMatrixElems + i];
        in.mqk[dst] = args.mqk[ws * kMatrixElems + i];
    }

    for (int i = tid; i < kDim; i += kRecurThreads)
        in.gt[i] = args.gt[ws * kDim + i];
}

// recurrence: all writers finish before consumers proceed.
load_input(s.input,args,tile,head);
__syncthreads();
if (warp < kComputeWarps) recur_tile(s,args,s.out,tile,head);
__syncthreads();

// recur_tile: declare once, replacing the four global address declarations.
InputShared& in = s.input;

// Existing key-product loop: preserve the state load and MMA.
Reg ka = load_a<kChunk>(in.kd,0,k*kChunk,lane);
Reg sb = load_b<kDim>(s.state,value,k*kChunk,lane);
mma(u,ka,sb);
```
