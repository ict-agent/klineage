---
skill_id: kda.tma-preparation-state-transfer
intent: Offload preparation and boundary-state global/shared transfers to TMA.
preconditions:
- 'Data types: tensor-map encodings must copy each stored representation without conversion;
  otherwise staging changes values or rounding. No particular arithmetic dtype or
  reduction order is required by the copy technique.'
- 'Layout: each CTA owns a transfer region with known strides and matching shared
  readers/writers. The region must admit descriptors within the encoder''s rank, extent
  and box limits: contiguous innermost elements, 16-byte-aligned global bases and
  outer byte strides, innermost box bytes divisible by 16, and 128-byte-aligned shared
  box bases. Violations misaddress data or make tensor copies invalid.'
- 'Storage: data already moves between device global memory and CTA-local shared buffers;
  those allocations must remain valid throughout transfers. TMA cannot replace register-only
  exchanges or communicate through another CTA''s ordinary local pointers.'
- 'Pipeline: producers must finish before consumers read, all participating CTA threads
  must reach publication barriers, and transfer sources/destinations must remain unmodified
  until readers/copies finish. These dependencies prevent stale reads and premature
  buffer reuse; no additional stage count is required.'
- 'Hardware: SM90-or-newer TMA with tensor-map driver encoding, transaction-count
  mbarriers and bulk async groups; CTA shared capacity must cover existing live buffers
  plus 8-byte barrier slots aligned to 8 bytes; tensor-map objects need 64-byte alignment.
  Without these facilities the transfer instructions cannot execute or their completion
  cannot be tracked.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace cooperative global/shared copies with Tensor Memory Accelerator (TMA)
loads and stores in preparation and initial/final-state transfer. One issuer
submits each descriptor box; computation retains its existing shared addresses.
This restores one transfer mechanism across its remaining call sites.

In `solution/prepare.cuh`, replace `load_prepare` and `store_prepare` with the
TMA blocks below. In `solution/recurrence.cuh`, replace `load_state` and
`store_state`. Delete those four cooperative helper definitions. Keep
`load_input`, `store_output`, `state_in`, `state_out`, `invert`, `recur_tile`,
and every arithmetic helper unchanged.

## Example configuration

Replay the existing specialization: batch 1, 4096 tokens, 96 heads, dimension
128, chunks of 16, 256 chunks/head, and 32 beta elements per transfer. Preparation
uses 256 threads; recurrence uses 192, with four compute warps. Keep
`prepare_beta` and its FP32-to-BF16 head-major conversion. Keep BF16 operands,
intermediates and recurrent working state, FP32 accumulation/conversion staging,
all quantization points, scalar BF16 additions, normalization reduction grouping,
approximate nonlinear functions and fixed scalar constants.

Keep the launches on the caller stream:
`prepare<<<dim3(kTiles,kHeads),kPrepareThreads,sizeof(PrepareShared),stream>>>`
and `recurrence<<<dim3(1,kHeads),kRecurThreads,sizeof(RecurShared),stream>>>`.
Keep compile flags, launch bounds, per-stream scratch allocation, workspace
partitioning and pybind11 ABI unchanged. There is one input buffer and no
inter-chunk copy/compute overlap. Shared sizes remain 42368 bytes for preparation,
18048 for `InputShared`, and 124672 for recurrence. The reserved `ready` and
`state_ready` slots become barriers; retain their existing 16-byte alignment.
Do not change the structure layouts or add buffers.

Preparation CTA `(tile,head)` owns one chunk; recurrence CTA `(0,head)` owns that
head's state and processes chunks in order. Input q/k/g use contiguous BTHD
storage. Beta scratch is `[head,token]`. Workspace tensors are contiguous
`[head*kTiles+tile,row,col]`; state is `[head,value,key]`. Preparation q/k/gate
shared inputs are row-major. Prepared matrices and both state buffers retain
`offset<Rows>(row,col) = row*8 + (col&7) + (col/8)*Rows*8`.
The existing shared writers and readers agree on these eight-column slabs.

## Host descriptors and transfer coverage

Add `<cuda.h>` to `native.cuh` and `<dlfcn.h>` to `native.cu`. Change
`PrepareArgs` to `CUtensorMap q, k, beta, g, bias, kd, qd, kr, gt, inv, mqk;`.
Change only `RecurArgs.state_in` and `.state_out` to `CUtensorMap`; all other
recurrence fields stay pointers. Both kernels retain `const __grid_constant__`
arguments so descriptors reside in kernel parameter storage. `CUtensorMap`
provides descriptor alignment; do not pack these structures.

Add the encoder below inside `native.cu`'s anonymous namespace. In `launch`,
replace assignments to preparation fields and the two state pointers with the
request block below. Keep all ordinary recurrence pointer assignments. Rebuild
every descriptor from the current pointers before launching; propagate encoding
errors without reading device tensor values or synchronizing the host.

The encoder uses these dimensions (fastest first); strides are bytes.
`D=kDim`, `C=kChunk`, `T=kTokens`, `H=kHeads`, `W=kHeadTiles`,
`b=sizeof(BF16)`, `f=sizeof(float)`. Element strides are all one.

| Fields / kind | Global dimensions | Outer strides | Box dimensions |
| --- | --- | --- | --- |
| q/k/g: PlainInput | `(D,T,H)` | `(D*H*b,D*b)` | `(D,C,1)` |
| beta: Beta | `(T*H)` | none | `(kBetaElems)` |
| bias: Bias | `(D,H)` | `(D*f)` | `(D,1)` |
| kd/qd/kr: Workspace | `(D,C,W)` | `(D*b,C*D*b)` | `(8,C,1)` |
| inv/mqk: Matrix | `(C,C,W)` | `(C*b,C*C*b)` | `(8,C,1)` |
| gt: Gate | `(D,W)` | `(D*f)` | `(D,1)` |
| state_in/state_out: State | `(D,D,H)` | `(D*f,D*D*f)` | `(8,D,1)` |

Use BF16 descriptors except FP32 for bias/gt/state. Keep interleave and swizzle
NONE, L2 promotion `L2_128B`, floating OOB fill NONE, and the load cache policy
shown below. Do not substitute FTZ or TF32 encodings. The BF16 beta box extends
past the allocation only for the final head/chunk; TMA must zero-fill those
lanes, matching `src < kTokens*kHeads`. Other boxes are entirely in bounds.
Only the first chunk's worth of beta entries is consumed. No new tail path is
needed for this fixed workload.

For a slab transfer, issue one box per column `c=0,8,...,Cols-8` at global
coordinate `(c,row,tile)` and shared pointer `base+c*Rows` in elements. This
maps each contiguous box row to `offset<Rows>` without introducing a swizzle.
Plain preparation loads instead issue one full-width box to row-major storage.

## Ordering and completion

Preparation thread 0 initializes `ready` with one arrival, publishes its
initialization, sets expected bytes to
`3*kTileBytes + kBetaElems*sizeof(BF16) + kGateBytes`, and issues five loads.
Every thread waits for phase 0 before normalizing. For recurrence, one elected
lane in warp `kComputeWarps` initializes `state_ready`, expects
`kStateElems*sizeof(float)` bytes, and issues all state slabs. Every thread
waits before `state_in` reads or overwrites the staging buffer.

All shared producers execute `async_fence()` before the CTA barrier preceding
TMA stores. Preparation thread 0 issues and commits the six output groups;
`wait_store()` on that thread finishes source reads before CTA exit. Recurrence
uses one elected lane in warp `kComputeWarps+1` for the final-state store group.
Its source has no later writers or reuse in the kernel; retain the final CTA
barrier and kernel completion. Same-stream ordering makes preparation output
visible to recurrence and final outputs visible to subsequent callers. Do not
replace a source-read wait with a global-consumer visibility assumption.

The recurrence chunk loop and its load/compute/store barriers are unchanged.
This replay introduces no buffer rotation or producer/consumer pipeline stages.
TMA alignment and completion rules are documented in the
[CUDA asynchronous-copy guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html).

# Precondition

- Data types: choose tensor-map encodings that preserve each element's bits.
  A converting or flushing encoding changes the data seen by unchanged consumers.
  The copy mechanism adds no arithmetic dtype or reduction-order requirement;
  the example's arithmetic and rounding remain its independent numerical contract.
- Layout: each CTA's transfer region has known strides and shared ownership.
  Descriptor boxes must reproduce the addresses used by its readers and writers.
  Innermost elements are contiguous; global bases and outer byte strides are
  multiples of 16 bytes; innermost box bytes are multiples of 16; shared box bases
  are aligned to 128 bytes. Illegal descriptors or misaligned boxes cannot perform
  this transfer. Tensor ranks, extents and boxes must fit the encoder's limits;
  oversized regions may be split into legal boxes. No particular slab width or
  thread count is intrinsic to the technique.
- Storage: the preceding implementation already transfers between device global
  memory and CTA-local shared buffers. These allocations must outlive their
  transfers. Tensor copies use those memory spaces, so register-only exchanges
  and another CTA's ordinary local addresses are outside this transformation.
- Pipeline: producers finish before consumption, and every thread participating
  in a CTA publication barrier reaches it. Sources remain stable until reads
  finish; destinations are not consumed or reused before copies finish. These
  existing dependencies must also hold with asynchronous completion, preventing
  stale data and overwrites. No additional stage count is required.
- Hardware: the device supports SM90-or-newer TMA, the driver can encode tensor
  maps, and transaction-count mbarriers and bulk async groups can track completion.
  Shared capacity covers the simultaneously live buffers plus aligned barrier
  storage; otherwise the launch or transfer cannot execute. Barriers require
  8-byte slots aligned to 8 bytes; tensor-map objects require 64-byte alignment.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

The four cooperative helpers currently copy bits with CTA-strided loops. For
example, `load_state` contains:

```cuda
for (int i = threadIdx.x; i < kStateElems; i += kRecurThreads)
    s.fp32[fp32_offset(i / kDim, i % kDim)] = args.state_in[head * kStateElems + i];
```

Replace these sites, keeping the intervening computation:

```cuda
// prepare: input site
load_prepare(s,args,tile,head);
float gain = expf(a_log[head]);
__syncthreads();
__syncthreads();

// prepare: output site
invert(s,tid);
__syncthreads();
store_prepare(s,args,tile,head);
__syncthreads();

// recurrence: initial-state site
load_state(s,args,head);
__syncthreads();
state_in(s);
__syncthreads();
__syncthreads();

// recurrence: final-state site
__syncthreads();
state_out(s);
__syncthreads();
store_state(s,args,head);
__syncthreads();
```

## After

Add these constants in namespace `kda` in `native.cuh`, retaining all other
constants. The transaction sizes below evaluate to 12864 and 65536 bytes for
preparation and initial-state transfer, respectively.

```cuda
constexpr int kPrepareTx = 3 * kTileBytes + kBetaElems * sizeof(BF16) + kGateBytes;
constexpr unsigned kWaitTicks = 0x989680;
constexpr int kLoadWarp = kComputeWarps;
constexpr int kStoreWarp = kLoadWarp + 1;
constexpr uint64_t kEvictNormal = 0x1000000000000000;
```

Add these transfer helpers in `native.cuh`, after `sigmoid` (where `shared_addr` is already defined):

```cuda
__device__ __forceinline__ void async_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void init_bar(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(shared_addr(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void expect_bar(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(shared_addr(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait_bar(uint64_t* bar, unsigned phase) {
    asm volatile("{ .reg .pred p; again: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; @!p bra again; }"
      :: "r"(shared_addr(bar)), "r"(phase), "r"(kWaitTicks) : "memory");
}
__device__ __forceinline__ unsigned elect() {
    unsigned leader;
    asm volatile("{ .reg .pred p; elect.sync _|p, %1; selp.u32 %0, 1, 0, p; }" : "=r"(leader) : "r"(kAllLanes));
    return leader;
}

// Each call issues one descriptor box; callers retain expert transfer groups.
__device__ __forceinline__ void load3(const CUtensorMap& map, void* dst, uint64_t* bar, int x, int y, int z) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
      :: "r"(shared_addr(dst)), "l"(&map), "r"(shared_addr(bar)), "r"(x), "r"(y), "r"(z), "l"(kEvictNormal) : "memory");
}
__device__ __forceinline__ void load2(const CUtensorMap& map, void* dst, uint64_t* bar, int x, int y) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
      :: "r"(shared_addr(dst)), "l"(&map), "r"(shared_addr(bar)), "r"(x), "r"(y), "l"(kEvictNormal) : "memory");
}
__device__ __forceinline__ void load1(const CUtensorMap& map, void* dst, uint64_t* bar, int x) {
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3}], [%2], %4;"
      :: "r"(shared_addr(dst)), "l"(&map), "r"(shared_addr(bar)), "r"(x), "l"(kEvictNormal) : "memory");
}
__device__ __forceinline__ void store3(const CUtensorMap& map, const void* src, int x, int y, int z) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
      :: "l"(&map), "r"(shared_addr(src)), "r"(x), "r"(y), "r"(z) : "memory");
}
__device__ __forceinline__ void store2(const CUtensorMap& map, const void* src, int x, int y) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      :: "l"(&map), "r"(shared_addr(src)), "r"(x), "r"(y) : "memory");
}
__device__ __forceinline__ void commit_store() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void wait_store() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}
template<int Rows, int Cols, class T> __device__ __forceinline__ void load_slabs(
    const CUtensorMap& map, T* dst, uint64_t* bar, int row, int tile) {
    #pragma unroll
    for (int c = 0; c < Cols; c += 8) load3(map, dst + c * Rows, bar, c, row, tile);
}
template<int Rows, int Cols, class T> __device__ __forceinline__ void store_slabs(
    const CUtensorMap& map, const T* src, int row, int tile) {
    #pragma unroll
    for (int c = 0; c < Cols; c += 8) store3(map, src + c * Rows, c, row, tile);
    commit_store();
}
```

Add the host encoder in `native.cu`:

```cuda
using Encode = decltype(&cuTensorMapEncodeTiled);

Encode encoder() {
    static void* driver = dlopen("libcuda.so.1",RTLD_NOW | RTLD_LOCAL);
    static Encode encode = driver ? reinterpret_cast<Encode>(dlsym(driver,"cuTensorMapEncodeTiled")) : nullptr;
    return encode;
}

enum class MapKind { PlainInput, Workspace, Matrix, Gate, Bias, Beta, State };

CUresult encode_map(CUtensorMap& map, const void* data, MapKind kind) {
    cuuint64_t shape[3] = {kDim,kTokens,kHeads};
    cuuint64_t strides[2] = {kDim*kHeads*sizeof(BF16),kDim*sizeof(BF16)};
    cuuint32_t box[3] = {8,kChunk,1}, elem[3] = {1,1,1};
    unsigned rank = 3;
    CUtensorMapDataType type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    switch (kind) {
        case MapKind::PlainInput: box[0] = kDim; break;
        case MapKind::Workspace:
            shape[1] = kChunk; shape[2] = kHeadTiles;
            strides[0] = kDim*sizeof(BF16); strides[1] = kTileBytes;
            break;
        case MapKind::Matrix:
            shape[0] = kChunk; shape[1] = kChunk; shape[2] = kHeadTiles;
            strides[0] = kChunk*sizeof(BF16); strides[1] = kMatrixBytes;
            break;
        case MapKind::Gate:
        case MapKind::Bias:
            rank = 2; type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
            shape[1] = kind == MapKind::Gate ? kHeadTiles : kHeads;
            strides[0] = kGateBytes; box[0] = kDim; box[1] = 1;
            break;
        case MapKind::Beta:
            rank = 1; shape[0] = kTokens*kHeads; box[0] = kBetaElems;
            break;
        case MapKind::State:
            type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
            shape[1] = kDim; strides[0] = kDim*sizeof(float); strides[1] = kStateElems*sizeof(float);
            box[1] = kDim;
            break;
    }
    auto encode = encoder();
    if (!encode) return CUDA_ERROR_NOT_INITIALIZED;
    return encode(&map,type,rank,const_cast<void*>(data),shape,strides,box,elem,
                  CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
```

Replace the pointer assignments identified above with:

```cuda
struct Request { CUtensorMap* map; const void* data; MapKind kind; };
    Request requests[] = {
        {&prep.q,inputs[0],MapKind::PlainInput}, {&prep.k,inputs[1],MapKind::PlainInput},
        {&prep.beta,b->beta,MapKind::Beta}, {&prep.g,inputs[3],MapKind::PlainInput},
        {&prep.bias,inputs[7],MapKind::Bias}, {&prep.kd,kd,MapKind::Workspace},
        {&prep.qd,qd,MapKind::Workspace}, {&prep.kr,kr,MapKind::Workspace},
        {&prep.gt,gt,MapKind::Gate}, {&prep.inv,inv,MapKind::Matrix}, {&prep.mqk,mqk,MapKind::Matrix},
        {&recur.state_in,inputs[9],MapKind::State},
        {&recur.state_out,outputs[1],MapKind::State}
    };
    for (auto r : requests) {
        if (encode_map(*r.map,r.data,r.kind) != CUDA_SUCCESS) return cudaErrorInvalidValue;
    }
```

In `prepare`, add `int token = tile*kChunk;`, then replace the input site with:

```cuda
if (tid == 0) {
        init_bar(&s.ready,1);
        init_fence();
        expect_bar(&s.ready,kPrepareTx);
        load3(args.q,s.q,&s.ready,0,token,head);
        load3(args.k,s.k,&s.ready,0,token,head);
        load1(args.beta,s.beta,&s.ready,head*kTokens+token);
        load3(args.g,s.gate,&s.ready,0,token,head);
        load2(args.bias,s.bias,&s.ready,0,head);
    }
    float gain = expf(a_log[head]);
    __syncthreads();
    wait_bar(&s.ready,0);
    async_fence();
    __syncthreads();
```

Replace its output site with:

```cuda
invert(s,tid);
    async_fence();
    __syncthreads();

    if (tid == 0) {
        int ws = head*kTiles+tile;
        store_slabs<kChunk,kDim>(args.kd,s.kd,0,ws);
        store_slabs<kChunk,kDim>(args.qd,s.qd,0,ws);
        store_slabs<kChunk,kDim>(args.kr,s.kr,0,ws);
        store2(args.gt,s.gt,0,ws);
        commit_store();
        store_slabs<kChunk,kChunk>(args.inv,s.inv,0,ws);
        store_slabs<kChunk,kChunk>(args.mqk,s.mqk,0,ws);
    }
    wait_store();
    __syncthreads();
```

In `recurrence`, add `unsigned leader = elect();` before its initial-state site.
All lanes call `elect()` before branching; exactly one elected lane issues each
state transfer group. Replace the initial-state site with:

```cuda
if (warp == kLoadWarp && leader) {
        init_bar(&s.state_ready,1);
        init_fence();
        expect_bar(&s.state_ready,kStateElems*sizeof(float));
        load_slabs<kDim,kDim>(args.state_in,s.fp32,&s.state_ready,0,head);
    }
    __syncthreads();
    wait_bar(&s.state_ready,0);
    async_fence();
    state_in(s);
    __syncthreads();
    __syncthreads();
```

Replace its final-state site with:

```cuda
// Finish all chunk accesses before converting the final state.
    __syncthreads();
    state_out(s);
    async_fence();
    __syncthreads();
    if (warp == kStoreWarp && leader) store_slabs<kDim,kDim>(args.state_out,s.fp32,0,head);
    __syncthreads();
```
