---
skill_id: fmha.tma-warp-specialized-pipeline
intent: Overlap operand transfers and attention computation with a warp-specialized
  asynchronous pipeline.
preconditions:
- 'Data types: tensor-map transfers must support the operand representation without
  changing bits; overlapping computation must preserve required reduction order and
  rounding, and masked speculative loads must remain numerically harmless.'
- 'Layout: global operands must have tensor-map-encodable affine strides and contiguous
  inner slices, with 16-byte-aligned bases and outer byte strides divisible by 16;
  the shared layout must admit matching TMA panels with inner byte widths divisible
  by 16 and 128-byte-aligned destinations; descriptor and barrier storage must permit
  64-byte and 8-byte alignment, respectively, so encoding and copies are legal.'
- 'Storage: operands reside in device global memory and are reused from CTA-local
  shared tiles; source values must remain stable until transfers finish, and all tile
  consumers must be reachable by the same CTA synchronization.'
- 'Pipeline: independent tile transfers and compute work must exist to overlap; each
  synchronization collective must retain its participants, publication must precede
  reads, async reads must finish before reuse, and register operands/results must
  respect tracked compute completion.'
- 'Hardware: tensor-memory transfers, transaction-counted shared barriers, named CTA
  barriers, and asynchronous compute-group completion tracking must be available;
  CTA resources must fit Q plus every live K/V stage, synchronization state, scratch,
  and participating threads.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Overlap global-to-shared TMA transfers, QK/PV matrix work, and online softmax.
Replace the synchronous tile loop with a dedicated producer warp, rotating K/V
storage, and two alternating math groups. This is one pipeline transformation;
its copies, buffer rotation, participant mapping, and completion protocol travel
together.

Apply the declarations and helpers below in `solution/ops.cuh`, append the work
handoff and producer to `solution/scheduler.cuh`, and replace `consumer` and
`attention` in `solution/attention.cuh`. Restore the tensor-map encoder in
`solution/kernel.cu`. All required additions appear below; retained functions
come from the supplied deoptimized bundle.

## Layout and ownership

Global Q/K/V remain packed `[token, head, column]`. A scalar address is
`token*kRow + head*kDim + column`. Both copy paths produce unswizzled panels:
`(column/kInputPanelCols)*(Rows*kInputPanelCols) + row*kInputPanelCols
+ column%kInputPanelCols`. Q uses `Rows=kM`; K/V use `Rows=kN`. Each K/V stage
occupies `kN*kDim` elements. Keep WGMMA descriptor leading/stride arguments.

Reserve the first `kGroupSize` thread IDs for producers; only the first warp
runs `producer`. The other producer-reserved warps return. Math threads shift
up by `kGroupSize`; retain their original logical IDs by subtracting that value
in `exchange`, `consumer`, and `epilogue`. Math group `wg` still owns its Q rows
beginning at `wg*kMmaRows`. Each CTA still handles one `tile(blockIdx.x, metadata)`.
No persistent scheduler, tile reorder, or input/output swizzle is introduced.

Make these accompanying edits:

- Add `int stage` after `wg` in `query_key`; use
  `s.k + stage*kN*kDim` for its K descriptor.
- Add `int stage` after `s` in `prob_value`; use
  `s.v + stage*kN*kDim` for its V descriptor.
- Add `Shared& s` after `p` in `epilogue`; append `signal(&s.o_empty)` after
  its stores. Preserve FP16 conversion, scalar stores, row guards, and LSE writes.
- In `kernel`, remove assignments to `p.q`, `p.k`, and `p.v`. After `prepare`
  and its launch check, call `encode(p.qmap, q.data_ptr<at::Half>(), kM)`,
  `encode(p.kmap, k.data_ptr<at::Half>(), kN)`, and
  `encode(p.vmap, v.data_ptr<at::Half>(), kN)`.
- Retain the host ABI, device guard, caller stream, workspace allocation,
  `cudaFuncSetAttribute(..., sizeof(Shared))`, grid expression,
  `config.blockDim=dim3(kThreads)`, and `config.dynamicSmemBytes=sizeof(Shared)`.
  Updating the declarations therefore updates launch resources.

## Ordering and bounds

Producer lane zero expects a whole tile's byte count, then issues one TMA copy
per panel against the same full barrier. Consumers wait on that barrier before
reading. Each math-group leader releases K only after its QK group completes,
and V only after its PV group completes. Both releases are required before a
producer overwrites the slot. Slot `pos%kStages` uses full-barrier phase
`(pos/kStages)&1` and the opposite empty-barrier phase. Initialize barriers once,
publish initialization, and synchronize the CTA before splitting participants.

The producer loads the highest K tile, publishes Q, then keeps K one tile ahead
of V. Each math group queues QK for the next tile followed by PV for the previous
tile. `mma_wait<1>` makes QK results readable while PV remains eligible to run;
`mma_wait<0>` completes PV before output rescaling or probability-register reuse.
Keep operand compiler fences and WGMMA fences/commits. Named Math0/Math1 barriers
alternate the groups; WorkFull/WorkEmpty publish the work record, QueryEmpty
protects Q loading, and `o_empty` records epilogue completion. The final sentinel
handoff and producer drain remain required by this protocol despite one tile
per CTA. TMA writes already use the async proxy; the synchronous copy path's
`proxy_fence()` and per-tile CTA barriers disappear with that path.

Retain descending KV-tile order, QK/PV instruction order within each reduction,
online max/sum arithmetic, exponent scaling, scalar round-to-nearest FP16
conversions, and final normalization. Moving softmax relative to independent PV
work must not change the recurrence: finish previous PV, rescale its accumulated
output for the new maximum, then add the next PV contribution. Preserve the
partial highest-tile score mask (`-INFINITY`) and output row guards.

TMA bounds describe the whole packed tensor: globally out-of-range input is
zero-filled, while loads across a sequence boundary can read another sequence.
Such key positions are masked before exponentiation; their V contributions must
be harmless under zero probability. This workload supplies finite random FP16
values. Invalid Q rows never store output. The synchronous version explicitly
zero-fills sequence tails; restoring these speculative loads is part of replacing
its copy path. Keep the complete problem, oracle, tolerances, seed, and timing policy.
Use the Kernel evaluator for correctness and latency, with evidence outside this card.

The tensor-map and alignment rules follow the
[CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
and [CUDA asynchronous-copy guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html).

## Example configuration

This replay uses eight sequences, 16,384 packed tokens, 64 heads, and head dimension
128. Inputs and output are FP16; score, output accumulator, max, sum, and LSE are
FP32. Packed token stride is `kRow=8192` half elements (16,384 bytes); head stride
is 256 bytes. The supplied offsets have 2,048 tokens per sequence; guards remain.

Retain `kM=128`, `kN=176`, `kInputPanelCols=8`, `kCoreRows=8`, `kMmaRows=64`,
`kMmaK=16`, 88 QK registers, 64 PV registers, and 44 packed probability registers
per math thread. QK uses `m64n176k16` WGMMA; PV uses `m64n128k16`. Shared-memory
peer reductions, scalar half conversion/stores, metadata preparation, and serial
sequence lookup remain unchanged except for the logical thread-ID adjustment.

Restore two K/V stages, 384 launched threads, 256 math threads, 128-thread math
groups, and 32 active producer threads. Named work/query barriers count 288
participants; alternating math barriers count 256. Full input barriers start
with arrival count one; K/V empty barriers count two math-group leaders;
`o_empty` counts all 256 math threads. Keep the code's producer-loop unroll factor
two, consumer-loop unroll factor one, `kWaitTicks=0x989680`, and unused padding
fields for this replay. These settings specify this schedule, not general
pipeline prerequisites.

Shared operand storage changes from `kM*kDim + 2*kN*kDim` to
`kM*kDim + 2*kStages*kN*kDim` half elements. The shown declarations use 123,904
bytes before and 214,272 bytes after, including reduction scratch, barrier/work
state, and structure padding. Keep shared base alignment 1,024, structure
alignment 128, and tensor-map member alignment 64. The grid remains
`kMaxQTiles*kHeads` (8,640 CTAs); excess CTAs terminate through the work sentinel.
Retain `__launch_bounds__(kThreads,1)` and compile flags `-O3`, `-std=c++17`,
`--use_fast_math`, `--resource-usage`, `-lineinfo`, `-DNDEBUG` for SM90a.

# Precondition

- Data types: the tensor-map format must represent the stored operands so TMA
  moves their bits unchanged. Preserve arithmetic order and rounding when
  rearranging independent work; otherwise overlap can change the numerical
  contract. Speculative masked values must remain harmless (for example,
  zero probability times a finite V value); masking alone does not neutralize
  NaNs.
- Layout: tensor-map encoding needs affine byte strides, a contiguous inner
  slice, global bases aligned to 16 bytes, and outer strides divisible by 16.
  Shared tile indexing must admit matching TMA panels: a different permutation
  feeds the wrong operands. The panel's inner byte width must be divisible by
  16 for non-interleaved tensor-map encoding. Storage must permit 128-byte-aligned
  panel destinations, 64-byte-aligned descriptors, and 8-byte-aligned mbarriers
  for legal instruction addresses. These constraints require allocatable
  placement, not pre-existing descriptors or pipeline buffers.
- Storage: values currently come from device global memory into shared tiles
  reused within one CTA. Transfers require stable source storage through
  completion. All readers must be covered by that CTA's synchronization;
  the local empty barriers cannot protect readers in another CTA.
- Pipeline: overlapping operations must have independent work to execute.
  Preserve participation in each collective to avoid incomplete barriers.
  Publish descriptors, work, initialization, and tile contents before reads;
  wait for all async tile readers before overwrite. Respect compute-group
  completion before reading results or modifying register operands. These
  dependencies protect both shared buffers and register lifetimes; they do
  not impose the example's stage count.
- Hardware: TMA, transaction-counted shared mbarriers, named CTA barriers, and
  async compute-group completion tracking are needed for the new copy and wait
  protocol. Available per-CTA storage must cover
  `bytes(Q) + S*(bytes(K_tile)+bytes(V_tile)) + bytes(sync_state_and_scratch)`,
  including alignment padding for the chosen live-stage count `S`; thread and
  register limits must cover all participants. Without those resources, the
  launch or overlap protocol cannot execute.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

`solution/ops.cuh`: one buffer per operand, raw source pointers, and synchronous
cooperative copies. The current `consumer` is shown to locate the schedule being
replaced; retain its arithmetic helpers.

```cuda
struct alignas(128) Shared {
    __half v[kN * kDim];
    __half q[kM * kDim];
    __half k[kN * kDim];
    float reduction[kMathThreads];
};

struct Params {
    const __half* q;
    const __half* k;
    const __half* v;
    __half* output;
    float* lse;
    int* metadata;
    const int* q_offsets;
    const int* k_offsets;
};
```

```cuda
// Copy one panel layout synchronously; zero-fill sequence tails.
template<int Rows>
__device__ __forceinline__ void load(const __half* src, __half* dst, int valid) {
    for (int i = threadIdx.x; i < Rows * kDim; i += kThreads) {
        const int row = (i / kInputPanelCols) % Rows;
        const int col = (i / (Rows * kInputPanelCols)) * kInputPanelCols
                        + i % kInputPanelCols;
        dst[i] = row < valid ? src[row * kRow + col] : __float2half(0.f);
    }
}
```

```cuda
__device__ __forceinline__ void consumer(const Params& p, Shared& s) {
    const int tid = threadIdx.x;
    const int wg = tid / kGroupSize;
    const int4 work = tile(blockIdx.x, p.metadata);
    if (work.w >= kBatch) return;

    const int batch = p.metadata[kBatchOffset + work.w];
    const int kstart = p.k_offsets[batch];
    const int length = p.k_offsets[batch + 1] - kstart;
    const int qstart = p.q_offsets[batch] + work.y * kM;
    const int qvalid = p.q_offsets[batch + 1] - qstart;
    const int last = (length + kN - 1) / kN - 1;
    float score[kQkRegs], out[kPvRegs], maximum[2], sum[2], scale[2];
    uint32_t prob[kProbRegs];
    load<kM>(p.q + qstart * kRow + work.z * kDim, s.q, qvalid);
    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) out[i] = 0.f;

    // Finish each tile before loading another into the same storage.
    #pragma unroll 1
    for (int n = last; n >= 0; --n) {
        const int base = (kstart + n * kN) * kRow + work.z * kDim;
        const int valid = length - n * kN;
        load<kN>(p.k + base, s.k, valid);
        load<kN>(p.v + base, s.v, valid);
        proxy_fence();
        __syncthreads();

        query_key(score, s, wg);
        mma_wait<0>();
        operands(score);
        if (n == last) {
            #pragma unroll
            for (int i = 0; i < kQkRegs; ++i) {
                const int col = (tid % 4) * 2 + (i / 4) * 8 + i % 2;
                if (n * kN + col >= length) score[i] = -INFINITY;
            }
            softmax<Step::First>(score, maximum, sum, scale, s);
        } else {
            softmax<Step::Next>(score, maximum, sum, scale, s);
            rescale(out, scale);
        }
        convert(score, prob);
        prob_value(out, prob, s);
        mma_wait<0>();
        operands(out);

        // Both math groups finish reading before K/V storage is reused.
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        sum[r] += exchange(sum[r], Peer::Alternate, s);
        sum[r] += exchange(sum[r], Peer::Adjacent, s);
        scale[r] = sum[r] == 0.f || isnan(sum[r]) ? 0.f : 1.f / sum[r];
        sum[r] = maximum[r] * kScale + __logf(sum[r]);
    }
    rescale(out, scale);
    epilogue(p, work, out, sum);
}

__global__ __launch_bounds__(kThreads, 1) void attention(const __grid_constant__ Params p) {
    extern __shared__ __align__(1024) char storage[];
    auto& s = *reinterpret_cast<Shared*>(storage);
    consumer(p, s);
}
```

## After

`solution/ops.cuh`: replace `kThreads`, insert the other constants below, replace
`Shared` and `Params`, and add `Named`. Existing non-pipeline constants remain.

```cuda
constexpr int kStages = 2;
constexpr uint32_t kWaitTicks = 0x989680;
constexpr int kThreads = 384;
constexpr int kActiveThreads = kMathThreads + 32;
constexpr int kTileBytes = kN * kDim * sizeof(__half);
constexpr int kQueryBytes = kM * kDim * sizeof(__half);

enum class Named : int { Epilogue = 1, WorkEmpty = 4, WorkFull = 5, QueryEmpty = 8, Math0 = 9, Math1 = 10 };

struct alignas(128) Shared {
    __half v[kStages * kN * kDim];
    __half q[kM * kDim];
    __half k[kStages * kN * kDim];
    alignas(16) uint64_t q_full;
    alignas(16) uint64_t qv_unused;
    alignas(16) uint64_t o_empty;
    alignas(16) uint64_t k_full[kStages];
    uint64_t k_empty[kStages];
    alignas(16) uint64_t v_full[kStages];
    uint64_t v_empty[kStages];
    uint64_t unused[12];
    alignas(16) int4 work;
    float reduction[kMathThreads];
};

struct Params {
    alignas(64) CUtensorMap qmap, kmap, vmap;
    __half* output;
    float* lse;
    int* metadata;
    const int* q_offsets;
    const int* k_offsets;
};
```

Keep `shared_addr`; add these helpers immediately after it.

```cuda
__device__ __forceinline__ void sync(Named id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(int(id)), "r"(count) : "memory");
}

__device__ __forceinline__ void arrive(Named id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(int(id)), "r"(count) : "memory");
}

__device__ __forceinline__ void init(uint64_t* p, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(shared_addr(p)), "r"(count));
}

__device__ __forceinline__ void signal(uint64_t* p) {
    asm volatile("{ .reg .b32 remote; mapa.shared::cluster.u32 remote, %0, 0; "
                 "mbarrier.arrive.shared::cluster.b64 _, [remote]; }"
                 :: "r"(shared_addr(p)) : "memory");
}

__device__ __forceinline__ void wait(uint64_t* p, int phase) {
    asm volatile("{ .reg .pred done;\nloop: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n@!done bra loop; }"
                 :: "r"(shared_addr(p)), "r"(phase), "r"(kWaitTicks) : "memory");
}

__device__ __forceinline__ void ready(uint64_t* p, int phase) {
    uint32_t done;
    asm volatile("{ .reg .pred pred; mbarrier.try_wait.parity.shared::cta.b64 pred, [%1], %2; "
                 "selp.b32 %0, 1, 0, pred; }"
                 : "=r"(done) : "r"(shared_addr(p)), "r"(phase) : "memory");
    if (!done) wait(p, phase);
}

__device__ __forceinline__ void expect(uint64_t* p, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(shared_addr(p)), "r"(bytes) : "memory");
}
```

Replace the synchronous `load` with this copy path and its slot guards.

```cuda
// Compact unswizzled panels share one transaction barrier per tile.
template<int Rows>
__device__ __forceinline__ void load(const CUtensorMap& map, __half* dst,
                                    uint64_t* full, int token, int head) {
    expect(full, Rows * kDim * sizeof(__half));
    #pragma unroll
    for (int col = 0; col < kDim; col += kInputPanelCols) {
        asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4, 0}], [%5];"
                     :: "r"(shared_addr(dst + col * Rows)), "l"(&map), "r"(col),
                        "r"(token), "r"(head), "r"(shared_addr(full)) : "memory");
    }
}

__device__ __forceinline__ void load_k(const Params& p, Shared& s, int pos, int token, int head) {
    const int stage = pos % kStages;
    wait(&s.k_empty[stage], ((pos / kStages) & 1) ^ 1);
    load<kN>(p.kmap, s.k + stage * kN * kDim, &s.k_full[stage], token, head);
}

__device__ __forceinline__ void load_v(const Params& p, Shared& s, int pos, int token, int head) {
    const int stage = pos % kStages;
    wait(&s.v_empty[stage], ((pos / kStages) & 1) ^ 1);
    load<kN>(p.vmap, s.v + stage * kN * kDim, &s.v_full[stage], token, head);
}
```

`solution/scheduler.cuh`: retain `prepare` and `tile`; append these functions
inside `namespace native`.

```cuda
__device__ __forceinline__ int4 next(Shared& s) {
    sync(Named::WorkFull, kActiveThreads);
    int4 work = s.work;
    arrive(Named::WorkEmpty, kActiveThreads);
    return work;
}

__device__ __forceinline__ void producer(const Params& p, Shared& s) {
    const int lane = threadIdx.x % 32;
    int4 work = tile(blockIdx.x, p.metadata);
    if (lane == 0) s.work = work;
    arrive(Named::WorkFull, kActiveThreads);
    int pos = 0, iteration = 0;

    if (work.w < kBatch) {
        int batch = p.metadata[kBatchOffset + work.w];
        int kstart = p.k_offsets[batch];
        int length = p.k_offsets[batch + 1] - kstart;
        int qstart = p.q_offsets[batch] + work.y * kM;
        int n = (length + kN - 1) / kN - 1;
        if (lane == 0) {
            load_k(p, s, pos, kstart + n * kN, work.z);
        }
        sync(Named::QueryEmpty, kActiveThreads);
        if (lane == 0) load<kM>(p.qmap, s.q, &s.q_full, qstart, work.z);
        wait(&s.o_empty, (iteration + 1) & 1);

        // K is one tile ahead of V. Preserve the two-stage producer order.
        #pragma unroll 2
        for (--n; n >= 0; --n) {
            int prev = pos++;
            if (lane == 0) {
                load_k(p, s, pos, kstart + n * kN, work.z);
                load_v(p, s, prev, kstart + (n + 1) * kN, work.z);
            }
        }
        if (lane == 0) load_v(p, s, pos, kstart, work.z);
        ++pos;
        ++iteration;
        // End this CTA after its tile; retain the consumer handoff.
        work = make_int4(0, 0, 0, kBatch);
        sync(Named::WorkEmpty, kActiveThreads);
        if (lane == 0) s.work = work;
        arrive(Named::WorkFull, kActiveThreads);
    }
    wait(&s.o_empty, (iteration + 1) & 1);
    if (lane != 0) return;
    #pragma unroll
    for (int i = 0; i < kStages; ++i, ++pos) {
        wait(&s.k_empty[pos % kStages], ((pos / kStages) & 1) ^ 1);
        wait(&s.v_empty[pos % kStages], ((pos / kStages) & 1) ^ 1);
    }
}
```

`solution/attention.cuh`: after applying the helper signature, stage-address,
and epilogue edits above, replace `consumer` and `attention` with the following.

```cuda
__device__ __forceinline__ void consumer(const Params& p, Shared& s) {
    const int tid = threadIdx.x - kGroupSize;
    const int wg = tid / kGroupSize;
    const int leader = tid % kGroupSize == 0;
    const Named own = wg == 0 ? Named::Math0 : Named::Math1;
    const Named other = wg == 0 ? Named::Math1 : Named::Math0;
    arrive(Named::QueryEmpty, kActiveThreads);
    if (wg == 0) arrive(Named::Math0, kMathThreads);
    int pos = 0, iteration = 0;
    int4 work = next(s);
    if (work.w < kBatch) {
        const int batch = p.metadata[kBatchOffset + work.w];
        const int length = p.k_offsets[batch + 1] - p.k_offsets[batch];
        int n = (length + kN - 1) / kN - 1;
        float score[kQkRegs], out[kPvRegs], maximum[2], sum[2], scale[2];
        uint32_t prob[kProbRegs];
        wait(&s.q_full, iteration & 1);
        ready(&s.k_full[pos % kStages], (pos / kStages) & 1);
        query_key(score, s, wg, pos % kStages);
        mma_wait<0>();
        operands(score);
        if (leader) signal(&s.k_empty[pos % kStages]);
        #pragma unroll
        for (int i = 0; i < kQkRegs; ++i) {
            int col = (tid % 4) * 2 + (i / 4) * 8 + i % 2;
            if (n * kN + col >= length) score[i] = -INFINITY;
        }
        softmax<Step::First>(score, maximum, sum, scale, s);
        convert(score, prob);
        #pragma unroll
        for (int i = 0; i < kPvRegs; ++i) out[i] = 0.f;

        #pragma unroll 1
        for (--n; n >= 0; --n) {
            const int prev = pos++;
            if (wg == 0) ready(&s.k_full[pos % kStages], (pos / kStages) & 1);
            sync(own, kMathThreads);
            query_key(score, s, wg, pos % kStages);
            if (wg == 0) ready(&s.v_full[prev % kStages], (prev / kStages) & 1);
            prob_value(out, prob, s, prev % kStages);
            arrive(other, kMathThreads);
            mma_wait<1>();
            operands(score);
            if (leader) signal(&s.k_empty[pos % kStages]);
            softmax<Step::Next>(score, maximum, sum, scale, s);
            mma_wait<0>();
            operands(out);
            if (leader) signal(&s.v_empty[prev % kStages]);
            convert(score, prob);
            rescale(out, scale);
        }
        arrive(Named::QueryEmpty, kActiveThreads);
        ready(&s.v_full[pos % kStages], (pos / kStages) & 1);
        prob_value(out, prob, s, pos % kStages);
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            sum[r] += exchange(sum[r], Peer::Alternate, s);
            sum[r] += exchange(sum[r], Peer::Adjacent, s);
            scale[r] = sum[r] == 0.f || isnan(sum[r]) ? 0.f : 1.f / sum[r];
            sum[r] = maximum[r] * kScale + __logf(sum[r]);
        }
        mma_wait<0>();
        operands(out);
        if (leader) signal(&s.v_empty[pos % kStages]);
        rescale(out, scale);
        ++pos;
        ++iteration;
        const int4 previous = work;
        work = next(s);
        epilogue(p, s, previous, out, sum);
    }
}

__global__ __launch_bounds__(kThreads, 1) void attention(const __grid_constant__ Params p) {
    extern __shared__ __align__(1024) char storage[];
    auto& s = *reinterpret_cast<Shared*>(storage);
    if (threadIdx.x == 0) {
        init(&s.q_full, 1);
        init(&s.o_empty, kMathThreads);
        #pragma unroll
        for (int i = 0; i < kStages; ++i) {
            init(&s.k_full[i], 1);
            init(&s.k_empty[i], 2);
            init(&s.v_full[i], 1);
            init(&s.v_empty[i], 2);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x < kGroupSize) {
        if (threadIdx.x < kWarpSize) producer(p, s);
        return;
    }
    consumer(p, s);
}
```

`solution/kernel.cu`: insert the encoder after `check`, before `Workspace`;
replace raw-pointer setup with the three encode calls described above.

```cuda
using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

Encode encoder() {
    static Encode encode = [] {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult result;
        constexpr unsigned kTensorMapVersion = 12000;
        check(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, kTensorMapVersion,
                                               cudaEnableDefault, &result));
        TORCH_CHECK(result == cudaDriverEntryPointSuccess, "Tensor map encoder unavailable");
        return reinterpret_cast<Encode>(fn);
    }();
    return encode;
}

void encode(CUtensorMap& map, const void* ptr, int rows) {
    const cuuint64_t shape[] = {kDim, kTokens, kHeads, 1};
    const cuuint64_t strides[] = {kRow * sizeof(__half), kDim * sizeof(__half), 0};
    const cuuint32_t box[] = {kInputPanelCols, static_cast<unsigned>(rows), 1, 1};
    const cuuint32_t step[] = {1, 1, 1, 1};
    auto result = encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, const_cast<void*>(ptr),
                            shape, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "Tensor map encoding failed: ", int(result));
}
```
