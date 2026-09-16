---
skill_id: sparse-mla.hopper-wgmma
intent: Accelerate shared-memory matrix products with warpgroup tensor-core instructions.
preconditions:
- 'Data types: operands must be BF16 and accumulators FP32 for the selected WGMMA
  opcodes; the numerical contract must allow their accumulation rounding instead of
  scalar-FMA bitwise equivalence.'
- 'Layout: shared operands must admit the selected WGMMA core layouts and 16-byte-aligned
  descriptor bases and strides; accumulator ownership must match the instruction register
  mapping across a complete 128-thread warpgroup, or operands/results are misaddressed.'
- 'Storage: Q, K, probabilities, and V must be available in CTA shared memory with
  stable addresses and descriptor-encodable offsets; these shared/shared instructions
  cannot read global operands.'
- 'Pipeline: the existing schedule must permit collective issue by every warpgroup
  thread, publication after operand production, and completion before result reads
  or buffer reuse; asynchronous execution otherwise diverges or races, including scalar
  access to in-flight accumulators.'
- 'Hardware: the compiler and GPU must support the selected Hopper WGMMA instructions;
  CTA shared-memory capacity must cover all simultaneously live operands, and launch
  resources must support the full participating warpgroup.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace scalar shared-memory matrix products in `solution/attention.cuh`
(`qk_tile`, `pv_smem`) with BF16 WGMMA and FP32 accumulators. Restore only
matrix instructions and their descriptors, fences, commit groups, and waits.
Retain the existing shared layouts, register tiles, online softmax, sparse loads,
masking, reduction grouping, and consumer partition.

## Replay

Create `solution/mma.cuh` with the After header below. Copy the NVIDIA copyright
and BSD-3-Clause notice already present at the top of `solution/hopper.cuh` to
its beginning. In `attention.cuh`, replace `#include "hopper.cuh"` with
`#include "mma.cuh"`, then replace `qk_tile` and `pv_smem` with their After bodies.
Remove the four scalar-only global constants `kPairElems`, `kLanesPerRow`,
`kColsPerStep`, and `kRegsPerStep`; keep similarly named function-local constants.

Restore ordering at these exact sites:

1. Append `shared_fence()` to `load_query` and `load_kv`, after their stores.
   The existing Stage barrier publishes all operands and validity flags to both
   consumer groups; the added proxy fence makes operand stores visible to WGMMA.
2. Append `commit()` after the final `qk_tile` call in each of `qk_left`,
   `qk_right`, and `qk_peer`. Preserve every tile call and its Clear/Add mode.
3. In `consume<0>`, replace the Local0 barrier immediately after `qk_right`
   with `wait<0>()`. This wait completes every asynchronous key reader before
   the last K0 tile is repurposed as probabilities. Keep the later Local0
   barrier after `save_prob`. In `consume<1>`, insert `wait<0>()` immediately
   after `qk_peer`, before masking its accumulator results.
4. Insert `shared_fence()` immediately after each of the three `save_prob`
   calls in `consume`, before its following Local0/Local1 barrier or Prob0
   arrival. Preserve those barriers and the peer Prob0/Prob1 rendezvous.
5. After each `pv_smem` call in `consume`, insert `commit(); wait<0>();`,
   except the final peer PV in Group 1: insert `commit(); shared_fence();`
   before its existing `arrive<Barrier::Prob1>()`, then insert `wait<0>()`
   after that arrival. Thus Group 0 can start its independent peer PV while
   Group 1's PV finishes. Both groups must wait before the closing Stage barrier
   permits Q/KV/probability storage reuse.
6. Insert `wait<0>()` at the start of `store_output`, before conversions.
   Keep every other barrier, rescaling operation, reduction, and output store.

No host ABI, launch, tensor layout, or bounds changes are needed. Inputs and
outputs stay on the caller's device and stream. Invalid sparse indices remain
zero-filled in KV and masked to negative infinity before softmax. Keep the
existing BF16 round-to-nearest conversions for probabilities and final output,
FP32 softmax/statistics, model scale, and ordered QK/PV tile traversal. WGMMA's
internal sum rounding can differ from scalar FMA; preserve the problem's
numerical requirements and evaluate with the unchanged oracle and tolerances.
Check generated code for matrix instructions and retained ordering; do not infer
instruction selection from timing alone.

## Example configuration

The supplied workload is sparse MLA prefill with 8192 tokens, 128 heads,
QK width 576, value width 512, and 2048 supplied indices. Inputs Q/KV are BF16,
indices int32; output is BF16 and maximum/LSE are FP32. The model scale is
`0.1352337788608801f`; retain `kScaleLog2`, `kInitialMax`, fast-math flags,
and all rescaling/reduction expressions.

The launch remains 16384 CTAs, 256 threads per CTA, two 128-thread consumer
groups, a single-CTA cluster, and `sizeof(Shared) == 231296` dynamic shared
bytes. Keep the existing launch bounds and compile flags; build for `sm_90a`.
Each CTA owns one token and a 64-head slice. Group 0 computes the first
256 output columns and Group 1 the second 256. Each lane owns two rows,
32 score accumulators, and 128 output accumulators. For register index `i`,
its row is `row_index((i % 4) / 2, lane)` and column is
`8 * (i / 4) + (lane % 4) * 2 + i % 2` within its matrix tile.

Preserve the current unswizzled 8-row by 8-element BF16 cores. Q and probability
cores use `q_index`/`prob_index`; KV uses `kv_index`. All shared arrays have
16-byte-aligned tile bases. Q/K columns are split into nine 64-column tiles;
PV spans four adjacent 64-column value tiles per consumer. QK uses
`m64n64k16`, and PV uses `m64n256k16`, each with four K steps per call.
The descriptors below encode these exact writer layouts. Q and P use K-major
cores, K uses its existing K-major reader mapping, and V uses the matching
MN-major mapping with the PV transpose operand set to 1.

Keep Q's 73728 bytes, two KV buffers totaling 147456 bytes, the 8192-byte
probability buffer, 128 validity bytes, 256 maximum bytes, 512 sum bytes,
and 1024 reduction bytes. P0 reuses K0's final 64-column RoPE tile after QK
completes. P1's shared tile also serves Group 1's local PV. Preserve this reuse.
Each of 16 iterations processes two 64-key tiles. Group 0's QK visits columns
0 through 575; Group 1 visits 256 through 575 then 0 through 255. PV visits K
in increasing order. Keep scalar volatile KV transfers, sparse-index reloads,
query reloads per pair, shared peer reductions, and per-element exponentials
and normalization. These settings are replay details, not extra WGMMA prerequisites.

# Precondition

- Data types: the selected instruction suffixes require BF16 operands and FP32
  accumulators. Replacing serial FMA with their internal accumulation requires
  tolerance for its rounding; strict scalar bitwise equivalence would invalidate
  this substitution. Preserve representation and every conversion outside MMA.
- Layout: the shared arrays must admit WGMMA's selected core organization, with
  descriptor bases and encoded strides aligned to 16 bytes. Each complete
  128-thread warpgroup must own accumulator registers in the instruction's
  mapping. Incorrect descriptor strides or register ownership select wrong
  operands or output coordinates. The After descriptors match the preceding
  scalar kernel's existing core layouts; no global alignment restriction is added.
- Storage: all shared/shared operands must already be available in CTA shared
  memory. Their addresses and descriptor-encodable offsets must remain valid
  throughout asynchronous consumption; the selected instructions cannot address
  global storage. No extra shared buffer is introduced by this substitution.
- Pipeline: the preceding schedule must allow every warpgroup thread to issue
  matching collective operations. Operand producers must finish before readers;
  addresses and contents must remain stable until those readers finish. There
  must be boundaries for publishing completed stores and delaying result reads
  or buffer reuse until completion. Scalar access to in-flight accumulators must
  be avoidable. Otherwise asynchronous execution diverges or races. The existing
  rendezvous permit the ordering calls specified under Replay; no fixed stage
  count is needed.
- Hardware: compilation and execution must support these Hopper WGMMA opcodes.
  Available shared memory must cover the simultaneously live operand allocations;
  launch resource limits must permit every participating warpgroup thread.
  Otherwise instructions cannot issue legally or the CTA cannot launch. No
  particular tuning budget or workload extent is an additional prerequisite.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

These two functions are in `solution/attention.cuh`, inside `namespace spa`.
Their constants and layout helpers already exist in the deoptimized bundle.

```cuda
template<int Buffer, int Tile, Accum Mode>
__device__ __forceinline__ void qk_tile(Shared& sm, float (&p)[kScoreRegs]) {
    const int lane = threadIdx.x % kWarpgroup;
    const Bf16* q = sm.q_o + Tile * kTileElems;
    const Bf16* key = sm.kv[Buffer] + Tile * kTileElems;

    if constexpr (Mode == Accum::Clear) {
#pragma unroll
        for (int i = 0; i < kScoreRegs; ++i) p[i] = 0.0f;
    }

    // Retain each lane's output tile and accumulate directly from shared memory.
#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
#pragma unroll
        for (int row = 0; row < kRowsPerThread; ++row) {
            const float a = __bfloat162float(q[q_index(row_index(row, lane), k)]);
#pragma unroll
            for (int i = row * kPairElems; i < kScoreRegs; i += kRegsPerStep) {
                const int col = kColsPerStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
                p[i] = fmaf(a, __bfloat162float(key[kv_index(col, k)]), p[i]);
                p[i + 1] = fmaf(a, __bfloat162float(key[kv_index(col + 1, k)]), p[i + 1]);
            }
        }
    }
}

__device__ __forceinline__ void pv_smem(
    const Bf16* s, const Bf16* v, float (&o)[kOutputRegs]) {
    const int lane = threadIdx.x % kWarpgroup;

    // Preserve BF16 probabilities and the per-lane FP32 output accumulators.
#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
#pragma unroll
        for (int row = 0; row < kRowsPerThread; ++row) {
            const float a = __bfloat162float(s[prob_index(row_index(row, lane), k)]);
#pragma unroll
            for (int i = row * kPairElems; i < kOutputRegs; i += kRegsPerStep) {
                const int col = kColsPerStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
                const int base = (col / kTile) * kTileElems;
                o[i] = fmaf(a, __bfloat162float(v[base + kv_index(k, col % kTile)]), o[i]);
                o[i + 1] = fmaf(a, __bfloat162float(v[base + kv_index(k, col % kTile + 1)]), o[i + 1]);
            }
        }
    }
}
```

## After

Add this instruction/descriptor header as `solution/mma.cuh`, with the existing
NVIDIA notice as described above. It contains only the restored matrix adapters.

```cuda
#pragma once
#include "hopper.cuh"

namespace hopper {
constexpr uint32_t kDescAddressMask = 0x3fff;
constexpr int kDescUnitBytes = kCoreBytes;
constexpr int kKvCoreStride = kTile * kCoreElems * sizeof(Bf16) / kDescUnitBytes;
constexpr int kKvRowStride = kCoreTileElems * sizeof(Bf16) / kDescUnitBytes;
constexpr uint64_t kKDesc = (uint64_t(kKvCoreStride) << 16) | (uint64_t(kKvRowStride) << 32);
constexpr uint64_t kMnDesc = (uint64_t(kKvRowStride) << 16) | (uint64_t(kKvCoreStride) << 32);
constexpr int kKvMmaK = 16;
constexpr int kKStep = kKvMmaK / kCoreElems * kKvCoreStride;
constexpr int kVStep = kKvMmaK / kCoreRows * kKvRowStride;
constexpr int kProbMmaK = 16;
constexpr int kProbStep = kProbMmaK / kCoreElems * kCoreTileElems * sizeof(Bf16) / kDescUnitBytes;
constexpr uint64_t kProbDesc =
    (uint64_t(kCoreTileElems * sizeof(Bf16) / kDescUnitBytes) << 16) |
    (uint64_t(kCoreRows * kTile * sizeof(Bf16) / kDescUnitBytes) << 32);
constexpr uint64_t kQDesc =
    (uint64_t(kCoreTileElems * sizeof(Bf16) / kDescUnitBytes) << 16) |
    (uint64_t(kCoreRows * kTile * sizeof(Bf16) / kDescUnitBytes) << 32);
constexpr int kQMmaK = 16;
constexpr int kQStep = kQMmaK / kCoreElems * kCoreTileElems * sizeof(Bf16) / kDescUnitBytes;

__device__ __forceinline__ uint64_t desc_q(const void* p) {
    return kQDesc | ((shared_addr(p) >> 4) & kDescAddressMask);
}

__device__ __forceinline__ uint64_t desc_prob(const void* p) {
    return kProbDesc | ((shared_addr(p) >> 4) & kDescAddressMask);
}

__device__ __forceinline__ uint64_t desc_k(const void* p) {
    return kKDesc | ((shared_addr(p) >> 4) & kDescAddressMask);
}

__device__ __forceinline__ uint64_t desc_mn(const void* p) {
    return kMnDesc | ((shared_addr(p) >> 4) & kDescAddressMask);
}

__device__ __forceinline__ uint64_t desc_step(uint64_t descriptor, uint32_t offset) {
    const uint32_t lower = uint32_t(descriptor) + offset;
    return (descriptor & 0xffffffff00000000ULL) | lower;
}

__device__ __forceinline__ void shared_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mma_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template<int Pending>
__device__ __forceinline__ void wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(Pending) : "memory");
}

template<int N>
__device__ __forceinline__ void reg_fence(float (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i]) :: "memory");
}

__device__ __forceinline__ void qk_mma(uint64_t a, uint64_t b, float (&d)[32], Accum mode) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7"
        ", %8, %9, %10, %11, %12, %13, %14, %15"
        ", %16, %17, %18, %19, %20, %21, %22, %23"
        ", %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 0; }"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a), "l"(b), "r"(int(mode)));
}

__device__ __forceinline__ void pv_shared(uint64_t a, uint64_t b, float (&d)[128]) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7"
        ", %8, %9, %10, %11, %12, %13, %14, %15"
        ", %16, %17, %18, %19, %20, %21, %22, %23"
        ", %24, %25, %26, %27, %28, %29, %30, %31"
        ", %32, %33, %34, %35, %36, %37, %38, %39"
        ", %40, %41, %42, %43, %44, %45, %46, %47"
        ", %48, %49, %50, %51, %52, %53, %54, %55"
        ", %56, %57, %58, %59, %60, %61, %62, %63"
        ", %64, %65, %66, %67, %68, %69, %70, %71"
        ", %72, %73, %74, %75, %76, %77, %78, %79"
        ", %80, %81, %82, %83, %84, %85, %86, %87"
        ", %88, %89, %90, %91, %92, %93, %94, %95"
        ", %96, %97, %98, %99, %100, %101, %102, %103"
        ", %104, %105, %106, %107, %108, %109, %110, %111"
        ", %112, %113, %114, %115, %116, %117, %118, %119"
        ", %120, %121, %122, %123, %124, %125, %126, %127}, "
        "%128, %129, 1, 1, 1, 0, 1;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
          "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]),
          "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
          "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]),
          "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
          "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]),
          "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
          "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]),
          "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
          "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]),
          "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
          "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]),
          "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
          "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
          "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),
          "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(a), "l"(b));
}
}  // namespace hopper
```

Replace the two Before functions inside `namespace spa` with these bodies.
Apply the commit/wait/proxy-fence call-site edits listed under Replay.

```cuda
template<int Buffer, int Tile, Accum Mode>
__device__ __forceinline__ void qk_tile(Shared& sm, float (&p)[kScoreRegs]) {
    const uint64_t q = desc_q(sm.q_o + Tile * kTileElems);
    const uint64_t k = desc_k(sm.kv[Buffer] + Tile * kTileElems);
    reg_fence(p);
    mma_fence();
#pragma unroll
    for (int step = 0; step < kTile / 16; ++step)
        qk_mma(desc_step(q, step * kQStep), desc_step(k, step * kKStep), p, step == 0 ? Mode : Accum::Add);
    reg_fence(p);
}

__device__ __forceinline__ void pv_smem(
    const Bf16* s, const Bf16* v, float (&o)[kOutputRegs]) {
    const uint64_t a = desc_prob(s);
    const uint64_t b = desc_mn(v);
    reg_fence(o);
    mma_fence();
#pragma unroll
    for (int step = 0; step < kTile / 16; ++step)
        pv_shared(desc_step(a, step * kProbStep), desc_step(b, step * kVStep), o);
    reg_fence(o);
}
```
