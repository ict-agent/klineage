---
skill_id: fmha.wgmma-register-probabilities
intent: Forward probability fragments directly from registers into WGMMA.
preconditions:
- 'Data types: the producer representation and accumulation type must have a matching
  register-A WGMMA form; preserve operand bits, prior rounding, and reduction order
  so forwarding does not change arithmetic.'
- 'Layout: each producer thread must already own the elements, in packed fragment
  order, required by its register-A MMA lane; otherwise direct substitution selects
  wrong matrix entries. No additional global contiguity or alignment is required because
  global accesses stay unchanged.'
- 'Storage: the probability fragment is available in producer registers before its
  shared copy, and that copy serves only the corresponding PV operation; otherwise
  deleting it loses data needed by another reader. The other MMA operand must remain
  accessible through its shared descriptor.'
- 'Pipeline: all warpgroup participants must reach the same MMA sequence; producer
  writes must be ordered before consumption and fragments must remain live until completion.
  The probability publication barrier must have no independent duty; other producer
  visibility and completion-before-reuse ordering must remain intact.'
- 'Hardware: the compiler and GPU must support the selected register-A WGMMA variant
  and its 128-thread warpgroup. Per-thread and per-CTA register limits must accommodate
  the instruction operands and accumulators; no extra shared capacity is needed because
  forwarding removes a buffer.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Forward the rounded probability fragments into PV WGMMA as register operand A.
Remove their shared-memory materialization, descriptor, and publication barrier.
Keep V as shared operand B and preserve the existing MMA shapes and instruction order.

In `solution/attention.cuh`, replace `prob_value` and its call sequence with the
After snippets. Delete `store_prob`. Retain `convert`, `half_pair`, and the
non-inlined scalar FP32-to-FP16 round-to-nearest conversions in `solution/ops.cuh`.
Delete only `__half p[kM * kN];` from `Shared`; leave its other members intact.
In `solution/mma.cuh`, replace only `pv` with the register-A wrapper below.
Its accumulator outputs remain operands 0–63; A becomes four 32-bit operands
64–67 and B becomes 64-bit operand 68. Drop the shared-A transpose argument;
retain accumulation enabled, positive operand scales, and transposed B.

Each thread already owns `prob[i] = pack_rn(score[2*i], score[2*i+1])`, with
`score[2*i]` in the low half. For CTA warp `w`, lane `l`, pair `i`, its row is
`16*w + l/4 + 8*(i%2)` and first column is `8*(i/2) + 2*(l%4)`.
These are the required register fragments for the existing PV operation:
four consecutive packed words supply each K=16 slice. Advance A by four
words and B by `kMmaK` descriptor units per slice. Do not transpose, shuffle,
re-round, or recompute these fragments.

The deleted shared copy uses half-element offset
`(col/kInputPanelCols)*kM*kInputPanelCols + row*kInputPanelCols + col%kInputPanelCols`.
Its descriptor starts at `s.p + wg*kMmaRows*kInputPanelCols`, has leading/stride
fields `kM`/`kCoreRows`, and advances by
`(kMmaK/kInputPanelCols)*kM` per slice. Removing this path changes no Q/K/V
layout or lane ownership.

Keep the empty read/write inline-assembly constraints around `prob` in the After
function; they expose register dependencies to the compiler around asynchronous
MMA. Retain `mma_fence`, commit, the caller's `mma_wait<0>()`, and accumulator
constraints. The existing Q/K/V proxy fence and CTA barrier still publish input
tiles. The final CTA barrier still follows MMA completion before Q/K/V reuse.
Remove only the proxy fence and CTA barrier immediately after `store_prob`;
they publish P alone. Registers must not be overwritten before the PV wait.

No grid, block, host ABI, stream, or compiler-flag edits are needed.
Both host shared-memory settings already use `sizeof(Shared)` and shrink automatically.
Keep masked KV scores at negative infinity before exponentiation, zero-filled
input tails, uniform invalid-CTA returns, and guarded output stores. All math
threads, including rows outside a partial Q tile, still participate. Preserve
reverse KV traversal, online-softmax reduction/rescaling order, FP32 arithmetic,
FP16 probability rounding, and scalar FP16 output rounding/stores.

Validate the supplied problem with `klineage.harness.evaluate`; preserve its
workload and numerical policy. Inspect generated SASS: PV must use register-A
`HGMMA.64x128x16`, with the same eleven operations and unchanged QK operations.
Keep measurements outside this card.

## Example configuration

The supplied problem is noncausal packed-NHD FMHA without dropout:
Q/K/V/output are FP16 `[16384,64,128]`; offsets are int32 `[9]`.
The workload has eight sequences of 2048 tokens. Element row stride is
`kHeads*kDim = 8192`. Scale constants remain `kScale=0.0883883461f` and
`kLog2Scale=0.127517432f`.

Retain `kM=128`, `kN=176`, `kDim=128`, `kMmaK=16`,
`kInputPanelCols=kCoreRows=8`, `kMmaRows=64`, and 256 threads in two math groups.
The launch has `kMaxQTiles*kHeads=8640` CTAs and no fixed SM assignment.
Each CTA owns one Q tile/head/sequence and processes KV tiles synchronously.
QK uses eight `m64n176k16` instructions; PV uses eleven `m64n128k16` instructions
per group per KV tile. Retain 88 score, 64 output, and 44 packed probability
words per thread in the source. Keep shared Q/K/V tiles, unswizzled descriptors,
shared warp reductions, online softmax, and the scheduler.

Shared storage decreases from 168960 to 123904 bytes by deleting the 45056-byte
P tile; Q/K/V and reduction offsets remain unchanged. Preserve `alignas(128)`,
dynamic storage alignment 1024, `__launch_bounds__(kThreads,1)`, and the opt-in
shared-memory attribute. Retain native SM90a compilation and flags
`-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.

# Precondition

- Data types: a matching register-A instruction must accept the producer's
  representation and the current accumulation type. Unsupported combinations
  cannot replace the shared-A instruction. Forward the same rounded bits and
  preserve reduction order; a different conversion or arithmetic grouping could
  change numerical results.
- Layout: producer ownership and packed order must match the consuming
  instruction's per-lane A fragment. Direct forwarding performs no redistribution,
  so mismatched fragments select wrong entries. Global contiguity and alignment
  have no additional requirement: the transformation changes no global access.
- Storage: producer registers must contain P before its shared materialization,
  and the corresponding PV operation must be the shared copy's only consumer.
  Otherwise removing that copy discards a needed communication path. B must
  remain accessible through a valid shared descriptor because the selected
  register-A form still obtains B from shared memory.
- Pipeline: every warpgroup participant must issue the same collective MMA
  sequence. Producer writes must precede consumption, and fragments must stay
  live until completion, preventing stale or overwritten operands. The removed
  P publication barrier must order nothing else. Preserve other shared producers'
  visibility and wait for their readers before buffer reuse; deleting a barrier
  with another duty would introduce a race.
- Hardware: GPU and assembler support for the selected register-A WGMMA form
  and its 128-thread warpgroup is required to issue that operation. Its operand
  and accumulator registers must fit per-thread and per-CTA architectural limits.
  There is no additional shared-capacity requirement: the transformation deletes
  shared storage rather than adding it.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are replacements at three locations: the `pv` driver in `mma.cuh`,
`prob_value` in `attention.cuh`, and the shown sequence inside `consumer`.
Keep the surrounding code. Delete `store_prob` and the `Shared::p` member as
specified above. The register groups below retain all existing accumulator bindings.

## Before

```cuda
__device__ __forceinline__ void pv(float (&d)[kPvRegs], uint64_t a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(a), "l"(b) : "memory");
}

__device__ __forceinline__ void prob_value(float (&out)[kPvRegs], const Shared& s, int wg) {
    operands(out);
    mma_fence();
    uint64_t a = descriptor(s.p + wg * kMmaRows * kInputPanelCols, kM, kCoreRows);
    // MN-major panels advance by eight rows in K and one panel in N.
    uint64_t b = descriptor(s.v, kCoreRows, kN);
    #pragma unroll
    for (int i = 0; i < kN / kMmaK; ++i) {
        uint64_t ai = a + i * (kMmaK / kInputPanelCols) * kM;
        pv(out, ai, b + i * kMmaK);
    }
    mma_commit();
    operands(out);
}

// Inside consumer, after softmax and any output rescale.
convert(score, prob);
store_prob(prob, s);
proxy_fence();
__syncthreads();
prob_value(out, s, wg);
mma_wait<0>();
operands(out);
__syncthreads();
```

## After

```cuda
__device__ __forceinline__ void pv(float (&d)[kPvRegs], uint32_t const* a, uint64_t b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b) : "memory");
}

__device__ __forceinline__ void prob_value(float (&out)[kPvRegs], uint32_t (&prob)[kProbRegs], const Shared& s) {
    #pragma unroll
    for (int i = 0; i < kProbRegs; ++i) asm volatile("" : "+r"(prob[i]) :: "memory");
    operands(out);
    mma_fence();
    // MN-major panels advance by eight rows in K and one panel in N.
    uint64_t b = descriptor(s.v, kCoreRows, kN);
    #pragma unroll
    for (int i = 0; i < kN / kMmaK; ++i) pv(out, prob + i * 4, b + i * kMmaK);
    mma_commit();
    operands(out);
    #pragma unroll
    for (int i = 0; i < kProbRegs; ++i) asm volatile("" : "+r"(prob[i]) :: "memory");
}

// Inside consumer, after softmax and any output rescale.
convert(score, prob);
prob_value(out, prob, s);
mma_wait<0>();
operands(out);
__syncthreads();
```
