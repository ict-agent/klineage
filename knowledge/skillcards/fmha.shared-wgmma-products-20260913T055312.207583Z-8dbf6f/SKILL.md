---
skill_id: fmha.shared-wgmma-products
intent: Accelerate attention matrix products with shared-memory WGMMA.
preconditions:
- 'Data types: operand and accumulator representations must match a supported WGMMA
  opcode, and its reduction grouping must satisfy the numerical contract; required
  intermediate rounding remains explicit.'
- 'Layout: operands must admit the instruction major order and descriptor offsets
  with 16-byte-aligned bases and 16-byte offset units; accumulator ownership must
  match its lane/register mapping, and reductions must cover complete 16-element slices
  or use padding and masking.'
- 'Storage: operands are resident in CTA-shared memory for descriptor reads; simultaneously
  live tiles cannot alias storage overwritten before their readers finish.'
- 'Pipeline: control flow permits complete 128-thread warpgroups to participate uniformly
  in aligned MMA, fence, commit, and wait operations; shared producers publish before
  issue, register updates precede MMA, results are read after completion, and every
  reader completes before buffer reuse.'
- 'Hardware: the CUDA target and toolchain support the selected SM90a WGMMA shapes
  and async-proxy synchronization; live shared storage fits the configured per-CTA
  limit, MMA accumulator tuples can reside in registers during issue, and allocated
  register resources fit launch limits.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace scalar shared-memory FMAs in `solution/attention.cuh::query_key` and
`prob_value` with shared/shared WGMMA. Each warpgroup computes the same output
register tile. Restore only tensor-core computation and its descriptor, compiler
ordering, and async-proxy adapters.

## Example configuration

Preserve packed contiguous FP16 Q/K/V/output `[16384,64,128]`, eight sequences,
int32 sequence offsets, noncausal attention without dropout, and the full problem
oracle. A token row spans `kRow=kHeads*kDim=8192` half elements. Preserve the
scale constants `kScale=0.0883883461f` and `kLog2Scale=0.127517432f`.

Keep `kM=128`, `kN=176`, `kDim=128`, `kMmaK=16`, panel width 8, and core height 8.
Each CTA has 256 threads, split into two 128-thread warpgroups. The grid remains
`kMaxQTiles*kHeads=8640`; `tile` derives sequence ownership from Q offsets and
extra CTAs return uniformly. Launch bounds remain `(kThreads,1)`. Preserve the
SM90 device check, `sm_90a` compilation, current CUDA stream, pybind11 ABI,
`config.toml`, and all input compile flags.

Retain `Shared` unchanged: V, Q, K, reduction slots, then P; 168960 bytes total.
The allocation has 1024-byte alignment and the struct has 128-byte alignment.
There is one synchronous shared tile per operand. Q is reloaded on each descending
KV iteration. Keep online softmax, shared peer exchange, scalar FP16 conversions,
scalar output stores, register tiling, sequence guards, and LSE stores.

The unpermuted half-element address of `(r,c)` in an operand with `R` rows is
`(c/8)*R*8+r*8+c%8`. Q and P have `R=kM`; K and V have `R=kN`.
Q/K/P use K-major descriptors. The same physical V panels are MN-major for PV.
No storage transpose or swizzle is introduced.

For thread lane `l`, warp `w` within its warpgroup, group `g`, and register `i`,
retain row `g*64+w*16+l/4+((i%4)/2)*8` and column
`(i/4)*8+(l%4)*2+i%2`. QK has 88 FP32 accumulators per thread; PV has 64.
These are the register orders of `m64n176k16` and `m64n128k16` respectively.
`store_prob`, softmax reductions, and the epilogue already use this ownership.

Keep reverse KV-tile traversal, increasing reduction-tile traversal, FP32
accumulators, and explicit round-to-nearest FP16 P/output conversions. WGMMA
changes scalar reduction grouping; use the supplied numerical oracle unchanged.
Clear QK on its first reduction instruction, then accumulate. PV always adds to
the initialized or rescaled output. Do not replace the retained probability
rounding with FP32 multiplication.

## Restore device adapters

In `solution/ops.cuh`, replace `panel` with the following functions inside
`namespace native`. They encode unswizzled shared descriptors in 16-byte units.
For Q/K/P, leading is the operand's row count and stride is `kCoreRows`.
For MN-major V, leading is `kCoreRows` and stride is `kN`.

```cuda
__device__ __forceinline__ uint32_t shared_addr(const void* p) {
    return uint32_t(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ uint64_t descriptor(const __half* p, int leading, int stride) {
    constexpr int kAddressShift = 4;
    constexpr int kLeadingShift = 16;
    constexpr int kStrideShift = 32;

    // No swizzle; offsets are in 16-byte descriptor units.
    return (uint64_t(stride) << kStrideShift) | (uint64_t(leading) << kLeadingShift)
           | (shared_addr(p) >> kAddressShift);
}
```

In `solution/mma.cuh`, retain includes and existing constants through
`kPanelCols`. Remove the four scalar ownership constants introduced after it:
`kFragmentRows`, `kFragmentSize`, `kLanesPerRow`, and `kPairElems`.
Insert these helpers inside `namespace native`:

```cuda
enum class Accum { Clear, Add };

template<int N>
__device__ __forceinline__ void operands(float (&x)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(x[i]) :: "memory");
}

__device__ __forceinline__ void mma_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void mma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template<int Groups>
__device__ __forceinline__ void mma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(Groups) : "memory");
}
```

Add two inline PTX drivers after the helpers. This source generator specifies all
register constraints without an opaque external dependency. Run it in the bundle
root once, after adding the helpers; it inserts the generated drivers before the
namespace terminator in `solution/mma.cuh`.

```python
from pathlib import Path


def emit_mma(name, count, width):
    registers = ", ".join(f"%{i}" for i in range(count))
    outputs = ", ".join(f'"+f"(d[{i}])' for i in range(count))
    inputs = '"l"(a), "l"(b)'
    signature = ""
    lines = []

    if name == "qk":
        signature = ", Accum mode"
        inputs += ', "r"(int(mode == Accum::Add))'
        lines.append(f"{{ .reg .pred p; setp.ne.b32 p, %{count + 2}, 0;")
        controls = "p, 1, 1, 0, 0"
    else:
        controls = "1, 1, 1, 0, 1"

    lines.append(
        "wgmma.mma_async.sync.aligned."
        f"m64n{width}k16.f32.f16.f16 {{{registers}}}, "
        f"%{count}, %{count + 1}, {controls};"
    )
    if name == "qk":
        lines.append("}")

    assembly = "\n".join('        "' + line + r'\n"' for line in lines)
    return (
        f"__device__ __forceinline__ void {name}(float (&d)[{count}], "
        f"uint64_t a, uint64_t b{signature}) {{\n"
        f"    asm volatile(\n{assembly}\n"
        f'        : {outputs}\n        : {inputs} : "memory");\n'
        "}\n"
    )


path = Path("solution/mma.cuh")
source = path.read_text()
marker = "} // namespace native"
assert source.count(marker) == 1
wrappers = emit_mma("qk", 88, 176) + "\n" + emit_mma("pv", 64, 128)
path.write_text(source.replace(marker, wrappers + marker))
```

QK uses descriptor operands A/B, positive input scales, no transposition, and a
predicate for accumulator clear/add. PV uses positive scales, always accumulates,
and transposes B's interpretation to consume V's MN-major layout. Preserve all
explicit `+f` accumulator constraints and memory clobbers.

## Restore issue and completion ordering

Replace both scalar math functions with the After snippet. In
`solution/attention.cuh::consumer`, add `proxy_fence()` before the existing CTA
barrier after Q/K/V loads and before the barrier after `store_prob`. These
publish generic shared writes to WGMMA's async proxy before any group reads them.

Immediately after `query_key(score, s, wg)`, insert:

```cuda
mma_wait<0>();
operands(score);
```

Immediately after `prob_value(out, s, wg)`, insert:

```cuda
mma_wait<0>();
operands(out);
```

Every warpgroup thread must issue the same aligned instructions. The helper
fences order ordinary register updates before MMA; commits form one group per
product call. Waits complete score/output writes before scalar consumers.
Keep the final CTA barrier in each KV iteration: both warpgroups must finish
reading operands before any thread overwrites Q/K/V/P on the next iteration.
Shared reduction exchange keeps both warp barriers.

Keep zero filling in `load`, last-KV-tile score masking, and bounded epilogue
stores. Padded K/V entries cannot become valid softmax probabilities; invalid Q
rows still participate in every CTA barrier. No grid or block change is needed.

Check the complete problem using `klineage.harness.evaluate` without changing its
policy. Inspect generated device code for both WGMMA shapes and their waits;
scalar source replacement alone does not establish tensor-core use.

# Precondition

- Data types: the selected instruction must support the operand and accumulator
  representations, and the numerical contract must permit its reduction grouping.
  Unsupported representations cannot be consumed by that opcode. Scalar FMA
  grouping is not promised bitwise equivalent to WGMMA. Required intermediate
  rounding remains explicit, since changing it changes the numerical path.
- Layout: shared operands must be expressible by the instruction's major order
  and descriptor offsets, with 16-byte-aligned descriptor bases and offsets in
  16-byte units. Otherwise descriptors address the wrong elements or are invalid.
  Accumulator ownership must match the instruction's lane/register mapping, or
  adapters must restore that mapping for scalar consumers. Reductions must cover
  complete 16-element instruction slices, padding and masking tails when needed;
  WGMMA cannot predicate individual elements of an incomplete slice.
- Storage: operands are available in CTA-shared memory and remain resident until
  consumption completes. The descriptor form reads shared operands, so direct
  global addresses cannot replace them. Simultaneously live tiles must not alias
  storage that producers overwrite while readers still use it.
- Pipeline: control flow must permit each complete 128-thread warpgroup to reach
  the same aligned MMA, fence, commit, and wait operations; partial or divergent
  participation violates their collective contract. Producers must finish shared writes and publish
  them to all consumers before issue; register writes must precede MMA and result
  reads must follow completion. Every reader must finish before shared-buffer
  reuse. These ordering dependencies apply regardless of stage count.
- Hardware: an SM90a-capable CUDA target and toolchain must support the chosen
  WGMMA shapes and async-proxy synchronization. Live shared allocation must fit
  the device's configured per-CTA limit. Each MMA accumulator tuple must be
  register-resident during issue, and allocated register resources must fit the
  chosen launch's limits. Other live state may spill; exceeding launch resource
  limits or lacking register operands prevents legal execution.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The adapters and consumer edits above are shared context. Replace these two
functions in `solution/attention.cuh`; other device and host code stays intact.

## Before

```cuda
__device__ __forceinline__ void query_key(float (&score)[kQkRegs], const Shared& s, int wg) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = (threadIdx.x % kGroupSize) / kWarpSize;
    #pragma unroll
    for (int i = 0; i < kQkRegs; ++i) score[i] = 0.f;

    // Keep each lane's output tile; reduce shared operands with scalar FMAs.
    #pragma unroll 1
    for (int d = 0; d < kDim; ++d) {
        #pragma unroll
        for (int i = 0; i < kQkRegs; ++i) {
            const int row = wg * kMmaRows + warp * kFragmentRows
                            + lane / kLanesPerRow + (i % kFragmentSize) / kPairElems * kCoreRows;
            const int col = (i / kFragmentSize) * kInputPanelCols
                            + (lane % kLanesPerRow) * kPairElems + i % kPairElems;
            score[i] = fmaf(__half2float(s.q[panel<kM>(row, d)]),
                            __half2float(s.k[panel<kN>(col, d)]), score[i]);
        }
    }
}

__device__ __forceinline__ void prob_value(float (&out)[kPvRegs], const Shared& s, int wg) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = (threadIdx.x % kGroupSize) / kWarpSize;

    // Accumulate the rounded probability tile in increasing key order.
    #pragma unroll 1
    for (int key = 0; key < kN; ++key) {
        #pragma unroll
        for (int i = 0; i < kPvRegs; ++i) {
            const int row = wg * kMmaRows + warp * kFragmentRows
                            + lane / kLanesPerRow + (i % kFragmentSize) / kPairElems * kCoreRows;
            const int col = (i / kFragmentSize) * kInputPanelCols
                            + (lane % kLanesPerRow) * kPairElems + i % kPairElems;
            out[i] = fmaf(__half2float(s.p[panel<kM>(row, key)]),
                          __half2float(s.v[panel<kN>(key, col)]), out[i]);
        }
    }
}
```

## After

```cuda
__device__ __forceinline__ void query_key(float (&score)[kQkRegs], const Shared& s, int wg) {
    operands(score);
    mma_fence();
    uint64_t a = descriptor(s.q + wg * kMmaRows * kInputPanelCols, kM, kCoreRows);
    uint64_t b = descriptor(s.k, kN, kCoreRows);
    #pragma unroll
    for (int i = 0; i < kDim / kMmaK; ++i) {
        uint64_t ai = a + i * (kMmaK / kInputPanelCols) * kM;
        uint64_t bi = b + i * (kMmaK / kInputPanelCols) * kN;
        qk(score, ai, bi, i == 0 ? Accum::Clear : Accum::Add);
    }
    mma_commit();
    operands(score);
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
```
