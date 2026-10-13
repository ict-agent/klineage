---
skill_id: fmha.shared-qkv-staging
intent: Reuse Q, K, and V operands across CTA threads through shared-memory staging.
preconditions:
- 'Data types: shared loads and stores must preserve operand bits, with unchanged
  arithmetic order and rounding; no particular floating-point format is intrinsic
  to staging.'
- 'Layout: known global strides and tile ownership must map each reused operand to
  one shared location read by its CTA consumers; accesses need the element type''s
  alignment.'
- 'Storage: reused Q/K/V operands are read from global memory, remain immutable while
  consumed, and have consumers within one CTA; shared storage cannot supply another
  CTA.'
- 'Pipeline: input producers must finish before operand reads; all participating CTA
  threads, including boundary threads, must be able to rendezvous before tile consumption
  and reuse, so copies become visible and reads finish before overwrite.'
- 'Hardware: CTA-shared memory and CTA barriers are required; aligned storage for
  all operand tiles, retained scratch, and compiler-reserved shared bytes must fit
  the configurable per-CTA shared-memory limit.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore cooperative Q/K/V staging in `solution/ops.cuh` and
`solution/attention.cuh`. Load each operand tile once per CTA, then reuse it
across the existing scalar QK and PV accumulators.

Add `v`, `q`, and `k`, in that order, before `Shared::reduction`. Add the
synchronous `load<Rows>` helper below. It assigns physical shared indices
`i = threadIdx.x + j*kThreads` and writes the panel mapping
`panel<Rows>(row,col) = (col/C)*Rows*C + row*C + col%C`, where
`C = kInputPanelCols`. Every tile element has one writer. Q uses `Rows=kM`;
K and V use `Rows=kN`. Readers use this same mapping.

In `consumer`, replace the direct `query_key` call with the three loads,
publication barrier, and staged call shown below. Load Q again for every KV
tile. Replace `prob_value`'s call and both device signatures as shown. Within
their existing innermost loops, replace only the operand reads and associated
FMA expressions; preserve the row/column calculations and loop order.

Global tile bases retain packed NHD addressing: Q starts at
`p.q + qstart*kRow + work.z*kDim`; K/V start at their pointer plus `base`.
The copy substitutes positive zero when its row is outside `valid`.
Use `qvalid` for Q and `valid` for K/V. Retain the last-tile score mask,
invalid-query output guards, and uniform empty-CTA return.

The added `__syncthreads()` publishes Q/K/V before QK reads. Keep the barrier
after `store_prob` to publish P before PV reads. Keep the end-of-tile barrier:
it completes Q/K/V/P reads before the next iteration overwrites storage.
Keep `exchange`'s warp barriers and dedicated reduction slots unchanged.
Inputs remain ordered on the caller stream.

`kernel.cu` needs no edit: its shared-memory attribute and launch both use
`sizeof(Shared)`, so they automatically allocate the expanded storage.
Retain the grid, block, ABI, stream, configuration, and compiler flags.

## Example configuration

This replay uses FP16 Q/K/V/P/output and FP32 scalar accumulators. Retain
increasing-D QK FMAs, increasing-key PV FMAs within each tile, descending KV
tile traversal, online maximum/sum rescaling, and scalar round-to-nearest
FP16 conversion. Preserve probability rounding before PV, grouped softmax
reductions, scale constants, and output conversion/store helpers.

The workload is packed `[16384,64,128]`, with eight sequences described by
the unchanged offset inputs. Tiles use `kM=128`, `kN=176`, `kDim=128` and
`kInputPanelCols=8`. Global row stride is `kRow=kHeads*kDim=8192` half
elements, or 16384 bytes. Q/K/V remain contiguous NHD tensors.

Retain 256 threads, two 128-thread groups, eight 32-thread warps, and the
existing lane ownership. With lane `l`, warp within group `w`, group `g`,
and accumulator `i`, the local output coordinates are
`row=g*64+w*16+l/4+((i%4)/2)*8` and
`col=(i/4)*8+(l%4)*2+i%2`. Each thread retains 88 QK accumulators,
64 PV accumulators, 44 packed probability values, and two softmax rows.

The launch stays `grid=(kMaxQTiles*kHeads)` and `block=(kThreads)`, with
`__launch_bounds__(kThreads,1)` and the existing offset-based tile scheduler.
Use one synchronous Q/K/V tile set per iteration. Q takes 32768 bytes;
K and V each take 45056 bytes. P takes 45056 bytes and reduction scratch
1024 bytes. `sizeof(Shared)` grows from 46080 to 168960 bytes. Retain
`alignas(128)` and the dynamic allocation's 1024-byte alignment.

Keep CUDA/SM90 targeting and the existing `-O3`, `-std=c++17`,
`--use_fast_math`, `--resource-usage`, `-lineinfo`, and `-DNDEBUG` flags.
No compiler-control change is needed for this storage transformation.
Validate with the unchanged problem through `klineage.harness.evaluate`;
inspect compiled operand loads to confirm shared reads replace direct reads.
Keep validation evidence outside this card.

# Precondition

- Data types: the shared copy must preserve operand bits. Keep arithmetic
  order and rounding unchanged, since staging does not justify numerical
  changes. No particular floating-point format is required by the copy;
  another representation needs matching load/store types and storage sizes.
- Layout: global strides and tile ownership must be known, so cooperative
  writers and consumers agree on each operand's shared address. Consumers
  within a CTA must reuse these values for staging to provide reuse.
  Element-aligned accesses are required for typed loads/stores; this scalar
  copy imposes no vector alignment or particular panel width.
- Storage: Q/K/V currently come from global memory and must remain immutable
  throughout consumption, otherwise staging changes which values are seen.
  Each staged tile serves consumers within its own CTA because another CTA
  cannot address this shared allocation.
- Pipeline: upstream input writes must finish before operand reads. Existing
  control flow must allow all participating CTA threads, including boundary
  threads, to rendezvous before tile consumption and reuse. The first barrier
  publishes completed copies; the second completes reads before overwrite.
  Missing either ordering allows uninitialized reads or overwrites. This
  dependency imposes no fixed stage count.
- Hardware: the device must provide CTA-shared memory and CTA barriers.
  The aligned allocation containing retained scratch and all simultaneous
  operand tiles must fit the configurable per-CTA shared-memory limit;
  otherwise the launch cannot reserve the required storage. Operand bytes are
  `(Q_rows*D + 2*KV_rows*D)*sizeof(element)`; include retained scratch and
  any alignment padding and compiler-reserved shared bytes when checking capacity.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are replacement excerpts, not complete functions. Signature lines name
the existing definitions to update. Inside the arithmetic loops, `row`, `col`,
`d`, `key`, and `i` retain their existing definitions and traversal order.
Keep all code between the two consumer call sites, including P publication.

## Before

```cuda
// solution/ops.cuh
struct alignas(128) Shared {
    float reduction[kMathThreads];
    __half p[kM * kN];
};

// solution/attention.cuh: query_key signature and inner-loop expressions.
__device__ __forceinline__ void query_key(float (&score)[kQkRegs], const __half* q,
                                        const __half* k, int qvalid, int kvalid, int wg);
const float qvalue = row < qvalid ? __half2float(q[row * kRow + d]) : 0.f;
const float kvalue = col < kvalid ? __half2float(k[col * kRow + d]) : 0.f;
score[i] = fmaf(qvalue, kvalue, score[i]);

// prob_value signature and inner-loop expressions.
__device__ __forceinline__ void prob_value(float (&out)[kPvRegs], const Shared& s,
                                         const __half* v, int valid, int wg);
const float value = key < valid ? __half2float(v[key * kRow + col]) : 0.f;
out[i] = fmaf(__half2float(s.p[panel<kM>(row, key)]), value, out[i]);

// consumer: after base/valid declarations, before softmax.
query_key(score, p.q + qstart * kRow + work.z * kDim, p.k + base,
          qvalid, valid, wg);

// consumer: after the unchanged P publication barrier.
prob_value(out, s, p.v + base, valid, wg);

// Both math groups finish reading before P storage is reused.
__syncthreads();
```

## After

```cuda
// solution/ops.cuh
struct alignas(128) Shared {
    __half v[kN * kDim];
    __half q[kM * kDim];
    __half k[kN * kDim];
    float reduction[kMathThreads];
    __half p[kM * kN];
};

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

// solution/attention.cuh: query_key signature and inner-loop expressions.
__device__ __forceinline__ void query_key(float (&score)[kQkRegs], const Shared& s, int wg);
score[i] = fmaf(__half2float(s.q[panel<kM>(row, d)]),
                __half2float(s.k[panel<kN>(col, d)]), score[i]);

// prob_value signature and inner-loop expression.
__device__ __forceinline__ void prob_value(float (&out)[kPvRegs], const Shared& s, int wg);
out[i] = fmaf(__half2float(s.p[panel<kM>(row, key)]),
              __half2float(s.v[panel<kN>(key, col)]), out[i]);

// consumer: after base/valid declarations, before softmax.
load<kM>(p.q + qstart * kRow + work.z * kDim, s.q, qvalid);
load<kN>(p.k + base, s.k, valid);
load<kN>(p.v + base, s.v, valid);
__syncthreads();
query_key(score, s, wg);

// consumer: after the unchanged P publication barrier.
prob_value(out, s, wg);

// Both math groups finish reading before K/V/P storage is reused.
__syncthreads();
```
