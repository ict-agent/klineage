---
skill_id: kda.cooperative-row-norm
intent: Share row-normalization partial sums across threads through shared memory.
preconditions:
- 'Data types: partial sums must retain their representation, leaf arithmetic, and
  ordered addition tree; changing these can change normalized outputs. Operand bit
  width itself is not a requirement of exchanging sums.'
- 'Layout: each row''s contributors occupy a power-of-two group G of consecutive,
  G-aligned lanes inside one warp, with disjoint known input segments and unique partial
  slots; XOR partners must stay within the same row.'
- 'Storage: each assigned contributor can read its row segment; otherwise distributed
  leaf evaluation cannot replace independent full-row reads. No additional operand
  placement requirement applies.'
- 'Pipeline: operands are complete before reduction and remain stable throughout it,
  and every lane named by the warp mask can execute matching barriers. Partial writes
  must be visible before partner reads, and reads must finish before slot reuse.'
- 'Hardware: CUDA warp barriers and CTA shared memory are required for the exchange.
  Free shared capacity must cover two partial slots per active contributor: 2*T*sizeof(partial),
  in addition to other live allocations.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Distribute each query/key row's norm across its contributing threads. Each
thread computes one leaf sum, then exchanges partials through shared memory.
This replaces every thread's independent evaluation of the complete reduction
tree while preserving its own ordered result.

In `solution/prepare.cuh`, remove `NormSum` and `norm_tree`. Keep namespace
constants `kNormElems` and `kNormLanes`. Replace the normalization block from
its ownership calculation through `qs`/`ks` initialization with After below.
Reuse the reserved `PrepareShared::norm_q` and `norm_k` arrays in
`solution/native.cuh`; change their comment to describe the active exchange.
Each thread owns slot `tid` in both arrays. At each decreasing XOR distance,
store both partials, synchronize the full warp, read `tid ^ delta`, add, then
synchronize before overwriting either array. No thread exits this phase early.

The leaf products accumulate in increasing element order. The XOR distances
reproduce the recursive tree: the largest distance joins leaves first, and
the smallest joins the final subtrees. Preserve operand conversions, compiler
flags, multiply/add contraction, and the subsequent `norm_each` calls and BF16
stores. Do not replace this tree with a differently associated row sum.

Launches, output ownership, layouts, and phase boundaries stay unchanged.
The caller stream makes inputs visible; the existing CTA barrier after
normalization publishes `s.q` and `s.k` to their later consumers. The final
warp barrier finishes scratch reads before later work or reuse. The fixed
workload has complete rows and tiles, so these snippets need no new bounds
guards. With partial rows, zero-fill invalid leaf elements and keep every
masked lane participating.

## Example configuration

Preserve batch 1, 4096 tokens, 96 heads, dimension 128, chunk 16, and 256
preparation threads. Each CTA owns one `(tile, head)` with 16 token rows;
each 16-lane row group owns eight contiguous elements per lane. Query/key
inputs are BF16 BTHD, with element strides `(TOKENS*HEADS*DIM, HEADS*DIM,
DIM, 1)`. Leaf sums and shared partials are FP32. Keep `kNormElems=8`,
`kNormLanes=kDim/kNormElems=16`, `kAllLanes=0xffffffffu`, and distances
8, 4, 2, 1. The two reserved arrays each contain 256 floats (2048 bytes
together); their byte offsets within `PrepareShared` are 39560 and 40584.

Keep `prepare<<<dim3(kTiles,kHeads),kPrepareThreads,sizeof(PrepareShared),stream>>>`,
its `__launch_bounds__(kPrepareThreads,8)`, and 41856 dynamic shared bytes.
Keep the existing 3072 static shared bytes for inversion and transposition.
The supplied build targets SM90a with the original flags, including
`--use_fast_math` and ptxas register-usage level 10. The norm epsilon remains
`1e-6f`; reciprocal norms remain per element, and normalized values round to
BF16 before later preparation.

Retain scalar volatile consumer loads, gate-prefix recomputation, two-block
triangular inversion, BF16 MMA with FP32 accumulators, shared matrix staging,
correction reuse, 256 separately launched recurrence chunks, and BF16 carry
handoffs. The recurrence uses 256 threads and 124672 dynamic shared bytes.
These retained mechanisms do not create prerequisites for the norm exchange.

# Precondition

- Data types: preserve the partial-sum representation, leaf arithmetic, and
  ordered addition tree. Shared transport must not round partials differently;
  reassociation or different leaf contraction can alter the normalized result.
  Exchanging partials does not inherently require a particular operand bit width.
- Layout: a row uses a consecutive, G-aligned power-of-two group of G lanes
  within one warp. Each contributor has a known disjoint input segment and
  distinct scratch slots. These properties keep every XOR partner in the same
  row and prevent missing, duplicate, or overwritten contributions.
- Storage: each contributor must be able to read its assigned row segment.
  Otherwise it cannot supply the leaf sum previously recomputed by every
  output thread. No additional operand placement is required; shared scratch
  is introduced by the optimization.
- Pipeline: the input producer finishes before reduction, and operands remain
  stable throughout it. All lanes named by the warp mask reach matching
  barriers. Publication before partner reads prevents stale partials; completion
  of reads before reuse prevents the next round overwriting live values.
- Hardware: the exchange requires CUDA warp barriers and CTA shared memory.
  With T active contributors, reserve at least `2*T*sizeof(partial)` bytes
  beyond other live allocations, or the two sets of partial slots cannot fit.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The Before helper is removed; replace its following `prepare` block with After.
Both snippets use the existing `PrepareArgs`, `PrepareShared`, `bf_float`,
`kNormElems`, `kNormLanes`, `tid`, `tile`, `head`, `args`, and `s` definitions.
Leave the immediately following per-element normalization loop unchanged.

## Before

```cuda
struct NormSum { float q = 0.0f, k = 0.0f; };

template<int Delta>
__device__ __forceinline__ NormSum norm_tree(
    const PrepareArgs& args, int base, int lane) {
    if constexpr (Delta == kNormLanes) {
        NormSum sum;
        #pragma unroll
        for (int j = 0; j < kNormElems; ++j) {
            const int src = base + lane * kNormElems + j;
            float q = bf_float(args.q[src]);
            float k = bf_float(args.k[src]);
            sum.q += q*q;
            sum.k += k*k;
        }
        return sum;
    } else {
        // Recompute both subtrees locally in the original XOR addition order.
        const NormSum a = norm_tree<Delta * 2>(args,base,lane);
        const NormSum b = norm_tree<Delta * 2>(args,base,lane ^ Delta);
        return {a.q + b.q, a.k + b.k};
    }
}

// Inside prepare:
    // Each thread recomputes its row norm without exchanging partial sums.
    int row = tid / (kDim/kNormElems);
    int col = tid % (kDim/kNormElems) * kNormElems;
    const int norm_base = ((tile*kChunk+row)*kHeads+head)*kDim;
    const NormSum sums = norm_tree<1>(args,norm_base,tid % kNormLanes);
    float qs = sums.q, ks = sums.k;
```

## After

```cuda
// Each lane computes one disjoint segment of its row.
int row = tid / kNormLanes;
int col = tid % kNormLanes * kNormElems;
float qs = 0.0f, ks = 0.0f;
#pragma unroll
for (int j = 0; j < kNormElems; ++j) {
    const int src = ((tile*kChunk+row)*kHeads+head)*kDim+col+j;
    float q = bf_float(args.q[src]);
    float k = bf_float(args.k[src]);
    qs += q*q;
    ks += k*k;
}

#pragma unroll
for (int delta = kNormLanes / 2; delta >= 1; delta >>= 1) {
    // Publish partials before reading the same row's XOR partners.
    s.norm_q[tid] = qs;
    s.norm_k[tid] = ks;
    __syncwarp(kAllLanes);
    qs += s.norm_q[tid ^ delta];
    ks += s.norm_k[tid ^ delta];

    // Finish all reads before the next round overwrites the slots.
    __syncwarp(kAllLanes);
}
```
