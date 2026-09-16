---
skill_id: cuda.shared-store-row-rotation
intent: Rotate shared-memory store row ownership across lane groups to avoid bank
  conflicts.
preconditions:
- 'Data types: no additional numerical dtype requirement; reassigning independent
  elements must preserve their representation and per-element arithmetic order, so
  ownership changes cannot alter rounding.'
- 'Layout: the gather/store region permits a bijection of row ownership, with matching
  gather and store coordinates and known strides and packed-access alignment; destination
  conflicts must depend on a row-address contribution that can select distinct banks,
  or row rotation cannot remove them. Consumers address coordinates independently
  of the writer.'
- 'Storage: all reassigned source elements already reside in shared memory visible
  across their CTA, and destination storage is writable by that CTA; a thread cannot
  gather another row from another thread''s private registers.'
- 'Pipeline: source producers complete and publish before reassigned reads; destination
  writes become visible before consumers, and all readers finish before storage reuse
  or lifetime end. Participating threads can reach the retained barriers, preventing
  cross-warp races.'
- 'Hardware: CUDA warp execution and banked CTA shared memory with a known address-to-bank
  mapping; the row permutation must distribute simultaneously issued stores across
  distinct banks. No extra storage capacity is required because the same buffers are
  retained.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Rotate row ownership by lane group while converting row-major shared inputs into
slab-layout shared outputs. This removes bank conflicts from the packed stores
without changing physical layouts or arithmetic.

In `solution/prepare.cuh`, inside `prepare`, change **both** row expressions in
the two nested `m`/`n` loop nests between normalization/gate preparation and the
lower-triangular products. The first gathers `s.g`, `s.gt`, `s.q`, and `s.k` into
`rg`, `rgt`, `rq`, and `rk`; the second computes and stores `s.qd`, `s.kd`, `s.ki`,
and `s.kr`. Replace `m*8 + warp` and `m*8+warp` with
`m*8 + (warp+group)%8` and `m*8+(warp+group)%8`, respectively. Update the preceding
comment to describe the rotated ownership. Change neither loop's column expression.
Changing only one row expression associates gathered values with the wrong output.

For each fixed lane group, rotation permutes the participating warps' rows.
Each `(row, column pair)` therefore still has exactly one writer. All elementwise
expressions, intermediate BF16 rounding, and later reduction order stay unchanged.
`offset<kChunk>` in `solution/native.cuh` remains the destination layout.
The `load_a`/`load_b` consumers and `store_slabs` export path continue reading the
same coordinates; no MMA fragment adapter or tensor-map edit is needed.

Retain the CTA barrier after `s.gt` exponentiation: it publishes normalized q/k
and completed gate data before threads gather reassigned rows. Keep the barrier
between the gather and store loops and the barrier after their stores. The latter
publishes all four matrices before the lower-triangular products. Preserve the
later barriers before inversion reuses `s.ki`, and the async fence, CTA barrier,
and store completion wait before shared storage ends. Input/output stage rotation
and recurrence synchronization are unchanged. No allocation or launch changes occur.

## Example configuration

Preserve `BATCH=1`, `kTokens=4096`, `kHeads=96`, `kDim=128`, and `kChunk=16`.
The host checks contiguous BTHD tensors and launches preparation with
`grid=(kTiles,kHeads)=(256,96)`, `block=kPrepareThreads=256`, and
`sizeof(PrepareShared)=40192` dynamic shared bytes. The eight warps each execute
`m,n` in `[0,2)`, with `lane=tid%kWarp`, `warp=tid/kWarp`, `group=lane/4`,
`pair=lane%4`, and `c=n*64+group*8+pair*2`.
Before rotation a warp owns rows `m*8+warp`. Afterwards each group owns
`m*8+(warp+group)%8`; both mappings cover every pair of the 16-by-128 tile once.
The rotated row remains in `[0,kChunk)` and `c+1<kDim`, so retain the full-tile
loops without new predicates. Other dimensions need a complete bijective mapping
and consistent bounds handling in both loops.

Inputs `s.q`, `s.k`, and `s.g` are row-major, indexed by `r*kDim+c`;
`s.gt` is indexed by `c`. These accesses retain aligned `uint32_t` BF16 pairs
and `float2` FP32 pairs. The four BF16 destinations use
`offset<R>(r,c)=r*8+(c&7)+(c/8)*R*8`, with `R=kChunk`.
In byte terms, a source row stride is `kDim*sizeof(element)` and a destination
slab stride is `R*8*sizeof(BF16)`. Keep the existing aligned shared allocation.

For this packed store on 32 banks of four bytes, the bank index, up to a common
base offset, is `(4*r+pair)%32`: each slab contributes a multiple of 32 words.
Without rotation, all eight lane groups in a warp target the same four banks at
different addresses. With rotation, `(4*((warp+group)%8)+pair)%32` visits all
32 banks. Source row strides contribute whole bank cycles, so reassigned rows do
not change the gather's bank pattern. This argument concerns these stores and
strides; different packing or layouts require deriving their bank mapping again.

Retain BF16 q/k/v/g and decayed matrices, FP32 gate working values and MMA
accumulators, BF16 conversion points, normalization epsilon, scales, activation
instructions, and accumulation order. Preserve the full ProblemSpec, including
its FP32 oracle and FP32 final-state output contract. Ownership rotation requires
no numerical relaxation.

Keep the BF16 `mma.sync`, `ldmatrix`, `stmatrix`, register prefetch, triangular
block inversion, TMA transfers, and shared buffers. Recurrence retains 192 threads,
four compute warps, three input stages, two output stages, and 160768 shared bytes.
Keep `solution/native.cu`, the pybind11 ABI, caller stream, tensor maps, compiler
flags, and launch bounds unchanged. These retained mechanisms are configuration,
not prerequisites of row rotation.

After replay, check that both coordinate expressions match, tile coverage remains
bijective, and all consumers retain their original layout. Use the existing
`klineage.harness.evaluate` with the unchanged problem for correctness and timing;
keep results outside this card.

# Precondition

- Data types: no additional numerical dtype requirement. Each reassigned element
  must retain its representation and arithmetic order, including rounding, because
  the transformation changes its executing thread only.
- Layout: the elementwise region must admit a bijection of row ownership. Gather
  and store coordinates must agree, with known physical strides and alignment for
  retained packed accesses; otherwise values are misplaced or accesses misaligned.
  Conflicting destination banks must have a row-address contribution that can
  select distinct banks, or rotation cannot resolve the conflict. Consumers must
  locate values by coordinates independently of the producing thread.
- Storage: reassigned sources already occupy CTA-visible shared memory, and the
  CTA can write their destinations. Values available only in another thread's
  private registers cannot be gathered by this ownership change.
- Pipeline: producers must finish and publish sources before reassigned reads;
  stores must become visible before consumers. All readers must complete before
  buffers are overwritten or their lifetime ends. Participating threads must reach
  the retained barriers; these conditions prevent cross-warp visibility and reuse
  races as ownership moves between warps.
- Hardware: CUDA warp execution and banked CTA shared memory with a known
  address-to-bank mapping are required to distribute simultaneous stores across
  distinct banks. No extra capacity requirement is introduced: existing buffers
  and their allocation are unchanged.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both snippets identify the two coordinate blocks inside the existing `m`/`n`
loops in `prepare`; the loop bodies and intervening barrier remain in place.
`tid`, `warp`, `lane`, `group`, and `pair` retain their existing definitions.

## Before

```cuda
// Gather loop: rg/rgt/rq/rk receive this coordinate's inputs.
int t = m*2+n;
int r = m*8 + warp;
int c = n*64+group*8+pair*2;

// Store loop: unchanged arithmetic produces qd/kd/ki/kr.
int t = m*2+n;
int r = m*8+warp;
int c = n*64+group*8+pair*2;
// Store the packed results at offset<kChunk>(r,c).
```

## After

```cuda
// Gather loop: rotate rows across lane groups to spread store banks.
int t = m*2+n;
int r = m*8 + (warp+group)%8;
int c = n*64+group*8+pair*2;

// Store loop: use the same rotation to preserve coordinate ownership.
int t = m*2+n;
int r = m*8+(warp+group)%8;
int c = n*64+group*8+pair*2;
// Store the packed results at offset<kChunk>(r,c).
```
