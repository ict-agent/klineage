---
skill_id: cuda.packed-bf16-fragment-add
intent: Combine independent BF16 fragment additions into packed two-element instructions.
preconditions:
- 'Data types: operands and results use BF16 with independent round-to-nearest-even
  additions; __hadd2 implements these componentwise semantics, not wider accumulation
  or saturation.'
- 'Layout: one thread owns two independent additions whose corresponding operands
  occupy matching halves of packed register words; mismatched pairing adds the wrong
  elements. No global/shared contiguity or alignment requirement is added.'
- 'Storage: both operand pairs are already available to their owning thread in registers;
  this replacement introduces no memory transfer or shared allocation.'
- 'Pipeline: no additional synchronization or collective participation is required.
  Operands must be ready before addition, and existing publication and buffer-reuse
  ordering must remain because downstream readers still consume the results.'
- 'Hardware: the CUDA target and toolchain must support packed BF16 arithmetic for
  __hadd2 to combine the operations; no additional shared-memory capacity is required
  because packing uses registers.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace each scalar BF16 pair-add loop with one `__hadd2` call. Each
thread adds corresponding halves of two register words independently. This
reduces addition instructions and avoids scalar extraction/repacking around
the additions while retaining BF16 round-to-nearest-even results.

Apply the two replacements shown below:

1. In `solution/prepare.cuh`, function `invert`, replace the
   `add_pair` call in the final fragment-add loop, immediately before
   `o.x[j] = c.u`.
2. In `solution/recurrence.cuh`, function `recur_tile`, replace the
   `add_pair` call after `Reg prod = quantize(o[i])`, immediately before
   `out_bf[i].x[j] = c.u`.
3. Remove the now-unused `add_pair` helper from `solution/native.cuh`.

Keep the surrounding `Pair` declarations and result-word assignments.
`Pair`, already defined in `solution/native.cuh`, provides three views:

```cuda
union Pair { uint32_t u; __nv_bfloat162 b; BF16 h[2]; };
```

The low and high BF16 components of `a` correspond respectively to the
low and high components of `b`. Ownership stays with the same CUDA lane;
no inter-lane shuffle or matrix-fragment remapping is needed. The existing
`Pair::b` view supplies the packed operands without a memory load.

Keep all launches, tensor maps, shared layouts, bounds handling, fences,
barriers, and buffer rotations unchanged. Addition is synchronous register
arithmetic. In preparation, preserve the warp synchronization and later CTA
publication before TMA stores. In recurrence, preserve the input-ready wait,
output-empty acquisition, `compute_sync`, async-proxy fence, output-full
arrival, and store completion before output-buffer release. These already
make producers visible and prevent reuse while readers remain active.

Both complete register pairs are valid at these sites. There is no tail
case or new bounds check in this replay. An adapted fragment with an odd
valid element count must handle its unpaired element separately.

Preserve every operand conversion, BF16 rounding point, reduction grouping,
and MMA accumulation order. Do not move either addition across conversion
or fuse it with a product. Keep packed FP32-to-BF16 conversions and all
other arithmetic unchanged. Rebuild the complete source bundle with the
existing problem and compile flags; use the registered evaluator for
correctness and timing, with evidence outside this card.

## Example configuration

The scalar `add_pair` helper iterates over two 16-bit components with
`#pragma unroll 1`, issuing `add.rn.bf16` per component and repacking the
result. The rolled loop prevents compiler pairing; remove it with the
helper when restoring packed addition. This compiler control is part of
the scalar implementation, not a technique prerequisite.

The supplied case has batch 1, 4096 tokens, 96 heads, head dimension 128,
and 16-token chunks. Inputs/output use the existing BF16 and FP32 ABI;
initial/final state remain FP32 tensors with the existing internal BF16
state conversions. Normalization, bounded gates, scale, and beta processing
remain unchanged.

`Reg` holds four packed words; `Acc` holds eight FP32 values. The preparation
site adds four word pairs per lane in the warp running `invert`. The
recurrence site adds four word pairs for each of two output fragments per
compute lane. Each recurrence compute warp owns 32 value columns. Preserve
the existing eight-column shared-memory slabs and matrix load/store adapters.
These fragment shapes are replay settings, not packed-add prerequisites.

Preparation keeps grid `(256,96)`, 256 threads, launch bounds
`(kPrepareThreads,8)`, and 42368 dynamic shared-memory bytes. Recurrence
keeps grid `(1,96)`, 192 threads, four compute warps, one load warp, one
store warp, three input stages, two output stages, and 160768 dynamic
shared-memory bytes. Scratch buffers remain distinct.

Retain BF16 Tensor Core MMA with FP32 accumulators, TMA transfers, the
recurrence pipeline, register fragments and prefetch ordering, and the
scalar preparation transfers and shared-memory lane exchanges already in
the bundle. Preserve `config.toml`, the `kernel.cu::kernel` entry point,
destination passing, caller device/stream, and all compile flags, including
`--use_fast_math`. On this SM90 target, CUDA's `__hadd2` lowers to packed
BF16 addition. None of the retained MMA, TMA, launch, or stage settings is
an extra dependency of register-pair addition.

# Precondition

- Data types: both operands and each result are BF16, with separate
  round-to-nearest-even addition for each component. `__hadd2` preserves
  this arithmetic; replacing wider accumulation or saturating addition
  would change the contract.
- Layout: a single thread owns both independent additions. Corresponding
  elements must occupy matching halves of the packed operands, or the
  instruction combines unrelated values. No additional memory alignment,
  contiguity, or particular MMA-fragment arrangement is required: this
  operation addresses registers.
- Storage: both pairs are already present in the owning thread's registers.
  The replacement therefore needs no new transfer, shared buffer, or
  cross-thread visibility mechanism.
- Pipeline: there is no additional synchronization or collective requirement.
  Operand production must precede the addition in thread order. Preserve
  surrounding publication and reuse ordering so downstream shared-memory
  readers see completed results and finish before buffers are overwritten.
  Register packing changes none of these dependencies.
- Hardware: the CUDA compiler and device must implement packed BF16
  arithmetic through `__hadd2`; otherwise the two component operations
  cannot be combined by this recipe. There is no extra shared-memory
  capacity requirement because the operands and result use register words.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// solution/prepare.cuh: invert, final fragment-add loop.
Pair a{p.x[j]}, b{o.x[j]}, c;
c.u = add_pair(a.u,b.u);
o.x[j] = c.u;

// solution/recurrence.cuh: recur_tile, after quantize(o[i]).
Pair a{out_bf[i].x[j]}, b{prod.x[j]}, c;
c.u = add_pair(a.u,b.u);
out_bf[i].x[j] = c.u;
```

## After

```cuda
// solution/prepare.cuh: invert, same loop and ownership.
Pair a{p.x[j]}, b{o.x[j]}, c;
c.b = __hadd2(a.b,b.b);
o.x[j] = c.u;

// solution/recurrence.cuh: recur_tile, same loop and ownership.
Pair a{out_bf[i].x[j]}, b{prod.x[j]}, c;
c.b = __hadd2(a.b,b.b);
out_bf[i].x[j] = c.u;
```
