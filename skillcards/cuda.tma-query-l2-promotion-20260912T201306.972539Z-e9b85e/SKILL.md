---
skill_id: cuda.tma-query-l2-promotion
intent: Promote query TMA loads to 128-byte L2 fetches.
preconditions:
- 'Data types: no additional representation or arithmetic requirement; L2 promotion
  changes fetch granularity without converting tensor elements.'
- 'Layout: no additional contiguity, alignment, or thread-ownership requirement beyond
  the existing valid TMA mapping; promotion changes neither tensor coordinates nor
  shared-memory addressing.'
- 'Storage: global-memory inputs are already loaded through a tensor map; the policy
  applies to those loads'' DRAM-to-L2 fills.'
- 'Pipeline: the descriptor must be encoded before launch and remain valid during
  TMA use; existing producer completion, consumer visibility, and completion before
  buffer reuse must hold because a cache policy supplies no synchronization.'
- 'Hardware: CUDA TMA loads and a tensor-map encoder supporting CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  otherwise this fetch policy cannot be requested. No additional shared-memory capacity
  or whole-input L2-residency requirement applies.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Request 128-byte L2 fills for query TMA loads. In `solution/kernel.cu`,
inside `kernel`, change only the promotion argument used to construct
`Maps::query` from `CU_TENSOR_MAP_L2_PROMOTION_NONE` to
`CU_TENSOR_MAP_L2_PROMOTION_L2_128B`. The existing `tensor_map` helper forwards
this enum to `cuTensorMapEncodeTiled`; `load_q` consumes the encoded map.

The policy controls DRAM-to-L2 fetch granularity. Adjacent useful query bytes
can benefit from a larger fill; a speedup is workload-dependent. It adds no
explicit prefetch instruction or cache-residency guarantee.
[CUDA tensor-map API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
documents the promotion field.

Keep the helper, tensor coordinates, 128-byte shared swizzle, launch, ownership,
bounds handling, and all device instructions unchanged. This is a host descriptor
change; no compiler-control change is needed. Preserve the complete problem,
compiler flags, BF16 conversions, FP32 accumulation/reduction order, softmax
scale, and output semantics. Changing cache fills introduces no arithmetic.

## Example configuration

Retain these replay settings; they are not prerequisites of L2 promotion:

- `q[8192,128,576]` is contiguous BF16. The descriptor uses dimensions
  `{kWidth,kHeads,kTokens}`, byte strides
  `{kWidth*sizeof(Bf16),kWidth*kHeads*sizeof(Bf16)}` = `{1152,147456}`,
  box `{64,64,1}`, unit element strides, no interleave, and
  `CU_TENSOR_MAP_SWIZZLE_128B`. Keep `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.
- One CTA owns one token and 64 heads. Launch 16384 CTAs, 384 threads per CTA,
  a one-CTA cluster, and 231376 dynamic shared-memory bytes on the caller stream.
  Two 128-thread consumer groups share Q; a third group gathers KV.
- The elected lane of consumer group 0's first warp issues nine Q TMA loads
  into `sm.q_o + tile*kTileElems` at coordinates
  `(tile*kTile,head,token)`. All Q boxes are in bounds. Query storage remains
  resident throughout the CTA; it is not rotated or overwritten.
- Keep `sm.query.init(1)`, the initialization fence and CTA barrier, transaction
  expectation of `kTile*kWidth*sizeof(Bf16)` = 73728 bytes, and both consumer
  groups' `sm.query.wait(0)` before QK. The launch carries the encoded map by
  grid-constant value. The caller stream orders input producers before reads.
- Retain two KV buffers, four asynchronous KV transfer groups with their
  ready/free barriers, 16-byte `cp.async.cg` copies and invalid-index zero fill,
  Q/KV swizzles, WGMMA QK/PV, register accumulators, online softmax, shared
  probability exchange, and direct output stores. KV has one head; TOPK is
  2048, value width is 512, and selected indices are checked before masking.
  Keep WGMMA waits before operand-buffer release and probability publication
  fences/barriers before asynchronous readers.
- Preserve BF16 Q/KV/probabilities/output, FP32 accumulators/maxima/LSE,
  `kScale=0.1352337788608801f`, the existing reduction grouping and rounding,
  and SM90a compilation with the serialized build flags. TMA and WGMMA remain
  native CUDA operations behind the existing destination-passing pybind11 ABI.

Inspect the compiled host call to the tensor-map encoder to verify its promotion
argument. Device load instructions alone do not expose this descriptor field.
Use the existing Kernel evaluator for correctness and latency with the unchanged
problem and policy; store observations outside this card.

# Precondition

- Data types: no additional representation or arithmetic requirement. The enum
  controls fetching bytes into L2; it neither interprets nor converts elements.
- Layout: no additional contiguity, alignment, or thread-ownership requirement
  beyond the existing valid TMA mapping. Its coordinates and shared destination
  remain identical, so no new addressing or fragment rule is introduced.
- Storage: global-memory inputs already use tensor-map loads. The changed
  field controls their DRAM-to-L2 fills; it cannot
  affect a register-only or shared-only read.
- Pipeline: encode the descriptor before launch and keep it valid during TMA
  use. Preserve input-producer completion, TMA completion visible to consumers,
  and completion of all readers before destination-buffer reuse. The cache
  policy creates no ordering or visibility guarantee, so it cannot replace
  those existing dependencies. No additional synchronization is required.
- Hardware: the device and CUDA encoder must support TMA loads and
  `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`; an unsupported encoder/device cannot
  request this policy. No extra shared-memory allocation or whole-input L2
  capacity is required: promotion changes fill granularity, not storage size
  or guaranteed residency.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace this initializer in `solution/kernel.cu::kernel`. The existing
`tensor_map(address, width, promotion, swizzle)` helper and `Maps` definition
are shared context and need no edits.

## Before

```cuda
const Maps maps{tensor_map(q.data_ptr<at::BFloat16>(), kWidth, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B)};
```

## After

```cuda
const Maps maps{tensor_map(q.data_ptr<at::BFloat16>(), kWidth, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                           CU_TENSOR_MAP_SWIZZLE_128B)};
```
