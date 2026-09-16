---
skill_id: cuda.warpgroup-register-redistribution
intent: Redistribute CTA registers from producer to consumer warpgroups.
preconditions:
- 'Data types: no additional dtype or arithmetic requirement; register allocation
  changes neither values nor operations.'
- 'Layout: donor and receiver roles occupy complete, aligned 128-thread warpgroups
  with uniform control flow; setmaxnreg requires whole-warpgroup participation. Tensor
  contiguity and alignment impose no additional requirement.'
- 'Storage: register capacity is transferable only within one CTA; donor live values
  must remain valid under the reduced allocation and acquired registers must be initialized
  before use. Tensor placement and visibility need no change.'
- 'Pipeline: donors must reach their release without waiting on blocked receivers.
  Existing producer completion, reader visibility, and completion-before-buffer-reuse
  ordering must remain; register adjustment supplies no memory fence.'
- 'Hardware: compiler and target support setmaxnreg (sm_90a here). Budgets are multiples
  of 8 in [24, 256], with donor <= initial <= receiver. The launch must establish
  a valid initial allocation, and sum(threads_in_role * role_budget) must fit the
  CTA register pool or acquisition can block indefinitely.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Redistribute a CTA's register allocation from its lighter producer warpgroup to
its heavier consumer warpgroups. This reduces consumer spilling while retaining
warp specialization. Restore only the register helpers, their constants, and
the two call sites below.

In `solution/hopper.cuh`, add `kConsumerRegs` and `kProducerRegs` beside
`kWarpgroup`, then add `compute_registers()` and `load_registers()` before
`shared_addr()`. In `solution/attention.cuh`, insert `compute_registers()` as the
first statement of `consume<Group>()`, and `load_registers()` as the first
statement of `produce()`. The existing `attention()` dispatch gives each role
whole warpgroups. Do not put either call inside an elected-lane branch.

Keep the launch, compiler flags, ownership, bounds handling, operand layouts,
and arithmetic unchanged. The compiler handles any resulting spill changes.
No tensor moves or new shared buffers are required. Preserve query and KV
ready waits, copy-completion arrivals, WGMMA waits before releasing KV buffers,
and probability proxy fences and barriers. Register acquisition may wait for
the producer's release, which occurs before the producer waits on free buffers.
Each warpgroup executes one adjustment here, so no inter-adjustment barrier
is added.

Build using the kernel's saved flags and `TORCH_CUDA_ARCH_LIST=9.0a`. Inspect
`attention` in the compiled binary: consumers must contain `USETMAXREG.TRY_ALLOC.CTAPOOL`
and the producer `USETMAXREG.DEALLOC.CTAPOOL`. Confirm the initial register allocation
supports the budget equation below. Use the existing evaluator with the
unchanged problem for correctness and timing; keep evidence outside this card.

## Example configuration

Preserve `kConsumerRegs=216`, `kProducerRegs=72`, three 128-thread warpgroups,
and `__launch_bounds__(kThreads, 1, 1)`. Warpgroups 0 and 1 consume; warpgroup 2
produces. The initial 168-register allocation provides
`384*168 = 128*(216+216+72) = 64512` registers per CTA. Keep the existing
`--register-usage-level=10` ptxas option and all other saved compile flags.

The grid has 16384 CTAs, each owning one token and 64 heads, with a one-CTA
cluster and 230352 shared-memory bytes. Preserve 8192 tokens, 128 heads,
QK width 576, value width 512, and 2048 selected indices. Input tensors remain
contiguous; query rows have 576 BF16 elements, and output rows 512. Consumers
own opposite 256-column output halves; `row_index()` and fragment ownership
remain unchanged. Invalid indices retain zero-filled KV copies and masked
scores; no boundary participation changes are needed.

Retain resident Q, two KV buffers split into four transfer groups, 128-byte
shared swizzling, TMA query loads, 16-byte asynchronous KV copies, L2 policies,
WGMMA QK/PV, register P operands for local PV, shared P for peer PV, and online
softmax. Preserve FP32 accumulations, their reduction order, BF16 probability
and output round-to-nearest conversions, scale `0.1352337788608801f`, and FP32
maximum/LSE outputs. Preserve the pybind11 ABI and caller CUDA stream.

# Precondition

- Data types: no additional requirement. Allocating registers changes resource
  ownership without changing representation or arithmetic.
- Layout: each role must cover complete aligned 128-thread warpgroups and take
  uniform branches at its adjustment. Partial participation violates the
  instruction contract. Tensor alignment and contiguity add no condition.
- Storage: donors and receivers share one CTA register pool. Donor live values
  must survive the reduced allocation, and newly acquired registers need
  initialization. No tensor placement or visibility changes are required.
- Pipeline: a donor must release registers independently of receivers waiting
  to acquire them, avoiding a circular wait. Preserve producer completion,
  reader visibility, and completion before buffer reuse: register adjustment
  cannot replace memory ordering.
- Hardware: compiler and target must support `setmaxnreg`, including `sm_90a`
  for this target. Each budget is an 8-register multiple in [24, 256], and
  donor <= initial <= receiver makes the adjustment directions legal. A valid
  launch allocation must cover `sum(threads_in_role * role_budget)` from the
  CTA pool; otherwise receivers may wait indefinitely.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The ellipses denote existing bodies, retained verbatim. Helpers go in namespace
`hopper`; call sites remain in namespace `spa`. There are no launch edits.

## Before

```cuda
// solution/hopper.cuh: shared_addr follows group_index; no budget helpers.

// solution/attention.cuh: function openings.
template<int Group>
__device__ __forceinline__ void consume(
    Shared& sm, const Params& params, const Maps& maps, int lane, int warp, int head, int token) {
    if constexpr (Group == 0) {
        // ...
    }
    // ...
}

__device__ __forceinline__ void produce(Shared& sm, const Params& params, int lane, int token) {
    const int group = lane / kCopyGroup;
    // ...
}
```

## After

```cuda
// solution/hopper.cuh: constants beside kWarpgroup; helpers before shared_addr.
constexpr int kConsumerRegs = 216;
constexpr int kProducerRegs = 72;

__device__ __forceinline__ void compute_registers() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(kConsumerRegs));
}

__device__ __forceinline__ void load_registers() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(kProducerRegs));
}

// solution/attention.cuh: retain everything after the inserted calls.
template<int Group>
__device__ __forceinline__ void consume(
    Shared& sm, const Params& params, const Maps& maps, int lane, int warp, int head, int token) {
    compute_registers();
    if constexpr (Group == 0) {
        // ...
    }
    // ...
}

__device__ __forceinline__ void produce(Shared& sm, const Params& params, int lane, int token) {
    load_registers();
    const int group = lane / kCopyGroup;
    // ...
}
```
