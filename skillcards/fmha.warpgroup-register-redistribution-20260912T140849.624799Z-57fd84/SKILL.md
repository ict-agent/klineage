---
skill_id: fmha.warpgroup-register-redistribution
intent: Redistribute registers from producer to consumer warpgroups.
preconditions:
- 'Data types: no additional dtype or arithmetic requirement; register redistribution
  changes capacity, not values or arithmetic order.'
- 'Layout: donor and recipient roles occupy complete, aligned warpgroups within one
  CTA, with uniform role selection; partial participation makes setmaxnreg invalid.
  Tensor strides and alignment impose no additional requirement because addresses
  stay unchanged.'
- 'Storage: donor live state can survive a smaller register budget through compiler
  allocation or spills, and recipient temporaries are defined before use; released
  registers cannot preserve live values and acquired registers have undefined contents.
  No additional tensor-memory placement is required.'
- 'Pipeline: donors can release registers without waiting for recipient progress,
  preventing blocking acquisition from deadlocking. Existing producer completion,
  reader visibility, and completion before buffer reuse remain enforced; register
  adjustment supplies no tensor-memory ordering.'
- 'Hardware: target and compiler support setmaxnreg and a valid initial CTA register
  allocation. Counts are multiples of 8 in [24,256], with donor_count <= initial_count
  <= recipient_count; the total requested registers must fit the CTA pool, including
  allocation granularity, or acquisition can block indefinitely.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Redistribute a CTA's register allocation from its producer warpgroup to its
consumer warpgroups. The producer has little live state; consumers hold attention
fragments. Larger consumer budgets reduce compiler spills without changing the
operator or its arithmetic.

In `solution/attention.cuh`, replace only the final dispatch block of `attention`,
starting at its existing `__syncthreads()`, with the After snippet. All producer
threads release registers before the inner producer-warp test or return. Each
consumer warpgroup acquires registers before entering `consumer`. The immediate
operands are absolute per-thread budgets. Keep the existing barrier initialization
and the `__launch_bounds__(kThreads, 1)` declaration.

No launch, tensor-address, fragment-ownership, or shared-layout change is needed.
No other register adjustments occur. Preserve all existing synchronization:
initialization becomes visible at the CTA barrier; full barriers establish input
copy completion before reads; empty barriers follow completed consumers before
buffer reuse. Preserve work and output handshakes. Register allocation does not
replace these memory dependencies.

Keep the sequence-length masks, guarded output stores, descending KV traversal,
FP32 accumulation and softmax order, FP16 probability conversion, and final
round-to-nearest FP16 conversion. The complete ProblemSpec, oracle, and numerical
requirements remain unchanged.

## Example configuration

The bundle specializes packed contiguous NHD FP16 attention to 16384 tokens,
64 heads, dimension 128, and eight sequences described by nine int32 offsets.
NHD element strides are `(8192,128,1)`; the token byte stride is 16384. Preserve
`kM=128`, `kN=176`, `kStages=2`, and the two 64-column input panels.

Retain the one-dimensional 384-thread CTA: threads 0–127 form the producer
warpgroup, with threads 0–31 running `producer`; threads 128–255 and 256–383
form the two consumer warpgroups. All 128 producer-group threads must execute the
release, including those that immediately return. Every consumer keeps its
existing fragment and output ownership.

Use producer budget 24 and consumer budget 240. The existing launch bound and
SM90a build reserve an initial 168 registers per thread:
`128*24 + 256*240 = 384*168 = 64512` registers per CTA.
Keep this launch bound and compilation configuration for replay; verify that the
compiler preserves a sufficient initial allocation. These budgets are this
configuration's settings, not universal prerequisites.

Keep `grid=(kMaxQTiles*kHeads)`, or 8640 CTAs, and one query tile per valid CTA.
Shared memory remains `sizeof(Shared)`: one query tile, two K buffers, two V
buffers, barriers, and work metadata. Preserve TMA with SW128 inputs, L2 hints,
WGMMA QK/PV, register probability fragments, online softmax, overlapping producer
and consumer work, scalar output stores, and programmatic launch dependencies.
Keep the destination-passing pybind11 ABI, caller device/stream, config.toml,
SM90a target, and compile flags `-O3 -std=c++17 --use_fast_math --resource-usage
-lineinfo -DNDEBUG`.

Rebuild from the complete final source bundle. Check generated code for both
register release and acquire operations and their budgets. Use
`klineage.harness.evaluate` with the unchanged problem to check correctness and
latency; keep measurements outside this card.

# Precondition

- Data types: no additional requirement. Redistribution only changes register
  capacity; it introduces no conversion, reduction, or arithmetic instruction.
- Layout: roles must align to complete hardware warpgroups in one CTA and select
  the adjustment uniformly. Otherwise some required participants miss the
  instruction. No extra tensor contiguity, stride, or alignment requirement
  arises because data addresses and ownership do not change.
- Storage: the compiler must preserve donor live state within the reduced budget
  or spill it, and recipient values must have definitions before reads. Returning
  registers cannot retain live data; acquiring registers does not initialize it.
  No additional global/shared tensor placement or visibility requirement arises.
- Pipeline: donors must reach their release without depending on recipient
  progress, since acquisition may wait for that release. Existing dependencies
  must still establish producer completion, reader visibility, and finished reads
  before storage reuse. Register adjustments supply none of that memory ordering.
- Hardware: the target/compiler must support `setmaxnreg` with a valid initial
  CTA allocation. Legal immediate budgets are multiples of 8 in [24,256], and
  release/acquire directions require `Rdonor <= Rinitial <= Rrecipient`.
  For donor and recipient thread counts `Tdonor` and `Trecipient`, require
  `Tdonor*Rdonor + Trecipient*Rrecipient <= Rcta`, accounting for register
  allocation granularity. `Rcta` is the register pool reserved for this CTA;
  spare registers belonging to another CTA cannot satisfy the request.
  Insufficient capacity can leave acquisition waiting forever.

Instruction rules: [NVIDIA PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg).

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both snippets replace the dispatch tail inside `native::attention` in
`solution/attention.cuh`. Existing `p`, `s`, `kGroupSize`, `kWarpSize`,
`producer`, and `consumer` remain in scope. The named constants encode the two
literal instruction budgets; no helper or other source edit is required.

## Before

```cuda
__syncthreads();
if (threadIdx.x < kGroupSize) {
    if (threadIdx.x < kWarpSize) producer(p, s);
    return;
}
consumer(p, s);
```

## After

```cuda
constexpr int kProducerRegs = 24;
constexpr int kConsumerRegs = 240;

__syncthreads();
if (threadIdx.x < kGroupSize) {
    // Release across the whole warpgroup before narrowing producer participation.
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"
                 :: "n"(kProducerRegs) : "memory");
    if (threadIdx.x < kWarpSize) producer(p, s);
    return;
}

// Acquire before creating the consumer's attention fragments.
asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"
             :: "n"(kConsumerRegs) : "memory");
consumer(p, s);
```
