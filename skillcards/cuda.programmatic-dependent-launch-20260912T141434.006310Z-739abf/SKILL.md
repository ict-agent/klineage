---
skill_id: cuda.programmatic-dependent-launch
intent: Overlap dependent kernel startup through programmatic dependent launch.
preconditions:
- 'Data types: no additional requirement; launch scheduling does not change stored
  representations, arithmetic, or rounding.'
- 'Layout: no additional contiguity, alignment, or thread-mapping requirement; the
  transformation changes when grids run, not their addresses or ownership.'
- 'Storage: producer results occupy memory accessible to both grids and remain live
  through consumer reads; private CTA storage cannot carry this cross-grid dependency.'
- 'Pipeline: the kernels launch in order on one stream, the consumer has independent
  startup work, every producer CTA can signal or finish, and every dependent reader
  can wait for producer completion and visibility. The producer must finish without
  consumer progress, and results must not be overwritten until readers finish; otherwise
  overlap can race or deadlock.'
- 'Hardware: SM90 or newer and a toolchain/runtime supporting griddepcontrol and programmatic
  launch attributes are required. Actual overlap needs available execution resources
  while the producer runs; correctness must also hold without overlap. No additional
  storage capacity is required.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore programmatic dependent launch (PDL) between `prepare` and `attention`.
Currently, ordinary caller-stream ordering completes `prepare` before starting
`attention`. PDL lets attention's independent startup overlap metadata production
or launch latency, then waits before consuming metadata. Concurrency is optional.

## Replay

Apply all three edits shown below together:

1. In `solution/scheduler.cuh::prepare`, insert
   `griddepcontrol.launch_dependents` immediately after `lane` is assigned, before
   input reads and early returns. Every thread of this example's producer CTA
   reaches it; repeated signals within a CTA are harmless.
2. In `solution/scheduler.cuh::producer`, insert `griddepcontrol.wait` immediately
   after `lane` is assigned and before `tile(blockIdx.x, p.metadata)`. All threads
   of the producer warp execute the wait. Keep the inline assembly `memory`
   clobbers to prevent compiler motion across these scheduling operations.
3. In `solution/kernel.cu::kernel`, replace the stream-order comment before
   `cudaLaunchKernelEx` with the attribute block below. The local attribute stays
   alive through that call. Retain the existing zero-initialized configuration,
   grid, block, shared-memory allocation, stream, and launch error checks.

The primary signal permits scheduling; it does not publish unfinished metadata.
The secondary wait supplies completion and visibility before dependent reads.
See NVIDIA's [PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
and [griddepcontrol specification](https://docs.nvidia.com/cuda/archive/13.0.0/pdf/ptx_isa_9.0.pdf).

## Ownership and ordering

`prepare` writes sequence metadata in global memory. The attention producer warp
reads it in `tile`, publishes `s.work`, and participates in `Named::WorkFull`.
Consumers pass that barrier in `next` before their metadata reads. Thus their
existing handoff remains downstream of the producer's dependency wait.
Attention's tensor-map prefetches, barrier initialization, and initial CTA barrier
precede the wait and do not read metadata.

Keep `prepare` before `attention` on the caller stream. Keep metadata allocated
through all readers; a later invocation may reuse it only after those readers
finish. Preserve the existing stream-ordered invocation discipline. PDL does not
make the shared workspace safe for concurrent callers on unrelated streams.

No data moves, address permutations, ownership changes, or additional buffers are
introduced. Keep sequence guards, inactive-tile exits, KV tail masking, and output
row guards. Keep every TMA transaction barrier, WGMMA completion wait, named
barrier, and buffer-reuse handshake. They synchronize the separate within-CTA
pipeline and cannot be replaced by PDL.

## Example configuration

Preserve this replay's CUDA13 SM90 target and build flags:
`-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.
The configured export remains `kernel.cu::kernel`, with destination-passing
arguments `q, k, v, qo, ko, output` and the PyTorch caller stream.

The workload has 8 sequences, 16,384 packed tokens, 64 heads, and head dimension
128. Q/K/V/output are FP16 contiguous NHD tensors with strides
`(8192, 128, 1)`; offsets are int32. Metadata holds `3*kBatch` integers.
`prepare<<<1,32,0,stream>>>` retains its guarded writes and warp shuffle.
Attention keeps `grid=kMaxQTiles*kHeads` (8,640 CTAs), `block=kThreads` (384), and
`dynamicSmemBytes=sizeof(Shared)`. Each active CTA computes one query tile/head.
The producer uses 32 threads; two 128-thread math groups consume its tiles.

Retain tiles `kM=128`, `kN=176`, `kDim=128`, two K/V stages, SW128 TMA layouts,
L2 cache policies, QK/PV WGMMA, register fragments, warp-shuffle reductions, and
online softmax. Preserve descending KV tile order, FP32 accumulations, probability
and output FP16 round-to-nearest conversion, and scalar output stores. PDL changes
none of these numerical operations. Preserve the complete problem and evaluator
policy; validate through `klineage.harness.evaluate`, with evidence outside this card.

# Precondition

- Data types: no additional requirement. Scheduling leaves data representations,
  arithmetic instructions, reduction order, and rounding unchanged.
- Layout: no additional contiguity, alignment, or thread-mapping requirement.
  No load/store address or value owner changes; reader ordering is covered below.
- Storage: producer results must occupy memory accessible to both grids and remain
  live through consumer reads. Private CTA storage cannot communicate these results
  across kernel launches. This recipe's existing global metadata meets that need.
- Pipeline: launches must be ordered in one stream to establish the programmatic
  dependency. Independent consumer startup supplies work that can execute early.
  Every producer CTA must be able to signal or finish before the runtime permits
  the dependent launch. Every dependent reader must execute after producer
  completion and visibility, directly through the wait or through a synchronized
  handoff from waiting threads. The producer must finish without consumer progress,
  since concurrent execution is not guaranteed. Results cannot be overwritten
  before readers finish; early scheduling does not establish safe buffer reuse.
- Hardware: SM90 or newer supports the required `griddepcontrol` instructions;
  the compiler and runtime must support these instructions and programmatic launch
  attributes. Available execution resources while the producer runs determine
  whether overlap occurs. No extra storage capacity is needed, and correctness
  must hold when execution remains serial.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The fragments replace only the indicated statements. Keep each function's
remaining body unchanged; the third fragment follows `config.stream = stream`.

## Before

```cuda
// solution/scheduler.cuh: start of prepare
const int lane = threadIdx.x;

int q = lane <= kBatch ? qo[lane] : 0;

// solution/scheduler.cuh: start of producer
const int lane = threadIdx.x % 32;
int4 work = tile(blockIdx.x, p.metadata);

// solution/kernel.cu: launch in kernel
// Stream order makes prepare's metadata visible before attention starts.
check(cudaLaunchKernelEx(&config, attention, p));
```

## After

```cuda
// solution/scheduler.cuh: start of prepare
const int lane = threadIdx.x;
asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

int q = lane <= kBatch ? qo[lane] : 0;

// solution/scheduler.cuh: start of producer
const int lane = threadIdx.x % 32;
asm volatile("griddepcontrol.wait;" ::: "memory");
int4 work = tile(blockIdx.x, p.metadata);

// solution/kernel.cu: launch in kernel
cudaLaunchAttribute attr{};
attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr.val.programmaticStreamSerializationAllowed = 1;
config.attrs = &attr;
config.numAttrs = 1;
check(cudaLaunchKernelEx(&config, attention, p));
```
