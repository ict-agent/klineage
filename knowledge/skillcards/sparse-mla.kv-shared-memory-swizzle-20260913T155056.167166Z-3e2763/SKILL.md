---
skill_id: sparse-mla.kv-shared-memory-swizzle
intent: Reduce shared-memory bank conflicts by swizzling KV operands.
preconditions:
- 'Data types: no additional arithmetic requirement; permute complete 16-byte payloads
  without conversion so element bits and reduction order remain unchanged.'
- 'Layout: known KV ownership and aligned, contiguous 16-byte copy segments; all readers
  must support the same 128-byte swizzle atoms (eight rows by eight segments), with
  consistent base phase and bounds, or they will address different values.'
- 'Storage: KV tiles already reside in CTA shared memory visible to both consumers;
  the permutation must stay within each tile''s allocation and lifetime, including
  any aliased storage.'
- 'Pipeline: finish prior readers before overwrite, complete and publish copies before
  either consumer reads, and finish all readers before buffer reuse; otherwise the
  new addresses can expose incomplete or overwritten operands.'
- 'Hardware: shared-memory descriptor readers must support 128-byte swizzling; capacity
  must cover the padded tile footprint, since the address permutation cannot escape
  its allocation.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Swizzle KV tiles in shared memory to spread accesses across banks. Update the
producer's destination mapping and both WGMMA readers together. This changes
addresses only: retain element bits, logical operands, reduction order, and rounding.

In `solution/hopper.cuh`, replace the constants from `kKvCoreStride` through
`kVStep` and replace `kv_index` with the After versions. Keep the preceding
core constants. In `solution/attention.cuh::copy_tiles`, change the row increment
from `kCopyGroups * kCoreElems` to `kCopyGroups * kTile`. Its `kv_index` call stays.
The existing `qk_tile` and `pv_smem` calls already use `kKStep` and `kVStep`;
the new constants update their descriptor advances. No other source changes are needed.

The preceding layout is
`I(r,c) = floor(c/E)*R*E + r*E + c%E`, where `E=16/sizeof(element)` and
`R=kTile`. Its core columns span every selected row before advancing columns;
this lets one N-major PV descriptor span adjacent value tiles. The optimized
example uses `I(r,c) = (r*kTile+c) XOR ((r%8)*E)`. Each XOR exchanges whole
16-byte segments within an eight-row, 128-byte-wide atom. Tile bases remain fixed.

QK reads KV as K-major B. PV reads the same bytes as N-major B. Set both
descriptor swizzle fields to mode 1, keep base phase zero, and update leading,
stride, and reduction-step offsets as below. Descriptor offsets use 16-byte units.
The descriptor rules are specified in NVIDIA's
[PTX shared-memory layout documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout).

Retain `free*.wait` before writes, `ready*.cp_arrive` after each copy group,
and consumer `ready*.wait` before WGMMA. Retain WGMMA waits before free arrivals.
The last K0 tile becomes P0 only after its QK readers finish; keep its probability
stores, fences, barriers, and unswizzled probability descriptor unchanged. That
later lifetime has its own layout. Keep mask publication and all phase toggles.

Invalid selected indices still issue zero-filled copies with byte count zero;
valid indices retain byte count 16. Preserve score masking and guarded numerical
outputs. Use the existing evaluator with the complete problem and build flags;
inspect generated descriptor modes and copy addressing to confirm the change.

## Example configuration

Preserve 8192 tokens, 128 heads, QK width 576, value width 512, and TOPK 2048.
The launch remains 16384 CTAs, 384 threads each, cluster `(1,1,1)`, caller stream,
and 231376 dynamic shared-memory bytes. Each CTA owns one token and 64 heads.
Groups 0/1 consume; group 2 produces. All three contain 128 threads.

Each KV buffer contains nine contiguous 64x64 BF16 tiles. Within the producer,
`group=lane/8`, `column=(lane%8)*8`, and copy row `j` owns logical row
`group+j*16`, for four rows per lane. Each copy moves eight BF16 elements.
Global KV row stride remains 576 elements (1152 bytes); tile stride in shared
memory remains 4096 elements (8192 bytes). KV origins at offsets 73728 and
147456 bytes have zero swizzle phase; every tile origin is 1024-byte aligned.

Unswizzled KV descriptors have encoded `(leading,stride)` values `(64,8)` for
QK and `(8,64)` for PV; their reduction-step advances are 128 and 16 units.
Restore mode 1 and `(1,64)` for QK, `(512,64)` for PV, with advances 2 and 128.
Keep Q and probability descriptors and their step sizes unchanged.

Retain two KV buffers, four transfer groups, resident Q, P0/K0 storage aliasing,
shared probability staging, online softmax, shared reduction exchanges, and direct
global output stores. Retain BF16 operands/probabilities/output, FP32 accumulators
and statistics, BF16 round-to-nearest conversion sites, softmax scale
`0.1352337788608801f`, and the existing QK/PV accumulation sequence.
QK uses `m64n64k16`; PV uses `m64n256k16`. Each consumer retains 32 score and
128 output registers per thread. Keep `mma.cuh`, the host ABI, all build flags,
and the SM90a compilation target unchanged.

# Precondition

- Data types: no additional arithmetic requirement. This permutation moves
  complete 16-byte payloads without interpreting or converting their elements.
  Moving partial elements or changing reduction order would alter numerical behavior.
- Layout: copy ownership and logical KV coordinates must be known; each copied
  segment is contiguous and 16-byte aligned. Every reader must interpret the same
  eight-row by eight-segment swizzle atom. Origins need a consistent swizzle phase
  (a 1024-byte-aligned origin permits phase zero), and partial atoms need valid
  padding or bounds handling. Mismatched descriptors or uncovered segments read
  incorrect values. These constraints concern shared addressing, not global row order.
- Storage: KV operands already occupy CTA shared memory accessible by both
  consumers. Each permutation stays within its tile allocation and lifetime.
  Aliased users must start after the KV readers finish and retain their own layouts;
  otherwise one lifetime can corrupt another.
- Pipeline: previous consumers must finish before producers overwrite their
  storage. Copies must complete and become visible before either reader starts.
  All readers must finish before reuse, including transitions to aliased users.
  These are ordering dependencies; no particular buffer count is required.
- Hardware: the operand readers must implement the matching 128-byte descriptor
  swizzle. Otherwise swizzled writes cannot be decoded. Available shared storage
  must cover `padded_rows * padded_row_bytes` per live tile; the permutation adds
  no storage but cannot address beyond that footprint.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are replacement regions, not a complete translation unit. Existing
`kDescUnitBytes`, `kCoreRows`, `kCoreElems`, `kCoreTileElems`, `kTile`,
`kTileElems`, and `Bf16` supply shared context. Keep the existing `kv_index`,
`kKStep`, and `kVStep` call sites.

## Before

```cuda
// solution/hopper.cuh: KV descriptor constants.
constexpr int kKvCoreStride = kTile * kCoreElems * sizeof(Bf16) / kDescUnitBytes;
constexpr int kKvRowStride = kCoreTileElems * sizeof(Bf16) / kDescUnitBytes;
constexpr uint64_t kKDesc = (uint64_t(kKvCoreStride) << 16) | (uint64_t(kKvRowStride) << 32);
constexpr uint64_t kMnDesc = (uint64_t(kKvRowStride) << 16) | (uint64_t(kKvCoreStride) << 32);
constexpr int kKvMmaK = 16;
constexpr int kKStep = kKvMmaK / kCoreElems * kKvCoreStride;
constexpr int kVStep = kKvMmaK / kCoreRows * kKvRowStride;

__device__ __forceinline__ int kv_index(int row, int col) {
    // Pack KV core columns contiguously for both unswizzled WGMMA readers.
    return (col / kCoreElems) * kTile * kCoreElems +
           row * kCoreElems + col % kCoreElems;
}

// solution/attention.cuh: copy_tiles inner call.
copy_kv(row_src + tile * kTile,
        dst + tile * kTileElems + row * kCopyGroups * kCoreElems,
        valid[Buffer][row] ? 16 : 0);
```

## After

```cuda
// solution/hopper.cuh: matching swizzled QK/PV descriptors and steps.
constexpr uint64_t kSwizzle128 = 1ULL << 62;
constexpr int kKvAtomUnits = kCoreRows * kTile * sizeof(Bf16) / kDescUnitBytes;
constexpr int kKvTileUnits = kTileElems * sizeof(Bf16) / kDescUnitBytes;
constexpr uint64_t kKDesc = kSwizzle128 | (1ULL << 16) | (uint64_t(kKvAtomUnits) << 32);
constexpr uint64_t kMnDesc = kSwizzle128 | (uint64_t(kKvTileUnits) << 16) | (uint64_t(kKvAtomUnits) << 32);
constexpr int kKvMmaK = 16;
constexpr int kKStep = kKvMmaK * sizeof(Bf16) / kDescUnitBytes;
constexpr int kVStep = kKvMmaK * kTile * sizeof(Bf16) / kDescUnitBytes;

__device__ __forceinline__ int kv_index(int row, int col) {
    // Exchange complete segments within each swizzle atom.
    const int offset = row * kTile + col;
    return offset ^ ((row % kCoreRows) * kCoreElems);
}

// solution/attention.cuh: copy_tiles inner call.
copy_kv(row_src + tile * kTile,
        dst + tile * kTileElems + row * kCopyGroups * kTile,
        valid[Buffer][row] ? 16 : 0);
```
