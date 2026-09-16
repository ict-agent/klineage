---
skill_id: cuda.output-shared-swizzle-128b
intent: Reduce output shared-memory bank conflicts with a 128-byte XOR swizzle.
preconditions:
- 'Data types: no additional dtype or arithmetic requirement; permute existing output
  bits in 16-byte atoms without changing conversion or rounding.'
- 'Layout: row-major staged tiles have known, unique writer coordinates and a tensor
  TMA reader; the inner byte extent fits a 128-byte swizzle span, global/shared addresses
  are 128-byte aligned, and the shared base phase within the 1024-byte pattern is
  known so writer and reader mappings agree.'
- 'Storage: output is already staged in CTA shared memory before tensor TMA writes
  global output; storage must cover whole 128-byte swizzle spans, including padding,
  to keep permuted addresses within allocation.'
- 'Pipeline: no additional stage-count requirement; preserve output-producer completion,
  completion of prior workspace readers, publication by every writer to the TMA issuer,
  and storage lifetime until asynchronous reads finish, so changing layout introduces
  no visibility or reuse race.'
- 'Hardware: the tensor TMA reader must support the 128-byte swizzle; 32 four-byte
  shared-memory banks let its 16-byte atom permutation spread row groups across banks,
  and capacity must cover padded staged spans plus other simultaneously live storage.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reduce shared-memory bank conflicts in the output epilogue with a 128-byte XOR
swizzle. Keep the same output elements, owners, conversion, staging storage, and
TMA transfers. Change both the shared writer and the output tensor-map reader;
changing only one permutes the global result incorrectly.

In `solution/attention.cuh`, replace the `save_output(converted,...)` call in
`store_output` with the existing `save_prob(converted,...)` call shown below.
Delete the now-unused `save_output` helper. `save_prob` already maps identical
fragment coordinates through `swizzle(real_row,col)`. Leave its probability
callers and implementation unchanged.

In `solution/kernel.cu`, set the output map's last argument to
`CU_TENSOR_MAP_SWIZZLE_128B`. Keep the query map at that same existing mode and
retain the configurable `tensor_map` helper. Its encoder must continue forwarding
the `swizzle` argument. No launch, shared-allocation, ABI, or compiler-flag change
is needed. Keep `solution/hopper.cuh` and `solution/mma.cuh` unchanged.

## Mapping and ordering

For the current element indexing, `u=row*kTile+col` becomes
`u ^ ((u & (7<<6))>>3)`. In bytes, a zero-phase span uses
`b ^ ((b & (7<<7))>>3)`: each row permutes eight 16-byte atoms, preserving bytes
inside each atom. The 128-byte TMA reader reverses the same permutation into
unchanged row-major global output. A nonzero base phase must be included in the
row component; the phase is `(shared_base/128)%8`. Preserve this instance's
zero-phase shared base and tile offsets.

The writer ownership is `row_index(row,lane)=(lane/32)*16+row*8+(lane%32)/4`.
For `i=row*2, row*2+4, ...`, that lane writes columns
`8*(i/4)+(lane%4)*2` and the next column. With linear 128-byte rows, lanes writing
the same within-row word across different rows collide in the same bank group.
The XOR distributes those row groups across the bank span. This depends on the
writer coordinates, not on which arithmetic instructions produced the fragment.

Keep `wait<0>()` before consuming output registers, normalization, BF16 conversion,
`shared_fence()`, the group-specific store barrier, elected writer, and
`store_commit()`. Every writer publishes its stores before the TMA issuer reads.
`reduce_sum` remains before the epilogue; its consumer barrier follows the final
QK/PV completions, allowing Q storage to become output storage. The consumer groups
write disjoint tiles, and no output tile is overwritten again in this CTA.
Preserve that lifetime through asynchronous reads and the caller-stream ordering
of downstream consumers. The existing bulk commit is not a completion wait;
any later design that recycles these tiles must wait for their reads first.
This replay adds no synchronization or buffer reuse.

Build the final frozen bundle and use `klineage.harness.evaluate` with the
unchanged problem. Inspect generated shared-store addressing and the output
map mode together; timing alone does not establish the permutation.

## Example configuration

Preserve `TOKENS=8192`, `HEADS=128`, QK width `576`, value width `512`, and
`TOPK=2048`. Keep BF16 Q/KV/output, FP32 accumulators and statistics, the scale
`0.1352337788608801f`, online softmax reduction order, probability conversion,
and final `__float2bfloat16_rn` after normalization. The permutation changes no
arithmetic and adds no rounding. Keep the complete problem and numerical gates.

Keep `kTile=64`, `kTileElems=4096`, `kHalfTiles=4`, 384 threads, two 128-thread
consumer groups, one producer group, consumer/producer register budgets 216/72,
32 probability elements and 128 output elements per consumer thread, and two rows
per thread. Retain QK/PV WGMMA, register/shared probability operands, both KV
buffers, four copy groups, async QK/PV overlap, query TMA, KV cache policy,
probability and operand swizzles, and bulk statistics stores.

Launch 16384 CTAs with cluster `(1,1,1)` and 230864 dynamic shared bytes. Keep
`9.0a` compilation, the recorded compiler flags, tensor checks, and caller stream.
Each CTA owns one token and 64 heads; group `Group` owns output columns starting
at `Group*kHalfTiles*kTile`. Each group writes four separate 64-by-64 tiles.
The existing `sm.q_o` allocation is 73728 bytes; output uses its first 65536 bytes
as eight 8192-byte tiles. Tile offsets preserve the zero swizzle phase. Keep the
allocation and offsets, including the prior Q lifetime.

The output tensor map keeps BF16 encoding, dimensions `{kValue,kHeads,kTokens}`,
byte strides `{kValue*sizeof(Bf16),kValue*kHeads*sizeof(Bf16)}`, box
`{kTile,kTile,1}`, unit element strides, no interleave, no L2 promotion, and no
floating OOB fill. Only its swizzle mode changes. These rows occupy exactly
128 bytes, so this replay needs no padding. Global output remains contiguous
`[token,head,value]`. The workload fills all output tiles; retain current output
bounds behavior and all sparse-index masking. Query maps stay swizzled throughout.

# Precondition

- Data types: no additional dtype or arithmetic requirement. This technique
  permutes 16-byte atoms of already converted output and leaves every bit inside
  them intact. Preserve conversion and rounding; operand or accumulator types do
  not determine this memory permutation.
- Layout: the preceding row-major shared tiles need known, unique writer
  coordinates and a tensor TMA reader, or the two sides cannot agree on logical
  element placement. The inner byte extent must fit a 128-byte span, and both
  global and shared addresses must meet 128-byte alignment for this swizzle mode.
  Account for the shared base phase within the 1024-byte repeating pattern;
  ignoring it changes which atoms the reader associates with a row.
- Storage: output is already staged in CTA shared memory and copied to global
  memory by tensor TMA. Allocate complete 128-byte swizzle spans, padding a
  narrower inner extent when needed; otherwise valid logical elements can be
  permuted beyond the available shared storage. No storage-level move is required.
- Pipeline: no additional stage-count requirement. Preserve producer completion
  before output reads, completion of previous workspace readers before writes,
  publication from every writer to the TMA issuer, and completion of asynchronous
  reads before storage reuse. The permutation cannot make incomplete, invisible,
  or overwritten values safe; retain the preceding ordering and participation.
- Hardware: the existing tensor TMA reader must implement the 128-byte swizzle;
  otherwise it cannot decode the writer layout. The mapping of 32 four-byte
  shared-memory banks lets the 16-byte atom permutation separate colliding row
  groups; a different bank mapping may not reduce these conflicts. Available capacity
  must cover all padded swizzle spans plus unrelated live data, or the shared
  layout exceeds its allocation. Retained arithmetic hardware adds no dependency.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The complete Before helper is deleted by the forward change. `save_prob`,
`swizzle`, and `tensor_map` already exist in the deoptimized bundle.

## Before

```cuda
__device__ __forceinline__ void save_output(const Bf16* p, Bf16* dest, int lane) {
    constexpr int kLanesPerRow = 4;
    constexpr int kPairElems = 2;
    constexpr int kColStep = kLanesPerRow * kPairElems;
    constexpr int kRegsPerStep = kRowsPerThread * kPairElems;

    // Store each lane's elements at the row-major coordinates.
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
        const int real_row = row_index(row, lane);
#pragma unroll
        for (int i = row * kPairElems; i < kScoreRegs; i += kRegsPerStep) {
            const int col = kColStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
            dest[real_row * kTile + col] = p[i];
            dest[real_row * kTile + col + 1] = p[i + 1];
        }
    }
}
```

```cuda
// attention.cuh: inside store_output, after converting this tile.
save_output(converted, sm.q_o + (Group * kHalfTiles + tile) * kTileElems, lane);

// kernel.cu: Maps initialization in the host wrapper.
const Maps maps{tensor_map(q.data_ptr<at::BFloat16>(), kWidth, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                           CU_TENSOR_MAP_SWIZZLE_128B),
                tensor_map(output.data_ptr<at::BFloat16>(), kValue, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                           CU_TENSOR_MAP_SWIZZLE_NONE)};
```

## After

```cuda
// attention.cuh: delete save_output; reuse the existing swizzled writer.
save_prob(converted, sm.q_o + (Group * kHalfTiles + tile) * kTileElems, lane);

// Existing save_prob writes these coordinates inside its unchanged loops:
// dest[swizzle(real_row, col)] = p[i];
// dest[swizzle(real_row, col + 1)] = p[i + 1];

// kernel.cu: only the output map's mode changes.
const Maps maps{tensor_map(q.data_ptr<at::BFloat16>(), kWidth, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                           CU_TENSOR_MAP_SWIZZLE_128B),
                tensor_map(output.data_ptr<at::BFloat16>(), kValue, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B)};
```
