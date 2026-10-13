---
skill_id: fmha.shared-output-redistribution
intent: Redistribute output fragments through shared memory for coalesced vector stores.
preconditions:
- 'Data types: output payloads must be representable as pairs of 16-bit elements in
  32-bit registers for stmatrix.b16; preserve existing conversion and rounding. No
  additional arithmetic dtype is required because redistribution copies bits.'
- 'Layout: register ownership must match the non-transposed m8n8 matrix-store fragment,
  or admit an explicit adapter. Shared row addresses and contiguous global output
  runs must support aligned 16-byte accesses; known strides and unique tile ownership
  are needed to gather and store each element once.'
- 'Storage: completed outputs reside in registers before global stores, and CTA-local
  scratch is available for the output tile after its prior users finish. Cross-CTA
  exchange cannot use this shared-memory redistribution.'
- 'Pipeline: output producers and prior scratch users must finish before staging;
  all warp lanes must execute each matrix store together, and all participating consumers
  must reach the visibility barriers. Every shared read must finish before scratch
  is released to its next producer, with required proxy ordering for asynchronous
  reuse.'
- 'Hardware: SM90 or later with stmatrix.m8n8.b16, CTA shared memory, and barriers.
  Reusable scratch must cover tile_rows * tile_columns * sizeof(output_element), and
  simultaneous live allocations must fit the CTA shared-memory limit.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Redistribute completed output fragments through CTA shared memory, then gather
contiguous vectors for global stores. This replaces each lane's scattered pair
stores with fewer, wider stores while preserving the rounded output bits.

In `solution/attention.cuh::epilogue`, replace everything after the existing
`packed` conversion loop with the After snippet. Retain the function signature,
local index setup, conversion loop, and caller. All needed helpers already exist
in `solution/ops.cuh`: `offset<kM>`, `shared_addr`, `sync`, `proxy_fence`, and
`signal`. No other source or launch change is needed.

## Ownership and storage

For consumer lane `l` in warp `w`, packed pair `i` owns tile row
`16*w + l/4 + 8*(i%2)` and columns `8*(i/2) + 2*(l%4) + {0,1}`.
Each four consecutive packed registers describe four non-transposed 8-by-8
matrices: upper-left, lower-left, upper-right, lower-right. The matrix store
writes these into the linear panel layout already defined by
`offset<Rows>(row,col) = row*64 + col%64 + (col/64)*Rows*64`.

Reuse the beginning of `s.v` for the output tile. This introduces output/V
storage aliasing; keep the input TMA layout and tensor-map swizzle unchanged.
Output staging uses the linear `offset` layout, independently of the old V
layout. The After snippet supplies each lane's matrix-row address explicitly.

After staging, consumer `tid` gathers vectors at tile rows `tid/8 + 32*r`
and columns `8*(tid%8) + 64*c`, for `r=0..3`, `c=0..1`. Each vector contains
eight consecutive output elements. Write to
`p.output + (start + row)*kRow + work.z*kDim + col`, where global row stride is
`heads * head_dim * sizeof(output_element)` bytes. The writer and reader
mappings cover the same tile exactly once.

## Ordering and bounds

The existing consumer finishes its final PV operation with `mma_wait<0>()`
before the epilogue. The first epilogue barrier joins all consumers before
reusing V storage. Execute every matrix store in every participating warp;
the second barrier makes all staged values visible before the vector gather.

Keep the producer's existing `o_empty` initialization, parity waits, and
per-consumer arrival count. Move the epilogue signal from after the direct
stores to after every shared gather and `proxy_fence()`. All consumers must
release the buffer before the next producer overwrites V. Gathered vectors
then survive in private storage while global stores finish. No final shared
barrier is needed after those global stores.

Keep all threads participating for partial query tiles. Stage and gather the
whole allocated tile; guard global output and LSE writes with `row < length`.
Column bounds remain covered by this specialization. Preserve the LSE stores,
FP32 arithmetic, softmax order, scale, and `half_pair` round-to-nearest conversion.
The transformation changes only output movement. Keep the ABI, device guard,
caller stream, workload, oracle, and numerical tolerances. Validate with
`klineage.harness.evaluate`; keep results outside this card.

## Example configuration

Preserve the supplied packed contiguous FP16 `[16384,64,128]` tensors, eight
sequences, int32 offsets, FP32 accumulators/LSE, noncausal attention, no dropout,
and scale `1/sqrt(128)`. Tiles use `kM=128`, `kN=176`, `kDim=128`.
The launch uses `grid=scratch.sms`, `block=kThreads=384`, and
`dynamicSmemBytes=sizeof(Shared)`. Consumers are threads 128..383: two groups,
eight warps, 256 threads. Their 64 output registers become 32 packed pairs;
each consumer gathers eight `uint4` vectors.

The output tile occupies 32768 bytes within the existing 90112-byte V allocation;
no extra shared allocation is needed. Preserve two K/V stages, Q/K/V TMA loads,
SW128 input layouts, WGMMA instructions and fragment adapters, online softmax,
warp-specialized overlap, 24/240 register budgets, persistent work assignment,
cache hints, and programmatic launch ordering. Preserve compiler flags and
`config.toml`. These instance settings are retained for this replay, not general
requirements of output redistribution.

# Precondition

- Data types: the matrix store uses 16-bit payloads packed two per 32-bit register.
  Other element widths do not fit this instruction. Preserve conversion before
  redistribution; no further arithmetic dtype restriction applies to moving bits.
- Layout: each warp must supply the non-transposed m8n8 fragment expected by
  `stmatrix`, directly or through an adapter. Incorrect ownership permutes values.
  Matrix-row addresses and vector accesses require 16-byte alignment; global
  vector runs must be contiguous. Known strides and unique tile ownership prevent
  misplaced or duplicate stores. Physical writer/reader indices must agree.
- Storage: completed results must be available in registers, with CTA-local
  scratch available after its previous users finish. This enables the intermediate
  store/gather; this CTA-local exchange cannot serve consumers in other CTAs.
- Pipeline: finish output production and previous scratch accesses before staging.
  All warp lanes must reach the same matrix-store instructions; all participating
  consumers must reach barriers that make staged writes visible. Delay reuse until
  every shared gather completes, including proxy ordering when the next user is
  asynchronous. These rules prevent incomplete fragments, deadlock, and overwrite
  races; no fixed stage count is required.
- Hardware: the matrix-store instruction requires SM90 or later. CTA shared memory
  and barriers implement the exchange. Reusable scratch must hold
  `tile_rows * tile_columns * sizeof(output_element)` bytes, and concurrently
  live allocations must fit the CTA limit; otherwise staging overlaps live data
  or exceeds launch capacity.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both snippets replace the tail of `epilogue` after its unchanged `packed` loop.
They use existing `tid`, `lane`, `warp`, `start`, `length`, and `packed` locals.

## Before

```cuda
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        int row = work.y * kM + warp * 16 + lane / 4 + r * 8;
        if (lane % 4 == 0 && row < length) p.lse[work.z * kTokens + start + row] = lse[r];
    }

    // Store each lane's rounded pairs directly from its accumulator layout.
    constexpr int kRowsPerLane = 2;
    constexpr int kLanesPerRow = 4;
    constexpr int kPairElems = 2;
    constexpr int kGroupRows = kWarpSize / kLanesPerRow;
    constexpr int kGroupCols = kLanesPerRow * kPairElems;
    #pragma unroll
    for (int i = 0; i < kPvRegs / kPairElems; ++i) {
        int row = work.y * kM + warp * kRowsPerLane * kGroupRows
                  + lane / kLanesPerRow + (i % kRowsPerLane) * kGroupRows;
        int col = (i / kRowsPerLane) * kGroupCols + (lane % kLanesPerRow) * kPairElems;
        if (row >= length) continue;
        *reinterpret_cast<uint32_t*>(p.output + (start + row) * kRow + work.z * kDim + col) = packed[i];
    }

    // Retain the producer's tile-completion handshake.
    signal(&s.o_empty);
```

## After

```cuda
    // V and O share storage. All consumers finish PV before the STSM epilogue.
    sync(Named::Epilogue, kMathThreads);
    #pragma unroll
    for (int i = 0; i < kDim / kMmaK; ++i) {
        int row = warp * 16 + lane % 16;
        int col = i * 16 + (lane / 16) * 8;
        uint32_t addr = shared_addr(s.v + offset<kM>(row, col));
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};"
                     :: "r"(addr), "r"(packed[i*4]), "r"(packed[i*4+1]),
                        "r"(packed[i*4+2]), "r"(packed[i*4+3]) : "memory");
    }
    sync(Named::Epilogue, kMathThreads);
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        int row = work.y * kM + warp * 16 + lane / 4 + r * 8;
        if (lane % 4 == 0 && row < length) p.lse[work.z * kTokens + start + row] = lse[r];
    }
    uint4 values[8];
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int row = tid / 8 + r * 32;
            int col = (tid % 8) * 8 + c * 64;
            values[c*4+r] = *reinterpret_cast<uint4*>(s.v + offset<kM>(row, col));
        }
    }
    proxy_fence();
    signal(&s.o_empty);
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int row = work.y * kM + tid / 8 + r * 32;
            int col = (tid % 8) * 8 + c * 64;
            if (row < length) *reinterpret_cast<uint4*>(p.output + (start + row) * kRow + work.z * kDim + col) = values[c*4+r];
        }
    }
```
