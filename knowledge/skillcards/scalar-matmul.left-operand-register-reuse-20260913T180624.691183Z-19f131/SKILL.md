---
skill_id: scalar-matmul.left-operand-register-reuse
intent: Reuse each scalar matrix-product left operand across a thread's output columns.
preconditions:
- 'Data types: no specific operand dtype is required; each repeated load and conversion
  must yield the same scalar representation, so reuse preserves each FMA operand and
  its rounding.'
- 'Layout: a thread uses the same left-operand address for multiple output columns
  at one reduction coordinate; the address must be independent of the column loop.
  No additional contiguity or alignment is required because loads remain scalar.'
- 'Storage: the left operand is repeatedly read from global memory by the same thread;
  its value must remain stable across those reads to permit one private register value
  to replace them.'
- 'Pipeline: the producer must finish and publish the operand before the hoisted read,
  and its storage cannot be overwritten until the last consumer finishes. Preserve
  existing visibility and buffer-reuse barriers; register reuse needs no collective
  participation or fixed stage count.'
- 'Hardware: scalar registers must hold the cached converted operand with the other
  live values; insufficient register capacity spills the cache and defeats register
  reuse. No special matrix or asynchronous-copy instruction is required.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reuse one loaded and converted left operand across a thread's output columns in
scalar QK and PV accumulation. In `solution/attention.cuh`, edit `qk_tile` and
`pv_smem`: inside each reduction-coordinate and row iteration, load `a` before
the output-column loop, then use it in both FMAs. This restores the same technique
at both matrix products; keep the per-thread accumulator tiles.

Replace the two loop regions shown below. Delete `load_operand` from
`solution/hopper.cuh` once neither region calls it. Its volatile global PTX load
exists only to prevent elimination or packing of repeated reads in the preceding
implementation. Use the ordinary scalar loads and BF16-to-FP32 conversion shown
in After; preserve all other volatile helpers and compiler flags.

QK reads `q[row_index(row, lane) * kWidth + k]`; PV reads
`s[prob_index(row_index(row, lane), k)]`. Neither address depends on output column
`i`. A thread privately owns `a`, and reuses it only within that row and reduction
coordinate. Keep `row_index`, `kv_index`, `prob_index`, all right-operand loads,
output ownership, and the physical global layouts unchanged. No launch, allocation,
or synchronization changes are needed.

Queries are ready on the caller stream before launch. Probability producers publish
their stores through the existing local and cross-group barriers before PV reads.
Keep those barriers and the completion ordering before probability/KV buffer reuse.
Do not hoist a load across its producer, across `k` or `row`, or across a barrier.
Keep sparse-index validity checks, zero-filling, score masking, and output bounds
unchanged. Each FMA receives the same scalar bits; retain reduction traversal,
FP32 `fmaf`, softmax arithmetic, and BF16 round-to-nearest conversion points.

Inspect compiled code to confirm one left-operand load feeds the column FMAs, with
no remaining calls to the forced-reload helper. Use the Kernel evaluator with the
unchanged problem for correctness and timing; keep its records outside this card.

## Example configuration

Preserve this replay's CUDA target `nvidia-sm90a-cuda13`, build flags, pybind11 ABI,
and caller stream. The fixed workload has 8192 tokens, 128 heads, QK width 576,
value width 512, and 2048 sparse selections. Q and KV are contiguous BF16;
indices are int32; accumulators, maxima, and LSE are FP32. Preserve the softmax
scale `0.1352337788608801f`, intermediate BF16 probabilities, and final BF16 output.

The launch has 16384 CTAs, 256 threads per CTA, and a one-CTA cluster. Each CTA
owns one token and 64 heads. Two groups of 128 threads use the existing lane
mapping; each thread owns two rows, 32 score registers, and 128 output registers.
For each row and `k`, QK replaces 16 repeated left loads with one; PV replaces 64
with one. The inner column loops remain unrolled, while the 64-coordinate
reduction loops retain `#pragma unroll 1`.

Keep QK's nine 64-wide calls and their order: group zero traverses tiles 0 through
8; group one traverses 4 through 8, then 0 through 3. PV retains increasing `k`
within each 64-wide tile and the existing block order. Each thread's output
accumulators, online softmax, shared reductions, and group handshakes remain.

Keep two global probability buffers of 4096 BF16 elements and two global gathered
KV buffers of 36864 BF16 elements per CTA. Keep the 1920-byte shared workspace,
all masks, layouts, scalar gather transfers, and scratch lifetimes. These settings
identify this replay; the reuse technique does not require their exact values.

# Precondition

- Data types: no specific operand dtype is required. Repeated loads and their
  conversions must yield the same scalar representation; otherwise replacing
  them with one value could change the FMA operands or rounding. Arithmetic and
  conversions of other operands remain unchanged.
- Layout: within one thread, the left-operand address must be independent of the
  output-column iteration at a fixed reduction coordinate. Reuse would select
  the wrong operand if the address varied. Scalar access introduces no additional
  contiguity or alignment requirement.
- Storage: the same thread repeatedly reads a stable left operand from global
  memory. One private register may replace those reads only while the value stays
  unchanged; there is no communication between threads in this transformation.
- Pipeline: the producer must complete and make its operand visible before the
  hoisted read. Every consumer must finish before that storage is overwritten.
  Keep the existing visibility and buffer-reuse barriers to prevent stale reads
  or premature writes. Reuse within a thread adds no collective participation
  requirement and imposes no fixed stage count.
- Hardware: available scalar registers must cover the cached converted operand
  and other simultaneously live values. Spilling the cache prevents the intended
  register reuse. Matrix instructions and asynchronous-copy hardware are not
  needed by this transformation.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets replace the reduction loops in the named functions. Keep their
signatures, pointer setup, accumulator initialization, and surrounding code.
Delete only the now-unused `load_operand` helper after both replacements.

## Before

```cuda
// qk_tile: forced reload at every score FMA.
    // Reload Q for each FMA; retain each lane's output tile and reduction order.
#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
#pragma unroll
        for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
            for (int i = row * kPairElems; i < kScoreRegs; i += kRegsPerStep) {
                const int col = kColsPerStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
                p[i] = fmaf(load_operand(q + row_index(row, lane) * kWidth + k),
                            __bfloat162float(key[kv_index(col, k)]), p[i]);
                p[i + 1] = fmaf(load_operand(q + row_index(row, lane) * kWidth + k),
                                __bfloat162float(key[kv_index(col + 1, k)]), p[i + 1]);
            }
        }
    }

// pv_smem: forced reload at every value FMA.
    // Reload each BF16 probability per FMA; retain the FP32 accumulators.
#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
#pragma unroll
        for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
            for (int i = row * kPairElems; i < kOutputRegs; i += kRegsPerStep) {
                const int col = kColsPerStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
                const int base = (col / kTile) * kTileElems;
                o[i] = fmaf(load_operand(s + prob_index(row_index(row, lane), k)),
                            __bfloat162float(v[base + kv_index(k, col % kTile)]), o[i]);
                o[i + 1] = fmaf(load_operand(s + prob_index(row_index(row, lane), k)),
                                __bfloat162float(v[base + kv_index(k, col % kTile + 1)]), o[i + 1]);
            }
        }
    }
```

## After

```cuda
// qk_tile: one query load per row and reduction coordinate.
    // Read Q and gathered KV globally; retain each lane's output tile.
#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
#pragma unroll
        for (int row = 0; row < kRowsPerThread; ++row) {
            const float a = __bfloat162float(q[row_index(row, lane) * kWidth + k]);
#pragma unroll
            for (int i = row * kPairElems; i < kScoreRegs; i += kRegsPerStep) {
                const int col = kColsPerStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
                p[i] = fmaf(a, __bfloat162float(key[kv_index(col, k)]), p[i]);
                p[i + 1] = fmaf(a, __bfloat162float(key[kv_index(col + 1, k)]), p[i + 1]);
            }
        }
    }

// pv_smem: one probability load per row and reduction coordinate.
    // Preserve BF16 probabilities and the per-lane FP32 output accumulators.
#pragma unroll 1
    for (int k = 0; k < kTile; ++k) {
#pragma unroll
        for (int row = 0; row < kRowsPerThread; ++row) {
            const float a = __bfloat162float(s[prob_index(row_index(row, lane), k)]);
#pragma unroll
            for (int i = row * kPairElems; i < kOutputRegs; i += kRegsPerStep) {
                const int col = kColsPerStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
                const int base = (col / kTile) * kTileElems;
                o[i] = fmaf(a, __bfloat162float(v[base + kv_index(k, col % kTile)]), o[i]);
                o[i + 1] = fmaf(a, __bfloat162float(v[base + kv_index(k, col % kTile + 1)]), o[i + 1]);
            }
        }
    }
```
