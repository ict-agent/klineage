---
skill_id: topk.radix-threshold-selection
intent: Find the Top-K cutoff by radix refinement before assigning output positions.
preconditions:
- 'Data types: keys need an order-preserving unsigned encoding whose full bit width
  can be refined; counters and indices must represent the row length, or bucket selection
  and output ranks can overflow. Output values must retain their original representation.'
- 'Layout: known row bounds, strides, and original indices must let one CTA enumerate
  all keys without omissions or duplication. Input/output storage must not overlap
  during rereads, and the output contract must permit unsorted results because threshold
  emission does not sort survivors.'
- 'Storage: the complete input row must remain readable in global memory by every
  participating thread; partial row visibility cannot determine its Top-K cutoff.'
- 'Pipeline: input production must finish before selection and the input must remain
  unchanged until emission finishes. Every CTA thread must be able to join barriers
  before histogram reads, cutoff consumption, and histogram reuse; otherwise readers
  can observe incomplete or overwritten state.'
- 'Hardware: CUDA CTA-shared memory and block barriers are required for bucket exchange;
  available shared capacity must cover the histogram plus aligned selection state,
  and the participating thread count must fit one CTA.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace the all-key rank loop in `solution/topk.cuh::emit` with a CTA-wide
radix cutoff and survivor-only position counting. Refine the unsigned keys
returned by the existing `ordered`: smaller encodings select larger scores.
The Kth encoding is `splitter`; `remaining` counts how many keys equal to that
cutoff are needed. Scan all radix digits, including the final partial digit.
Do not change `ordered`, input/output types, or the problem oracle.

The direct kernel assigns ranks in encoded-value order. The optimized emission
places keys strictly better than the cutoff first, in input-index order, then
the earliest required cutoff ties. Both choose the same keys and preserve each
value's bits and original index. Output order and tied indices are unspecified
by this problem. No score arithmetic, rounding, or tolerance changes are needed.

## Example configuration

Preserve `kRows=1`, `kLength=131072`, `kSelected=2048`, `kThreads=512`,
`kBlocks=1`, `kKeyBits=32`, `kChunkBytes=16*1024`, and `kAlign=128`.
The contiguous FP32 input is `[1,131072]`; FP32 values and int64 indices are
`[1,2048]`. Row strides are their row extents in elements. Keep the existing
32 chunks of 4096 items, head/tail bounds, offset/count arrays, and striped
emission ownership `index=threadIdx.x; index<count; index+=kThreads`.

Restore `kRadixBits=11`, `kPasses=(kKeyBits+kRadixBits-1)/kRadixBits`,
`kBuckets=1<<kRadixBits`, and `kBucketsPerThread=kBuckets/kThreads`.
This gives three passes at bit positions 21, 10, and 0; the last digit uses
10 bits. Histogram/reset owner `tid` owns `tid+i*kThreads`; prefix/choice owner
`tid` inspects `tid*kBucketsPerThread+i`. Each covers all buckets once.
Preserve four buckets per thread for this replay.

Keep `config.toml`, `solution/kernel.cu`, and `solution/native.cuh` unchanged.
The launch remains grid `(1,1,1)`, block `(512,1,1)`, zero dynamic shared bytes,
and `__launch_bounds__(kThreads,1)`. Keep the host's 25% shared carveout hint,
tensor/device checks, device guard, destination ABI, and caller CUDA stream.
The restored static allocation is 8208 bytes, including the aligned state.
Preserve scalar global rereads, single-key processing, exclusive histogram
writers, direct bucket-prefix sums, and direct survivor position counts.
No atomics, cooperative scan, input cache, async copy, register key batches,
cluster communication, early pass termination, or launch change is introduced.

## Restore the cutoff helpers

Add the radix constants above beside the existing constants. Before `ordered`,
add the following shared types and pass enum:

```cuda
struct alignas(8) State {
  unsigned candidates;
  unsigned remaining;
  unsigned bucket;
};

struct Shared {
  unsigned hist[kBuckets];
  State state;
};
constexpr int kStaticBytes = 8208;
static_assert(sizeof(Shared) == kStaticBytes);

enum class Pass { first, later };
```

Add device-forceinline helpers after `ordered`:

- `int start_bit(int pass)` returns
  `max(0, kKeyBits-(pass+1)*kRadixBits)`; `start_bit(-1)` is the key width.
- `void reset(Shared& sm)` zeroes `sm.hist[threadIdx.x+i*kThreads]` for
  `i` in `[0,kBucketsPerThread)`, with `#pragma unroll`.
- `void scan(Shared& sm, unsigned (&counts)[kBucketsPerThread],
  unsigned (&prefixes)[kBucketsPerThread])` sets
  `first=threadIdx.x*kBucketsPerThread`, loads `counts[i]=sm.hist[first+i]`,
  and directly sums `sm.hist[0:first]` starting from `unsigned prefix=0`. Then sets
  `prefixes[i]=prefix; prefix+=counts[i]` in increasing `i` order.
  Unroll the two fixed-size array loops; keep the preceding-bucket sum scalar.
- `void choose(Shared& sm)` first snapshots `target=sm.state.remaining`, then
  calls `scan` on local `counts`/`prefixes` arrays. In an unrolled array loop,
  continue when `prefixes[i]>=target || prefixes[i]+counts[i]<target`.
  The unique surviving bucket writes `sm.state.candidates=counts[i]`,
  `sm.state.remaining=target-prefixes[i]`, and
  `sm.state.bucket=threadIdx.x*kBucketsPerThread+i`. Preserve these direct
  prefix counts; do not substitute a collective scan.

Use this histogram helper. Each bucket owner rereads every chunk independently;
later passes include only keys with the previously selected prefix.

```cuda
template <Pass mode>
__device__ __forceinline__ void histogram(Shared& sm, const float* keys, int count,
    unsigned splitter, int prior_bit, int bit, unsigned mask) {
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    const unsigned bucket = threadIdx.x + i * kThreads;
    unsigned total = 0;

    #pragma unroll 1
    for (int index = 0; index < count; ++index) {
      const unsigned bits = ordered(keys[index]);
      if constexpr (mode == Pass::later) {
        if (((bits >> prior_bit) << prior_bit) != splitter) continue;
      }
      if (((bits >> bit) & mask) == bucket) ++total;
    }

    sm.hist[bucket] += total;
  }
}
```

## Restore selection and ordering

At the start of `select`, declare `__shared__ Shared sm` and
`const bool leader_thread=threadIdx.x==0`. Preserve the existing chunk setup.
After that setup, insert:

```cuda
if (leader_thread) sm.state = {kLength, kSelected, 0};
reset(sm);
__syncthreads();

#pragma unroll 1
for (int chunk = 0; chunk < kChunks; ++chunk)
  histogram<Pass::first>(sm, input + offsets[chunk], counts[chunk],
      0, 0, start_bit(0), kBuckets - 1);
histogram<Pass::first>(sm, input, head_count, 0, 0, start_bit(0), kBuckets - 1);
histogram<Pass::first>(sm, input + kLength - tail, tail_count,
    0, 0, start_bit(0), kBuckets - 1);
__syncthreads();

unsigned splitter = 0;
int final_bit = 0;
for (int pass = 0; pass < kPasses; ++pass) {
  const int bit = start_bit(pass);
  const int prior_bit = start_bit(pass - 1);
  const unsigned mask = (1u << (prior_bit - bit)) - 1;
  if (pass != 0) {
    #pragma unroll 1
    for (int chunk = 0; chunk < kChunks; ++chunk)
      histogram<Pass::later>(sm, input + offsets[chunk], counts[chunk],
          splitter, prior_bit, bit, mask);
    histogram<Pass::later>(sm, input, head_count, splitter, prior_bit, bit, mask);
    histogram<Pass::later>(sm, input + kLength - tail, tail_count,
        splitter, prior_bit, bit, mask);
  }
  __syncthreads();
  choose(sm);

  if (pass + 1 < kPasses) {
    __syncthreads();
    reset(sm);
  }
  __syncthreads();
  splitter |= sm.state.bucket << bit;
  final_bit = bit;
}
__syncthreads();
```

Reset completion precedes histogram accumulation. Histogram completion precedes
prefix reads. Choice readers finish before histogram reuse, and state writes
become visible before cutoff consumption. All CTA threads participate, including
those without an emission item. First-pass specialization avoids shifting a key
by its full width; later prefix shifts and digit-mask shifts remain in range.
Unused buckets in the last partial digit stay zero. Process head and tail once
per histogram pass and emission; their existing counts guard bounds.

Change `emit` to accept `(Shared& sm, const float* keys, int count,
int global_base, unsigned splitter, int bit, float* values, int64_t* indices)`.
In all three existing calls, prepend `sm` and insert `splitter,final_bit` after
`global_base`. Preserve chunk/head/tail traversal and the outer `emit` loop.
Replace only its body as shown below. The full-digit cutoff makes `bit=0`
at emission; `kSelected-sm.state.remaining` is the count of strictly better keys.
Counting earlier keys within each output region gives disjoint slots. The final
output bound discards excess ties. Keep the original float stores and int64 cast.

For an application of this recipe, use the existing Kernel evaluator on the
unchanged problem for correctness and latency. Store evidence outside this card.

# Precondition

- Data types: radix refinement needs an unsigned encoding that preserves the
  required key ordering across its full bit width. Otherwise the selected digit
  path need not contain the Kth key. Counts and indices must represent the row
  length to avoid overflow in histogram sums and output positions. Copy values
  in their original representation. FP32 itself is an example setting, not a
  restriction of radix selection.
- Layout: known row bounds, strides, and original indices must let one CTA
  enumerate all keys without omissions or duplication; otherwise histogram
  totals and output positions are wrong. Input/output overlap would let emission
  corrupt later rereads. The output contract must permit unsorted results because
  threshold emission does not sort survivors. Specific contiguity, alignment,
  bucket counts, and thread counts can change with adapted indexing and ownership.
- Storage: the entire input row must remain globally readable by all threads
  participating in selection. A cutoff based on only part of the row cannot
  select its global Top-K. No preexisting shared input cache is required.
- Pipeline: input production must complete before selection, and input values
  must remain stable through emission. All CTA threads must be able to reach
  histogram-publication, cutoff-publication, and reuse barriers. These enforce
  producer completion, reader visibility, and completion before overwrite;
  divergent participation would invalidate the exchange or deadlock. No fixed
  pass or buffer-stage count is a prerequisite.
- Hardware: the participating threads must fit one CUDA CTA with shared memory
  and CTA barriers. Shared capacity must cover
  `align_up(bucket_count*sizeof(counter),alignof(state))+sizeof(state)`, rounded
  for the containing structure's alignment. Otherwise the histogram and state
  cannot coexist. This technique requires no architecture-specific copy or
  cluster instructions.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These replace the body of the existing striped loop in `emit`; the cutoff
helpers, shared allocation, synchronization, and call-site edits are above.

## Before

```cuda
const float value = keys[index];
const unsigned bits = ordered(value);
const float* input = keys - global_base;
unsigned out = 0;

#pragma unroll 1
for (int other = 0; other < kLength; ++other) {
  const unsigned other_bits = ordered(input[other]);
  if (other_bits < bits || (other_bits == bits && other < global_base + index)) ++out;
}

if (out >= kSelected) continue;
values[out] = value;
indices[out] = static_cast<int64_t>(global_base + index);
```

## After

```cuda
const float value = keys[index];
const unsigned bits = (ordered(value) >> bit) << bit;
if (bits > splitter) continue;

const float* input = keys - global_base;
unsigned out = bits == splitter ? kSelected - sm.state.remaining : 0;
#pragma unroll 1
for (int prior = 0; prior < global_base + index; ++prior) {
  const unsigned prior_bits = (ordered(input[prior]) >> bit) << bit;
  if (bits == splitter ? prior_bits == splitter : prior_bits < splitter) ++out;
}

if (out >= kSelected) continue;
values[out] = value;
indices[out] = static_cast<int64_t>(global_base + index);
```
