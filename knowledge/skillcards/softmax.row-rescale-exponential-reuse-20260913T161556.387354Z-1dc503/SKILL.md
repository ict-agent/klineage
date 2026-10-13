---
skill_id: softmax.row-rescale-exponential-reuse
intent: Reuse each row’s softmax rescaling exponential across its consumers.
preconditions:
- 'Data types: repeated exponentials must have identical arguments, result representation,
  and rounding semantics; sharing their result must preserve every consuming operation.'
- 'Layout: each consuming element must map to the correct row within its owning thread;
  otherwise a cached factor can rescale another row. No additional contiguity or alignment
  is required.'
- 'Storage: discarded argument loads must not observe external updates or implement
  required communication; otherwise caching changes memory behavior. No particular
  memory level is required.'
- 'Pipeline: maxima must be complete and visible before computing their difference,
  and each cached factor must remain unchanged until its last consumer. Scalar reuse
  introduces or repurposes no buffer, so reader completion and buffer reuse gain no
  additional requirement.'
- 'Hardware: no additional device feature or fixed capacity requirement; the transformation
  uses ordinary thread-local scalar storage and the existing exponential operation.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Compute each row's rescaling factor once, then reuse it for probabilities,
output accumulators, and the normalization sum. This eliminates repeated
`exp2f(old_max - new_max)` evaluations within the same thread. Keep the
maximum difference and all consuming arithmetic unchanged.

Apply these edits in `solution/attention.cuh`:

1. In `softmax<Group>`, replace `const float delta = m[row] - next[row];`
   with `const float scale = exp2f(m[row] - next[row]);`. Replace every
   `rescale_exp(delta)` in that function with `scale`. This includes both
   output elements in the inner loop and the final `l[row]` update.
2. In the `Group == 0` branch of `consume<Group>`, following
   `sync<Barrier::Max1>()`, change `float next[2], delta[2];` to
   `float next[2], scale[2];`. Replace
   `delta[row] = m[row] - next[row];` with
   `scale[row] = exp2f(m[row] - next[row]);`, before assigning `m[row]`.
   Replace every `rescale_exp(delta[row])` there with `scale[row]`, including
   both probability elements, both output elements, and `l[row]`.
3. Remove `rescale_exp` from `solution/hopper.cuh` once it has no callers.
   Its volatile local argument prevents compiler reuse in the preceding code;
   deleting the helper removes those local stores and loads and lets one
   explicitly hoisted exponential serve all uses.
   Keep `output_scale` and its separate reciprocal-recomputation behavior.

Each thread owns its factors; do not exchange them across lanes. Keep
`row_index`, the interleaved fragment ownership, and every shared-memory
layout unchanged. There are no launch, host-allocation, or shared-allocation changes. The helper's private
volatile local argument exists only to prevent compiled common-subexpression
elimination. Each call initializes it synchronously before reading it; no other
thread supplies updates. Removing the helper also removes this compiler control.

Keep WGMMA completion waits before consuming scores or modifying accumulators.
The existing maximum barriers publish the peer maximum before its use. Preserve
probability stores, async-proxy fences, probability barriers, and free/ready
arrivals: scalar caching does not authorize early shared-buffer reuse. The
Group 0 factors must survive the probability exchange through their final sum
updates. Preserve all mask handling and index bounds checks.

Keep FP32 differences, `exp2f`, multiplication/FMA order, BF16 round-to-nearest
conversions, and the existing compiler flags. In particular, do not recompute
the difference after overwriting `m[row]`, change score exponentials, or
reassociate reductions. The volatile local roundtrip preserves the FP32 argument's bits.

## Example configuration

The supplied sparse MLA workload has 8192 tokens, 128 heads, query/key width
576, value width 512, and 2048 selected indices per token. The scale is
`0.1352337788608801f`; `kLog2E` is `1.4426950408889634f`. Inputs, stored
probabilities, and output are BF16; scores, maxima, sums, and accumulators are
FP32. Preserve the supplied oracle and numerical requirements.

Keep 64-element tiles and 32 selected-index tiles processed in 16 paired iterations.
The grid has 16384 CTAs, each owning one token and 64 heads. Each CTA has
384 threads: two 128-thread consumer groups and one 128-thread producer.
Each consumer thread has two interleaved rows, 32 score elements, and 128
output elements. In `softmax`, one factor serves 64 output elements and one
sum per row. At the Group 0 peer-maximum update, one factor serves 16
probability elements, 64 output elements, and one sum per row.

Retain 231376 shared-memory bytes per CTA, resident Q, both KV buffers,
probability storage aliasing, unswizzled WGMMA descriptors, asynchronous KV
copies, pipeline phases, all barriers, scalar shared reductions/statistics,
and per-element output reciprocal evaluation. Keep the SM90 WGMMA instructions,
register compilation settings, singleton cluster, host ABI, and caller stream.
These are replay settings, not prerequisites of scalar exponential reuse.

# Precondition

- Data types: all reused evaluations must receive identical arguments and use
  the same result representation and rounding semantics. Otherwise replacing
  several results with one can change the consuming arithmetic. This technique
  does not itself require BF16 inputs or a particular scalar precision.
- Layout: associate every consumer with its owning thread's correct row factor;
  a factor from another row changes its rescaling. No additional physical
  contiguity or alignment is required because this change moves no tensor data.
- Storage: discarded argument loads must not observe external updates or
  implement required communication. Caching would otherwise suppress required
  memory behavior. No particular memory level is required; the temporary
  argument in this implementation has no external producer or consumer.
- Pipeline: maximum producers must finish and make their values visible before
  the difference is formed. The cached factor must remain unchanged
  until all consumers finish; overwriting it early would rescale later
  consumers incorrectly. This transformation introduces or
  repurposes no buffer, so reader completion and buffer reuse gain no additional
  requirement. No particular stage count or new synchronization is required.
- Hardware: no additional device feature or fixed capacity requirement. Ordinary
  thread-local scalar storage holds the reused value; the exponential operation
  already exists in the preceding implementation.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These excerpts show the edited statements in their existing functions. Keep
intervening score conversion, publication, and synchronization code in place.
The first probability loop below is the Group 0 peer-maximum update, not the
score-exponential loop inside `softmax`.

## Before

```cuda
// solution/hopper.cuh: helper used at every rescaling consumer.
__device__ __forceinline__ float rescale_exp(float delta) {
    volatile float argument = delta;
    return exp2f(argument);
}

// softmax<Group>: inside the existing row loop.
const float delta = m[row] - next[row];
#pragma unroll
for (int i = row * 2; i < kOutputRegs; i += 4) {
    o[i] *= rescale_exp(delta);
    o[i + 1] *= rescale_exp(delta);
}
// Keep the existing score-exponential loop that computes sum here.
l[row] = l[row] * rescale_exp(delta) + sum;

// consume<Group>, Group 0: immediately after sync<Barrier::Max1>().
float next[2], delta[2];
load_stats(sm.maximum + lane / 4, next);
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
    delta[row] = m[row] - next[row];
    m[row] = next[row];
}
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
    for (int i = row * 2; i < kScoreRegs; i += 4) {
        s[i] = __float2bfloat16_rn(p[i] * rescale_exp(delta[row]));
        s[i + 1] = __float2bfloat16_rn(p[i + 1] * rescale_exp(delta[row]));
    }
}
// Keep save_prob, shared_fence, Prob0 arrival, and Prob1 synchronization here.
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
    for (int i = row * 2; i < kOutputRegs; i += 4) {
        o[i] *= rescale_exp(delta[row]);
        o[i + 1] *= rescale_exp(delta[row]);
    }
    l[row] *= rescale_exp(delta[row]);
}
```

## After

```cuda
// Delete rescale_exp from solution/hopper.cuh; retain output_scale.

// softmax<Group>: inside the same row loop.
const float scale = exp2f(m[row] - next[row]);
#pragma unroll
for (int i = row * 2; i < kOutputRegs; i += 4) {
    o[i] *= scale;
    o[i + 1] *= scale;
}
// Keep the existing score-exponential loop that computes sum here.
l[row] = l[row] * scale + sum;

// consume<Group>, Group 0: immediately after sync<Barrier::Max1>().
float next[2], scale[2];
load_stats(sm.maximum + lane / 4, next);
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
    scale[row] = exp2f(m[row] - next[row]);
    m[row] = next[row];
}
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
    for (int i = row * 2; i < kScoreRegs; i += 4) {
        s[i] = __float2bfloat16_rn(p[i] * scale[row]);
        s[i + 1] = __float2bfloat16_rn(p[i + 1] * scale[row]);
    }
}
// Keep save_prob, shared_fence, Prob0 arrival, and Prob1 synchronization here.
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row) {
#pragma unroll
    for (int i = row * 2; i < kOutputRegs; i += 4) {
        o[i] *= scale[row];
        o[i + 1] *= scale[row];
    }
    l[row] *= scale[row];
}
```
