---
skill_id: triangular-inverse.register-working-rows
intent: Keep private triangular-inverse working rows in registers across substitution
  steps.
preconditions:
- 'Data types: shared-to-register promotion must preserve each working element''s
  representation and arithmetic rounding; narrowing or reassociation would change
  the substitution.'
- 'Layout: each thread exclusively accesses its own fixed-size row, with indices resolvable
  at compilation; exclusive ownership permits privatization and constant indices permit
  register scalarization. No additional alignment or global-contiguity requirement
  applies.'
- 'Storage: working rows reside in shared memory without escaping aliases; cross-thread
  values pass through separate pivot storage that remains shared. Direct readers in
  other threads would lose access after promotion.'
- 'Pipeline: input-matrix production and row initialization finish before their reads;
  pivot publication precedes consumption, and consumers finish before pivot storage
  reuse. Preserve participation and barriers because registers provide no cross-thread
  visibility.'
- 'Hardware: sufficient thread registers for the live row and other simultaneous values,
  within thread and CTA allocation limits; spills would defeat register residency.
  No specialized matrix instruction is required by this promotion.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Promote each thread's triangular-inverse working row from shared memory to
registers. This removes shared loads and stores between substitution steps.

In `solution/prepare.cuh`, change only the working-row declaration in `invert`,
after `i` and `base`: replace `rows` and its volatile pointer with `float inv[8]`.
Remove the now-unused `kInverseCols` declaration and its shared-storage comment.
Keep all following accesses named `inv`, so initialization, substitution, and
final BF16 conversion stay unchanged. Removing `volatile` lets the existing
unrolled, constant-index accesses become register scalars.

Each participating lane owns one row. Other lanes receive its values only through
`s.pivot`, which must remain shared. Preserve the caller's barrier after producing
`s.l`, both warp barriers around every pivot exchange, and barriers publishing
the inverse to later consumers. The private row needs no additional barrier;
it is initialized before use and has no consumers after `invert` returns.

Retain the `tid >= kWarp` early return and every existing index and predicate.
Do not change launches, global layouts, dynamic shared structures, caller-stream
ordering, arithmetic expressions, FMA order, or rounding. The removed array is
static shared storage, so no host allocation or launch adjustment is needed.

Inspect generated code: row updates should use register operands; shared accesses
to `s.l`, `s.pivot`, and final matrices remain. Check that the promoted row does
not spill to local memory. Use the existing Kernel evaluator with the complete
problem for correctness and timing; retain its numerical and timing policies.

## Example configuration

The workload is B=1, T=4096, H=96, D=128, with 16-token chunks. Preserve BF16
operands/intermediates and FP32 inverse-row arithmetic, the BF16 conversion points,
and the later BF16 block merge with FP32 MMA accumulators.

`prepare` launches `grid=(256,96)`, 256 threads, and 41856 dynamic shared bytes,
with `__launch_bounds__(kPrepareThreads,8)`. Only its first 32 lanes enter `invert`.
Lane `tid` uses `i=tid&7`, `base=tid&8`, and an eight-element row initialized from
row-major `s.l[(base+i)*kChunk+base+p]`. Four groups of eight lanes retain their
existing output duties; the two 8x8 diagonal substitutions and subsequent merge
remain intact. Preserve the seven substitution steps, increasing `p` traversal,
`fmaf(scale,pivot,inv[p])`, and every `#pragma unroll`.

Before promotion, `rows[32][8]` occupies 1024 static shared bytes with a 32-byte
row stride. Promotion removes that allocation while preserving the separate
pivot and transpose scratch, shared layouts, and all `sizeof` assertions.
The existing compile flags, fast-math setting, and launch bounds remain unchanged.
Keep tensor-core products and their fragment adapters, cooperative staging,
normalization reduction grouping, and the 256 ordered recurrence launches.
These are replay settings and retained mechanisms, not promotion prerequisites.

# Precondition

- Data types: register storage must preserve the working elements' representation
  and arithmetic rounding. Moving storage does not justify narrowing elements or
  reassociating substitution updates; either could change the result.
- Layout: every row belongs exclusively to one thread, and its fixed extent and
  indices must permit compile-time scalarization. Cross-thread row accesses would
  break privatization; unresolved indexing can leave the array in local memory.
  No additional alignment or global-contiguity requirement applies because this
  change introduces no vector transfer or global addressing.
- Storage: the shared working rows have no escaping aliases. Cross-thread values
  already travel through separate pivot storage, which remains shared. Otherwise
  replacing rows with private registers would strand their readers.
- Pipeline: finish input-matrix production and initialize each row before reading
  it. Retain pivot publication before consumption and completion of all readers
  before pivot scratch reuse, with the same participation and barriers. Register
  promotion supplies no cross-thread visibility and cannot replace those barriers.
- Hardware: the row and other simultaneously live values must fit the available
  register budget (`R_row + R_other <= R_available`) under thread and CTA allocation
  limits. Otherwise spills defeat register residency. No specialized matrix
  instruction is needed for the storage promotion itself.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace this declaration block inside `invert`; retain the following loops and
all other function code verbatim.

## Before

```cuda
    int i = tid & 7;
    int base = tid & 8;
    constexpr int kInverseCols = kChunk / 2;
    __shared__ volatile float rows[kWarp][kInverseCols];
    // Keep private working rows in shared memory throughout substitution.
    volatile float* inv = rows[tid];
```

## After

```cuda
    int i = tid & 7;
    int base = tid & 8;
    float inv[8];
```
