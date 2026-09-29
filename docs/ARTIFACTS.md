# Where the paper's results live

The paper is [`KLineage.pdf`](../KLineage.pdf). Most of its measurements are on
`master`; the transfer study is on its own branch. This page maps every reported
number to the directory that holds it.

## Table 3: AscendC kernels against the vendor reference

Each report states the operator contract, the baseline it was measured against, the
timing policy, and the raw samples behind the median.

| Operator | Report | Baseline | Speedup |
| --- | --- | --- | ---: |
| GEMM | `experiment/ascend/optimization/gemm/README.md` | torch-npu `matmul` | 1.048× |
| FMHA | `experiment/ascend/optimization/fmha/README.md` | `npu_fused_infer_attention_score` | 1.155× |
| GDN | `experiment/ascend/optimization/README.md` | vLLM-Ascend Triton chain | 1.404× |
| Top-K | `experiment/ascend/optimization/topk/README.md` | torch-npu `topk` | 1.281× |
| Conv2d | `experiment/ascend/optimization/conv2d/README.md` | CANN `aclnnConvolution` | 1.496× |

Conv2d is still on the `feat/conv2d-tle-kernel` branch; the other four are merged.

Baselines differ per operator because each one has a different vendor path. Read the
report before comparing latencies across operators: devices (910B1 and 910B3),
CANN versions, and timing scopes are not uniform.

## Figure 4 and Table 4: transfer to new operators

On the `ascend-generation` branch, under `experiment/ascend/generation/`. Setting A
(`without_memory`) gets no expert material; setting B (`with_memory`) gets the
induced skills. Both had a 7,200-second budget.

| Operator | A best (ms) | B best (ms) | Control |
| --- | ---: | ---: | --- |
| KDA | 44.894 | 31.564 | PyTorch reference |
| SparseAttention | 5648.614 | 99.356 | initial AscendC version |
| Top-P | 0.3428 | 0.2481 | PyTorch reference |
| FusedAddRmsNorm | 0.565930 | 0.457040 | PyTorch reference |
| GQA | 39.208 | 35.143 | initial AscendC version |

SparseAttention A kept the correct starting version and never replaced it, which is
why the paper marks it a failure rather than a 1.00× result.

Each operator directory carries its `submission/`, version snapshots, result
metadata, and plots. `experiment/ascend/generation/optimization-techniques/` holds
the technique audit behind Table 4. Trace availability varies by operator; the
per-operator report says what was exported and what was redacted.

## CUDA side

- `skillcards/` — 141 `SKILL.md` cards, every one scoped `nvidia-sm90a-cuda13`.
  Grouped by source family: fmha 42, kda 31, cuda 29, sparse-mla 17, topk 11,
  rmsnorm 5, and six singletons. A card records `skill_id`, `intent`,
  `preconditions` over data types, layout, storage, pipeline and hardware, and
  `scope` over cases, languages and platforms.
- `experiment/transfer_*/` — timestamped transfer runs, one directory per run,
  then per operator, then `with_memory/` and `without_memory/`.
- `experiment/_transfer/` — the consolidated view across runs.
- `problems/` — the five problem definitions in FlashInfer Trace format, with
  workloads and fixed input samples. See [`problems/README.md`](../problems/README.md).

## Reproduction

`master` carries the framework and all but two of the result artifacts. Build and
CLI instructions are in the [top-level README](../README.md),
and the [workflow contract](../src/klineage/message/workflow.md) defines the artifacts,
stop rules, and failure handling that every run follows.
