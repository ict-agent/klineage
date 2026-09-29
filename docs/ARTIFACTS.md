# Where the paper's results live

The paper is in [`paper/`](../paper). Its measurements are spread across branches,
because each operator study was run and published on its own. This page maps every
reported number to the branch and directory that holds it.

Clone once and switch branches; nothing here needs a second checkout.

```bash
git fetch origin
git switch ascend-generation    # or any branch below
```

## Branch map

| Branch | Holds | Paper |
| --- | --- | --- |
| `master` | Framework, CLI, problem definitions, CUDA skill cards, CUDA transfer runs | §3, Fig. 4 (top) |
| `ascend-generation` | Five AscendC transfer targets, with and without memory | Fig. 4 (bottom), Tab. 4 |
| `ascend-gemm-optimization` | GEMM expert kernel on Ascend | Tab. 3 |
| `feat/conv2d-tle-kernel` | Conv2d expert kernel on Ascend | Tab. 3 |
| `fmha-artifact` | FMHA expert kernel on Ascend | Tab. 3 |
| `ascend-gdn-optimization` | GDN expert kernel on Ascend | Tab. 3 |
| `ascend-topk-optimization` | Top-K expert kernel on Ascend | Tab. 3 |
| `archive/kda-setup-2026-09-06` | Stopped KDA setup, kept for provenance | — |

## Table 3: AscendC kernels against the vendor reference

One branch per operator. Each report states the operator contract, the baseline it
was measured against, the timing policy, and the raw samples behind the median.

| Operator | Branch | Report | Baseline | Speedup |
| --- | --- | --- | --- | ---: |
| GEMM | `ascend-gemm-optimization` | `experiment/ascend/optimization/gemm/README.md` | torch-npu `matmul` | 1.048× |
| Conv2d | `feat/conv2d-tle-kernel` | `experiment/ascend/optimization/conv2d/README.md` | CANN `aclnnConvolution` | 1.496× |
| FMHA | `fmha-artifact` | `experiment/ascend/optimization/fmha/README.md` | `npu_fused_infer_attention_score` | 1.155× |
| GDN | `ascend-gdn-optimization` | `experiment/ascend/optimization/README.md` | vLLM-Ascend Triton chain | 1.404× |
| Top-K | `ascend-topk-optimization` | `experiment/ascend/optimization/topk/README.md` | torch-npu `topk` | 1.281× |

Baselines differ per operator because each one has a different vendor path. Read the
report before comparing latencies across operators: devices (910B1 and 910B3),
CANN versions, and timing scopes are not uniform.

## Figure 4 and Table 4: transfer to new operators

`ascend-generation`, under `experiment/ascend/generation/`. Setting A
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

On `master`:

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

`master` is the only branch you need in order to run the framework; the others are
result artifacts. Build and CLI instructions are in the [top-level README](../README.md),
and the [workflow contract](../src/klineage/message/workflow.md) defines the artifacts,
stop rules, and failure handling that every run follows.
