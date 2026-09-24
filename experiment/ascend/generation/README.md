# Ascend generation: Task 2 A/B results

This directory reports five fixed-workload AscendC operator studies. Setting A
(`without_memory`) receives no pre-injected expert pack; setting B
(`with_memory`) receives operator-specific expert material. The reported
latencies are the best versions that passed each operator's fixed correctness
gate. Both settings had a nominal 7,200-second budget, but run protocols and
effective iteration time differ; see the operator reports.

## Results

| Operator | Setting | Correct | Best latency (ms) | Control | Speedup |
| --- | --- | --- | ---: | --- | ---: |
| [KDA](kda/README.md) | A | Yes | 44.894 | PyTorch reference | 39.19x |
| [KDA](kda/README.md) | B | Yes | 31.564 | PyTorch reference | 59.16x |
| [SparseAttention](sparse_attention/README.md) | A | Yes, initial version | 5648.614 | Initial AscendC version | 1.00x |
| [SparseAttention](sparse_attention/README.md) | B | Yes | 99.356 | Initial AscendC version | 56.86x |
| [Top-P](top_p/README.md) | A | Yes | 0.3428 | PyTorch reference | 13.21x |
| [Top-P](top_p/README.md) | B | Yes | 0.2481 | PyTorch reference | 18.51x |
| [FusedAddRmsNorm](fused_add_rmsnorm/README.md) | A | Yes | 0.565930 | PyTorch reference | 10.62x |
| [FusedAddRmsNorm](fused_add_rmsnorm/README.md) | B | Yes | 0.457040 | PyTorch reference | 13.17x |
| [GQA](gqa/README.md) | A | Yes | 39.208 | Initial AscendC version | 2.175x |
| [GQA](gqa/README.md) | B | Yes | 35.143 | Initial AscendC version | 2.433x |

Speedup uses each setting's own measured control, not a common denominator
across operators or devices. SparseAttention A retained its correct initial
version: its 1.00x is **not** a successful optimization. Against the PyTorch
reference, SparseAttention A and B measure 0.0832x and 4.6938x, respectively.
The PyTorch references are correctness controls, not claimed optimized
operator baselines.

## Operator reports

| Operator | Fixed workload | Report |
| --- | --- | --- |
| KDA | 4,096 tokens, 96 heads, 128-wide state | [Result, trace, and curves](kda/README.md) |
| SparseAttention | 8,192 queries, 128 heads, 2,048 sparse indices | [Result, audit, and curves](sparse_attention/README.md) |
| Top-P | 32 rows, 128,256 vocabulary entries | [Result, trace, and curves](top_p/README.md) |
| FusedAddRmsNorm | 8,192 rows, 7,168 hidden units | [Result, audit, and curves](fused_add_rmsnorm/README.md) |
| GQA | 16,384 tokens, 32 query and 8 KV heads | [Result, trace, and curves](gqa/README.md) |

The per-operator reports give tensor contracts, devices, timing policies,
first-pass and best-version milestones, plots, and reproduction steps. The
[AscendC technique matrices](optimization-techniques/README.md) compare the
selected A/B source for all five operators; they do not establish which
technique caused a speedup.

## Reading the comparison

- B has a lower best latency in each published pair, but this is five
  observations, not a controlled estimate of the effect of expert knowledge.
- KDA and FusedAddRmsNorm B reached a passing version earlier. Top-P B
  passed later than A but reached its near-best latency earlier; GQA A reached
  its near-best latency earlier despite B's faster final version.
- SparseAttention B improved its shared initial AscendC version substantially;
  A found no correct faster replacement.
- KDA and SparseAttention/FusedAddRmsNorm used Ascend 910B3; Top-P and GQA
  used 910B1. KDA also used a shorter NPU-event timing policy than the other
  reports. Compare settings within each operator's documented protocol, not
  raw latencies across operators.
- SparseAttention had unequal development feedback and scope corrections;
  FusedAddRmsNorm had a mid-run protocol correction. Both reports disclose
  these deviations. Their A/B gaps cannot be attributed solely to the
  injected material.

## Evidence and reproduction

Each operator directory contains its selected `submission/`, version
snapshots, result metadata, and plots. Availability of `events.jsonl` and
`trace.jsonl` varies by publication: some exports are redacted, and private
raw sessions are not included. Consult each operator's artifact inventory and
audit before using token counts or trace fields. Tokens include cached input
and are not generated-token counts or cost.

For new runs, follow [NEXT_RUNS.md](NEXT_RUNS.md); it supersedes the historical
defaults in [AGENTS.md](AGENTS.md). For published results, use the reproduction
instructions in the corresponding operator report. Neither this overview nor
the technique matrices rerun the experiments.
