# Ascend generation — expert-knowledge ablation (task 2)

## SparseAttention / FusedAddRmsNorm 正式结果

两个算子的正式结果分别位于 [SparseAttention](sparse_attention/README.md) 和 [FusedAddRmsNorm](fused_add_rmsnorm/README.md)：AscendC，DeepSeek-V4.1-Flash + Codex，每组7200秒。

| Kernel | Setting | Correct | 最佳延迟(ms) | vs. PyTorch reference |
|---|---|---|---:|---:|
| SparseAttention | A | ✓（保留初始实现） | 5648.614014 | 0.0832× |
| SparseAttention | B | ✓ | 99.355560 | 4.6938× |
| FusedAddRmsNorm | A | ✓ | 0.565930 | 10.6181× |
| FusedAddRmsNorm | B | ✓ | 0.457040 | 13.1701× |

SparseAttention两组初始源码相同，相对本组初始实现分别为1.0000×与56.8563×。
定义、复现、A/B曲线、优化技巧及汇总数据均位于对应算子目录。
公开包包含源码、版本快照和脱敏元数据；原始会话与私有资料仅本地保留。不同开发反馈条件及无效resume在报告中披露。
本节替换这两个算子的旧结果；以下KDA/Top-P记录保留原文。

Two kernels, `kda` and `top_p`, each generated twice by one Codex session:
`without_memory` (bare prompt) and `with_memory` (same prompt plus an expert
pack the agent may read). Same gate, same 2 h budget, same delivery rules; the
expert material is the only difference, and `expert/source.json` names where
each pack came from.

## What the curves have to show

The x axis is the session's wall clock (minutes) or the tokens it has spent so
far; the y axis is the gated latency of the version the session holds at that
moment, or its speedup over the torch-npu baseline. The baseline is a single
flat line — the torch reference never improves — so every point on our curve is
a comparison against it. `plots/latency-vs-tokens.png` draws both settings of
both kernels on those axes.

Two readings carry the claim this study is about:

- **Far above the baseline.** Each setting leaves the baseline line at its
  first passing version, 6.5× to 40× ahead, and ends 13× to 75× ahead: the
  reference is never competitive at any point of the session, not just at the
  end.
- **Peak before the budget.** The plateau — the first gate within 5 % of that
  setting's final best — arrives at minute 76–84 of the 120 min budget, so the
  last 15–27 min of a session buy ~5 % at most: the gap to the reference is
  clear in the first half of a session, the peak in the second.

## Results

| Kernel | Setting | Correct | Latency (us) | vs. Baseline |
| --- | --- | --- | --- | --- |
| KDA | Without Expert Knowledge | ✓ | 22312.0 us | 74.72× |
| KDA | With Expert Knowledge | ✓ | 23075.3 us | 66.44× |
| TOP_P | With Expert Knowledge | ✓ | 248.1 us | 18.51× |
| TOP_P | Without Expert Knowledge | ✓ | 342.8 us | 13.21× |

Best gated sample per session; the top_p with_memory artifact measured 0.2541 ms
(18.08×) on its own last gate call, 0.2481 ms at best.

## Trajectory

Time and tokens are clocked from the session's first API response. `peak` is
the first gate within 5 % of the best; `best` is the final artifact's own
measurement.

| Kernel | Setting | First passing gate | Peak | Best gate | Gate calls (rejected) |
| --- | --- | --- | --- | --- | --- |
| KDA | Without Expert Knowledge | 36.6 min · 11.50 M · 38.70× | 75.9 min · 19.58 M | 95.8 min · 25.46 M | 10 (1) |
| KDA | With Expert Knowledge | 5.1 min · 2.45 M · 40.22× | 75.9 min · 15.79 M | 102.1 min · 20.30 M | 35 (7) |
| TOP_P | With Expert Knowledge | 29.6 min · 7.48 M · 9.18× | 78.0 min · 17.05 M | 87.8 min · 19.61 M | 14 (2) |
| TOP_P | Without Expert Knowledge | 21.6 min · 6.59 M · 6.51× | 83.5 min · 16.26 M | 100.3 min · 18.89 M | 8 (1) |

Sessions spent 19–26 M tokens and stopped at 96.9–103.3 min, i.e. before the
2 h kill; the gate rejected 1–7 candidates per session, and those calls consume
tokens without moving the curve.

### Reading the two settings against each other

At equal tokens the comparison is:

- **top_p** — the effect the task predicts. with_memory passes the gate later
  (29.6 vs 21.6 min) but 1.4× faster through it (9.18× vs 6.51×), overtakes
  without_memory at ~58 min and stays ahead at every later checkpoint: ~16.5 M
  tokens 0.2663 vs 0.3474 ms (1.30×), ~19 M tokens 0.2481 vs 0.3428 ms (1.38×).
  It also peaks 5.5 min earlier (78.0 vs 83.5 min).
- **kda** — mixed. with_memory is much earlier to a passing version (5.1 vs
  36.6 min; 2.45 vs 11.50 M tokens) and at ~11.7 M tokens measures 24.51 ms
  (62.6×) where without_memory has just landed its first pass at 43.07 ms
  (38.7×). At ~19.6 M tokens the two are level (23.08 vs 23.36 ms), and
  without_memory ends 3.4 % ahead (22.31 ms / 74.7× vs 23.08 ms / 66.4×): the
  expert pack sped the search up but did not raise its ceiling here.

## Artifacts

```text
generation/
|- README.md       this file
|- plots/          table.md (the four rows above) + csv per setting + curves
|- kda/            per-kernel tree: one directory per setting
`- top_p/
```

Each `kda/` and `top_p/` tree holds `README.md` (problem, protocol, trajectory),
one directory per setting with `events.jsonl`, `trace.jsonl`, `baseline.json`,
`versions/version<N>/` and `submission/`, plus `plots/`.

## Reproducing

```sh
scripts/ascend/run.sh start --kernels kda --settings without_memory,with_memory --devices 2,3
scripts/ascend/collect.py --kernel kda     # run root -> experiment/ascend/generation
scripts/ascend/plot.py --root <run root> --x tokens --y latency
```

## Deviations and caveats

- The two settings of a kernel ran on different NPUs (without_memory on device
  2, with_memory on device 3), so each is scored against the baseline measured
  on its own device: KDA 1667.08 vs 1533.14 ms, top_p 4.5267 vs 4.5922 ms.
  Speedups are comparable, absolute latencies are not.
- KDA is pinned to triton-ascend (`language = "python"`), top_p to AscendC
  (`language = "ascendc"`). The task leaves the language open; the pin keeps
  the two settings of a kernel comparable, and the gate rejects torch-only
  bundles.
- Expert packs: KDA reads vllm-ascend's triton KDA (`vllm_ascend/ops/triton/kda`,
  v0.23.0); top_p reads CANN `ops-nn`'s AscendC sources for
  `apply_top_k_top_p_with_sorted` and `top_k_top_p_sample` (606a4d5d5).
- The baseline column is the definition's torch reference on the NPU, as the
  task's table asks — deliberately not a fused CANN operator.
- No data ships with either problem, so the workload's `random` entries are
  filled locally: for top_p `probs` is the softmax of seed-0 logits, because
  noise would break the sum-to-one the reference relies on; kda keeps `scale`
  and `lower_bound` in a small safetensors file. Shapes and references are the
  definition's.
- top_p's gate is stricter than `allclose`: the support must match the
  reference element for element, with no tolerance, and rows must sum to 1.
- The kda without_memory unit was resumed once by the batch after its first
  session ended, which is why its curve continues past minute 85.
