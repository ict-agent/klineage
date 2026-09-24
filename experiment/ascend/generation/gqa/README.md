# GQA - AI-generated AscendC implementation

DeepSeek-V4.1-Flash / Codex App 分别运行 A（无专家材料）和 B（有专家材料）。两组从相同的已验证 AscendC 初始源码出发，使用相同题目、固定评测器、NPU 2 和 7200 秒预算。B 额外读取 [专家材料](with_memory/expert/source.json)。

## 题目

固定 workload 为 causal GQA，BF16 连续 BSHD：Q `[1,16384,32,128]`，K/V `[1,16384,8,128]`，每 4 个 Q head 对应 1 个 KV head，scale=`1/sqrt(128)`。输出与 Q 的 shape、dtype、device 相同。输入文件 SHA256：`8a741f0fee7b0cc579f1b990e5bb000e508ce372c03d335e8376b0cb845cb827`。

## 评测口径

- 正确性：完整输出与题目 PyTorch FP32 reference 比较，`torch.allclose(atol=rtol=0.01)`，同时核对 shape、dtype、device；仅通过固定 workload 的版本进入曲线。
- 计时：当前 NPU stream 的 Event interval，完整算子调用；3 轮，每轮预热 10 次、计时 50 次，先取每轮中位数，再取 3 轮中位数的中位数。编译、输入构造与正确性比较不计入。
- 环境：Ascend 910B1、CANN 9.1.0、torch 2.10.0+cpu、torch-npu 2.10.0.post4。

## 结果

| Setting | 固定校验 | version0 | 最佳版本 | vs. version0 |
| --- | ---: | ---: | ---: | ---: |
| A，无专家材料 | 27/36 通过 | 85.285 ms | version33，39.208 ms | 2.175× |
| B，有专家材料 | 32/39 通过 | 85.504 ms | version34，35.143 ms | 2.433× |

在这两次运行中，B 的最佳耗时比 A 低 4.065 ms，A/B = **1.116×**。

| Setting | 首次通过 | 首次达到最佳耗时 1.05 倍内 | 最佳版本 |
| --- | --- | --- | --- |
| A | 3.3 分钟 / 0.132M token / 85.285 ms | 49.8 分钟 / 9.291M token / 39.463 ms | 112.6 分钟 / 41.904M token / 39.208 ms |
| B | 2.9 分钟 / 0.166M token / 85.504 ms | 61.5 分钟 / 21.684M token / 36.083 ms | 106.3 分钟 / 38.826M token / 35.143 ms |

[时间曲线](plots/latency-vs-time.png)、[token 曲线](plots/latency-vs-tokens.png)、[逐次评测 CSV](plots/)由 `events.jsonl` 生成。时间从首个模型请求起算；token 为 API 记录的累计输入加输出 token，包含缓存输入。阶梯线表示截至该次评测最快的通过版本，散点表示各次通过版本的耗时。

## 产物与复现

`without_memory/`、`with_memory/` 各含 `events.jsonl`、`trace.jsonl`、`result.json`、`versions/version<N>/` 和最佳 `submission/`。B 的专家资料及来源位于 `with_memory/expert/`。

在本目录重绘结果：

```sh
python -m pip install matplotlib
python plots/plot.py
```
