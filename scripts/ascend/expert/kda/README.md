# KDA expert knowledge (Ascend 910B1, triton-ascend)

目标：用 triton-ascend 在 910B1 上实现并优化 KDA prefill（B=1, T=4096, H=96,
K=V=128, bf16 输入, fp32 状态）。本目录是 Setting B 的专家知识，与 Setting A 共用
同一评测入口；这里只放实现策略，不放评测接口。

阅读顺序：

1. `kda-math.md`：算子语义 + chunked 等价形式（这是并行的唯一来源）。
2. `ascend-kda-sota.md`：已开源的 Ascend KDA 实现及其分解方式
   （本地 CANN `ops-transformer` / vllm-ascend / FlagGems / FLA）。
3. `ascend-kernel-design.md`：Ascend 上的分核、tile、状态常驻、GM 流量估算与实测经验。
4. `ascend-constraints.md`：910B1 硬约束与性能启发式（取自本地优化库，已按框架过滤）。
5. `ascend-triton-gdn.md`：本地 GDN 融合 Triton 实现的结构、实测与负面结论（FwdH 同构）。
6. `triton-ascend.md`：triton-ascend 编程模型、autotune、编译选项、调试与常见退化。

规模事实（用于估算）：

```text
q/k/v/g  : [1, 4096, 96, 128] bf16     每个 ~100.7 MB
beta     : [1, 4096, 96] fp32
state h  : [1, 96, 128, 128] fp32      每个 head 64 KB
chunk    : 64 -> 每 head 64 个 chunk
```

先做正确性，再谈性能：任何 tile/流水改动都保留可回退的版本。
