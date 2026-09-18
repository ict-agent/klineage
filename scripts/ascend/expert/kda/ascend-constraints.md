# 910B1 平台约束与启发式（triton-ascend 视角）

来源（本机）：
`~/codex/Ascend/ascend-competition-harness/optimization_constraints/`、
`~/codex/Ascend/ascend-competition-harness/optimization_tips/`、
`~/codex/triton-tle/optimization_tips/`。

只保留与框架无关、在 A2（910B1）上仍成立的条目；AscendC 专有 API（`TQue`/`TPipe`/
`DataCopyPad` 重载、`K_MAX_SHAPE_DIM`、`KERNEL_TASK_TYPE_*`）不搬。
原始卡片里的实测数据多来自 910B3，本机是 910B1：**结论方向可借，数字必须复测**。

## 1 硬约束（违反即错或直接退化）

```text
对齐      片上 buffer 起始地址 32B 对齐；GM 搬运按 32B 粒度处理
          热点尾轴按 32B 对齐设计 -> KDA 的 D=128 bf16 = 256B，天然满足
          [T, H, K] 连续布局下 head 维步长 256B，别引入更窄的热点行
Scatter   A2/A3 无 Direct Scatter；离散写/gather 退化为标量（triton 侧同样成立）
同地址     多核同时访问同一 512B 段会被串行化，核越多越慢
          对策：核内起始 chunk 轮转 (pid + k) % numChunks，别让所有核同步从头扫
```

## 2 性能启发式（可用实验证伪，不是禁令）

```text
搬运   单次搬运 >= 16KB 更接近带宽上限；用 stride 描述规则搬运，不要循环发小 copy
Tile   热点主 tile 按 512B 对齐设计；不要为此牺牲 UB 容量或核间均衡
L1     左右矩阵不能同时常驻时，让更小且可复用的那个常驻 L1，另一个流式搬
L2     工作集超 L2 时按波次切分，让单波尽量驻留；只读一次的流数据可试 CACHE_MODE_DISABLE
       （必须 A/B，禁用 L2 在跨核/跨轮复用时反而变慢）
归约   连续 inner axis 在单核单 tile 内做完；outer/跨核 axis 用分段 merge
核数   微秒级小算子减少启动核数、加大单核工作量；纯 Vector 路径别带 Cube 核头开销
       910B3 实测：启动核数 != 工作核数，两者要分开调
布局   先转置/重排成连续布局，再复用高性能内核；小 shape 摊不薄转换成本
专门化  分支/资源差异留到 host 或 tiling；热点循环只保留一条路径
       禁止按 case ID、固定 shape 或只对某个测试点成立的条件做特化
```

## 3 落到 KDA 的检查表

- `q/k/v/g` 每个 100.7 MB：热点搬运按 256B 行宽、整 chunk（`64x128` bf16 = 16 KB）设计。
- `Aqk/Akk/A` 在 chunk 内是下三角，别做全矩阵搬运；mask 用编译期常量表达
  （`T % 64 == 0`，无尾块时不要运行时 mask）。
- 状态 `h` 每个 head 64 KB：分核按 `(head, V 块)`；同一 head 的 chunk 序列串行。
- `g`/`a_log`/`dt_bias` 是逐 key 维广播：先在 UB 里算好 `2^gk` 再进 dot，
  不要把广播展开进 Cube 输入。
- 反例优先读 `triton-tle/optimization_tips/cards/rejected/`：
  `l2-hit-without-gm-traffic-reduction`、`fusion-without-residency-or-parallelism`。
