# Ascend 910B1 上的 KDA 设计要点

## 1. 并行度盘点（B=1, T=4096, H=96, D=128, BT=64）

```text
intra-chunk 任务： 96 head x 64 chunk = 6144 个独立任务（可完全并行）
state 递推任务   ： 96 head x (V/BV)  ，每个任务串行 64 个 chunk
开核规模        ： 本机 24 AIC（GDN 实测用满 24 核）；grid 固定在这个量级 + 核内 stride 循环
```

结论：**第一阶段（Prepare/PostWU）并行度绰绰有余；瓶颈在第二阶段（FwdH）的串行链**。
优化优先级：先把 FwdH 的串行链做短（状态常驻、满 tile、少同步），再考虑第一阶段。

## 2. GM 流量估算（决定融合还是拆 kernel）

| 方案 | GM 流量 | 量级估算（用实测定标，别当结论） |
|---|---|---|
| 两 kernel：w/u/qg/kg 经 GM 中转（各 100.7 MB bf16） | 写 ~503 MB + 读 ~503 MB ≈ **1.0 GB** | ~0.85 ms（按 1.2 TB/s 估） |
| 单 kernel：chunk 内算完即用，只写 output | 读 4x100.7 MB + 写 100.7 MB ≈ **0.5 GB** | ~0.42 ms（按 1.2 TB/s 估） |

- 拆 kernel 换来 6144 路并行，但每个中间量都是「整张 [4096, 96, 128]」级别的 GM 往返；
  融合后并行度仍有 96~192 路（够 24 核），且省掉约一半流量。**先试融合，再按 profiling 决定**。
- 无论哪种方案，`Aqk/Akk` 只在 chunk 内用（[64,64]），不要落 GM。
- 输入 q/k/v/g 各 100.7 MB 的读无法避免；`output` 100.7 MB 的写无法避免。

## 3. 状态常驻：h 绝不能每 chunk 从 GM 读写

- 每 head 状态 `[K, V]` fp32 = 128x128x4 = **64 KB**；若每 chunk 从 GM 读+写：
  `64 KB x 64 chunk x 96 head x 2 ≈ 786 MB` 额外流量（≈0.65 ms），直接吃掉全部收益。
- 做法：按 V 切块（BV=64 -> 32 KB）或按 K 切块（FlagGems 用 4 个 `[64, BV]` fp32 常驻），
  在 chunk 循环外加载一次 initial_state，循环内只更新片内副本，循环结束后写 final_state。
- BV=64 时每 head 2 个 worker，合计 192 个 state 任务；每 worker 顺序处理 32 个 chunk。

## 4. Cube/Vector 分工与 tile

- Cube（`tl.dot`）：`Aqk/Akk`（[BT,K]x[K,BT]）、`u/w` 的 `A @ X`（[BT,BT]x[BT,K]）、
  `w @ h`、`qg @ h`、`kg^T @ v_new`。这些都能落到满 tile：K=128、V=64/128、BT=64。
- Vector：L2norm、gate 激活与 `cumsum`、`2^gk`/`2^-gk`、掩码、逐元素 decay、最终 cast。
- 实测经验（同仓 GDN lineage）：**整 K/整 V 的满 tile 优于源码层拆成两个 K64 dot 再累加**
  （`1.057x-1.076x`，见 `~/codex/triton-tle/optimization_tips/cards/compute/cube/gdn-full-kv-cube-tile.md`）。
- L0 形态参考本仓 GEMM 实测：`128x256x64`（L1 `128x256x256`）。
- 对齐：UB 张量起始地址 32B 对齐；热点 GM tile 按 512B 对齐设计；尾轴 128 个 bf16 = 256B，天然对齐。
- T=4096 是 64 的整数倍：快路径可以完全去掉 chunk 尾块掩码（保留通用路径即可）。

## 5. MIX kernel 的流水与同步（来自本仓 GDN 正式实现）

- 单 MIX kernel 内用显式 `al.sync_block_set/wait(role, pipe)` 组织 AIC/AIV 生产-消费；
  两槽 workspace 交替，消费 task i 时预生产 i + num_workers。
- 实测收益 `1.365x-1.550x`（纯 device 单变量，`initial_state=True`）。启动配置：

```python
num_warps=4, num_stages=2, enable_sync_block_lock=True,
disable_auto_inject_block_sync=True, multibuffer=False
```

- 风险：set/wait 不是计数型 join，错 flag 会死锁或静默错；只有把跨 pipe 依赖枚举清楚才手排。
  先让编译器自动同步跑通正确性，再做显式流水。
- 固定 worker + stride loop（`for task in range(worker_idx, total, num_workers)`），
  workspace 按 worker 分配，避免随 chunk 数增长。

## 6. 有明确负向证据的做法（不要重复踩）

- 只提高 L2 命中率、不减少 GM 流量的改动：`-0.24% ~ +0.31%`。
- 只做 kernel 融合、没有改善常驻/并行：负收益（barrier 与统一 grid 抵消 launch 收益）。
- persistent worker 在短任务（T=1024）持平，长任务才有 1.04x-1.10x；本算子 T=4096 属于长任务。
- 不要为「看起来对称」把可常驻的小矩阵也切开重复搬运（L1 容量不足时让较小的可复用矩阵常驻）。

## 7. 优化顺序建议

1. 逐 token golden 对齐的小 shape 正确性（BT=16, T=64）。
2. 融合版单 kernel 跑通：per (head, BV) worker，串行 64 chunk，状态常驻。
3. 满 tile（K=128/V=64、BT=64），去掉尾部掩码。
4. 再上显式 AIC/AIV 两槽流水；每次只改一个变量并记录 latency。
