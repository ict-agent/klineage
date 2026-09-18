# 开源 Ascend KDA 实现（可直接借鉴的分解）

## 1. 本地 CANN `ops-transformer`（AscendC，官方，离线可读）

路径：`/Users/rudy/codex/Ascend/references/ops-transformer/attention/`

- `chunk_kda_fwd/`：KDA prefill 官方实现，与下面 vllm-ascend 同源同接口族。
  文档 `docs/ChunkKdaFwd算子设计介绍.md`（阶段 / 模板 / 流水）、`README.md`（公式 / 属性 / dtype）。
- `recurrent_kda/`：KDA decode/MTP 递推版；`kda_input_proj/`：qkv+beta/gate/g 前处理（A5 only）。
- `chunk_gated_delta_rule/`、`recurrent_gated_delta_rule/`：GDN 家族；前者三 stage 拆分
  （chunk 内准备 / 状态递推 / 带 mask 的 QK^T 收尾），tiling key 分 `WITH_GAMMA`（逐通道 gate）
  与 state dtype，AIC/AIV 分工 + 双缓冲 + 核内 chunk 并行的写法可直接对照。

设计文档里对 triton 实现有效的结论：

```text
编译期模板   key=1 通用  |  key=2 = chunk64/K128/V128（本算子正是 key=2）
快路径条件   BF16 + 非 varlen + T % 64 == 0 -> dense 模板；否则 tail/varlen 通用路径
Gate 变体    safe_gate=true  -> lower_bound(-5) * sigmoid(exp(A_log) * (g + dt_bias))
             safe_gate=false -> -exp(A_log) * softplus(g + dt_bias)      # 本算子只用前者
数值          L=(I+Akk) 前的 L、求逆、状态传播全 FP32；公开低精度 h 不回灌状态
状态布局      内部统一 [K, V]，只在公开边界按 state_v_first 转置
流水         右矩阵常驻 L1；AIC L1/L0 双缓冲；AIV 输入 staging ping-pong
性能目标      A5 / H96 / T16K / chunk64 / BF16：10 ms 目标，11 ms 门禁（A2 上不适用）
```

A5(arch35) 的特化（regbase、head-pair、sub-16 尾块模板）在 910B1(A2) 上不存在；
A2 走 key2 通用实现，仍用编译期常量。把上面的阶段划分与数值约定搬过来即可，模板细节别照抄。

## 2. vllm-ascend `ChunkKdaFwd`（AscendC，同源，仓库在线）

- 仓库：<https://github.com/vllm-project/vllm-ascend> → `csrc/attention/chunk_kda_fwd/`
  （同目录 `README.md`、`docs/design.md`、`docs/api.md`；另有 `csrc/attention/kda_gate_cumsum/`）
- 顶层语义对齐 FLA `chunk_kda_fwd`（commit `0f0f0c97af39343855b43bbbaddcedfda5cb9d77`）。
- 属性：`chunk_size=64`（支持 128）、`layout=BSND/BNSD/TND/NTD`、`state_v_first`、
  `safe_gate`（= 本算子的 `-5 * sigmoid(exp(A_log) * (g + dt_bias))`）、`use_gate_in_kernel`。

阶段职责（每个阶段都是独立可复用的 kernel 边界）：

```text
GateCumsum : gk = cumsum(gate) / ln2          # chunk-local，fp32，BNSD
Prepare    : Aqk, Akk, qg, qg_scaled, w_seed, u_seed   # 三角求逆 fp32 累加
Post-WU    : w, u, kg, v_new_seed              # Akk head 循环按 H_v（GQA 时避免漏算）
FwdH       : v_new = u - w @ h_prev
             h_next = 2^gk_last * h_prev + kg^T @ v_new     # 与 GDN 的 FwdH 同一实现
Finalize   : attn_out = qg_scaled @ h + Aqk @ v_new
```

设计要点（design.md 摘录）：

- 内部递推状态固定 `[..., K, V]`，`state_v_first` 只在进出边界转置；公开 `h` 为 sequence-major。
- 单 launch 时 Prepare→Finalize 在同一 L0 kernel 内完成，中间量走 workspace；
  多 chunk 场景才拆四段 launch（阶段间用物理边界重置事件状态）。
- tiling key 只按 `chunk`、`K`、`V` 选择：key=2 = `chunk=64, K=V=128`（dense/tail/varlen）。
- 性能设计：Prepare 的右矩阵常驻 L1（避免 K/K^T 重复搬运与转置）；AIC 用 L1/L0 双缓冲组织
  MTE2/MTE1/Cube/Fixpipe；AIV 用输入 staging ping-pong 让下一 tile 的 MTE2 与当前 VEC 重叠；
  数值主计算保持 FP32；性能结论只用 `msopprof`。
- 适用平台 A2/A3/A5；dtype FP16/BF16；K=128，V=128/256。

对 triton-ascend 的启示：**四段分工照搬，融合与否按 GM 流量决定**（见 `ascend-kernel-design.md`）。

## 3. FlagGems-vllm `chunk_kda.py`（Triton，算法分解最贴近你要写的代码）

- 本地：`~/codex/triton-tle/FlagGems-vllm/src/flaggems_vllm/ops/FLA/chunk_kda.py`
  （origin `gitcode.com/Triton-TLE/FlagGems-vllm`；910b1 上另有
  `~/FlagGems-vllm-pr1065-autosync-20260909` 同文件 + `tests/test_FLA/test_chunk_kda.py`）
- 其中 `strict_tle` / `tle` 路径要求 CUDA，**不要在 Ascend 上复用其代码**；只借两个纯 Triton kernel：

```text
_kda_fwd_intra_triton_kernel    grid=(NT, B*H)
  phase0: q/k L2norm、beta sigmoid
  phase1: 逐 BK 块 cumsum(gate)->gk，得到 Aqk(因果) 与 Akk(严格下三角)
  phase2: (I + Akk)^-1，fp16 里做倍增（(I-L)(I+L^2)(I+L^4)...；BT=64 到 L^32）
  phase3: u = A @ (beta*v)、w = A @ (beta*k*2^gk)、qg、kg
  autotune: BK∈{16,32,64}, BV∈{16,32,64}, num_warps∈{1,2,4}, num_stages∈{1,2,4}

_kda_fwd_h_o_triton_kernel      grid=(V/BV, B*H), BV∈{32,64,128}
  state 按 K 切成 4 个 [64, BV] fp32 常驻寄存器，跨 chunk 循环不落 GM
  per chunk: v_new = u - w@h; o = scale*(qg@h) + Aqk@v_new
             h *= 2^gk_last; h += kg^T @ v_new
```

- `BT=16`（推理默认）/64（FLA 训练路径）；`do_not_specialize=["T"]` 避免 T 进入特化键。
- 这个文件里 state 分块与寄存器常驻的写法是 Ascend 上同样成立的策略：**h 不能每 chunk 从 GM 读写**。

## 4. klineage 仓内的 CUDA 侧材料（算法，不含 Ascend 映射）

- `skillcards/kda.*`（31 张）里有 chunk 级结论可直接迁移：
  `kda.chunk-decay-materialization`、`kda.recurrence-temporal-fusion`、
  `kda.cooperative-row-norm`、`kda.state-decay-register-reuse` 等。
  CUDA 专属的 TMA/warp-specialization 卡片（`kda.tma-*`）在 Ascend 上不适用。
- 本地 910B1 上的高性能 Triton 参考（GDN 融合 Stage6+7）见 `ascend-triton-gdn.md`：
  结构与实测结论与 KDA 的 FwdH 同构。
- `experiment/_transfer/kda/{with,without}_memory/apply/0/` 是 CUDA 运行的完整产物，
  可用于确认真实算子的输入输出形状，但实现是 CUDA 的。

## 5. 复现/对照

- 逐 token 参考实现（评测用的那个）在 problem definition 内；chunked 实现必须与它
  在小 shape（BT=16，T=64，H=2）上逐元素对齐后再放大。
