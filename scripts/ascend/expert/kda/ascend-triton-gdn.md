# 本地高性能 Triton 参考：GDN 融合 Stage6+7（910B1 实测）

代码：`/Users/rudy/codex/gdn/workspace/optimized/fused.py`（唯一实现，约 1500 行；
`fused_stage6_7*.py` / `pto_stage67*.py` 都是指向它的兼容 shim）。
实验报告：`/Users/rudy/codex/gdn/reports/`（head_pair / cross_aicore_pipeline / multibuffer / scan）。

KDA 的 FwdH 与这里的 Stage6 是同一个递推（`h = decay ⊙ h + kg^T · v_new`），
所以结构与实测结论可直接搬到 KDA 的 FwdH+输出阶段。

## 1 结构（可照搬）

```text
grid = (1, 24)  persistent worker，核内 stride 循环遍历 (sequence, head) 任务
单 kernel 内融合 Stage6（状态递推）+ Stage7（输出）
单 kernel 内分 AIC/Cube 与 AIV/Vector 两个角色，靠显式 pipe 对同步：
  al.sync_block_set/wait("cube"|"vector", flag_id, sender_pipe=..., receiver_pipe=...)
  例：AIV 写 gated_v -> MTE3 通知 -> AIC 取用 -> FIX 回 credit
状态 b_h_k0/b_h_k1（fp32，64x128 各 32 KB）在 AIV 侧寄存器常驻跨 chunk 循环，
  另存 per-worker GM state_workspace 供 Cube 侧读取（Cube 读不到 AIV 寄存器）
wh / qk / qh / gated_v / kv 各一块 per-worker workspace，buffer_id 做双缓冲，核间零共享
tile：BK=64、BV=128、BT=64（K=V=128）
尾块统一 block_ptr + boundary_check，不为尾块开分支内核
```

launch（`chunk_h_o_fused`）：

```python
num_warps=4, num_stages=2, enable_sync_block_lock=True,
multibuffer=False, disable_auto_inject_block_sync=True
```

## 2 实测（910B1，B=1,H=16,Hg=8,K=V=128,BT=64）

- `T=4096`：融合 708 us vs 拆开 Stage6+7 803 us（1.13x）；11 个 `T>=4096` case 平均 1.245x。
- 短序列（T=257~1024）机械融合反而慢 0.90~0.99x：按 shape 分派（融合 / 双 kernel），
  不要一刀切融合。
- 精度：新路径 `o` 最大差 4.88e-4（atol=rtol=1e-2 通过），`final_state`/`h`/`v_new` 完全一致。

## 3 负面结论（别重复踩）

- 跨 AICore 生产者/消费者不可用：910B1 上 `tl.atomic_*` 不更新 GM；普通
  `tl.store(flag,1)` + `volatile` 读能看到 flag 但看不到数据（`o` 出 NaN）。
  同核 `al.debug_barrier` 与 `sync_block_*` 都只约束本核流水。
- `al.sync_block_all("all_vector"/"all_sub_vector", 0)` 在 24 block 下直接报
  507035 失败；可用形态是 `sync_block_set/wait`（同核 Cube<->Vector）与收尾的
  `sync_block_all("all", 0)`。
  -> KDA 的 chunk 递推必须在一个 program 内跑完，不要用 GM flag 串多核。
- `multibuffer=True` / `set_workspace_multibuffer=2/4`：T=4096 更慢，跨 shape 无稳定收益。
- affine scan 重写（`H_{c+1} = P_c H_c + Q_c`）：数学成立，但 `P_c` 是 128x128 稠密矩阵，
  matmul 与 workspace 流量过大，已否决。KDA 同样别走这条路。

## 4 与 KDA 的差异（不要照抄的部分）

- 该实现面向 GQA（`Hg = H/2`），head-pair 是为了在 24 核下凑够任务；
  KDA 是 `H=96`、无 GQA，任务充足，跳过 head-pair。
- 它把每个 chunk 的入口状态 `h` 落 GM（bf16, `chunk_slots`），因为 Stage7 与反向要用；
  KDA 只要 `output`(bf16) 与 `final_state`(fp32)：融合后中间 `h` 不必落 GM。
- 变长负载均衡（3-sequence 分组、critical-token 排序、5% 阈值、metadata-free 地址）
  只在多序列 varlen 下有意义；本算子单序列 dense 且 `T % 64 == 0`，全部不需要。
