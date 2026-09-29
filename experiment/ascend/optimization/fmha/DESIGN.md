# FMHA 执行结构

## 运行时分派

Python wrapper 校验 packed-NHD 输入、D 的范围和完整序列边界，随后选择路径：

```text
q/k/v + cumulative lengths
            |
       validate + dispatch
            |
     +------+-----------------------+
     |                              |
D128, uniform, aligned         tails / ragged / other D
     |                              |
FusedInferAttentionScore       FusionAttention (official)
     |
CANN tiling key selection
     |
resident output for the replaced non-split-KV key
```

常驻路径要求 Q/K 边界相同，序列长度是 512 的倍数且至少 1536。硬件 tile 是 Q256/KV512/D128，batch、heads、序列长度不再编译成常量。通用路径使用独立算子，避免不满足 UB 条件的输入进入被替换的 FIA key。形状分派覆盖前向正确性；性能验收仍只针对规定的 B8/H64/S2048/D128。

## 动态任务与工作区

`fmha_tile_config.hpp` 集中定义硬件 tile、流水深度和工作区跨度。batch/head 数来自 CANN tiling；序列长度与 token 总数来自实际边界。每批任务数为 `H * (S / 256)`，总任务数为 `B * H * (S / 256)`，保持 head 最快变化的次序。KV 循环次数为 `S / 512`，不再固定为四轮。

工作区按实际启动的 AIC 数分段：S、P、partial O 均使用三槽流水。24 AIC 时分别为 36 MiB、18 MiB、9 MiB，共 63 MiB；宿主仍使用官方 workspace 分配。核数较少时分段随之缩小，避免用固定 24 核跨度访问较小的宿主工作区。

## UB 常驻输出

每个 AIV 负责 128 行，为 128×128 FP32 输出保留 64 KiB，在所有 KV 轮次之间保持。没有降低精度或改变 attention 语义。针对原 S2048 验收 shape，相比 GM running-O 更新路径，省去每次调用累计 3 GiB 的逻辑 GM 读写；这不是实测 HBM 流量。

| UB 区间（KiB） | softmax 阶段 | 输出更新阶段 |
|---|---|---|
| 0–64 | 双缓冲 S | 当前 PV 结果（128 行 FP32） |
| 64–96 | 双缓冲 P | 保留 |
| 96–112 | 未使用 | FP16 输出暂存（64 行） |
| 112–176 | 常驻 running O | 常驻 running O |
| 176–184 | 归约/广播临时空间 | 行缩放因子广播 |
| 184–189 | max/sum | 读取统计量 |
| 189–190.5 | 三槽 delta-max | 按槽读取 |

S 在 softmax 调用结束后没有跨调用存活的数据，输出阶段借用该空间；V→MTE2 事件控制缓冲复用。最终 FP16 暂存与 S 分离，防止下一个 Q 任务覆盖未完成的输出 DMA。128 行 Add 拆成两条 128-repeat 指令，避开 uint8 repeat 上限；输出按两次 64 行写回。

FP32 更新保留独立 Mul、Add，最后 Div、Cast，不以 FMA 或倒数近似改变运算顺序。

## 模块

- `src/fused_infer_attention_score/op_kernel/fmha_tile_config.hpp`：硬件 tile、动态任务解码、工作区约束。
- `.../attn_infra/epilogue/block/fmha_fixed_ub_layout.hpp`：固定硬件 tile 的 UB 布局与容量检查。
- `.../attn_infra/epilogue/block/fmha_resident_output.hpp`：常驻累加、归一化、分块输出、流水同步。
- `flash_attention_regular.h`：读取运行时形状并接入输出路径。
- `fmha.py`：输入验证和安全分派；`check_shapes.py`：独立 FP32 参考校验。

`src/ascendc`、`src/common`、其他 attention 目录和生成入口沿用上游，不计为本次优化代码。完整补丁路径相对 `src/fused_infer_attention_score/`，统计见 `diff_upstream.json`。
