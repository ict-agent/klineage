# FMHA 内核结构

## 算法与资源布局

保留Q256/KV512、FP32 softmax与累加、FP16 P。没有降低精度、减少head、缩短序列或改变attention语义。

上一版每个AIV负责128行，但输出缓冲仅能容纳64行，因此每次调用新增3GiB逻辑GM读写：四轮KV中，前三轮写running O，后三轮读running O。本版为完整128×128 FP32输出保留64KiB，跨四轮KV更新始终留在UB，完全绕过这部分gUpdate读写。

| UB字节区间（KiB） | softmax阶段 | 输出更新阶段 |
|---|---|---|
| 0–64 | 双缓冲S | 当前PV结果（128行FP32） |
| 64–96 | 双缓冲P | 保留，不与输出别名 |
| 96–112 | 不使用 | 最终FP16输出暂存（每次64行） |
| 112–176 | 常驻running O | 常驻running O |
| 176–184 | 归约/广播临时空间 | 行缩放因子广播 |
| 184–189 | max/sum统计量 | 读取sum等统计量 |
| 189–190.5 | 三槽delta-max，每槽128行 | 按槽读取delta-max |

S在每次softmax调用结束后没有跨调用存活的数据。输出阶段借用该空间，入口和出口用V→MTE2事件交还所有权。最终FP16暂存与S分离，避免下一个Q任务加载S时覆盖尚未完成的输出DMA。128行更新的Add拆为两条128-repeat指令，避开uint8 repeat上限；最后输出拆为两次64行。

FP32更新仍为独立Mul、Add，最后Div、Cast；没有以FMA或倒数近似改变计算顺序。

固定任务数量4096，保持原先head最快变化的任务顺序。工作区实际使用S36MiB、P18MiB、partial O9MiB，共63MiB；**宿主仍按官方tiling分配88MiB**，本版不宣称减少了宿主实际分配量。这里的3GiB累计逻辑访问量也不等于3GiB实测HBM流量。

## 模块

- `src/fused_infer_attention_score/op_kernel/fmha_fixed_case.hpp`：固定几何、任务映射、工作区和编译期约束。
- `src/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/fmha_fixed_ub_layout.hpp`：共享 UB 布局与容量检查。
- `src/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/fmha_resident_output.hpp`：常驻累加、归一化、分块输出和流水同步。
- `flash_attention_regular.h`：接入输出路径并解码固定任务。

## 源码归属

`src/ascendc`、`src/common`、其他 attention 目录与生成的编译入口为构建依赖；沿用上游文件，不计为本次创新代码。`kernel_vs_upstream.patch` 的路径相对 `src/fused_infer_attention_score/`，改动统计见 `diff_upstream.json`。
