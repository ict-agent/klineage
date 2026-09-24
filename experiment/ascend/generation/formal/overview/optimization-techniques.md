# 四组正式结果：NPU 优化技巧对照

以最终选中源码为准：SparseAttention A=version0（给定初始实现）、B=version20；FusedAddRmsNorm A=version10、B=version9。
**✓**：最终源码中存在；**✓\***：给定初始实现已有，不是A组新增成果；**—**：未观察到或不适用。未通过或未选中的尝试不记作已采用。

| 类别 | NPU优化技巧 | SparseAttention A | SparseAttention B | FusedAddRmsNorm A | FusedAddRmsNorm B |
|---|---|:---:|:---:|:---:|:---:|
| 并行与计算 | 多核任务划分 | ✓* | ✓ | ✓ | ✓ |
| 并行与计算 | Vector向量运算与归约 | ✓* | ✓ | ✓ | ✓ |
| 并行与计算 | Cube矩阵计算（QK/PV） | — | ✓ | — | — |
| 并行与计算 | Add+RMSNorm融合，中间和驻留UB | — | — | ✓ | ✓ |
| 搬运与流水 | UB暂存与数据复用 | ✓* | ✓ | ✓ | ✓ |
| 搬运与流水 | 块式/连续DMA搬运 | ✓* | ✓ | ✓ | ✓ |
| 搬运与流水 | 双缓冲与下一批输入预取 | — | ✓ | ✓ | ✓ |
| 搬运与流水 | 跨stream事件驱动流水 | — | ✓ | — | — |
| 搬运与流水 | 权重常驻UB，跨行复用 | — | — | ✓ | ✓ |
| 搬运与流水 | 跳过掩码/无效索引的数据搬运 | ✓* | ✓ | — | — |
| 局部计算与调度 | 分块/整行UB调度 | ✓* | ✓ | ✓ | ✓ |
| 局部计算与调度 | BF16存储、FP32向量中间计算 | ✓* | ✓ | ✓ | ✓ |
| 局部计算与调度 | 向量树形归约+标量尾部求和 | — | — | ✓ | — |
| 局部计算与调度 | 显式设置输出L2写缓存提示 | — | — | — | ✓ |

最适合归纳为共性的是：**多核分工、向量化、UB复用、块式搬运、分块调度和混合精度**。双缓冲预取在SparseAttention B及RMSNorm A/B中出现。

## 优先展示：专家组特有的差异

| 技巧 | SparseAttention A | SparseAttention B | FusedAddRmsNorm A | FusedAddRmsNorm B |
|---|:---:|:---:|:---:|:---:|
| 用Cube Matmul同时计算QK和PV | — | ✓ | — | — |
| Cube/Vector分工并使用独立stream | — | ✓ | — | — |
| 事件驱动的跨chunk流水 | — | ✓ | — | — |
| KV一次gather，供QK与PV共同复用 | — | ✓ | — | — |
| 显式配置L1深度及L0双缓冲 | — | ✓ | — | — |
| 设置输出L2写缓存提示 | — | — | — | ✓ |
| 查询硬件Vector核数设置launch grid | — | — | — | ✓ |
| 余数均摊的行划分，各核行数最多差1 | — | — | — | ✓ |
| 使用TQue管理生产者/消费者缓冲 | — | ✓ | — | ✓ |
| 使用ReduceSum API而非手写求和树 | ✓* | ✓ | — | ✓ |

**前五项是SparseAttention最值得展示的结构性差异。** A在Vector上做QK点积/PV累加，按head tile反复加载KV；B使用Cube Matmul，先将稀疏KV变为连续块，供两个矩阵阶段共同使用，再配合跨stream事件调度和双缓冲。

**RMSNorm两组已经共享融合、权重常驻和双缓冲，因此不能把这些标成B独有。** B的明确差异是输出写缓存提示、硬件核数查询、均衡行划分以及ReduceSum API。固定shape下，A用ceil分行导致40核中最后一核处理197行、其他核205行；B分配204或205行。动态核数查询本身不一定增加核数，也未证明有单独加速。

TQue是两组B共同采用、两组A均未采用的缓冲管理方式，但RMSNorm A通过手工HardEvent也实现了双缓冲流水。因此该行表示实现方式差异，不能等同“只有专家组会流水”。ReduceSum在SparseAttention A初始源码已经存在，也不能包装成跨算子B独有。

主图优先展示上述差异，上一张完整表保留共性。没有单项消融或profiling证据，不对每项技巧宣称独立加速收益。

## 源码依据

- [SparseAttention A](../sparse_attention/without_memory/submission/solution/kernel.asc)：L101按token跨核步进；L22–27设置head/key tile；L112起Q tile跨多个key复用、K/V tile跨head复用；L146使用DataCopyParams；L156起ReduceMax/Exp/ReduceSum完成softmax；L126/L189跳过无效索引。QK由Mul+ReduceSum完成，PV使用Axpy，不调用Cube Matmul。阶段间score/probability写回GM，不能称为无中间GM的融合attention。
- [SparseAttention B](../sparse_attention/with_memory/submission/solution/kernel.asc)：L27–31设置CHUNK=256、Cube baseM/N/K=128；L93跳过无效KV读取，L100附近将UB批量写成连续DataCopy；L119–123及L252–256设置Cube tiling、depthA1/B1=4和L0双缓冲；L128/L261调用Matmul；L160–200用深度2的TQue预取softmax下一行；L276用原KV stride复用已gather数据；L287起创建QK/PV streams，L357起以事件控制双缓冲阶段依赖。此处是多kernel分阶段流水，不是单kernel融合FlashAttention。
- [FusedAddRmsNorm A](../fused_add_rmsnorm/without_memory/submission/solution/kernel.asc)：L89将连续行分给各核；L98/L119每核加载一次weight；L104–110用两个UB槽预取下一行，HardEvent配对控制生产/消费；L165起融合Cast/Add/归一化，保留FP32和；L174–188按1024元素tile累加平方，向量树归约到32项再标量求和；L171提前发起residual输出，L197附近才等待写回完成。
- [FusedAddRmsNorm B](../fused_add_rmsnorm/with_memory/submission/solution/kernel.asc)：L34的SplitStart均匀分配行；L214查询Vector核数；L27/L74–77输入与output队列深度为2（residual输出为1），L93预取下一行；L101每核缓存weight；L131起在UB内完成融合计算，L147调用ReduceSum；L68–69仅对yGm/zGm设置WRITE缓存禁用提示。注释提到输入读缓存，但实际代码未设置输入读缓存，表格按代码标注。

## 解读边界

1. ✓表示机制存在，不代表每项技巧都做过独立消融，也不代表单项收益已测定。跨流并发和预取具有重叠设计，不据此宣称硬件始终满重叠。
2. SparseAttention A保留seed，不能把其已有技巧归功于本轮AI优化。B相对其初始版本增加Cube和流水等机制，但跳过无效索引不是初始版本完全没有的技巧。
3. 四组均仍有标量控制逻辑；不能写“完全消除标量循环”。RMSNorm A明确保留32项标量归约尾部，B也读取归约标量。
4. 未发现足够源码证据支持Split-K、显式循环展开、bank-conflict专项消除或workgroup swizzling，未沿用示例图中的这些勾选。
5. 图中使用Ascend术语，不能将AMD的MFMA、AGPR等名称直接替换成已实现能力。

[图片PNG](optimization-techniques.png) · [PDF](optimization-techniques.pdf) · [SVG](optimization-techniques.svg) · [绘图脚本](build_techniques.py)
