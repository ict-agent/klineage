# Task 2：两个算子、四组正式实验

正式范围：SparseAttention相同初始实现A/B优化；FusedAddRmsNorm AscendC A/B。
A=无预注入专家知识，B=有专家知识。模型均为DeepSeek-V4.1-Flash，Agent Harness为Codex，1M上下文，每组7200秒。

| 算子 | Setting | 正确性 | Baseline (ms) | 实现 (ms) | 加速比 |
|---|---|---|---:|---:|---:|
| sparse_attention | A | ✓（初始实现） | 470.202118 | 5648.614014 | 0.0832× |
| sparse_attention | B | ✓ | 466.355347 | 99.355560 | 4.6938× |
| fused_add_rmsnorm | A | ✓ | 6.009090 | 0.565930 | 10.6181× |
| fused_add_rmsnorm | B | ✓ | 6.019270 | 0.457040 | 13.1701× |

Baseline均为题目PyTorch reference，并非已有最优专用融合算子。SparseAttention两组使用相同初始源码，相对各自初始实现A为1.0000×、B为56.8563×；A未获得有效优化。FusedAddRmsNorm两组均有有效加速。

- [SparseAttention：定义、复现、曲线、限制](../sparse_attention/README.md)
- [FusedAddRmsNorm：定义、复现、曲线、限制](../fused_add_rmsnorm/README.md)
- [四组优化技巧对照表](optimization-techniques.md) · [PNG](optimization-techniques.png) · [PDF](optimization-techniques.pdf)
- [四组汇总CSV](results.csv)
- [四组时间对比图](four-groups-vs-minutes.png) · [PDF](four-groups-vs-minutes.pdf)
- [四组Token对比图](four-groups-vs-tokens.png) · [PDF](four-groups-vs-tokens.pdf)
- [正式结果清单](selection.json)
- [文件校验清单](SHA256SUMS.json)

每个算子包含A/B源码、全部评测快照、脱敏事件、时长审计、一键benchmark脚本及时间/Token两套PNG/PDF/CSV。
完整API与Codex Trace保留在本地。此汇总及算子目录不分发输入上下文与原始会话，属于脱敏交付包，不应描述为公开完整Trace。
这两个算子的旧公开产物已由本目录替换；历史原始记录保留本地私有归档。
每组预算满两小时，但末段存在无效resume；具体偏差与正确性失败在算子README中列明。


统一布局为 generation/<kernel>/{without_memory,with_memory,plots}。联合汇总与复测工具位于 next-plots/。

离线重画：`.venv-ascend/bin/python experiment/ascend/generation/next-plots/build_report.py`。
