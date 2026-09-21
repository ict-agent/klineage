# Conv2d：Ascend / FlagTree TLE

## 算子与实现

| 项目 | 约定 |
| --- | --- |
| 输入 | 连续 NHWC `[8,56,56,64]`，FP16 |
| 权重 | 连续 `[576,128]`，FP16；按 Kh、Kw、C 顺序展平 HWCF |
| 输出 | 连续 NHWC `[8,56,56,128]`，FP16 |
| 运算 | 前向 3×3 cross-correlation；stride=1，padding=1，dilation=1，groups=1，bias=None |
| 主 case | N=8，C=64，H=W=56，F=128 |
| 累加 | 自定义算子 FP32 累加，FP16 输出 |

入口为 `src.conv2d.run(x, weight)`。主 case 使用 `tle.scope` 分配 Vector/Cube：
Vector 补零到 `[8,58,64,64]`，最多 24 个 Cube program 循环处理 256 个空间位置的 tile，
经 `tle.dsa.ascend.L1` 和 buffer frontend 完成九次矩阵乘法，再由 Vector 裁剪为连续 NHWC。
每次调用生成 padding、scratch、output；launcher 按设备、batch 缓存，接收当前指针与 stream。

## Baseline

[Ascend 官方 torch-npu](https://github.com/Ascend/pytorch) 的
`torch.nn.functional.conv2d`，底层为 CANN `aclnnConvolution`，采用库默认调度。
`src/baseline.py` 将 NHWC 输入变为 NCHW view，将 HWCF 权重转为连续 OIHW，
卷积输出转为连续 NHWC。权重转换每次执行。
接口与 `problems/definitions/conv2d.json` 中的 reference 一致。

布局转换改变数据排列：`permute` 调整 shape/stride，`.contiguous()` 及 CANN 内部格式转换
执行实际搬运。完整接口比较使用相同的 NHWC 输入、展平 HWCF 权重和连续 NHWC 输出。

## 环境与复现

实测环境：Linux aarch64，Ascend 910B1，驱动 25.5.2，CANN 9.1.0；
Python 3.12.13，torch 2.10.0+cpu，torch-npu 2.10.0.post4；
FlagTree `0.6.0+ascend.gitf56cd1bd`，Triton 3.5.1；主机分配器通过 `LD_PRELOAD` 使用 jemalloc。

在已安装上述 torch / torch-npu 和 CANN 的环境中安装固定版本编译器：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
git clone https://github.com/flagos-ai/FlagTree.git /tmp/conv2d-flagtree
git -C /tmp/conv2d-flagtree checkout f56cd1bdb98a88eedc9e3367620a6f8d68ab585b
python -m pip install 'setuptools>=40.8' 'cmake>=3.20,<4' 'ninja>=1.11.1' 'pybind11>=2.13.1' wheel
FLAGTREE_BACKEND=ascend python -m pip install /tmp/conv2d-flagtree --no-build-isolation
```

一键运行该 case 的正确性、完整接口计时和 kernel profiler：

```bash
TRITON_CACHE_DIR=$(mktemp -d /tmp/conv2d-triton.XXXXXX) \
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
PYTHON=/path/to/venv/bin/python bash run_benchmark.sh --device 0
```

输出为 `results/run.XXXXXX/` 下的 `full.json`、`kernels.json`、`kernels.csv`、
profiler trace；可用 `--output-dir DIR` 指定新目录。运行产物由 Git 忽略。
逻辑卡号通过 `--device` 指定，物理卡号默认按 `ASCEND_RT_VISIBLE_DEVICES` 映射，
也可通过 `--physical-device` 指定。Python 与 `npu-smi` 使用相同 PID 命名空间。

## 验证

CPU FP32 convolution 为精度参考，分别校验 baseline、自定义输出以及两者差异，
逐元素阈值为 `abs(a-b) <= 0.01 + 0.01*abs(b)`，NaN 判失败。
同一 Shape 的七组输入覆盖三个随机种子、零值、全 1、角落脉冲、非对齐连续 view；
每组额外重复五次并检查逐 bit 一致。另校验两个非默认 stream，以及 Graph 捕获后
原地址更新输入和权重时的全部捕获输出。

## 性能口径

所有加速比均为**同一行 baseline 时间 / 自定义时间**。四种口径的边界如下：

| 口径 | Baseline 计入范围 | 自定义计入范围 | 计时工具 |
| --- | --- | --- | --- |
| 卷积核心 | `aclnnConvolution_Conv2dWithFlag_Conv2D` task | `_conv_cube` task | torch-npu profiler 的设备 `Duration(us)` |
| 设备 kernel 总和 | 题目完整接口的全部 `aclnn*` task，含权重、输出转换及内部格式转换 | 每次完整调用的 `_prepare` + `_conv_cube` + `_crop` | 逐次调用求 task duration 之和 |
| 完整接口 Graph | HWCF→OIHW、卷积、输出→连续 NHWC 的捕获操作 | 补零、卷积、裁剪的捕获操作 | `perf_counter_ns`；每次 replay 后设备同步 |
| 完整接口 Eager | NHWC→NCHW view、HWCF→OIHW、卷积、输出→连续 NHWC | 补零、卷积、裁剪 | `perf_counter_ns`；每次调用后设备同步 |

- **输入边界**：全部测试使用同一题目 ABI：连续 NHWC 输入、展平 HWCF 权重、连续 NHWC 输出。每次调用执行权重转换或补零；核心耗时从该完整调用中提取。
- **统计量**：主结果用算术平均值，对齐 SparseAttnSharedKV 的平均耗时口径。每轮先求每次调用均值，再对等样本数的七轮求均值，等价于总时间除以总调用数。加速比为 baseline 均值 / TLE 均值。保留所有测量轮次；JSON 的 `median_us`、`min_us`、`max_us` 为七个轮均值的辅助统计，`samples_us` 保存轮均值。
- **核心/设备总和**：先各预热 20 次；七轮交替先后顺序，每轮每个实现紧邻测量前再预热 20 次，测 20 次，每块结束时同步。CSV 保留预热记录，统计时按块剔除；每个实现有效调用 140 次。设备总和先按调用累计全部 task duration（含同名 task 的每次出现），再求均值。`per_call_us` 保留逐次样本。
- **Graph**：各实现先 Eager 预热 20 次，默认每张图捕获一次完整调用，再预热 20 次 replay，每次 replay 后同步。七轮交替顺序，每轮开始先同步，然后计时 50 次 replay，每次 replay 后同步；主机总时间除以 50。`--graph-calls K` 可指定每张图捕获 K 次，分母变为 `50*K`，此时结果是 K 次调用的摊销时间，须连同 K 一起报告。
- **Eager**：复用初始化预热；七轮交替顺序，每轮开始先同步，然后计时 50 次 Python 函数调用，每次调用后同步；主机总时间除以 50。Graph/Eager 各有 350 次有效 replay/调用。
- **时间边界**：task 总和度量设备执行工作量；Graph 主机时间计入 replay 提交、设备执行、调度间隙及同步等待，capture 与捕获时分配在计时区间外；Eager 计入每次 Python 调度、分配、设备执行及同步等待。同步 API 的开销计入两种主机口径。
- **数据与缓存**：FP16 固定输入地址、warm-cache；每次执行所需转换与补零。JIT 编译、H2D、CPU 精度参考在计时区间外。主 case 计时数据种子为 4321（Graph 更新后），profiler 为 9182；CPU 线程数为 4。
- **运行控制**：每阶段开始及每轮前后检查设备健康与进程占用，要求目标卡独占；快照粒度为轮边界。`torch.inference_mode()`；CANN 默认策略，`allow_hf32_conv=True`、`allow_hf32_matmul=False`；自定义显式 `enable_auto_blockify=False`，Cube 使用 `sync_solver=False` 自动同步注入路径。
- **记录**：JSON 保存各轮样本、均值/中位数/最小/最大值、正确性、源码 SHA256 和设备占用；`full.json` 记录运行版本及框架开关，CSV 保留原始设备 task。

卷积核心指标用于评价计算 kernel。完整接口 Graph 对应已捕获的固定 Shape 调用，
Eager 对应逐次 Python 调用并等待完成的延迟。生产效果应按应用实际使用的执行模式、
张量布局和同步边界测量完整调用链。

## 性能结果

2026-09-21 03:56–03:57 UTC，Ascend 910B1 物理 NPU 2；
各轮占用快照均只有测试进程，全部正确性验证通过。
设备结果各统计 140 次有效调用，Graph/Eager 各统计 350 次；均为算术平均值。单位 μs。

| 口径 | torch-npu/CANN | TLE | 加速比 |
| --- | ---: | ---: | ---: |
| 卷积核心 | 26.5391 | 17.7352 | **1.496×** |
| 设备 kernel 总和 | 138.4524 | 29.2197 | **4.738×** |
| 完整接口 Graph | 177.8662 | 80.4040 | **2.212×** |
| 完整接口 Eager | 257.9054 | 277.0980 | **0.931×** |

Baseline 每次调用含八个设备 task：卷积核心均值 26.5391 μs，
布局/内部格式转换及初始化合计均值 111.9133 μs。
TLE 的补零/卷积/裁剪均值分别为 5.7671 / 17.7352 / 5.7174 μs。
4.738× 对应题目完整接口的设备 task 总和；卷积核心加速比为 1.496×。

七轮 Graph 均值范围：baseline 175.2194–181.9740 μs，TLE 75.6704–84.9293 μs；
Eager 均值范围：baseline 247.5752–311.6924 μs，TLE 261.5671–322.6779 μs。
