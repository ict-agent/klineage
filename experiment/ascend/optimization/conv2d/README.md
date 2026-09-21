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
原地址更新输入和权重时的全部八个输出。

## 性能口径

所有加速比均为**同一行 baseline 时间 / 自定义时间**。四种口径的边界如下：

| 口径 | Baseline 计入范围 | 自定义计入范围 | 计时工具 |
| --- | --- | --- | --- |
| 卷积核心 | `aclnnConvolution_Conv2dWithFlag_Conv2D` task | `_conv_cube` task | torch-npu profiler 的设备 `Duration(us)` |
| 设备 kernel 总和 | 题目完整接口的全部 `aclnn*` task，含权重、输出转换及内部格式转换 | 每次完整调用的 `_prepare` + `_conv_cube` + `_crop` | 逐次调用求 task duration 之和 |
| 完整接口 Graph | NHWC→NCHW view、HWCF→OIHW、卷积、输出→连续 NHWC | 补零、卷积、裁剪 | NPU start/end events 包围 Graph replay |
| 完整接口 Eager | 与 Graph 相同的完整接口 | 与 Graph 相同的完整接口 | 首尾同步的 `perf_counter_ns` 主机时钟 |

- **输入边界**：全部测试使用同一题目 ABI：连续 NHWC 输入、展平 HWCF 权重、连续 NHWC 输出。每次调用执行权重转换或补零；核心耗时从该完整调用中提取。
- **核心/设备总和**：先各预热 20 次；七轮交替先后顺序，每轮每个实现紧邻测量前再预热 20 次，测 20 次。每轮取 20 次中位数，再取七轮中位数。CSV 保留预热记录，统计时按块剔除。总和先按调用相加，再取中位数。
- **Graph**：各实现先 Eager 预热 20 次，每张图捕获八次调用，再预热 20 次 replay。七轮交替顺序，每轮测 50 次 replay，event 时间除以 `50*8` 得每次均值；最终取七轮均值的中位数。
- **Eager**：复用上述初始化预热；每轮连续调用 50 次，首尾设备同步，主机总时间除以 50；七轮交替顺序，取七轮均值的中位数。
- **时间边界**：task 口径仅累计 kernel 执行；Graph 计入设备操作及期间空隙，主机 capture/分配在计时区间外；Eager 计入 Python 调度、分配和设备等待。CPU 提交造成的设备空隙可反映在 event 区间内。
- **数据与缓存**：FP16 固定输入地址、warm-cache；每次执行所需转换与补零。JIT 编译、H2D、CPU 精度参考在计时区间外。主 case 计时数据种子为 4321（Graph 更新后），profiler 为 9182；CPU 线程数为 4。
- **运行控制**：每阶段开始及每轮前后检查设备健康与进程占用，要求目标卡独占；快照粒度为轮边界。`torch.inference_mode()`；CANN 默认策略，`allow_hf32_conv=True`、`allow_hf32_matmul=False`；自定义显式 `enable_auto_blockify=False`，Cube 使用 `sync_solver=False` 自动同步注入路径。
- **记录**：JSON 保存各轮样本、中位数/最小/最大值、正确性、源码 SHA256 和设备占用；`full.json` 记录运行版本及框架开关，CSV 保留原始设备 task。

## 性能结果

2026-09-21，Ascend 910B1 物理 NPU 2；上述验证全部通过。单位 μs。

| 口径 | torch-npu/CANN | TLE | 加速比 |
| --- | ---: | ---: | ---: |
| 卷积核心 | 25.8705 | 17.7300 | **1.459×** |
| 设备 kernel 总和 | 136.2235 | 29.1110 | **4.679×** |
| 完整接口 Graph | 134.3604 | 30.3817 | **4.422×** |
| 完整接口 Eager | 143.9784 | 227.9415 | **0.632×** |

七轮核心配对加速比为 1.453–1.476×。补零/卷积/裁剪各自中位数为 5.7100 / 17.7300 / 5.6905 μs。
阶段独立中位数之和与逐次调用求和的中位数存在差异。

七轮 Graph 均值范围：baseline 134.002–135.351 μs，TLE 30.336–40.502 μs；
Eager 均值范围：baseline 142.633–212.723 μs，TLE 208.959–277.172 μs。表中报告七轮中位数。
