# Hybrid CP 动机实验：8 卡标准性能结果

记录日期：2026-10-21。实验代码提交：`0eac13f2`，分支：`motivation-v3`。
结果来源为本轮提供的控制台汇总表，配置按本轮确认的标准性能测试命令记录。

## 测试配置

| 项目 | 设置 |
| --- | --- |
| 并行规模 | 单节点 8 卡 |
| 数据集 | arxiv、freelaw、github、pile、prolong |
| Token budget | 每个 case 全局目标 131072 tokens（128K） |
| Case 数 | 每个数据集 20 个，共 100 个 |
| Attention | Causal，BF16，QH=32，KVH=8，head dimension=128 |
| 调度 | Megatron hybrid CP；CP-sizing target 为每 rank 8192 tokens，CP size 上限为 8 |
| 随机种子 | 0 |
| 预热与采样 | 每个 case、模式和阶段预热 10 次，测量 40 次 |
| 正确性检查 | `CHECK=0`；本轮采集性能，四卡正确性验收见[验证记录](../../baseline/megatron_hybrid_cp/SMOKE_H20.md) |

128K 是全局采样目标。序列对齐及调度后的 padding 会使实际执行 token 数
随 case 变化；同一 case 的三种模式使用相同的输入、CP assignment、padding、
样本顺序和 execution groups。

## 三种模式

| 模式 | KV / dKV 传输 | Step 执行 | 保留的本地操作 |
| --- | --- | --- | --- |
| `normal` | 保留正常 P2P ring 通信及其与计算的重叠 | 按 CP size 分 step 计算 | 每 step attention、O/LSE 合并、FP32 梯度累加 |
| `compute-only` | KV 在计时前就绪，计时内移除 KV 和 dKV 传输 | 保留相同的 step 切分 | 保留每 step attention、O/LSE 合并和 FP32 梯度累加 |
| `one-step` | KV 前缀在计时前就绪，计时内移除 KV 和 dKV 传输 | 每个样本将两个 zigzag Q 半段及各自 KV 前缀打包为一次 varlen FA3 调用 | 保留两个前缀重叠部分的 dKV 归并 |

三者形成 `normal → compute-only → one-step` 的渐进对照。CP1 使用相同的本地
attention 路径；不同样本和 execution groups 仍按原有顺序分别执行，组间
world barrier 保留在测量范围内。

无通信模式的 backward 保留本 rank 对各 KV owner 的梯度贡献。完整 owner
dKV 的跨 rank 汇总用于单独的正确性检查；`CHECK=0` 的性能运行关闭该检查。

## 测试阶段与计时范围

| Phase | 测量范围 |
| --- | --- |
| `forward` | 完整 `forward_all()`，包含所有样本和 execution groups |
| `backward` | 完整 `backward_all()`；所需 forward 在计时前完成 |
| `forward+backward` | 在同一个计时间隔内，先执行完整 forward，再执行完整 backward |

计时使用 CUDA 同步后的 wall time，单位为毫秒，包含 Python 调度、kernel 执行，
以及对应模式的通信、合并和组间同步。调度计划构造、process-group 创建、输入
分配、KV 预取和打包、梯度 buffer 重置、阶段开始前的同步均在计时范围外。
`forward+backward` 独立测量，其数值反映同一次联合执行。

## 指标定义

对给定数据集、模式和阶段，令 `t[c,i,r]` 表示 case `c` 第 `i` 次测量中
rank `r` 的耗时。先计算每次测量的最大 rank 耗时，再对该 case 的 40 次测量
求平均，最后对数据集的 20 个 case 等权求平均：

```text
case_latency[c] = mean_i(max_r(t[c,i,r]))
dataset_latency = mean_c(case_latency[c])
```

| 表头 | 含义 |
| --- | --- |
| `Normal ms` | `normal` 的 dataset latency，越小越好 |
| `Compute ms` | `compute-only` 的 dataset latency，越小越好 |
| `One-step ms` | `one-step` 的 dataset latency，越小越好 |
| `Norm/comp` | `Normal ms / Compute ms`；大于 1 表示移除通信后的加速倍数 |
| `Comp/one` | `Compute ms / One-step ms`；大于 1 表示进一步消除 step 切分后的加速倍数 |

两个比值均由数据集平均耗时相除得到。比如 `Norm/comp=1.213` 表示
compute-only 相对 normal 加速 1.213 倍，对应时延降低约 `1-1/1.213=17.6%`。
normal 与 compute-only 的时间差衡量移除通信对整段执行时延的影响，其中包含
通信与计算重叠的作用；compute-only 与 one-step 的差值反映 step 切分引入的
多次调用、合并及 kernel 执行形状的影响。

## 控制台汇总结果

时间和比值保留原始控制台输出的三位小数。

| Dataset | Phase | Normal ms | Compute ms | One-step ms | Norm/comp | Comp/one |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| arxiv | forward | 10.763 | 9.065 | 6.889 | 1.187 | 1.316 |
| arxiv | backward | 28.702 | 23.150 | 21.137 | 1.240 | 1.095 |
| arxiv | forward+backward | 39.238 | 32.362 | 27.683 | 1.213 | 1.169 |
| freelaw | forward | 5.050 | 4.384 | 3.538 | 1.152 | 1.239 |
| freelaw | backward | 13.160 | 11.350 | 10.635 | 1.159 | 1.067 |
| freelaw | forward+backward | 18.076 | 15.702 | 14.030 | 1.151 | 1.119 |
| github | forward | 5.507 | 4.852 | 3.957 | 1.135 | 1.226 |
| github | backward | 14.967 | 12.761 | 12.045 | 1.173 | 1.059 |
| github | forward+backward | 20.374 | 17.648 | 15.759 | 1.154 | 1.120 |
| pile | forward | 3.043 | 2.697 | 2.231 | 1.128 | 1.209 |
| pile | backward | 8.187 | 7.136 | 6.781 | 1.147 | 1.052 |
| pile | forward+backward | 11.177 | 9.838 | 8.897 | 1.136 | 1.106 |
| prolong | forward | 20.786 | 17.808 | 14.460 | 1.167 | 1.232 |
| prolong | backward | 57.234 | 45.804 | 43.657 | 1.250 | 1.049 |
| prolong | forward+backward | 77.834 | 64.012 | 57.028 | 1.216 | 1.122 |

## Forward+backward 的总体收益

根据上表展示的联合阶段耗时计算：

| Dataset | Normal / One-step | One-step 相对 Normal 的时延降低 |
| --- | ---: | ---: |
| arxiv | 1.417× | 29.4% |
| freelaw | 1.288× | 22.4% |
| github | 1.293× | 22.7% |
| pile | 1.256× | 20.4% |
| prolong | 1.365× | 26.7% |

五个数据集的三个阶段均呈现 `normal > compute-only > one-step` 的耗时关系。
在联合 forward+backward 中，移除通信带来 1.136–1.216× 加速，进一步合并
step 带来 1.106–1.169× 加速，合计对应 20.4%–29.4% 的时延降低。

## 标准性能测试命令

在仓库根目录、8 张完整 GPU 可用的节点上执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GPUS=8 \
DATASETS="arxiv freelaw github pile prolong" \
TARGET_TOKENS=131072 \
NUM_CASES=20 \
QHEAD=32 \
KVHEAD=8 \
MEGATRON_MAX_SEQLEN_PER_RANK=8192 \
SEED=0 \
WARMUP_ITERS=10 \
NUM_ITERS=40 \
CHECK=0 \
LOG_DIR="benchmark_logs/$(date +%Y%m%d-%H%M%S)-cp-motivation-8gpu" \
bash scripts/benchmark_cp_motivation.sh
```

测量与汇总代码：[benchmark_cp_motivation.py](../../ring_test/benchmark_cp_motivation.py)。
计算路径：[motivation.py](../../baseline/megatron_hybrid_cp/motivation.py)。
