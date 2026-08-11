# DCP Mega 性能分析与优化记录

> 测试日期：2026-08-06 至 2026-08-07
>
> 测试平台：8 x NVIDIA H100 80GB HBM3，CUDA 12.8，PyTorch 2.10.0+cu128
>
> 仓库版本：`bc26b79b21cbd92df0c2d01c3c83c3f5d70a84e9`
>
> 文档范围：`simple_bench.sh` 实测、case 004/007 根因分析、split/communication 调优、history combine 与 receive scheduler 优化及后续设计

本文记录本轮 DCP Mega 性能排查的完整结论。当前实现结构和同步协议的背景说明见
[`DCP_MEGA_CONVERSATION_NOTES.md`](DCP_MEGA_CONVERSATION_NOTES.md)；本文只讨论性能实验和后续优化，不重复描述整个实现。

## 1. 结论摘要

1. 原始 `simple_bench.sh` 的 10-case 结果中，Mega eager 平均 p50 为 `0.200558 ms`，vLLM A2A CUDA Graph 平均 p50 为 `0.190262 ms`，Mega 平均慢 `5.41%`。排除唯一的大工作量 case 004 后，其余 9 个 speculative-decode case 中，Mega 平均慢约 `14.18%`。
2. 最大退化来自 case 007：Mega auto 为约 `0.200 ms`，vLLM A2A graph 为约 `0.128 ms`，Mega 慢约 `56.4%`。
3. case 007 的 Mega auto 实际 history split 是 `[1,22,2]`。当前 auto heuristic 只优化 attention CTA occupancy，没有计入 split partial-state 的后续 reduction 成本。
4. auto 相比 split 3 只节省约 `20.2 us` attention，却增加约 `110.1 us` history-combine tail。手动 split 3 将 Mega p50 从约 `0.200 ms` 降到 `0.115472 ms`，降低约 `42.5%`。
5. split 4 配合 `num_comm_sm=4` 是本轮 Mega 最优点，p50 为 `0.113632 ms`；同样的 split 4 配合 `num_comm_sm=8` 是 `0.136016 ms`。差异主要位于 publish/receive 调度，不是 reduction 算术。
6. vLLM A2A 对 split 的响应与 Mega 不同：split 从 1 增至 16 时，独立 FA3 combine 仍能高效处理更多 partial；最优区间是 16 到 auto 22 的平台。干净长测中，vLLM A2A auto 为 `0.124864 ms`，手动 split 16 为 `0.125776 ms`，差异仅 `0.912 us`，属于噪声量级。
7. 用双方各自最优实测点比较：Mega split 4 / comm 4 为 `0.113632 ms`，比 vLLM A2A graph auto 的 `0.124864 ms` 低 `8.995%`，即约 `1.0988x` speedup。
8. vLLM 路径的 split combine 更快，准确地说，是它调用的 FA3 `FlashAttnFwdCombine` 更适合大 split reduction：CTA 更细、并行度更高、每个 vector 使用一个 warp、LSE reduction 不重复、partial O 使用 128-bit `cp.async` 四级流水，并采用预归一化权重后线性累加。Mega 当前以通信 publish tile 为 reduction 粒度，case 007 只有 24 个 history-combine task，无法占满 H100 的 132 个 SM。
9. 最重要的 combine 重构方向是：**把数值 reduction 粒度与通信 publication 粒度解耦**。建议每个 reduction subtask 处理 8 或 16 个 vector，多个 CTA 写入同一个 64-vector publish tile 的不重叠区域，通过 completion counter 让最后一个 CTA 执行 system-scope publish。

## 2. 比较口径与测量约束

### 2.1 baseline 的准确含义

本文中的“vLLM A2A”不是原生 vLLM engine 或原生 vLLM attention backend 的性能。它是仓库中的 vLLM-style orchestration baseline：

- local attention 与 Mega 使用同一个 `min_fa3_op.forward_kvcache_varlen`；
- history split reduction 调用本仓库复制并裁剪的 FA3 `FlashAttnFwdCombine`；
- Q 使用 all-gather；
- output 使用 BF16 packed all-to-all；
- CUDA Graph 捕获整条 baseline orchestration。

因此这里比较的是“相同 local FA3 attention 下，两种 DCP orchestration 与 combine 组织方式”，不能外推为 Mega 对原生 vLLM 的结论。

### 2.2 计时口径

原始 `simple_bench.sh` 比较使用：

| 路径 | 执行模式 | 计时范围 |
|---|---|---|
| Mega | eager | persistent mega-kernel only，workspace/metadata 已准备 |
| vLLM A2A | CUDA Graph | graph 内完整 A2A orchestration end-to-end |

这不是完全相同的 launch 形态，但与原始脚本的目标比较口径一致。Mega 的 kernel 内阶段由 SM90 `%globaltimer` 记录；baseline 阶段由 CUDA event 记录。所有 p50/p90 都先对同一次 iteration 的 8 个 rank 取最大值，再跨 iteration 取 percentile。

性能长测使用：

```text
warmup = 40
iters  = 60
CHECK  = 0
```

`CHECK=0` 只用于避免 correctness reference 干扰性能运行；此前实现已经完成单独的正确性验证。短 sweep 使用 `warmup=10,iters=20`，只用于定位趋势，最终比较采用长测。

### 2.3 测试环境

| 项目 | 值 |
|---|---|
| GPU | 8 x NVIDIA H100 80GB HBM3 |
| SM 数量 | 132 / GPU |
| CUDA runtime | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12.13 |
| OS | Linux 5.15 / glibc 2.35 |
| world size / TP / DCP | 8 / 8 / 8 |
| dtype | BF16 input/output，FP32 partial O/LSE |
| head dim | 128 |
| Q heads / KV heads | 32 / 1 |

主日志为：

- [`eager.log`](benchmark_logs/dcp_mega_trace_20260806-160903/eager.log)
- [`graph.log`](benchmark_logs/dcp_mega_trace_20260806-160903/graph.log)

## 3. 原始 `simple_bench.sh` 结果

### 3.1 10-case Mega eager 对 vLLM A2A graph

下表直接对齐原始比较口径。`差异` 为 `(Mega / vLLM - 1) * 100%`，正值表示 Mega 更慢。

| Case | Mega eager p50 (ms) | vLLM A2A graph p50 (ms) | Mega 差异 |
|---:|---:|---:|---:|
| 000 | 0.149 | 0.191 | -22.0% |
| 001 | 0.186 | 0.143 | +30.1% |
| 002 | 0.148 | 0.136 | +8.8% |
| 003 | 0.183 | 0.141 | +29.8% |
| 004 | 0.506 | 0.589 | -14.1% |
| 005 | 0.172 | 0.150 | +14.7% |
| 006 | 0.135 | 0.133 | +1.5% |
| **007** | **0.200** | **0.128** | **+56.4%** |
| 008 | 0.151 | 0.141 | +7.1% |
| 009 | 0.176 | 0.151 | +16.6% |
| **平均** | **0.200558** | **0.190262** | **+5.41%** |

case 004 的工作量明显更大，Mega 在该 case 反而更快。排除 case 004 后，9 个小 Q/speculative-decode case 的平均值约为：

```text
Mega eager       ~0.1667 ms
vLLM A2A graph   ~0.1460 ms
Mega slower      ~14.18%
```

这说明原始平均差距不是 attention 吞吐的单一问题。对短 Q case，固定调度成本、split reduction 和通信 ready/publish tail 会占据更高比例；case 007 是这一问题最清晰的样本。

## 4. Case 007 输入与原始阶段时间

### 4.1 Shape

```text
B                         = 3
Sq                        = [16, 16, 16]
global history            = [1139, 44536, 3167]
total_q                   = 48
DCP / TP                  = 8 / 8
QH / Hq_local / KVH       = 32 / 4 / 1
head_dim                  = 128
BlockN                    = 128
num_comm_sm               = 8
```

rank-local history 长度为：

```text
rank 0-2: [143, 5567, 396]
rank 3-6: [142, 5567, 396]
rank 7:   [142, 5567, 395]
```

### 4.2 Mega auto 原始 profile

```text
q_allgather_done                  5.200 us
attention_done                   17.088 us
history_combine_done            153.568 us
receive_done                    161.888 us
final_combine_done              187.920 us
kernel_done                     188.208 us

attention -> history tail       136.768 us
publish -> receive tail          14.144 us
history -> final tail            39.872 us

CUDA event p50                    0.200 ms
CUDA event p90                    0.206 ms
```

`history_combine_done` 与 `publish_done` 在当前实现中使用同一个时间戳，它表示所有 fused history-combine task 已完成并发出 remote-ready release，不表示存在一个额外的独立 publish pass。

最突出的事实是：attention 在约 `17 us` 已完成，而 history-combine 直到约 `154 us` 才完成。`136.768 us` 的 post-attention tail 是 case 007 的主要瓶颈。

## 5. Auto split 为什么得到 `[1,22,2]`

### 5.1 当前 heuristic

当前 split 计算位于 [`dcp_mega_metadata.py`](dcp_mega_metadata.py#L208)，核心形式为：

```text
m_i = ceil(q_i * heads / 128)
n_i = ceil(k_i / BlockN)

total_blocks = sum_i(m_i * n_i)
blocks_per_sm = max(ceil(total_blocks * 1.1 / num_sms), 1)
S_i = clamp(ceil(n_i / blocks_per_sm), 1, split_upper_bound)
```

case 007 history 路径中：

```text
n_blocks = [ceil(143/128), ceil(5567/128), ceil(396/128)]
         = [2, 44, 4]

m_blocks = ceil(16 * 32 / 128)
         = 4

total_blocks = 4 * (2 + 44 + 4)
             = 200

blocks_per_sm = ceil(200 * 1.1 / 132)
              = 2

actual splits = [ceil(2/2), ceil(44/2), ceil(4/2)]
              = [1, 22, 2]
```

8 个 rank 都得到 `[1,22,2]`。手动 `--num-splits=N` 也不是强制三条序列都使用 N，而是为逐序列动态 split 提供上限；因此例如请求 split 4，实际是 `[1,4,2]`。

### 5.2 heuristic 的目标与缺项

该 heuristic 的设计目标是让 attention work descriptor 足够多，改善 attention occupancy。对 44 个 N-block 的长序列，拆成 22 个 split 能快速铺满 SM，所以 attention 本体确实变快。

问题是它没有计入 split 的下游成本：

```text
partial O bytes
partial LSE loads
exp/log/isfinite 数学
history-combine task 的串行 split loop
publish ticket 与 ready dependency
通信 tile 生成和发布
```

在 FA3 独立 combine 足够高效时，这一缺项未必致命；在 Mega 当前 coarse-grained fused combine 中，它会使 attention occupancy 的收益被 reduction tail 数倍反噬。

## 6. Mega 手动 split 实测

### 6.1 长测结果

以下结果均为 `warmup=40,iters=60`。除特别标注外，`num_comm_sm=8`。

| 请求 split | 实际 history split | num_comm_sm | Mega p50 (ms) | Mega p90 (ms) |
|---:|---:|---:|---:|---:|
| 2 | `[1,2,2]` | 8 | 0.118848 | 0.122986 |
| 3 | `[1,3,2]` | 8 | 0.115472 | 0.121837 |
| 4 | `[1,4,2]` | 8 | 0.136016 | 0.144794 |
| **4** | **`[1,4,2]`** | **4** | **0.113632** | **0.117194** |
| 5 | `[1,5,2]` | 8 | 0.115936 | 0.120070 |
| 6 | `[1,6,2]` | 8 | 0.122272 | 0.125312 |
| 7 | `[1,7,2]` | 8 | 0.121376 | 0.126854 |
| auto | `[1,22,2]` | 8 | ~0.200 | ~0.206 |

结论不是“所有 Mega workload 都应该固定 split 3 或 4”，而是当前 auto heuristic 在 case 007 上严重过分拆分。最优值依赖 Q、各序列 local K 长度、head 数、BlockN、SM 数以及 combine 实现。

### 6.2 Attention 收益小于 reduction 代价

auto 与 split 3 的阶段对比表明：

```text
auto 相比 split 3：
attention 更快                  ~20.2 us
history reduction tail 更慢    ~110.1 us
净结果                         明显更慢
```

因此 case 007 的长尾不是 history attention 自身耗时，而是 attention 输出的 partial state 数量与 Mega combine 实现共同造成。

### 6.3 Reduction work 近似线性增长

定义 token-split state 数：

```text
R = sum_i(q_i * actual_split_i)
```

case 007 中：

| 请求 split | 实际 split | R |
|---:|---:|---:|
| 2 | `[1,2,2]` | 80 |
| 3 | `[1,3,2]` | 96 |
| 4 | `[1,4,2]` | 112 |
| 5 | `[1,5,2]` | 128 |
| 6 | `[1,6,2]` | 144 |
| 7 | `[1,7,2]` | 160 |
| 8 | `[1,8,2]` | 176 |
| 16 | `[1,16,2]` | 304 |
| auto | `[1,22,2]` | 400 |

较大采样点的 history-reduction tail 基本随 R 增长：

```text
R= 96   -> ~26.7 us
R=176   -> ~52.6 us
R=304   -> ~97.5 us
R=400   -> ~134-137 us
```

每个 token-split state 包含 `32 * 128` 个 FP32 partial O 元素。只计算 partial O，不含 LSE 和写回：

```text
auto:
400 * 32 * 128 * 4 bytes = 6,553,600 bytes / GPU

split 3:
96 * 32 * 128 * 4 bytes  = 1,572,864 bytes / GPU

ratio = 4.17x
```

这解释了为什么 auto 的 combine tail 大幅增长；但仅有 6.55 MB 数据不应在 H100 上花 130 us 以上，因此“数据变多”还不是完整答案。更深层原因是当前 Mega combine 的 CTA 粒度、重复数学、同步 load 和依赖链都不适合高 split reduction，详见第 9 节。

### 6.4 Split 1 暴露独立问题

Mega split 1 多次运行都挂住，GPU 随后处于 idle，且没有生成 JSON。split 1 会选择 non-split specialization，因此它并不是“split 2 比 split 1 更快”的正常性能现象，而更像另一条 specialization 中的同步或 progress bug。

在修复并独立验证该问题之前：

- 不应把 Mega split 1 纳入性能曲线；
- 不应让 cost model 自动选择 non-split specialization；
- 应用最小 case 在 timeout、CUDA memcheck/compute-sanitizer 和 phase counter 下单独排查。

## 7. `num_comm_sm` 为什么影响 split 4

split 4 的 reduction 算术完全相同，但 comm CTA 数改变了尾部：

| 配置 | attention -> history | publish -> receive | p50 |
|---|---:|---:|---:|
| split 4, comm 8 | ~31.6 us | ~38.0 us | 0.136016 ms |
| split 4, comm 4 | ~31.8 us | ~14.1 us | 0.113632 ms |

case 007 只有：

```text
final tiles       = 3
remote sources    = 7
receive tasks     = 3 * 7 = 21
```

每个 communication CTA 有 `kNumCommChunks=6` 个 receive chunk。receive task 的初始映射见
[`dcp_mega_min_fa3_varlen_launch.h`](include/dcp_mega_min_fa3_varlen_launch.h#L610)：

```text
comm CTA 0 -> task 0..5
comm CTA 1 -> task 6..11
comm CTA 2 -> task 12..17
comm CTA 3 -> task 18..20
comm CTA 4..7 -> 没有初始 receive task
```

因此 comm 8 并没有为 receive 增加有效并行度，却减少了 compute CTA 数，并改变了 compute CTA 的 block offset、initial attention descriptor 映射以及哪些 CTA 领取 history-combine ticket。当前通信 loop 又是 receive-first，且 history combine 只检查 queue head 是否 ready，不会越过未 ready head 扫描后续任务。这些因素共同造成明显的 CTA/ticket/publish 调度敏感性。

这一实测支持动态 comm CTA 数：

```text
num_comm_sm ~= ceil(max(q_transfer_tasks, receive_tasks) / kNumCommChunks)
```

case 007 中 `q_transfer_tasks=24`、`receive_tasks=21`、`kNumCommChunks=6`，公式恰好得到 4。实际实现还应做上下限 clamp，并在大 workload 上重新标定，不能直接视为普适最优公式。

## 8. vLLM A2A split sweep

### 8.1 为什么增加独立 `vllm_a2a` selector

原来的 `--implementations vllm` 会在同一进程中先执行 AG+RS runner，再执行 A2A runner。为了避免前一个 collective runner 对 A2A 测量状态的影响，本轮增加了 benchmark-only 的 `--implementations vllm_a2a` 选择器：

- parser：[`dcp_test/benchmark_dcp_varlen.py`](dcp_test/benchmark_dcp_varlen.py#L102)
- runner 选择：[`dcp_test/utils.py`](dcp_test/utils.py#L531)
- 使用说明：[`dcp_test/README.md`](dcp_test/README.md#L55)

该改动只隔离 benchmark runner，没有改变 vLLM-style baseline 的 kernel 或 collective 算法。

### 8.2 短 sweep

以下为 CUDA Graph 短测 `warmup=10,iters=20` 的 A2A 结果。`local history phase` 包含 history attention 和 FA3 combine，**不是纯 combine kernel 时间**。

| 请求 split | 实际 history split | A2A p50 (ms) | local history phase (us) |
|---:|---:|---:|---:|
| 1 | `[1,1,1]` | 0.198448 | 73.888 |
| 2 | `[1,2,2]` | 0.180224 | 46.880 |
| 3 | `[1,3,2]` | 0.167664 | 37.760 |
| 4 | `[1,4,2]` | 0.136976 | 32.224 |
| 5 | `[1,5,2]` | 0.134368 | 30.336 |
| 6 | `[1,6,2]` | 0.132800 | 28.640 |
| 8 | `[1,8,2]` | 0.131888 | 25.024 |
| 12 | `[1,12,2]` | 0.128352 | 23.360 |
| 16 | `[1,16,2]` | 0.126976 | 22.112 |
| 22 | `[1,22,2]` | 0.127536 | 22.144 |

趋势很清楚：vLLM/FA3 combine 能承受更多 split，attention 的并行收益一直持续到 16 左右；16 到 22 已进入平台，继续增加 split 不再有收益。

### 8.3 干净长测与双模态噪声

最终隔离长测：

```text
vLLM A2A manual split 16:
p50                 = 0.125776 ms
p90                 = 0.129296 ms
local history phase = 22.560 us
Q all-gather        = 25.552 us

vLLM A2A isolated auto:
p50                 = 0.124864 ms
p90                 = 0.128058 ms
local history phase = 22.288 us
Q all-gather        = 26.224 us
```

手动 split 16 是最好的手动点，但 auto 比它快 `0.912 us`。该差值低于 collective run-to-run 波动，合理结论是 **vLLM 的最优区间为 16 到 22，auto 已经近似最优**，而不是“22 严格优于 16”。

实验中还观察到部分新进程 launch 进入 rank-grouped collective 慢模式：

```text
正常 Q all-gather      25-27 us
慢模式 Q all-gather    47-55 us
慢模式 A2A p50         0.147-0.156 ms
```

auto 和多个手动 split 都出现过该模式，所以它与某个 split 值没有因果关系。最终比较使用与原始 `simple_bench.sh` 一致的干净 collective path，同时保留这一噪声说明，避免选择性忽略异常结果。后续严谨评估应重复启动多轮进程，并同时报告 clean-mode 分布、慢模式发生率和全样本分布。

## 9. 为什么 vLLM/FA3 的 split combine 更高效

严格地说，更快的是 vLLM-style baseline 使用的专用 FA3 `FlashAttnFwdCombine`，入口位于：

- [`min_fa3_kvcache_combine_launch.h`](include/min_fa3_kvcache_combine_launch.h#L23)
- [`min_fa3_fwd_combine_kernel.h`](include/hopper_compat/min_fa3_fwd_combine_kernel.h#L25)

它复制自 FlashAttention commit `c75d019dea9d910312974417bc28f190dfdda6d9` 的 Hopper forward combine 路径。Mega 的 fused history combine 位于
[`dcp_mega_min_fa3_varlen_launch.h`](include/dcp_mega_min_fa3_varlen_launch.h#L423)。

### 9.1 CTA 粒度和全芯片 occupancy

FA3 combine 的 tile 是 `8 vectors x 128D`，每 CTA 256 threads。case 007 有：

```text
48 tokens * 32 heads = 1536 vectors
1536 / 8             = 192 combine CTAs
```

192 个 CTA 足以覆盖 H100 的 132 个 SM，并让尾部还有第二波 CTA。

Mega 的 publish tile 固定是：

```text
16 tokens * Hq_local(4) = 64 vectors / task
3 token blocks * 8 destination ranks = 24 history-combine tasks
```

所以即使还有大量 partial O 要 reduce，Mega 也最多只有约 24 个 active combine worker，远低于 132 个 SM。将通信 tile 直接作为数值 reduction tile，是当前最主要的并行度限制。

### 9.2 每个 vector 的线程分工

FA3 使用 32 threads 处理一个 128D vector。每个 thread 通过 128-bit load 读取 4 个 FP32 值：

```text
32 threads * 4 floats = 128 floats
```

Mega 使用 8 lanes 处理一个 vector，每 lane 串行持有并更新 16 个 FP32 accumulator：

```text
8 lanes * 16 floats = 128 floats
```

Mega 虽然减少了一个 vector 占用的线程数，但在 case 007 中 reduction task 数本来就太少，节省线程并不能转化为更多有效 CTA；反而增加了每个 lane 的寄存器、指令和串行依赖。

### 9.3 LSE reduction 是否重复

FA3 先把 partial LSE 以 swizzled layout 搬到 shared memory，再由同一 warp 内的线程并行做 max/sum reduction。归一化权重随后写回 shared memory，partial O accumulation 直接复用这些权重。

Mega 的 8 个 lane 都执行完整的 runtime split loop。对于同一个 vector，以下量在 8 个 lane 上完全相同，却被重复计算 8 次：

```text
state_lse load
isfinite checks
max_lse / next_max
expf(previous scale)
expf(state scale)
denominator
combined LSE state
```

只有 128D O 的不同维度需要 lane 私有处理。把 LSE 数学放在一个 warp 上做一次，再广播 normalized scale，能直接消除这部分重复。

### 9.4 在线 recurrence 的依赖链

Mega 当前每个 split 使用在线合并：

```text
accum = accum * previous_scale + partial * state_scale
denominator = denominator * previous_scale + state_scale
```

这意味着每进入一个 split，都必须重缩放当前 16 个 accumulator，且下一个 split 依赖上一个 split 的 `max_lse`、`denominator` 和整个 accumulator 状态。长 split loop 形成严格串行链。

FA3 先完成：

```text
weight_s = exp(lse_s - max_lse) / sum_s(exp(lse_s - max_lse))
```

随后执行：

```text
accum += weight_s * partial_s
```

这样每个 split 不再重缩放既有 accumulator，主循环只有 load、convert 和 FMA，依赖链更短，也更适合流水化。

### 9.5 Memory pipeline

FA3 combine 明确使用：

- 128-bit `cp.async` partial O load；
- 4-stage shared-memory pipeline；
- load future split 与当前 split accumulate 重叠；
- shared-to-register vectorized copy；
- compile-time fixed tile layout。

相关代码见 [`min_fa3_fwd_combine_kernel.h`](include/hopper_compat/min_fa3_fwd_combine_kernel.h#L35) 和同文件的
[partial O pipeline](include/hopper_compat/min_fa3_fwd_combine_kernel.h#L516)。

Mega 当前直接从 global memory 做同步 FP32 scalar load 到 register array。每个 lane 每个 split 读取 16 个标量，中间没有等价的多级 shared-memory pipeline，无法有效隐藏 partial O load latency。

### 9.6 Compile-time specialization

FA3 根据 max split 选择 32/64/128 等 compile-time 模板，layout、shared memory 和循环上界都能展开和调优。其 split loop 已显式使用 tuned unroll。

Mega 的实际 split 是 per-batch runtime 值，由 `history_sequence_splits[batch]` 读取，然后进入 runtime loop。它没有等价的 max-split specialization，也会为每个 vector 通过 `batch_for_token()` 做一次 token 到 batch 的二分搜索。

### 9.7 Mega history timestamp 包含更多工作

Mega 的 `attention_done -> history_combine_done` 不只是纯数学 reduction，还包含：

- attention dependency wait；
- shared atomic ticket 竞争；
- 每个 vector 的 `batch_for_token()`；
- partial O/LSE reduction；
- FP32 到 BF16 communication tile 转换；
- CTA `__syncthreads()`；
- shared tile 的 TMA store/wait；
- local 或 system-scope ready release；
- receive-first loop 带来的调度影响。

FA3 combine kernel 只生成 local O/LSE。A2A pack、collective、unpack/final state merge 是其他 graph node。因此不能把 Mega 的 `136.8 us` tail 与 vLLM 的 `22.3 us local_history_attention_ms` 当成两个“纯 combine kernel”直接相除。

即便考虑这一口径差异，split sweep 仍提供强证据：vLLM/FA3 在 split 16-22 时 local history phase 仍约 `22 us`，而 Mega 的 reduction tail 随 token-split state 增至 `100 us` 以上，说明 combine kernel 结构本身确实是关键差异。

### 9.8 CUDA Graph 与 PDL

FA3 attention 后调用独立 combine kernel，但该路径启用 Programmatic Dependent Launch：

- 调用位置：[`min_fa3_kvcache_bindings.cu`](csrc/min_fa3_kvcache_bindings.cu#L592)
- combine 内等待 dependent grid：[`min_fa3_fwd_combine_kernel.h`](include/hopper_compat/min_fa3_fwd_combine_kernel.h#L485)

CUDA Graph 又消除了 Python 和普通逐 kernel launch 的大部分 host overhead。因此“Mega 把一切融合进一个 kernel”并不自动意味着更快：如果融合使 reduction 粒度被通信粒度绑死、降低 occupancy，独立但专门优化的 combine kernel可以更快。

## 10. 调优后的最终比较

采用本轮双方最优且干净的长测结果：

| 实现 | 配置 | p50 (ms) | p90 (ms) |
|---|---|---:|---:|
| vLLM A2A graph | auto，实际约 `[1,22,2]` | 0.124864 | 0.128058 |
| Mega eager | split 3，comm 8 | 0.115472 | 0.121837 |
| **Mega eager** | **split 4，comm 4** | **0.113632** | **0.117194** |

Mega split 3 使用默认 comm budget 时：

```text
latency reduction = (0.124864 - 0.115472) / 0.124864
                  = 7.52%
speedup           = 1.081x
```

最佳 Mega split 4 / comm 4：

```text
p50 reduction     = (0.124864 - 0.113632) / 0.124864
                  = 8.995%
speedup           = 1.0988x
p90 reduction     = (0.128058 - 0.117194) / 0.128058
                  ~= 8.48%
```

因此最终结论是：**case 007 的原始差距不是 Mega 架构必然落后，而是 auto split 与当前 combine/publish 粒度失配；手动校正 split 和 comm CTA 数后，Mega 已快于调优后的 vLLM A2A baseline。**

## 11. Mega combine 优化设计

### 11.1 第一原则：拆开 reduction tile 与 publish tile

当前一个 `PublishWorkDesc` 同时定义：

```text
数值 reduction 粒度 = 64 vectors
BF16 communication tile = 64 vectors
ready publication 粒度 = 64 vectors
```

后两者需要 64-vector tile，但前者没有必要。建议保留现有 16-token x `Hq_local` publication ABI，同时增加更细的 reduction subtask：

```text
ReductionWorkDesc:
    publish_id
    vector_offset_in_publish_tile
    valid_vectors          # 8 或 16
    batch/token base       # 避免运行时二分搜索
```

case 007 的任务量将变为：

```text
8-vector subtask:   1536 / 8  = 192 tasks
16-vector subtask:  1536 / 16 = 96 tasks
```

两种粒度都比当前 24 tasks 更能覆盖 132 个 SM。优先测试 8 和 16，而不是先假设其中一个普适最优。

### 11.2 多 CTA 填充同一个 publish tile

不同 CTA 不能共享普通 shared memory，因此“多个 reduction CTA 填一块 communication tile”需要明确的 global staging/publication 协议。推荐两个实现选项。

**选项 A：直接写最终 global send tile，最后一个 CTA 发布 ready。**

1. 每个 subtask 将自己的 BF16 vector range 用 vectorized global store 写入 `history_send_local[dst_rank]` 的不重叠区域。
2. 同时将 combined FP32 LSE 写入对应的不重叠位置。
3. CTA 完成写入后，对 `publish_completion[publish_id]` 做 release-capable atomic increment。
4. 观察到 completion 达到 expected subtasks 的最后一个 CTA 执行必要的 device/system fence。
5. 最后一个 CTA 才更新 local `publish_ready` 或 remote `tile_ready`。

这个方案不再需要先在单 CTA shared memory 中形成完整 64-vector tile，也省掉 shared-to-global TMA store。remote reader 只在 ready release 后访问，因此 publication 原子和 system-scope memory ordering 必须经过 litmus/correctness test 验证。

**选项 B：global staging + 最后一个 CTA TMA publish。**

1. subtasks 把 BF16/LSE 写到本地 global staging tile；
2. completion counter 选出最后一个 CTA；
3. 最后一个 CTA 将完整 staging tile 搬到 shared memory；
4. 从 shared memory 发起现有 TMA store，并在完成后 release ready。

该方案更接近当前 TMA publication 路径，改动风险较低，但多一次 global round trip，预期性能弱于选项 A。可以先用它验证调度解耦收益，再迁移到直接写最终 send tile。

### 11.3 Warp-per-vector LSE 与 O 分工

建议每个 128D vector 使用一个完整 warp：

```text
lane 0..31
每 lane load 4 x FP32 partial O
```

每个 vector 的 LSE reduction 只做一次：

1. warp 协作加载各 split LSE；
2. warp reduce max；
3. 计算 `exp(lse-max)`；
4. warp reduce sum；
5. 生成 normalized split weights；
6. weight 存入 shared memory或保留在协作 lane/register；
7. 所有 lane 对各自 4 个 O 维度做 `accum += weight * partial`。

这样同时解决当前 8-lane 重复 LSE 数学和每 lane 16 accumulator 的串行压力。

### 11.4 Partial O 的异步流水

在 reduction subtask 内复制 FA3 combine 已验证的结构：

- 128-bit aligned `float4`/CuTe copy atom；
- 3 或 4 个 shared-memory stage；
- 预取 split `s+kStages-1`，同时计算 split `s`；
- 先归一化 LSE scale，再进入纯 FMA partial O loop；
- 根据 max split 模板化 32/64/128 上界。

这项改动应放在任务粒度解耦之后。当前只有 24 个 coarse task 时，仅做 cp.async 未必能解决全芯片 occupancy 不足。

### 11.5 去除 descriptor 热路径开销

当前每个 vector 调用 `batch_for_token(token, cu_seqlens_q, batch_size)`。metadata 已经知道 publish task 对应的 token block 和 batch，可以把以下字段直接编码进 work descriptor：

```text
batch_id
token_begin
history_head_begin 或 dst_rank
actual_splits
```

这样可移除 per-vector binary search 和多次地址推导。对于 case 007 这类 `Sq=16` 的规则 tile，descriptor 甚至天然不会跨 batch；若未来允许跨 batch，需要 metadata 在边界处分裂 descriptor，而不是在 device 热路径搜索。

### 11.6 Ready queue 与 head-of-line blocking

通信 CTA 的 `try_run_ready_history_combine()` 当前只看 queue head：

```text
head ready     -> consume and run
head not ready -> return -1，不扫描后续 ready task
```

当不同序列或 destination 的 attention 完成时间不一致时，这会造成 head-of-line blocking。可以按风险从低到高考虑：

1. 在 head 后有限窗口内扫描 4-8 个 descriptor；
2. 每个 publish task 使用 ready bitmap，CTA 通过 `ffs` 领取；
3. attention 完成时把最后解除依赖的 publish task push 到 ready queue。

推荐先做 bounded scan/bitmap，不要立刻引入复杂的全局 MPMC queue。需要保持 ticket 唯一领取和 graph phase monotonicity。

### 11.7 Split cost model

新的 auto split 不能只估算 attention blocks，应优化近似 critical path：

```text
T_total(S) ~= T_attention(S)
           + alpha * sum_i(q_i * S_i) * heads * head_dim
           + beta  * reduction_task_count
           + gamma * publish_receive_tail
```

其中第一项随 split 增加先快速下降后饱和，第二项随 split 近似线性增加。`alpha/beta/gamma` 应从多 shape 实测拟合，并分别为现有 combine 和新 combine 标定。

在新模型上线前，可以使用保守的两阶段 heuristic：

1. 先按现有公式得到 occupancy-oriented split；
2. 用 `sum(q_i*S_i)` 预算做 cap，使 predicted reduction bytes/tail 不超过阈值；
3. 同时限制 split 增加必须带来足够多的新 attention CTA；
4. case 007 预期会从 22 限制到 3-5 区间。

不要把 case 007 的 split 4 写死为全局常量；它只应作为 cost model 的回归样本。

### 11.8 Dynamic `num_comm_sm`

建议初始公式：

```text
num_comm_sm = clamp(
    ceil(max(q_transfer_tasks, receive_tasks) / kNumCommChunks),
    min_comm_sm,
    max_comm_sm)
```

还应加入：

- communication CTA 参与 history combine 的预期比例；
- compute CTA 至少保留多少个；
- DCP size 和 NVLink topology；
- receive task 是否足以覆盖所选 CTA；
- 大 Q 时 Q all-gather TMA task 数。

case 007 的 comm 4 结果说明这是低风险、高收益的第一阶段改动，但必须在全部 trace case 上做回归，尤其关注 case 004 等大工作量 case。

## 12. 推荐实施顺序与验收标准

### Phase 1：heuristic 与动态 comm CTA

- split cost 加入 `sum(q_i*S_i)*heads*head_dim`；
- 加入 dynamic `num_comm_sm`；
- 保留环境变量/CLI override，便于 A/B；
- 单独修复 split 1 specialization hang。

验收：10-case 平均不退化，case 007 auto 落到 3-5，达到接近当前手调 `0.114-0.116 ms` 的水平。

### Phase 2：低风险 combine 微优化

- descriptor 携带 batch/token base，移除 `batch_for_token()`；
- LSE 数学每 vector 只做一次；
- 128-bit partial O load；
- normalized-scale FMA 替换 online recurrence。

验收：固定 split 3/4/8/16 分别测 history tail，确认每 token-split state 成本下降，并用 correctness matrix 覆盖 `-inf` LSE、空 history、不同 actual split。

### Phase 3：reduction subtasks + publish completion

- 引入 8/16-vector reduction descriptor；
- 为每个 64-vector publish tile 建 completion counter；
- 实现选项 A 或先实现选项 B；
- 严格验证 system-scope visibility 和 graph phase 复用。

验收：case 007 产生 96-192 reduction tasks；Nsight Systems/Compute 显示 combine wave 覆盖接近全芯片；split 16/22 的 tail 不再近似按当前斜率增长。

### Phase 4：cp.async 多级流水与 max-split specialization

- 3/4-stage partial O pipeline；
- 32/64/128 max-split 模板；
- register/shared-memory occupancy 调优；
- 对 8-vector 与 16-vector tile 做系统 sweep。

验收：在固定 task 粒度下进一步降低 combine cycles，并确认 shared-memory 增长没有降低 CTA residency 到抵消收益。

### Phase 5：ready 调度重构

- bounded scan、bitmap 或 ready queue；
- 消除 queue-head 未 ready 导致的 combine 空转；
- 联合调整 receive-first 策略。

验收：构造不同序列 K 长度极不均衡的 workload，验证 publish/receive tail 和 p90/p99 收敛，而不仅是 case 007 p50 改善。

## 13. 复现命令

运行原始 10-case trace：

```bash
cd /home/hychen/min_fa3_demo
./scripts/test_dcp/simple_bench.sh
```

复现 Mega 最优 case 007 配置：

```bash
cd /home/hychen/min_fa3_demo
torchrun --standalone --nproc_per_node=8 \
  --module dcp_test.benchmark_dcp_varlen \
  --b 3 --sq 16,16,16 --seqlen 1139,44536,3167 \
  --qhead 32 --kvhead 1 --headdim 128 \
  --tp-size 8 --dcp-size 8 --workload chunk \
  --implementations mega --no-cuda-graph \
  --mega-phase-timestamps --mega-block-n 128 --mega-num-comm-sm 4 \
  --num-splits 4 --warmup 40 --iters 60 --no-check \
  --output-json benchmarks/results/case007_mega_split4_comm4_repro.json
```

复现隔离的 vLLM A2A auto：

```bash
cd /home/hychen/min_fa3_demo
torchrun --standalone --nproc_per_node=8 \
  --module dcp_test.benchmark_dcp_varlen \
  --b 3 --sq 16,16,16 --seqlen 1139,44536,3167 \
  --qhead 32 --kvhead 1 --headdim 128 \
  --tp-size 8 --dcp-size 8 --workload chunk \
  --implementations vllm_a2a --cuda-graph --baseline-phase-timing \
  --num-splits 0 --warmup 40 --iters 60 --no-check \
  --output-json benchmarks/results/case007_vllm_a2a_auto_repro.json
```

运行前应先用 `nvidia-smi` 确认 8 张 GPU 没有其他进程。本轮末尾曾观察到 GPU 4-7 上出现无关的 root-owned GPT pretraining 任务，因此本文没有在该状态下继续追加 8-GPU 数据。

## 14. 结果与源码索引

### 14.1 主要结果

- 原始 Mega auto：[`case 007 eager JSON`](benchmark_logs/dcp_mega_trace_20260806-160903/results/eager/case_000007_dcp8_hkv1_eager.json)
- 原始 vLLM A2A graph：[`case 007 graph JSON`](benchmark_logs/dcp_mega_trace_20260806-160903/results/graph/case_000007_dcp8_hkv1_graph.json)
- Mega split 2：[`case007_split2_long.json`](benchmarks/results/case007_split2_long.json)
- Mega split 3：[`case007_split3_long.json`](benchmarks/results/case007_split3_long.json)
- Mega split 4 / comm 8：[`case007_split4_mega_only_long.json`](benchmarks/results/case007_split4_mega_only_long.json)
- Mega split 4 / comm 4：[`case007_mega_split4_comm4_final_long.json`](benchmarks/results/case007_mega_split4_comm4_final_long.json)
- Mega split 5：[`case007_split5_long.json`](benchmarks/results/case007_split5_long.json)
- Mega split 6：[`case007_split6_long.json`](benchmarks/results/case007_split6_long.json)
- Mega split 7：[`case007_split7_long.json`](benchmarks/results/case007_split7_long.json)
- vLLM A2A isolated auto：[`case007_vllm_a2a_isolated_auto_long.json`](benchmarks/results/case007_vllm_a2a_isolated_auto_long.json)
- vLLM split 16 clean repeat：[`case007_vllm_graph_split16_long_repeat.json`](benchmarks/results/case007_vllm_graph_split16_long_repeat.json)

vLLM 短 sweep 文件统一位于 `benchmarks/results/case007_vllm_graph_split*_smoke.json`。

### 14.2 关键源码

- split heuristic：[`dcp_mega_metadata.py`](dcp_mega_metadata.py#L208)
- history split metadata 构造：[`dcp_mega_metadata.py`](dcp_mega_metadata.py#L318)
- Mega history combine：[`dcp_mega_min_fa3_varlen_launch.h`](include/dcp_mega_min_fa3_varlen_launch.h#L423)
- Mega communication loop：[`dcp_mega_min_fa3_varlen_launch.h`](include/dcp_mega_min_fa3_varlen_launch.h#L610)
- communication CTA 分类：[`dcp_mega_min_fa3_varlen_launch.h`](include/dcp_mega_min_fa3_varlen_launch.h#L1000)
- initial scheduler 映射：[`dcp_mega_min_fa3_varlen_scheduler.h`](include/dcp_mega_min_fa3_varlen_scheduler.h#L121)
- FA3 combine launch：[`min_fa3_kvcache_combine_launch.h`](include/min_fa3_kvcache_combine_launch.h#L23)
- FA3 combine kernel：[`min_fa3_fwd_combine_kernel.h`](include/hopper_compat/min_fa3_fwd_combine_kernel.h#L25)
- FA3 combine PDL 调用：[`min_fa3_kvcache_bindings.cu`](csrc/min_fa3_kvcache_bindings.cu#L592)

## 15. 最终判断

case 007 原始 `0.200 ms` 并不是 history attention 太慢，而是三个因素叠加：

```text
attention-only auto split 过大
        x
Mega coarse-grained、串行化的 split reduction
        x
comm CTA / ticket / publish 调度敏感性
```

短期最有效的方案是让 split heuristic 感知 reduction 成本，并按实际 Q/receive task 数动态选择 comm CTA。中期决定上限的方案是把 64-vector communication publication tile 拆成 8/16-vector numerical reduction subtasks，再以 completion counter 恢复 64-vector ready publication。

完成这种解耦后，Mega 可以同时保留 persistent kernel 内 attention/communication overlap 的优势，并获得接近 FA3 dedicated combine 的全芯片 reduction 并行度。届时 auto split 才有条件像 vLLM/FA3 一样使用 16-22 的高 split 区间；在当前 combine 结构下，盲目增加 split 只会重复 case 007 的长尾。

## 16. History combine 优化实现（2026-08-06）

本节记录上述中期方案的第一版实际实现。它保留单个 persistent Mega kernel、现有 FA3 attention mainloop/epilogue 和公开 runner API，优先替换 history split reduction 热路径；本轮没有修改 auto split heuristic、`num_comm_sm` 策略或 final combine。

### 16.1 Metadata v3

metadata header 仍为 40 个 `int32`，版本从 v2 升到 v3。原 header 最后两个保留槽现在编码：

```text
history_combine_count
history_combine_offset
```

新增的 8-int `HistoryCombineWorkDesc` 为每个 `(dst_rank, vector)` 保存：

```text
publish_id, vector, history_head,
dependency_begin, dependency_count,
batch_idx, actual_splits, reserved
```

每个 vector 使用精确的 history attention completion IDs，不再让一个 64/128-vector publish tile 先等待所有 vector 的依赖。metadata validation 保证：

- `history_combine_count == dcp_size * total_vectors`；
- 每个 `(dst_rank, vector)` 恰好出现一次；
- `dependency_count == actual_splits`；
- descriptor 的 batch/head/publish tile 映射和 split 数一致；
- publish descriptor 的 `combine_task_count == valid_vectors`。

序列化顺序变为：

```text
attention -> q_tasks -> q_dependencies
-> publish -> history_combine -> publish_dependencies
-> final -> final_dependencies -> chunk_splits -> history_splits
```

runner metadata capacity 相应增加 `world_size * max_total_q * Hq_local * 8` 个 int，原 `publish_ready` workspace 不扩容，改为每个 publish tile 的完成计数器。

关键实现见 `dcp_mega_metadata.py`、`include/dcp_mega_min_fa3_varlen_params.h` 和 `min_fa3_dcp.py`。

### 16.2 Warp-per-vector combine

compute CTA 的 12 个 warps 现在独立消费 history combine 队列。一个 warp 完成一个 128-d vector：

1. lane 0 用全局 ticket counter 领取 descriptor，并 acquire-wait 该 vector 的精确 attention completion IDs；
2. `__syncwarp()` 将依赖可见性传递到本 warp；
3. 32 lanes 各负责连续 4 个维度；
4. 每个 lane 读取 `lane + 32*k` 对应的 LSE，warp shuffle reduction 得到 max/sum；
5. 归一化权重保存在寄存器中，按 split 广播；
6. FP32 partial-O 使用每 lane 16-byte `cp.async` 进入 warp 私有的 4-stage shared pipeline；
7. 采用 FA3 的 `accum += weight * partial` 形式，`weight == 0` 时跳过未定义 partial；
8. 直接写最终 BF16 IPC send O 和 FP32 LSE，不再构造整块 BF16 shared tile，也不再执行 producer-side TMA store。

`actual_splits == 1` 保留直接复制 attention BF16 O/LSE 的 fast path，不读取 FP32 partial workspace。

shared pipeline 使用：

```text
12 warps * 4 stages * 128 floats = 24 KiB
```

它覆盖原有 communication tile shared region，因此不增加 kernel 动态 shared-memory 上限。

split kernel 按 `history_num_splits` 选择编译期 32/64/128 bucket；nonsplit 使用 bucket 1。host binding 和 launch 同时检查 split 上界，异常 metadata 不会越过 bucket 的寄存器权重数组。

### 16.3 轻量级同步和 publication 顺序

history combine task loop 内没有 `__syncthreads()`。使用的同步只有：

- descriptor 依赖等待后的 `__syncwarp()`；
- O/LSE store 与 completion RMW 之间的 `__syncwarp()`；
- 所有 warps 退出 combine queue 后、进入现有 CTA-oriented final combine 前的一次 CTA barrier。

communication CTA 在本版本中只执行 receive，不再抢 history combine ticket，从而解除 receive 扫描与 producer combine 的 CTA-wide 控制流耦合。现有 receive/final-combine barrier 未在本轮改动。

publish counter 与远端 ready 的内存顺序为：

```text
warp generic global O/LSE stores
-> __syncwarp
-> atom.acq_rel.gpu.add(publish_ready)
-> last warp: fence.acq_rel.sys
-> st.release.sys(remote tile_ready phase)
-> peer ld.acquire.sys(tile_ready phase)
-> fence.proxy.async.global
-> peer TMA load
```

最后一个 completion RMW acquire 前面 warps 构成的 release sequence，再用 system fence 把该 tile 的普通 global stores 提升到 peer 可见范围。local final combine 则 acquire-wait：

```text
publish_ready[publish_id] >= publish.combine_task_count
```

对应实现位于 `include/dcp_mega_min_fa3_varlen_launch.h` 的 completion helper、`combine_history_splits` 和 `run_history_combine`。

### 16.4 已完成的静态验证

以下检查通过：

```bash
python -m unittest scripts.test_min_fa3.test_dcp_mega_metadata
python -m py_compile dcp_mega_metadata.py min_fa3_dcp.py \
  scripts/test_min_fa3/test_dcp_mega_metadata.py \
  scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py
git diff --check
make -j2
```

结果：

```text
metadata tests: 14/14 passed
extension build: passed, incremental rebuild reports no work
Python extension import: passed
```

纯 CPU case007 metadata 实例检查也通过：auto 得到 dispatch upper bound 29、实际 history splits `[1,22,2]`，split16 得到实际 `[1,16,2]`；两者均生成 1536 个逐-vector combine descriptors。对应 v3 image 分别为 26285 和 23021 个 int。

`cuobjdump` 对 DCP split kernels 的结果为：

```text
registers/thread: 168
LOCAL:            0
STACK:            120-136 bytes（按 DCP/CommHeads specialization）
32/64/128 bucket: register count 相同
```

SASS 中确认存在 128-bit `LDGSTS`（`cp.async`）、device-scope strong atomic、system/device membar 和 warp sync。源码检查确认 history combine task loop 内没有 CTA barrier。

### 16.5 GPU 验证状态

正确性 matrix 已扩展为六个 case，新增 case 007 auto 和 split16，并保留 eager、prepared replay、CUDA Graph、O/LSE reference 和 phase invariant 验证。测试入口还修复了直接执行脚本时 repo root 不在 `sys.path` 的问题。

本轮没有得到有效 GPU correctness 或 after-performance 数据：

- 第一次 matrix 尝试在 import 阶段失败，没有启动 kernel；修复了测试入口路径；
- 第二次尝试前 8 张 GPU 一度显示空闲，但随后另一组 root-owned `pretrain_gpt.py` 抢占了 Exclusive Process H100；rank 6/7 在 `torch.cuda.set_device()` 即返回 `cudaErrorDevicesUnavailable`，仍未启动 DCP kernel；
- 此后外部训练任务持续轮换占用 8 卡，因此按“静态检查优先、控制 GPU 测试次数”的约束停止重试。

因此，本文前述 `0.124864 ms`、`40 us history tail` 等均仍是验收目标，不是本实现的已测结果。GPU 空闲后应严格按以下顺序补测：

1. 一次六-case correctness matrix；
2. correctness 通过后，一次进程内 case007 split/comm sweep；
3. 最后一次 `simple_bench.sh` eager/graph batch；
4. 只有目标未达到时再使用 Nsight。

在这些数据完成前，不应宣称 Mega 已追平 vLLM A2A；当前可以确认的是设计已落地、ABI/构建/SASS 静态门禁通过，GPU correctness 和性能收益仍待空闲机器实测。

## 17. Metadata v4：tile-major 顺序与编译期 vectors/task（2026-08-06）

本节记录在 v3 warp-per-vector combine 之后实施的两个后续优化，以及同一台 8x H100 机器上的完整实测结果。v3 的 case007 收益已经成立，但 case004 从历史基线约 `0.506 ms` 回退到 `0.721 ms`；根因不是 history copy 本身变慢，而是逐-vector、destination-major 队列产生了 124,576 个 task，并让高 destination rank 的 publication 和 receive 严重滞后。

### 17.1 实现

metadata 从 v3 升级到 v4，header 仍保持 40 个 `int32`。`publish[]` 继续使用 rank-major ABI：

```text
publish_id = dst_rank * final_count + final_id
```

只有 `history_combine[]` 的展平顺序改为：

```text
for final_id:          # 16-token tile major
    for dst_rank:      # destination round-robin
        append all tasks belonging to publish[dst_rank, final_id]
```

这恢复了旧 publish ticket 映射的 tile-major/destination-minor 性质，同时不改变 receive 侧对 `publish_id` 的解释。

新的 8-int descriptor 为：

```text
publish_id
vector_begin
valid_vectors
dependency_begin
dependency_count
batch_idx
actual_splits
reserved
```

每个 task 的 vector 数由 kernel specialization 固定：

```cpp
static constexpr int kHistoryVectorsPerTask = Split ? 1 : CommHeads;
```

对应映射为：

```text
split kernel:        1 vector/task
non-split Hq_local4: 4 vectors/task，即一个 token
non-split Hq_local8: 8 vectors/task，即一个 token
```

non-split task 的 dependencies 是该 token 所有 local heads 所需 history completion IDs 的排序去重并集。warp 只等待一次 dependencies，编译期展开复制 4/8 个 vector，最后只做一次 `publish_ready` completion atomic。split 的 FA3-style reduction 仍保持一个 vector/task，避免把多个 split reduction 串在同一 ticket 中。

同步协议没有加重：history task loop 中仍只有依赖等待后的 `__syncwarp()` 和 completion 前的 `__syncwarp()`，没有新增 `__syncthreads()`。原 device-scope acq_rel atomic、system fence 和 remote system release 保持不变。

### 17.2 静态与正确性验证

以下静态检查全部通过：

```text
metadata unit tests: 16/16
python py_compile:    passed
git diff --check:     passed
full make -j2:        passed
extension import:     passed
```

单元测试覆盖 Hq_local 4/8、DCP 2/4/8、BlockN 128/176、ragged batch 和 tail tile，并显式验证：

- split descriptor 始终为 1 vector/task；
- non-split descriptor 始终为 Hq_local vectors/task；
- 一个完整 non-split publish tile 恰好有 16 个 combine tasks；
- 展开后的 `(dst_rank, vector)` 坐标严格为 tile-major/destination-minor；
- grouped dependency 是覆盖 task 内所有 heads 的精确去重并集；
- v4 offset、count、capacity 和 40-int header 连续且一致。

`cuobjdump` 结果：

```text
split:     REG=168, LOCAL=0, STACK=120-136 B
non-split: REG=168, LOCAL=0, STACK=88-128 B
```

因此本轮没有新增 cubin local allocation。源码检查确认 history hot loop 内仍无 CTA barrier。

8 卡在 `Exclusive_Process` 下运行一次完整 correctness matrix，最终 run 目录为：

```text
benchmark_logs/dcp_mega_matrix_when_idle/20260806-235019
```

六个配置全部通过，包含 eager、prepared replay、CUDA Graph、O/LSE reference 和 phase invariant：

```text
dcp2_h4_bn128_split1
dcp4_h8_bn176_split2
dcp8_h4_bn176_auto
dcp2_h8_bn128_split2_tail
case007_dcp8_h4_bn128_auto
case007_dcp8_h4_bn128_split16
```

### 17.3 完整 benchmark

实测目录：

```text
benchmark_logs/dcp_mega_trace_20260806-235432
```

配置与前测一致：10 个确定性 trace cases、eager+graph、warmup 40、iters 60、`CHECK=0`，并开启 Mega 与 baseline phase timing。完整脚本 exit code 为 0。

Mega eager p50 与 v3 逐-vector版本 `20260806-225429` 的逐 case 对比：

| Case | v3 p50 (ms) | v4 p50 (ms) | 变化 |
|---|---:|---:|---:|
| 000 | 0.086976 | 0.087616 | +0.74% |
| 001 | 0.099520 | 0.100544 | +1.03% |
| 002 | 0.091008 | 0.092192 | +1.30% |
| 003 | 0.099296 | 0.099936 | +0.64% |
| 004 | 0.721344 | 0.488624 | -32.26% |
| 005 | 0.092256 | 0.093808 | +1.68% |
| 006 | 0.083856 | 0.086080 | +2.65% |
| 007 | 0.084688 | 0.086032 | +1.59% |
| 008 | 0.092512 | 0.092800 | +0.31% |
| 009 | 0.099984 | 0.098064 | -1.92% |

10-case Mega arithmetic mean 为 `0.132570 ms`，优于验收线 `0.155144 ms`。除目标 case004 外，其余 case 的 p50 最大回退为 case006 的 `2.65%`，低于 3%。case008 的 p90 从 `0.095386` 增至 `0.107200 ms`，虽然 p50 只增加 0.31%，仍应记录为一次尾延迟噪声/残余风险；按控制测试次数的约束，本轮没有为这个单独重复整套 GPU benchmark。

### 17.4 Case004

case004 是 non-split、`total_q=3893`、`Hq_local=4`、DCP=8。task 数变化完全符合设计：

```text
v3: 3893 * 4 * 8 = 124576 tasks
v4: 3893 * 8     =  31144 tasks
```

关键结果：

| 指标 | v3 | v4 | 变化 |
|---|---:|---:|---:|
| Mega p50 | 0.721344 ms | 0.488624 ms | -32.26% |
| Mega p90 | 0.728416 ms | 0.501664 ms | -31.12% |
| rank p50 max-min | 198.128 us | 16.144 us | -91.85% |
| attention_done | 157.296 us | 156.368 us | 基本不变 |
| history_combine_done | 444.672 us | 305.600 us | -139.072 us |
| receive_done | 692.944 us | 470.208 us | -222.736 us |
| final_combine_done | 708.896 us | 476.368 us | -232.528 us |

这证明两个优化分别消除了主要开销：

1. 4 vectors/task 把 ticket、dependency wait、descriptor load、completion atomic 缩减为原来的 1/4，直接提前 history completion。
2. tile-major/destination-minor 顺序把 rank p50 spread 从约 198 us 压到 16 us，消除了 destination-major 队列造成的 rank7 starvation。

case004 的总时延已经优于历史未回退水平约 `0.506 ms`，但严格的 `publish_done -> receive_done <= 40 us` 目标没有达到，当前仍为 `167.392 us`。这里不能再归因于 destination starvation：rank spread 已经只有 16 us。剩余尾段来自长 Q case 中 receive-only communication CTAs 对大量 tile 的服务进度，以及 phase 指标使用“每轮先跨 rank 取 max，再取 percentile”的全局完成语义。下一步若继续优化，应针对 receive readiness/scan 调度，而不是重新增大 combine task 粒度或加入 CTA-wide barrier。

### 17.5 Case007

case007 是 split kernel，因此仍保持 1536 个逐-vector tasks，本轮主要只受到队列顺序变化影响：

| 指标 | v3 | v4 | 验收 |
|---|---:|---:|---:|
| p50 | 0.084688 ms | 0.086032 ms | <= 0.087229 ms，通过 |
| p90 | 0.086022 ms | 0.087395 ms | <= 0.090000 ms，通过 |
| attention -> history | 19.360 us | 19.840 us | <= 22 us，通过 |
| rank p50 max-min | 3.488 us | 4.992 us | 保持很小 |

因此保守策略 `Split ? 1 : CommHeads` 是必要的：它在 non-split case004 上取得 4 倍 task 压缩，同时没有把多个昂贵的 split reductions 串进 case007 的单个 task，保住了之前 FA3-style combine 的性能。

### 17.6 结论和后续边界

本轮计划中的两个修改已经实现并通过静态、8 卡 correctness 和完整性能验证。主要目标均满足：case004 总时延和 rank 均衡恢复，case007 p50/p90/history tail 保持，10-case mean 改善且其他 case p50 回退不超过 3%。

如果继续优化 case004，范围应限制在 receive 调度：优先研究 ready receive 的低成本发现方式、减少无效 scan，以及在不引入 `__syncthreads()` 热循环同步的前提下改善 receive CTA 的 tile 服务顺序。当前不建议改变 split combine 数值路径、remote release protocol 或再次扩大 non-split task；这些部分已经由本轮 correctness 和性能数据验证。

## 18. v5：receive warp-pair 独立推进与 bounded scan

本节实现上一节确定的 receive 第一阶段方案。修改范围刻意限制在 communication CTA 的 post-Q receive 调度；history combine 仍只由 compute CTA 执行，attention/history queue、metadata ABI、split policy、publish 顺序、`num_comm_sm` 和 Python API 均未改变。

### 18.1 实现

旧实现让六组 producer/consumer warp pair 经过 CTA-wide scan、选择和传输阶段。即使某个 pair 已有 ready tile，它也可能被另一个没有 ready task 的 pair 拖住；case004 每个静态 slot 又有 35/36 个 task，lane 0 在 ready 稀疏时会反复执行大量 `receive_ready` 和 remote `tile_ready` acquire load。

新实现保留 `6 * num_comm_sm` 个静态 strided receive slot，但将每个 slot 改成独立流水线：

```text
producer warp:
  wait finished
  -> scan at most 8 circular candidates
  -> publish task ID and source LSE in shared memory
  -> arrive.release work_ready
  -> peer TMA load, completion on arrived

consumer warp:
  wait work_ready
  -> wait arrived
  -> write LSE and issue local TMA store
  -> wait for store completion
  -> release receive_ready
  -> arrive finished
```

关键实现点：

1. `kReceiveScanWindow=8` 是编译常量。`task_count > 1` 时每轮最多探测 8 个候选，miss 后循环 cursor 前进并短暂 `__nanosleep(64)`；连续窗口仍会覆盖完整 task ring。
2. `task_count == 1` 直接等待唯一 remote `tile_ready`，避免 case007 的 21 个单-task slot 进入通用 scan。
3. 六组 pair 各自使用 `work_ready`、`arrived`、`finished` mbarrier，不再用 CTA-wide 状态聚合。hot loop 只使用 warp shuffle、`__syncwarp()` 和 mbarrier。
4. `finished` 只在 local TMA store 完成且 `receive_ready=1` release 之后发出，防止 producer cursor 环回时重新选择仍在 flight 的 task。
5. 函数中只保留初始化后和 receive phase 结束前两个 `__syncthreads()`；scan/选择/TMA 的 hot loop 没有 CTA-wide barrier。

跨 GPU 可见性顺序保持为：

```text
source O/LSE stores
-> atom.acq_rel.gpu publish completion
-> fence.acq_rel.sys
-> st.release.sys remote tile_ready
-> ld.acquire.sys remote tile_ready
-> fence.proxy.async.global
-> peer TMA load
-> local TMA store completion
-> fence.proxy.async.global
-> store_release receive_ready
-> final combine acquire wait
```

### 18.2 静态验证

CPU receive scheduler model 覆盖以下情况：

- DCP 2/4/8 和 `num_comm_sm` 1/4/8；
- empty、single-task、short、tail 和较长 task list；
- forward、reverse 和 interleaved readiness order；
- case007 的 21 tasks / 48 slots，以及 case004 的 1708 tasks / 48 slots；
- exactly-once completion、cursor wraparound 和每个 scan window 不超过 8 probes。

静态测试与编译结果：

```text
python -m unittest scripts/test_min_fa3/test_dcp_mega_metadata.py
17/17 passed

python -m py_compile dcp_mega_metadata.py min_fa3_dcp.py \
  scripts/test_min_fa3/test_dcp_mega_metadata.py \
  scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py
passed

git diff --check
passed

make -j2
passed

python -c "import torch; import _min_fa3_op"
passed
```

最终 cubin 资源与 v4 相同，所有 DCP Mega specialization 均为 `REG=168, LOCAL=0`；non-split stack 为 88-128 B，split stack 为 120-136 B，仍保持每个 SM 一个 384-thread CTA 的 residency。

### 18.3 8 卡 correctness

测试在 8x H100、`Exclusive_Process` 下运行。首次 idle-poll launch 在最终空闲检查和 `torch.cuda.set_device()` 之间被外部任务抢占，因此未启动 DCP kernel；重试目录为：

```text
benchmark_logs/dcp_mega_matrix_when_idle/20260807-receive-warp-pairs-retry1
```

重试 exit code 为 0，六个配置全部通过 eager、prepared replay、CUDA Graph、O/LSE reference 和 phase invariant：

```text
dcp2_h4_bn128_split1
dcp4_h8_bn176_split2
dcp8_h4_bn176_auto
dcp2_h8_bn128_split2_tail
case007_dcp8_h4_bn128_auto
case007_dcp8_h4_bn128_split16
```

### 18.4 定向 smoke

为控制 GPU 测试次数，先在单个 `torchrun` 中连续测 case004/case007，仅运行 Mega eager、`warmup=10,iters=20`、`num_comm_sm=8`、auto split 和 phase timestamps。结果目录：

```text
benchmark_logs/dcp_mega_receive_smoke_20260807-1139
```

结果如下：

| Case | p50 | p90 | attention -> history | publish -> receive |
|---|---:|---:|---:|---:|
| 004 | 0.484 ms | 0.493 ms | 150.512 us | 141.856 us |
| 007 | 0.084 ms | 0.087 ms | 20.112 us | 10.640 us |

case004 的方向性指标相对 v4 长测 `167.392 us` 已缩短 25.536 us，因此才继续执行一次完整 benchmark。临时 smoke config 在测试后删除，结果 JSON 保留。

### 18.5 完整 10-case benchmark

完整实测目录：

```text
benchmark_logs/dcp_mega_trace_20260807-114034
```

配置与 v4 一致：相同确定性 10-case trace、eager+graph、`warmup=40,iters=60`、`CHECK=0`、`num_comm_sm=8`、auto split，并开启 Mega 和 baseline phase timing。`simple_bench.sh` exit code 为 0。

Mega eager p50 对比：

| Case | v4 p50 (ms) | v5 p50 (ms) | 变化 |
|---|---:|---:|---:|
| 000 | 0.087616 | 0.086672 | -1.08% |
| 001 | 0.100544 | 0.097488 | -3.04% |
| 002 | 0.092192 | 0.088832 | -3.64% |
| 003 | 0.099936 | 0.097344 | -2.59% |
| 004 | 0.488624 | 0.480992 | -1.56% |
| 005 | 0.093808 | 0.091728 | -2.22% |
| 006 | 0.086080 | 0.084112 | -2.29% |
| 007 | 0.086032 | 0.084416 | -1.88% |
| 008 | 0.092800 | 0.089696 | -3.34% |
| 009 | 0.098064 | 0.096048 | -2.06% |

10-case arithmetic mean 从 `0.132570 ms` 降到 `0.129733 ms`，改善 `2.14%`。所有 case 的 p50 都改善，因而满足“其他 individual p50 regression <= 3%”和 mean regression 验收条件。

case004 的 phase 对比：

| 指标 | v4 | v5 | 变化 |
|---|---:|---:|---:|
| Mega p50 | 0.488624 ms | 0.480992 ms | -1.56% |
| Mega p90 | 0.501664 ms | 0.487690 ms | -2.79% |
| rank p50 max-min | 16.144 us | 10.176 us | -36.97% |
| q_allgather_done | 143.152 us | 143.536 us | +0.384 us |
| attention_done | 156.368 us | 156.688 us | +0.320 us |
| history_combine_done | 305.600 us | 305.984 us | +0.384 us |
| receive_done | 470.208 us | 442.960 us | -27.248 us |
| final_combine_done | 476.368 us | 468.352 us | -8.016 us |
| publish -> receive | 167.392 us | 140.384 us | -27.008 us / -16.13% |

Q all-gather、attention 和 history combine 均只变化约 0.4 us，而 receive milestone 提前 27.248 us。这把收益定位到 receive scheduler，排除了 combine 算术或 attention workload 波动。`publish -> receive=140.384 us` 达到本阶段 `<=142.3 us` 的验收线，但距离最初理想的 40 us 仍有明显差距。

case007 的 p50 从 `0.086032` 降到 `0.084416 ms`，`attention -> history` 从 `19.840` 降到 `19.712 us`，`publish -> receive` 从 `12.640` 降到 `10.896 us`，证明 receive 修改没有破坏 split combine 主路径。p90 从 `0.087395` 增至 `0.090266 ms`，比 `0.090000 ms` 门槛高 0.266 us；同时 p50、phase p50 和 rank spread 均改善，故记录为本轮唯一尾延迟残余风险。按照“静态测试优先、控制 GPU 测试次数”的约束，没有为这 0.266 us 单独重跑完整矩阵。

### 18.6 结论与下一步

v5 已解决第一阶段的两个结构性问题：六组 warp pair 不再被最慢 chunk CTA-wide 锁步，ready discovery 也不再每次扫描 slot 的全部 35/36 个 task。实现保留了 system-scope release/acquire 顺序，hot loop 没有 `__syncthreads()`，并在完整 trace 上获得全 case p50 改善。

剩余问题是静态 slot ownership 本身：ready 到达顺序仍不一定匹配静态 strided 子集，某个 pair 空转时不能领取其他 pair 的 ready task。下一阶段应优先评估轻量 bitmap 或分层 ready queue，并允许 pair 级 work stealing；仍需避免在 hot loop 引入 CTA-wide barrier，且必须保留 exactly-once receive ownership 和 graph phase monotonicity。

communication CTA 恢复 history-combine assist 仍是独立优化方向。本轮保持 compute-only combine，因此 first combine/publish 只能在 compute CTA 退出全局 attention 领取循环后开始；它可以与最后一波尚在执行的 attention 以及后续 O receive 重叠，但不能恢复旧版本 Q all-gather 完成后立即 opportunistic combine 的更早重叠。该方向不应与 receive scheduler 修改混在同一次变更中。

## 19. v6 候选设计：per-m-block split 完成计数与 ready-driven history combine

本节记录一个尚未实现的后续优化设计：对同一个 history attention tile 的所有 split 使用完成计数；最后一个 split 完成时立即发布 ready group，使已经退出 attention pipeline 的 compute warp 可以提前执行该 group 的 history combine，继而提前 publish remote tile 并允许 communication CTA receive。

该思路方向正确，但原始表述中的两个粒度需要修正：

1. 完成计数必须属于 `(batch_idx, history_m_block)`，不能只属于整个 sequence。
2. 最后完成 split 的 CTA 应作为 ready group 的发布者，并在退出 attention pipeline 后优先参与 combine；不应在 FA3 attention pipeline 中途独占并执行整个 group 的 combine。

推荐的数据流如下：

```text
history split epilogue store
  -> fence.proxy.async.global
  -> atom.acq_rel.gpu completion_count[group_id]++
  -> last split publishes group_id READY
  -> drained compute warps claim ready vectors
  -> warp-per-vector history combine
  -> publish_ready[publish_id]++
  -> last vector in publish tile releases remote tile_ready
  -> communication CTA receives O/LSE
```

### 19.1 为什么不能只设置 sequence 完成计数

当前 history attention descriptor 的基本坐标是：

```text
(batch_idx, m_block, split_idx)
```

一个 split descriptor 只生成一个 128-row packed attention tile 的 partial O/LSE，不覆盖整个 sequence。case007 中每个 sequence 的形状为：

```text
Q length                         = 16
history physical heads          = DCP 8 * Hq_local 4 = 32
packed rows per sequence        = 16 * 32 = 512
history m-blocks per sequence   = 512 / 128 = 4
actual history splits           = [1, 22, 2]
```

因此 case007 需要 12 个独立的 history group counter：

```text
sequence 0: 4 counters, target = 1
sequence 1: 4 counters, target = 22
sequence 2: 4 counters, target = 2
```

如果 sequence 1 只使用一个 target 为 22 的 counter，它实际会收到 `4 * 22 = 88` 个 descriptor completion。计数达到 22 只能证明任意 22 个 tile-split 完成，不能证明某个 m-block 的全部 22 个 split 已完成；此时 combine 可能读取尚未写入的 partial O/LSE。

若把该 sequence counter 的 target 改成 88，虽然正确，但必须等待整个 sequence 的全部 m-block 完成，失去按 m-block 提前 combine 的价值。

当前 metadata 已经保留了正确的逻辑分组：`_append_attention_domain()` 为同一个 m-block 生成连续 split completion IDs，`completion_for_vector` 将属于该 m-block 的完整 completion tuple 关联到其 128 个 packed vector。case007 当前共有 12 个不同的 history dependency set，因此可以在不改变 attention tile 定义的前提下生成稳定的 `history_group_id`。

### 19.2 当前已经存在的 overlap

新方案不是第一次让 attention 和 history combine 重叠。当前 compute CTA 在自己的 attention queue 耗尽后便独立执行：

```text
attention_kernel
  -> CTA-local __syncthreads
  -> run_history_combine
```

这里没有 compute grid-wide barrier。`attention_done` phase timestamp 只表示最后一个 compute CTA 已退出 attention，并不表示此前没有 CTA 或 warp 在处理 history combine。

case007 当前 metadata 为：

```text
chunk attention descriptors     = 3
history attention descriptors   = 100
total attention descriptors     = 103
compute CTAs                    = 132 - 8 = 124
```

因此有 21 个 compute CTA 没有有效 initial attention descriptor，它们会很早进入 `run_history_combine()`。现有实现已经可能让 first combine 与仍在执行的 attention 重叠。

当前限制来自 ready discovery 和队列行为：

1. 每个 split history vector task 会重新遍历其 `dependency_count` 个 completion IDs；case007 长 sequence 的相同 22 个 acquire loads 会被大量 vector task 重复执行。
2. `history_combine[]` 使用全局静态 ticket。warp 领取未 ready task 后会在该 task 上等待，不能转去处理另一个已经 ready 的 group。
3. 最后一个 split 完成者只 release 一个独立 completion flag，不会直接发布“该 m-block 的 128 个 vector 已可 combine”这一调度事件。

所以 v6 的主要收益是把现有的被动 dependency polling 改为 ready-driven scheduling，并减少 head-of-line blocking，而不是简单地在 attention 后增加一个新阶段。

### 19.3 case007 的机会与上限

v5 case007 实测为：

```text
q_allgather_done          5.008 us
attention_done           16.512 us
history_combine_done     35.712 us
receive_done             44.800 us
final_combine_done       72.608 us

attention -> history     19.712 us
publish -> receive       10.896 us
kernel p50               84.416 us
```

因此 history-combine 关键路径最多有约 `19.7 us` 可以尝试隐藏，但实际收益必然小于这一理论上限：

- 一个 remote publish tile 包含 `16 * Hq_local = 64` 个 vector。
- case007 的一个 16-token sequence 对每个 destination 的 64 个 vector 依赖该 sequence 的 4 个 history m-block。
- 单个 m-block ready 后可以提前完成其中一部分 vector reduction，但只有对应 publish tile 的全部 64 个 vector 完成后，现有 `publish_ready` 才会 release remote `tile_ready`。
- current final combine 仍在所有 compute CTA 完成 `run_history_combine()` 后开始，v6 第一阶段不会改变这个全局阶段边界。

不过 `[split=1]` 和 `[split=2]` 的两个短 sequence 有机会在长 sequence 的 `[split=22]` attention 尚未完成时完成全部 4 个 group，进而提前 publish 和 receive。长 sequence 内部的 4 个 m-block 也可以按实际完成顺序提前 reduction，减少 attention 结束后的剩余 combine 工作。

### 19.4 为什么不在 attention epilogue 中直接做完整 combine

FA3 producer/consumer pipeline 会提前领取并预取后继 attention work：

- producer warp group 在当前 task 的 load 路径中调用 `prefetch_next_work()`；
- consumer warp group 在当前 task epilogue 前调用 `get_next_work()`；
- 下一项 K/V 或 scheduler hand-off 可能已经处于 flight 状态。

如果最后完成 split 的 consumer warp group 在 epilogue completion 后直接跳入完整 history combine，会产生以下问题：

1. producer 和 consumer 可能在 named barrier 或 pipeline barrier 上失去配对。
2. 后继 attention work 已经占用 pipeline stage，插入 combine 会拉长或破坏 pipeline 生命周期。
3. `HelperSharedStorage::comm_tiles` 与 attention dynamic shared storage 复用同一块内存；当前 FA pipeline active 时，combine 的 `cp.async` staging 会覆盖 attention shared data。
4. 只有 consumer warp group观察到 epilogue completion；强制整 CTA 转入 combine 需要额外的 CTA-wide 协调，违背 hot loop 使用轻量同步的约束。

因此完整 combine 必须发生在该 CTA 的 attention kernel 已经安全返回、现有 CTA-local `__syncthreads()` 完成之后。最后完成者如果仍有后继 attention work，只发布 READY 并继续 FA3 pipeline；如果它恰好已经耗尽 attention work，则退出 pipeline 后自然成为 ready group 的第一批消费者。

### 19.5 为什么不能让最后一个 CTA 独占 ready group

一个 history m-block 最多解锁 128 个 vector combine task。当前高性能路径为一个 warp 处理一个 128-d vector；一个 12-warp CTA 独自处理 128 个 vector 需要约 11 轮。

若每个 group 固定由最后一个 split CTA 独占：

- case007 总共只有 12 个 history group；
- 长 sequence 只有 4 个 group；
- 长 sequence reduction 峰值可能只有 4 个 CTA；
- H100 上大量已经退出 attention 的 compute CTA 无法协助。

这会丢掉 v3 之后建立的 warp-per-vector 全芯片并行度。因此“最后完成者执行”应解释为“最后完成者发布并参与”，而不是“最后完成者拥有整个 group”。同一个 ready group 应允许多个 compute warp/CTA 通过 group-local vector cursor 协作领取任务。

### 19.6 推荐调度状态与轻量同步

每个 history m-block group 建议至少维护：

```text
completion_count[group_id]   # 已完成 split 数
next_vector[group_id]        # 下一个待领取 vector ordinal
completed_vectors[group_id]  # 已完成 combine vector 数
ready[group_id]              # 或 ready bitmap
```

全局阶段状态建议维护：

```text
completed_group_count
history_group_count
```

执行协议：

1. history attention epilogue 完成后，单个 leader 执行 device-scope acq_rel atomic add。
2. `old == actual_splits - 1` 的最后完成者 release-publish group ready。
3. 已经退出 attention 的 combine warp acquire 检查 ready group，并通过 `atomicAdd(next_vector, 1)` 领取一个 vector。
4. 领取成功的 warp沿用当前 FA3-style `combine_history_splits()`；不同 CTA 可以协作处理同一个 group。
5. 每个 vector 完成后沿用当前 `publish_ready[publish_id]` completion protocol，最后一个 vector继续执行 system fence 和 remote `tile_ready` release。
6. group 的最后一个 vector 更新 `completed_group_count`；空闲 warp 只有在全部 group 完成后退出 history 阶段。
7. 暂时没有 ready group 但仍有未完成 group 时使用短 `__nanosleep(64)`，不能提前退出。

hot loop 不需要新增 `__syncthreads()`。同步应限制为 warp-level `__syncwarp()`、device-scope atomic/release/acquire，以及现有 attention-to-helper-shared 生命周期边界上的单次 CTA barrier。

ready queue 若采用多 producer FIFO，需要处理 tail reservation 先于 slot write 的可见性，不能仅依赖一个 tail counter。第一版更适合使用静态 group state、ready bitmap 和 group-local cursor，避免引入复杂的 MPMC slot sequence protocol。group 数较小时可以 bounded round-robin 扫描；group 数较大时再考虑分层 bitmap。

### 19.7 内存序要求

推荐保留以下完整可见性链：

```text
partial O/LSE epilogue stores
  -> fence.proxy.async.global
  -> atom.acq_rel.gpu completion_count++
  -> last completion release-publishes READY
  -> combine warp acquire-observes READY
  -> cp.async loads partial O and generic loads partial LSE
  -> history_send O/LSE stores
  -> atom.acq_rel.gpu publish_ready++
  -> last publish task fence.acq_rel.sys
  -> st.release.sys remote tile_ready
  -> peer ld.acquire.sys remote tile_ready
  -> peer TMA load and local TMA store
  -> release receive_ready
```

device scope 足以保护本 GPU 的 partial O/LSE 和 ready group；只有向 peer GPU 发布 communication tile 时需要现有 system-scope fence/release。不能用 relaxed atomic 代替最后 split 的 acq_rel RMW，否则最后完成者不一定获得其他 split epilogue stores 的传递可见性。

### 19.8 workspace、graph replay 与终止条件

现有 `attention_done` workspace 按 attention descriptor 数分配并在 eager/prepared replay/CUDA Graph replay 前清零。v6 可以评估两种布局：

1. 显式增加 history group state workspace，接口和语义最清楚。
2. 在 split specialization 中重新布局 `attention_done`，复用不再需要的逐-split binary flag 空间。

第二种方案不能仅凭 case007 容量充足就直接采用。若 workload 包含大量 split=1 group 和少量 split>1 group，`counter + cursor + ready state` 总量可能超过原始逐-descriptor flag 数，因此必须先对全支持范围证明容量上界。若不能静态证明，优先使用显式 workspace，避免脆弱的隐式 alias。

CUDA Graph replay 必须重置所有 group counter、cursor、ready state 和 completed count；remote `tile_ready` 继续使用现有单调 phase，不能改回零值复用。history worker 的退出条件必须是：

```text
completed_group_count == history_group_count
```

不能使用“当前没有 ready group”或“所有 vector 已被领取”作为退出条件，因为仍在运行的 attention CTA 可能随后发布新 group，且已领取 vector 可能尚未完成 history stores 和 remote publication。

### 19.9 建议实施顺序与验收

为隔离风险并控制 GPU 测试次数，建议分两阶段实现。

第一阶段只引入正确的 per-m-block completion counter 和 ready group 调度，保留：

- 当前 attention descriptor 顺序；
- warp-per-vector combine 数值路径；
- tile-major、destination round-robin publication 语义；
- communication CTA receive-only 分工；
- final combine 阶段边界；
- system-scope remote publication protocol。

第二阶段仅在第一阶段证明收益后，再评估 final combine 是否也能按 final tile readiness 提前启动，或 communication CTA 是否恢复受限的 history-combine assist。不能把这两个调度变化和 per-m-block counter 混在同一次修改中。

静态验证应先覆盖：

1. DCP 2/4/8、Hq_local 4/8、split 1/2/多 split。
2. 每个 group 的 completion 数严格等于 `actual_splits`。
3. 每个 history vector exactly once，并映射到正确的 publish ID。
4. ready group 的提前到达、逆序到达和稀疏到达。
5. worker 在暂时 empty 时不提前退出。
6. graph replay reset 和 phase monotonicity。
7. split kernel 的 register、local memory 和 stack 不恶化到降低 occupancy。

GPU 验证仍遵循先 correctness matrix、后一次定向 case007 benchmark、最后仅在达到方向性门槛时运行完整 10-case benchmark。关键观测指标为：

```text
attention_done
history_combine_done
first/last remote tile_ready（可选调试 instrumentation）
publish_done -> receive_done
final_combine_done
kernel p50/p90
rank p50 spread
```

### 19.10 设计结论

per-m-block split completion counter 值得作为下一项 history-combine 优化。它能把重复的逐-split dependency polling 转换成一次明确的 ready event，并允许短 sequence 和先完成的长 sequence m-block 提前 reduction、publish 和 receive。

推荐方案不是在 FA3 epilogue 中强行插入完整 combine，而是：

> 最后一个 split CTA 以 acq_rel counter 确认 group 完成并 release-publish READY；所有已经退出 attention pipeline 的 compute warp，包括最后完成者退出后的 warp，通过 group-local cursor 协作执行该 group 的 vector combine。

该方案兼顾早期 overlap、warp-per-vector 并行度、copied-and-trimmed FA3 pipeline 生命周期和轻量同步要求。它仍不能单独消除 final combine 的全局阶段边界，也不能保证隐藏全部 `19.712 us` history tail；这两点应通过第一阶段实测后再决定是否继续扩展。

## 20. Metadata v5：自适应 history copy 宏任务（2026-08-07）

本节记录针对新 trace 大负载实施的 history copy 优化。它解决的是
`actual_splits == 1` 的 O/LSE transpose-pack，不替代上一节面向真实 split
reduction 的 ready-group 候选设计。

### 20.1 新 trace 上的瓶颈

`simple_bench.sh` 的 10-case eager timestamp 显示，大 non-split case 的
`attention_done -> history_combine_done` 达到 `331-497 us`。该 tail 与
`total_q` 的 Pearson 相关系数为 `0.998`，与估算 O/LSE 读写量的相关系数为
`0.999`；大 case 的有效读写带宽只有约 `424-434 GB/s`。

history combine 除数值 reduction 外还负责 token-major history O/LSE 到
destination-major IPC send layout 的搬运。v4 的一个 token 一个任务在原负载上
有效，但新 trace 的 8K-13K query token 仍产生 69K-103K 个任务。每个任务都
需要 queue ticket atomic、descriptor/dependency 读取和 publish completion atomic。

另一个问题是粒度由全局 `dispatch.split` 控制。case000001 的 chunk domain
需要 split，而 history sequence 的 `actual_splits` 全为 1；旧实现仍为 history
copy 生成 46,976 个逐-vector任务。

### 20.2 自适应选择

v5 只对 `actual_splits == 1` 合并连续 copy，真实 reduction 始终保持一个
vector 一个任务。copy 候选为：

```text
Hq_local=4: 1, 4, 8, 16, 32 vectors/task
Hq_local=8: 1, 8, 16, 32 vectors/task
```

最大任务覆盖 32 vectors，即 Hq_local4 的 8 tokens 或 Hq_local8 的 4 tokens。
Host metadata 用以下模型从大到小选择候选：

```text
worker_warps = (num_sms - num_comm_sm) * 12
target_claims = ceil(0.8 * worker_warps)

total_claims(g) = real_split_vector_tasks
                + copy_tasks_after_tile_and_sequence_clipping(g)
```

选择满足 `total_claims(g) >= target_claims` 的最大 `g`；如果逐-vector任务也
不足目标波次，则使用 `g=1` 提供最大可用并行度。H100、`num_comm_sm=8` 时
worker 数为 1,488 warps，目标为 1,191 tasks。

粒度绑定每个 descriptor 的 `actual_splits`，不再绑定全局 kernel split
specialization。因此同一个 split kernel 内可以同时执行 32-vector普通 copy 和
one-vector真实 split reduction。

### 20.3 对齐与边界

trace schema v2 将物理 `q_lens` 对齐到至少 8 且为 8 的倍数，同时保留
`logical_q_lens` 推进 replay 状态。16-token publish tile 内恰有两个 8-token
region，宏任务不会跨 sequence 或 publish 边界。

CUDA API 不把对齐作为正确性前提。metadata builder 仍先按 publish tile 和
sequence region 截断，再按选定粒度分组；非对齐输入会产生较小尾任务。每个
descriptor 只属于一个 `batch_idx`、`actual_splits` 和 `publish_id`。

宏任务 dependency 是所覆盖 vector completion IDs 的精确去重并集。
`publish.combine_task_count` 记录实际宏描述符数，因此每个宏任务仍只执行一次
completion atomic。最后任务的 device acq_rel RMW、system fence 和 remote
release store 完全保留；hot loop 没有新增 CTA barrier。

Metadata image 升级到 v5，但 40-int header 和 `HistoryCombineWorkDesc` 的
8-int 布局不变。dispatch/benchmark JSON 新增
`history_copy_vectors_per_task`，queue profile 新增 worker warp 和 task wave 数。

### 20.4 验证结果

CPU suite 覆盖 DCP 2/4/8、Hq_local 4/8、五档 copy 粒度、non-split、真实
split、chunk split/history copy、mixed split 和非对齐 batch fallback，共 31 项
相关测试通过。SM90 扩展成功编译；non-split 32-vector copy 与同批 mixed
copy/reduction 均在 8x H100 上通过 full-KV correctness。

对齐后的确定性 10-case trace 使用 eager Mega、`warmup=10,iters=20`、auto
split、`num_comm_sm=8` 和 phase timestamps。与 2026-08-07 修改前诊断 run
对比如下：

| Case | Tasks old -> v5 | Copy vectors/task | History tail old -> v5 | 变化 |
| --- | ---: | ---: | ---: | ---: |
| 000000 | 101,248 -> 12,656 | 32 | 489.232 -> 306.400 us | -37.4% |
| 000001 | 46,976 -> 1,472 | 32 | 111.392 -> 28.656 us | -74.3% |
| 000002 | 14,336 -> 4,416 | 32 | 33.120 -> 28.928 us | -12.7% |
| 000003 | 21,504 -> 5,136 | 32 | 54.768 -> 37.152 us | -32.2% |
| 000004 | 16,992 -> 4,752 | 32 | 42.176 -> 30.048 us | -28.8% |
| 000005 | 102,912 -> 12,864 | 32 | 496.800 -> 313.360 us | -36.9% |
| 000006 | 70,912 -> 8,864 | 32 | 348.032 -> 219.184 us | -37.0% |
| 000007 | 17,920 -> 6,016 | 32 | 43.504 -> 34.256 us | -21.3% |
| 000008 | 43,440 -> 5,432 | 32 | 210.560 -> 134.688 us | -36.0% |
| 000009 | 69,248 -> 8,656 | 32 | 331.616 -> 210.128 us | -36.6% |

所有 case 的 history tail 均下降。case000000/000005/000006/000009 达到大负载
至少 30% 的方向性门槛；case000001 证明按 `actual_splits` 解耦修复了全局 split
导致的伪细粒度 copy。剩余大 case tail 仍包含不可消除的约 210 MB O/LSE 搬运，
下一步应先用 hardware counter 区分 HBM 带宽与 receive backpressure，不应重新
把真实 split reduction 合并进一个 warp task。

## 21. Metadata v7：自适应 final combine 与 pull/final 同序（2026-08-11）

本节记录 DCP8 decode-only final combine 的实际实现和 100-case 验证。它建立在
前述 history combine、receive bounded scan 和自适应 copy task 之上，不改变
16-token IPC communication tile、remote ready phase 或 FA3 attention mainloop。

### 21.1 原始问题

arrival rate 1 的 DCP8 decode-only trace 中，原始 Mega 的
`receive_done -> final_combine_done` 平均约为 `27.590 us`，而整个 kernel body
平均约为 `87.675 us`；对应 CUDA event latency 为 `99.304 us`。final 阶段已经
接近 body 的三分之一。

同一原始 phase run、固定 `comm_sm=12` 时，DCP2/4/8 的 decode-only 结果如下。
这里先对每个 case 的 global milestone p50 做差，再对 87 个 decode cases 求算术
平均；`Final/body` 使用相同 case 集合的均值之比：

| DCP | CUDA event | Kernel body | `receive_done -> final_done` | Final/body |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 79.152 us | 68.907 us | 10.758 us | 15.6% |
| 4 | 82.823 us | 72.224 us | 15.994 us | 22.1% |
| 8 | 99.304 us | 87.675 us | 27.590 us | 31.5% |

因此原始 final critical tail 随 DCP size 明显放大；DCP2 并非主要瓶颈，DCP4 已占
body 约五分之一，而 DCP8 达到约三分之一。这也是本轮优先优化 DCP8、同时保留
大 task-count 动态 claim 路径的原因。

旧 final queue 固定一个 16-token task 对应一个 CTA。低负载 decode 的 parent
task 数远小于 H100 上的 compute CTA 数，因此大量 CTA 无工作；有工作的 CTA
又需要串行遍历 8 个 DCP state。单纯增加 communication task 数不能解决这个
问题，因为 remote publication 和 TMA pull 仍应保持 16-token 对齐。

### 21.2 自适应 final task 粒度

Metadata v7 将 communication parent 和 final compute subtask 分离。parent 数仍为：

```text
parent_task_count = ceil(total_q / 16)
num_compute_ctas  = num_sms - num_comm_sm
```

final task 的 token 粒度使用严格小于比较：

```text
parent_task_count <     num_compute_ctas ->  4 tokens/final task
parent_task_count < 2 * num_compute_ctas ->  8 tokens/final task
otherwise                                -> 16 tokens/final task
```

边界行为为：

```text
N - 1  -> 4
N      -> 8
2N - 1 -> 8
2N     -> 16
```

Q all-gather、history publish、remote output pull、`tile_ready` 和
`receive_ready` 仍基于 16-token parent tile。4/8-token final descriptor 不跨 parent
边界；同一 parent 的 sibling subtasks 共享 publish/receive readiness。
`FinalWorkDesc` 新增 `parent_token_block`，packed header 仍保持 40 个 int，metadata
semantic version 升到 v7。`FINAL_TOKENS_PER_TASK=4` 继续作为容量上界中的最小粒度，
实际选择通过 runtime queue diagnostics 的 `final_tokens_per_task` 报告。

### 21.3 Pull 与 final descriptor 使用同一 parent 顺序

旧实现即使为 final queue 增加 subtasks，如果 remote output pull 和 final descriptor
采用不同 parent 顺序，排在 final queue 前面的 task 仍可能长期等待未被优先 pull
的 parent。

v7 使用同一个 `heuristic_q_block_order` 控制：

```text
Q all-gather/pull
  -> history combine/publish
  -> remote output pull logical order
  -> final descriptor parent order
```

CUDA helper `receive_task_id_from_pull_ordinal()` 将 logical pull ordinal 通过
`q_tasks[parent_ordinal * DCPSize]` 映射回原有 physical receive layout：

```text
receive_id = physical_parent_token_block * (DCPSize - 1) + source_ordinal
```

因此 IPC 地址和 `receive_ready[parent, source]` 布局没有变化。final descriptors
先按 `heuristic_q_block_order` 分组，再按 parent 内 token offset 排列。

这里对齐的是 logical scheduling priority，不是强制完成顺序。receive loop 仍是
readiness-aware bounded scan；后排但已经 ready 的 parent 可以先完成，不引入
device-side completion FIFO。

### 21.4 Final CTA 调度与 DCP-state reduction

final queue 使用两种调度：

```text
final_count <= num_compute_ctas:
    static CTA i -> final task i

final_count > num_compute_ctas:
    atomic counter claims descriptors in metadata order
```

静态路径去掉低负载 decode 中每个 CTA 的 final ticket atomic。大 batch 保留动态
claim，避免 task 数超过 CTA 数时只处理第一波。

每个 128-d output vector 使用 16-lane subgroup。前 8 lanes 分别拥有一个 DCP
state 的 LSE，先通过 subgroup shuffle 并行求 max 和 denominator，再对 O 做加权
累加。该实现避免由一个 lane 串行加载 8 个 LSE，并减少原路径通过 shared memory
广播 LSE 所需的 warp 同步。

完全 unroll 最大 final task 的尝试导致明显 register spill，因此最终保留：

```cpp
#pragma unroll 1
for (int vector_in_task = vector_in_wave;
     vector_in_task < work.valid_vectors;
     vector_in_task += 16)
```

代表性 DCP8、Hq-local=4、non-split 实例为 `288B spill stores / 300B spill loads`，
没有采用 spill 更严重的全展开版本。

### 21.5 正确性验证

CPU metadata suite：

```text
python -m unittest scripts.test_min_fa3.test_dcp_mega_metadata
Ran 39 tests
OK
```

测试覆盖 DCP2/4/8、4/8/16-token 分支、严格阈值、non-16 tail、parent 不交叉、
pull/final parent 同序和 metadata capacity。

DCP8 GPU correctness 额外覆盖：

1. `q=(16,16,16)`、history `(1139,44536,3167)`、实际 parent order `(1,2,0)`，
   验证非平凡 release-LPT 顺序和 4-token final。
2. `108 x 16-token` decode requests，命中 `num_compute_ctas=108` 的 8-token 边界。
3. `216 x 16-token` decode requests，命中 16-token 边界。

三组均通过 eager、prepared replay 和 CUDA Graph，并验证 O/LSE reference。
完整 SM90 build 覆盖 DCP2/4/8、split/non-split、BlockN 128/176、Hq-local 4/8。

### 21.6 100-case benchmark

原始日志删除后，以以下 run ID 和本节内嵌结果作为归档记录：

| Run ID | 代码状态与范围 | Manifest 完整性 | Timing source |
| --- | --- | ---: | --- |
| `20260811-150046-arrival1-phases-graph` | 原始 Metadata v5；DCP2/4/8 Mega phase sweep 和 Graph baselines | 每个 DCP 的 Mega 500/500；Graph 100/100 | Mega `internal_cpp_cuda_events`；Graph capture 内 events |
| `20260811-174805-final4-arrival1-dcp8` | 固定 4-token final，尚未 pull/final 同序 | 500/500，5 个 comm-SM variants | `internal_cpp_cuda_events` |
| `20260811-193634-adaptive-final-order-arrival1-dcp8` | 自适应 4/8/16-token final + pull/final 同序 | 100/100，`comm_sm=12` | `internal_cpp_cuda_events` |
| `20260811-launch-adjacent-events-dcp8` | 相同 final kernel，加 direct-launch/event 优化 | 100/100，`comm_sm=12` | `internal_cpp_reused_launch_adjacent_cuda_events` |

四次 run 的共同环境和工作负载为：8x NVIDIA H100 80GB HBM3、CUDA 12.8、
PyTorch 2.10.0+cu128、Python 3.12.13、TP=8、Q heads=32、head dim=128、
arrival rate 1、seed 42、warmup 40、iterations 60。trace 共 100 cases，其中 87 个
decode-only、13 个 mixed；trace SHA256 为
`b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df`。
各 DCP 的生成配置 SHA256 为：

| DCP | Trace config SHA256 |
| ---: | --- |
| 2 | `fe04ed59234d34eef061618bed6a82399b31a7c5f5449186998d93c519117505` |
| 4 | `46ad02286b580356a65ec08220a4eb041925b0a6d4de91b11b37461ca8f73155` |
| 8 | `dd35c6a45eae1863c671cfe3023173c39b4515b9c11a411851be7d362738f301` |

原始 phase run 记录的 repository commit 为 `3c014397fae8b016d7d13e8ec9cdf2b3b136d9ba`；
后三次 run 记录为 `caa5b0c580b2472decc1e0137711908d677aad54`。final 和 launch
修改当时尚在 working tree 中，因此 commit 字段不能单独标识 kernel 版本，必须
同时使用上表的 run ID 和代码状态。baseline source commits 为 vLLM
`a89015c6df8eeb37a843b717c97a5be1355de83d`、SGLang
`8d6549bc4039d33635844495d86684677a4f0df8`。

DCP8 headline 对比统一选择 `comm_sm=12`。每个 iteration 先对 8 个 rank 的 local
latency 取最大值，每个 case 再跨 iterations 取 p50，最后对 100 cases 的
global-rank-max p50 求算术平均。fixed-final4 和原始 Mega 虽运行了
`comm_sm=4,8,12,16,20` 五档 sweep，下表没有进行 per-case best-of-sweep 选择。

| Scope | Adaptive + order | Fixed final4 | Original Mega | vLLM A2A Graph |
| --- | ---: | ---: | ---: | ---: |
| All 100 | 0.132733 ms | 0.135494 ms | 0.154088 ms | 0.208366 ms |
| Decode-only 87 | 0.075730 ms | 0.076910 ms | 0.099304 ms | 0.139419 ms |
| Mixed 13 | 0.514212 ms | 0.527554 ms | 0.520725 ms | 0.669776 ms |

用于复核 phase 差值的 DCP8 decode-only 绝对 milestone 均值如下。所有 milestone
均相对各 rank 的 `%globaltimer kernel_start`，表中仍采用每 case global-rank-max
p50 后跨 case 求均值：

| Variant | Event | Q done | Attention done | History/publish done | Receive done | Final done | Kernel done |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Mega | 99.304 us | 7.806 us | 38.599 us | 50.215 us | 59.812 us | 87.402 us | 87.675 us |
| Fixed final4 | 76.910 us | 7.789 us | 38.556 us | 50.223 us | 59.261 us | 65.079 us | 65.360 us |
| Adaptive + order | 75.730 us | 7.850 us | 39.009 us | 50.757 us | 57.715 us | 63.942 us | 64.239 us |

相对 fixed-final4：

```text
All:    1.0208x, wins 83/100
Decode: 1.0156x, wins 76/87
```

相对 original Mega：

```text
All:    1.1609x, wins 96/100
Decode: 1.3113x, wins 87/87
```

相对仓库中的 vLLM A2A CUDA Graph orchestration baseline：

```text
All:    1.5698x, wins 100/100
Decode: 1.8410x, wins 87/87
```

这里的 vLLM baseline 使用同一个 `min_fa3_op` attention kernel，只比较仓库中的
A2A orchestration，不代表 production vLLM 原生 kernel 性能。

decode-only phase 平均值：

| Phase | Adaptive + order | Fixed final4 | Original Mega |
| --- | ---: | ---: | ---: |
| `publish_done -> receive_done` | 6.959 us | 9.039 us | 9.597 us |
| `history_combine_done -> final_done` | 13.185 us | 14.856 us | 37.188 us |
| `receive_done -> final_done` | 6.226 us | 5.818 us | 27.590 us |

顺序对齐主要改善 remote output arrival：publish-to-receive 相对 fixed-final4 下降
约 23%。纯 terminal final compute 比 fixed-final4 慢约 `0.41 us`，但相对 original
Mega 的 terminal tail 改善约 `4.43x / 77.4%`。仓库 vLLM A2A Graph 的 decode
`a2a_unpack_combine` 平均为 `6.375 us`，当前 Mega final arithmetic 已处于同一量级。

原始 Graph phase run 中，vLLM A2A baseline 的完整 decode-only stage 均值如下。
每项都是 87 个 case 的 per-case global-rank-max p50 算术平均；stage event 独立
测量，存在 stream overlap 和 event boundary，因此各列不要求严格相加等于 E2E：

| DCP | Graph E2E | Q AG + reorder | History attention | A2A pack | A2A collective | Unpack + combine | Chunk attention | State merge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 123.678 us | 19.505 us | 29.671 us | 6.112 us | 14.369 us | 5.870 us | 13.992 us | 5.873 us |
| 4 | 127.250 us | 22.989 us | 27.201 us | 6.727 us | 15.908 us | 5.929 us | 13.741 us | 5.769 us |
| 8 | 139.419 us | 29.441 us | 26.383 us | 7.834 us | 21.277 us | 6.375 us | 13.752 us | 5.760 us |

`a2a_unpack_combine` 在 DCP2/4/8 decode 中分别占 Graph E2E 的约 4.75%、4.66% 和
4.57%。它没有像原始 Mega final 一样随 DCP8 放大到 body 的三分之一；但该 kernel
只完成 A2A 后的 unpack/LSE merge，不能脱离 Graph orchestration 总时间单独比较
端到端优劣。

同一次 Graph run 中的 SGLang-style baseline 没有同构的单个 final-combine kernel。
它先 all-gather LSE，在 PyTorch 中计算 global LSE 和 partial-output scale，再对
FP32 partial output 执行 all-reduce；distributed combine 成本应看
`LSE all-gather/correction + FP32 all-reduce` 两段之和：

| DCP | Graph E2E | LSE AG + correction | FP32 all-reduce | Distributed combine 合计 | Chunk state merge |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 178.091 us | 45.717 us | 18.382 us | 64.099 us | 29.499 us |
| 4 | 188.838 us | 47.909 us | 26.207 us | 74.116 us | 29.161 us |
| 8 | 229.207 us | 55.237 us | 51.591 us | 106.829 us | 29.661 us |

这些也是 87 个 decode-only case 的 per-case global-rank-max p50 算术平均。该
SGLang-style 数字来自仓库内使用相同 `min_fa3_op` attention kernel 的 pinned
orchestration，并非 production SGLang 原生 kernel benchmark。其优点是逻辑直接、
利用成熟 collectives；但在此低负载 DCP8 trace 上，FP32 payload、单独 LSE
all-gather 和高层 correction 使它不适合作为 Mega final 的低延迟实现模板。相较之下，
vLLM A2A 的 packed BF16 O + FP32 LSE 传输及单个 unpack/combine kernel 更接近当前
Mega final 所需的数据流。

该 trace 的粒度分布为：

```text
4-token:  93 cases
8-token:   0 cases
16-token:  7 cases
```

87 个 decode-only case 全部选择 4-token。6 个小 mixed case 选择 4-token，相对
fixed-final4 基本持平；7 个大 mixed case 选择 16-token，总延迟改善 2.9%，且
`receive_done -> final_done` 从 `73.461 us` 降到 `27.872 us`。8-token 中间分支
由 correctness 覆盖，但该 100-case trace 没有实际命中。

`case_000000` 相对 fixed-final4 有一个明显 decode 回归：`78.640 -> 99.872 us`，
几乎全部来自 `receive_done: 61.056 -> 80.976 us`。8 个 rank 生成的 parent order
均为 `(0,4,1,3,5,2,6)`，不是跨 rank metadata disagreement。本轮按要求暂不围绕
该反例回退整体 ordering，因为 ordering 在 76/87 decode cases 和 83/100 overall
cases 上获胜。

## 22. Direct-launch 热路径与 event/body gap（2026-08-11）

本节记录 final combine 优化完成后，对 Mega eager CUDA event 计时边界和 direct
launch 热路径的第一阶段优化。该阶段不改变 kernel body、workspace reset、IPC
phase 或 metadata 内容。

### 22.1 固定约 11.5 us 差值的来源

在上一节 100-case run 中，event 时间和 `%globaltimer` body 为：

| Scope | CUDA event | `kernel_done-kernel_start` | Gap |
| --- | ---: | ---: | ---: |
| All 100 | 132.733 us | 121.186 us | 11.547 us |
| Decode-only 87 | 75.730 us | 64.239 us | 11.491 us |
| Mixed 13 | 514.212 us | 502.289 us | 11.922 us |

decode gap 的范围只有 `10.96-11.92 us`，与 workload 大小基本无关。旧 C++
binding 的顺序为：

```text
cudaEventRecord(start)
  -> construct two FA3 KernelParams
  -> cudaGetDevice twice
  -> construct TK PGL/GL and Mega KernelParams
  -> cudaFuncSetAttribute
  -> kernel<<<...>>>()
  -> cudaGetLastError
cudaEventRecord(end)
```

CUDA event timestamp 在 GPU 上执行。如果 GPU 已经处理 start event，而 host 仍在
构造参数或调用 runtime，stream 会出现可见 idle bubble，并被 event elapsed time
计入。

内部 timestamp 也不是完整 grid 生命周期。`kernel_start` 只由
`blockIdx.x == 0 && threadIdx.x == 0` 写入，不保证等于最早 CTA entry；
`kernel_done` 由最后一个 CTA 的 completion atomic winner 在 CTA retirement 前写入。
因此 event-body gap 同时包含 event command、direct launch dispatch、grid ramp-up、
timestamp 边界偏差和最后 CTA retirement。

### 22.2 第一阶段修改

修改保持公开 runner API 和 kernel 参数语义不变：

1. C++ binding 使用 thread-local、per-device `ReusableTimingEvents`，不再每次
   replay 执行 `cudaEventCreate/Destroy`。
2. Binding 将 `q.get_device()` 写入 `DCPMega_fwd_params.device`，两个
   `make_attention_kernel_params()` 不再各自调用 `cudaGetDevice()`。
3. `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` 使用每个
   kernel specialization、每个 device 的 `std::call_once`，只设置一次。
4. Typed launcher 完成 `KernelParams` 构造和一次性 attribute setup 后，执行：

```text
cudaEventRecord(start)
kernel<<<...>>>()
cudaEventRecord(end)
cudaGetLastError
```

5. Benchmark JSON 将 eager timing source 标记为
   `internal_cpp_reused_launch_adjacent_cuda_events`。

本阶段没有引入 type-erased C++ prepared-launch handle。typed `KernelParams` 仍在
每次 backend call 重建，只是构造发生在 device event 之前。完整缓存需要按
DCP/BlockN/split/Hq specialization 保存不同 C++ 类型，并定义 dynamic metadata、
header 和 pointer 更新协议；当前 device event 指标也不能衡量该 host-only 收益，
因此没有在本轮扩大修改面。

### 22.3 构建与验证

完整 extension build 覆盖 DCP2/4/8、split/non-split、BlockN 128/176、Hq-local
4/8。PTXAS register、stack 和 spill 与 final-combine 修改后的 build 一致；代表性
DCP8、Hq-local=4、non-split 实例仍为 `288B spill stores / 300B spill loads`。

聚焦 DCP8 correctness 使用 `q=(1,8,16)`、history `(129,258,515)`、Hq-local=4、
split=1、BlockN=128、`comm_sm=8`，通过 eager、prepared replay 和 CUDA Graph。

### 22.4 Paired 100-case benchmark

归档 run ID 为 `20260811-launch-adjacent-events-dcp8`；原始日志可删除，必要的环境、
trace hash、manifest 完整性和 timing source 已归档在 21.6。配置与
`20260811-193634-adaptive-final-order-arrival1-dcp8` 相同：DCP8、arrival rate 1、
`comm_sm=12`、warmup 40、iterations 60、同一 trace、Mega eager、phase timestamps
enabled，100/100 cases 完成。

| Scope | Old event | New event | Old body | New body | Old gap | New gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All 100 | 132.733 us | 128.741 us | 121.186 us | 121.533 us | 11.547 us | 7.208 us |
| Decode-only 87 | 75.730 us | 71.283 us | 64.239 us | 64.111 us | 11.491 us | 7.172 us |
| Mixed 13 | 514.212 us | 513.268 us | 502.289 us | 505.819 us | 11.922 us | 7.449 us |

decode-only event 平均下降 `4.447 us / 5.87%`，约 `1.062x`；body 仅变化
`-0.128 us`，可视为持平。event-body gap 下降 `4.319 us / 37.6%`。87 个 decode
case 中 86 个 event 更短；全部 100 cases 中 97 个更短。

新 decode gap 分布为：

```text
mean  7.172 us
p50   7.168 us
min   6.816 us
max   7.504 us
```

三个 event 未获胜的 case 为 mixed `case_000014`、mixed `case_000044` 和 decode
`case_000069`。三者的新 gap 均下降，event 回归来自该轮 kernel body 波动；其中
`case_000014` body 增加约 `41 us`。

约 `4.3 us` gap 收缩不能全部解释成真实 serving latency 收益。缓存 function
attribute、删除 runtime device query 和复用 events 是实际 host 热路径优化；把
event 移到 typed params 构造之后则修正了计时边界。真实 host-to-output latency
仍需 CPU wall time、CUPTI 或 Nsight Systems 单独测量，不能用新的纯 device
kernel-command event 直接代替。

### 22.5 Reset、CUDA Graph 与 baseline 计时语义

Mega eager prepared replay 的以下 reset 发生在 internal event 之前：

```text
q_ready.zero_()
attention_done.zero_()
publish_ready.zero_()
receive_ready.zero_()
queue_state.zero_()
phase_timestamps.zero_()  # profiling enabled 时
```

因此 reset 不是旧 `11.5 us` gap 的来源。CUDA Graph 也支持 capture 这些
`cudaMemsetAsync`；现有 Mega fixed-shape Graph 已经 capture：

```text
workspace reset
  -> device phase advance
  -> pre IPC barrier
  -> Mega kernel
  -> post IPC barrier
```

动态 batch 使用 Graph 的主要障碍是 dynamic metadata/header、kernel
specialization 和 pointer 更新，不是 buffer reset。

baseline Graph 当前也不是把多次 replay 放在一对 event 中再平均。每个 sample
执行一次 `graph.replay()`；capture 内的 `attention_start/end` event nodes 记录
单次完整 Graph 的时间。该口径：

- 排除 Python、逐 kernel host launch、Graph capture/instantiate 和后续 rank
  aggregation；
- 通常排除 `cudaGraphLaunch` host API 和内部 start event 执行前的初始部分；
- 包含 start/end event nodes 的扰动、Graph 内 node scheduling、每个 kernel 的
  GPU front-end dispatch、grid ramp-up/retirement、NCCL/A2A kernel 和实际计算；
- 开启 `--baseline-phase-timing` 时，还包含额外 phase event nodes 的扰动。

因此 CUDA Graph 消除的是逐 kernel host submission，不会消除 device-side kernel
dispatch 和 grid lifecycle。现有 Mega Graph benchmark 使用 Graph 外部 Python
events 包住 replay，与 baseline 的 Graph-internal start/end events 仍不是严格同一
边界。若继续做 apples-to-apples Graph 比较，应同时报告：

```text
graph_internal_ms:
    capture 内 start/end event，匹配 baseline 当前口径

external_replay_ms:
    外部 event 包住 graph replay，包含 GPU 可见 graph submission 和 stream waits
```

headline latency 应只保留 start/end events；完整 phase events 单独 profile，避免
大量 event nodes 扰动几十微秒级 decode。当前 batch CLI 的
`--mega-num-comm-sms` sweep 仍限制为 eager-only，尚未产出新的 Mega Graph
100-case 结果。
