# DCP Mega Critical-Wave 调度设计

## 1. 文档目标

本文描述 DCP Mega host metadata 调度优化的稳定设计，包括：

- 为什么原生 split 与 FIFO 顺序在 ragged batch 上容易产生尾部；
- critical-wave cost model 如何估算 attention 与 history combine 的重叠执行；
- 多序列迭代 split、Q 解锁排序和 release-aware LPT 如何工作；
- split、history order 和 BlockN 三个决策如何组合；
- metadata 生成必须维持的正确性约束与当前模型边界。

本文是一份特性设计文档，不记录逐轮实验时间线、单个 case 数字或 benchmark 日志位置。实验数据只用于确定默认策略，后续调整模型时应重新进行独立、严格配对的验证。

主要实现位于：

- `dcp_mega_metadata.py`：split 选择、cost model、队列排序、metadata 构造与校验；
- `min_fa3_dcp.py`：runner 参数、metadata 上传和运行时诊断；
- `include/dcp_mega_min_fa3_varlen_scheduler.h`：device 端 attention descriptor scheduler；
- `include/dcp_mega_min_fa3_varlen_launch.h`：persistent kernel、combine 和通信队列执行。

## 2. 背景

DCP Mega 将一个 chunk-prefill batch 拆成两个 attention domain：

1. `chunk attention`：本 rank 的 causal chunk K/V；
2. `history attention`：DCP group 聚合后的 Q 对本 rank history K/V 做 noncausal attention。

host 为两个 domain 构造统一的 attention descriptor 队列。compute CTA 完成 attention 后继续处理 history combine 和 final combine；communication CTA 负责 Q all-gather、接收远端 history partial，并在空闲时协助 combine。

一个 batch 通常同时包含：

- 很短的 decode query；
- 较长的 chunk query；
- 差异很大的 history 长度；
- 不同数量的 M-block、N-block 和 split task。

因此总 FLOPs 或总 task 数接近，并不代表各 CTA 的完成时间接近。真正影响 kernel 尾部的是最后一个 wave 中 task 的离散装箱结果，即 wave quantization。

## 3. 原始策略及局限

### 3.1 FA3 native dynamic split

原始 split 路径先根据总工作量、SM 数量和 split upper bound 估计每个 SM 应承担的 N-block 数，再为每条序列独立计算：

```text
n_blocks_i = ceil(k_len_i / BlockN)
split_i = clamp(ceil(n_blocks_i / blocks_per_sm), 1, split_upper_bound)
```

它适合作为通用 occupancy heuristic，但没有模拟最终 descriptor 顺序和每个 CTA 的实际 load。对于 mixed batch，多个序列的短 task 与少量长 task 会共同决定最后一个 wave，单独按序列计算 split 容易出现过拆或漏拆。

### 3.2 FIFO descriptor 顺序

原始 metadata 顺序为：

```text
全部 chunk descriptor
    -> history sequence 0
    -> history sequence 1
    -> ...
```

history 内部保持 batch、M-block、split 的构造顺序。Q all-gather task 同样按 token block 升序排列。

FIFO 简单、确定，且对 mixed batch 通常有较好的局部性；但 decode-only batch 中，不同 history task 的成本差异很大，长 task 可能集中进入同一个晚期 wave。此外，FIFO 没有优先发送能够解锁更多或更重 history work 的 Q block。

### 3.3 split 的代价

split 可以缩短单个 attention task，却不是免费操作：

- descriptor 数增加；
- scheduler claim 和 task 固定开销增加；
- 每个输出向量需要读取更多 partial O/LSE；
- history combine 的计算和访存量增加；
- 过多小 task 可能让 attention makespan 下降，但总 kernel latency 上升。

因此调度目标不能只是最小化最长 attention descriptor，也不能只最大化 occupancy。

## 4. 设计目标

critical-wave 调度器的主要目标按优先级排列为：

1. 降低 attention 与 dependent combine 的整体 makespan；
2. 改善 compute CTA 的负载均衡，减少最后一个未填满 wave 的尾部；
3. 在收益接近时选择更少的 attention task 和更小的 split 向量；
4. 保留 FA3 native dynamic split 作为候选和运行时回退；
5. 不修改 CUDA attention 数学实现和 packed metadata ABI；
6. 保持 host 侧决策确定性，便于 CUDA Graph replay 和离线复现。

非目标：

- cost model 不是 CUDA cycle simulator；
- 不尝试精确预测绝对微秒数；
- 不在这一层改变通信协议、attention kernel tile shape 或 combine 数学；
- 不根据一次 benchmark 自动在线训练参数。

## 5. 调度决策概览

metadata 生成将三个决策分开处理：

1. `split policy`：critical-wave 或 FA3 native；
2. `history order`：FIFO 或 Q 解锁排序 + release-aware LPT；
3. `BlockN`：auto、固定 128 或固定 176。

前两个轴可以形成完整的 2 x 2 组合：

| Split policy | History order |
|---|---|
| critical-wave | FIFO |
| critical-wave | release-LPT |
| FA3 native | FIFO |
| FA3 native | release-LPT |

Q 解锁排序和 release-LPT 被视为同一个顺序策略。当前不提供“只排序 Q”或“只排序 history descriptor”的中间模式，因为两者共同定义 descriptor 的 release epoch。

## 6. 无量纲 attention 模型

### 6.1 task 展开

模型按照最终 metadata 使用的相同规则展开 chunk/history attention descriptor。

对序列 `i`：

```text
chunk_m_blocks_i   = ceil(q_len_i * Hq_local / 128)
history_m_blocks_i = ceil(q_len_i * DCP_size * Hq_local / 128)
history_n_blocks_i = ceil(history_len_i / BlockN)
```

每个 history M-block 生成 `split_i` 个 descriptor。第 `s` 个 split 覆盖的 N-block 数为：

```text
blocks_per_split = ceil(history_n_blocks_i / split_i)
split_n_blocks(i, s) = clamp(
    history_n_blocks_i - s * blocks_per_split,
    0,
    blocks_per_split,
)
```

### 6.2 task cost

attention descriptor 使用无量纲成本：

```text
attention_work = n_blocks + attention_task_overhead
attention_task_overhead = 4
```

`n_blocks` 表示随 K/V 长度增长的主体工作；固定开销近似 descriptor claim、pipeline 启停和短 task 中无法忽略的常量成本。

这里保留无量纲表达，而不直接拟合微秒，原因是不同 BlockN、DCP size、batch shape 和系统负载下绝对时间不稳定；相对调度关系比绝对标定更可靠。

### 6.3 CTA list scheduling

compute CTA 数为：

```text
num_compute_ctas = num_sms - num_comm_sm
```

模型维护一个按当前 load 排序的 CTA min-heap，按照候选方案的真实 descriptor 顺序逐个放置 task：

```text
cta = pop_min_load_cta()
start = cta.load
finish = start + attention_work
cta.load = finish
push(cta)
```

模型同时保留：

- 每个 CTA 的 attention finish time；
- 每个 completion ID 的 finish time；
- makespan；
- makespan CTA 最后处理的 history sequence 集合。

最后一项构成 `critical_history_sequences`，用于识别下一轮值得继续拆分的序列。

## 7. Q 解锁排序与 release-aware LPT

release-LPT 不是普通的全局 LPT。history descriptor 依赖 Q all-gather 的 token-block ready counter，尚未就绪的 descriptor 会等待依赖，因此排序必须同时考虑 release time 和 task 长度。

### 7.1 Q block 解锁权重

对每个 history descriptor，将其 `attention_work` 平均分配给它依赖的 Q block：

```text
unlock_work[q_block] += attention_work / dependency_count
```

Q task 按以下 key 排序：

```text
(-unlock_work, original_q_block_id)
```

即优先传输能够解锁更多 history work 的 Q block，原始 ID 用作确定性 tie-break。

### 7.2 release epoch

根据新的 Q block 顺序，history descriptor 的 release position 为其最晚依赖的位置：

```text
release_position = max(q_position[dependency])
q_counters_per_epoch = max(1, num_comm_sm // dcp_size)
release_epoch = release_position // q_counters_per_epoch
```

这不是实际的 Q ready 时间预测，而是与通信并行度一致的离散 release bucket。

### 7.3 history descriptor 顺序

全部 chunk descriptor 仍然放在 history descriptor 之前。history 部分按以下 key 排序：

```text
(release_epoch, -attention_work, batch_idx, m_block, split_idx, completion_id)
```

含义是：

1. 不跨越 release epoch 抢跑；
2. 同一 epoch 内先调度长 task，降低晚期 wave 出现单个重 task 的概率；
3. 使用完整坐标保证顺序稳定。

cost model 和最终 metadata 必须使用完全相同的顺序。history order 不能在 split 选择完成后作为无成本后处理，否则模型评分与实际队列不一致。

## 8. 依赖感知的 attention/combine 重叠模型

只看 attention makespan 会系统性偏好过多 split。当前 score 继续模拟 history combine，并保留它对 attention completion ID 的依赖。

### 8.1 combine task 工作量

combine task 按实际 packed metadata 顺序构造：

```text
final token tile
    -> destination rank
        -> sequence region
            -> vector task
```

NoSplit 路径可以在一个 task 中复制多个 vector；split 路径每个 vector 独立 combine。每个 task 的无量纲成本为：

```text
combine_work = combine_task_overhead + valid_vectors * actual_splits
combine_task_overhead = 4
```

这同时计入固定 task 开销和随 partial vector 数增长的工作。

### 8.2 worker release 与依赖

每个 compute CTA 提供 `MEGA_COMPUTE_WARPS = 12` 个 combine worker。一个 CTA 的 combine worker 只有在该 CTA 完成 attention 后才可用：

```text
worker.available_time = cta_attention_finish_time
```

对 FIFO combine task `j`：

```text
worker = pop_earliest_available_worker()
dependency_ready = max(
    attention_completion_time[id]
    for id in task_j.dependencies
)
start_j = max(worker.available_time, dependency_ready)
finish_j = start_j + combine_work_j
worker.available_time = finish_j
```

先领取 task 再等待 dependency，模拟 device 端 FIFO claim 后 warp 被该 task 占住的行为。模型不允许同一个 warp 绕过未就绪 task 去执行后续 ready task。

最终评分为：

```text
score = max(attention_makespan, latest_combine_finish)
combine_penalty = score - attention_makespan
```

因此 attention 和 combine 可以出现在同一个时间区间，`combine_penalty` 只表示未被 attention 覆盖并落在关键路径上的尾部。

## 9. 多序列迭代 split

### 9.1 初始候选

critical-wave 总是构造 NoSplit 候选。允许 native automatic split 时，还会构造 FA3 legacy dynamic split 向量，并把它作为：

- 每条序列 split 搜索的基础 upper bound；
- 最终必须参与比较的完整候选。

对满足以下条件的长 decode-history 序列，搜索上限至少扩展到 4：

```text
q_len <= 16 and history_n_blocks >= num_compute_ctas
```

这避免 native occupancy heuristic 过早把真正的 critical decode sequence 限制为 NoSplit。

### 9.2 迭代步骤

迭代从 NoSplit 向量开始：

```text
current = [1, 1, ..., 1]

while true:
    candidates = {}

    for each sequence i below its split cap:
        candidates += current with split_i increased by 1

    if all tied critical sequences can still split:
        candidates += current with every tied critical split increased by 1

    best = minimum candidate by deterministic ordering

    if best.score < current.score:
        current = best
    else:
        stop
```

“所有并列 critical sequences 联合增加”用于处理 critical plateau：如果多个等重序列共同构成最后一个 wave，只拆其中一个通常不会降低 makespan，却会增加 combine 工作；联合候选可以跨过这一平台。

接受条件使用严格 `<`。score 相等时继续 split 只会增加 task/partial，不能带来关键路径收益。

### 9.3 候选 tie-break

候选按以下顺序比较：

```text
(
    score,
    attention_makespan,
    attention_task_count,
    sum(sequence_splits),
    sequence_splits,
)
```

这保证在总体 score 相同时优先选择：

1. attention 尾部更短；
2. descriptor 更少；
3. split 总量更小；
4. 字典序稳定的向量。

### 9.4 最终选择和收益门槛

最终比较：

- NoSplit；
- multi-sequence iterative；
- FA3 native dynamic split（若启用）。

只有相对 NoSplit 的完整 score 收益达到 `10%`，且 selected score 严格更小时，才接受 split：

```text
gain = (nosplit_score - selected_score) / nosplit_score
accept = gain >= 0.10 and selected_score < nosplit_score
```

门槛用于抵抗无量纲模型未覆盖的 launch、cache、atomic 和通信扰动。它不是从单个 case 拟合出的精确常数。

## 10. History order 的策略解析

Python metadata API 使用三态 order：

```text
reorder_history_override = None   -> auto
reorder_history_override = False  -> FIFO
reorder_history_override = True   -> Q 解锁排序 + release-LPT
```

命令行对应：

```text
--mega-history-order auto|fifo|release-lpt
```

batch 类型定义为：

```text
decode-only: all(q_len <= 16)
mixed:       any(q_len > 16)
```

auto 顺序策略：

```text
critical-wave NoSplit:
    FIFO

critical-wave split + decode-only:
    release-LPT

critical-wave split + mixed:
    FIFO

FA3 native / fixed split:
    FIFO
```

显式 FIFO 或 release-LPT 覆盖 auto。显式 release-LPT 即使最终为 NoSplit 也保持启用，以便顺序策略可以独立消融。

critical-wave 候选评分必须与 resolved order 一致：

| Order 设置 | NoSplit 候选 | Split 候选 |
|---|---|---|
| 显式 FIFO | FIFO | FIFO |
| 显式 release-LPT | release-LPT | release-LPT |
| auto + decode-only | FIFO | release-LPT |
| auto + mixed | FIFO | FIFO |

因此切换 history order 可能改变最终 split 向量，而不只是改变 descriptor 排列。

## 11. BlockN 策略

BlockN 同时影响：

- 每个序列的 N-block 数；
- attention task cost；
- wave quantization；
- kernel tile 效率。

如果先用一个 BlockN 选择 split，再切换 BlockN 后重新选择，会形成循环依赖。auto 策略因此固定一个规范决策模型：

```text
critical-wave model BlockN = 128

if selected plan is NoSplit:
    dispatch BlockN = 176
else:
    dispatch BlockN = 128
```

显式 `--mega-block-n 128|176` 同时固定模型和 dispatch。FA3 native 和固定 split 在 auto 下保留 BlockN=128 fallback。

最终默认组合为：

| 最终方案 | History order | Dispatch BlockN |
|---|---|---:|
| Critical-wave NoSplit | FIFO | 176 |
| Critical-wave decode-only split | release-LPT | 128 |
| Critical-wave mixed split | FIFO | 128 |

## 12. Metadata 构造与正确性约束

优化只改变 host 侧 split 向量和队列顺序，不改变 attention 数学。metadata 必须维持以下不变量。

### 12.1 descriptor 完整覆盖

每个 `(kind, batch, m_block, split_idx)` 必须恰好出现一次。completion ID 必须稠密且唯一；排序只移动 descriptor，不能重编号其逻辑 completion。

### 12.2 Q dependency

history descriptor 必须引用覆盖其全部有效 packed Q rows 的 Q-ready counter。tail tile 的无效行不能进入依赖集合，也不能参与输出 combine。

### 12.3 combine dependency

每个 history combine task 必须依赖它读取的所有 history partial completion ID。排序后 dependency 仍按 completion ID 查找，不能按新的 queue ordinal 推断。

### 12.4 Q task 与 attention order 一致

release-LPT 启用时：

- Q task 必须按 `q_block_order` 生成；
- history descriptor 必须按同一个 order 推导的 release epoch 排序。

FIFO 模式下两者都必须保持原 token-major/batch-major 顺序。

### 12.5 CUDA Graph 稳定性

同一组 shape、host cu-seqlens 和策略输入必须生成完全确定的 metadata。所有 tie-break 都包含原始 ID 或完整坐标，禁止依赖 Python set/dict 的非确定遍历顺序。

## 13. 运行时接口与诊断

split 轴：

```text
--mega-scheduler-heuristic       critical-wave
--no-mega-scheduler-heuristic    FA3 native
```

history order 轴：

```text
--mega-history-order auto|fifo|release-lpt
```

BlockN 轴：

```text
--mega-block-n auto|128|176
```

host 诊断使用正交字段，不应解析组合字符串来反推策略：

```text
split_policy:
    critical_wave | fa3_native

history_order_policy:
    fifo | release_lpt
```

同时保留：

- model/effective BlockN；
- baseline/selected attention task 数；
- combine task、partial vector 和 work；
- baseline/selected attention makespan；
- combine penalty 和完整 gain；
- selected split vector、plan source 和 Q block order。

这些字段用于解释决策和回归测试，不构成 device metadata ABI。

## 14. 踩坑总结

以下问题曾导致模型或实测结论偏离，后续修改应避免重复：

1. **只看总体 FLOPs 或平均 occupancy。** 它们无法描述最后一个 wave 的离散装箱，必须模拟 descriptor 顺序和 CTA load。
2. **只拆一条最重序列。** 多条并列 critical sequence 会形成平台；需要联合候选，也需要允许下一轮 critical sequence 发生变化。
3. **把 native split 只当 baseline。** native dynamic 向量在部分 workload 上仍是有效候选，不能只比较 iterative 与 NoSplit。
4. **只惩罚 split 数。** combine 成本取决于 task 粒度、partial vector 数和 attention dependency；聚合常数 penalty 会误判可被 attention 隐藏的工作。
5. **先选 split，再改变排序。** FIFO 与 release-LPT 的 makespan 不同；模型必须按最终会发射的顺序评分。
6. **decode 与 mixed 共用 LPT 默认。** release-LPT 对 decode wave 有帮助，但 mixed batch 的 chunk/history 组合不同，默认策略必须区分 batch 类型。
7. **所有方案统一 BlockN。** NoSplit 与 split 的最佳 tile 选择不同；需要规范 model BlockN 和明确的 dispatch 规则。
8. **用 `<=` 接受等分候选。** score 不下降时继续 split 会引入纯额外工作，因此迭代接受条件必须是严格 `<`。
9. **忽略 benchmark 隔离。** 多用户 GPU 上的外部进程会污染尾延迟；参数结论必须来自同环境、同 case、严格配对且无竞争的测试。

## 15. 当前模型边界

critical-wave 已显式建模 attention descriptor、CTA wave、completion dependency 和 history combine overlap，但仍未包含：

- Q all-gather 的实际 ready timestamp；release epoch 只是离散近似；
- HBM/L2 contention、TMA pipeline 状态和不同 task 间 cache reuse；
- atomic claim 的运行时非确定性和 warp scheduling 细节；
- receive、remote publication 与 final combine 的完整成本模型；
- BlockN=128/176 在不同 N-block 长度上的真实 kernel throughput 曲线；
- 多节点通信和非 SM90 平台。

因此模型适合做候选排序和保守过滤，不适合输出绝对 latency。任何阈值或默认策略变化都必须同时满足：

1. CPU metadata/依赖不变量测试通过；
2. DCP=2/4/8 correctness 通过；
3. decode-only 与 mixed workload 分开报告；
4. 与现有默认进行逐 case 配对，而不是只比较聚合吞吐；
5. GPU 环境无外部竞争，并记录 effective split/order/BlockN。

## 16. 维护原则

- cost model 与 metadata 构造共享相同的 task 展开和排序规则；
- 新增成本项时优先保持无量纲，并解释它对应的 device 工作；
- 不通过单个反例直接调参，应先确认是模型缺项、测量污染还是 kernel 特性；
- 保留 NoSplit 和 FA3 native 候选，确保优化可回退；
- 新策略必须增加正交诊断字段和 CPU 行为测试；
- 文档只维护当前设计和关键取舍，实验明细放在 benchmark 产物或单独分析报告中。
