# Mega DCP Post-Q History Combine Assist 实验复盘

> 状态：实验已结束，功能代码已回退，不属于当前 kernel 行为
>
> 实验基线：`197dee5`
>
> 日期：2026-08-08
>
> 适用范围：SM90、BF16、head dimension 128、forward-only Mega DCP varlen

## 1. 结论

本实验验证了以下设想：communication CTA 完成 Q all-gather 后，使用部分 warp 与
compute CTA 共同消费 history split combine/no-split copy queue。

该方案在固定 `num_comm_sm=32` 时，相对同样配置但不参与 history combine 的
baseline 有明确收益：目标 split case 的 history tail p50 改善 18.65%，kernel p50
改善 6.78%。但是这不代表应把默认的 `num_comm_sm=8` 调到 32：在同一个目标 case
上，`comm=32 + assist` 仍比 `comm=8 + no-assist` 慢 3.27%。

`num_comm_sm=8` 时，通信 CTA 只能增加 32 个 combine worker warp，相对 1488 个
compute combine worker warp 仅增加 2.15%。这点并行度不足以缩短 history tail，反而
由于 receive/TMA 竞争使目标 split case 的 kernel p50 回退 29.16%。20-case trace 的
无条件 assist 平均 p50 也回退 2.06%，并出现最大 28.60% 的单 case 回退。

综合收益、复杂度和风险后，决定不保留本次实现。kernel 和 CPU test 修改均回退；
本文件只保存实验过程、证据和后续约束。

## 2. 实验目标与约束

目标是利用 communication CTA 在 Q all-gather 后的部分执行窗口，协助处理统一的
history combine queue，同时继续推进 remote history receive。

实现遵循以下约束：

- 保持 copied-and-trimmed Hopper/FA3 主路径，不重写 attention mainloop；
- split reduction 和 no-split copy 两个 specialization 都支持 assist；
- 不增加公开 runtime 参数；
- 不改变 params、metadata v5、workspace、binding、runner API 或 ABI；
- Q all-gather 继续使用原有 6 组 producer/consumer warp pair；
- receive task 最终仍由 `6 * num_comm_sm` 个静态 slot exactly once 消费；
- 保持 eager、prepared replay 和 CUDA Graph 行为一致。

预设性能门槛：

- split `attention_done_to_history_combine_done` p50 至少改善 10%；
- split max-rank kernel p50 至少改善 3%；
- representative no-split 和单个 full-trace case p50 回退不超过 2%。

## 3. 实现过程

### 3.1 Post-Q warp 分工

Q all-gather 阶段保持原始 6 对 warp，不做修改。Q all-gather 完成后，实验版本首先
使用以下布局：

```text
warp 0..3    receive producer，slot 0..3
warp 4..7    receive consumer，slot 0..3
warp 8..11   history combine/copy worker
```

4 个 combine warp 全部退出统一 queue 后，使用 warp-group completion counter 进行
角色转换：

```text
warp 8..9    receive producer，slot 4..5
warp 10..11  receive consumer，slot 4..5
```

这样前 4 对 receive warp 可以和 history assist 并发，后 2 对 receive warp 在 combine
staging 释放后接管原有 slot 4、5。receive 的静态 stride 和最终任务所有权保持不变。

### 3.2 统一 history queue

compute CTA 的 12 个 warp 与 communication CTA 的 4 个 assist warp 共享已有的
`kHistoryCombineCounter`。每个 warp 用 lane 0 原子领取 ticket，然后执行：

```text
领取 ticket
-> 等待 descriptor 对应的 attention completion
-> split reduction 或 no-split copy
-> 写 IPC send O/LSE
-> 更新 publish completion
```

没有新增 metadata queue 或 runtime 参数。split 路径仍是 warp-per-vector；no-split
路径仍使用 metadata v5 的 grouped copy task。

### 3.3 Shared-memory staging 分区

原 `run_history_combine()` 隐式使用 warp id 选择 staging。为了让 communication warp
复用同一实现，实验版本将 staging base 和 logical worker id 显式传入。

```text
compute CTA combine staging        comm_tiles[0..]
communication receive slot 0..3   comm_tiles[0..3]
communication combine staging     comm_tiles[4..5]
```

communication CTA 的 4 个 combine warp 共需要：

```text
4 warps * 4 stages * 128 floats = 8 KiB
```

`comm_tiles[4..5]` 正好提供 8 KiB，并且与并发 receive 使用的 `comm_tiles[0..3]`
不重叠。combine warp 全部退出后，slot 4、5 才复用这块 shared memory。

### 3.4 Completion 与 phase timestamp

原实现的 history completion 只统计 compute CTA。assist 版本需要保证：

- 每个 compute CTA 在 12 个 warp 汇合后贡献一次；
- 每个 communication CTA 在 4 个 assist warp 全部退出后贡献一次；
- history/publish completion 的 expected CTA count 在 assist 开启时为 `num_sms`；
- assist 关闭时仍为 `num_sms - num_comm_sm`；
- receive completion 独立统计，不能被 history completion 或最终 CTA barrier 延迟。

因此实验增加了 communication CTA 内部的 combine-warp 和 receive-pair completion
counter，以及由指定 leader 提交全局 phase completion 的路径。该协议没有引入
compute-grid barrier，但显著增加了角色和计数状态。

### 3.5 后期 profitability guard

`num_comm_sm=8` 的负面结果出现后，实验版本最后加入了内部静态阈值：

```text
communication assist workers >= 10% * compute combine workers
```

对应判断为：

```text
4 * num_comm_sm * 100 >= 12 * (num_sms - num_comm_sm) * 10
```

在 H100 `num_sms=132` 时：

```text
comm=8:   32 / 1488 = 2.15%，关闭 assist
comm=32: 128 / 1200 = 10.67%，开启 assist
```

guard 关闭 assist 时恢复原始 post-Q 布局：warp 0..5 为 receive producer，warp 6..11
为 receive consumer。该 guard patch 在决定终止实验前尚未完成重新编译、正确性和性能
验证，因此它不是一个已经验证的最终方案，也随功能代码一起回退。

## 4. 正确性与静态验证

guard 加入前完成了以下验证：

- split/no-split、BlockN 128/176、DCP 2/4/8、Hq local 4/8 和所有 split bucket
  specialization 编译通过；
- `scripts/test_min_fa3/test_dcp_mega_metadata.py` 共 21 个 CPU test 通过；
- 统一 queue CPU 模型验证 compute/communication workers 不重复、不遗漏领取 ticket；
- receive ownership 验证最终仍为 6 个 slot/CTA；
- 8-rank correctness matrix 连续通过两次；
- matrix 覆盖 DCP 2/4/8、split 1/2/16/auto、BlockN 128/176、tail case、eager、
  prepared replay 和 CUDA Graph；
- O/LSE correctness 和所有 phase invariants 通过；
- `git diff --check` 通过。

验证命令：

```bash
PYTHONPATH=. python scripts/test_min_fa3/test_dcp_mega_metadata.py

MAX_JOBS=2 python setup.py build_ext --inplace

PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
  scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py --matrix
```

### 4.1 Ptxas spill 调查

实验过程中一度怀疑新增控制流造成了较大的 local-memory spill。为避免用不同构建参数
误判，使用原始 `HEAD` header 和完全相同的 CUDA 12.8 flags 构建了独立 baseline
extension，再逐 specialization 对比 ptxas 输出。

结果表明 baseline 和 assist 版本的 stack/spill 数值一致。例如：

```text
no-split B128 DCP8/Hq8: baseline/new 2272 spill stores, 2276 loads
no-split B128 DCP2/Hq4: baseline/new  432 spill stores,  468 loads
```

因此观察到的 spill 是当前构建配置下原有问题，不是本实验引入的回退。为诊断临时添加
的 `__noinline__` helper 已删除，没有进入最终实验 patch。

## 5. 性能实测

### 5.1 测试环境和统计口径

- 8 x NVIDIA H100 80GB HBM3；
- CUDA 12.8；
- PyTorch 2.10.0+cu128；
- BF16，head dimension 128；
- eager，Mega kernel only，CUDA event；
- 每轮先取 DCP ranks 中的最大时间，再跨 iteration 统计 p50/p90；
- baseline 是使用原始 `HEAD` header、相同对象和编译参数构建的独立 extension。

focused case：

```text
B=3
Sq=[16,16,16]
Sk=[1139,44536,3167]
DCP=8
QH=32
KVH=1
Hq_local=4
BlockN=128
```

### 5.2 Split=16，num_comm_sm=8

| 指标 | Baseline | Assist | Assist 变化 |
| --- | ---: | ---: | ---: |
| kernel p50 | 81.328 us | 105.040 us | 慢 29.16% |
| kernel p90 | 83.811 us | 111.104 us | 慢 32.56% |
| attention -> history done p50 | 16.864 us | 16.832 us | 快 0.19% |
| attention -> history done p90 | 17.600 us | 17.440 us | 快 0.91% |
| receive done p50 | 43.328 us | 65.840 us | 慢 51.96% |

这里只增加了 32 个 assist warp，相对 1488 个 compute combine warp 为 2.15%。history
tail 几乎没有变化，但 receive 路径明显受干扰，导致 kernel 端到端回退。

### 5.3 Split=16，num_comm_sm=32

| 指标 | Baseline | Assist | Assist 变化 |
| --- | ---: | ---: | ---: |
| kernel p50 | 90.096 us | 83.984 us | 快 6.78% |
| kernel p90 | 93.290 us | 88.736 us | 快 4.88% |
| attention -> history done p50 | 24.624 us | 20.032 us | 快 18.65% |
| attention -> history done p90 | 25.283 us | 21.216 us | 快 16.09% |
| receive done p50 | 51.952 us | 46.176 us | 快 11.12% |

固定 `comm=32` 时，128 个 assist warp 相对 1200 个 compute combine warp 增加
10.67%，达到两个 split 性能门槛。

但是跨 `num_comm_sm` 比较时：

```text
comm=8  + no-assist kernel p50 = 81.328 us
comm=32 + assist    kernel p50 = 83.984 us，仍慢 3.27%

comm=8  + no-assist history tail = 16.864 us
comm=32 + assist    history tail = 20.032 us，仍慢 18.79%
```

原因是增加 communication CTA 会减少更多 compute warp：

```text
comm=8  + no-assist: 12 * (132 - 8)      = 1488 combine warps
comm=32 + assist:    12 * (132 - 32)+4*32 = 1328 combine warps
```

总 combine worker 数减少 10.75%，compute CTA 数从 124 降到 100。`comm=32` 的
assist 收益只能说明它可以补偿一部分已经划给通信的 SM，不能说明应主动把
`num_comm_sm` 从 8 增加到 32。

### 5.4 No-split，num_comm_sm=8

| 指标 | Baseline | Assist | Assist 变化 |
| --- | ---: | ---: | ---: |
| kernel p50 | 122.832 us | 122.736 us | 快 0.08% |
| kernel p90 | 126.566 us | 125.251 us | 快 1.04% |
| attention -> history done p50 | 8.384 us | 8.128 us | 快 3.05% |
| attention -> history done p90 | 9.248 us | 9.091 us | 快 1.70% |
| receive done p50 | 89.296 us | 89.184 us | 快 0.13% |

该 focused no-split case 在 2% 门槛内，但完整 trace 中仍出现 no-split 回退，说明单个
shape 不能代表实际 workload。

### 5.5 20-case trace，num_comm_sm=8

配置为 DCP2/Hkv4、QH32、auto split、BlockN128、warmup 20、iterations 40、eager。

| 指标 | Baseline | 无条件 Assist | 变化 |
| --- | ---: | ---: | ---: |
| 20-case mean p50 | 0.166878 ms | 0.170312 ms | 慢 2.06% |
| case p50 中位数 | 0.087912 ms | 0.091384 ms | 慢 3.95% |
| workload-weighted TFLOPS | 1103.945 | 1081.688 | 下降 2.02% |

主要回退：

| Case | Baseline p50 | Assist p50 | 回退 |
| --- | ---: | ---: | ---: |
| case003 | 0.100352 ms | 0.109680 ms | 9.30% |
| case005 | 0.080832 ms | 0.103952 ms | 28.60% |
| case010 | 0.468896 ms | 0.492672 ms | 5.07% |
| case011 | 0.481616 ms | 0.505808 ms | 5.02% |

其中 case010、case011、case019 为 no-split；再加上长耗时的 split case012，这四个
case 占 baseline 总 p50 时间约 51.3%。它们更容易受 compute CTA 数量下降或 post-Q
通信竞争影响。因此不能用 `comm=32` focused case 相对自身 baseline 的收益，推断
20-case 加权平均会受益。

## 6. 实现暴露的问题

### 6.1 Worker 交换比例不成立

每增加一个 communication CTA，会减少一个具有 12 个 combine warp 的 compute CTA，
但 communication CTA 只提供 4 个 assist warp。若提高 `num_comm_sm` 是为了获得
assist，本质上是用 12 个通用 compute/combine warp 换 4 个 post-Q combine warp。

assist 只有在 `num_comm_sm` 已因其他通信瓶颈而必须较大时，才可能作为补偿有意义；
它不应成为主动增加 `num_comm_sm` 的理由。

### 6.2 Receive 与 combine 争用同一个 communication CTA

虽然 shared-memory staging 已物理分离，4 个 receive pair 和 4 个 combine warp 仍
共享 CTA 调度、内存系统和 TMA 活动。`comm=8` split case 中 receive done p50 回退
51.96%，说明“shared memory 不重叠”并不等于“执行资源没有争用”。

### 6.3 Warp 角色转换使控制流和同步显著复杂化

为了保留 6 个 receive slot，需要让 combine warp 在 queue drain 后变成后两组
receive pair。这引入了：

- CTA 内 combine warp completion counter；
- assist/fallback 两套 warp-id 到 producer/consumer/slot 的映射；
- staging 生命周期和角色转换之间的隐式契约；
- 前 4 个 receive pair、combine group、后 2 个 receive pair 三类并发进度；
- phase timestamp 必须由不同 leader 独立提交。

这些状态没有表现出 correctness failure，但审查、维护和后续修改成本明显增加。

### 6.4 全局 completion participant 数变为条件式

assist 开启时 history completion 统计所有 CTA；关闭时只统计 compute CTA。所有 CTA
必须对 guard 得出完全一致的结果，communication CTA 也必须恰好贡献一次。任何计数
错误都会表现为过早 publish、timestamp 永不完成或 CUDA Graph replay hang，风险远高于
获得的平均性能收益。

### 6.5 Profitability guard 只覆盖 worker 数量

10% guard 来自 `comm=8` 与 `comm=32` 的单 focused shape。它没有考虑：

- history task 数、task 大小和 actual split；
- attention 完成时间分布；
- receive ready 顺序和 backpressure；
- Q all-gather/receive 实际通信量；
- no-split copy 的 HBM/L2 行为；
- 不同 DCP、BlockN 和 ragged shape。

因此即使 guard 通过 correctness，也不能证明完整 trace 一定有收益。为这个启发式保留
两套复杂执行路径不符合本次实验的收益/复杂度比例。

### 6.6 性能 telemetry 没有完整反映 assist workers

实验 JSON 中的 `history_combine_worker_warps` 来自既有 host-side 统计，只记录 compute
CTA 的 worker 数，没有把 post-Q communication assist warp 纳入。实验分析时必须手工
补上 `4 * num_comm_sm`。如果保留实现，还需要同步修改 telemetry 语义，否则性能报告
容易误导；本实验禁止 ABI/metadata 扩展，因此没有为此增加新字段。

## 7. 最终决策和回退范围

本实验不合入功能实现，回退以下内容：

- communication CTA 的 post-Q 4 receive pairs + 4 combine warps 布局；
- combine warp 到 receive slot 4、5 的角色转换；
- compute/communication CTA 共享 history queue 的调用路径；
- communication CTA 专用 combine staging offset；
- communication CTA history/publish completion 计数；
- 动态 history completion participant count；
- 10% worker-ratio profitability guard；
- 对应 CPU queue simulation test 和 receive 常量修改。

回退后恢复原始行为：Q all-gather 后，每个 communication CTA 使用 warp 0..5 作为
6 个 receive producers，warp 6..11 作为 6 个 receive consumers；history combine/copy
只由 compute CTA 的 12 个 warp 执行。

本实验从始至终未改变公开 API、params、metadata v5、workspace 或 ABI，因此回退不
需要数据格式迁移。

## 8. 后续重新评估的前置条件

只有出现以下证据时才建议重新考虑类似方案：

1. 生产配置因 Q/receive 通信瓶颈本来就需要较大的 `num_comm_sm`；
2. 完整 trace 证明 post-Q communication CTA 存在稳定、足够长的空闲窗口；
3. assist 可以不引入 warp 角色转换，或能用明显更简单的 ownership 协议实现；
4. A/B 必须比较 `comm=N + assist` 与当前生产最优 `comm=M + no-assist`，不能只比较
   相同大 `comm=N` 下的局部收益；
5. split history tail、kernel p50、no-split 和每个 trace case 同时达到既定门槛。

优先级更高的方向仍应是减少 history task/atomic 成本、改善 split cost model，或在
不减少 compute CTA 的前提下缩短 combine tail。

