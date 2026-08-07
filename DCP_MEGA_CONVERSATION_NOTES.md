# DCP Mega 实现与技术记录

> 状态：已实现并通过静态、编译和 8-GPU 正确性验证
>
> 更新日期：2026-08-05
>
> 适用范围：当前仓库中的 experimental `dcp_mega_varlen` forward 路径

本文档记录 DCP Mega 当前已经落地的实现，而不是未来设计草案。重点包括：

- causal chunk attention 与 noncausal history attention 共用一个 persistent kernel；
- history PackGQA Q tile 使用 TMA，chunk PackGQA Q tile 继续使用 cp.async；
- 通信 CTA 在 Q all-gather 后既负责 receive，也可以参与 history combine；
- history combine 内直接执行 remote TMA store 和 ready publish，不存在单独的 publish pass；
- history Q TMA 尾部允许 speculative overfetch，但 `q_ready` 仍只覆盖有效 packed rows。

2026-08-04 版本中的 TMA 可行性推理仍有部分背景价值，但其中“尚未修改代码”“两条 Q 路径都需要改为 TMA”“TMA footprint 必须扩大 `q_ready` dependency”和“publish 调度尚待核查”等描述已经过时，以本文当前状态为准。

## 1. 当前范围与约束

DCP Mega 仍是一个窄范围的 Hopper 特化：

- GPU：SM90；
- dtype：BF16；
- head dimension：128；
- forward only；
- packed varlen；
- `Hkv_group == 1`；
- `DCPSize in {2, 4, 8}`；
- `Hq_local in {4, 8}`；
- `BlockN in {128, 176}`；
- PackGQA 固定开启；
- 支持 split-capable 和 no-split 两类 kernel 实例；
- Q/O 通信 tile 固定为 `[16, Hq_local, 128]`。

本次 history Q-TMA 改动没有改变 Python/CUDA 公共 API、metadata header ABI、`q_group` 布局或这些输入约束。共享 mainloop 的其他 BSHD、varlen、ring 和普通 PackGQA 实例依靠默认关闭的 opt-in 保持原行为。

关键输入和中间布局为：

```text
local Q:   [total_q, Hq_local, 128]
q_group:   [capacity_q, DCPSize * Hq_local, 128]
chunk K/V: [total_q, 1, 128]
history K/V:
           [total_history_on_rank, 1, 128]
```

chunk attention 是 causal，读取本 rank 的 local Q 和 chunk K/V。history attention 是 noncausal，读取 all-gather 后的 `q_group` 和本 rank 的 history K/V。

## 2. 总体任务与依赖图

一次 persistent launch 使用 `num_sms` 个 CTA。前 `num_comm_sm` 个 CTA 是通信 CTA，其余是计算 CTA：

```cpp
bool const communication_cta = int(blockIdx.x) < params.num_comm_sm;
```

当前执行关系可以概括为：

```text
communication CTA:
Q all-gather
  -> 优先处理已经 ready 的 remote receive
  -> 暂无 ready receive 时，尝试领取 ready history-combine task
  -> history combine + remote TMA store + direct publish
  -> 重复，直到 receive 和 combine 两个队列都完成

compute CTA:
unified chunk/history attention queue
  -> 领取剩余 history-combine task
  -> history combine + remote TMA store + direct publish
  -> final combine

每个 history 数据块的依赖：
q_ready(valid packed rows)
  -> history attention_done
  -> history combine + publish_ready/tile_ready
  -> receive_ready
  -> final combine
```

这不是一个全局严格分段的：

```text
all attention -> all publish -> all receive -> all final combine
```

通信 CTA 可以在计算 CTA 仍执行 attention 时参与已经 ready 的 history combine，也可以接收对端已经 publish 的 tile。计算 CTA 自身仍先耗尽统一 attention queue，再帮助清空 history-combine queue，最后执行 final combine。

## 3. Metadata 与任务粒度

metadata ABI 仍使用 40 个 `int32_t` 的 `MetadataHeader`。主要 descriptor 粒度如下：

| descriptor / signal | 粒度 | 作用 |
|---|---|---|
| `AttentionWorkDesc` | sequence x 128 packed-Q rows x split | 描述 chunk 或 history attention 工作和 completion ID |
| `QTaskDesc` | source rank x 16 tokens | 将每个 rank 的 Q 搬到 `q_group` |
| `PublishWorkDesc` | destination rank x 16 tokens x `Hq_local` | combine 对应 history contribution，并直接发布给目标 rank |
| `FinalWorkDesc` | 16 tokens x `Hq_local` | 合并 local chunk、local history 和所有 remote history |
| `q_ready` | 16-token block | 表示该 block 的各 rank Q 已写入 `q_group` |
| `attention_done` | attention descriptor | 发布 chunk/history attention completion |
| `publish_ready` | local publish descriptor | 表示本 rank 的 local history contribution 已 combine |
| `tile_ready` | source rank x 16-token block，位于目标 rank IPC arena | 表示 remote history tile 已写入并可接收 |
| `receive_ready` | final tile x remote source | 表示 remote tile 已落入本地 receive workspace |

attention descriptors 仍按 chunk 在前、history 在后的顺序生成，completion ID 稠密且唯一。每个计算 CTA 先按 CTA ID 领取一个 initial descriptor，之后通过 `kAttentionDynamicCounter` 领取剩余 descriptor。

chunk 和 history 使用同一个 scheduler 队列，但 descriptor 的 `kind` 决定：

- 选择 causal chunk mainloop 还是 noncausal history mainloop；
- 选择对应的 sequence split 数；
- 是否等待 `q_dependencies`；
- completion 写入哪个 `attention_done[completion_id]`。

只有 history descriptor 等待 gathered Q；chunk descriptor 的 `q_dependency_count` 始终为 0。

### 3.1 Split 与 PackGQA 的选择粒度

`PackGQA=true` 是整个 launch 的 compile-time 选择，不是逐序列选择。

一次 launch 也只选择一个 `Split=true` 或 `Split=false` kernel 实例。选择 split-capable 实例时，各序列仍可通过 `chunk_sequence_splits[]` 和 `history_sequence_splits[]` 使用不同的实际 split 数，包括 1。

当前动态 split 估算使用：

```text
m_i = ceil(q_i * heads / 128)
n_i = ceil(k_i / BlockN)

total_blocks = sum_i(m_i * n_i)
blocks_per_sm = max(ceil(1.1 * total_blocks / num_sms), 1)
S_i = clamp(ceil(n_i / blocks_per_sm), 1, split_upper_bound)
```

PackGQA 下 scheduler head 数折叠为 1。chunk 的 `heads=Hq_local`，history 的 `heads=DCPSize * Hq_local`；两者分别计算逐序列 split，launch 使用二者的有效上界选择模板实例。

## 4. 通信 CTA 与计算 CTA 的职责

### 4.1 通信 CTA

通信 CTA 完全不初始化 FlashAttention pipeline。其控制流是：

```text
run_q_allgather()
  -> record q_allgather_done
  -> run_communication_post_q()
  -> record kernel_done
  -> return
```

`run_q_allgather()` 对每个 16-token tile：

1. 从 source rank 的 IPC Q allocation 发起 TMA load 到 shared memory；
2. 从 shared memory 发起 TMA store 到本地 `q_group` 的 source-rank head 槽；
3. 等待 store 完成；
4. 执行 `fence.proxy.async.global`；
5. 对对应 `q_ready` 项执行 release signal。

history scheduler 使用 acquire wait，要求对应 `q_ready` 累积到 `DCPSize`，之后才允许该 history descriptor 发起 Q load。

### 4.2 计算 CTA

计算 CTA 的控制流是：

```text
unified chunk/history attention
  -> __syncthreads()
  -> run_history_combine()
  -> run_final_combine()
```

attention pipeline 结束后，CTA 全体线程同步，随后复用同一 dynamic shared-memory 区域执行 combine。计算 CTA 通过共享的 `kHistoryCombineCounter` 清空通信 CTA 未完成的 history-combine 工作。

final combine 只由计算 CTA 执行。通信 CTA 不进入 FlashAttention mainloop，也不进入 final-combine queue。

## 5. 通信 CTA 参与 History Combine

### 5.1 Receive-first 策略

`run_communication_post_q()` 同时推进 receive 和 history combine，但优先 receive：

1. 每个 communication chunk 扫描自己负责的 receive tasks；
2. 若发现对端 `tile_ready` 达到当前 monotonic phase，立即执行 receive；
3. 只要本轮存在 ready receive，整个 CTA 先处理 receive，然后重新扫描；
4. 若没有 ready receive，则调用 `try_run_ready_history_combine()`；
5. receive 或 combine 尚未完成且无即时工作时，短暂 `__nanosleep(64)` 后重试。

所以“通信 CTA 参与 combine”不是额外 kernel，也不是 Q all-gather 中插入的路径，而是 Q all-gather 完成后的 post-Q loop 中的 fallback 工作。

### 5.2 共享 combine queue

通信 CTA 和计算 CTA 共享 `queue_state[kHistoryCombineCounter]`。

计算 CTA 的 `run_history_combine()` 使用 `atomicAdd` 直接领取 ticket；领取后等待该任务的全部 history `attention_done` dependencies。

通信 CTA 使用非阻塞的 `try_run_ready_history_combine()`：

- 先读取队首 ticket，不立即消费；
- 检查该 publish descriptor 的全部 history attention dependencies；
- 只有 ready 时才用 compare-exchange 消费 ticket；
- 成功后执行一个完整 combine task。

返回值语义为：

```text
publish_id >= 0 : 成功执行一个 ready history-combine task
-1              : 队首 task 尚未 ready
-2              : 所有 combine tickets 已被领取
```

当前通信 helper 只检查队首任务，不越过未 ready 的队首去扫描后续 combine task。这是当前明确的调度行为，不影响正确性，但可能影响不同序列完成时间差很大时的 overlap。

## 6. History Combine 与 Direct Publish

`run_history_combine_task()` 把 combine 和 publish 融合在同一个任务内。

对 `PublishWorkDesc(dst_rank, vector_begin, valid_vectors, ...)`：

1. 等待该 16-token tile 所需的 history attention completion IDs；
2. 对每个有效 `(token, local_head)` 读取目标 `history_head = dst_rank * Hq_local + local_head`；
3. 若 history kernel 使用 split，读取该序列的实际 split 数和 partial O/LSE；
4. 使用 numerically stable 的 LSE 加权公式合并各 split；
5. 将 BF16 O 写入 shared communication tile，将 FP32 combined LSE 写入 send workspace；
6. 从 shared communication tile 发起 TMA store 到 `history_send_local[dst_rank]`；
7. 等待 remote-visible store 完成；
8. 发布对应 ready 状态。

ready 发布分两类：

- `dst_rank == dcp_rank`：对本地 `publish_ready[publish_id]` 执行 release store；
- remote destination：对目标 rank 的 IPC `tile_ready` arena 执行 system-scope release store，值为当前 monotonic phase。

因此没有单独的 publish descriptor 执行阶段或 publish kernel。metadata 中沿用 `PublishWorkDesc`、`publish_count` 和 `publish_dependencies` 命名，但运行时该任务就是“history combine + TMA store + ready release”的完整单元。

计时中的 `history_combine_done` 和 `publish_done` 有意记录同一个时间戳：

```text
publish_done = 所有 remote ready release 已发出
```

它不表示另一个独立 publish pass 已结束。

## 7. Receive 与 Final Combine

receiver 使用当前 rank 的 IPC `tile_ready` 视图，按 source rank 和 token block 轮询 system-scope ready phase。ready 后：

1. 读取 source rank 发布的 FP32 history LSE；
2. 对 source rank 的 history O 发起 remote TMA load 到 shared memory；
3. 把 O 从 shared memory TMA store 到本地 `history_receive_o[source]`；
4. 写本地 `history_receive_lse[source]`；
5. 等待本地 store 完成并执行 `fence.proxy.async.global`；
6. release `receive_ready[task_id]`。

每个 `FinalWorkDesc` 在计算前等待三类条件：

- 对应 local chunk attention 的全部 `attention_done`；
- 本 rank local history contribution 的 `publish_ready`；
- 其他 `DCPSize - 1` 个 source 的全部 `receive_ready`。

随后 final combine 用同样的 stable LSE-weighted 方式合并：

```text
local causal chunk
+ local noncausal history
+ every remote noncausal history contribution
-> final_o / final_lse
```

无效 tail vectors 通过 `valid_vectors` predicate 排除，不写入有效输出之外的行。

## 8. History PackGQA Q-TMA

### 8.1 Compile-time opt-in

`CollectiveMainloopFwdSm90` 新增了 trailing、默认关闭的模板参数：

```cpp
bool UseTmaPackGQAQ_ = false
```

它只在 `PackGQA` 且没有 Qv 时有效。Mega config 使用：

```text
Mainloop<true>  / causal chunk     : UseTmaPackGQAQ = false
Mainloop<false> / noncausal history: UseTmaPackGQAQ = true
```

因此当前实际模式是：

| attention domain | Q source | Q load |
|---|---|---|
| chunk | local `q` | `PackGQAManager::load_Q()` + cp.async |
| history | gathered `q_group` | packed-Q TMA |

这项改动仅针对 Mega history。普通 PackGQA、BSHD、varlen 和 ring mainloop 不会因为默认模板参数而改变。

### 8.2 Descriptor 布局

Mega 的 `Hkv_group == 1`，因此当前 attention group 的所有 Q heads 在物理内存中连续：

```text
Q[token][head][d]
address = ((token * G + head) * 128 + d)

chunk G   = Hq_local
history G = DCPSize * Hq_local
```

history 使用已有的逻辑 `ShapeQPacked` / `StrideQPacked`：

```text
((G, total_q), 128, Hkv=1, batch=1)
```

CuTe TMA descriptor 构建时把连续的 `(G, total_q)` 暴露成一个 active row extent：

```text
[total_q * G, 128]
```

descriptor 通过 `make_tma_copy()`、`SmemLayoutQ` 和
`select<0, 2>(TileShape_MNK{})` 构建。active extent 使用 `total_q * G`，不是 IPC allocation 的 capacity。

### 8.3 Device tile 起点与传输大小

history tile 的 packed row 起点是：

```text
packed_offset
  = seqlen_info.offset_q * G
  + m_block * 128
```

每次都发起完整的：

```text
128 rows x 128 BF16
= 32768 bytes
```

实现不要求 `q_len * G` 整除 128，也不假定尾 tile 固定有 64 个有效 packed rows。最后一个全局 tile 超过 descriptor active extent 的部分由 TMA OOB zero-fill。

## 9. cp.async / TMA 混合 Barrier 协议

chunk 和 history 会在同一个 Hopper shared pipeline 中交替出现，所以两条 Q-load 路径必须满足同一个 producer hand-off 和 shared-storage contract。

当前不变量为：

- chunk 和 history 都保留 `NumProducerThreads = 128`；
- 两者的 `QueryEmpty` arrival count 都是 `NumMmaThreadsQK + 128`；
- 只要任一路径使用 TMA，fused kernel 的 Q barrier 类型就是 `ClusterTransactionBarrier`；
- Q barrier 固定以 128 arrivals 初始化；
- Q shared-memory swizzle alignment 在 PackGQA cp.async 和 opt-in TMA 路径间保持一致；
- chunk/history 的 `TensorStorage` size、alignment 和 Q/K/V member offset 必须完全一致。

chunk tile 的 producer 行为保持原样：

```text
128 producers:
  QueryEmpty sync
  -> PackGQAManager::load_Q()
  -> cpasync_barrier_arrive()
  -> barrier_Q.arrive()
```

history tile 的 producer 行为是：

```text
128 producers:
  QueryEmpty sync
  -> one elected thread:
       barrier_Q.arrive_and_expect_tx(32768)
       issue packed-Q TMA
  -> other 127 producers:
       barrier_Q.arrive()
```

fused Mega kernel 的 static assertions 明确要求 chunk 为 cp.async、history 为 packed-Q TMA，并同时检查 producer 数、named-barrier count、Q barrier arrivals、transaction bytes、shared layouts、storage size/alignment 和成员 offset。这样 phase 从 chunk 切到 history 或从 history 切回 chunk 时仍使用同一套合法的 hand-off。

## 10. Speculative Overfetch 与 `q_ready` 不变量

这是 history Q-TMA 正确性的核心约定。

对某条序列的 history M tile，数学上有效的 packed rows 只有：

```text
[m_block * 128,
 min((m_block + 1) * 128, q_len * G))
```

metadata 只把这些有效 rows 映射到现有 16-token `q_ready` blocks。它不会因为 TMA 总是读取完整 128-row footprint 而增加下一序列的依赖。

因此一个尾 tile 可以发生：

```text
有效 rows:
  已被 q_ready acquire wait 覆盖

当前序列尾部之后、仍位于 descriptor active extent 内的 rows:
  speculative overfetch
  可能属于下一序列
  可能尚未 ready
  允许读取任意值

超过最后一个全局 active row 的部分:
  TMA OOB zero-fill
```

不扩大 dependency 的依据不是“overfetch 数据一定为零”，而是这些数据只落在当前 tile 的无效 M rows。正确性依赖以下既有语义继续成立：

- attention mask 按 row 排除序列范围外的 Q rows；
- softmax 和 QK/PV 计算不进行跨 Q-row reduction；
- split history combine 只遍历 metadata 标记的有效 vectors；
- `store_O()` 和 `store_LSE()` 使用有效-row predicate；
- final combine 使用 `valid_vectors`，不会把无效 tail 写回有效输出。

后续修改不得让 speculative rows 进入跨-row reduction、completion dependency 推导或有效输出写回。若未来引入这类行为，就必须重新设计 readiness 或 padding，而不能继续依赖当前约定。

这个设计刻意不采用以下方案：

- 不按 128 packed rows 给每条 sequence 增加物理 padding；
- 不让尾 tile 回退到 cp.async；
- 不把 `q_dependencies` 扩大到完整 TMA footprint；
- 不增加 metadata 字段或 workspace capacity。

## 11. 当前调度事实与性能含义

已经确认的正确性事实：

- history attention 可以按自己的有效 Q dependencies 提前开始，不等待全局 Q all-gather 完成；
- history combine task 按自己的 history completion IDs gate；
- 通信 CTA 可以在其他 attention descriptor 仍运行时执行 ready combine；
- remote store 和 ready release 在同一个 combine task 中完成；
- receiver 可以在其他 rank 仍计算其他 tile 时拉取已发布 tile；
- final combine 仍由计算 CTA 在其 attention phase 结束后执行，并逐 tile 等待 chunk/local-history/remote-history 依赖。

这提供了细粒度 overlap，但不等于每个可运行 task 都会立刻被调度：

- compute CTA 在耗尽 attention queue 前不会帮助 combine；
- communication CTA 优先 ready receive；
- communication combine helper 只检查共享 combine queue 的队首；
- final combine 要等计算 CTA 完成其 attention loop 后才开始领取。

这些选择已经通过正确性验证，但其性能优劣仍应通过独立 benchmark 判断。

一次完整 Q TMA tile 和原 cp.async 完整 tile 都搬运 32 KiB。history Q-TMA 的预期收益主要来自减少 packed-row `divmod`、逐行指针计算、warp shuffle 和 producer copy 指令，而不是减少有效 HBM 字节数。ragged tail overfetch、split 重复 Q load、barrier 成本、`num_comm_sm`、receive-first 策略和 combine 队首阻塞都可能影响最终收益。

## 12. 验证状态

history Q-TMA 实现后按“静态测试和完整编译通过，再运行 GPU”的顺序完成了验证。

### 12.1 CPU metadata

```bash
python -m unittest scripts/test_min_fa3/test_dcp_mega_metadata.py
```

结果：14 tests passed。

新增 tail dependency case 在同一个测试中覆盖 24、64、120 个有效 packed rows，确认：

- 完整 128-row TMA footprint 即使越过当前序列尾部，也不会引用下一 `q_ready` block；
- history dependency 只覆盖有效 packed rows；
- chunk dependency 仍为空。

### 12.2 完整编译

```bash
make -j2
```

结果：NVCC/PTXAS 编译和最终链接通过，共 17 个目标；覆盖 Mega 的：

- split / no-split；
- `BlockN=128 / 176`；
- 共享 mainloop 的其他现有实例。

编译期 static assertions 同时确认 Mega chunk 仍是 cp.async Q load、history 是 TMA Q load。

### 12.3 8-GPU 正确性 matrix

```bash
PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
  scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py --matrix
```

四个 case 均通过，每个执行两轮：

```text
dcp2_h4_bn128_split1
dcp4_h8_bn176_split2
dcp8_h4_bn176_auto
dcp2_h8_bn128_split2_tail
```

matrix 覆盖：

- DCP 2/4/8；
- `Hq_local` 4/8；
- split/no-split；
- `BlockN` 128/176；
- ragged sequence tail；
- output 和 LSE reference；
- prepared replay；
- monotonic phase wrap；
- CUDA Graph replay；
- fused `history_combine_done == publish_done` timing invariant。

本文档更新本身只修改 Markdown，因此没有重复运行 CUDA 构建或 8-GPU matrix。

## 13. 关键源码位置

- `dcp_mega_metadata.py`
  - dispatch、逐序列 split、attention/Q/publish/final descriptor 和 dependency 生成；
  - history TMA tail 只对有效 packed rows 建立 `q_ready` dependency。
- `include/dcp_mega_min_fa3_varlen_params.h`
  - metadata ABI、descriptor 定义、ready/completion workspace。
- `include/dcp_mega_min_fa3_varlen_scheduler.h`
  - unified attention queue、history Q acquire wait、`attention_done` release。
- `include/dcp_mega_min_fa3_varlen_launch.h`
  - Mega mainloop 特化；
  - Q all-gather；
  - communication-assisted history combine；
  - direct TMA publish；
  - remote receive、final combine 和 persistent CTA 分工。
- `include/dcp_mega_min_fa3_kernel.h`
  - chunk/history mainloop 切换；
  - mixed Q barrier 类型、初始化和 shared-storage compatibility assertions。
- `include/min_fa3_mainloop.h`
  - 默认关闭的 `UseTmaPackGQAQ`；
  - packed-Q TMA descriptor、tile offset 和 mixed cp.async/TMA producer protocol。
- `include/min_fa3_prologue.h`
  - 通用 Q barrier 使用显式 `QBarrierArrivalCount` 初始化。
- `scripts/test_min_fa3/test_dcp_mega_metadata.py`
  - valid-row `q_ready` dependency 和 metadata invariants。
- `scripts/test_min_fa3/test_dcp_mega_varlen_multi_rank.py`
  - 多 rank output/LSE、replay、phase 和 CUDA Graph matrix。

## 14. 后续工作边界

当前 correctness implementation 已完成。后续工作应作为独立的性能阶段，不应再把下列事项描述成 correctness blocker：

- history Q-TMA 相对 cp.async 的 H200 kernel-time 收益；
- ragged tail overfetch 对带宽的影响；
- split 场景下重复 Q load 的代价；
- 最优 `num_comm_sm`；
- receive-first 与 combine-first 的调度权衡；
- combine 队首未 ready 时是否值得扫描后续任务；
- phase timestamp 中 attention、combine/直接 publish、receive、final combine 的实际重叠程度。

任何性能优化都必须继续保持以下三个已验证不变量：

1. chunk Q 保持当前 cp.async 路径，除非另有独立设计和验证；
2. history speculative tail rows 不参与有效输出或跨-row reduction；
3. remote TMA store 完成、ready release、receiver acquire、local receive store 和 `receive_ready` 之间的内存可见性顺序不能削弱。

## 15. Case007 history combine 优化实现补记（2026-08-06）

本轮按 `DCP_MEGA_PERFORMANCE_OPTIMIZATION.md` 的计划完成了第一版实现，范围严格限制在 DCP Mega history combine 及其 metadata/dispatch/test 配套：

- metadata 升级到 v3，但 header 保持 40 个 int；新增逐 `(dst_rank, vector)` 的 8-int history combine descriptor；
- publish dependency 从 tile union 改为每个 vector 的精确 split completion IDs；
- 12 个 compute warps 独立领取 vector task，任务循环只使用 warp sync；
- split>1 使用 FA3 风格 normalized LSE weights、128-bit partial-O `cp.async` 和 4-stage pipeline；
- split==1 直接复制现有 BF16 O/LSE；
- 结果直接写 IPC send O/LSE，移除旧 history combine 的 shared BF16 tile 和 producer TMA store；
- `publish_ready` 改为 tile completion counter，最后一个 warp 以 device acq_rel RMW、system fence、remote system release 发布 ready；
- communication CTA 改为 receive-only；combine queue 和 final combine 之间仅保留一次 CTA convergence barrier；
- split kernel 增加 32/64/128 编译期 combine bucket，nonsplit 使用 bucket 1；
- runner API 和 CUDA binding tensor 参数保持不变。

静态结果：metadata unit test 14/14、py_compile、`git diff --check`、`make -j2` 和扩展 import 均通过。`cuobjdump` 显示 split kernel 为 168 registers/thread、`LOCAL:0`，三个 bucket 的 register count 相同；SASS 确认 128-bit `LDGSTS`、strong GPU atomic 和 warp sync。

case007 的纯 CPU metadata 检查同样通过：auto 实际 splits 为 `[1,22,2]`，split16 为 `[1,16,2]`，两者都覆盖 1536 个 `(dst_rank, vector)` combine tasks。

GPU 结果尚未产生。正确性 matrix 已加入 case007 auto/split16，但两次启动都没有进入 kernel：第一次暴露并修复了脚本 repo-root import 问题；第二次被 22:11 后启动的 root-owned `pretrain_gpt.py` 抢占 8 张 Exclusive Process H100，失败于 `torch.cuda.set_device()`。为遵守“静态优先、尽量减少 GPU 测试次数”，没有在占卡状态继续重试，也没有运行性能 sweep、`simple_bench.sh` 或 Nsight。

后续验收顺序固定为：一次六-case matrix，通过后一次 case007 split/comm sweep，再运行一次完整 `simple_bench.sh`。在此之前，本文只记录实现和静态证据，不把性能目标写成已达成结果。
