# Mega Ring Hybrid Ready-Once 设计与审查总结

## 1. 文档目的

本文档记录 `mega ring CP hybrid` ready-once 路径的设计目标、当前实现、代码审查结果、已知风险、建议修复顺序和 H100 验证计划。

本文档对应的代码状态为：

- 基线提交：`1f167d7 dynamic role switch for comm SMs to comp SMs`
- 开发分支：`ready-once`
- 基线分支 `main` 不包含本文档所述的未提交 ready-once 工作区修改
- ready-once 分支保留 compact KV、连续前缀计数器、local-first 调度和 causal front-first 调度实现

本文档的主要目标不是证明当前实现已经可以上线，而是明确：

1. 当前方案解决了什么问题。
2. 当前实现中哪些部分是正确且值得保留的。
3. 哪些并发问题可能导致死锁或错误结果。
4. 哪些实现细节会削弱预期性能收益。
5. 后续应按什么顺序修改和验证。

## 2. 原始性能问题

旧的 mega-ring 路径把一个 CP 序列拆成多个 ring step。每个 Q tile 在每个 step 上分别执行一段 attention，然后在 epilogue 中执行在线归并：

1. 从全局内存读取上一个 step 的 O 和 LSE。
2. 将当前 step 的局部 O/LSE 与历史结果合并。
3. 把新的 O/LSE 写回全局内存。
4. 使用 `mega_ring_step_ready` 串行化同一个 Q tile 的多个 step。

这种方式有三个主要问题：

- 同一个 Q tile 被重复调度和重复执行 epilogue。
- 每个 step 都产生 O/LSE 的额外全局内存读写。
- 同一个 Q tile 的 step reduction 需要严格排序，产生等待和 semaphore 开销。

Ready-once 的目标是：

- 每个 Q tile 只计算一次。
- 计算开始前，确保该 Q tile 依赖的连续 KV 范围已经写入本地 compact KV buffer。
- 使用普通 FA3 softmax 和直接 epilogue，一次性产生最终 O/LSE。
- local-only 序列优先，覆盖 kernel 启动初期的通信延迟。
- causal zigzag 中先计算 front Q，再计算依赖更长的 back Q。
- 通信 CTA 完成通信任务后转入公共计算任务队列。

## 3. 当前实现概览

### 3.1 Ready-once 模式开关

Python API 在 `min_fa3_op.forward_varlen_mega_ring()` 中新增 `ready_once` 参数，hybrid 模式默认启用。

当前只有在同时满足以下条件时才实际启用 ready-once：

```text
use_ready_once = ready_once && hybrid_mode
```

没有 `global_seqlens_host` 的 legacy all-CP 路径仍使用旧的 step/reduction 实现。

### 3.2 Compact KV buffer

Ready-once 不再直接让 FA3 mainloop 把 KV 看成 `[world_size][local_total_k]` 的 rank-block 布局，而是为当前 rank 构造一个按 attention 访问顺序排列的 compact KV buffer。

Noncausal CP 序列的 compact 顺序为：

```text
rank 0 KV, rank 1 KV, ..., rank W-1 KV
```

Causal zigzag CP 序列的 compact 顺序为：

```text
front(rank 0), front(rank 1), ..., front(rank W-1),
back(rank W-1), back(rank W-2), ..., back(rank 0)
```

Local-only 序列只在 compact buffer 中保留本 rank 的本地 KV。

Host 侧为 compact buffer 重新构造 `cu_seqlens_k`，并把本 rank 已有的 KV 通过 `pack_ready_once_local_kv_kernel` 写入 compact buffer。远端 KV 由 fused kernel 内的通信 CTA 直接写入相应 compact row。

### 3.3 连续就绪前缀

每个依赖区间维护以下元数据：

- `chunk_done[interval][chunk]`：chunk 内已经完成的 K/V row 写入次数。
- `ready_end[interval]`：从 compact interval 起点开始，已经完全就绪的连续 row 数。
- `publish_lock[interval]`：推进 `ready_end` 时使用的设备锁。
- `ready_interval_rows[interval]`：该 interval 的有效 row 总数。

一个 KV row 的 K 和 V 都完成后，该 row 对 `chunk_done` 的总贡献为 2。只有一个 chunk 的计数达到 `2 * rows_in_chunk`，该 chunk 才被视为完成。

计算 CTA 根据 Q tile 的 causal 范围计算 `required_end`，然后等待：

```text
ready_end[interval] >= required_end
```

该 acquire/release 链用于保证计算 CTA 在读取 compact KV 前可以观察到通信 CTA 的 TMA store。

### 3.4 Ready-once 工作流

Ready-once scheduler 把工作划分成以下 work kind：

```text
0: local-only
1: noncausal CP full
2: causal CP front
3: causal CP back
```

全局任务顺序为：

```text
local-only tiles -> CP front tiles -> CP back tiles
```

Noncausal 模式没有 front/back 划分：

```text
local-only tiles -> CP full tiles
```

Ready-once 模板把 epilogue 的 reduction 编译期关闭，因此每个 Q tile 直接写出最终 O/LSE，不再读取和合并旧结果。

### 3.5 通信 CTA 动态转计算 CTA

Kernel grid 由两类初始 CTA 组成：

```text
[0, num_comp_sm): compute CTA
[num_comp_sm, num_comp_sm + num_comm_sm): communication CTA
```

通信 CTA 完成所有分配给自己的 remote-load task 后，重新初始化共享内存并调用 attention kernel body。它通过 `get_initial_work_from_queue()` 从全局 tile semaphore 获取尚未领取的计算任务。

初始计算 CTA 使用 block index 领取 `[0, num_comp_sm)` 的 tile。后续公共队列从 `virtual_grid_dim_x == num_comp_sm` 开始，因此正常情况下不会与初始 tile 重复。

## 4. 当前实现中值得保留的部分

### 4.1 一次 attention 和一次 epilogue 的方向正确

Compact KV 让 FA3 mainloop 可以把一个 Q tile 的所有有效 KV 当作一个连续 attention 问题处理。相比逐 step reduction，这直接消除了重复 O/LSE read-modify-write，是本次优化最核心的收益来源。

### 4.2 Local-only 序列优先调度已经实现

Scheduler 会先过滤并解码 local-only batch，再进入 CP stream。这符合使用本地计算覆盖通信启动延迟的目标。

### 4.3 Causal front 和 back 的位置映射基本一致

当前 compact causal 排列、Q row offset、有效 `seqlen_q`、有效 `seqlen_k` 和 bottom-right causal mask 的组合在设计上是一致的：

- front Q 使用本地序列前半。
- back Q 使用本地序列后半。
- front 和 back 分别使用与其全局 zigzag 位置对应的 KV prefix。

### 4.4 通信完成后转计算的队列起点设计基本合理

动态转换 CTA 不重新使用自己的 `blockIdx.x` 作为初始 tile，而是从全局队列领取 `num_comp_sm` 之后尚未分配的任务。这比直接把通信 CTA 映射为固定计算 tile 更安全，也能适应不同通信完成时间。

## 5. 高优先级正确性问题

### 5.1 `ready_end` 发布存在 lost wakeup

相关代码：

- `include/mega_ring_min_fa3_varlen_ring_launch.h`
- `try_advance_ready_end()`
- `signal_ready_once_row()`

当前逻辑在获取 `publish_lock` 失败时直接返回：

```cpp
if (atomicCAS(params.publish_lock + interval_id, 0, 1) != 0) {
    return;
}
```

只有完成某个 chunk 的最后一次 signal 才会尝试推进 `ready_end`。以下时序会丢失唤醒：

1. 后面的 chunk B 完成并获得锁。
2. B 的线程从当前 `ready_end` 开始扫描，发现前面的 chunk A 尚未完成。
3. A 的最后一次 signal 与 B 的扫描并发发生。
4. A 的线程尝试获取锁，但因为 B 仍持锁而失败，然后直接返回。
5. B 释放锁，但不会重新扫描。
6. A 和 B 实际都已经完成，但 `ready_end` 永久停留在 A 之前。
7. 等待更大 `required_end` 的计算 CTA 永久自旋。

这是功能性死锁，不是偶发的性能抖动。

#### 最小修复

完成 chunk 的线程必须最终获得发布锁，例如：

```cpp
while (atomicCAS(lock, 0, 1) != 0) {
    __nanosleep(64);
}
scan_and_advance_ready_end();
atomicExch(lock, 0);
```

发布区间很短，通信 CTA 数也较少，因此首先使用可证明正确的自旋锁方案更合适。确认 correctness 后再考虑 lock-free CAS。

#### 更优的无锁方向

可以把 `ready_end` 作为单调 CAS cursor：

1. acquire 读取当前 cursor。
2. 从 cursor 开始扫描连续完成的 chunk。
3. 使用 CAS 把 cursor 从旧值推进到新值。
4. CAS 失败时基于新的 cursor 重试。

无锁实现必须保持 TMA store -> release chunk counter -> acquire scan -> release ready_end -> acquire consumer 的传递顺序。

### 5.2 Grid oversubscription 可能阻止通信 CTA 驻留

当前 launch grid 为：

```text
num_comp_sm + num_comm_sm
```

但绑定只检查参数为正和 int32 范围，没有检查 kernel 的最大 resident CTA 数。

可能的死锁时序为：

1. 计算 CTA 先占满所有 resident block slot。
2. local-only 任务完成后，计算 CTA 开始等待远端 KV。
3. 通信 CTA 仍未被调度。
4. 计算 CTA 不会退出，因此通信 CTA 永远无法驻留。

需要在拿到具体模板 kernel 后调用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`，计算：

```text
max_resident_ctas = active_blocks_per_sm * multiprocessor_count
```

然后确保初始 compute/communication 配额不会让所有通信 CTA 被排除在 resident set 之外。

在当前“一 CTA 对应一个 SM 配额”的设计下，建议同时执行保守检查：

```text
num_comp_sm + num_comm_sm <= props->multiProcessorCount
```

Benchmark 参数验证也应查询实际 GPU SM 数，不能假设所有 H100/H200 都有相同的 132 个 SM。

### 5.3 使用 default stream 会破坏 PyTorch stream 语义

当前绑定使用：

```cpp
at::cuda::getDefaultCUDAStream(q.get_device())
```

PyTorch 自定义算子应在 current stream 上执行。否则在非默认 stream、NCCL overlap、pipeline parallel 或用户显式 stream 场景中可能发生：

- pack kernel 在 Q/K/V 生产完成前读取输入。
- attention 完成前，其他 stream 开始消费输出。
- 临时 tensor 的 caching allocator 生命周期与实际使用 stream 不一致。

应改为：

```cpp
at::cuda::getCurrentCUDAStream(q.get_device())
```

所有 pack、metadata 初始化、ready counter 初始化和 fused attention launch 必须使用同一个 current stream。

### 5.4 Public API 没有验证 hybrid 布局前提

当前 `global_seqlens_host` 只用于生成 CP mask，没有验证：

- CP batch 满足 `global_len == local_len * world_size`。
- CP batch 位于 rank-dependent local-only batch 之前。
- 不同 rank 的 CP offset 相同。
- 不同 rank 的 TK rank-block extent 相同。

通信 kernel 使用本 rank 的 `cu_seqlens_k` 解码所有远端 source row。如果某个 CP batch 前面存在 owner-only local batch，各 rank 的 CP offset 会不同，kernel 将静默读取错误的 KV row。

`ring_test/benchmark_hybrid_forward.py` 已经实现 CP-first、整除和 local balance 检查，但这些约束没有下沉到绑定层。

建议：

1. 绑定层验证每个 CP batch 的 `global_len == local_len * world_size`。
2. 在当前单 `cu_seqlens_k` 协议下，绑定层要求所有 CP batch 出现在 local-only batch 之前。
3. 如果未来需要任意 batch 顺序，应显式传递每个 source rank 的 cu-seqlens 或预构造 source-row descriptor，不能继续假设所有 rank offset 一致。

## 6. 主要性能问题

### 6.1 通信顺序与 compact prefix 顺序相反

`ready_end` 只能发布从 interval 起点开始的连续前缀，但通信仍按旧 ring step 顺序加载。

Noncausal compact 顺序是 rank 升序，通信顺序却是：

```text
rank-1, rank-2, ..., 0, W-1, ..., rank+1
```

Causal front compact 顺序是 rank 升序，但 lower-rank 通信顺序是降序。Higher-front 同样存在反向问题。

因此：

- rank 0 对应 compact prefix 的最前段，却通常最后到达。
- `ready_end` 在大部分通信时间内停在较小值。
- noncausal Q tile 通常要等几乎所有远端 KV 完成。
- causal 模式大多只能得到 front/back 两阶段重叠，而不是更细的 Q tile 粒度重叠。

#### 建议修改

Ready-once 不应继续使用 legacy ring-step task decoder。应按 compact destination row 升序生成通信任务：

1. task id 首先映射到 compact interval/chunk/row。
2. 根据 compact row 反解 source rank、source batch row 和 K/V selector。
3. 多个通信 CTA 以连续 chunk 为基本单位领取任务。
4. 最早的 compact chunk 必须优先发出，后续 chunk 可以并行完成。

这样通信优先级、ready prefix 和 Q tile dependency 使用同一个坐标系。

### 6.2 Causal LPT 顺序优先领取依赖更长的 Q tile

当前 causal scheduler 使用 `LPT=true`，并在 tile decoder 中反转 M block：

```text
block = num_m_blocks - 1 - block
```

对于 causal back half，M block 越靠后，所需 KV prefix 越长。当前顺序会让最早驻留的计算 CTA 等待最长依赖，而依赖较短的 Q tile 仍留在任务队列中。

建议把以下两个概念拆开：

- 是否启用 causal zigzag。
- 是否使用 LPT M-block 逆序。

Ready-once CP front/back 应按 `required_end` 升序调度。Local-only 任务仍可保留原来的 LPT 策略。

### 6.3 每次调用重新分配和初始化 workspace

每次 ready-once 调用都会执行：

- 分配 compact K/V。
- 分配并清零 ready counter storage。
- CPU 构造 compact cu-seqlens。
- CPU 构造 interval row metadata。
- CPU 到 GPU metadata copy。
- local KV pack kernel。
- ready prefix 初始化 kernel。
- fused communication/attention kernel。

该仓库 benchmark 测量端到端 op 时间，因此这些开销全部会计入 ready-once 性能。

建议提供显式可复用 workspace，至少包含：

- compact K/V storage。
- compact `cu_seqlens_k`。
- `ready_end`。
- `chunk_done`。
- `publish_lock`。
- `ready_interval_rows`。

对于训练中重复出现的相同 batch layout，host metadata 应缓存，不应每次重新构造和复制。

### 6.4 Ready-once 仍分配 legacy counter

当前 ready-once 模板不会消费：

- `mega_ring_kv_ready_counts`
- `mega_ring_step_ready`

但 host 仍为它们分配 tensor、执行 zero initialization，并设置本地 ready count。

应根据 `use_ready_once` 条件化分配，ready-once 路径允许这些指针为 null，同时调整 launch validation。

### 6.5 Mainloop 参数携带未使用的 ready metadata

FA3 mainloop 实际只需要：

- `mega_ring_ready_once`
- `mega_ring_ready_end`

`ready_intervals`、`ready_max_chunks` 和 `ready_chunk_rows` 仅由通信 wrapper 使用，不需要进入 `CollectiveMainloop::Arguments/Params`。删除这些未使用字段可以减少参数结构膨胀，并让 ready protocol 的 ownership 更清晰。

## 7. 代码结构建议

### 7.1 不要继续复用 `ring_step` 表示 work kind

Ready-once scheduler 当前使用 `ring_step` 字段传递 magic value：

```text
0 local
1 CP full
2 CP front
3 CP back
```

Producer mainloop、MMA mainloop 和 epilogue 分别解释这些值。后续修改很容易漏掉其中一个分支。

建议定义共享枚举：

```cpp
enum class MegaRingWorkKind : int {
    Local,
    CpFull,
    CpFront,
    CpBack,
};
```

Legacy step path 保留 `ring_step`，ready-once 路径使用 `work_kind`。为了保持 copied-and-trimmed 结构，可以只扩展现有 `WorkTileInfo`，不需要重写 scheduler family。

### 7.2 把 ready protocol 限制在 mega-ring wrapper

以下逻辑属于通信协议，不应扩散到通用 FA3 mainloop：

- chunk completion。
- ready prefix publication。
- publish lock。
- compact row 到 source row 的映射。

通用 mainloop 只需要接收一个可以 acquire-load 的 `ready_end` 和当前 tile 的 interval/target。

### 7.3 为 ready-once 单独定义通信 task decoder

当前 `run_mega_ring_remote_load()` 同时承载 legacy step path和ready-once compact path，内部存在大量运行时分支。

由于 `ReadyOnce` 已经是模板参数，可以把 task decoder 分成两个编译期路径：

- legacy ring-step decoder。
- ready-once compact-prefix decoder。

这样可以移除 ready-once hot loop 中的部分运行时判断，也能避免未来修改 legacy 路径时破坏 compact 顺序。

## 8. 推荐修复顺序

### 阶段 1：先保证不会死锁或跨 stream 算错

1. 修复 publish-lock lost wakeup。
2. 改用 current CUDA stream。
3. 验证 resident CTA 配额，拒绝可能饿死通信 CTA 的配置。
4. 下沉 hybrid layout 的必要输入检查。

完成标准：

- 2/4 GPU correctness 长时间循环不挂起。
- 非默认 stream correctness 通过。
- 无效 SM 配额和无效 hybrid layout 明确报错。

### 阶段 2：验证 ready-once 数值正确性

1. Ready-once 对比 PyTorch reference。
2. Ready-once 对比 legacy step/reduction 路径。
3. 覆盖 causal/noncausal、GQA、多个 CP batch 和 mixed local batch。
4. 运行 memcheck 和 racecheck。

完成标准：

- 所有 rank 输出通过既定 `atol=2e-1, rtol=2e-1`。
- 无非法访问和可报告数据竞争。

### 阶段 3：让 counter 真正产生细粒度收益

1. 按 compact prefix 重排通信任务。
2. 按 `required_end` 重排 causal Q tile。
3. 用 Nsight Systems 验证 local compute、communication、front Q 和 back Q 的时间线。

完成标准：

- `ready_end` 在通信期间多次推进，而不是只在阶段末尾跳变。
- 短依赖 Q tile 在长依赖 Q tile 之前启动。
- 通信 CTA 完成后能领取剩余计算任务。

### 阶段 4：减少 host 和 workspace 开销

1. 移除 ready-once 不使用的 legacy counter。
2. 引入可复用 workspace。
3. 缓存静态 layout metadata。
4. 分别测量 pack、metadata、communication、attention 和完整 op 时间。

完成标准：

- Ready-once 的端到端收益稳定高于 legacy step path。
- 短序列场景不会因 workspace 管理明显退化。

## 9. H100 Slurm 验证矩阵

登录节点只有 CUDA toolkit 和 build-time stub，没有 GPU 或 CUDA runtime。以下测试必须通过 Slurm 在 H100 上执行。

### 9.1 基础 correctness

至少覆盖：

| 维度 | 取值 |
| --- | --- |
| world size | 1, 2, 4, 8 |
| mode | causal, noncausal |
| path | ready-once, legacy step |
| batch | all-local, all-CP, mixed |
| Q heads | 8, 16, 32 |
| KV heads | 8 |
| head dim | 128 |
| sequence | 128, 256, 2048, 4096, 16384, 65536 |

重点覆盖 rank 0、中间 rank 和最后 rank，因为 causal front/back prefix 长度依赖 rank。

### 9.2 并发压力测试

为了触发 ready publication 竞争，应使用：

- 多个 CP batch。
- 每个 CP 序列包含多个 128-row chunk。
- `num_comm_sm` 取 4、8、16、24。
- world size 4 或 8。
- 连续执行至少数千次。

测试需要设置 watchdog 或 Slurm timeout，发生永久等待时保留 rank、mode、SM 配额和序列配置。

### 9.3 Stream correctness

测试流程：

1. 创建非默认 `torch.cuda.Stream()`。
2. 在该 stream 上生产 Q/K/V。
3. 在同一 stream 上调用 ready-once。
4. 在同一 stream 上立即消费输出。
5. 与 default-stream reference 比较。

### 9.4 Sanitizer

建议执行：

```bash
compute-sanitizer --tool memcheck \
    torchrun --standalone --nproc_per_node=2 \
    mega_ring_test_min_fa3_varlen_hybrid_multi_rank.py ...

compute-sanitizer --tool racecheck \
    torchrun --standalone --nproc_per_node=2 \
    mega_ring_test_min_fa3_varlen_hybrid_multi_rank.py ...
```

如果 `compute-sanitizer` 无法直接包裹 `torchrun`，应由每个 rank 单独启动被测 Python 进程并保存独立日志。

### 9.5 性能分解

至少记录以下时间：

- compact metadata 准备。
- local KV pack。
- ready counter 初始化。
- remote KV load。
- attention kernel body。
- 完整 Python op。

对比对象：

- PyTorch reference。
- FA3 hybrid Python baseline。
- mega-ring legacy step path。
- mega-ring ready-once path。

需要同时查看端到端时间和 kernel timeline，避免只看到 attention kernel 变快，但完整 op 因 pack/alloc 退化。

## 10. 当前验证状态

在登录节点已完成：

- `git diff --check`
- 相关 Python 文件 `py_compile`
- `make`
- CUDA 12.8 toolkit stub 链接

构建时未发现模板实例化或链接错误。由于登录节点没有 GPU 和 CUDA runtime，尚未完成：

- H100 correctness
- 多 rank runtime
- 非默认 stream correctness
- compute-sanitizer
- Nsight profiling
- ready-once 与 legacy 的端到端性能对比

## 11. 最终判断

Ready-once 的核心方向值得继续：compact KV 加一次完整 attention 可以从根本上删除逐 step O/LSE reduction，local-first 和 causal front-first 也符合通信计算重叠目标。

当前版本不能直接作为默认生产路径，主要原因是：

1. `publish_lock` 竞争可能丢失最后一次唤醒并永久死锁。
2. SM 配额没有 residency 保护，可能让通信 CTA 永远无法运行。
3. default stream 不符合 PyTorch 自定义算子语义。
4. 通信顺序和 Q tile 顺序没有与 dependency prefix 对齐，counter 的细粒度收益尚未兑现。
5. 每次调用的 compact workspace 和 metadata 开销可能抵消 reduction 节省。

应先完成并发正确性修复，再在 H100 上验证 compact-prefix 时间线，最后优化 workspace。不要在 lost wakeup、residency 和 stream 问题修复前依据单次 benchmark 得出 ready-once 的最终性能结论。
