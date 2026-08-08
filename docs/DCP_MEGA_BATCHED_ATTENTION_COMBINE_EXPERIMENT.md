# Mega DCP 分批 Attention/Combine 实验复盘

日期：2026-08-08

状态：focused gate 失败，实验功能回退，仅保留本文档。

## 1. 目标与假设

目标 workload 是一个 batch 中包含少量长 chunk-prefill 序列和大量短序列。原始 Mega DCP compute CTA 先完成整个 attention queue，再进入 history split combine；通信 CTA 在 Q allgather 后等待 publish tile 并执行 receive。

实验假设是把 attention/combine 拆成两个 epoch：

```text
Attention(E0) -> History Combine/Publish(E0)
              -> Attention(E1) -> History Combine/Publish(E1)
              -> Final Combine
```

E0 选择短 query 对应的一组 publish-complete tile，使通信 CTA 可以在 E1 attention 期间开始 receive。功能默认关闭，仅通过显式 flag 启用。

## 2. 实现概要

Metadata 升级到 version 6，在原 header slot 31/32 记录 E0 attention 和 history-combine 边界。selector 使用以下约束：

- `max(Sq) >= 4 * median(Sq)`。
- E0 只选择 `Sq <= median(Sq)` 的 final tile。
- E0 attention 至少覆盖四分之一个 compute CTA wave。
- E0 combine 至少覆盖一个 compute CTA wave。
- E0 receive 至少覆盖一个 communication CTA wave。
- E1 attention 至少保留一个 compute CTA wave。
- E0 估算 attention 成本不超过总成本的三分之一。

Kernel 为 E1 使用独立 dynamic counter，并允许 history combine 在指定 `[work_begin, work_count]` 范围内运行。启用路径没有新增 `__syncthreads()`，而是复用当前静态配置中未使用的 named barrier：

- `AppendKV` barrier：attention pipeline 初始化后的 CTA fence。
- `QueryRotated` barrier：attention、mbarrier invalidate、combine 之间的 CTA join。

在 shared memory 被 combine 覆盖前，显式 invalidate Q/O 和 K/V/Vt pipeline mbarrier。默认关闭路径继续使用原始单 attention、单 combine 和两次 `__syncthreads()`。

## 3. 实现期间发现的问题

### 3.1 Attention re-entry named-barrier 尾态

第一次实现会在 E1 re-entry 死锁。queue snapshot 证明 E0 attention/combine、first publish 和 first receive 已经完成，compute CTA 卡在第二次进入 attention mainloop。

FA mainloop 退出后会留下供下一 tile 使用的 named-barrier arrival：

- MMA threads 留下 `QueryEmpty` arrival，需要 producer WG 补齐。
- `IntraWGOverlap=true` 时，每次 WG hand-off 会留下 WG2 对 `WarpSchedulerWG1` 的 arrival，需要 WG1 补齐。
- WG2 barrier 在当前 hand-off 路径中已经被消费，不能再次 drain。

错误地让 WG1 和 WG2 都 drain 时，恰好执行过 E0 work 的 48 个 CTA 卡住；只让 WG1 drain 后，selector-positive correctness 通过。

### 3.2 mbarrier invalidate 不是完整根因

只增加 Q/O、K/V/Vt pipeline mbarrier invalidate 不能解除 re-entry 死锁。mbarrier 生命周期需要正确结束，但 named barrier phase 也必须单独收敛。

### 3.3 Benchmark host 异常被误判为 kernel 死锁

第一次 focused off/on 运行中，off 结果已经完成并写出，但 rank 0 打印 phase table 时对 `epoch0_*: null` 调用了 `.get()`，抛出：

```text
AttributeError: 'NoneType' object has no attribute 'get'
```

非 rank0 已进入 finally 中的 NCCL barrier，因此表现为 GPU 1-7 持续 100%、GPU 0 空闲。这不是 attention kernel 死锁。临时修复为打印时跳过未写的可选 milestone，并保证 rank0-only 序列化/打印失败时所有 rank 对称进入 finally barrier。

## 4. Correctness 结果

以下验证通过：

- Metadata CPU tests：22/22。
- Runner/benchmark CPU tests：16/16。
- Python compile 和 `git diff --check`。
- 缩小 selector-positive DCP=4、BlockN=128、split=2 用例。
- 完整 8 卡 correctness matrix，一次运行通过。
- Matrix 覆盖 DCP 2/4/8、BlockN 128/176、auto/显式 split、重复 eager/CUDA Graph，以及 `20 x Sq=16 + 1 x Sq=4096` 的 selector-positive DCP=8 case。

因此实验失败原因不是数值错误或稳定复现的 kernel deadlock，而是性能 gate 失败。

## 5. Focused 实测

Focused workload：`case_000010`，DCP=8，Hq_local=4，BlockN=128，auto split（实际 non-split），num_comm_sm=8，eager，20 warmup，40 iterations，off/on 位于同一 process group。

| 指标 | off | on | 变化 |
| --- | ---: | ---: | ---: |
| p50 kernel latency | 0.563 ms | 2.922 ms | +419% |
| p90 kernel latency | 0.571 ms | 2.960 ms | +418% |
| Aggregate TFLOPS | 406.2 | 78.2 | -80.7% |
| Avg TFLOPS/GPU | 50.8 | 9.8 | -80.7% |

On 路径 phase p50：

| Phase | 时间 |
| --- | ---: |
| Q allgather done | 201.104 us |
| E0 attention done | 174.592 us |
| E0 history combine done | 204.656 us |
| first publish | 79.728 us |
| first receive | 164.800 us |
| overall attention done | 2850.496 us |
| final combine done | 2909.200 us |

通信 overlap 的功能目标已经达到：first publish 和 first receive 都早于 overall attention done。但 E0 combine 到 overall attention done 的尾段达到 2666.928 us，第二次 attention 进入后的成本远大于原单次 attention 路径，完全抵消 overlap。

Focused gate 要求 p50 至少提升 5%，p90 回退不超过 2%。实测远未达到，因此没有继续运行 20-case A/B，避免无意义地增加 GPU 测试次数。

## 6. Spill 说明

Ptxas 报告的高 stack/spill 在本实验修改前已经存在。按约定，本实验没有调整 launch bounds、寄存器配置或编译参数，也没有为了 spill 重构 kernel。Focused 回退不能归因于本轮新增 spill 调优缺失；直接可见的主要问题是 attention mainloop re-entry 后 E1 尾段极长。

## 7. 结论

该策略能够提前 publish/receive，但代价是：

- 必须管理 FA mainloop 的 named barrier 和 mbarrier re-entry 生命周期。
- 需要额外 metadata selector、queue counter 和 phase 状态。
- non-split focused case 中第二次 attention 进入产生数量级性能回退。
- 复杂度和性能风险显著高于实际通信 overlap 收益。

最终决定是回退两阶段 attention/combine 功能、metadata v6、相关 CLI/benchmark sweep 和 correctness 扩展，只保留本文档。后续若再次探索，应优先避免同一 CTA 内重新进入完整 FA mainloop，例如在单次 mainloop 内暴露可发布 tile，或在独立 kernel/launch 边界上验证 pipeline 生命周期和成本。
