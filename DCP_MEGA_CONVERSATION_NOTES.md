# DCP Mega 对话与技术分析记录

> 记录日期：2026-08-04  
> 仓库目录：`/home/hychen/min_fa3_demo`  
> 范围：DCP test 中 Mega DCP 的负载拆分、PackGQA、split、计算/通信任务队列，以及 packed-Q 使用 TMA 的可行性。  
> 状态说明：本文区分“已经通过当前代码确认的事实”和“对话中提出、但仍需要继续读代码验证的问题”。本文不是实现方案提交，本轮没有修改 kernel 或 metadata 代码。

## 1. 对话背景和问题脉络

本轮讨论从一个总问题开始：

> 当前 `dcp test` 的 Mega DCP，是如何将输入的一组负载转化为计算 tile 和通信 tile 负载的？

随后问题被细化为六组调度和流水问题：

1. 如果一组序列中只有一个序列需要 split，是否整个调用都会选择 split kernel 实例，即使其他序列并不需要 split？目前是否先逐序列计算 split 数，再取最大值作为实例上界，而不是把全部序列作为一个整体共同决定？
2. PackGQA 的选择是整组序列统一选择，还是每条序列单独选择？
3. 如何准确理解下面的 split 估算过程：

   ```text
   m_i = ceil(q_i * heads / 128)
   n_i = ceil(k_i / BlockN)

   blocks_per_sm =
       ceil(1.1 * sum_i(m_i * n_i) / num_sms)    # PackGQA 情况

   S_i = clamp(ceil(n_i / blocks_per_sm), 1, split_upper_bound)
   ```

4. 是否可以让部分 publish 工作在对应 attention 完成后立即开始，而不是等待全部 attention 完成，从而使对应 receive 更早开始？
5. 当前通信 CTA 等待远端 `tile_ready` 时，轮询的是本地映射/本地地址上的 ready，还是持续读取远端显存？
6. 当前任务队列是否并不是严格的 `attention -> publish -> final combine` 三个全局阶段，而是依赖 `attention_done` 逐 tile 推进？理想目标是否应当是：例如 `seq0` 的 attention 完成后，立即进行该序列的 history combine，再 publish，而不必等待其他序列？

在此基础上，讨论进一步聚焦到 PackGQA 的 Q 加载：

> 当前 PackGQA 的一个 attention tile，其 Q 是通过普通 load/cp.async 读取，而不是 TMA。假如可以保证 Q 长度至少为 8 且是 8 的倍数，每个 rank 至少有 4 个 Q head，并且 all-gather 时可以重排成每 token、每 Q head 的连续顺序，是否可以用 TMA 将 Q tile 加载到 shared memory？

本文先完整记录目前已经确认的 packed-Q/TMA 结论，再列出前述调度问题的后续核查清单，避免将尚未验证的推断写成当前实现事实。

## 2. 关键术语和符号

- `q_i`：第 `i` 条序列的 Q token 数。
- `k_i`：第 `i` 条序列对应的 K/V token 数。
- `Hq_local`：每个 DCP rank 上的本地 Q head 数。当前 Mega DCP 代码只接受 4 或 8，不是任意“至少 4”。
- `DCP`：参与当前 DCP group 的 rank 数；当前代码支持 2、4、8。
- `G`：PackGQA 因子，即一个 KV head 对应的 Q head 数，通常为 `Hq / Hkv`。
- `BlockM`：Q 的 packed-row tile 大小。当前 Mega DCP 使用 128。
- `BlockN`：K/V 方向 tile 大小。当前配置支持 128 或 176。
- `m_i`：序列 `i` 在 packed Q/M 方向上的 tile 数。
- `n_i`：序列 `i` 在 K/V/N 方向上的 tile 数。
- `S_i`：序列 `i` 最终采用的 split 数。
- `q_group`：Q all-gather 后供 history attention 使用的本地连续缓冲区。

## 3. 已确认：当前 PackGQA 的 attention Q 加载路径

### 3.1 当前确实没有用 TMA 将 Q tile 搬入 shared memory

主循环明确设置：

```cpp
static constexpr bool Use_TMA_Q = !PackGQA;
```

见 [`include/min_fa3_mainloop.h`](include/min_fa3_mainloop.h#L62)。因此启用 PackGQA 时：

```text
PackGQA == true
Use_TMA_Q == false
```

attention producer 随后进入 `Load Q with cp.async` 分支，并调用 `PackGQAManager::load_Q()`，见 [`include/min_fa3_mainloop.h`](include/min_fa3_mainloop.h#L1026)。

需要准确区分以下两层：

- Mega DCP 的 Q all-gather 本身已经通过 ThunderKittens TMA load/store 在通信路径中搬运数据。
- all-gather 完成后，attention CTA 从本地 Q 或 `q_group` 将一个 Q tile 搬到 shared memory，目前使用的是 128-bit 向量化 `cp.async`，不是 TMA Q load。

因此，将当前路径简称为“普通 load”不够准确；它是多线程协作、带 zero-fill copy atom 的异步全局到共享内存拷贝。

### 3.2 当前 cp.async 路径如何定位每个 packed row

`PackGQAManager::load_Q()` 把逻辑 packed row 编号还原成 token 和 group 内的 Q head：

```text
packed_row = token * G + group_head

token, group_head = divmod(packed_row, G)
q_ptr = &Q[token, group_head, 0]
```

对应代码位于 [`include/hopper_compat/pack_gqa.h`](include/hopper_compat/pack_gqa.h#L58)。它先为部分线程计算行指针，再通过 warp shuffle 将指针分发给负责同一行不同向量段的线程，最后发出 128-bit `cp.async`，见同文件的 [`load_Q`](include/hopper_compat/pack_gqa.h#L79)。

该路径的主要额外成本不是 Q 数据传输量，而是：

- 每个 packed row 的 `divmod`；
- 行首地址计算；
- warp shuffle 分发指针；
- 由较多 producer threads 发出的 copy 指令。

## 4. 为什么通用 PackGQA 不能直接复用普通 Q 的 TMA descriptor

通用 GQA 的物理 Q 布局通常是：

```text
Q[token][all_qheads][D]
```

其中 `D = 128`。对于一个给定 KV head，只需要它对应的 `G = Hq / Hkv` 个 Q head。设该 KV head 对应 `kG ... kG+G-1`，packed 访问顺序为：

```text
token 0, head kG
token 0, head kG+1
...
token 0, head kG+G-1
token 1, head kG
...
```

同一个 token 内，相邻 Q head 行的地址差是 `D` 个元素。但是从当前 token 的最后一个 group head 跳到下一个 token 的第一个 group head时，地址差是：

```text
(Hq - G + 1) * D
```

因此，把 `(token, group_head)` 简单压平为一个 packed M 维后，这个 M 维通常不是固定 stride 的连续二维维度。当前普通 Q TMA descriptor 描述的是 token 维和单独的 head 维，不能仅通过把 `Use_TMA_Q` 改为 `true` 就正确加载 PackGQA tile。

这不是“TMA 原理上不支持 PackGQA”。仓库内 vendored MagiAttention 已经实现 packed-Q TMA：

- 定义 `ShapeQPackedTMA = ((PackGQAFactor, seqlen), headdim, khead)`；
- 为其构建 `TMA_Q_Packed`；
- varlen 加载时把序列 offset 乘以 `PackGQAFactor`。

相关代码见 [`third_party/MagiAttention/.../mainloop_fwd_sm90_tma_gmma_ws.hpp`](third_party/MagiAttention/magi_attention/csrc/flexible_flash_attention/mainloop_fwd_sm90_tma_gmma_ws.hpp#L257) 和同文件的 [descriptor 构建](third_party/MagiAttention/magi_attention/csrc/flexible_flash_attention/mainloop_fwd_sm90_tma_gmma_ws.hpp#L447)、[TMA Q 发射](third_party/MagiAttention/magi_attention/csrc/flexible_flash_attention/mainloop_fwd_sm90_tma_gmma_ws.hpp#L548)。

这证明了：当 pack factor 和物理 stride 可以被 descriptor 正确表达时，PackGQA Q 使用 TMA 是可实现的。当前 minimal FA3 只是没有接入相应 descriptor 和同步路径。

## 5. 为什么 Mega DCP 特化更适合 flattened packed-Q TMA

当前 Mega DCP 对 chunk 和 history K/V 都要求：

```text
Hkv_group == 1
```

shape 检查见 [`csrc/dcp_mega_min_fa3_varlen_bindings.cu`](csrc/dcp_mega_min_fa3_varlen_bindings.cu#L381)。因此：

```text
chunk attention:
    G_chunk = Hq_local

history attention:
    G_history = DCP * Hq_local
```

因为只有一个 KV head，当前 attention 调用里的全部 Q head 都属于同一个 PackGQA group。如果物理布局为连续的：

```text
Q[token][head][d]
```

那么元素地址为：

```text
address(token, head, d)
    = ((token * G + head) * 128 + d)
```

定义：

```text
packed_row = token * G + head
```

就得到一个完全连续的二维矩阵：

```text
Q_packed[total_q * G][128]
```

因此可以用一个逻辑上的 `128 x 128` BF16 TMA tile，从下面的位置读取：

```text
packed_sequence_offset = cu_seqlens_q[batch] * G
packed_tile_offset     = packed_sequence_offset + m_block * 128

Q_packed[packed_tile_offset : packed_tile_offset + 128, 0 : 128]
```

这个方案不再需要逐 row `divmod` 来计算 Q 的全局内存地址。token/head 的反解仍可能用于 mask、输出地址和 epilogue，但不再位于 Q tile 的数据搬运热路径上。

## 6. 已确认：当前 all-gather 输出已经基本是目标顺序

当前绑定要求 `q_group` 的 shape 为：

```text
[capacity_q, DCP * Hq_local, 128]
```

见 [`csrc/dcp_mega_min_fa3_varlen_bindings.cu`](csrc/dcp_mega_min_fa3_varlen_bindings.cu#L495)。这些 Q tensor 还被要求是 contiguous，见同文件的 [`check_packed_bf16`](csrc/dcp_mega_min_fa3_varlen_bindings.cu#L43)。

Q all-gather 把每个远端 rank 的 16-token、`Hq_local * 128` tile 存入 `q_group` 的 `src_rank` 槽位：

```cpp
params.q_group, shared.comm_tiles[chunk],
{0, task.token_begin / 16, task.src_rank, 0}
```

见 [`include/dcp_mega_min_fa3_varlen_launch.h`](include/dcp_mega_min_fa3_varlen_launch.h#L353)。其物理顺序等价于：

```text
[token]
    [src_rank 0][local_head 0 ... Hq_local-1]
    [src_rank 1][local_head 0 ... Hq_local-1]
    ...
[d]
```

即：

```text
[token][global_qhead][d]
```

所以，如果目标只是让 history Q 成为每 token、每 Q head、每 head-dimension 连续，当前 `q_group` 实际上已经满足这个条件。当前本地 chunk Q 也是 contiguous 的 `[token, Hq_local, 128]`，同样可以压平为 `[total_q * Hq_local, 128]`。

结论是：Mega DCP 的关键优势不是“未来可以重排”，而是当前单 KV-head 加上现有 contiguous layout 已经消除了通用 GQA 的 head-group 间空洞。

## 7. `q_len >= 8` 且 `q_len % 8 == 0` 的真实作用

Q attention tile 的 M 方向是 128 个 packed rows，不是固定 8 个 token。对序列 `i`：

```text
valid_packed_rows_i = q_i * G
m_i = ceil(q_i * G / 128)
```

如果希望每条序列完全由整 tile 构成，不发生 partial tile，则要求：

```text
q_i * G % 128 == 0
```

当前 Mega DCP 实际支持 `Hq_local in {4, 8}`，见 [`csrc/dcp_mega_min_fa3_varlen_bindings.cu`](csrc/dcp_mega_min_fa3_varlen_bindings.cu#L136)，而不是任意 `Hq_local >= 4`。

| 路径 | `G` | 一个 128-row tile 对应的 token 数 | 完全无尾部的条件 |
|---|---:|---:|---:|
| chunk，`Hq_local=4` | 4 | 32 | `q_i % 32 == 0` |
| chunk，`Hq_local=8` | 8 | 16 | `q_i % 16 == 0` |
| history，`DCP * Hq_local=8` | 8 | 16 | `q_i % 16 == 0` |
| history，`G=16` | 16 | 8 | `q_i % 8 == 0` |
| history，`G=32` | 32 | 4 | `q_i % 4 == 0` |
| history，`G=64` | 64 | 2 | `q_i % 2 == 0` |

所以 `q_i % 8 == 0` 的结论是：

- 对 `G >= 16` 的 history attention，足以消除 Q 方向 partial tile。
- 对最小 history 配置 `DCP=2, Hq_local=4, G=8`，仍需 `q_i % 16 == 0`。
- 对 chunk attention，`Hq_local=4` 时仍需 `q_i % 32 == 0`，`Hq_local=8` 时仍需 `q_i % 16 == 0`。
- `q_i >= 8` 本身不是 TMA 可用性的必要条件；它更多是在限制小序列和减少极低有效率 tile。

因此，这组条件有利于 TMA，但既不是“能否用 TMA”的必要条件，也不足以保证所有路径都没有尾部。

## 8. Varlen 尾部的三种可行策略

### 8.1 每条序列按 128 个 packed rows 填充

为每条序列分配：

```text
padded_rows_i = ceil(q_i * G / 128) * 128
```

优点：

- 每次 TMA load 的 footprint 都完全位于本序列分配区间；
- 不会读到下一条尚未 all-gather 完成的序列；
- ready dependency 和内存正确性最容易证明。

代价：

- Q/all-gather buffer 增大；
- 通信和显存流量包含 padding；
- `cu_seqlens_q` 不能再直接作为物理 Q offset，需要额外 packed/padded offsets。

### 8.2 完整 tile 用 TMA，末尾 partial tile 保留 cp.async

优点：

- 不需要改变 Q 的序列间物理布局；
- 不发生跨序列读取；
- 完整 tile 可以移除绝大多数逐 row 地址计算。

代价：

- kernel 内同时存在 TMA Q 和 cp.async Q 两套加载协议；
- transaction barrier、cp.async barrier、phase 和 named barrier 的组合更复杂；
- 小序列或大量短 varlen 序列下，可能有较高比例仍然走 cp.async。

### 8.3 固定读取完整 TMA tile，允许越过当前序列尾部

从 attention 数学看，不同 Q row 相互独立，无效 row 不会影响有效 row；当前 PackGQA epilogue 也不会写出序列范围外的 row。因此，只考虑算子数学，尾部加载到下一序列的 Q 不会污染当前序列的有效输出。

但是 Mega DCP 的 Q all-gather 与 attention 是重叠执行的。若 TMA tile 越过当前序列边界，可能访问到尚未由 all-gather 发布的下一段 Q。当前 cp.async 实现对 `idx < seqlen_q * G` 做判断，不会产生这种访问；切换成固定大小 TMA 后必须额外处理。

可选处理包括：

- 把整个 TMA footprint 覆盖的 Q token block 都加入 `q_ready` dependency；
- 保证被 overfetch 的下一段在 attention 发射前已发布；
- 在序列之间增加 padding；
- 只对最后一个 partial tile 回退到 cp.async。

最后一条全局序列还需要保证 TMA tensor map 的全局 extent 能提供安全 OOB zero-fill，或者在分配末尾预留足够 padding。

## 9. TMA Q 实现需要同时调整的组件

这不是将 `Use_TMA_Q` 从 `false` 改成 `true` 的单行修改。完整实现至少涉及：

1. 为 PackGQA/Mega DCP 引入 packed Q TMA descriptor。
2. descriptor 应描述连续的 `[total_q * G, 128]`，或者等价的嵌套 `((G, total_q), 128, kvhead)`。
3. chunk 和 history 需要分别构建 descriptor，因为二者的 `G` 分别是 `Hq_local` 和 `DCP * Hq_local`。
4. varlen sequence offset 必须乘以 `G`：

   ```text
   packed_offset_q = cu_seqlens_q[batch] * G
   ```

5. tile 起点应为：

   ```text
   packed_offset_q + m_block * 128
   ```

6. Q shared-memory barrier 必须使用 transaction barrier。当前 kernel 已经通过 `Use_TMA_Q` 在 `ClusterTransactionBarrier` 和 `ClusterBarrier` 之间选择，见 [`include/min_fa3_kernel.h`](include/min_fa3_kernel.h#L69)。
7. TMA issuer 需要调用：

   ```text
   arrive_and_expect_tx(128 * 128 * sizeof(bfloat16))
   ```

   一次完整 Q tile 的 transaction bytes 是 32768。
8. shared-memory Q layout 必须由 TMA descriptor 正确映射，不能只假设它适用于 position-independent cp.async swizzle tensor。
9. `NumProducerThreads`、`QueryEmpty` named barrier 的 arrival count 和 Q barrier 初始化必须随 TMA 路径一致变化。当前相关选择见 [`include/min_fa3_mainloop.h`](include/min_fa3_mainloop.h#L127) 和 [`include/min_fa3_prologue.h`](include/min_fa3_prologue.h#L21)。
10. 必须保留 all-gather TMA store 完成、`fence.proxy.async.global`、release signal、attention acquire wait 和后续 TMA read 之间的可见性关系。
11. metadata 中的 `q_dependencies` 必须覆盖实际 TMA load footprint，而不只是数学上的有效 Q row。
12. chunk Q 和 history `q_group` 都要验证，不能只优化 all-gather 后的 history Q。

## 10. Q all-gather 与 attention 的同步关系

当前 Q all-gather 每次搬运一个 16-token communication tile。完成本地 `q_group` TMA store 后，代码执行：

```text
tma_store_async_wait
fence.proxy.async.global
signal_release(q_ready)
```

见 [`include/dcp_mega_min_fa3_varlen_launch.h`](include/dcp_mega_min_fa3_varlen_launch.h#L353)。

history attention scheduler 在领取 descriptor 后，根据该 descriptor 的 `q_dependencies` 等待对应 `q_ready` 达到 `DCPSize`，见 [`include/dcp_mega_min_fa3_varlen_scheduler.h`](include/dcp_mega_min_fa3_varlen_scheduler.h#L105)。这意味着当前设计不是简单地“全部 Q all-gather 完成后再统一启动 history attention”，而是具备按 descriptor/Q block dependency 推进的结构。

若改用 TMA Q load，最需要重新审计的是：

```text
metadata 声明的 ready footprint
    是否覆盖
实际 128 packed-row TMA load footprint
```

当 `128 / G` 个 token 与 16-token communication block 对齐时，依赖关系较简单；当序列起点、尾部或 overfetch 跨越 communication block 时，需要覆盖多个 ready 项。

## 11. 性能判断

一次完整 BF16 Q tile 的数据量固定为：

```text
128 * 128 * 2 bytes = 32 KiB
```

从 cp.async 改为 TMA 不会减少有效完整 tile 的字节数。主要潜在收益来自：

- 去掉逐 packed row 的 `divmod`；
- 去掉逐 row 指针计算和 warp shuffle；
- 减少 producer 发出的 load/copy 指令；
- 使用单线程 TMA issue 和 transaction barrier；
- 可能降低 producer 路径的寄存器和指令压力。

但现有 cp.async 已经是 16-byte 向量化、按 head dimension 合并的读取，因此不能只根据“TMA 指令更高级”就断定性能一定提高。还要考虑：

- TMA descriptor 构建和访问形式；
- partial tile 的额外流量；
- split descriptor 是否重复读取同一个 Q tile；
- Q all-gather/attention overlap 是否因更宽 dependency 而下降；
- producer/consumer barrier 开销；
- H200 上不同 `q_len`、`G`、`BlockN`、split 数下的实际测量。

当前最合理的性能假设是：TMA Q 的收益主要是降低地址生成和 producer instruction overhead，而不是降低 HBM 字节数。

## 12. packed-Q TMA 问题的最终结论

可以将结论压缩为：

> 对当前 Mega DCP 的 `Hkv_group == 1` 特化，连续的 `[token][qhead][128]` 可以直接压平为 `[q_len * G][128]`，因此 Q tile 从 global memory 到 shared memory 可以使用 TMA。当前 history `q_group` 的 all-gather 输出实际上已经接近或等同于所需顺序，本地 chunk Q 也满足连续条件。真正需要解决的不是 TMA 是否能够描述该布局，而是 varlen 尾部、跨序列 overfetch、Q readiness footprint、transaction barrier 和 producer 同步。

同时：

> `q_len >= 8`、`q_len % 8 == 0` 和每 rank 至少 4 个 Q head 并不是充分的“无尾部”条件。当前代码实际支持每 rank 4 或 8 个 Q head；是否恰好整 tile，应检查 `q_len * G % 128 == 0`。

## 13. 尚待继续核查的原始调度问题

以下问题在本轮可见对话中被明确提出，但尚未形成经过完整源码核查的最终答复。后续分析时应以 metadata 生成器、launch dispatch、scheduler 和 fused kernel 队列实现为依据。

### 13.1 输入负载如何生成计算和通信 tile

需要从 Python metadata 生成入口开始，逐层说明：

```text
输入序列集合
  -> 每序列 q_i / history_k_i / chunk_k_i
  -> PackGQA 决策与 BlockN 选择
  -> m_i / n_i
  -> 每序列 split S_i
  -> chunk/history attention descriptors
  -> Q all-gather tasks
  -> history combine/publish tasks
  -> receive/final combine tasks
  -> dependency arrays 和 ready/completion IDs
  -> compute CTA 与 communication CTA 的领取方式
```

核查时需要明确每个 task descriptor 的粒度究竟是：sequence、M tile、N split、16-token communication block，还是多个维度的组合。

### 13.2 split kernel 实例是全局选择还是逐序列选择

需要分别回答两个层次：

- 编译/launch 层：一次 kernel launch 是否只能统一选择 `Split=true` 或 `Split=false` 的模板实例。
- metadata/runtime 层：在 `Split=true` 实例内，是否允许某条序列的 `S_i=1`，而另一条序列 `S_j>1`。

还需要确认 launch 使用的 `effective_num_splits` 是否是 `max_i(S_i)`，以及各序列实际 split 数是否保存在 `chunk_sequence_splits` 和 `history_sequence_splits` 中。不能把“选择 split-capable kernel”误解为“所有序列都必须真的拆成相同 split 数”。

### 13.3 PackGQA 是统一还是逐序列选择

需要确认 PackGQA 是否影响模板实例、tensor layout、scheduler 解释和 workspace shape。若它是 compile-time 模板参数，则同一次 launch 中通常必须统一；即使 metadata 的成本模型可以逐序列估算，也不能直接让一个 kernel 内部分序列 PackGQA、部分序列 non-PackGQA，除非存在显式的双路径 descriptor 或拆成两个 launch。

需要从 metadata 的 `pack_gqa` 字段、bindings 的 dispatch 和 kernel specialization 三处交叉验证。

### 13.4 split 公式的逐项含义

待验证公式为：

```text
m_i = ceil(q_i * heads / 128)
n_i = ceil(k_i / BlockN)

blocks_per_sm = ceil(1.1 * sum_i(m_i * n_i) / num_sms)

S_i = clamp(ceil(n_i / blocks_per_sm), 1, split_upper_bound)
```

后续完整解释应回答：

- `heads` 在 PackGQA 下究竟是 `Hq/Hkv`、本地 head 数，还是当前 attention group 的总 Q head 数；
- `m_i * n_i` 为什么代表未 split 的 tile work；
- `sum_i` 为什么在所有序列间聚合，用于估算整个 launch 的平均 CTA waves；
- `1.1` 是怎样的 oversubscription/负载均衡安全系数；
- `blocks_per_sm` 为什么反过来限制一个 split 应包含的 N blocks；
- `ceil(n_i / blocks_per_sm)` 如何得到序列自己的 split 数；
- `split_upper_bound`、空 history、causal chunk 和不同 BlockN 如何影响最终值；
- chunk 和 history 是否分别计算 `S_i`，以及 launch 的 split 上界如何合并。

### 13.5 publish 是否可以按已完成 tile/序列提前执行

需要检查：

- `PublishWorkDesc` 的 dependency 是整个序列的所有 attention descriptors，还是单个 history/chunk tile；
- compute CTA 在 attention 队列耗尽前是否允许领取 publish 队列；
- publish counter 是否只在全局 attention phase 后开启；
- `attention_done` 是否已经支持按 completion ID 细粒度等待；
- 提前 publish 是否会与仍在写同一 output/LSE partial workspace 的 attention CTA 冲突。

设计目标可以表述为：

```text
seq0 所需 attention descriptors 完成
  -> seq0 history/split combine
  -> seq0 publish
  -> 对端 seq0 receive
  -> seq0 final combine

同时 seq1/seq2 的 attention 仍可继续
```

是否已经做到这一点，必须区分“依赖结构允许”和“当前 CTA phase loop 实际会不会及时领取”。

### 13.6 `tile_ready` 位于哪里、通信 CTA 在读什么

需要沿这些对象核查：

- `ipc_tile_ready` 的每 rank IPC allocation；
- `tile_ready_remote[rank]` 指针数组如何建立；
- publisher 写入目标 rank 的 ready 地址；
- receiver 使用哪个 rank 视图轮询；
- NVLink/IPC 映射下“本地虚拟地址”和“物理上位于远端 GPU 显存”之间的区别。

准确回答应避免简单地说“本地”或“远端”。CUDA IPC/NVLink 场景中，指针对当前 GPU 是可访问的本地虚拟地址，但其 backing allocation 可能属于另一张 GPU；每次系统作用域 load 是否产生远端访问，还取决于 ready buffer 的所有权、映射方式和缓存/一致性协议。

### 13.7 attention、publish、final combine 是否是严格分段队列

需要结合 fused kernel 的实际控制流回答：

- attention CTA 完成一个 descriptor 后是否只继续领取 attention；
- attention queue 何时被判定耗尽；
- publish/final counter 何时开始领取；
- publish/final descriptor 虽然逐项等待 `attention_done`，是否仍因领取顺序形成宏观 phase barrier；
- communication CTA 的 Q all-gather、receive 是否与 compute queue 并行；
- graph replay phase signal 是否增加额外全局阶段。

核心区别是：

```text
细粒度 dependency 存在
```

并不自动等价于：

```text
调度器会在 dependency 满足后立刻执行该任务
```

如果所有 compute CTA 必须先把 attention queue 取空，才切换到 publish/final counter，那么依赖虽然逐 tile，宏观上仍可能近似分段。若 CTA 可以在 attention 未全局结束时跨队列取任务，才是真正的序列级流水。

## 14. 后续建议的源码核查顺序

为完整回答第 13 节的问题，建议按以下顺序继续，不先做代码修改：

1. 阅读 `dcp_mega_metadata.py` 的输入规范、PackGQA/BlockN/split 决策和 descriptor 生成。
2. 对照 metadata header、descriptor struct 和 dependency array 的 C++ 定义。
3. 从 bindings dispatch 确认 `Split`、`BlockN`、`DCPSize`、`CommHeads` 和 PackGQA 的实例选择粒度。
4. 展开 scheduler 的初始 descriptor、动态 counter 和 completion ID 映射。
5. 展开 fused kernel 中 attention、history combine/publish、receive、final combine 的 phase/control flow。
6. 逐项追踪 `q_ready`、`attention_done`、`publish_ready`、`tile_ready` 和 `receive_ready` 的 writer、reader、作用域及内存序。
7. 用一个最小例子手工展开 metadata，例如两条序列、不同 `q_i/k_i`、仅一条需要 split，列出生成的全部 compute/communication tasks 和 dependency edges。

## 15. 本轮工作区变更

本轮只新增本文档：

```text
DCP_MEGA_CONVERSATION_NOTES.md
```

没有修改任何 C++、CUDA、Python、构建文件或测试文件，也没有改动工作区中原有的未跟踪 `test.sh`。
