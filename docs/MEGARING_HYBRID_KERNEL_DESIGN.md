# MegaRing Hybrid Megakernel 设计

本文档描述当前仓库中已经实现的 **MegaRing Hybrid** 核心 kernel，包括
forward、backward、层级 ring 调度、通信计算重叠、在线归约和跨 GPU 梯度归约。
它与 [BR-PBS 负载均衡器设计](../balancer/design.md) 共同构成完整方案：BR-PBS
决定每条序列放到哪个 buddy ring，MegaRing Hybrid 在一次层级化执行中消费这些
ring，并尽量把通信隐藏在 attention 计算之后。

本文以当前代码为准，同时区分三类结论：

- **已实现机制**：可以直接映射到当前源码。
- **设计动机**：解释当前选择解决了什么问题，但不等同于性能结论。
- **论文假设**：需要通过消融实验和 profiler 数据验证后才能写成定量结论。

早期的 forward 方案和 backward 专项说明仍保留在
[HIERARCHICAL_HYBRID_MEGA_RING_DESIGN.md](./HIERARCHICAL_HYBRID_MEGA_RING_DESIGN.md)
与
[HIERARCHICAL_HYBRID_MEGA_RING_BACKWARD_DESIGN.md](./HIERARCHICAL_HYBRID_MEGA_RING_BACKWARD_DESIGN.md)，
但涉及当前统一架构、causal multi-segment 和 SM 角色转换时，以本文为准。

## 1. 设计目标与核心结论

### 1.1 要解决的问题

常规 ring attention 通常把一次 attention 拆成多个 ring step：

```text
for step in ring:
    communicate K/V
    launch attention block
    merge partial O/LSE
```

这种组织在训练系统中有四类开销：

1. 每个 step 都有 kernel launch、host 调度和 stream 依赖。
2. 通信与计算由独立 kernel 或通信库管理，短 step 很难稳定重叠。
3. 每个 step 都需要重复读取 Q，并对 O/LSE 做一次全局内存归并。
4. 当同一 world 中同时存在 G8/G4/G2/G1 多个重叠 ring 时，逐 ring 启动会增加
   launch 数量，且难以统一利用空闲 SM。

MegaRing Hybrid 的目标不是重新设计 attention 数值算法，而是在保留裁剪版 FA3
Hopper mainloop 的前提下，重构其**执行组织**：

```text
一个 persistent grid
    = compute CTAs
    + communication CTAs
    + device-side work scheduler
    + device-side readiness protocol
    + forward O/LSE 在线归约
    + backward dKV owner-directed reduce-add
```

### 1.2 核心设计

当前实现包含五个关键点：

1. **Megakernel 融合**。Forward 将远端 K/V 搬运、FA3 attention 和 O/LSE
   在线归约放入一次 persistent launch。Backward 的核心 megakernel 将远端 K/V
   ingress、FA3 attention backward 和 dKV 远端 reduce-add 放入一次 launch；FA3
   preprocess、跨卡完成等待与梯度格式 postprocess 仍是外围辅助 kernel。
2. **SM 角色专门化与 forward 动态转换**。启动时逻辑上划分 compute SM 和
   communication SM。Forward communication CTA 完成自己的搬运任务后，不退出，
   而是转换为 compute CTA，加入同一个动态 work queue。
3. **层级 hybrid ring**。G8、G4、G2、G1 使用同一个 kernel 和固定四层 descriptor，
   不按 ring size 启动不同 attention kernel。
4. **Causal 动态 multi-segment**。同一个 Q tile 的若干连续、已就绪 ring step
   可以合并成一次 FA3 mainloop，减少 scheduler claim、Q 重载和 O/LSE 归并次数。
5. **与 attention tile 对齐的通信任务**。通信调度按 128 或 176 token row 的
   logical task 计数，物理传输拆为固定 `16 x 1024` 2D TMA subtile，在通信粒度、
   shared memory 和流水并行度之间取得可实现的平衡。

### 1.3 “Hybrid”的含义

本文中的 Hybrid 同时包含两个维度：

- **拓扑 hybrid**：同一 world 中并存 G8/G4/G2/G1，不同序列使用不同 CP degree。
- **执行资源 hybrid**：同一 grid 中并存 communication CTA 和 compute CTA，forward
  还允许前者在运行中转换为后者。

最核心的语义是拓扑 hybrid。SM 角色转换是 forward 对这种 workload 的资源利用
优化，不应把二者混为同一个概念。

## 2. 系统边界

### 2.1 从 balancer 到 kernel

端到端数据流为：

```text
dataset sequence lengths
        |
        v
BR-PBS balancer
  - choose ring_size in {8, 4, 2, 1}
  - choose aligned ring_start
  - reorder by non-increasing ring size
        |
        v
global_seqlens / ring_sizes / ring_starts
        |
        v
per-rank compact Q + rank-major IPC K/V arena
        |
        v
MegaRing Hybrid forward
        |
        +--> O + FP32 LSE
        |
        v
MegaRing Hybrid backward
        +--> dQ + owner-local dK/dV
```

Balancer 负责“放在哪里”，kernel 负责“如何执行”。当前 kernel 不在 device 上搜索
ring placement，也不在热路径中运行跨 rank metadata collective。

### 2.2 已支持范围

| 项目 | Forward | Backward |
| --- | --- | --- |
| GPU | Hopper SM90 | Hopper SM90 |
| dtype | BF16 | BF16 输入/输出，FP32 累加 |
| head dim | 128 | 128 |
| layout | varlen packed | varlen packed |
| attention | causal、noncausal | causal |
| GQA/MQA | `QH % KVH == 0` | `QH % KVH == 0` |
| 通信行宽 | `KVH * D == 1024` | `KVH * D == 1024` |
| physical world | 2、4、8 | 1、2、4、8 |
| hierarchy | 能放入 world 的 G8/G4/G2/G1 | 能放入 world 的 G8/G4/G2/G1 |

完整 G8/G4/G2/G1 hierarchy 需要 8 GPU。Forward 的 world size 1 应使用普通本地
varlen attention；backward 保留 G1-only 的 world size 1 路径。

### 2.3 Public entry 与核心张量

Python 扩展的两个入口是：

```python
forward_varlen_mega_ring(
    q, k, v, remote_k, remote_v,
    cu_seqlens_q, cu_seqlens_k,
    cu_seqlens_q_host, cu_seqlens_k_host,
    max_seqlen_q, max_seqlen_k, is_causal,
    num_comp_sm, num_comm_sm,
    global_seqlens_host, ring_sizes_host, ring_starts_host,
    out=None, lse=None, return_lse=False,
)

backward_varlen_mega_ring(
    dout, q, k, v, out, softmax_lse,
    cu_seqlens_q, cu_seqlens_k,
    cu_seqlens_q_host, cu_seqlens_k_host,
    max_seqlen_q, max_seqlen_k,
    remote_k, remote_v,
    remote_dk_accum, remote_dv_accum, remote_dkv_completion,
    num_comp_sm, num_comm_sm,
    global_seqlens_host, ring_sizes_host, ring_starts_host,
)
```

三组 topology metadata 是 CPU contiguous int32 `[B]`，所有 rank 必须一致。
`ring_sizes` 会复制到 device；`global_seqlens` 和 `ring_starts` 主要用于 host validation。
Forward 返回 compact O，并可返回 backward 复用的 FP32 LSE；backward 返回 compact
dQ/dK/dV。

## 3. 拓扑与数据布局契约

### 3.1 Buddy ring

每条全局序列 $i$ 由三元组表示：

\[
(L_i, G_i, S_i),
\]

其中 $L_i$ 是全局长度，$G_i\in\{1,2,4,8\}$ 是 ring size，$S_i$ 是
ring start。合法性要求：

\[
S_i \bmod G_i = 0,\qquad S_i+G_i\le W,\qquad L_i\bmod G_i=0.
\]

成员 rank 集合为：

\[
\mathcal R_i=\{S_i,S_i+1,\ldots,S_i+G_i-1\}.
\]

若 rank $r\in\mathcal R_i$，其 local length 为 $L_i/G_i$；否则本地
`cu_seqlens` 中该 batch 长度为 0。

对一个成员 rank，subring-local rank 为：

\[
r_G=r-\left\lfloor\frac{r}{G}\right\rfloor G.
\]

step $s$ 对应的 K/V owner 为：

\[
owner(G,r,s)=\left\lfloor\frac{r}{G}\right\rfloor G
             +(r_G-s+G)\bmod G.
\]

固定对齐的 buddy ring 使 device 端不必读取 `ring_starts` 就能恢复 subgroup。
`ring_starts_host` 仍用于 host 侧合法性和 membership 检查。

### 3.2 Batch 顺序为什么是 G8 -> G4 -> G2 -> G1

全局 batch 必须按 ring size 非递增排列：

```text
all G8 batches, then all G4 batches, then all G2 batches, then all G1 batches
```

这不只是便于调度。它保证同一 exact-size ring 的成员在 compact local K/V 中，
该 level 对应一个连续 row range，并且同一 subgroup 内的 rank 对该 range 使用相同
batch-relative offset。于是远端地址可以写成：

```text
owner * rank_kv_capacity + compact_batch_row
```

不需要额外的 `[source_rank, batch] -> remote_offset` 查找表。

### 3.3 Rank-major K/V arena

Q、O、dO 和 dQ 保持 rank-local compact layout：

```text
q, o, dout, dq: [local_total_q, QH, 128]
```

K/V 使用每个进程 shape 相同的 IPC arena：

```text
k, v: [world_size * rank_kv_capacity, KVH, 128]
```

rank $r$ 的 owner-local 数据放在：

```text
[r * rank_kv_capacity,
 (r + 1) * rank_kv_capacity)
```

远端 K/V 只跨 NVLink/IPC 搬运一次到本 GPU arena，随后所有 Q block 和 GQA Q head
都通过本地 HBM TMA 复用。直接让每个 compute CTA 从 peer memory 取 K/V 会按 Q tile
和 Q head 放大远端流量，因此当前实现选择“远端 ingress + 本地复用”的两级路径：

```text
peer HBM -> communication CTA shared memory -> local rank-major arena
          -> compute CTA FA3 TMA -> compute shared memory -> WGMMA
```

### 3.4 固定四层 descriptor

Host binding 为 G8/G4/G2/G1 构造 `MegaRingHierarchyDesc`。每层记录：

```text
ring_size
batch_begin / batch_end
row_begin / full_rows
half_row_begin / half_rows
full_tiles / half_tiles
reduction_base
kv_ready_base
```

固定层数避免引入通用动态 group abstraction，也使 11 个 K/V readiness section
和 15 个 backward dKV section 可以静态编号。

## 4. Megakernel 总体结构

### 4.1 一个 grid 中的 CTA 角色

给定：

```text
C = num_comp_sm
M = num_comm_sm
```

核心 grid 为：

```text
grid.x = C + M

blockIdx.x in [0, C)      -> compute CTA
blockIdx.x in [C, C + M)  -> communication CTA
```

这些是逻辑 SM 配额。由于 kernel 是高资源 persistent CTA，设计上按一 CTA/SM 使用，
但 CUDA 并没有把某个物理 SM 永久命名为 compute 或 communication SM。

Forward 与 backward 的 CTA 生命周期不同：

```text
Forward:
  compute CTA:  attention work -> ... -> exit
  comm CTA:     K/V ingress -> role conversion -> attention work -> exit

Backward:
  compute CTA:  attention backward + local dQ/dKV accumulation -> exit
  comm CTA:     K/V ingress -> wait local dKV -> remote dKV reduce-add -> exit
```

Backward communication CTA 后半程还承担 dKV egress，不能在 ingress 结束后立即转为
compute，因此当前只在 forward 实现动态角色转换。

### 4.2 为什么使用 persistent scheduler

Varlen 与 hybrid placement 会让不同 Q tile 的工作量不同。固定 CTA 到固定 batch
容易产生尾部不均衡。Persistent scheduler 让所有 compute CTA 从一个全局 ticket
源取 work，并复用 FA3 producer/consumer warp-group pipeline。

对 forward causal，ticket 在 step 0 后不再直接表示 `(level, step, tile)`，而是触发
一次“扫描哪个 Q tile 已有可运行连续 segment”的动态 claim。对 backward，ticket
仍精确表示一个 `(level, ring_step, KV tile)`。

## 5. 多个重叠 ring 的执行顺序

多个 ring 的顺序分成三个层次，不能只用一个全局 `for ring` 描述。

### 5.1 Host layout 顺序：大 ring 在前

Batch 的物理顺序固定为 G8 -> G4 -> G2 -> G1。该顺序首先服务于地址一致性，
同时让每个 level 的 batch 和 row range 连续。

### 5.2 Communication task 顺序：优先发布长 critical path

Forward 和 backward 的 K/V ingress task decoder 都按下列 section 建立线性任务空间：

```text
G8 step 1..7
G4 step 1..3
G2 step 1
```

每个 section 内包含 K tasks 和 V tasks。多个 communication CTA/warp 以 stride
方式并行领取 task，因此实际完成顺序不是严格串行，但低 task id 对 G8 的偏置会让
最大 ring 的远端数据较早开始传输。

设计动机是：大 ring 有更多串行依赖 step，且 BR-PBS 通常把更长、更重的序列放入
更大 ring。优先启动大 ring 有利于提前展开最长 critical path。这个动机是否带来
稳定收益，需要用不同 level order 的消融实验验证，不能仅凭代码顺序得出结论。

### 5.3 Compute 顺序：step 0 先行，远端工作按 readiness 驱动

所有模式先处理 step 0：

```text
step 0 = 本 rank 所有非空 G8/G4/G2/G1 batch 的 owner-local attention
```

这里的“先处理”指所有 step-0 ticket 位于 remote-work ticket 之前，不表示存在全局
step-0 completion barrier。当所有 base ticket 已经被领取后，scheduler 可以开始扫描
remote work，即使别的 step-0 CTA 仍在运行；某个 Q tile 自己的 state 只有在其 step 0
epilogue 完成后才从 0 变为 1，因此它的 remote segment 不会越过本 tile 的依赖。

之后：

- **Noncausal forward** 使用线性 exact-size replay：G8 的 7 个 step、G4 的 3 个
  step、G2 的 1 个 step。Persistent queue 允许不同 CTA 并行推进，但 work id 的
  逻辑布局仍按大 ring 到小 ring。
- **Causal forward** 不设置全局 ring barrier，也不要求所有 G8 完成后才执行 G4。
  Scheduler 在 G8/G4/G2 的 Q-tile state 空间中轮转扫描，谁的下一个连续 step 已经
  ready，谁就能被 claim。`scan_cursor` 每次分配 8 个候选，兼顾 level 前缀和跨 tile
  公平性。
- **Causal backward** 使用精确 `(level, step, KV tile)` 线性 stream。它没有
  forward multi-segment claim；每个 work ticket 的梯度归属和 dKV readiness section
  都是确定的。

因此当前方案不是“把多个 ring 严格串行执行”，而是：

```text
大 ring 优先布置和发起通信
        +
step 0 建立每个 Q/KV tile 的本地初始状态
        +
远端工作按局部 readiness 和 persistent queue 动态穿插
```

该策略避免一个较慢 ring 的全局 barrier 阻塞同一 rank 上已经就绪的其他 subring。

### 5.4 一个重叠 hierarchy 的执行例子

考虑：

```text
batch:          b0    b1    b2    b3
global length: 8192  4096  2048  2048
ring size:       G8    G4    G2    G1
ring start:       0     4     2     7
```

成员关系为：

```text
b0: ranks 0..7
b1: ranks 4..7
b2: ranks 2..3
b3: rank 7
```

于是 rank 2 有 G8 和 G2 work，rank 7 有 G8、G4 和 G1 work。其 K/V ingress 的
逻辑 section 分别是：

```text
rank 2: G8 step 1..7, then G2 step 1
rank 7: G8 step 1..7, then G4 step 1..3
```

在 causal forward 中，rank 2 对 G8 的 local rank 是 2：front-Q tile 的 remote
最后一步为 2，back-Q tile 的最后一步为 7；它对 G2 的 local rank 是 0，所以
G2 front-Q tile 只有 step 0，G2 back-Q tile 还有 remote step 1。Rank 7 对 G8/G4
分别是 local rank 7/3，因此其 remote steps 都使用 front-half KV。

Communication decoder 的 task id 仍偏向先发布 G8，但 compute scheduler 不会等
rank 2 的所有 G8 Q tile 完成后才运行 G2。如果 G2 step 1 先 ready，而某些 G8 tile
仍 busy 或等待数据，扫描器可以 claim G2 back-Q tile；同理 rank 7 的 G1 在 step 0
已经完成全部 attention，不参与任何 remote replay。这正是“大 ring 优先启动、多个
ring readiness-driven 穿插”的具体含义。

## 6. Forward 设计

### 6.1 Forward 数据流

```text
                         one persistent launch
  +----------------------------------------------------------------+
  | communication CTAs                                             |
  | peer K/V --2D TMA--> smem --2D TMA--> local rank-major arena   |
  |                                  |                              |
  |                                  +--> release kv_ready          |
  |                                  +--> join compute work queue    |
  |                                                                 |
  | compute CTAs                                                    |
  | step 0 or ready segment                                         |
  |   -> local-HBM TMA -> FA3 online softmax/WGMMA                  |
  |   -> direct first store or in-place O/LSE merge                 |
  |   -> publish per-Q-tile progress                                |
  +----------------------------------------------------------------+
```

Forward kernel 保留 FA3 的 producer warp group、MMA consumer warp groups、TMA
pipeline 和 online softmax。MegaRing 主要扩展 scheduler、K/V 地址映射、epilogue
归约和 kernel wrapper。

这里的“一次 launch”专指核心 `mega_ring_flash_attn_varlen_kernel`。Binding 仍可能在
它之前运行 varlen scheduler metadata preparation，并通过 PyTorch op 初始化 O/LSE。
因此论文若报告完整 op latency，应把这些外围 kernel 计入或明确排除，不能把核心
megakernel 的 launch 数等同于整个 Python API 只有一次 CUDA launch。

### 6.2 Causal zigzag 分解

对 ring size $G$、subring-local rank $r_G$，每个 local shard 分为等长的
front/back 两半。当前对 G2/G4/G8 要求 local half 是 128-row aligned。

执行语义为：

```text
step 0:
  full local Q x local full KV
  apply local diagonal causal mask

step 1..r_G:
  full local Q x remote front-half KV
  no diagonal causal mask

step r_G+1..G-1:
  local back-half Q x remote full KV
  no diagonal causal mask
```

从单个 Q tile 观察：

- front-half Q tile 的最后一步是 `r_G`；当 `r_G == 0` 时它只有 step 0。
- back-half Q tile 的最后一步是 `G - 1`。

这使各 rank 的 causal 有效 attention FLOPs 保持负载均衡，同时避免计算未来 token。

### 6.3 Per-Q-tile 状态机

每个 G8/G4/G2 full-Q tile 有一个 `tile_state`。低位存 `next_step`，最高使用位
`kTileStateBusy` 表示该 tile 正被某个 CTA claim。

```text
initial state = 0
      |
      | step 0 computes and stores O/LSE
      v
state = 1
      |
      | scheduler finds ready [begin, end]
      | CAS(state, state | BUSY)
      v
BUSY with claimed segment
      |
      | one FA3 mainloop over all KV blocks in [begin, end]
      | one epilogue merge
      v
state = end + 1
      |
      +--> if end == last_step, completed_tiles += 1
      +--> otherwise wait for next contiguous ready range
```

CAS lock 保证同一个 Q tile 不会由两个 CTA 并发更新。不同 Q tile 完全独立，不需要
全局 step barrier。

状态发布使用 GPU-scope release store，scheduler 使用 acquire load/CAS。这样观察到
新 `next_step` 的 CTA 也观察到前一个 segment 对 O/LSE 的写入。

### 6.4 动态合并连续 step segment

当 scheduler 看到某个 tile 的 `next_step = b` 时，它依次检查：

```text
b, b+1, ..., last_step
```

对每个 step，读取对应 `(level, step)` 的 K/V readiness counter。扫描在第一个未
ready step 停止，得到最大连续区间 `[b,e]`。只有 `e >= b` 时才尝试 CAS claim。

核心逻辑可以概括为：

```text
while completed_tiles < remote_tiles:
    reduction_idx = next_round_robin_candidate(scan_cursor)
    state = acquire(tile_state[reduction_idx])
    if state <= 0 or state has BUSY:
        continue

    begin = state
    end = largest_contiguous_ready_step(begin, last_step)
    if end < begin:
        continue

    if CAS(tile_state, state, state | BUSY) succeeds:
        return (q_tile, begin, end, end == last_step)
```

Claim 结果压入一个 32-bit `segment_meta`：低 4 bit 是 begin，接下来 4 bit 是 end，
bit 8 表示 terminal segment。Ring size 最大为 8，因此 4 bit 足以编码合法 step 和
invalid sentinel。Producer 与 consumer warp group 通过 scheduler shared-memory slot
读取同一份 packed metadata。

合并条件强调“连续”：不能跳过未就绪 step 去执行后续 step，因为 O/LSE running
state 和 tile_state 只维护单调前缀。连续前缀也使状态只需要一个整数，而不需要
每 tile 的 bitset。

一次 merged segment 的 KV block 数为：

\[
N_{seg}=N_{half}\cdot n_{half}+N_{full}\cdot n_{full},
\]

其中 causal 当前 `BlockN=128`，一个 half segment 含 `half_len/128` 个 block，
一个 full segment 含两倍 block。

Mainloop 把 `[b,e]` 看作一个虚拟连续 K 序列。对每个 virtual N block，通过 O(1)
算术恢复：

```text
(ring_step, source_rank, source-local n_block)
```

而不是构造临时拼接 K/V。映射先处理 step `b..min(e,r_G)` 的 half-KV 区域，再处理
其余 full-KV 区域，并转换为 rank-major arena 中的实际 block index。

### 6.5 Multi-segment 为什么能减少开销

若 $k=e-b+1$ 个 step 分开执行，同一个 Q tile 需要大致经历 $k$ 次：

```text
scheduler claim
Q load
FA3 prologue/mainloop/epilogue
previous O/LSE load
O/LSE merge and store
```

合并后，区间内多个 source rank 的 K/V block 在同一次 mainloop online softmax 中
消费，只在 segment 结尾做一次 epilogue merge。它不减少 attention 的核心矩阵
乘 FLOPs，也不减少每个唯一 K/V 元素的本地读取；主要减少的是 per-step 固定开销、
Q 重复读取和中间 O/LSE 全局内存流量。

实际 segment 长度由通信 readiness 决定：

- 通信足够领先时，一个 claim 可以覆盖多个 step。
- 通信与计算接近时，scheduler 会用较短 segment 保持前进，不等待整个 ring ready。
- 当只有当前 step ready 时，机制自然退化为单 step，不改变正确性。

这是一个 data-ready 自适应优化，而不是固定 `merge_k_steps` 超参数。

### 6.6 O/LSE 在线归约

Step 0 直接写出当前 tile 的 O 和 FP32 LSE，避免先读取调用方预清零的 O。后续
segment 产生 block result $(O_b,L_b)$，与已有 running state $(O_p,L_p)$
做：

\[
L=\log(\exp L_p+\exp L_b),
\]

\[
\alpha=\frac{\exp L_b}{\exp L_p+\exp L_b},
\]

\[
O=O_p+\alpha(O_b-O_p).
\]

LSE 的 row owner 计算 `logaddexp` 和 `alpha`，通过 epilogue shared memory 把每行
scale 交给持有 coalesced O fragment 的线程，再完成 vectorized O load/store。

该归约是 attention 数学上的在线 softmax 合并，不是对局部 O 做简单加法。

### 6.7 Forward SM 角色转换

初始 compute CTA 使用静态起始 work id：

```text
0, 1, ..., C-1
```

后续 work 由同一个 `tile_count_semaphore` 动态分配，并以 `C` 作为 virtual grid
stride。Communication CTA 完成自己 stride 分配的全部 K/V ingress task 后：

1. 执行 CTA-wide `__syncthreads()`，结束 communication shared-memory 生命周期。
2. 调用同一个 FA3 `AttnKernel`，但设置 `start_from_work_queue=true`。
3. 第一个 work 也从全局动态队列领取，而不是使用 `blockIdx.x` 生成静态 id。
4. 之后与原 compute CTA 使用完全相同的 persistent scheduler。

这保证 converted CTA 不会重复计算 `[0,C)` 的初始 work。不同 communication CTA
可以在不同时间完成并逐个加入 compute pool，不需要等待所有通信 CTA 同时转换。

从物理上看，这不是把 CTA 迁移到另一个 SM，而是同一个 persistent CTA 在完成
communication phase 后复用寄存器/共享内存执行 compute phase。Grid 的动态 shared
memory 按 `max(attention_smem, communication_smem)` 分配，确保两种 phase 都合法。

### 6.8 Noncausal forward 的差异

Noncausal 使用 FA3 `BlockM=128, BlockN=176`，每个 remote step 对 full Q 和 full KV
执行。它保留 exact-step replay 和逐 Q tile 的 O/LSE 顺序计数，不启用 causal
multi-segment：

```text
total work = T_all + 7*T8 + 3*T4 + T2
```

Forward SM 角色转换仍然适用。将 noncausal 也改为 multi-segment 是潜在演进方向，
但当前实现没有该分支，文档和论文不能把 causal 优化泛化为所有模式。

## 7. TMA 通信 tile 选择

### 7.1 两级粒度

当前实现区分 logical task 和 physical transfer：

| 路径 | Logical task | Physical TMA subtile |
| --- | --- | --- |
| causal forward K/V ingress | 128 token rows | `16 x 1024` BF16 |
| noncausal forward K/V ingress | 176 token rows | `16 x 1024` BF16 |
| causal backward K/V ingress | 128 token rows | `16 x 1024` BF16 |
| backward dK/dV egress | 128-token block / KV head | `16 x 1024` FP32 |

Forward noncausal 最后一个 logical task 可以少于 176 rows，但 host 保证 row range
至少 128-row aligned，因此 tail 仍是若干完整 16-row subtile，不存在单行 fallback。

### 7.2 为什么 logical task 跟随 FA3 BlockN

Logical task 选择 128/176，与 compute mainloop 的 `BlockN` 一致。其作用是：

1. readiness 以 compute 可消费的 K/V 粒度增长，而不是每行做一次原子更新。
2. 一个 K logical task 和一个 V logical task 完成后，对应一份完整 attention K/V
   tile 的传输进度。
3. 通信 task 数从 row-granular 路径显著下降，减少 decoder、semaphore、TMA commit/
   wait 和 ready-counter 开销。

每个 `(level, step)` 有一个累计 counter。K 或 V logical task 完成后各 `+1`，完整
section 的 ready target 为：

\[
target=2\left\lceil\frac{rows}{BlockN}\right\rceil.
\]

乘 2 对应 K 和 V。Scheduler 只有在达到 target 后才把该 step 纳入 merged segment。

### 7.3 为什么物理 subtile 是 16 rows

固定约束 `KVH * D = 1024` 允许把一个 token row 展平为 1024 个元素：

```text
BF16:  1 row  = 1024 * 2 B = 2 KiB
FP32:  1 row  = 1024 * 4 B = 4 KiB

16-row BF16 tile = 32 KiB
16-row FP32 tile = 64 KiB
```

当前 forward/backward specialization 的 attention CTA 有 12 warps。通信实现把
warps 成对用作 TMA load/store pipeline：

```text
BF16 K/V: up to 6 slots * 32 KiB = 192 KiB
FP32 dKV: up to 3 slots * 64 KiB = 192 KiB
```

再加 barrier 区域仍能放入 Hopper 约 227 KiB 的 shared-memory 上限。在保持当前
6 个 BF16 slot 和 3 个 FP32 slot 并行度时，若把 subtile 扩为 32 rows，shared
memory footprint 会超限；若缩为 8 rows，footprint 更小，但物理 TMA、barrier 和
循环次数翻倍。选择 16 rows 可以保留当前 warp 配对并行度，同时统一 BF16 ingress
和 FP32 egress 的实现。更大的 subtile 可以通过减少 slot 数实现，但是否更快必须
由消融实验判断，当前设计不把 16 rows 声称为所有 workload 的全局最优。

Logical task 不会整体驻留 shared memory。以 causal K/V 为例，一个 128-row task
依次通过 8 个 16-row transfer，复用少量 slot。因此“128-row 通信 tile”不意味着
为 K 或 V 分配一个完整 256 KiB shared tile。

### 7.4 Load/store warp 配对与可见性

每个 slot 有两个 semaphore phase：

```text
load warp:
  wait slot finished
  expect 16-row bytes
  peer global -> shared TMA load

store warp:
  wait load arrived
  shared -> local global TMA store
  wait store finished reading shared
  release slot
  wait global store completion
  publish logical-task readiness
```

`store_async_read_wait` 只保证 shared slot 可复用，不保证 compute CTA 已能观察本地
HBM 写入。因此 readiness 必须在 `store_async_wait` 之后用 release increment 发布，
compute/scheduler 通过 acquire load 消费。

## 8. Backward 设计

### 8.1 Backward 不是单一 kernel 的完整替代

Backward 沿用 FA3 的多阶段结构：

```text
1. preprocess kernel
   O, dO, LSE -> dPsum / LSE-log2 / zero or prepare accumulators

2. MegaRing backward core megakernel
   compute CTAs: attention backward -> dQaccum + step-local dK/dV
   comm CTAs:    K/V ingress -> remote FP32 dK/dV reduce-add

3. one-thread completion wait kernel
   wait until this owner has received all peer contributions

4. postprocess kernels
   FP32 dQaccum -> BF16 dQ
   owner FP32 dK/dV accum -> compact BF16 dK/dV
```

因此准确表述是“backward 的核心 attention/通信/跨卡 dKV 归约由一个 megakernel
融合”，而不是“整个 backward 只有一个 CUDA kernel”。

### 8.2 Backward scheduler 为什么以 KV tile 为 work

FA3 backward 以 K/V 的 N block 为外层 work tile，并在 mainloop 中遍历相关 Q block。
MegaRing backward 保持该结构。一个 ticket 精确表示：

```text
(ring level, ring step, KV block, Q head, batch)
```

工作流为：

```text
step 0: all local full-KV tiles of G8/G4/G2/G1
G8: step 1..7
G4: step 1..3
G2: step 1
G1: no remote replay
```

对于 subring-local rank $r_G$，每层 work 数为：

\[
T_{full}+r_G T_{half}+(G-1-r_G)T_{full}.
\]

这里 step `1..r_G` 调度 front-half KV tile，后续 step 调度 full KV tile。当前 backward
没有 multi-segment claim，也不让 fragment 跨 ring step 存活。

### 8.3 Causal backward 区域

Backward 的 causal 区域与 forward 一致：

```text
step 0:
  local full KV against causally valid local Q

step 1..r_G:
  remote front-half KV against full local Q

step r_G+1..G-1:
  remote full KV against local back-half Q
```

Compute mainloop 根据 `(ring_level, ring_step)` 修改 K/V owner offset、Q offset 和
sequence length。Remote step 在读取 K/V 前等待精确的 ingress readiness section。

### 8.4 dQ 累加

不同 K/V tile 和 ring step 都会对同一 local Q 产生 dQ contribution。Compute CTA
把 contribution reduce-add 到本地 FP32 `dq_accum`，最后由 postprocess 转换成 BF16
compact dQ。

由于当前 mega-ring backward 是 non-deterministic specialization，不保证跨 CTA
浮点加法顺序固定。Deterministic backward 不在支持范围。

### 8.5 Step-local dK/dV buffer

每个 compute work 对当前 K/V owner 产生 dK/dV。为避免多个 ring step 在同一 buffer
中混淆，核心 kernel 使用：

```text
dk_steps, dv_steps: [world_size, step_stride]
step_stride = KVH * padded_rank_capacity * 128 floats
```

不同 hierarchy level 复用 step dimension，但 batch range 不重叠。每个 batch 后加入
128 个 zero padding rows：

```text
padded_row(batch b) = compact_row + b * 128
```

Padding 让每个 level 的范围都能按 128-token block 解码，避免 dKV egress 的非对齐
tail。Padding 初始为 0，像普通数据一样 reduce-add，不影响 compact 输出。

### 8.6 15 个 dKV readiness section

Backward 为每个 `(level, step)` 分配 section：

```text
G8 step 0..7 -> section 0..7
G4 step 0..3 -> section 8..11
G2 step 0..1 -> section 12..13
G1 step 0    -> section 14
```

每个 compute epilogue 在完成一份 step-local dK/dV tile 后对 `local_ready[section]`
做 release increment。Communication CTA 在 section 达到 host 计算的 expected count
后，才开始读取该 step buffer。

### 8.7 Owner-directed remote dKV reduce-add

Communication CTA 在 K/V ingress 后进入 dKV egress phase。对每个有效
`(level, step)`：

1. 等待该 section 的所有本地 dK/dV tile ready。
2. 解码 level 的 padded batch row range、KV head 和 128-token block。
3. 从 step-local FP32 buffer 通过 TMA load 到 shared memory。
4. 对 `owner(G,r,step)` 的 IPC FP32 accumulator 执行 remote TMA reduce-add。

对于一个 KV head 的 128-token block：

```text
128 tokens * 128 dims = 16384 FP32 values
                       = 16 descriptor rows * 1024 values
```

因此一次 dK 或 dV egress task 正好是一笔 `16 x 1024` FP32 TMA reduce-add。

这种 owner-directed 设计使最终每个 rank 只 postprocess 自己原始 K/V shard 的梯度，
无需在 Python 中 materialize 所有 step 的 dK/dV 再做 collective。

当前 dKV egress 按 G8 -> G4 -> G2 -> G1、step 递增的固定 section 顺序推进。若前面
section 的 compute 尚未完成，communication CTA 会等待，即使后面 section 已 ready。
这是用简单 completion protocol 换取潜在 head-of-line blocking 的明确权衡；改成
readiness-driven dKV section queue 是未来优化，而不是当前已实现能力。

### 8.8 跨 GPU completion 协议

每个 communication CTA 完成 section 中自己的 strided tasks 后，增加本地
`dkv_comm_done[section]`。最后一个 CTA 对目标 owner 的 IPC completion scalar 做
system-scope release increment。

一个 owner 的期望完成值为：

\[
E_r=\sum_{\ell:\,rows_\ell(r)>0}G_\ell.
\]

原因是对 size $G$ 的一个本地 level，最终会收到该 subring 中 $G$ 份
rank/step contribution。Core megakernel 之后，同一 CUDA stream 上的单线程 wait
kernel 用 system-scope acquire 轮询本 rank completion。达到 $E_r$ 后才允许 dK/dV
postprocess 读取 owner accumulator。

调用方必须在每次 backward 前把 owner FP32 accumulators 和 completion scalar 清零，
完成 CUDA 同步和 distributed barrier。这个准备过程不属于 core megakernel。

### 8.9 为什么 backward 不做 forward 式角色转换

Backward communication CTA 在 ingress 完成时，compute CTA 通常刚开始产生 dK/dV。
它们必须继续存活，等待 15 个 section 的 local readiness，并执行远端 reduce-add。
若此时转为 compute，会失去专门处理 dKV egress 的 CTA，或者需要第二次角色反转和
更复杂的全局状态机。

当前选择是保持 compute/communication 角色稳定，以换取简单、可证明的 completion
协议。未来可以研究阶段化 CTA pool，但必须同时证明不会降低 dKV egress 并行度或
引入跨 rank deadlock。

## 9. 同步协议与正确性不变量

### 9.1 Counter 总览

| 状态 | 粒度 | Producer | Consumer | Scope |
| --- | --- | --- | --- | --- |
| forward/backward `kv_ready` | `(level, remote step)`，按 logical K/V task 累计 | comm CTA | scheduler/compute CTA | GPU |
| forward `tile_state` | Q tile | compute epilogue | causal dynamic scheduler | GPU |
| forward `completed_tiles` | terminal remote Q tile | compute epilogue | scheduler exit | GPU |
| backward `local_ready` | `(level, step)`，按 dKV tile 累计 | compute epilogue | comm CTA | GPU |
| backward `dkv_comm_done` | `(level, step)`，按 comm CTA 累计 | comm CTA | last comm CTA | GPU |
| backward owner completion | owner rank | last comm CTA | owner wait kernel | System |

### 9.2 关键 happens-before

Forward K/V：

```text
peer TMA load
  -> local TMA store complete
  -> release kv_ready
  -> acquire kv_ready
  -> compute TMA reads local arena
```

Forward O/LSE：

```text
segment epilogue store/merge
  -> release tile_state = end + 1
  -> acquire/CAS next claim
  -> next segment reads running O/LSE
```

Backward dKV：

```text
compute epilogue writes step-local dKV
  -> release local_ready
  -> acquire local_ready by comm CTA
  -> remote TMA reduce-add
  -> system release owner completion
  -> system acquire owner wait
  -> owner dKV postprocess
```

### 9.3 必须保持的不变量

1. 同一个 forward Q tile 在任意时刻最多有一个 remote segment owner。
2. `tile_state` 只按连续前缀单调增长，不能跳 step。
3. K/V readiness 只能在 local global store 完成后发布。
4. Step 0 是 forward O/LSE 的唯一初始化者；later empty step 不能覆盖 running state。
5. Backward dKV section 只有在所有对应 compute tile 都发布后才能远端归约。
6. Owner dK/dV postprocess 只能在所有相关 subring contribution 到达后运行。
7. 所有 rank 必须使用一致的 `global_seqlens/ring_sizes/ring_starts`。

## 10. 性能模型与设计权衡

### 10.1 理想重叠模型

记 compute 工作时间为 $T_c$，通信时间为 $T_m$，串行外围开销为 $T_o$。理想
megakernel 时间接近：

\[
T\approx \max(T_c,T_m)+T_o,
\]

而非：

\[
T_c+T_m+T_o.
\]

实际还受到以下因素影响：

- `num_comp_sm:num_comm_sm` 分配是否匹配 workload。
- 不同 rank 的 critical path 和负载是否由 BR-PBS 平衡。
- K/V readiness 是否足够早，使 causal segment 能合并多个 step。
- rank-major arena 的本地 HBM 二次写入和读取成本。
- forward O/LSE 中间态的全局内存流量。
- backward dKV FP32 remote reduce-add 带宽和 system-scope completion 尾部。

### 10.2 角色转换的收益边界

Forward communication phase 的工作量有限，若 M 个 SM 在搬运结束后直接退出，
剩余长尾只能由 C 个 compute SM 完成。角色转换使后半程最多可用 C+M 个 CTA 消化
queue，主要针对 compute-heavy tail。

但它不是无条件收益：converted CTA 进入 compute 前有同步、FA3 prologue 和 scheduler
开销；若 compute 已接近结束，额外 CTA 可能贡献有限。论文应按 sequence length、
ring mix 和 SM 配额报告消融，而不是只给单点结果。

### 10.3 Multi-segment 的收益边界

设一个 Q tile 的远端 step 数为 $s$，实际形成 $c$ 个 segment，$1\le c\le s$。
相对逐 step 路径，理论上 per-step epilogue/claim 次数从 $s$ 降到 $c$。但 $c$
由通信 ahead distance 决定：

- 更多 communication SM 可能增大 segment，但会减少初始 compute SM。
- 更少 communication SM 可能提高前期 compute 并行度，却让 segment 退化为单 step。

因此 `num_comp_sm:num_comm_sm` 与 segment fusion 存在耦合，调优时不能独立看待。

### 10.4 为什么不直接使用 NCCL ring

当前设计针对单节点 Hopper IPC/TMA：

- 通信 task 可以在同一 kernel 中通过 device counter 与 compute CTA 同步。
- K/V 可以直接写入 rank-major local arena。
- Backward 可以对 peer FP32 accumulator 做 TMA reduce-add。

NCCL 提供更通用的跨节点 transport 和 collective 算法，但不能直接替代本设计的
kernel 内 tile readiness、Q-tile segment claim 和 in-kernel dKV owner completion。
两者应作为不同系统边界的 baseline 比较，而不是声称一个普遍替代另一个。

## 11. 失败模式与防御

### 11.1 Deadlock 风险

最危险的循环依赖是：scheduler 等待某 Q tile progress，而该 tile 的 consumer
epilogue 尚未发布 progress，consumer 又提前向 producer 请求下一份 work。当前
causal chunked 路径在 epilogue 完成、清除 busy bit 并发布新 state 后，才调用
`get_next_work()`，打破该环。

Backward 的主要风险是 comm CTA 等待错误的 expected count。Host binding 为每个
level/step 精确计算 full/half tile 数，empty level 不参与 owner completion。

### 11.2 地址越界风险

Host 侧拒绝：

- 非 2/4/8 的 forward world 或非 1/2/4/8 的 backward world。
- 非法/越界/未对齐 buddy ring。
- 全局长度不能整除 ring size。
- local length 与 membership 不一致。
- batch 未按 ring size 非递增排列。
- local sequence、arena 或 causal half 不满足 128-row alignment。
- `KVH * D != 1024`。
- arena capacity 不足或 TMA 指针不满足对齐。

### 11.3 Empty rank

某 rank 可以没有任何有效 batch，但仍参与 IPC setup 和 kernel launch。Binding 使用
一行 dummy Q/O/LSE backing 避免构造零 extent TMA descriptor；device `cu_seqlens`
保持所有 batch 长度为 0，所以 scheduler 不产生逻辑 tensor access。

## 12. 实现映射

| 机制 | 主要文件 |
| --- | --- |
| Python/CUDA forward binding、hierarchy 构造 | [`csrc/mega_ring_min_fa3_varlen_ring_bindings.cu`](../csrc/mega_ring_min_fa3_varlen_ring_bindings.cu) |
| Forward fused wrapper、通信 CTA、角色转换 | [`include/mega_ring_min_fa3_varlen_ring_launch.h`](../include/mega_ring_min_fa3_varlen_ring_launch.h) |
| Forward hierarchy scheduler、segment claim | [`include/mega_ring_min_fa3_varlen_scheduler.h`](../include/mega_ring_min_fa3_varlen_scheduler.h) |
| Forward persistent FA3 wrapper | [`include/min_fa3_kernel.h`](../include/min_fa3_kernel.h) |
| Forward virtual N-block mapping、KV wait | [`include/min_fa3_mainloop.h`](../include/min_fa3_mainloop.h) |
| Forward O/LSE online merge | [`include/min_fa3_epilogue.h`](../include/min_fa3_epilogue.h) |
| Packed segment metadata、acquire/release 原语 | [`include/mega_ring_semaphore.cuh`](../include/mega_ring_semaphore.cuh) |
| G8/G4/G2/G1 shared descriptor | [`include/min_fa3_mega_ring_hierarchy.h`](../include/min_fa3_mega_ring_hierarchy.h) |
| Backward binding、buffer 与 expected count | [`csrc/backward/min_fa3_bwd_bindings.cu`](../csrc/backward/min_fa3_bwd_bindings.cu) |
| Backward fused wrapper、K/V ingress、dKV egress | [`include/backward/min_fa3_bwd_launch.h`](../include/backward/min_fa3_bwd_launch.h) |
| Backward exact-size persistent scheduler | [`include/backward/min_fa3_bwd_scheduler.h`](../include/backward/min_fa3_bwd_scheduler.h) |
| Backward rank/step address 与 causal 区域 | [`include/backward/min_fa3_bwd_mainloop.h`](../include/backward/min_fa3_bwd_mainloop.h) |
| Backward step-local dKV store/readiness | [`include/backward/min_fa3_bwd_epilogue.h`](../include/backward/min_fa3_bwd_epilogue.h) |
| Backward producer/consumer kernel | [`include/backward/min_fa3_bwd_kernel.h`](../include/backward/min_fa3_bwd_kernel.h) |

## 13. 验证与论文实验建议

### 13.1 正确性验证

至少覆盖：

1. G8/G4/G2/G1 同时存在并互相重叠的 8-GPU case。
2. 所有合法 G4 start 和 G2 start。
3. G1-only、all-CP 和 empty-rank。
4. Forward causal/noncausal，backward causal。
5. MHA 与 GQA，特别是当前 `QH=16/32, KVH=8, D=128`。
6. 连续多轮执行，确保所有 counter 每次初始化且无跨轮串扰。
7. Arena padding sentinel 和 remote K/V bitwise copy 检查。
8. Forward O/LSE 以及 backward dQ/dK/dV 与完整逻辑 attention reference 对比。
9. `compute-sanitizer` 的 racecheck/synccheck，尤其是 TMA slot 和 named barrier。

当前入口可参考根 [README](../README.md) 中的 hierarchical mega-ring tests。

### 13.2 必做消融

为了把设计写入论文，建议至少报告以下 A/B：

| 消融 | 对照 | 目的 |
| --- | --- | --- |
| Megakernel | Python/NCCL ring 或逐 step kernel | 分离 launch 与融合收益 |
| Forward role conversion | comm CTA 搬运后退出 | 衡量 compute tail 利用率 |
| Causal multi-segment | 固定单 step claim | 衡量 claim、Q load、O/LSE merge 减少 |
| Logical tile copy | row-granular ingress | 衡量 TMA/atomic/task 开销 |
| 16-row subtile | 8-row，资源允许时测试 32-row | 验证 shared memory 与指令数权衡 |
| Ring order | G8-first 对比小 ring-first/round-robin | 验证 critical-path 假设 |
| Hybrid placement | all-CP、G1-only、threshold hybrid | 分离 balancer 与 kernel 贡献 |
| SM split | 多组 `comp:comm` | 展示 workload-dependent 最优点 |

### 13.3 建议 profiler 指标

- end-to-end iteration latency 和 max-rank latency。
- aggregate TFLOP/s 与 per-GPU TFLOP/s。
- forward `completed_tiles` 对应的平均/分位 segment 长度，需要增加低开销统计版。
- TMA load/store 指令数、NVLink throughput、本地 HBM throughput。
- compute CTA 等待 `kv_ready` 的 stall 时间。
- communication CTA 转换时间分布，以及转换后完成的 work tile 比例。
- O/LSE global load/store bytes。
- backward dKV reduce-add 带宽和 completion wait tail。
- 不同 rank 的 compute/token load 与最终 kernel time 相关性。

Benchmark 必须报告端到端 op time，并明确是否包含 scheduler metadata preparation、
buffer reset、distributed barrier 和 backward preprocess/postprocess。只比较 core kernel
时间容易把外围成本隐藏掉。

## 14. 可用于论文的贡献表述

在完成上述实验后，核心贡献可以组织为：

1. **Hierarchical hybrid ring execution**：在一个 physical world 中，以单个
   persistent megakernel 执行多个重叠、不同大小的 buddy ring，使 sequence-level
   CP degree 与 kernel execution 解耦。
2. **In-kernel communication/compute orchestration**：使用专门的 communication CTA
   和 tile readiness，把 peer K/V ingress 与 FA3 compute 重叠；forward 在通信完成后
   将 CTA 动态回收为 compute worker。
3. **Readiness-driven causal segment fusion**：以 per-Q-tile 单调状态机动态合并连续
   ready ring steps，在不 materialize 拼接 K/V 的情况下复用一次 FA3 online-softmax
   mainloop。
4. **Fused distributed gradient ownership**：backward 在核心 megakernel 中生成
   step-local dKV，并通过 remote TMA reduce-add 直接归约到 K/V owner，以 system-scope
   completion 保证 postprocess 顺序。
5. **Balancer-kernel co-design**：BR-PBS 输出的 buddy topology、batch ordering 和
   alignment 直接满足 kernel 的连续地址与固定 TMA tile 契约。

写作时应避免以下过度表述：

- 不应说 backward 全流程只有一次 kernel launch。
- 不应把 causal multi-segment 写成 noncausal 已支持。
- 不应声称跨节点支持；当前 transport 是单节点 IPC/TMA。
- 不应把解析上的通信计算重叠直接等同于完全隐藏通信。
- 不应把某个固定 SM split 声称为普遍最优。
- 不应只凭 G8-first 的实现顺序声称它一定优于其他顺序。

## 15. 当前非目标与演进方向

当前明确不支持：

- backward noncausal 或 deterministic mega-ring。
- FP16、FP8、非 128 head dim。
- 非 SM90、跨节点 transport。
- 任意 rank 子集或非 buddy ring。
- 不对齐通信 tail。
- backward multi-segment 和 backward communication CTA -> compute CTA 转换。
- paged KV、append KV、rotary、local attention、softcap、split-KV。

合理的后续方向包括：

1. 给 causal segment 长度增加 profile-only 统计，建立可验证的收益模型。
2. 将实测 kernel latency、NVLink 拥塞和最优 SM split 反馈给 BR-PBS 成本模型。
3. 研究 noncausal segment fusion，但保持独立 specialization 和 A/B 验证。
4. 研究 backward 阶段化 CTA pool，前提是完整保留 dKV completion 正确性。
5. 评估 G8-first、deadline/criticality-first 和 readiness-only 通信顺序。
6. 在不放大远端 K/V 流量的前提下，评估更细粒度的 compute-side tile readiness。

这些方向应继续遵守本目录的“copied + trimmed”原则：调度和通信扩展可以演进，
FA3 params、kernel、mainloop、epilogue 与 scheduler 的主体结构仍需保持可追溯。
