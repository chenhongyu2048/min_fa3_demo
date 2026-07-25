# MegaRing 论文设计参考（基于当前实现核查）

> 用途：本文是 MegaRing 论文“设计/方法”部分的统一技术底稿。它以当前仓库实现为事实依据，连接层级化 mega kernel 与 Buddy-Ring Pareto Beam Scheduler（BR-PBS），并显式区分：
>
> - **实现事实**：可在当前源码中定位，适合写入方法章节；
> - **设计动机**：解释为何采用该设计，但不自动构成性能结论；
> - **待验证假设**：必须由消融、profiler 或端到端实验支撑，不能仅因代码存在而声称有效。

本文不将裁剪自 FlashAttention-3（FA3）的 WGMMA、TMA pipeline、online softmax 等基础实现包装成 MegaRing 的原创贡献。MegaRing 的贡献在于：在保留 FA3 attention math kernel 的条件下，重新组织 heterogeneous context parallel（CP）的拓扑、通信、调度、在线归约与负载规划。

## 1. 核查范围与核心结论

本次核查覆盖以下实现和设计资料：

| 范畴 | 核查入口 | 作用 |
| --- | --- | --- |
| Hybrid megakernel 既有说明 | `docs/MEGARING_HYBRID_KERNEL_DESIGN.md` | 现有 forward/backward、TMA、同步协议的设计基线 |
| Forward host binding 与 launch | `csrc/mega_ring_min_fa3_varlen_ring_bindings.cu`、`include/mega_ring_min_fa3_varlen_ring_launch.h` | 验证 host 元数据、资源配置和真实 launch 边界 |
| Forward scheduler / FA3 接入 | `include/mega_ring_min_fa3_varlen_scheduler.h`、`include/min_fa3_kernel.h`、`include/min_fa3_mainloop.h` | 验证 causal segment claim、角色转换和实际 attention 工作计数 |
| Backward megakernel | `include/backward/`、`csrc/backward/` | 验证 step-local dKV 与 owner-directed reduce-add 路径 |
| BR-PBS | `balancer/load_balancer.py`、`balancer/sampler.py`、`balancer/test_balancer.py` | 验证 candidate、beam、relaxation、fill 和 repair 的真实行为 |
| 负载观测与 benchmark 接口 | `ring_test/forward_load_model.py`、`ring_test/benchmark_topology_forward.py` | 验证静态模型与 runtime tile counter 的对应关系 |

核查后的总体判断如下。

1. **系统是清晰的 co-design，而不是两个相互独立的组件。** BR-PBS 输出的 buddy ring、按 ring size 重排的 batch、对齐约束和 rank-local packed layout，直接减少了 device 端地址查询和动态分组；megakernel 则把这些静态决策变成设备端可执行的 fixed-level descriptor。
2. **forward 的核心技术重点是“readiness-driven persistent execution”，不是单纯把通信和 attention 放在同一个 CUDA kernel。** 它同时包含 K/V ingress readiness、per-Q-tile CAS 状态机、causal 连续 step segment 合并、online O/LSE 合并，以及 communication CTA 到 compute CTA 的动态角色转换。
3. **负载均衡器不是单目标的 token partitioner。** BR-PBS 将 token、二次 attention compute、短序列切分、通信 token-hop proxy 和拓扑分散度置于明确的字典序中；搜索过程采用 Pareto beam、2% 量化、多样性截断、渐进式候选解锁和层级局部修复。
4. **近期实现新增了一条重要的“模型到执行”观测闭环。** 可选的 stats-specialized forward kernel 真实统计 Q/O work visit 与 attention KV tile read；它不会计入 16-row 通信 TMA 子传输。该机制应成为论文验证 planner proxy 和 causal segment fusion 的关键测量工具。
5. **需要如实描述两阶段合法性。** Causal BR-PBS 的 `L mod (256G)=0` 与 kernel 的 local-half 对齐契约一致；noncausal BR-PBS 只保证 `L mod G=0`，而实际 kernel 还要求每个 local shard 为 128-row 对齐。因此 noncausal 方案必须经过 benchmark/frontend 或 binding 的 launchability 检查，不能写成“planner 单独保证所有 kernel shape 合法”。

## 2. 问题、符号与系统边界

### 2.1 问题

给定一个单节点 Hopper world，world size 为

\[
W\in\{2,4,8\},
\]

以及一个长度异构的 sequence batch，MegaRing 需要为每条序列选择 context-parallel degree，并在不同 CP degree 的序列互相重叠时保持：

1. attention 数值正确；
2. rank 间 token 与 attention compute 尽量平衡；
3. 不让短序列为追求 load balance 而被过度切分；
4. 将 peer K/V 传输与 attention compute 重叠；
5. 避免传统 ring attention 的逐 step host launch、Q 重载和重复 O/LSE 归并。

一条全局序列 \(i\) 由三元组表示：

\[
(L_i,G_i,S_i),
\]

其中 \(L_i\) 是 global length，\(G_i\in\{1,2,4,8\}\) 是 buddy ring size，\(S_i\) 是 aligned ring start。合法 buddy ring 的成员集合为

\[
\mathcal{R}_i=\{S_i,\ldots,S_i+G_i-1\},
\]

并要求

\[
S_i\bmod G_i=0,\qquad S_i+G_i\le W,\qquad L_i\bmod G_i=0.
\]

rank \(r\) 属于该集合时，local shard length 为 \(L_i/G_i\)；否则该 rank 在该 batch slot 的 local packed length 为零。令 \(r_G=r-\lfloor r/G\rfloor G\)，ring step \(s\) 所需的 K/V owner 是

\[
owner(G,r,s)=\left\lfloor\frac{r}{G}\right\rfloor G
 +(r_G-s+G)\bmod G.
\]

该 owner 公式只依赖 aligned buddy topology、rank 和 step。因此 device 端不需要读取 `ring_starts`；`ring_starts` 只在 host 验证 membership 与合法性。

### 2.2 支持范围

当前实现的真实边界应在论文实现章节或附录中明确：

| 项目 | Forward | Backward |
| --- | --- | --- |
| GPU / transport | 单节点 Hopper SM90；CUDA IPC / peer TMA | 同左 |
| 数据类型与 head dim | BF16，`D=128` | BF16 I/O，FP32 accumulation，`D=128` |
| GQA/MQA | `QH % KVH == 0` | `QH % KVH == 0` |
| 通信行宽 | `KVH * D == 1024` | `KVH * D == 1024` |
| physical world | 2、4、8 | 1、2、4、8 |
| attention mode | causal、noncausal | causal |
| hierarchy | 同一 launch 中的 G8/G4/G2/G1 | 同一 hierarchy 的 owner-directed dKV 归约 |

这些不是可选调参，而是当前模板实例化、TMA descriptor 与 host validation 的硬边界。尤其 `KVH*D=1024` 使一 token row 可以平铺为 1024 个元素，是 `16 x 1024` 通信 tile 的前提。

### 2.3 端到端分层

```text
dataset length statistics / user-provided lengths
                    |
                    v
           BR-PBS topology planner
      choose (L_i, G_i, S_i), reorder batch
                    |
                    v
  per-rank packed Q + rank-major K/V IPC arenas
                    |
                    v
  host validation + fixed G8/G4/G2/G1 descriptor
                    |
                    v
      one forward persistent mega kernel
  comm CTA: peer K/V ingress -> local arena -> ready
  compute CTA: FA3 attention -> online O/LSE reduction
                    |
                    +--> forward O and FP32 LSE
                    |
                    v
  backward helper kernels + backward core megakernel
  compute: dQ/dKV production; comm: remote dKV reduce-add
                    |
                    v
          owner-local compact dQ/dK/dV
```

BR-PBS 决定“序列放在哪个 buddy ring”，kernel 决定“已知 topology 如何在 GPU 上执行”。device 端不会重新搜索 placement，也不会在 hot path 中执行跨 rank metadata collective。

## 3. 贡献一：Hierarchical Hybrid Mega Kernel

### 3.1 设计原则和真正的新内容

MegaRing 保留裁剪 FA3 forward/backward 中已验证的 attention core：warp-specialized producer/consumer、TMA pipeline、WGMMA、online softmax 和 varlen prepared scheduler metadata。新设计集中于以下 execution organization：

1. 在一个 persistent grid 中共同驻留 compute CTA 与 communication CTA；
2. 用 rank-major arena 将 peer K/V ingress 一次搬入本地 HBM，供多个 Q tile 与 GQA heads 重用；
3. 用固定四层 hierarchy descriptor 表示重叠 G8/G4/G2/G1；
4. 用 tile-granular ready counter 与 per-Q-tile state，让 compute 只消费已就绪的 remote K/V；
5. 在 causal 模式合并连续 ready steps，减少 per-step FA3 固定开销；
6. 在 forward 回收通信 CTA 为 compute CTA，缩短 communication-tail 之后的 compute tail；
7. 在 backward 中将 dKV 直接 remote reduce-add 到 owner，而不是 materialize 后再做独立 collective。

因此论文中的正确定位是：**MegaRing 改造了 attention 的跨设备执行图与调度图，而没有改写 attention 数学或从零设计一个新 GEMM kernel。**

### 3.2 Rank-major K/V arena 与地址契约

Q、O、dO、dQ 保持 rank-local compact layout：

```text
q, o, dout, dq: [local_total_q, QH, 128]
```

而 K/V 是每个进程持有、但通过 IPC 可由 peer 访问的统一 arena：

```text
k, v: [W * rank_kv_capacity, KVH, 128]
```

rank \(r\) 的 owner-local K/V 区域为

```text
[r * rank_kv_capacity, (r + 1) * rank_kv_capacity)
```

其中 `rank_kv_capacity` 必须为 128 rows 的倍数，local packed K/V total 不得超过该 capacity。每次 remote step 的地址可以直接写为

```text
source_rank * rank_kv_capacity + compact_row
```

而不需要 `[source_rank, batch] -> remote_offset` indirection table。

这是一项同时服务性能和实现可控性的布局优化：

```text
peer HBM
  -> communication CTA shared-memory staging
  -> local rank-major K/V arena
  -> FA3 local-HBM TMA / shared memory / WGMMA
```

与每个 compute CTA 直接读取 peer HBM 相比，remote K/V 只跨设备 ingress 一次，之后可被多个 Q tiles 和 GQA Q heads 在本地 HBM 上复用。代价是多了一次本地 HBM store 与后续 local read；是否净收益取决于 remote bandwidth、reuse 和 workload，需通过 profiler 验证。

### 3.3 Batch order 是 kernel ABI 的一部分

BR-PBS 的输出在送入 kernel 前按以下键排序：

```text
descending ring_size, then ascending ring_start, then original_index
```

因此 batch 呈现为连续的 G8、G4、G2、G1 区段。这个排序不是展示层格式，而是 device-side descriptor 的正确性前提：host 可以为同 size batch 生成连续 row range，并使同一 exact-size level 的 tile space 连续。binding 会拒绝 `ring_sizes` 非递减的输入。

固定 descriptor 为每层记录：

```text
ring_size
batch_begin / batch_end
row_begin / full_rows
half_row_begin / half_rows
full_tiles / half_tiles
reduction_base
kv_ready_base
```

层数固定为 4，顺序为 G8/G4/G2/G1。远端 K/V readiness 只需要 11 个 section：

```text
G8: step 1..7  -> 0..6
G4: step 1..3  -> 7..9
G2: step 1     -> 10
G1: no remote K/V section
```

fixed-size descriptor 避免了通用动态 group object、device pointer chasing 和 variable-length topology decoding。它是“针对最多 8 GPU buddy hierarchy 的专用化”，不应被描述为任意 world size 或任意 rank subset 的通用方案。

### 3.4 一个 persistent grid 中的 CTA 角色

给定逻辑 SM 预算 `C=num_comp_sm`、`M=num_comm_sm`，forward grid 是

```text
grid.x = C + M

blockIdx.x in [0, C)   : initial compute CTAs
blockIdx.x in [C, C+M) : communication CTAs
```

这表达的是 persistent CTA 的逻辑资源配额，而不是将物理 SM 永久命名。binding 检查 \(C>0\)、\(M\ge0\) 和 \(C+M\le\) 设备 SM 数。实际共享内存配置为

\[
S_{launch}=\max(S_{FA3},S_{comm}),
\]

并通过 `cudaFuncAttributeMaxDynamicSharedMemorySize` 申请所需动态 shared memory。由此同一个 CTA 可先使用 communication staging，再复用同一 block 执行 FA3 compute。

初始 compute CTA 使用静态 work id `0..C-1`；其余 work 从全局 `tile_count_semaphore` 获取。communication CTA 结束自身 strided ingress task 后执行 CTA-wide 同步，并以 `start_from_work_queue=true` 进入同一个 attention persistent queue。该首个 work 也从队列领取，避免重复初始 `0..C-1` work。

这项 **forward role conversion** 的意义是：每个 comm CTA 完成 ingress 后可独立加入 compute pool，不必等待所有 communication CTA，也不会让这些 CTA 在 compute-heavy tail 中闲置。它不是 CTA 在物理 SM 间的迁移。

Backward 中不进行该转换：communication CTA 在 ingress 后还必须等待 local dKV section ready，再执行 remote reduce-add；若过早转为 compute 会破坏 dKV egress 并行度或需要更复杂的双向角色状态机。

### 3.5 K/V ingress：逻辑 attention tile 与物理 TMA subtile 分离

Forward communication 采用两级粒度：

| 模式 | logical K/V task | physical TMA transfer |
| --- | --- | --- |
| causal forward | 128 token rows | `16 x 1024` BF16 |
| noncausal forward | 176 token rows | `16 x 1024` BF16 |
| causal backward ingress | 128 token rows | `16 x 1024` BF16 |
| backward dK/dV egress | 128-token KV-head block | `16 x 1024` FP32 reduce-add |

`BlockN` 决定 logical task 大小，因此 readiness 对应“compute 可消费的一整个 attention K/V tile”，而不是 token row。每个 logical K task 或 V task 由若干 16-row transfer 完成后才发布一次 counter；一个 ready target 等于

\[
2\left\lceil\frac{rows}{BlockN}\right\rceil,
\]

系数 2 对应 K 和 V 各一份 task。`16 x 1024` BF16 tile 为 32 KiB；当前 12-warp attention CTA 将 warps 成对组织，得到最多 6 个 BF16 staging slots，即约 192 KiB，另留 barrier 空间。FP32 dKV path 使用 3 个 64 KiB slots，也约为 192 KiB。

通信 CTA 的 load/store warps 配对使用 phase semaphore：

```text
load warp:
  wait(slot reusable)
  peer-global -> shared TMA

store warp:
  wait(shared arrival)
  shared -> local-rank-major arena TMA
  wait(store has finished reading shared) -> release slot
  wait(global store completion) -> release kv_ready
```

最后一步非常关键。`store_async_read_wait` 只说明 shared slot 可重用；只有 `store_async_wait` 后的 GPU-scope release increment 才能把“local HBM 已可由 compute CTA 读取”发布给 acquire consumer。该 protocol 是通信计算重叠正确性的 happens-before 基础。

对于 causal half-KV task，通信 decoder 在 compact half-row space 选择 logical row，并通过 `half_cu_seqlens` 的二分反查恢复 full local K/V arena 的 row。这避免维护额外的 half-arena，同时保持 communication logical task 与 scheduler 的 causal 分区一致。

### 3.6 Causal zigzag 与 per-Q-tile state machine

对 \(G>1\) 的 local shard，causal 路径将 Q/KV 划分为等长 front/back half，要求 local half 为 128-row 对齐。给定 subring-local rank \(r_G\)：

```text
step 0:
  full local Q x full local KV, with local diagonal causal mask

step 1 .. r_G:
  full local Q x remote front-half KV, no diagonal mask

step r_G+1 .. G-1:
  local back-half Q x remote full KV, no diagonal mask
```

该 zigzag 规则覆盖所有全局 causal pairs，同时将每个 rank 的有效 attention area 尽量均衡。front Q tile 的终止 step 是 \(r_G\)，当 \(r_G=0\) 时只存在 step 0；back Q tile 需要推进至 \(G-1\)。

每个 G8/G4/G2 的 full Q tile 对应一个 `tile_state`：低位为下一个尚未处理 step，最高位 `kTileStateBusy` 为临时 claim lock。其状态转移为

```text
0
  -- step 0 store O/LSE --> 1
  -- CAS claim [begin,end] --> BUSY
  -- one FA3 mainloop + one online merge --> end + 1
  -- terminal segment --> completed_tiles += 1
```

tile state 使用 GPU-scope release store；scheduler 使用 acquire load 与 acquire CAS。故观察到 `state=end+1` 的 CTA 也会观察到前一 segment 对 O/LSE 的 store。一个 Q tile 同时最多只有一个 owner，独立 Q tiles 则可以完全并行。

### 3.7 Readiness-driven causal segment fusion

这是 forward 中最重要的 scheduler 优化。step 0 的所有 local work 先占据 base ticket；并不存在全局 step-0 barrier。某个 tile 的 remote work 只有在它自己的 step 0 epilogue 已将 state 发布为正数后才可能被 claim。

当 scheduler 读到 `next_step=b` 时，它依次检查 \(b,b+1,\ldots,last\_step\) 的 `kv_ready` count，在第一个未 ready step 停止，得到最长连续 ready span \([b,e]\)。只有成功执行

```text
CAS(tile_state, state, state | BUSY)
```

后，CTA 才获得该 range。begin/end/terminal 被压入一个 32-bit `segment_meta`：4 bit begin、4 bit end、1 bit terminal；最大 ring size 为 8，因而足够编码合法 step 与 invalid sentinel。

FA3 mainloop 把 span 看作虚拟连续 K/V 序列，按 O(1) 算术恢复

```text
(ring_step, source_rank, source-local n_block)
```

而不 materialize 拼接 K/V。一个 segment 内先遍历适用的 half-KV step，再遍历 full-KV step；最终只执行一次 prologue/mainloop/epilogue 与一次 O/LSE merge。

若 \(s\) 个远端 step 被拆成 \(c\) 个 segments，\(1\le c\le s\)，则该机制不减少注意力矩阵乘的数学 FLOPs，也不减少每个唯一 KV block 的读取；它减少的是 scheduler claim、Q 重载、intermediate O/LSE global-memory merge 等 per-step 固定开销。通信足够领先时 \(c\) 变小；仅当前 step ready 时自然退化为单 step，保持数值语义。

Noncausal 路径不启用该机制。它采用 `BlockN=176` 和 exact-step replay：全 Q 对全 KV，逻辑 work 为 G8 的 7 个 step、G4 的 3 个 step、G2 的 1 个 step。因此论文不能把 causal segment fusion 泛化为 noncausal 已实现能力。

### 3.8 在线 O/LSE 合并与空 tile 处理

step 0 初始化每个 Q tile 的 running O/LSE，后续 segment 使用 online softmax 合并。令已有状态为 \((O_p,L_p)\)，当前 segment 为 \((O_b,L_b)\)，则

\[
L=\log(\exp L_p+\exp L_b),
\]

\[
\alpha=\frac{\exp L_b}{\exp L_p+\exp L_b},\qquad
O=O_p+\alpha(O_b-O_p).
\]

这不是局部输出的简单相加。row owner 在 FP32 LSE 上计算 `logaddexp` 和 scale，epilogue 通过 shared memory 将 per-row scale 交给持有 coalesced O fragments 的线程。

有两个实现细节应如实保留：

1. later remote step 可能因 causal mask 没有有效 K/V；它不能覆盖已有 running O/LSE，只有 step 0 能写 neutral `(0,-inf)`；
2. host binding 每次 forward 仍执行 `out.zero_()` 与 `lse.fill_(-inf)`，包括 caller-provided buffer。step 0 不需要读取这份预清零 state，但这两个初始化属于实际 op 的外围开销，端到端 benchmark 不能忽略或误称为 kernel 内零成本。

### 3.9 Forward launch 的实现级优化与边界

当前 forward runtime 在以下 specialization 间显式 dispatch：

```text
world size: 2 / 4 / 8
mode: causal / noncausal
stats: normal / instrumented
```

`NumDevices` 是 TK PGL 的编译期模板参数，所以 world size 不能在一个通用 runtime kernel 内任意变化。所有 TMA 相关指针都进行 64-byte alignment diagnostics；`q/k/v/o`、remote K/V data 以及 descriptor layout 也存在对应的对齐要求。

若 prepared varlen metadata 启用了 PDL，launch 走 `cudaLaunchKernelEx` 的 programmatic stream serialization path；否则普通 `<<<grid, block, smem, stream>>>` launch。PDL 是 inherited FA3/CUDA launch capability 的接入，不应在没有数据的情况下作为 MegaRing 独立性能贡献。

empty rank 仍参与 IPC setup 和 persistent launch。binding 为 TMA descriptor 构造一行 dummy Q/O/LSE backing，但 scheduler 的 local lengths 全为零，不产生逻辑 attention access。这避免 zero-extent descriptor 的特殊分支。

### 3.10 Backward：fused core，而非整个 backward 只有一次 launch

Backward 沿用 FA3 的多阶段形式：

```text
1. preprocess: O, dO, LSE -> dPsum / LSE-log2 / accum preparation
2. backward core megakernel:
   compute CTAs: attention backward -> dQ accum + step-local dK/dV
   comm CTAs: K/V ingress -> remote FP32 dK/dV reduce-add
3. owner completion wait kernel
4. postprocess: FP32 accumulators -> compact BF16 dQ/dK/dV
```

Backward scheduler 的 ticket 是精确 `(level, ring_step, KV tile, Q head, batch)`，以 KV N-block 为外层，保留 FA3 backward 的 mainloop 结构。它没有 forward 式 multi-segment claim。

不同 KV tiles 和 remote steps 对相同 Q 都有 dQ contribution，因而 dQ 累加到 local FP32 `dq_accum`，最后才转 BF16。当前 path 为 non-deterministic specialization：跨 CTA floating-point add 顺序不固定，deterministic backward 不在支持范围。

对于 dKV，core kernel 使用 step-local FP32 buffer：

```text
dk_steps, dv_steps: [world_size, step_stride]
step_stride = KVH * padded_rank_capacity * 128 floats
```

每个 batch 后额外插入 128 zero padding rows，使所有 dKV egress decode 均按 128-token block 对齐。不同 hierarchy level 复用 step dimension，但 batch ranges 不重叠。Backward 有 15 个 local dKV readiness section：G8 step 0..7、G4 step 0..3、G2 step 0..1、G1 step 0。

communication CTA 等待一个 section 的所有 local dKV tile ready 后，将一个 KV-head 的 128-token FP32 block，即 `16 x 1024` FP32 tile，TMA reduce-add 到 `owner(G,r,step)` 的 IPC accumulator。最后一个 CTA 对目标 owner completion scalar 执行 system-scope release increment；owner-side wait kernel 使用 system-scope acquire，达到期望值后才允许 dKV postprocess。

该 owner-directed protocol 避免了将所有 step-local dKV materialize 到 Python 或单独 collective 后再归约。其当前限制是 dKV egress 固定按 G8 -> G4 -> G2 -> G1 的 section 顺序；前序 section 未 ready 时会造成 head-of-line blocking，即使后续 section 已 ready。论文应将其称为清晰的当前设计权衡，而非 readiness-optimal egress scheduler。

## 4. 贡献二：Buddy-Ring Pareto Beam Scheduler（BR-PBS）

### 4.1 规划器的角色

BR-PBS 是针对 \(W\in\{2,4,8\}\) 的小规模在线 planner。它不求解通用 CP placement，也不建模精确 kernel latency、NVLink congestion、SM allocation 或显存容量。它的任务是：在 kernel 支持的 aligned buddy tree 中，为每条 sequence 选择一个 `(ring_size, ring_start)`，在明确负载容差内尽量保护短序列，并输出 kernel 可以消费的重排 metadata。

对 \(W=8\)，合法位置恰好构成 buddy tree：

```text
G8: [0..7]
G4: [0..3] [4..7]
G2: [0,1] [2,3] [4,5] [6,7]
G1: [0] [1] [2] [3] [4] [5] [6] [7]
```

一般最多有 \(2W-1\) 个 concrete ring positions，因此 W=8 时最多 15 个候选位置。这是 planner 能采用 beam search 而不是引入大规模整数优化器的关键。

### 4.2 两阶段合法性和 sampler 对齐

`eligible_ring_sizes(L,W,is_causal)` 的硬 candidate rule 为：

\[
\text{causal: } G=1\ \text{or}\ L\bmod(256G)=0,
\]

\[
\text{noncausal: } L\bmod G=0.
\]

Causal 的更强约束确保每个 local shard 至少 256-aligned，故其 half shard 至少 128-aligned，匹配 zigzag kernel。Noncausal 的规划器规则则故意较弱，只表达 topology 可切分性。

`balancer/sampler.py` 从 256-token bucket 统计中按确定性 RNG stream 采样，并按长度分层向上对齐：

| sampled length range | alignment |
| --- | ---: |
| `<2K` | 256 |
| `[2K,4K)` | 512 |
| `[4K,8K)` | 1024 |
| `[8K,16K)` | 2048 |
| `>=16K` | 2048 |

这提高了 sampled workload 进入 kernel 的概率，但**不替代** forward binding 的 per-local-shard 128-row 检查。特别是 noncausal direct planner call 或某些 G4/G8 assignment 仍可能满足 `L%G==0` 却不满足 `(L/G)%128==0`。`ring_test` 的 mega-hybrid compatibility gate 与 C++ binding 负责拦截这类情况。

因此论文可将当前接口准确描述为：

```text
planner hard legality (topology/alignment proxy)
        +
kernel launch validation (local shard/arena/TMA legality)
```

而不是“负载均衡器单独保证所有 kernel ABI”。

### 4.3 双负载模型与通信 proxy

对于长度 \(L_i\)，BR-PBS 使用：

\[
A_i=\begin{cases}
L_i(L_i+1)/2,& \text{causal},\\
L_i^2,& \text{noncausal}.
\end{cases}
\]

将 sequence 放到 size \(G\) ring 后，每个 member rank 获得

\[
t_{i,G}=L_i/G,\qquad c_{i,G}=A_i/G.
\]

目标均值为 \(\bar T=\sum_iL_i/W\)、\(\bar C=\sum_iA_i/W\)。二次 compute 维度避免了“相同 token 数的一个长序列和多个短序列等价”的错误假设。

通信 proxy 为

\[
q_{i,G}=\begin{cases}
0,&G=1,\\
L_i(G-1)/G,&G>1.
\end{cases}
\]

最终 objective 的 `communication_cost` 为 \(Q=\sum_iq_{i,G_i}\)；与此同时 `rank_communication` 对 ring 的每个成员累加 \(q_{i,G}\)，故

\[
\text{communication amplification}
=\frac{\sum_r rank\_communication_r}{\sum_iL_i}
=\frac{\sum_i(G_i-1)L_i}{\sum_iL_i}.
\]

这些都是 token-hop proxy，而非字节、拥塞或 latency 模型。它们只能在高优先级目标相同时做合理的 topology tie-break，不能替代端到端通信测量。

### 4.4 可行性、短序列保护与最终 objective

定义最大绝对相对偏差：

\[
D_T=\max_r|T_r/\bar T-1|,\qquad
D_C=\max_r|C_r/\bar C-1|.
\]

给定 token tolerance \(\epsilon_T\) 和 compute tolerance \(\epsilon_C\)，违反量为

\[
V=\max(0,D_C-\epsilon_C,D_T-\epsilon_T).
\]

`V <= 1e-12` 才是 `feasible=True`。默认 tolerance 是 compute 5%、token 10%。

短序列按 `<=2K`、`2K-4K`、`4K-8K`、`8K-16K`、`>16K` 五个右闭 bucket 记录：

\[
N_b=\#\{i\in b:G_i>1\},\qquad
P_b=\sum_{i\in b,G_i>1}\log_2G_i.
\]

最终比较键严格按字典序：

\[
J=\operatorname{lex}(V,N_0,P_0,N_1,P_1,\ldots,N_4,P_4,Q,H,D_C,D_T),
\]

其中 \(H\) 是已使用的 `(ring_size, ring_start)` 数。它表达的优先级是：

```text
满足容差
  > 保护更短 bucket
  > 降低通信 token-hop proxy
  > 减少 active buddy locations
  > 继续改善 compute/token deviation
```

`H` 只表示 topology fragmentation，不是实际 CUDA kernel launch 数；hybrid megakernel 可在一个 core launch 中处理多个 active rings。

### 4.5 Job 收缩与 core/filler 分解

每个 job 计算结构强度：

\[
\kappa_i=\max(L_i/\bar T,A_i/\bar C).
\]

并计算最小必要 ring size：

\[
G_i^{min}=\min\left\{G:\frac{L_i}{G}\le(1+\epsilon_T)\bar T,
\frac{A_i}{G}\le(1+\epsilon_C)\bar C\right\}.
\]

若没有 legal size 满足该上界，取最大 hard-legal size。通常 level 中 structural job 只允许 `G >= G_min`，这是不可逆 overload 的有效剪枝；最后的 `all hard-legal candidates` level 会重新打开所有 hard-legal size，避免该剪枝永久排除低 violation 方案。

满足下列任一条件者进入 structural beam：

```text
kappa >= structure_threshold (default 0.5)
minimum_ring_size > 1
current progressive level has unlocked its bucket/size
final all-hard-legal level
```

其余 job 是 filler，初始只允许 G1。两类 job 均按照

```text
descending kappa, descending compute, descending length, ascending original_index
```

处理。该分解体现“长序列先决定结构、短序列补 residual capacity”的设计选择；它不是全局最优证明。

### 4.6 Pareto beam：状态、前缀指标与状态压缩

`_BeamState` 保存很小的 rank load tuple、每 bucket split statistics、communication cost、active ring set，以及 parent chain。parent chain 避免在每次扩展复制整个 placement history。

部分 assignment 不能直接用最终 deviation 判断质量：未放置 workload 可以填补低 load rank，已经超上界的 rank 却不可逆。因此 prefix metric 为

\[
O_T=\max\left(0,\frac{\max_rT_r-(1+\epsilon_T)\bar T}{\bar T}\right),
\]

\[
O_C=\max\left(0,\frac{\max_rC_r-(1+\epsilon_C)\bar C}{\bar C}\right),
\]

以及 token/compute spread、split key、communication 和 active-ring count。一个状态在这些维度上全部不差且至少一维严格更好时，支配另一个状态。

在 Pareto 过滤后，按最终平均负载的 2% 做 rank-order-preserving quantization：

\[
\Gamma_T(r)=\left\lfloor T_r/(0.02\bar T)\right\rfloor,
\qquad
\Gamma_C(r)=\left\lfloor C_r/(0.02\bar C)\right\rfloor.
\]

同一 signature 仅保留 split、communication、active ring 与 prefix metric 更优者。此处**保留 rank 顺序**，不能对 load vector 排序；因为 buddy topology 的物理位置和后续 candidate 都依赖 rank identity。

若 frontier 仍大于 `beam_width=64`，代码先固定若干 anchor：最低 overload、最低 token spread、最低 compute spread、最低通信、最少短序列切分、最少 active rings；再使用多维 crowding distance 保留稀疏区域。这防止 beam 只保留某一种 residual-capacity 形状。

### 4.7 Residual fill、渐进解锁与 hierarchical repair

每个 beam terminal 独立对 G1 filler 执行 greedy fill。候选 rank 的键为：

```text
(projected upper-bound violation,
 smooth-max token/compute potential,
 rank index)
```

其中 smooth max 采用 \(\lambda=8\)：

\[
\operatorname{smax}_8(z)=\frac{1}{8}\log\sum_r\exp(8z_r).
\]

这比只查看当前最大 rank 更平滑地考虑所有 rank 的双负载，并以 rank index 作为确定性 tie-break。filler 阶段只使用 G1，不会意外破坏短序列保护。

候选空间不是一开始全部打开。progressive relaxation 从初始保守空间开始，然后按**长 bucket 到短 bucket、每桶 G2 到 G4 到 G8**逐步解锁，最后才进入 all-hard-legal level。每个 level 运行 beam、fill 和 repair；一旦该 level 找到 feasible plan 即停止。这个外层停止规则比最终 \(J\) 更强地保护短序列：不会为了更低通信而无谓打开更短 bucket 的 split option。

局部 repair 只围绕 token/compute 最重与最轻的各两个 rank，并取相关 rank 上按 \(\kappa\) 排名前四的 jobs。其受限邻域包括：

```text
G1 move / G1 swap
same-size buddy relocation
parent promotion
child demotion
two-job sibling demotion
```

每轮只接受严格减小 \(J\) 的邻居，最大 32 轮；相同 objective 以按原始 index 排列的 `(ring_size, ring_start)` 作确定性 tie-break。有限 beam、2% quantization、greedy fill 与受限邻域意味着 BR-PBS 是可复现启发式，不是 MILP 级全局最优或完备可行性判定器。

### 4.8 输出与复杂度

最终 placements 按 kernel ABI 重排，输出：

```text
global_lengths
ring_sizes
ring_starts
rank_tokens / rank_compute / rank_communication
feasible / load_violation / relaxation level
split diagnostics / communication proxy / active_ring_count / repair_moves
```

当前 `HybridWorkload` 不公开从规划顺序回到 input order 的 permutation；benchmark 直接在规划后顺序构造 packed input。任何通用训练集成若要求恢复样本顺序，都必须在 planner 外维护该 permutation。

令 \(R\le8\)、最大 concrete ring positions \(G\le2R-1\)、beam width 为 \(B\)、structural/filler jobs 数为 \(N_H,N_F\)，忽略 frontier 比较时的主要工作量为

\[
O(BN_HGR)+O(BN_FR).
\]

当前 direct Pareto frontier 的最坏单轮代价可达 \(O((BG)^2d)\)，但 `R<=8`、`G<=15`、`B=64` 是该设计成立的重要实际边界。

## 5. Planner 与 Kernel 的协同契约

下面的映射建议作为论文中连接两个贡献的一张关键表。

| Planner / host 决策 | Kernel 中的消费方式 | 作用 |
| --- | --- | --- |
| aligned buddy `(G,S)` | `owner(G,r,s)` 算术恢复 source rank | 不需要 device-side topology lookup |
| ring-size 非递增 batch order | G8/G4/G2/G1 fixed descriptor 的连续 batch/row range | 消除通用 group metadata 与复杂索引 |
| causal `L mod 256G=0` | local half 128-row aligned，支持 zigzag 与 16-row transfer | 使 causal logical tile 和 TMA subtile 同时对齐 |
| rank-local packed `cu_seqlens` | varlen prepared scheduler metadata 与 compact Q/O | 允许 empty membership 表示为零长度 slot |
| rank-major K/V capacity | `source_rank * capacity + compact_row` | peer ingress 一次，local HBM 多次复用 |
| token / compute balance | 降低 max-rank critical-path 风险 | 仍需 runtime execution counter 和 latency 验证 |
| short-sequence protection | 避免无谓的小 ring communication / topology fragmentation | 是 policy 优先级，不是数学性能保证 |

从系统方法角度，最适合的叙述是：**BR-PBS 将 heterogeneous sequences 映射到一个 hardware-aware but deliberately compact buddy topology；MegaRing 将该 topology 编译为固定层级 descriptor，并以 data-ready persistent execution 消除多 ring 的全局串行化。**

## 6. 新发现：实现已具备、但应补入论文主线的细节

下表列出核查中发现的、未被旧设计主线充分串联的实现事实。它们不一定都是“算法创新”，但会直接影响论文的设计完整性、实验可解释性或表述边界。

| 细节 | 当前实现事实 | 论文中的正确写法 |
| --- | --- | --- |
| Runtime work counters | `stats` 非空时 dispatch 到单独编译的 stats kernel variant，统计 `[qo_visits, kv_tile_reads]` | 将其作为 validation/diagnostic instrumentation，而不是计时路径的一部分 |
| Counter 精确定义 | 有效 attention mainloop work 每次 `mma()` 增加一次 Q/O visit；KV read 加最终 `n_block_max-n_block_min`；16-row ingress TMA 不计入 | 用于衡量实际 attention tile work 与 segment merge，不可解释成总通信字节或总 TMA 指令数 |
| Low-overhead aggregation | 每个 persistent CTA 先在寄存器累积，结束时仅由固定 consumer thread 对全局 stats 做最多两次 atomic add | 可说明观测对 kernel hot loop 的扰动被限制，但不能声称完全零开销 |
| Causal segment 内部统计 scaffold | stats specialization 将 `completed_tiles` 扩为 4 个 int：index 0 仍是终止 tile count；`+1/+2/+3` 记累计 segment span、最大 span、segment count | 当前 public binding 未返回后三项；它们是可进一步暴露的内部测量点，不能报告成现有用户可见 metric |
| Static-to-runtime bridge | `forward_load_model.py` 给 causal mega-ring 计算 Q/O visit best/worst bounds；runtime stats 给出实际 dynamic scheduler 结果 | 应用作“planner proxy 与执行行为一致性”验证；KV read 理论上稳定，Q/O visit 会受 ready segment 合并影响 |
| 统计 probe 与 latency 隔离 | benchmark 在计时后额外运行一次 probe，随后 sync、all-gather 和 all-reduce；这些不计入 latency/TFLOPS | 所有论文图中需说明是否启用 stats，以及 probe 不进入计时 |
| Half-row address recovery | causal ingress 从 compact half row 经 `half_cu_seqlens` 二分映射回 full K/V row | 这是避免 half arena / lookup table 的实际地址优化，可写入 implementation detail |
| Dynamic shared memory reuse | launch 使用 `max(FA3 smem, comm smem)`；comm CTA 同一 block 完成 copy 后转 compute | 这是角色转换成立的资源条件，不是两个独立 kernels 的简单拼接 |
| Host-side op overhead | output/LSE reset、counter allocation/reset、scheduler metadata preparation、可能的 PDL launch 都在 core kernel 外 | 报告 end-to-end op time 时必须纳入，或明确 core-only 边界 |
| Two-stage noncausal legality | planner 的 noncausal rule 只有 `L%G==0`，binding 还检查 local length 128 alignment | 不应过度声称 planner 一步得到可 launch plan；应在系统图中标出 launchability gate |
| Planner determinism | stable sorting、rank tie-break、placement tie-break、连续 RNG stream 固化结果 | 可用于可复现实验设置；不能等同于全局最优 |

## 7. 运行时观测与负载模型闭环

### 7.1 为什么需要两种负载量

BR-PBS 的 \(A_i/G\) 是低成本解析 proxy，足以在 planner 中搜索；它没有表达：

```text
causal zigzag 的 tile 级边界
head count 与 BlockM/BlockN 离散化
dynamic ready timing
segment fusion 长度
empty / masked tile
communication CTA 与 compute CTA 资源竞争
```

因此项目中同时存在两套互补量：

| 层级 | 指标 | 用途 |
| --- | --- | --- |
| Planner | `rank_tokens`, `rank_compute`, token-hop proxy | 低成本 placement 搜索与可行性控制 |
| Static load model | effective/physical tokens、scores、FLOPs、bytes、`kv_tile_reads`、Q/O best-worst bounds | 运行前的可解释分析与 baseline 对齐 |
| Device runtime | `qo_visits`, `kv_tile_reads` | 验证真实 scheduler work shape 与 segment fusion |
| End-to-end benchmark | max-rank latency、aggregate TFLOP/s、profile counters | 性能结论 |

对 causal MegaRing，static model 的 worst Q/O visits 近似逐 step execution；best visits 近似将 local、front-half、back-half 可合并部分各以最少 segment 消费。真实 `qo_visits` 应在该离散范围内变化，取决于 `kv_ready` 领先距离和 dynamic claim 时机。`kv_tile_reads` 对同一有效 attention region 则不因 segment merge 改变。

### 7.2 推荐的论文验证链

```text
BR-PBS rank_compute / rank_tokens
        |
        v
static physical score and tile accounting
        |
        v
runtime [qo_visits, kv_tile_reads]
        |
        v
per-rank latency / max-rank latency / profiler stalls
```

建议在论文中报告 planner estimate 与 runtime execution metric 的相关性，而不只报告最终平均 latency。若某 planner placement 的 token/compute balance 很好，但 `qo_visits`、`kv_tile_reads/QO` 或 max-rank latency 仍显著偏斜，应将其解释为 cost model 未包含的 tile/overlap 效应，而不是掩盖该差异。

## 8. 正确性不变量与失败边界

### 8.1 Forward happens-before

```text
peer TMA load
  -> local TMA store completion
  -> GPU-scope release kv_ready
  -> acquire ready load / wait
  -> compute CTA local-HBM TMA read

segment O/LSE store or merge
  -> GPU-scope release tile_state=end+1
  -> acquire/CAS claim for next segment
  -> next segment reads running O/LSE
```

对应不变量：

1. 同一 Q tile 同时最多有一个 remote segment owner；
2. `tile_state` 只推进连续 prefix，不能跳过未 ready step；
3. remote K/V readiness 只能在 local global store 真正完成后发布；
4. step 0 是 O/LSE 唯一初始化者，later empty step 不得覆盖 state；
5. compute CTA 只能读取其 ready counter 已覆盖的 remote K/V section。

### 8.2 Backward happens-before

```text
compute epilogue writes step-local FP32 dKV
  -> GPU-scope release local_ready
  -> comm CTA acquire wait
  -> peer FP32 TMA reduce-add
  -> system-scope release owner completion
  -> owner system-scope acquire wait
  -> owner dKV postprocess
```

对应不变量：

1. dKV section 的 remote egress 必须等到该 section 所有 local compute tiles 完成；
2. owner postprocess 只能在所有 peer contribution 到齐后访问 accumulator；
3. 每轮 backward 前调用方必须清零 remote FP32 accumulator 与 completion scalar，并完成必要 CUDA 同步/distributed barrier；
4. 所有 rank 必须使用一致的 global sequence、ring size、ring start metadata。

### 8.3 重要限制

- 不支持跨节点。当前依赖单节点 IPC/TMA，不能将其与 NCCL 的通用跨节点能力混为一谈。
- 不支持任意 CP group；只支持 contiguous aligned buddy groups。
- Forward G1-only / world-size-1 应走普通 local varlen path；forward mega-ring entry 的 world size 为 2/4/8。
- Backward 不保证 determinism。
- 规划器不建模 memory capacity、head shape、SM split 与 physical topology；binding 才验证某些 ABI 约束。
- finite beam 未找到 feasible plan 不等价于该 relaxation level 在数学上无可行解。

## 9. 论文中的推荐组织与可主张贡献

### 9.1 建议章节结构

```text
3. Background and Motivation
   - heterogeneous sequence lengths make fixed CP degree inefficient
   - conventional ring attention causes repeated step launches and reductions

4. MegaRing Overview
   - buddy hierarchy, planner-to-kernel contract, rank-major arena

5. Hierarchical Hybrid Mega Kernel
   - persistent CTA roles and K/V ingress
   - causal zigzag and readiness-driven segment fusion
   - online O/LSE reduction and role conversion
   - backward owner-directed dKV reduction

6. Buddy-Ring Pareto Beam Scheduler
   - topology, dual load model, lexicographic policy
   - Pareto beam, progressive relaxation, residual fill, repair

7. Runtime Observability and Cost-Model Validation
   - static model, runtime Q/O-KV counters, profiling methodology

8. Evaluation
   - correctness, end-to-end latency, balance, ablation, limitations
```

### 9.2 可在实验完成后使用的贡献表述

1. **Hierarchical hybrid execution for heterogeneous CP.** 在一个 physical world 中，以固定 buddy hierarchy 表示并执行同时存在、相互重叠的 G8/G4/G2/G1 sequence-level CP groups。
2. **In-kernel communication/compute orchestration.** 在同一 persistent grid 内用 communication CTA 做 peer K/V ingress，用 tile readiness 驱动 compute，并在 forward 回收完成 ingress 的 CTA 作为 compute worker。
3. **Readiness-driven causal segment fusion.** 以 per-Q-tile 单调状态与 CAS claim 合并连续 ready ring steps，在不拼接 K/V 的情况下复用一次 FA3 mainloop 和 online softmax reduction。
4. **Fused owner-directed backward gradient reduction.** 将 step-local FP32 dKV 通过 peer TMA reduce-add 直接归约至 K/V owner，并以 system-scope completion 协议保障后处理顺序。
5. **Topology-aware load planner.** BR-PBS 在明确 token/compute tolerances 下，以 Pareto beam 和短序列优先的字典序目标选择 buddy placement；其输出直接满足 hierarchical execution 所需的排序与拓扑契约。
6. **Execution-aware observability.** 将解析 load model 与 device-side actual Q/O/KV tile counter 对接，显示 planner 的静态平衡与动态 segment execution 之间的关系。

### 9.3 必须避免的过度表述

- 不要说“整个 backward 只有一个 kernel”；准确说法是 backward **core** megakernel 融合 attention、ingress 和 dKV egress，仍有 preprocess/wait/postprocess。
- 不要说 causal segment fusion 也用于 noncausal；当前 noncausal 是 exact-step replay。
- 不要说角色转换迁移 CTA 到另一个物理 SM；它是同一 persistent CTA 的 phase reuse。
- 不要说通信完全隐藏；实际时间仍受 SM split、ready distance、HBM copy、NVLink、dKV egress 和 tail 影响。
- 不要将 `communication_cost` 或 `rank_compute` proxy 当作 latency 测量。
- 不要声称 BR-PBS 为全局最优，或 noncausal planner 单独保证所有 kernel launch shape。
- 不要将继承的 FA3 WGMMA/TMA pipeline/cache hint 记为 MegaRing 独有贡献。

## 10. 必做实验与报告口径

### 10.1 正确性

至少覆盖：

1. G8/G4/G2/G1 同时存在且 subgroup 重叠的 8-GPU case；
2. 所有合法 G4/G2 buddy start；
3. all-CP、G1-only、empty rank、连续多轮调用；
4. causal/noncausal forward，causal backward；
5. MHA 与 GQA；
6. O、LSE、dQ、dK、dV 对逻辑 reference 的一致性；
7. K/V remote copy sentinel 与 capacity/padding 检查；
8. `compute-sanitizer` racecheck/synccheck，特别是 TMA slot、named barrier 与 release/acquire counter。

### 10.2 消融

| 消融 | 对照 | 回答的问题 |
| --- | --- | --- |
| Core megakernel | Python/NCCL ring 或逐 step launch | 融合与 host launch/reduction 开销的贡献 |
| Forward role conversion | ingress 完成后 comm CTA 退出 | 是否缩短 compute tail |
| Causal segment fusion | 强制 single-step claim | 是否减少 Q/O visits、O/LSE merge 和 scheduler 开销 |
| Logical tile copy | row-granular ingress | attention-aligned readiness 的收益 |
| Physical TMA subtile | 8-row，资源允许时 32-row | 16-row slot/指令数/shared-memory 权衡 |
| Communication ordering | G8-first、small-first、round-robin | critical-path-first 假设是否成立 |
| BR-PBS | all-CP、G1-only、threshold baseline、无 repair 等 | planner 与 kernel co-design 的净收益 |
| SM split | 多组 `num_comp_sm:num_comm_sm` | workload-dependent overlap 与 fusion 的耦合 |
| Planner quality | 不同 beam/finalist/threshold | 规划开销、feasibility、short split、latency 的关系 |

### 10.3 测量口径

每张性能图都应写明：

```text
是否包含 scheduler metadata preparation
是否包含 out/LSE reset、counter reset、buffer preparation
是否包含 distributed barrier / IPC setup
是否为 core kernel、single op、或完整训练 iteration
统计 probe 是否开启，以及 probe 是否在计时外
报告 average-rank 还是 max-rank latency
```

推荐同时给出：end-to-end max-rank latency、aggregate/per-rank TFLOP/s、NVLink 与 HBM throughput、ready wait stall、runtime `[qo_visits,kv_tile_reads]`、segment span 分布（若后续暴露内部 counters）、planner `rank_tokens/rank_compute` 与最终 rank time 的相关性。

## 11. 主要实现映射

| 机制 | 主实现 |
| --- | --- |
| Forward Python/CUDA binding、host validation、hierarchy construction、stats reset | `csrc/mega_ring_min_fa3_varlen_ring_bindings.cu` |
| Forward fused grid、TMA ingress、role conversion、shared-memory launch policy | `include/mega_ring_min_fa3_varlen_ring_launch.h` |
| Causal readiness scan、CAS claim、fixed-level tile decode | `include/mega_ring_min_fa3_varlen_scheduler.h` |
| FA3 persistent wrapper、segment publish、CTA-local runtime stats aggregation | `include/min_fa3_kernel.h` |
| Virtual N-block mapping、ready wait、remote KV local-HBM TMA | `include/min_fa3_mainloop.h` |
| Online O/LSE merge | `include/min_fa3_epilogue.h` |
| GPU/system scope semaphore primitives | `include/mega_ring_semaphore.cuh` |
| Shared G8/G4/G2/G1 descriptor | `include/min_fa3_mega_ring_hierarchy.h` |
| Backward fused wrapper and dKV egress | `include/backward/min_fa3_bwd_launch.h` |
| Backward scheduler/mainloop/epilogue | `include/backward/min_fa3_bwd_scheduler.h`、`include/backward/min_fa3_bwd_mainloop.h`、`include/backward/min_fa3_bwd_epilogue.h` |
| BR-PBS core | `balancer/load_balancer.py` |
| Dataset alignment and deterministic sampling | `balancer/sampler.py` |
| Planner correctness tests | `balancer/test_balancer.py` |
| Static forward load model | `ring_test/forward_load_model.py` |
| Runtime stats collection | `ring_test/benchmark_topology_forward.py` |

## 12. 结论

MegaRing 的论文主线应围绕一个清楚的系统闭环：**BR-PBS 以短序列保护和双负载平衡为约束，将异构 sequence 映射到有限 buddy hierarchy；MegaRing megakernel 则以 rank-major arena、tile readiness、persistent scheduling 和 online reduction，将该 hierarchy 转化为可重叠执行的单节点 Hopper CP 路径。**

最值得强调的技术张力是：planner 的静态 placement 使 kernel 可以专用化且避免 metadata 开销；kernel 的动态 readiness 和 segment fusion 又避免静态 ring order 造成全局 barrier。runtime tile counters 让这种“静态拓扑决策 + 动态设备执行”的关系第一次可以被直接测量，而不必仅依赖解析 FLOP 或 token proxy。

在没有完整消融与 profiler 数据前，应把上述机制作为设计贡献，把性能收益作为待验证假设；这会使论文的论证更严谨，也更容易清楚地区分算法贡献、kernel engineering 与系统边界。
