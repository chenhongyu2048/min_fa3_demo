# BR-PBS 负载均衡器设计

本文档描述 `balancer/` 中当前已经实现的负载均衡算法，而不是一个与代码脱节的理想化方案。核心实现位于 [load_balancer.py](./load_balancer.py)，数据集长度采样位于 [sampler.py](./sampler.py)，CPU-only 行为测试位于 [test_balancer.py](./test_balancer.py)。

Balancer 输出如何由 forward/backward megakernel 消费，以及层级 ring、SM 角色、
TMA 通信和归约协议，见
[MegaRing Hybrid Megakernel 设计](../docs/MEGARING_HYBRID_KERNEL_DESIGN.md)。

算法名称为 **Buddy-Ring Pareto Beam Scheduler（BR-PBS）**。它解决的问题是：在 2、4 或 8 个 rank 上，为一批不同长度的序列选择合法的 G1/G2/G4/G8 buddy ring，在 token 负载和 attention compute 负载都满足容差的前提下，尽量避免切分短序列，并降低通信和拓扑复杂度。

## 1. 设计目标

### 1.1 主要目标

1. 每个序列只分配到一个 kernel 支持的 buddy ring。
2. 同时平衡每个 rank 的 token 工作量和 attention compute 工作量。
3. 将负载容差作为可行性条件，而不是仅优化一个没有明确含义的加权分数。
4. 优先保护短序列，只有在更保守的候选空间找不到可行解时才逐步允许切分。
5. 在小规模在线规划场景中保持确定性和可接受的 CPU 开销。
6. 产出可直接传给 hierarchical mega-ring benchmark/kernel 的 `global_seqlens`、`ring_sizes` 和 `ring_starts`。
7. 即使不存在满足容差的拓扑，也返回搜索到的最低违反方案，并显式标记为不可行。

### 1.2 非目标

当前 balancer 不试图完成以下工作：

- 不建立精确的 kernel latency、网络拥塞或 SM 利用率模型。
- 不求解全局最优的 MILP/CP-SAT。
- 不支持任意 GPU 数或任意 rank 子集；只支持 `world_size in {2, 4, 8}` 和连续对齐的 buddy ring。
- 不在规划器中建模显存容量、Q/K/V head 数、head dim、通信与计算重叠细节。
- 不保持输入序列的原始输出顺序；输出会按 kernel 要求重新排序。
- 不保证在有限 beam、量化合并和受限局部邻域下找到所有存在的可行解。

这些限制是有意的。当前规划器服务于固定规模的 hierarchical mega-ring demo，首先需要一个行为清楚、可观测、能在线运行的调度器，而不是一个覆盖所有硬件和 kernel 变体的通用优化框架。

## 2. 系统边界

完整的数据流如下：

```text
dataset bucket statistics
          |
          v
deterministic length sampling + ring-aware alignment
          |
          v
assign_hierarchical_rings(lengths, world_size, mode, tolerances, ...)
          |
          +--> build legal buddy-ring candidates
          +--> progressive BR-PBS search
          +--> residual singleton filling
          +--> hierarchical local repair
          |
          v
HybridWorkload
  global_lengths / ring_sizes / ring_starts
  per-rank loads / feasibility / split and communication diagnostics
          |
          v
dataset benchmark frontend -> explicit-topology mega-ring benchmark/kernel
```

`assign_hierarchical_rings()` 也可以直接接收调用方提供的长度；数据集采样不是调度算法的必要组成部分。`make_workload()` 和 `make_workloads()` 只是把采样器与该入口连接起来。

## 3. Buddy ring 拓扑

令 rank 数为：

\[
R \in \{2,4,8\}.
\]

一个合法 ring 由 `(ring_size, ring_start)` 表示，其中：

\[
p=\text{ring\_size}\in\{1,2,4,8\},\qquad p\le R,
\]

\[
s=\text{ring\_start}\in\{0,p,2p,\ldots,R-p\}.
\]

该 ring 覆盖连续 rank：

\[
G(p,s)=\{s,s+1,\ldots,s+p-1\}.
\]

因此 `ring_start % ring_size == 0`，所有 ring 都是 buddy tree 中的节点。对 `R=8`，候选拓扑为：

```text
G8: [0..7]
G4: [0..3] [4..7]
G2: [0,1] [2,3] [4,5] [6,7]
G1: [0] [1] [2] [3] [4] [5] [6] [7]
```

总 ring 数为：

\[
|\mathcal G|=2R-1,
\]

所以 8 ranks 时最多只有 15 个位置候选。固定的树形拓扑显著缩小了搜索空间，也与 kernel 的层次化通信结构一致；代价是不能表达 `[1,2]`、`[0,2,4,6]` 等任意子集。

## 4. 硬合法性约束

对于长度为 \(L_i\) 的序列，`eligible_ring_sizes()` 先确定合法 ring size，再枚举该 size 的所有 buddy 位置。

### 4.1 Causal

G1 总是合法。对于 `p > 1`：

\[
L_i \bmod (256p)=0.
\]

这保证每个 rank 的 local shard 满足 causal hierarchical ring 的分块要求。

### 4.2 Non-causal

当前规划器要求：

\[
L_i \bmod p=0.
\]

数据集采样器会额外执行 256 到 2048 token 的分层向上对齐，因此经 `make_workload(s)` 进入的长度通常比这个最低条件更强。直接调用 `assign_hierarchical_rings()` 时，调用方仍需负责满足下游 kernel 的其他 shape/alignment 约束。

### 4.3 不做隐式 padding

调度器不会在候选生成阶段改变输入长度。如果需要 padding，应在调用 balancer 前完成。`sampler.py` 的职责之一就是为采样 workload 做这件事。

这种选择使成本、输出长度和真实 benchmark metadata 一致，也避免 planner 悄悄增加 token；相应地，直接输入未对齐长度可能只能使用较小的 ring。

## 5. 负载模型

每个 job 保存：

```text
original_index
length
compute
legal_sizes and concrete candidates
length bucket
kappa
minimum_ring_size
```

### 5.1 Token 工作量

序列 \(i\) 放到大小为 \(p\) 的 ring 后，每个成员 rank 增加：

\[
t_{i,p}=\frac{L_i}{p}.
\]

合法性保证这里可以用整数除法。总平均 token 目标为：

\[
\bar T=\frac{\sum_i L_i}{R}.
\]

### 5.2 Attention compute 工作量

`attention_compute()` 使用以下解析 proxy：

\[
A_i=
\begin{cases}
L_i(L_i+1)/2, & \text{causal},\\
L_i^2, & \text{non-causal}.
\end{cases}
\]

序列放入 \(G(p)\) 后，当前模型假设每个成员 rank 增加：

\[
c_{i,p}=\frac{A_i}{p}.
\]

总平均 compute 目标为：

\[
\bar C=\frac{\sum_i A_i}{R}.
\]

使用二次 compute 维度是必要的：仅平衡 token 会低估长序列对 attention critical path 的影响。例如一个 8K 序列与八个 1K 序列 token 相同，但 non-causal attention proxy 相差 8 倍。

### 5.3 序列结构强度

定义：

\[
\kappa_i=\max\left(\frac{L_i}{\bar T},\frac{A_i}{\bar C}\right).
\]

`kappa` 同时用于识别需要进入结构搜索的“大块”序列，以及决定 beam 和 filler 的处理顺序。默认 `structure_threshold=0.5`。

### 5.4 通信 proxy

对 \(G(p)\) 序列，单个成员 rank 的通信 proxy 为：

\[
q_{i,p}=\begin{cases}
0, & p=1,\\
L_i\frac{p-1}{p}, & p>1.
\end{cases}
\]

最终目标中的通信成本为：

\[
Q=\sum_i q_{i,p_i}.
\]

`rank_communication[r]` 则对 ring 中每个成员都累加 \(q_{i,p}\)。因此报告中的：

\[
\text{communication\_amplification}
=\frac{\sum_r \text{rank\_communication}_r}{\sum_i L_i}
=\frac{\sum_i (p_i-1)L_i}{\sum_i L_i}.
\]

这里的单位是 token-hop proxy，不是测量得到的字节数或时间。它适合在其他目标相同的方案间判断“少切分/小 ring 通常通信更少”，但不能预测拥塞、链路拓扑或通信计算重叠后的端到端时间。

## 6. 可行性与最终目标

对完整方案 \(x\)，每个 rank 的负载为 \(T_r\) 和 \(C_r\)。定义最大绝对相对偏差：

\[
D_T=\max_r\left|\frac{T_r}{\bar T}-1\right|,
\qquad
D_C=\max_r\left|\frac{C_r}{\bar C}-1\right|.
\]

给定 `token_balance_tolerance` \(\epsilon_T\) 和 `compute_balance_tolerance` \(\epsilon_C\)，当前代码定义：

\[
V=\max(0,D_C-\epsilon_C,D_T-\epsilon_T).
\]

`V <= 1e-12` 时方案被标记为 `feasible=True`。这里的 \(V\) 是超出容差的相对负载差，不再除以容差本身。例如 compute 偏差 7%、容差 5% 时，违反量为 0.02。

### 6.1 短序列保护

长度桶按右闭区间划分：

```text
0: <=2K
1: 2K-4K
2: 4K-8K
3: 8K-16K
4: >16K
```

对每个桶 \(b\)，记录：

\[
N_b=\#\{i\in b:p_i>1\},
\]

\[
P_b=\sum_{i\in b,p_i>1}\log_2 p_i.
\]

`N_b` 先惩罚“有多少条序列被切分”，`P_b` 再区分切分程度。例如一条 G8 的 penalty 为 3，一条 G2 的 penalty 为 1。

### 6.2 字典序目标

完整方案的实际比较键为：

\[
J(x)=\operatorname{lex}(
V,
N_0,P_0,
N_1,P_1,
\ldots,
N_4,P_4,
Q,
H,
D_C,
D_T
).
\]

其中 \(H\) 是不同 `(ring_size, ring_start)` 的数量，包括使用过的 G1。

该顺序表达了明确的产品选择：

1. 可行性高于其他一切目标。
2. 在同等违反量下，优先不切最短桶；只有短桶指标相同时才比较更长桶。
3. 切分情况相同时，优先降低通信 proxy。
4. 再减少使用过的 ring 种类/位置数量。
5. 最后继续改善 compute 和 token 偏差。

选择字典序而不是单个加权和，是为了避免无法解释的换算关系，例如“切分一条 512 序列等价于多少 token-hop”或“1% compute imbalance 等价于多少通信”。代价是目标顺序是强偏好：只要更高优先级有极小差异，后续目标改善再大也不能反转结果。

`active_ring_count` 是拓扑分散程度的弱 proxy，不是实际 kernel launch 数。当前 hybrid 路径可以在一个 fused persistent launch 中处理多个 ring，因此不能将 \(H\) 直接解释为 launch latency。

## 7. 必要 ring size 与候选收缩

对每个 job，规划器寻找最小的硬合法 size，使该序列单独分摊后不超过两个负载上界：

\[
p_i^{\min}=\min\left\{p:\frac{L_i}{p}\le(1+\epsilon_T)\bar T,
\frac{A_i}{p}\le(1+\epsilon_C)\bar C\right\}.
\]

如果没有合法 size 能满足，则用该序列最大的合法 size 作为 `minimum_ring_size`。

在普通 relaxation level 中，结构 job 只允许：

\[
p\ge p_i^{\min}.
\]

原因是负载只会随着后续 job 增加；如果一个序列自身已经让某 rank 超过最终上界，更小 ring 通常没有修复价值。这个剪枝对在线搜索很有效，但它不是数学上对最终字典序最优的完整证明，尤其当没有可行解、需要比较不同违反模式时。因此最后还有一个 `all hard-legal candidates` level，重新允许所有满足对齐/拓扑硬约束的候选。

## 8. Structural job 与 filler

job 满足以下任一条件时进入 structural set：

```text
kappa >= structure_threshold
minimum_ring_size > 1
该长度桶/size 已在 progressive relaxation 中解锁
已经进入最终 all-hard level
```

其他 job 作为 filler，只允许 G1。两个集合都按以下确定性顺序处理：

```text
descending kappa
descending compute
descending length
ascending original_index
```

这种 core/filler 分解利用了 workload 的常见形态：少数长序列决定结构，大量短序列更适合作为残余容量填充块。如果让所有短序列一开始都进入 15-way beam，搜索宽度会迅速被大量近似等价状态消耗。

相应的权衡是，filler 的 singleton 放置是 greedy 的，不能回溯。规划器通过保留多个 structural beam 终态、对每个终态分别 fill，以及后续 local repair 来降低这个风险。

## 9. Pareto beam search

### 9.1 状态

`_BeamState` 保存：

```text
rank_compute[R]
rank_tokens[R]
split_counts[5]
split_penalties[5]
communication_cost
active_rings
parent + current placement
```

parent chain 避免每次扩展都复制整个 assignment；每次仍会复制只有 2、4 或 8 个元素的 rank load tuple。

### 9.2 前缀评价

由于 beam 中只放置了部分 structural jobs，不能直接用最终平均负载判断下溢。未放置的工作还可以填补低负载 rank，但已经出现的超载不可逆。因此前缀指标为：

\[
O_T=\max\left(0,\frac{\max_rT_r-(1+\epsilon_T)\bar T}{\bar T}\right),
\]

\[
O_C=\max\left(0,\frac{\max_rC_r-(1+\epsilon_C)\bar C}{\bar C}\right),
\]

\[
S_T=\frac{\max_rT_r-\min_rT_r}{\bar T},
\qquad
S_C=\frac{\max_rC_r-\min_rC_r}{\bar C}.
\]

实际前缀向量为：

\[
M(s)=(O_T,O_C,S_T,S_C,N_0,P_0,\ldots,N_4,P_4,Q,H).
\]

这里使用 spread 而不是“相对当前前缀平均值的绝对偏差”，避免前缀总工作量很小时产生不稳定比例。

### 9.3 Pareto dominance

如果状态 \(s_1\) 在 \(M\) 的所有维度都不差于 \(s_2\)，且至少一维严格更好，则 \(s_1\) 支配 \(s_2\)，后者被删除。完全相等的 metric 也只保留一个状态。

Pareto 过滤保留了 token、compute、切分与通信之间尚未决出的 trade-off。与单一加权 score 相比，它不容易在前缀阶段过早删除“当前通信稍高、但为后续 filler 留出更好形状”的状态。

这仍是启发式剪枝。metric 只保存 `active_rings` 的数量而非集合本身；两个当前 metric 相同但 active ring 集合不同的状态，未来复用 ring 的机会可能不同。删除等价 metric 状态可能损失仅在后续 \(H\) 上体现的方案，但 \(H\) 是较低优先级目标，这个状态压缩是有意接受的。

### 9.4 2% load 量化合并

每个 rank 的 token 和 compute load 分别按最终平均负载的 2% 量化：

\[
\Gamma_T(r)=\left\lfloor\frac{T_r}{0.02\bar T}\right\rfloor,
\qquad
\Gamma_C(r)=\left\lfloor\frac{C_r}{0.02\bar C}\right\rfloor.
\]

签名保留 rank 顺序：

\[
\Gamma(s)=(\Gamma_T(0),\Gamma_C(0),\ldots,\Gamma_T(R-1),\Gamma_C(R-1)).
\]

同一签名中按以下顺序保留一个代表：切分 key、通信、active ring 数、前缀 metric。

2% 是搜索质量与状态数量之间的工程折中。它比默认 5% compute / 10% token 容差更细，但足以合并大量近似状态。由于使用 floor，落在量化边界两侧的非常接近状态仍可能分开，而同一 cell 内有最多接近 2% 的 load 差异会被次要目标覆盖。

### 9.5 Beam 截断与多样性

如果 Pareto 与量化后仍超过 `beam_width`，规划器先保留若干锚点：

- 最小最大上界违反；
- 最小 token spread；
- 最小 compute spread；
- 最低通信；
- 最少短序列切分；
- 最少 active rings。

剩余位置使用类似 NSGA-II 的多维 crowding distance 选择，优先保留 Pareto 前沿中稀疏区域的状态。默认 `beam_width=64`。

这比“只取字典序最前的 64 个状态”更能保留不同形状的 residual capacity，但 crowding distance 对各维只做区间归一化，不代表真实效用距离。

## 10. Residual singleton fill

每个 beam 终态都独立填充 filler。filler 已按 `kappa` 降序排列，相当于二维 longest-processing-time-first。

对一个待放置 filler，枚举所有 G1 rank。放入 rank \(r\) 后先计算仅考虑上界的违反：

\[
U=\max\left(0,
\frac{\max T'}{\bar T}-(1+\epsilon_T),
\frac{\max C'}{\bar C}-(1+\epsilon_C)
\right).
\]

再计算平滑最大值势能：

\[
\operatorname{smax}_{\lambda}(z)=
\frac{1}{\lambda}\log\sum_r e^{\lambda z_r},
\]

\[
\Phi(T',C')=
\operatorname{smax}_{8}(T'/\bar T)+
\operatorname{smax}_{8}(C'/\bar C).
\]

选择键为：

```text
(upper-bound violation, smooth-max potential, rank index)
```

因此优先不制造不可逆超载；上界表现相同后，同时考虑所有 rank 的 token/compute 负载，而不是只看当前最重 rank。rank index 是确定性 tie-breaker。

Residual fill 只处理 singleton，故不会在这个阶段意外切分受保护序列。下溢是否满足要求要等完整方案评价时才能确定。

## 11. Finalist 与局部修复

所有完成 fill 的 beam 方案按完整 \(J\) 排序，只将前 `finalist_count` 个送入局部修复，默认值为 8。

### 11.1 受限邻域

每轮 repair 只关注：

- token 最重和最轻的各 2 个 rank；
- compute 最重和最轻的各 2 个 rank；
- 每个相关 rank 上按 `kappa` 排序的前 4 个 job。

在这些 job 上生成：

1. **G1 move**：从高负载 rank 移到低负载 rank。
2. **G1 swap**：交换高、低负载 rank 上的两个 singleton。
3. **同级 relocation**：ring size 不变，移动到与低负载 rank 相交的 buddy ring。
4. **Promotion**：从当前 ring 提升到其 parent。
5. **Demotion**：从当前 ring 下沉到与低负载 rank 相交的 child。
6. **Sibling demotion**：同一 parent ring 上的两个 job 分别下沉到左右 child，并尝试两种方向。

Sibling demotion 对通信优化尤其重要。例如两个负载接近的 G4 job 可以分别下沉到 `[0,1]` 和 `[2,3]`，大致保持 parent 范围内负载，同时降低通信。

repair 只能使用当前 relaxation level 允许的候选，因此不会绕过短序列保护策略。

### 11.2 接受规则

每轮枚举邻域后选择 \(J\) 严格变小的最佳方案；相同 objective 使用按原始序列索引排列的 `(ring_size, ring_start)` 作为确定性 tie-breaker。最多执行 `max_repair_iterations=32` 轮。

因为只接受严格 objective 改进且分配空间有限，repair 必然终止。受限邻域显著降低开销，但结果仅是该邻域下的局部最优；未处于极端 rank 上的 job 不会被完整枚举。

## 12. 渐进式 relaxation

BR-PBS 不一开始允许所有短序列使用所有 ring，而是从最保守的候选空间逐步解锁。

初始 level：

```text
structural jobs: ring_size >= minimum_ring_size
fillers:          G1 only
```

之后只对 batch 中实际存在的长度桶，按“长桶优先、小 ring 优先”的顺序解锁：

```text
>16K:   G2 -> G4 -> G8
8K-16K: G2 -> G4 -> G8
4K-8K:  G2 -> G4 -> G8
2K-4K:  G2 -> G4 -> G8
<=2K:   G2 -> G4 -> G8
```

一个原 filler 的桶一旦解锁，它会进入 structural beam，并可选择 G1 或该桶已经解锁的 sizes。原本已经是 structural 的 job 仍按 `ring_size >= minimum_ring_size` 约束；无实际候选变化的 level 会通过 signature 去重跳过。

最后追加：

```text
all hard-legal candidates
```

此时所有 job 都进入 beam，并恢复全部硬合法候选，包括先前因 `minimum_ring_size` 被剪掉的较小 ring。

每个 level 完成 beam、fill 和 repair。如果得到可行方案，立即返回，不再解锁更短序列。这一外层顺序实现了比单纯最终 objective 更强的保护语义：只要当前受限搜索找到了可行解，就不会为了降低通信而打开更多短序列切分选项。

有限 beam 意味着“未找到”不等于“该 level 数学上不存在可行解”。因此短序列保护是强启发式策略，而不是全局最优性证明。增大 `beam_width` 和 `finalist_count` 可以降低误判风险，但会增加规划时间。

## 13. 端到端算法

```text
BR-PBS(lengths):
    validate inputs
    compute average token/compute targets
    build every job and its hard-legal buddy candidates
    best = none

    for level in progressive_relaxation_levels:
        derive structural jobs, fillers, and allowed candidates
        skip level if the structural candidate signature did not change

        beam = {empty state}
        for structural job in descending kappa order:
            expand every beam state with every allowed placement
            remove Pareto-dominated states
            merge states in the same 2% per-rank load cell
            retain diverse representatives up to beam_width

        completed = residual-fill every beam state with G1 fillers
        evaluate completed plans with exact final objective J
        repair the best finalist_count plans
        update the best plan seen over all levels

        if the best plan at this level is feasible:
            stop and return it

    if no level was feasible:
        return the minimum-objective infeasible plan

    sort output by descending ring size, then ring start, then original index
```

最后排序是 kernel metadata 的硬接口要求：batch 必须按 non-increasing ring size 排列。三个输出数组始终作为一个整体解释；`global_lengths[k]`、`ring_sizes[k]` 和 `ring_starts[k]` 描述同一条重排后的序列。

调用方若需要把输出恢复到原始样本顺序，必须在外部同步维护 permutation。当前 `HybridWorkload` 不公开 `original_index` 列表，因为 dataset benchmark 直接按规划后的顺序构造输入。

## 14. 一个可复现例子

现有测试中的 workload：

```python
lengths = [512] * 8 + [4096, 4096]
workload = assign_hierarchical_rings(
    lengths,
    world_size=8,
    is_causal=True,
)
```

当前确定性结果为：

```text
4096 -> G4 [0..3]
4096 -> G4 [4..7]
512  -> G1 [0]
512  -> G1 [1]
...
512  -> G1 [7]
```

每个 rank 的 token load 都是 1536，compute 与 token 最大偏差均为 0。方案在 `initial` level 就可行，所以 8 条 512 序列保持 G1：

```text
split_counts    = [0, 2, 0, 0, 0]
split_penalties = [0, 4, 0, 0, 0]
```

两个 4096 序列属于 `2K-4K` 桶，各自使用 G4，因此该桶 split count 为 2、总 penalty 为 4。该例展示了设计意图：长块承担结构化并行，短块填充各 rank 的残余容量。

测试还覆盖了只有解锁 `2K-4K G2` 才能可行的案例；其中 512 序列仍保持 G1。单条未对齐的 1280 causal 序列在 8 ranks 上只有 G1 可用，规划器返回 `feasible=False` 和最低违反方案，而不是抛弃 workload。

## 15. 复杂度与在线开销

令：

```text
R  = rank 数，最大 8
G  = buddy ring 位置数，最大 2R-1 = 15
B  = beam_width
NH = structural job 数
NF = filler 数
K  = finalist_count
```

忽略剪枝内部比较时，beam 扩展和 load 更新约为：

\[
O(BN_HGR).
\]

Residual fill 为：

\[
O(BN_FR).
\]

当前 `_remove_dominated()` 使用直接的 Pareto frontier 扫描；一次扩展最多产生约 \(BG\) 个状态，最坏比较开销是二次的：

\[
O((BG)^2d),
\]

其中 \(d\) 是十几个固定 metric 维度。量化合并发生在 dominance 过滤之后，所以它主要限制下一轮 beam，而不能消除本轮 dominance 的最坏二次项。

实际边界很小：`R <= 8`、`G <= 15`、默认 `B=64`，并且 Pareto 过滤通常会快速缩小状态数。局部修复只围绕最多 2 个高/低 rank 和每 rank 4 个重点 job 构造邻域，避免全 assignment 的二次枚举。

如果未来将该算法扩展到更大 world size 或数百个 structural jobs，当前 frontier 实现会成为优先优化对象；在现有固定拓扑下，引入复杂索引结构的收益有限。

## 16. 参数与调优权衡

| 参数 | 默认值 | 增大后的主要影响 | 风险 |
| --- | ---: | --- | --- |
| `compute_balance_tolerance` | 0.05 | 更容易得到可行方案、减少 relaxation | compute critical path 差异可能更大 |
| `token_balance_tolerance` | 0.10 | 更容易用 G1 filler 完成计划 | token/内存相关负载更不均衡 |
| `beam_width` | 64 | 保留更多结构方案，提高找到可行解的机会 | beam 与 Pareto 比较时间、内存增加 |
| `finalist_count` | 8 | 更多完整方案获得 local repair | repair 时间近似线性增加 |
| `structure_threshold` | 0.5 | 更多 job 留作廉价 filler | structural 搜索看不到中等 job 的 ring 选择 |
| `max_repair_iterations` | 32 | 允许更多严格局部改进 | 单 batch 规划时间增加，仍不保证全局最优 |

`structure_threshold` 降低时，更多 job 会进入 beam，搜索质量通常提高但组合数增加。它升高时，更多 job 走 G1 greedy fill；不过 `minimum_ring_size > 1` 的 job 无论阈值如何都仍是 structural。

内部常数当前没有公开为 API 参数：

```text
load quantization       2%
smooth-max lambda       8
repair extreme ranks    2
repair jobs per rank    4
numeric epsilon         1e-12
```

在缺少实测证据前不应同时调整多个常数。建议固定数据集、seed、token budget 和硬件，先比较 planner feasibility/split 指标，再比较端到端 benchmark；不能仅凭 proxy objective 推断性能提升。

## 17. 关键设计权衡

### 17.1 双负载 proxy vs. 精确时间模型

token 与二次 attention compute 能捕获长度异构的主要矛盾，计算便宜且不依赖具体 GPU 频率。它没有建模 head 数、tile 边界、占用率、通信重叠和 forward/backward 差异，因此 `D_C` 不是运行时间不均衡的直接测量。

尤其是 causal 模型当前把 \(A_i/p\) 均匀分给 ring 成员。真实 per-rank tile 数如果因 zigzag、尾块或调度策略不完全均匀，planner 会低估这种差异。未来若有稳定的 kernel tile 计数，可以把标量 \(A_i/p\) 替换为预计算的 per-rank 向量，而不需要改变 beam/repair 框架。

### 17.2 Pareto + 字典序 vs. 加权和

当前目标能清楚解释“为什么切了这条短序列”，并避免手工标定异构单位权重。代价是 Pareto frontier 可能较大，字典序也不能表达柔性的业务交换，例如“允许多切一条 2K 序列来换取 30% 通信下降”。如果确有这种需求，应先定义产品级交换规则，而不是随意加入 magic weight。

### 17.3 Beam search vs. 精确求解

Beam search 适合每个 batch 在线执行，并能自然利用 `R <= 8` 的小拓扑。有限宽度、状态量化和等价合并意味着它没有全局最优或完备可行性保证。MILP/CP-SAT 更适合作为离线 oracle：在代表性小 workload 上评估 heuristic optimality gap，而不是放入 runtime critical path。

### 17.4 Core/filler 分解 vs. 全量联合搜索

先决定长序列结构，再用短序列补洞，符合长尾数据集的形态，也使 `N <= 128` 级别 workload 可在线规划。但 greedy filler 可能需要 structural 方案预留特定二维缺口，有限 beam 不一定保留该状态。多个 finalists、smooth-max fill 和 repair 是对此的补偿，而不是严格证明。

### 17.5 短序列保护 vs. 最低通信

外层 relaxation 保证搜索优先级：先尝试不切 filler，再从长桶开始解锁。最终 objective 又从最短桶开始比较 split key。这种双层保护会有意放弃某些通信更低但切分短序列更多的方案。设计依据是短序列通信/协调固定成本更难摊薄，且大量短序列切分会使执行结构复杂化。

“保护”不是绝对禁止：短序列如果 `kappa` 很大或 `minimum_ring_size > 1`，从初始 level 起就可能成为 structural 并被切分；如果更保守 level 不可行，短桶最终也会被解锁。

### 17.6 解析通信 proxy vs. 实测网络成本

\(Q\) 只表示单序列 ring critical-path token 量，`communication_amplification` 表示总 token hops。它们忽略不同 buddy ring 的物理链路差异、并发争用、消息启动成本和 overlap。因此通信指标用于 tie-break 和诊断，不应替代端到端 benchmark。

### 17.7 固定 2% 量化 vs. 容差自适应

固定量化让行为简单、确定且容易测试，但在非常严格的容差下可能过粗，在很宽松的容差下又可能保留过多状态。目前默认容差与 2% cell 尺度匹配。若暴露该参数，应同时增加 planner quality/latency 基准，避免只调到某一个数据集。

## 18. 确定性、回退与错误处理

相同的 lengths、参数和 Python 版本下，规划器通过稳定排序和显式 tie-breaker 保持确定性。数据集 `make_workloads()` 使用一个由 `seed` 初始化并连续推进的 RNG stream：第一个 case 与 `make_workload()` 一致，后续 case 不会简单使用 `seed + case_index`。

输入错误会立即抛出 `ValueError`，包括：

- 非 2/4/8 的 `world_size`；
- 空 lengths、非正整数长度；
- 非有限或负的 tolerance/threshold；
- 非正 beam/finalist 数；
- 负 repair iteration 数。

找不到满足容差的解不是输入错误。规划器会遍历到 all-hard level，并返回搜索到的最小 \(J\) 方案：

```text
feasible = False
load_violation > 0
relaxation_label = 产生 best plan 的 level
```

这让 benchmark 可以打印和分析拓扑受限案例，同时由上层决定是否拒绝执行。当前 benchmark frontend 不会仅因 `feasible=False` 自动终止。

## 19. 输出与可观测性

`HybridWorkload` 同时承载 kernel metadata 和 planner diagnostics：

| 字段 | 含义 |
| --- | --- |
| `global_lengths` | 按 ring size 降序重排后的全局序列长度 |
| `ring_sizes`, `ring_starts` | 与每个长度一一对应的 buddy placement |
| `rank_tokens`, `rank_compute` | 模型估计的 per-rank load |
| `rank_communication` | per-rank token-hop proxy |
| `average_*`, `peak_*` | 平均与峰值 load |
| `compute_deviation`, `token_deviation` | 最大绝对相对偏差属性 |
| `load_violation`, `feasible` | 相对容差的最终状态 |
| `relaxation_level`, `relaxation_label` | 为得到方案解锁到哪一级 |
| `split_counts`, `split_penalties` | 五个长度桶的切分数量与程度 |
| `communication_cost` | objective 使用的 \(Q\) proxy |
| `communication_amplification` | 总 token-hop / 原始 token 属性 |
| `active_ring_count` | 使用过的 `(size, start)` 数量 |
| `repair_moves` | 最终方案经历的严格 repair 轮数 |

`ring_test/benchmark_dataset_forward.py --print-workload` 会打印 placement、per-rank load、容差、relaxation、split protection、通信 proxy 和最终 kernel metadata。诊断调度问题时，应先查看这些结构化指标，再查看 GPU benchmark 时间。

## 20. Mode 语义

`make_workloads()` 中：

```text
mode="noncausal" -> non-causal compute/alignment model
mode="causal"    -> causal compute/alignment model
mode="both"      -> causal model
```

`both` 使用 causal planner 是保守且确定的共享拓扑选择，避免为同一个采样 case 生成两套 metadata。它不表示同时优化两个 mode 的实测 latency。Backward dataset frontend 当前固定使用 causal planner。

## 21. 验证策略

CPU-only 测试覆盖：

- sampler bucket、对齐、seed 和多 case 行为；
- causal/non-causal placement 与 per-rank 统计守恒；
- G8 下完整 15 个 buddy 候选；
- Pareto dominance 语义；
- 存在可行方案时短序列保持 local；
- 从长桶开始的 progressive unlock；
- 不可行拓扑返回最低违反方案；
- G1 move/swap、同级 relocation、promotion、demotion、sibling demotion；
- planner 确定性和 benchmark frontend metadata 交接。

运行：

```bash
python -m unittest balancer.test_balancer
```

这些测试验证的是调度器逻辑和接口，不验证 proxy 与 Hopper 实测时间的拟合程度。性能结论必须使用目标硬件上的 dataset benchmark，并注意 benchmark 测量的是端到端 op 时间，包含 host 准备、scheduler 和通信开销。

## 22. 实现映射

| 设计环节 | 主要实现 |
| --- | --- |
| cost 与合法 size | `attention_compute`, `ring_communication_per_rank`, `eligible_ring_sizes` |
| job/candidate 构造 | `_build_jobs` |
| progressive unlock | `_relaxation_levels`, `_allowed_for_level` |
| beam 状态扩展 | `_add_to_beam_state`, `_pareto_beam_search` |
| Pareto/量化/多样性 | `_remove_dominated`, `_merge_equivalent_states`, `_select_representatives` |
| filler | `_residual_fill` |
| 完整 objective | `_evaluate_solution` |
| repair | `_repair_neighbors`, `_hierarchical_local_repair` |
| 总入口 | `assign_hierarchical_rings` |
| dataset facade | `make_workload`, `make_workloads` |

## 23. 可演进方向

以下方向与当前框架兼容，但在有 benchmark 证据前不应直接扩大实现范围：

1. 用 kernel tile 计数或 profile 拟合值替换 \(L^2\) proxy，特别是提供 causal per-rank compute vector。
2. 增加显存/arena capacity 硬约束，使“合法候选”同时覆盖资源容量。
3. 用代表性小 batch 的 MILP/CP-SAT 结果离线衡量 beam optimality gap。
4. 单独建立 planner latency 与 solution quality benchmark，评估量化、beam width 和 threshold。
5. 若真实执行不再是单 fused launch，用实际 packing/launch 规则替换 `active_ring_count` proxy。
6. 如果通用调用方需要保序，在输出中公开从规划顺序到 `original_index` 的 permutation。

在这些扩展中，优先保留当前抽象边界：candidate 提供合法 placement 和 per-rank load，搜索层只消费向量并比较目标。这样可以改进成本模型，而不必重写 buddy ring 搜索、residual fill 和 repair 的整体结构。
