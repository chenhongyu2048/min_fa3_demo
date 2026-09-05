# MegaCP Motivation 实验统一方案

训练 / Full-prefill / Decode

版本：2026-09-05，统一设计版。
保存项目：F:\MegaCp。
远端源码根目录：/home/LOCAL/shixuan/hongyu/tmp/test/min_fa3_demo。
已审阅源码 commit：b56656af3c6307755c1ada771575fa327649ecfb。

本文整合此前的训练/full-prefill、负载均衡和 decode 讨论，作为后续实验实施依据。它规定目的、操作步骤、测量口径、预期结果、完成标准和可暂缓工作。文中的“预期”均为待验证假设；只有明确标注“已有日志”的数字是历史实测。本轮仅整理方案，没有启动新 GPU benchmark，没有修改远端实现。

## 1. 论文主线与范围

### 1.1 一个共同问题，三个可观察现象

论文的共同问题是：现有 CP 的负载组织、通信和依赖推进多以序列、分块、ring step 或完整算子为单位，而 GPU 实际按更细的 attention tile 和数据就绪状态执行。这种粒度不一致可能导致：

1. 训练/full-prefill：通信与计算分别执行，重叠期间产生资源干扰；ring step 又切断计算连续性，增加重复状态访问和阶段开销。
2. 变长训练/full-prefill：均衡序列、token 或解析 FLOPs，不一定同时均衡实际 tile 工作、通信和投影/MLP 工作，也不一定维持高效内核形状。
3. Decode：CUDA Graph 减少主机提交开销后，整算子依赖和中间物化仍可能留下阶段尾部和额外内存访问。

Motivation 的任务是用少量实验建立这些问题存在的证据，不提前解释完整 MegaRing/MegaDCP 算法，也不把完整 evaluation 搬入 1–2 页正文。

### 1.2 已确认的范围决定

- 训练实验包含 forward 和 backward；full-prefill 对应 forward。Decode mixed 中的 chunk-prefill 与这里的 full-prefill 分开命名。
- 通算重叠实验使用均匀输入，暂不混入负载均衡；MegaRing 使用 All-CP 配置。
- 负载均衡比较 Megatron-style、Zeppelin 和全均匀切分 All-CP；单层 Transformer 用于后续补充 token load 对投影/MLP 的影响。
- Decode 主要使用 CUDA Graph 基线，复用现有 q=16 workload，按 decode-only 与 mixed 分组。本轮不要求另做 q=1。
- q=16 可用于讨论小 query 在 Tensor Core 最小 tile 粒度下的执行特征；配置表仍如实记录 q=16，不将实际通信字节或输出数据量改记为 q=1。
- Decode 首先测逐 CTA/SM 的阶段开始、有效工作结束及退出时间；精确 tile-ready 仅在需要更强结论时追加。
- 中间数据流量通过 NCU 验证；weights prefetch 不作为第一轮必须完成的实验。

## 2. 实验清单与优先级

“首轮必须”指形成核心 motivation 证据需要完成；“条件必做”指只有保留对应强结论才需要完成；“可暂缓”指可以先写清范围，随后放 evaluation 或附录。

| 编号 | 实验 | 目的 | 首轮最低完成范围 | 优先级 |
|---|---|---|---|---|
| T1 | 均匀输入通算重叠 | 测量并发干扰和真实净重叠收益 | 三 baseline、FWD/BWD、COMM/COMP/SERIAL/OVERLAP；Mega All-CP 完整调用 | 首轮必须 |
| T2 | Ring step 计算代价 | 隔离分步执行与资源争抢 | Forward 合法的分步/合并 compute-only 对照；backward 分步及附加开销 | 首轮必须；合并 backward 可暂缓 |
| T3 | 变长负载的多目标冲突 | 比较 token、attention work、通信与执行效率 | 同批输入下三策略的负载统计和 attention 时间 | 首轮必须 |
| T4 | 单层 Transformer | 证明 token 偏斜传递到投影/MLP及层延迟 | 复用 T3 manifest 的代表 batch | 可暂缓；声称非 attention 实测收益时必做 |
| D1 | Decode Graph 逐 SM 阶段时间 | 观察尾部空闲与阶段推进 | Decode-only/mixed，vLLM A2A 与 Mega 的代表 trace；基线阶段事件 | 首轮必须 |
| D2 | NCU 中间物化流量 | 区分实际 HBM 往返与缓存访问 | 两类输入的整图 DRAM/L2 指标和正常运行时间 | 首轮必须；若暂缓则删除 HBM 强结论 |
| D3 | Tile-ready / stage gate 消融 | 建立阶段边界阻塞的直接因果证据 | 少量 tile 依赖就绪记录，或受控 gate 变体 | 条件必做 |
| D4 | 下一算子权重预取 | 验证预取机会与净延迟收益 | Attention→输出投影，Graph/Mega 各有无预取 | 可暂缓 |

首轮不扫描所有配置的笛卡尔积。先在一个固定 GPU 数和代表 workload 上完成测量链，再扩展少量长度或 DCP 点确认稳定性。

## 3. 所有实验共用的规则

### 3.1 硬件、配置与输入冻结

每次实验保存独立 manifest，至少记录：

- 仓库 commit、CUDA/PyTorch/NCCL 版本、GPU 型号与实际 SM 数、GPU UUID、MIG 状态、互联拓扑。
- World size、TP/CP/DCP 分组、dtype、Hq/Hkv、head dimension、causal/mask、后端和完整方法名。
- 原始序列长度、实际执行长度、padding、sample/case ID、seed、有效 token 数。
- 实际 tile shape、每序列 splits、group/step 顺序、通信 CTA 配额；不要只保存 auto 参数。
- 执行模式、warmup、迭代数、计时边界、是否插桩、profiler 配置和 correctness 结果。

当前 H20 节点与历史 H100 结果分开报告。此前观察到 GPU 7 有 MIG，先选择实际可分配的同构完整 GPU；不要为了跑八卡实验自动修改 MIG。训练可先用 W=4。Decode 历史 TP=8；如果当前只能分配四卡，需要重新生成合法 TP/DCP/head 配置并标为新实验，不能将其称为旧配置复现。

### 3.2 正常性能与诊断分开

- 正常性能：关闭内部时间戳与阶段诊断事件，测完整合法调用，包含每次必需的 reset、phase advance、归约、等待和后处理。
- 诊断：单独开启逐阶段 events、Nsight Systems 或 timestamp/NCU；用正常运行的性能作参照，报告插桩影响。
- 所有方法排除一次性分配、communicator 创建、固定输入构造及 Graph capture；动态元数据若每次执行必须更新，其成本应单列，不能由稳态 replay 外推完整服务开销。
- 多 stream 调用的结束事件必须等待所有相关工作完成；不在每个 step 插入全设备同步来测重叠。
- 固定配置先做输出和梯度 correctness，再采性能。Graph 要验证多次 replay 的状态复用，不能只检查第一次输出。

### 3.3 统计与原始数据

初轮可沿用 warmup=40、timed iterations=60；先检查稳定性，在必要时增加，不能把 60 次迭代视作 60 个独立 workload。正式代表结果使用数次独立进程运行，轮换方法顺序，报告波动。

完整调用每迭代先取 max-rank，再在迭代间取 p50/p90；不同 batch 的结果另行汇总并注明口径。主对比使用相同 case 的配对延迟比。

禁止把各阶段分别取 rank 最大值或 p50 后相加画成“完整关键路径”。堆叠必须来自同一 iteration、同一 rank，且各部分互斥。重叠区间用时间线表达。

不同 GPU 的 globaltimer 不直接相减。保留 rank-local 原始时间；跨 rank 用相对时长或经过校准的工具时间轴。

## 4. T1：均匀输入下的通算重叠

### 4.1 目的与假设

验证两个独立问题：

- 相同通信和 attention 在并发时是否各自延长？
- 即使发生延长，重叠后的完整调用仍获得多少净收益？

预期是部分工作点的通信和计算都变慢，使净重叠收益受限。不能预先要求 OVERLAP 比 SERIAL 更慢；两者各自变慢与整体仍加速可以同时成立。

### 4.2 方法和初始输入

| 方法 | 对应路径/配置 | 用途 |
|---|---|---|
| FA3 Ring | 完整 zigzag P2P ring | 独立通信与计算的 ring 基线 |
| AllGather-CP | AllGatherAttention | KV-head 分块流水基线 |
| Llama3-style AllGather-CP | Llama3AllGatherAttention | 对应 packed/two-block zigzag 基线 |
| MegaRing All-CP | min_varlen_mega_ring，每样本 G=W | 完整设备端执行参照 |

起始建议：BF16、causal、Hq=32、Hkv=8、D=128、B=4、W=4；global per-sequence L=8K/32K/128K。先实现 32K 代表点，再补其余点。K 按 1024 定义。

三 baseline 的每条序列长度相同；使用正确匹配的 causal sharding，确认每 rank 有效 attention 工作量。均匀长度不意味着任意 causal 分片都均衡。

现有 homogeneous microbench 的 case.seqlen 是 local length，global L=local length×W。配置和图必须统一 global context length。

当前 backward 的 Python ring 路径使用 trimmed min_fa3 block，不能直接标为 external FA3 backward。统一底层后端，或在方法名中说明差异；完整数学与输入必须一致。

### 4.3 具体操作

对三种 baseline 的 FWD 和 BWD 各实现四种模式：

1. COMM-ONLY：去除 attention，重放原通信的消息序列、大小、dtype、算法和拓扑；使用预先准备好的 payload。
2. COMP-ONLY：提前准备原执行中各 step 的真实 KV 输入，重放相同 shape/mask/stride/head chunk 的 attention 及必要合并，去掉跨 GPU 传输。
3. SERIAL：保留完整依赖、通信和计算，明确禁止它们并发。
4. OVERLAP：原来的合法异步流水，不插入逐 step device synchronize。

MegaRing 主测完整 All-CP 调用；不要将 fused kernel 任意拆出“纯通信/纯计算”，再认为成本与原 fused 执行相同。

Backward COMP-ONLY 必须使用正确的 forward O/LSE/dO，并复位梯度累加状态。COMM-ONLY 的 dKV payload 必须保持真实形状、精度和归约操作，但它不模拟原本的梯度生产时间。

阶段至少区分：KV ingress、attention、O/LSE merge 或 dQ/dKV 累加、dKV owner 回传/reduce-scatter、pack/reorder/cast/reset。Reduce-scatter 命名为通信/归约，不称为纯链路传输。

### 4.4 测量和输出

使用未插桩完整时间得到 Tserial、Toverlap、Tmega。独立 Nsight Systems/CUPTI trace 关联 rank、step、kernel，取得：

- C0/A0：独立执行时的匹配通信/计算 kernel duration。
- Cov/Aov：OVERLAP 中相同操作的 duration。
- Scomm=Cov/C0，Sattn=Aov/A0。
- 净重叠加速=Tserial/Toverlap。

CUDA API 调用时间、Work.wait 时间不是 GPU 通信时间；调用方 stream 的 event 也不必然覆盖 NCCL 内部 stream。通信 kernel duration 可能包含远端等待，不能解释为纯 wire time。

原始数据表：case、direction、method、mode、rank、iteration、step、kernel_kind、start/end、duration、完整调用时间。

推荐图：一个代表 context 的通信/计算 slowdown 配对柱，与 SERIAL/OVERLAP/Mega 完整调用时间。FWD/BWD 用两个小面板或并列分组。

### 4.5 预期结果与结论门槛

- 两种 slowdown>1 且 trace 确有并发：支持“并发资源干扰”。
- OVERLAP 仍快于 SERIAL：如实写“净收益受干扰限制”，不写“重叠无效”。
- 无明显 slowdown、仅有依赖间隙：改写为流水依赖或阶段边界问题。

若要明确归因 SM 争抢，再做一个轻量 co-run：所有 rank 的通信 payload 均已就绪，受控同时启动通信与 attention，减少远端生产等待混杂；在版本支持时取两三个通信 CTA 配置，并为每个配置重测 COMM-ONLY。单凭 CTA 配置变化仍不能完全排除 HBM/L2 干扰，因此保守表述为 SM/存储资源竞争。

### 4.6 哪些可暂缓

可暂缓：W=8、多节点、更多 B/L、全部 NCCL CTA 扫描、精确无干扰 DAG 模型、默认 transport 之外的对照。

不能省略：FWD/BWD 对照、合法通信序列、完整调用计时、后端标注与 correctness。

max(C0,A0) 只可作理想化参考，不是整个 ring 的可达最优时间；填充、排空和 DAG 依赖未被这个表达式建模。

## 5. T2：Ring step 是否降低计算效率

### 5.1 目的

在没有通信竞争时，检查将同一 attention 工作切成多个 step 是否产生重复 Q/O/LSE 访问、额外 merge、launch 和较差执行形状。该实验把“分步代价”与 T1 的“并发干扰”分开。

### 5.2 具体操作

沿用 T1 的均匀输入和匹配后端，预置完整所需 KV：

- COMP-STEP：仍按原 ring step 顺序执行并合并状态。
- COMP-SEGMENT：在数学等价的前提下合并可连续消费的 KV 段，减少重复状态访问。

先验证两个版本的 Q 区域、KV 范围、global causal mask 和数学工作一致，再比较输出。不能直接拿一次普通 causal FA3 替换任意 zigzag rank 的计算；不合法时先限制为可正确映射的代表子问题，并说明范围。

分别报告 attention kernel body 总量与包含 merge/pre/post 的 compute-only 完整时间，记录调用次数、tile 数、状态访问次数的实现统计及 useful TFLOP/s。

Forward 的分步/合并两边最好都用 Graph 做一个小对照，减少 host launch 差异。如果纯 kernel 数学不变而全调用变快，收益可能主要来自 merge/状态重访；如 kernel body 也改善，说明执行形状或局部性可能贡献。

### 5.3 预期结果和完成标准

预期 COMP-SEGMENT 在部分配置提高 useful TFLOP/s、减少状态重访，并缩短 compute-only 时间。完成标准是得到至少一个合法、可复现的代表点，以及清楚区分 kernel body 和附加处理的解释。

若只减少 host launch，则将结论限制为提交代价；若 Graph 后仍有差异，则进一步支持设备端执行分割成本。负结果保留，不能把不同数学工作量误认成效率收益。

### 5.4 哪些可暂缓

Forward 合法配对是首轮重点。Backward 先完成按 step 的数学、pre/post、累加成本统计；如果没有合法 merged backward，不要求本轮实现新算法，不能声称已经测出 backward 合并收益。

不同 step 数、更多 tile shape、详细 HBM 计数可留 evaluation。

## 6. T3：真实长度分布下的负载与效率

### 6.1 目的与假设

比较三种负载组织方式能否同时控制：token 偏斜、attention 执行成本和通信成本。预期不是“三种方法都无法均衡”，而是不同策略可能在这些目标之间存在冲突。

特别是 All-CP 可以很好地均衡 token 和工作，却可能因短任务、更多通信或状态访问而降低执行效率；这是需要保留的对照。

### 6.2 数据和公平性

方法：megatron_hybrid_cp、zeppelin、全均匀切分 all-cp。

先用项目已有长度数据选一个长尾数据集，冻结 30–50 个 batch；主线成立后增加短序列/长序列占主导的两个分布。固定 global token budget（起始可用 128K）、sample IDs、原始长度、seed 和容量限制，所有方法处理同一个 manifest。

现有 balancer/sampler.py 根据 sequence_length_buckets.json 的概率抽样，并进行 ring-aware alignment；这种输入应称“真实长度分布驱动的合成 batch”。若希望论文写“真实 batch”，从原始 tokenized length 列表构造 manifest，保留样本归属；不必为 shape benchmark 读取完整文本。

同时记录原始长度和执行/padding 长度，不允许各方法独立重新抽样。

### 6.3 具体操作和三类指标

第一步，对三策略输出的 placement 保存逐序列 ownership、group、degree、padding，以及每个 execution group 的工作量。

第二步，按实际执行计算：

| 指标 | 定义 | 作用 |
|---|---|---|
| I_token | max_r N_r / mean_r N_r，N_r 是投影/MLP 实际处理的 physical tokens | 观察非 attention 工作潜在偏斜 |
| I_attn | max_r A_r / mean_r A_r，A_r 为明确口径的有效 attention 工作 | 观察数学工作偏斜 |
| Tile work | 实际 mask/layout/tile shape 下的任务数及工作分布 | 区分解析均衡与实际执行 |
| Padding ratio | 多处理 token / 原始有效 token | 区分均衡和浪费 |
| B_attn | CP 必要 KV 与 backward dKV 交换/归约的 payload | 观察正常 CP 通信 |
| B_layout | placement 重分布、输出/梯度归位的额外 payload | 本文“带外通信”定义 |
| Attention latency | 完整调用每迭代 max-rank 的 p50/p90 | 将负载统计关联到实际性能 |

解析有用工作、padding 后工作与 tile 访问量分开；模型预测的 physical_flops 不是实测 Tensor Core 指令 FLOPs。Useful TFLOP/s 使用所有方法一致的有用数学工作定义及实测时间。

通信保存每 rank Tx 和 sum(Tx)，不要加 Tx+Rx 双计数；API payload、算法 byte-hop、链路 counter 分别命名。本地 pack/unpack 记本地内存操作，不计入跨 GPU 字节。

第三步，在相同 batch 上测 native/adapted execution 的 attention 时间。若用同一 Mega executor 比较 planner，应标成 planner-adapter 控制，不能冒充原生系统性能。Zeppelin 原生 placement 与 buddy 映射版本分开。

### 6.4 带外通信的实施边界

当前 Megatron adapter 没有执行被裁掉的 dataloader rerouting。未实现的 B_layout 必须记 excluded/not measured，不能当实测 0。

如果首轮不实现完整重分布，可以先用共同初始 ownership 计算精确的逻辑搬运 payload，图中明确标为 analytical bytes；时间留空。后续真正计时再实现分发与归位。

必须声明初始 ownership 和布局保持范围：跨所有层保持布局的搬运是一次性开销，不能对每层重复收费；仅实际每层发生的搬运才记为每层。

### 6.5 预期结果和展示

预期看到三种可能：

- Token/attention work 偏斜对应某些 rank 变慢。
- 总量均衡，但 execution group 的串行/barrier 暴露阶段性空闲。
- All-CP 均衡较好，但通信或短任务效率付出代价。

主图用 token imbalance、attention work imbalance、communication 三个紧凑面板，并给一行配对 attention latency/useful TFLOP/s。不能仅凭静态柱状图得出“性能差”。

如希望直接对应早先“相同 context、均匀与不均匀输入”的对照，可额外做固定总 token 的均匀/长尾合成 batch。必须分别统计真实 attention pairs：同总 token 并不意味着相同 sum(L²)，因此 raw latency 差不能全归因于不均衡。这是辅助实验，不替代真实分布对照。

### 6.6 哪些可暂缓

可先做一个数据集和固定 token budget；更多数据集、扩展长度、动态到达可暂缓。完整 B_layout 实现可暂缓，但必须以分析量/未测量准确标注。

不能省略：同批配对、padding、三类负载定义、实际 attention 性能、适配实现范围。

## 7. T4：单层 Transformer 补充实验

目的：证明 token load 影响投影/MLP，并检查 attention 级收益是否传递到完整层。

复用 T3 的相同 manifest、placement、padding、TP、模型宽度和 GEMM backend，选代表性 batch 即可。记录每 rank 的 QKV projection、CP attention、output projection、MLP、norm/residual 和完整层 FWD/BWD。

现有 benchmark_transformer_layer.py 的 SelfAttention 包括 QKV、core attention、output projection；others 是 norm/residual/MLP 等剩余项，不能直接标成 MLP。第一阶段可使用已有 full-layer/self-attention 数据；只有加了独立事件或 NVTX/GEMM 关联，才能解释投影/MLP 的单项耗时。

预期 physical token 较多的 rank 有更大的投影/MLP 工作，但耗时不要求严格线性；GEMM 形状可能改变利用率。完整层 critical rank 也可能与 attention critical rank 不同。

本实验可移至 evaluation，motivation 引用即可。暂缓时只能说 token load “预计影响”投影/MLP，不能写成已实测。

## 8. Decode 历史实验的可复用内容

日志绝对路径：
/home/LOCAL/shixuan/hongyu/tmp/test/min_fa3_demo/benchmark_logs/bench_dcp/20260812-125820-arrival4-phases-graph

### 8.1 已核实的配置

- 每个 DCP=2/4/8 各 100 cases：decode-only 54，mixed 46。
- 使用实际 q=16 的 speculative verification；mixed 加入不同长度的 chunk-prefill。本方案保留这些 workload，不把 q=1 当作首轮依赖。
- Arrival_time_scale=4 表示时间缩放，不能写成 4 requests/s。
- 历史硬件 H100 80GB HBM3，TP=8，CUDA 12.8，torch 2.10.0+cu128；历史项目 commit=32a7190b71fe33964677051a00cdbe6ee15d7b05。
- vLLM A2A 和 AG-RS 均是 cuda_graph，启用了阶段 CUDA events；SGLang baseline 也采用 Graph。
- Mega 是 eager、kernel-only，预先上传 metadata，reset 不计入。历史总延迟不可直接形成 Graph 对 Graph 的公平 speedup。
- baseline 使用相同 min_fa3_op，测的是编排/collective 差异，不是原生 vLLM/SGLang 全引擎性能。
- 原实验 check=false；新正式结果必须补 correctness。

### 8.2 可用的先导观察

下表单位 μs，为各 case 的跨 rank 每迭代最大延迟 p50，再在该类别内取中位数。

| 输入 | DCP | vLLM A2A Graph | vLLM AG-RS Graph | SGLang AG-AR Graph |
|---|---:|---:|---:|---:|
| Decode-only | 2 | 172.63 | 173.58 | 251.33 |
| Decode-only | 4 | 172.38 | 182.87 | 261.38 |
| Decode-only | 8 | 206.13 | 226.82 | 338.74 |
| Mixed | 2 | 567.42 | 589.90 | 743.88 |
| Mixed | 4 | 569.04 | 614.62 | 845.75 |
| Mixed | 8 | 677.83 | 755.52 | 1107.98 |

Mixed 中 vLLM A2A 的 history attention 从 DCP=2 的 311.78 μs 降到 DCP=8 的 151.33 μs，而总延迟从 567.42 μs 增至 677.83 μs。这支持调查非局部 attention 开销，不直接证明 wave quantization 或 HBM 瓶颈。

Mega mixed 的 kernel-only 延迟对 comm CTA 很敏感：DCP=8 从 4 CTA 的 1040.17 μs 降至 20 CTA 的 422.70 μs（同类 case p50 中位数）。因此新实验在独立调参集上选定资源配置，在测试集固定；不能逐 case 取最佳 comm CTA 再和固定 baseline 比。

这些旧日志可用于筛选代表 case、校验阶段趋势、保留原始 baseline 数据；不能代替新的逐 SM trace 或 NCU counter。

## 9. D1：逐 CTA/SM 阶段开始与结束时间

### 9.1 目的

直接观察在 Graph 执行中：上一阶段是否只有少数 SM 仍做有效工作，其他 SM 已空闲，而下一阶段仍等到整算子完成；Mega 是否让部分 SM 提前推进至后续阶段。

这项实验先建立“阶段尾部与整体推进”的证据。只有补充全部输入 ready 时间，才进一步声称“已就绪 tile 被整算子边界阻塞”。

### 9.2 对照与输入

首轮以 vLLM A2A Graph 和 Mega Graph 为详细 trace 对照；同时保留 vLLM AG-RS、SGLang AG-AR 的原有阶段时间和总延迟。无需立即为所有通信实现做内核插桩。

先固定一个合法 TP/DCP 配置，从旧 manifest 预先选 decode-only、mixed 各一个中位区域代表 case，并可追加一个尾部 case。选择依据使用长度/任务数或基线延迟分位，不根据 Mega 的最佳加速挑样本；全 case 汇总用于确认代表性。

自然 mixed 比 decode-only 工作更多，其绝对延迟差不能直接归因于不规则性。主对照是同一 case 的两种执行方式。

### 9.3 插桩设计

按 CTA 记录，带 SM ID，后处理再按 SM 聚合。不能直接用每个 SM 一个槽：baseline 一个 SM 可能执行多波 CTA。

最小字段：

iteration, rank, kernel_id, phase_id, CTA_id, SM_id, start_ns, useful_work_end_ns, exit_ns, valid_work_count。

- start：本 CTA 进入指定阶段的时间；它可能早于依赖满足或真正计算。
- useful_work_end：本 CTA 在此阶段最后一项有效工作结束的时间。
- exit：离开该阶段前的时间；与 useful_work_end 分开，避免将同步/polling 当成有用计算。
- 没有任务的 CTA 设 valid_work_count=0；图中不当作计算占用。

对自有 CUDA 内核使用 SM90 globaltimer 和实际 SM ID；固定大小预分配缓冲，每个 CTA/阶段拥有唯一记录槽，尽量不使用高频全局原子。不为了计时额外插入每 tile barrier；记录语义必须覆盖相关 producer/consumer 完成，无法准确捕获时明确标注 CTA 近似。

Graph 每次 replay 的缓冲清理和索引更新放在正确位置；profile 只抓少数迭代，热身后再开始。性能运行关闭该功能。记录插桩开关前后的完整延迟，检查是否改变寄存器/共享内存需求或 kernel 选择。

Baseline 的 attention、pack、combine 可在本地实现中插桩；NCCL 内部 kernel 第一轮用已有 CUDA events/NSYS GPU 时间区间表示。不把通信区间复制到每个 SM 并虚构逐 SM 数据。

Mega 的 compute CTA 可以先完成自身 attention 再进入 history combine；当前全局 attention_done 是最后一个 compute CTA 的完成记录，不表示此前所有 CTA 均被全局屏障锁住。

### 9.4 图和指标

时间图：横轴为该 GPU 一次调用的相对时间，纵轴 SM ID，颜色区分 Q transfer、attention、pack/combine、receive/final combine。保留一个 SM 的多段记录与中间空隙。

Baseline 通信整体时间单独放顶部轨道；自有阶段画精确 CTA/SM 区间。若 SM 同时承载多个 CTA，使用子轨道或区间并集，不能直接覆盖记录。

输出：

- 各阶段 CTA 有效结束时间的 P50/P90/max。
- T_tail=P100(useful_end)-P50(useful_end)，描述 CTA 完成尾部跨度。
- 有用工作 SM 数随时间的曲线；persistent block resident 不等于有用工作活跃。
- 前一阶段完成分布与后一阶段开始分布，观察推进是否集中在全局边界之后。
- 相同 case 的无插桩 Graph 总时间。

P100-P50 对异构 CTA 只描述尾部，不单独代表空闲损失。更完整的空闲面积需要逐 SM 有用工作区间，并限定哪些 SM 对下一阶段具有兼容资源。

历史 phase milestones 和 tails 的聚合次序不同，不可相减后拼时间图：max(B-A) 不等于 max(B)-max(A)，p50(B-A) 不等于 p50(B)-p50(A)。新图使用同一次 replay、同一个 GPU 的原始记录。

### 9.5 预期结果与可写结论

预期 baseline 有明显阶段尾部，下一阶段在整算子完成后启动；Mega 允许部分 CTA 较早进入后续工作，使部分尾部被利用。若只是阶段变快、没有交错，则收益应解释为执行效率，不能声称已测到细粒度推进。

逐 SM 图可以支持“阶段尾部”和“算子级推进”。仅有该图时，wave quantization 写作候选原因或更宽泛的尾部利用率问题；等长任务的整数波扫描完成后再使用严格量化结论。

### 9.6 哪些可暂缓

- 不强制本轮增加 q=1。
- 不必修改 NCCL 源码实现全部通信 CTA 计时。
- 不必追踪所有 tile，也不必扫描所有 DCP/comm CTA。
- 精确 tile-ready、gate 消融和 wave quantization 小扫描见 D3，只有对应强结论需要时再补。

## 10. D2：用 NCU 验证中间物化与 HBM 流量

### 10.1 目的和可证伪假设

验证跨算子的 output/LSE、pack/send、receive/unpack、state merge 缓冲是否增加实际 DRAM 访问，并且是否影响完整调用延迟。

不能把“写 global memory”直接等同于“往返 HBM”。中间结果可能留在 L2；NCU 必须区分 DRAM、L2 与逻辑数据量。

### 10.2 先列 buffer 账本

沿生产者→消费者列出 buffer、shape/dtype、owner、读写次数和生存期：

- History attention output/LSE → pack/send。
- Send/receive buffer → unpack/combine。
- Mixed 的 chunk output/LSE 与 history result → state merge。
- Mega 的 partial output、scratch、IPC publish、final combine。

将数学必需 KV/输出、DCP 必需远端 payload、额外中间 pass 分开。账本是实现推导，不是 HBM 实测；不能只给 baseline 计中间缓冲而遗漏 Mega 的 scratch。

### 10.3 最小测量矩阵

沿用 D1 的配置和输入。第一轮先做：

| 输入 | 方法 | 正常性能 | NCU 整图流量 |
|---|---|---|---|
| Decode-only | vLLM A2A Graph / Mega Graph | 必须 | 必须 |
| Mixed | vLLM A2A Graph / Mega Graph | 必须 | 必须 |

AG-RS 与 SGLang 的逐 kernel counter 可暂缓，后续补在代表点即可。先确认前两条链的指标可靠，再做更大的中间缓冲尺寸探针和 staging 消融。

固定数学工作、实际 splits、tile shape、通信配额；若方法无法使用相同 splits，必须记录差异，先把总流量差解释为完整方法差异，不能全部归因于物化。

### 10.4 NCU 采集流程

本节点已确认安装 /usr/local/cuda-12.8/bin/ncu，版本 2025.1，支持 graph-profiling node/graph。首轮只做少量 metric。

步骤一：能力检查。

- 用 ncu --query-metrics 确认当前 GPU/模式支持的 DRAM/L2 指标和 counter 权限。
- 候选 DRAM 指标：dram__bytes_read.sum、dram__bytes_write.sum。
- L2 选择对应架构的 read/write sectors、bytes 和 hit rate 指标。若只有 sectors，用文档确认 sector 大小后换算；不要凭名字猜公式。
- 不要求 NCU 提供 NVLink counter；其可用性依工具/平台而定。通信 payload 可由代码账本记录；若要链路硬件字节，另找可用工具并分开标注。

步骤二：准备单 case、单方法的 profiling 入口。

完成分布式初始化、固定输入、workspace、capture、warmup，然后仅让一个指定 Graph replay 进入采集区间。保持所有 rank 的 collective 顺序一致，将结果导出和 CPU 分析放在区间之外。推荐 profiler start/stop 或确认有效的 graph/range 过滤，不在完整 100-case sweep 上直接使用 --set full。

步骤三：先验证整图采集可行性。

优先评估 graph-profiling=graph，使一次完整 graph 成为一个 workload，保留节点间的真实缓存交接。正常缓存条件使用 cache-control=none；需要多 pass 时，优先采用可协调的应用重放，让每次从相同初始化/warmup 恢复。

这些是采集配置意图，不是未经验证即可套用的启动命令。多进程通信对重放进度敏感：先只采一个 metric/很少 pass，检查能否完成、是否发生额外 replay、collective 次序和总量是否合理，再扩展指标。不能让某一 rank 私自重放通信 graph，而其他 rank 已进入下一轮。

所有 rank 若需协同采集，必须共同进入相同采集 replay。若工具不能可靠支持该完整多 GPU 工作负载，则将 NCU 限于可安全重放的局部 producer/consumer 子图，明确称局部物化流量实验；完整通信性能仍由正常运行给出，不伪称完成整图测量。

步骤四：局部归因。

整图差异确认后，单独检查 pack、unpack/combine、merge 或等价局部子图的指标，定位额外流量。默认 node/kernel replay 可能清缓存或改变生产者→消费者关系，逐 kernel 数据不能未经验证相加替代整图稳态流量。

步骤五：正常性能复测。

相同配置关闭 NCU、关闭 timestamp 测总延迟；NCU 运行的 duration 只用于诊断。记录 replay、cache、clock-control、采集 rank/设备和 pass 数。不要用 NCU 默认串行化后的时间评价实际通算重叠。

### 10.5 汇总指标

- B_HBM_read、B_HBM_write、B_HBM_total：一次完整采集 workload 的 DRAM 字节。
- B_L2：定义清楚的 L2 读写流量，避免把 lookup hit rate 与字节量混为一项。
- B_logical：buffer 账本的逻辑流量，单列为模型。
- T_normal：无 profiler 的完整 Graph 调用时间。
- 若测齐所有 GPU，报告 sum_r(B_HBM,r) 与逐 rank 分布；只采到一个 rank 就只报告该 rank，不假设乘 world size 一定正确。

图上标清是整个 DCP 调用的流量还是局部子图流量。完整 baseline 与 Mega 的差值还可能包含 KV 重读、缓存政策、split、通信路径差异；它只能建立相关性，单一物化机制需要下一节控制。

### 10.6 因果消融与缓存控制

优先同一 Mega 路径比较：直接发布 vs 写 staging 再读取发布。保持通信 payload、数学工作、splits、tile shape、调度策略相同。或者在 baseline 中实现合法的 producer-side pack，减少一个中间 pass；若改动同时影响多项因素，标成复合优化。

至少保留正常 warmup/稳态条件。可以后续加入两个工作集尺寸：中间缓冲较小与明显大于有效 L2 容量，验证是否从缓存流量转成 DRAM 流量。是否驻留以 counter 为准，不用容量估计代替测量。

若要模拟真实服务的权重/KV缓存干扰，应使用可说明来源的工作集；人为 flush 可以作为冷缓存边界，不能成为唯一主结果。

### 10.7 预期结果与完成门槛

| 测量结果 | 允许的结论 |
|---|---|
| DRAM 字节减少，受控消融下正常延迟也下降 | 中间物化的 HBM 往返影响该配置性能 |
| 主要减少 L2 流量，DRAM 变化小 | 中间物化增加缓存/内存层级访问，不写成 HBM 瓶颈 |
| 字节下降但总延迟不降 | 降低流量成立，但不是该工作点主要延迟瓶颈 |
| 只有不同完整方法的总字节差 | 方法级差异成立，单一物化原因仍待消融 |
| 多 GPU 整图 counter 无法可靠采集 | 改为局部子图诊断并明确范围，不捏造整图流量 |

### 10.8 哪些可暂缓

先完成 decode-only/mixed 各一个代表点的 NCU 整体或明确定义的局部测量。其余基线的全 kernel 指标、所有 DCP 扫描、跨 L2 容量探针可以暂缓。

Staging 消融可以晚于首轮 counter；暂缓时只写方法级流量差。若 NCU 整项暂缓，motivation 暂用“中间结果物化存在额外访问路径”的代码事实，不写“HBM round trip 已被实验证明”。

## 11. 条件补证和可以先不用完成的项目

### 11.1 D3：更强的 tile 因果证据

若正文要明确写“tile 已 ready 却被整算子边界阻塞”，选少量输出 tile，记录它的全部真实输入 ready 和 consumer start：W_j=start_j-ready_j。ready 必须包含全部分片/LSE/跨 rank 依赖，不能拿第一个生产者完成代替。

同时观察当时是否有兼容资源；仅 W_j 大不能排除资源饱和。跨 GPU ready 由消费端观察到合法 ready 状态时在本地记录，不直接相减两 GPU globaltimer。

可另做 Mega 的 tile-ready vs 全阶段 gate。全阶段 gate 不能简单给任意 persistent kernel 加自旋屏障，必须保证所有参与 CTA 可合法取得进展，或使用明确合法的分阶段启动变体；避免用死锁或额外不对等资源制造“消融结果”。

若正文严格使用 wave quantization，补等长任务扫描：固定 split/tile，任务数围绕实测有效并发容量 C 的 C-1/C/C+1、2C-1/2C/2C+1，观察时间阶跃和末波利用率。变长 straggler 与整数波量化分别命名。

上述三项均可先暂缓；首版将结论限定为阶段尾部和算子级推进。

### 11.2 D4：Weights prefetch

当前 benchmark 不含 projection/MLP 权重，不能用现有 timestamp 验证 weights prefetch。暂按 HBM→L2 的下一算子权重预取作为后续假设；如果实际关注 CPU/NVMe offload，另开完整传输设计，不混入本轮。

后续最小扩展：DCP attention → 输出投影 GEMM。比较 Graph 无预取/有合法预取节点，以及 Mega 无预取/有预取。固定实际权重分片，测 attention+GEMM 的合计时间和缓存效果，不能只展示 GEMM 单独变快。

CUDA Graph 可以表达独立节点和依赖，因此不写“CUDA Graph 无法 weights prefetch”；只研究当前执行链是否未利用可用窗口。该项不阻塞首轮 motivation。

### 11.3 其他可暂缓项及措辞影响

| 可暂缓项 | 暂缓后如何限制结论 |
|---|---|
| q=1 decode | 主实验保留实际 q=16 配置，不声称逐字节等同 q=1 |
| Eager 主机开销分解 | 不声称已经证明 host overhead 主导，只说 Graph 为主要基线 |
| 完整 vLLM/SGLang engine 接入 | 使用“同核 orchestration baseline”名称 |
| 多层/端到端服务、TPOT/TTFT | 仅报告 attention/layer latency，不外推服务指标 |
| 完整投影/MLP 分项 | Token 影响作为预期，或只报告已有 full-layer 时间 |
| Megatron 被裁掉的布局通信 | 标为 analytical/excluded，不能写实测 0 |
| 合并 backward 新内核 | 报告 backward 分步与附加工作，不声称已测合并收益 |
| 全部参数、硬件、多节点扫描 | 结论限定在已测工作点，扩展放 evaluation |
| 纯 SM 瓶颈归因 | 用“并发资源干扰”，不把所有 slowdown 定为 SM 争抢 |

## 12. 实施顺序与交付文件

### 12.1 分批推进

第一批：冻结配置与采集口径。

- 复用旧 decode manifest，确认当前硬件与合法 TP/DCP 配置。
- 冻结训练均匀输入、T3 原始长度 manifest。
- 确认三训练 baseline、三 decode baseline 的底层后端和完整计时边界。
- 完成代表配置 correctness。

第二批：得到最小现象证据。

- T1 先跑一个 context 的 FWD，再覆盖 BWD 与其他 baseline 四模式。
- T2 做合法 forward compute-only 对照。
- T3 先出一个数据集的三策略负载与 attention 时间。
- D1 实现 CTA/SM 记录，先做两类输入与 vLLM A2A/Mega 的图。

第三批：补物理机制和收敛正文。

- D2 先验证 NCU 少量指标采集链，再采两个代表 case。
- 根据是否保留强结论选择 co-run、tile-ready、staging 消融，不一次全部展开。
- 补少量配置确认稳定性，将更广覆盖与单层实验移到 evaluation。

### 12.2 建议的结果目录结构

在项目下新建独立实验批次目录，避免覆盖历史日志：

benchmark_logs/motivation/<run_id>/
  manifest.json
  correctness.json
  training_overlap/
  training_step/
  load_balance/
  decode_sm_trace/
  decode_memory/
  figures/
  summary.md

各目录至少保留配置、per-rank/per-iteration 原始 CSV/JSON、汇总脚本版本、命令与 profiler 报告。Trace schema 和 timing boundary 写进 manifest，不能只留下 PNG。

建议汇总列：experiment_id、case_id、method、execution_mode、direction、rank、iteration、timing_boundary、latency_us、useful_flops、token_load、attention_work、payload_tx_bytes、dram_read/write_bytes、instrumentation、source_file。不同指标不适用时留空并解释，不填 0 冒充测量。

## 13. 现有代码入口与需要补的工作

以下相对路径均以本文开头的远端源码根目录为基准。

| 入口 | 可复用内容 | 首轮需要补充 |
|---|---|---|
| ring_test/homogeneous_all_cp_microbench/benchmark_ring_forward.py | 三 baseline、Mega All-CP 调用 | 四模式、step 关联、完整计时边界 |
| ring_test/homogeneous_all_cp_microbench/benchmark_ring_backward.py | BWD ring/all-gather/Mega | 后端命名、KV/dKV 分项、reset 计时 |
| ring_test/ring_common.py | Zigzag/P2P step | Compute-only 输入快照和诊断标记 |
| ring_test/allgather_attention.py | Head-chunk all-gather/BWD reduce-scatter | 对应通信/计算模式和实际 GPU trace |
| ring_test/benchmark_load_balance.py / forward_load_model.py | Placement 与静态负载 | 原始 manifest、分析量/实测量分离 |
| balancer/sampler.py | 长度分布驱动抽样 | 真实样本与桶抽样标签、padding 记录 |
| ring_test/benchmark_transformer_layer.py | Full-layer 和 self-attention 时间 | 后续增加 projection/core/MLP 分项 |
| scripts/benchmark_dcp_mega_arrival4_phase_graph.sh | 旧 workload 和参数来源 | 不直接沿用 Mega eager 与 baseline Graph 的不一致 |
| dcp_test/benchmark_dcp_varlen.py | Graph、全局 phase 汇总 | 统一边界、原始 trace 导出、单 case profiling 入口 |
| dcp_test/utils.py | CUDA events、capture/replay | 诊断开关、指定 replay 采集 |
| dcp_test/baselines.py | 同核 vLLM/SGLang 编排、pack/combine | 本地 CTA/SM 记录与 buffer 账本 |
| include/dcp_mega_min_fa3_varlen_launch.h | Mega 阶段调度与 globaltimer | 可关闭的 CTA/SM 开始/有效结束/退出记录 |
| dcp_mega_metadata.py | Task/split/queue 信息 | 导出实际配置；模型 makespan 不替代硬件时间 |

修改仅限 benchmark 支持和必要的可关闭诊断路径；不要为了 motivation 顺带重构核心算法或复制新的完整引擎。具体启动脚本必须在实现前检查工作目录与参数，本文不提供尚未实现的伪命令。

## 14. 论文 1–2 页如何呈现

实验工程文档可以详细，motivation 正文只保留三段观察：

1. 均匀输入下，分离通信/计算的重叠收益受干扰限制，step 切分还有独立计算代价。
2. 变长输入下，粗粒度均衡难以同时兼顾 token、实际 attention 执行和通信。
3. Decode 的 Graph 执行仍存在阶段尾部及中间数据交接成本，需要更贴近 tile 就绪的推进方式。

建议两张组合图：

- 图 1（训练/full-prefill）：重叠 slowdown 与完整调用时间为主面板；step 对照和负载三目标取紧凑子图。详细 FWD/BWD/长度结果留 evaluation，正文仍明确数据覆盖两方向。
- 图 2（decode）：同一代表 case 的逐 SM 阶段时间，加 decode-only/mixed 的 DRAM/L2 流量与正常延迟小图。另一类别的完整时间线可放附录。

如果版面接近 1 页，T2 用一句配对数字，T4 和全部强因果消融放 evaluation；不要靠缩小字体塞满所有曲线。若尚无可靠 NCU 数据，图 2 先只展示阶段时间，并在文本中将物化解释保留为待证实假设。

每段末尾回到“决策/依赖粒度与实际 tile 执行不匹配”，不展开 Mega 的调度数据结构。Weights prefetch 不进入首版主线。

## 15. 文档与证据完成检查

首轮完成需满足：

- [ ] T1 三基线 FWD/BWD 四模式，以及合法 Mega All-CP 完整调用时间。
- [ ] T2 一个合法 forward 分步/合并对照；backward 分步成本范围写清。
- [ ] T3 固定 batch 的三类负载和实际 attention 时间，未实现通信明确标注。
- [ ] D1 两类输入的 CTA/SM 阶段记录与无插桩 Graph 时间，图没有虚构通信 SM 轨迹。
- [ ] D2 至少代表点的可靠 NCU 内存层级数据；若未完成，正文不保留 HBM 强结论。
- [ ] 所有主结果有 correctness、配置、原始样本和可复现汇总。
- [ ] 原生系统、适配器、历史/新硬件、分析值/实测值清楚区分。
- [ ] 预期未被数据支持时收窄措辞，而不是选择性丢弃结果。

可后补：T4、D3、D4、q=1、原生引擎全链、多节点、所有参数扫描。暂缓这些项目不妨碍形成核心 motivation，但对应强结论必须同步暂缓。

参考工具文档：
- NVIDIA Nsight Compute Profiling Guide：https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- NVIDIA Nsight Compute CLI：https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- NVIDIA CUDA Graph：https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- NVIDIA NCCL buffer registration / transport 范围：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html
