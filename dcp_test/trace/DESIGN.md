# Mega DCP Trace Workload 设计

本文说明 `dcp_test/trace` 如何把公开的 Mooncake/Kimi conversation trace
转换为可直接驱动 Mega DCP packed-varlen benchmark 的 workload case。内容以
当前代码为准，覆盖前期调研结论、建模选择、实现结构、数据契约、benchmark
集成和使用方法。

## 1. 目标与边界

工具需要从真实请求 trace 中构造一组确定性的、可复现的 Mega DCP 输入：

- `batch_size`：当前 scheduler step 中可以进入 Mega DCP 的请求数；
- `q_lens`：各请求本次调度的 query 或 chunk 长度；
- `history_lens`：本次调度开始前已经存在的 KV history 长度；
- `total_kv_lens`：逐请求满足 `history_lens + q_lens`；
- 请求 ID、prefill/decode phase、prefix cache 和 MTP 状态等诊断信息。

这不是完整的在线 serving simulator。它是一个 CPU-only 的固定步长重放器，
用于生成 attention benchmark 输入。生成 workload 时不会初始化 CUDA，也不会
调用或修改 CUDA kernel。真正的 GPU 测试由
`dcp_test.benchmark_dcp_mega_batch` 和
`scripts/test_dcp/benchmark_dcp_mega_trace.sh` 完成。

实现保持以下边界：

- 只使用 trace 明确提供的 prompt block hash，不推断 output token hash；
- 不声称示例 scheduler、cache 或 MTP 参数等同于 Kimi 线上配置；
- 不模拟多实例路由、preemption、KV swap、网络传输和 active KV 内存压力；
- 采样对象是 Mega-eligible 的 scheduler step，不包含空闲时刻和完全不满足
  Mega DCP 条件的时刻。

## 2. 调研依据

以下资料在实现前完成调研，原始访问日期为 2026-08-05。

| 资料 | 与本实现的关系 | 来源 |
| --- | --- | --- |
| Mooncake FAST'25 | disaggregated serving 系统及 Kimi 生产背景 | https://www.usenix.org/conference/fast25/presentation/qin |
| Mooncake FAST'25 release | 本工具使用的公开 conversation trace | https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release |
| vLLM MLA DCP | decode context parallel 实现参考 | https://github.com/vllm-project/vllm/pull/23734 |
| vLLM GQA DCP | GQA decode context parallel 实现参考 | https://github.com/vllm-project/vllm/pull/24864 |
| vLLM packed A2A | 后续 DCP 通信实现参考 | https://github.com/vllm-project/vllm/pull/41160 |
| Sarathi-Serve | chunked prefill 及 decode/prefill 共存调度 | https://arxiv.org/abs/2403.02310 |
| LoongServe | 长上下文 serving 和弹性 sequence parallelism | https://arxiv.org/abs/2404.09526 |
| Infinite-LLM | 分布式长上下文 serving | https://arxiv.org/abs/2401.02669 |
| Vidur | trace-driven LLM serving simulation | https://github.com/microsoft/vidur |
| Azure LLM 2024 | 可替代的公开 inference workload | https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md |
| BurstGPT v2.0 | 带 session 信息的可替代 workload | https://github.com/HPMLL/BurstGPT/releases/tag/v2.0 |

### 2.1 Mooncake trace 的已确认事实

上游文件为 `FAST25-release/traces/conversation_trace.jsonl`。仓库中的
`conversation_trace.jsonl` 是该输入，`example_config.json` 使用以下 SHA
校验其来源：

- Git blob SHA：`5a371794b81d7fc7487ec8a02a2c631bcc7fbdf8`
- 文件 SHA-256：
  `b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df`
- 每行字段：`timestamp`、`input_length`、`output_length`、`hash_ids`
- 每条已检查记录满足
  `len(hash_ids) == ceil(input_length / 512)`

`timestamp` 的单位是从 trace 内容和总时长共同确认的毫秒。最后一个时间戳为
3,536,999，对应约 3,537 秒，而不是 3,536,999 秒。早期按秒解释的 smoke
replay 只能接纳 12,031 条请求中的 15 条，这一反例促成了当前代码中的
`timestamp * 1_000` 微秒转换。

对上述精确文件做出的本地统计如下。这些是离线分析结果，不是每行 trace
直接提供的字段：

- 3,537 秒内共 12,031 个请求，平均约 3.40 request/s；
- input length 均值 12,035，P50/P90/P99 为
  6,909/27,367/85,400，最大值 126,195；
- output length 均值 342.62，P50/P90/P99 为
  350/597/1,118.5，最大值 2,000；
- 时间戳通常聚合在约 3 秒的 bucket 中；
- 每个 timestamp 的请求数 P50 为 10，P90 为 15，最大值为 28；
- 假设无限 cache，并且只统计此前出现过的、从 prompt 起点连续命中的完整
  512-token block，推导出的 token hit ratio 约为 37.34%。

最后一个命中率不能当作线上 prefix-cache 指标。它没有包含有限容量、路由、
active block pinning 或 output KV。

### 2.2 为什么没有混合 BurstGPT

BurstGPT v2.0 中只有 `BurstGPT_3.csv` 同时包含 `Session ID` 和
`Elapsed time`。同一个 session 内的 `Request tokens` 可能下降，可能来自
截断、编辑或模型切换。因此 session ID 本身不足以恢复单调增长的 conversation
KV 状态，实现不会从 BurstGPT 推断 prefix history，也不会把它的分布混入
Mooncake workload。

## 3. 总体数据流

```text
example_config.json
        |
        v
严格配置校验 + effective config SHA
        |
        v
Mooncake SHA 校验、JSONL 解析、arrival 归一化
        |
        v
512-token prefix LRU + fixed-step scheduler replay
        |
        v
history_len >= dcp_size 过滤 + reservoir sampling
        |
        v
mega_dcp_workload/v2 JSONL
        |
        v
batch frontend 在 CUDA 初始化前校验 provenance 和 shape
        |
        v
每种模式一次 8-rank torchrun，进程组内顺序执行所有 case
        |
        v
per-case JSON + manifest + workload-weighted summary
```

配置和 trace case 之间通过两个哈希绑定：

- `trace_sha256` 标识输入 trace 的精确字节内容；
- `config_sha256` 标识 canonical JSON 形式的有效 replay 配置。

`--num-cases` 会先覆盖配置中的 `num_cases`，再计算 `config_sha256`。因此改变
case 数量会改变 provenance，也会重新运行 reservoir sampling。它不是对旧
JSONL 做前缀截断。

## 4. Replay 设计

### 4.1 配置加载和随机数流

`models.py` 使用严格字段集合加载 JSON 配置：缺字段、未知字段、错误类型、
非法范围或 PMF 不一致都会直接失败。相对 `trace_path` 以配置文件所在目录为
基准解析。

所有随机行为都由一个 `seed` 派生，但使用互相独立的 SHA-256 RNG stream：

- `arrival`：timestamp bucket jitter；
- `acceptance`：MTP accepted draft 采样；
- `sampling`：scheduler step 的 reservoir sampling。

独立 stream 避免某一阶段增加一次随机调用后，意外改变其他阶段的结果。
相同 trace、有效配置和 seed 会生成逐字节一致的 JSONL。

### 4.2 Trace 解析和 arrival time

加载顺序如下：

1. 在解析任何 row 之前计算完整文件 SHA-256，并与配置比较。
2. 按 JSONL 的零基 line index 分配稳定 `request_id`。
3. 校验 timestamp 非负且非递减，长度为整数，hash 数量与 512-token block
   数量一致。
4. 将源毫秒时间戳用 decimal arithmetic 转换为整数微秒。
5. 对 `input_length + output_length > max_model_len` 的请求计数并丢弃。
6. 以 `(arrival_us, request_id)` 排序后送入 FCFS waiting queue。

支持两种 timestamp policy：

- `preserve`：保留相同 timestamp 的原始 bucket；
- `uniform_bucket_jitter`：用确定性随机数把 bucket 内请求均匀分散到当前
  timestamp 与下一个不同 timestamp 之间。最后一个 bucket 使用所有正 bucket
  宽度的中位数作为 fallback。

归一化后先减去最早 arrival，再计算：

```text
arrival_us = round((timestamp_us - origin_us) / arrival_time_scale)
```

所以 `arrival_time_scale > 1` 会压缩时间轴并提高 offered load。

### 4.3 Prefix cache

`PrefixBlockCache` 模拟一个全局、有限容量的 LRU，block size 固定为 512
token。它只接受从 prompt 第一个 block 开始的连续命中，遇到第一个 miss 后
停止，不允许中间跳过 block。

为了保证一次 query 至少计算一个 token，可命中的完整 block 数上限为：

```text
max_cached_blocks = floor((input_length - 1) / 512)
```

这意味着 input length 恰好是 512 的整数倍时，最后一个完整 block 也不会被
全部视为 cached。只有 replay 实际计算完一个完整 prompt block 后，它才会被
insert 或 touch。Cache 不存 output token，也不表示 active-request KV 容量或
request admission 容量。

### 4.4 请求状态和 admission

每个 active request 保存：

- `cached_prefix_length`：admission 时的 prefix hit，之后不再变化；
- `computed_prompt_tokens`：已完成的 prompt token 数；
- `history_len`：当前已提交进 KV history 的 token 数；
- `generated_tokens`：已经提交的 output token 数。

Replay 从时间零开始，以 `fixed_step_us` 为间隔推进一个逻辑 DCP instance。
arrival request 进入 waiting queue，随后以 FCFS 顺序 admission，直到 active
request 数达到 `max_num_seqs`。当 active 和 waiting 都为空时，实现会直接跳到
下一个 arrival 所在的 fixed step，同时累计被跳过的 idle step。

### 4.5 Decode-first 调度和 token budget

每个 step 共享 `max_num_batched_tokens` 物理 query token budget，并最多为每个
request 调度一次 query。调度分两遍完成：

1. 按 active FCFS 顺序调度所有可 decode/MTP 的 request；
2. 使用剩余 budget 按 active FCFS 顺序调度 chunk prefill。

Replay 区分逻辑 query 长度和用于性能 workload 的物理长度：

```text
logical_q_len = min(prompt_remaining,
                    prefill_chunk_size,
                    aligned_remaining_budget)
physical_q_len = align_up(logical_q_len, q_len_alignment)
```

budget 扣减 `physical_q_len`，因此对齐 padding 不会让物理 batch 超过配置上限；
prompt progress、prefix cache、history state 和 decode commit 只使用
`logical_q_len`。`q_len_alignment=1` 保持原始 replay 行为，示例性能配置使用 8。

Decode/MTP query 是 atomic 的。如果对齐后的完整物理 query 放不进剩余 budget，
就停止 decode pass，不把它缩短。两种 `scheduled_q_rule` 定义逻辑长度：

- `target_plus_drafts`：`q_len = 1 + num_speculative_tokens`；
- `drafts_only`：`q_len = num_speculative_tokens`，这里配置值表示验证 query
  的总长度。

`accepted_draft_pmf[i]` 表示本步接收 `i` 个 draft 的概率：

- `target_plus_drafts` 需要 `num_speculative_tokens + 1` 项；
- `drafts_only` 需要 `num_speculative_tokens` 项，因为最多只能提交
  `q_len - 1` 个 draft。

在接近 output-length 边界时，scheduled `q_len` 保持不变，但实际提交数会被
限制为：

```text
committed = 1 + min(sampled_accepted,
                    remaining_outputs - 1,
                    q_len - 1)
```

Chunk prefill 完成整个 prompt 且 `output_length > 0` 时，首个生成 token 视为
已经产生，`generated_tokens` 置为 1。所有 snapshot 都记录状态转换之前的
`history_len` 和 `generated_tokens_before`。

### 4.6 Mega DCP eligibility

Scheduler 会正常执行该 step 的所有 query，但输出 snapshot 只保留：

```text
history_len >= dcp_size
```

Mega DCP 使用 token-interleaved history shard。该条件保证每个 DCP rank 至少
分到一个 history token，避免 rank-local history 为空。一个 mixed step 中不
满足条件的 query 仍然消耗全局 token budget、推进请求状态并影响后续 replay，
只是不会出现在 Mega case 中。因此 case 的 `batch_size` 可能小于同一步实际
调度的 request 数。

没有 eligible query 的 step 只进入统计，不产生 `B=0` case。

### 4.7 Reservoir sampling

Replay 会从时间零开始恢复 cache 和 request 状态，但只在半开区间
`[sampling_start_ms, sampling_end_ms)` 内采样。每个包含至少一个 eligible
query 的 fixed-width step 是一个候选样本。

实现使用单遍 reservoir sampling，对所有候选 step 做无放回均匀采样，空间
复杂度为 `O(num_cases)`，不需要保存全部候选。Replay 结束后按
`(sampled_time_us, scheduler_step)` 排序，再依次生成稳定的
`case_000000`、`case_000001` 等 ID。

如果窗口内 eligible step 少于请求的 `num_cases`，生成过程会失败，不重复
case，也不静默减少输出数量。

## 5. 配置契约

`example_config.json` 提供完整示例。所有字段都是必需字段，不能添加未知字段。

| 字段 | 语义 |
| --- | --- |
| `trace_path` | Mooncake JSONL 路径，相对路径以 config 目录为基准 |
| `trace_sha256` | 预期输入文件 SHA-256 |
| `timestamp_policy` | `preserve` 或 `uniform_bucket_jitter` |
| `arrival_time_scale` | 正数，arrival elapsed time 的除数 |
| `sampling_start_ms` | 采样窗口起点，包含 |
| `sampling_end_ms` | 采样窗口终点，不包含 |
| `fixed_step_us` | scheduler 固定步长 |
| `num_cases` | reservoir 最终保留的 case 数 |
| `seed` | 所有独立 RNG stream 的根 seed |
| `max_num_seqs` | active request 上限 |
| `max_num_batched_tokens` | 每个 scheduler step 的共享 query token budget |
| `prefill_chunk_size` | 单个 request 每步最大 prefill chunk |
| `q_len_alignment` | 物理 benchmark query 对齐，严格限制为 1 或 8 |
| `max_model_len` | 允许的 `input_length + output_length` 上限 |
| `dcp_size` | Mega eligibility 和 benchmark topology，限制为 2、4 或 8 |
| `prefix_cache_capacity_blocks` | 全局 512-token prefix LRU 容量，0 表示关闭 |
| `num_speculative_tokens` | MTP draft 数或 `drafts_only` 的 query 总长度 |
| `scheduled_q_rule` | `target_plus_drafts` 或 `drafts_only` |
| `accepted_draft_pmf` | accepted draft count 的离散概率分布 |

示例配置里的 cache capacity、scheduler limits、fixed step、arrival scaling 和
MTP acceptance 都是研究用输入。公开 Mooncake trace 没有提供这些 serving
参数，不能把示例值表述成 Kimi production default。

## 6. JSONL 输出契约

Schema version 为 `mega_dcp_workload/v2`。一行是一个 scheduler snapshot，
字段如下：

| 字段 | 内容 |
| --- | --- |
| `schema_version` | 固定为 `mega_dcp_workload/v2` |
| `case_id` | 按采样时间排序后分配的安全、唯一 ID |
| `source` | 固定为 `mooncake_kimi_conversation_fast25` |
| `trace_sha256` | 已验证输入 trace 的 SHA-256 |
| `config_sha256` | 含 `--num-cases` override 的有效配置 SHA-256 |
| `sampled_time_us` | snapshot 的 replay wall-clock time |
| `scheduler_step` | `sampled_time_us / fixed_step_us` 对应的 step |
| `batch_size` | 此 case 保留的 Mega-eligible query 数 |
| `q_lens` | 对齐后的物理 query/chunk 长度数组 |
| `logical_q_lens` | 推进 replay 状态的逻辑 query/chunk 长度数组 |
| `history_lens` | 状态转换前的 KV history 长度数组 |
| `total_kv_lens` | `history_lens + q_lens` |
| `request_ids` | 源 JSONL line index |
| `phases` | 每个 query 的 `decode` 或 `chunk_prefill` |
| `prompt_lengths` | 源 request input length |
| `output_lengths` | 源 request output length |
| `cached_prefix_lengths` | admission 时命中的 prefix token 数 |
| `generated_tokens_before` | 本 step 之前已提交的 output token 数 |
| `num_speculative_tokens` | replay 配置中的 MTP 参数 |
| `q_len_alignment` | 该 case 使用的物理 query 对齐 |
| `accepted_drafts` | decode 为整数，chunk prefill 为 `null` |
| `fixed_step_us` | replay fixed step |
| `timestamp_policy` | arrival normalization policy |
| `seed` | replay root seed |

必须始终满足：

```text
batch_size == len(q_lens)
           == len(logical_q_lens)
           == len(history_lens)
           == len(total_kv_lens)
           == len(request_ids)
           == len(phases)

total_kv_lens[i] == history_lens[i] + q_lens[i]
q_lens[i] % q_len_alignment == 0
logical_q_lens[i] <= q_lens[i]
q_lens[i] - logical_q_lens[i] < q_len_alignment
history_lens[i] >= dcp_size
```

Generator 先在目标目录写临时文件、flush 并 `fsync`，再用 `os.replace`
原子发布。默认拒绝覆盖已有文件，只有 `--force` 会替换它。

## 7. 实现结构

| 文件 | 职责 |
| --- | --- |
| `models.py` | strict config、dataclass、schema 常量、canonical config SHA、独立 RNG stream |
| `mooncake.py` | trace SHA 和 row 校验、timestamp normalization、model-length filtering、prefix LRU |
| `replay.py` | request state、admission、decode-first scheduling、MTP commit、eligibility、reservoir sampling、case 构造 |
| `generate.py` | workload CLI、覆盖保护和原子 JSONL 写入 |
| `show_example.py` | 对已有 workload 做确定性无放回抽样并显示，不重新 replay |
| `example_config.json` | 可运行但非 production 声明的示例 replay 参数 |
| `tests/test_trace_workload.py` | config、parser、cache、scheduler、MTP、sampling、I/O 和 viewer 的 CPU 测试 |
| `../benchmark_dcp_mega_batch.py` | 多 case benchmark frontend、trace 校验、进程组复用、manifest 和加权汇总 |
| `../../scripts/test_dcp/benchmark_dcp_mega_trace.sh` | 生成或复用 trace，并按 execution mode 各启动一次 `torchrun` |

`show_example.py` 的 display seed 只决定展示已有 JSONL 中的哪些 case，不影响
replay seed，也不修改输入文件。`--full` 关闭 focused field projection，打印
完整 source case。

## 8. Benchmark 集成

### 8.1 CUDA 初始化前校验

`dcp_test.benchmark_dcp_mega_batch` 在 trace 模式下同时要求
`--trace-config` 和 `--trace-cases`，并在初始化 CUDA 前完成：

- JSONL 可读、每行是 JSON object；
- schema version 正确；
- case ID 可安全用于文件名、没有重复；
- JSONL case 数精确等于有效配置中的 `num_cases`；
- `trace_sha256` 和 `config_sha256` 与有效配置一致；
- `batch_size` 与四个 length array 对齐；
- `total_kv_lens == history_lens + q_lens`；
- `q_lens` 是 `logical_q_lens` 按配置生成的最小对齐物理长度；
- 每个 `history_len >= dcp_size`；
- 最终选择的 benchmark topology 恰好等于 replay 的 DCP size。

Trace case 中除四个 kernel shape 字段外的元数据会保留到 manifest 的 per-case
`trace` 字段，便于从性能结果追溯到 scheduler step 和源 request。

### 8.2 一次 launch 执行多组 case

Wrapper 的默认 world size 为 8。每个 execution mode 只启动一次 distributed
job，并在同一个 process group 中依次执行全部 trace case：

- eager mode：一次 8-rank `torchrun`；
- CUDA Graph mode：另一次 8-rank `torchrun`。

Batch frontend 只初始化一次 distributed world，并缓存所需 DCP subgroup。
因此 `NUM_CASES=20 MODES=eager,graph` 是两次 launch，而不是 40 次 launch。

### 8.3 Workload-weighted 汇总

Manifest 按 topology 和 method 分组，不会把 DCP=2、4、8 混在一个聚合值里。
每个 case 先用所有 TP rank 的最大 CUDA Event latency 形成 global latency，再
对 iteration 求 case p50。最终吞吐按总工作量除以总时间计算：

```text
weighted_tflops
  = sum(global_effective_flops) / sum(case_p50_latency_seconds)
  = sum(case_tflops * case_p50_ms) / sum(case_p50_ms)
```

这与 ring dataset benchmark 的 total-work-over-total-time 定义一致。每个 method
还记录：

- case count；
- case p50 和 p90 latency 的 min/mean/p50/max；
- case effective TFLOPS 的算术均值；
- workload-weighted aggregate TFLOPS 和 per-GPU TFLOPS；
- total effective FLOPs 和 total p50 latency。

`BASELINE_PHASE_TIMING=0` 只关闭 non-Mega baseline 的 CUDA Event 分解计时，
仍保留 attention start/end Event 和端到端 latency。Mega 内部 `%globaltimer`
milestone 由独立的 `MEGA_PHASE_TIMESTAMPS` 控制。默认 Mega 只属于 eager batch；
graph batch 只运行 `ours,vllm,sglang`，并固定关闭该 Mega 选项。

## 9. 使用方法

所有命令都从仓库根目录运行：

```bash
cd /home/hychen/min_fa3_demo
```

### 9.1 只生成 workload

使用配置中的 `num_cases`：

```bash
python -m dcp_test.trace.generate \
  --config dcp_test/trace/example_config.json \
  --output ./mega_dcp_trace_cases.jsonl
```

在计算有效配置 SHA 和 reservoir sampling 之前覆盖 case 数量：

```bash
python -m dcp_test.trace.generate \
  --config dcp_test/trace/example_config.json \
  --output ./mega_dcp_trace_cases.jsonl \
  --num-cases 20
```

如果目标已存在并且确认需要替换：

```bash
python -m dcp_test.trace.generate \
  --config dcp_test/trace/example_config.json \
  --output ./mega_dcp_trace_cases.jsonl \
  --num-cases 20 \
  --force
```

Generator 的 replay summary 以单行 JSON 写到 stderr，其中包含输出路径、
有效 case 数、`config_sha256` 和完整 replay counters。

### 9.2 查看生成的 case

使用默认 display seed 确定性展示 5 个 focused example：

```bash
python -m dcp_test.trace.show_example \
  --input ./mega_dcp_trace_cases.jsonl
```

指定数量和 display seed：

```bash
python -m dcp_test.trace.show_example \
  --input ./mega_dcp_trace_cases.jsonl \
  --num-examples 8 \
  --seed 1234
```

打印完整 JSON object：

```bash
python -m dcp_test.trace.show_example \
  --input ./mega_dcp_trace_cases.jsonl \
  --num-examples 3 \
  --seed 1234 \
  --full
```

选择是无放回的。请求数量大于文件中的 case 数时，每个 case 最多输出一次，
不会失败。

### 9.3 完整两步用例：当前目录生成，然后批量测试

```bash
cd /home/hychen/min_fa3_demo

NUM_CASES=20
TRACE_CASES="$PWD/mega_dcp_trace_cases.jsonl"

python -m dcp_test.trace.generate \
  --config dcp_test/trace/example_config.json \
  --output "$TRACE_CASES" \
  --num-cases "$NUM_CASES"

GENERATE_TRACE=0 \
TRACE_CONFIG=dcp_test/trace/example_config.json \
TRACE_CASES="$TRACE_CASES" \
NUM_CASES="$NUM_CASES" \
MODES=eager,graph \
WARMUP=10 \
ITERS=40 \
CHECK=0 \
BASELINE_PHASE_TIMING=0 \
./scripts/test_dcp/benchmark_dcp_mega_trace.sh
```

`GENERATE_TRACE=0` 要求 `TRACE_CASES` 已存在。Shell wrapper 会在 `torchrun`
前检查文件，Python frontend 随后校验精确 case 数、有效 config SHA 和全部 shape
invariant。复用 JSONL 时，`TRACE_CONFIG`、`NUM_CASES` 或配置内容不能与生成时
不一致。

### 9.4 一条命令生成并测试

默认 `GENERATE_TRACE=1`。下面的命令会把 JSONL 写到当前目录，再运行 eager
和 graph 两个 batch：

```bash
GENERATE_TRACE=1 \
TRACE_CASES="$PWD/mega_dcp_trace_cases.jsonl" \
NUM_CASES=20 \
MODES=eager,graph \
./scripts/test_dcp/benchmark_dcp_mega_trace.sh
```

此模式准确执行：

```bash
python -m dcp_test.trace.generate \
  --config "$TRACE_CONFIG" \
  --output "$TRACE_CASES" \
  --num-cases "$NUM_CASES"
```

然后每个选中的 mode 各执行一次 8-rank `torchrun`。如果输出已存在，generator
会保护性失败；设置 `FORCE_TRACE=1` 才会向生成命令追加 `--force`。

### 9.5 Wrapper 参数

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TRACE_CONFIG` | `dcp_test/trace/example_config.json` | replay config |
| `TRACE_CASES` | `$RESULT_DIR/trace_cases.jsonl` | 生成或复用的 workload JSONL |
| `NUM_CASES` | `20` | 有效 replay case 数 |
| `GENERATE_TRACE` | `1` | 1 生成，0 复用已有 JSONL |
| `FORCE_TRACE` | `0` | 1 允许覆盖已有 JSONL |
| `MODES` | `eager,graph` | execution mode 列表 |
| `EAGER_IMPLEMENTATIONS` | `mega,ours,vllm,sglang` | eager method 集合 |
| `GRAPH_IMPLEMENTATIONS` | `ours,vllm,sglang` | graph baseline 集合，不包含 Mega |
| `WARMUP` | `10` | 每个 case 的 warmup 次数 |
| `ITERS` | `40` | 每个 case 的测量次数 |
| `CHECK` | `0` | 是否做 correctness precheck |
| `BASELINE_PHASE_TIMING` | `0` | non-Mega CUDA Event 分解计时 |
| `MEGA_PHASE_TIMESTAMPS` | `0` | eager Mega 内核 milestone |
| `NUM_SPLITS` | `0` | baseline split 参数 |
| `MEGA_BLOCK_N` | `auto` | 默认 NoSplit 用 176、critical-wave split=2/4 用 128；也可固定 128 或 176 |
| `MEGA_NUM_COMM_SM` | `8` | Mega communication CTA/SM budget |
| `LOG_DIR` | 带时间戳的 `benchmark_logs/dcp_mega_trace_*` | 日志根目录 |
| `RESULT_DIR` | `$LOG_DIR/results` | case JSON、trace JSONL 和 manifest 目录 |
| `DRY_RUN` | `0` | 1 只打印 generator 和 launcher 命令 |

只跑一种模式可以设置 `MODES=eager` 或 `MODES=graph`。检查最终展开命令而不
启动 replay 或 GPU benchmark：

```bash
DRY_RUN=1 NUM_CASES=20 MODES=eager ./scripts/test_dcp/benchmark_dcp_mega_trace.sh
```

## 10. 验证结果和回归范围

原始实现和后续 benchmark 集成完成过以下验证：

- `python -m compileall -q dcp_test/trace` 通过；
- consolidated trace CPU suite 覆盖 strict config、trace schema/SHA、两种
  timestamp policy、model-length filtering、prefix LRU、chunk-to-decode 状态、
  MTP atomicity、mixed phase、deterministic reservoir、sampling shortfall、
  1/8-token query alignment、逻辑/物理状态分离、atomic output、overwrite
  protection 和 example viewer；
- 纠正毫秒单位后的原始 64-case full-Mooncake smoke 在约 4.53 秒内完成，
  12,031/12,031 个请求均 admission 并完成；该次独立 smoke 配置产生
  1,240,338 个 sampling-window eligible step；
- 该 64-case 历史样本的 batch size min/P50/P90/max 为 1/1/2/3，76 个 query
  中 phase 计数为 72 decode/MTP 和 4 chunk prefill；这些数字只描述当时的
  illustrative smoke 配置，不代表当前示例配置或 Kimi production；
- trace benchmark 集成用真实 trace 和 `--num-cases 2` 验证过 JSONL 生成与
  `--print-cases` 加载，当次 replay 找到 162,125 个 eligible step，并正确选择
  `DCP=4/Hkv=2`；
- 8 张 H100 上分别完成 eager 和 CUDA Graph 的 2-case smoke，两个 manifest
  的 completion、trace provenance、case count、per-case metadata 和 weighted
  throughput 均做过机器复算；smoke 只有 2 次 iteration 且关闭 correctness，
  不能作为正式性能数据；
- `scripts/test_dcp/simple_test.sh --static-only` 覆盖 Python compile、CLI
  import/help、shell syntax、wrapper dry-run、whitespace 和 public API checks。

修改 replay 时，最低限度应重新运行：

```bash
python -m unittest dcp_test.trace.tests.test_trace_workload -v
./scripts/test_dcp/simple_test.sh --static-only
```

涉及 batch validation 或 weighted summary 时，还应运行：

```bash
python -m unittest scripts.test_min_fa3.test_dcp_mega_batch -v
```

真实 GPU 性能验证必须在支持的 8-GPU Hopper 环境中进行。

## 11. 已知建模限制

- `fixed_step_us` 是 scheduler 建模单位，不是实测 kernel duration；
- 只重放一个逻辑 DCP instance，不模拟多实例负载均衡；
- 不模拟 preemption、request migration、KV swap、network transfer 或 active
  KV memory pressure；
- Mooncake hash 只能确定 prompt block identity，不能确定生成 token identity；
- prefix cache 是可复用 prompt block 的全局 LRU，不是完整的 KV allocator；
- decode-first、token budget、MTP PMF 和 cache capacity 来自明确配置，不来自
  公开 trace；
- `q_len_alignment=8` 产生的是 synthetic performance padding；padding 参与物理
  attention shape 和 token budget，但不表示模型提交了额外 token；
- 样本条件是当前 step 至少有一个 Mega-eligible query，因此不能将 case 分布
  解读为包含 idle time 的无条件 wall-clock 分布；
- workload-weighted summary 汇总 attention case，不包含真实 serving 的 queueing
  delay、CPU 调度、模型其他算子和端到端 request latency。

## 12. 维护约束

- 改变 JSONL 字段含义或 invariant 时，应升级 `SCHEMA_VERSION`，同步 generator、
  viewer、batch loader 和 tests；
- 改变有效配置字段时，应保持 strict validation，并确认 canonical hash 能区分
  行为不同的 replay；
- 不要把 `--num-cases` 实现为生成后的 truncation，它必须继续在 replay 和
  config SHA 之前生效；
- 新增随机行为时应创建独立的 named RNG stream，避免破坏现有 stream 的调用
  序列；
- 任何被 Mega eligibility 过滤的 query 仍必须参与 scheduler budget 和状态
  转换，否则后续 sampled state 会失真；
- 文档中的 replay 统计必须标注所用配置，不得把 illustrative result 表述为
  production workload 特征。
