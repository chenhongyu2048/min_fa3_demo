# Motivation v2

基于 `main` commit `5b0bf71240d4adf0168f9d3d991ff4ed952b08e1`，在
`codex/motivation-v2` 重建。旧 `motivation-lab` 仅作为 T1/T2 控制实验参考。
公共 evaluation 和生产 MegaRing 算法保持 main 实现；目录外改动限于
D1 可选 CTA trace 接口和 T2 ablation 的 CP4/CP8、compute-only 支持。

## 入口和固定配置

在仓库根目录、项目现有 Python 环境中执行：

```bash
python -m motivation.run --gpus 4 --dry-run
python -m motivation.run --gpus 8 --dry-run
python -m motivation.run --gpus 8
python -m motivation.run --gpus 4 --experiments T1,T2
```

入口用当前 `sys.executable -m torch.distributed.run` 启动单节点多卡进程。
选择恰好 4 或 8 张可见、同构 SM90 GPU；沿用项目已有环境、子模块和依赖。
`--dry-run` 不导入 torch、不创建结果目录，输出命令、输入 manifest 和配置。

| 实验 | 4 卡 | 8 卡 | 默认 warmup / 测量 |
|---|---|---|---|
| T1/T2 | 全局 131072 tokens，Q32/KV8/D128 | 相同 | 40 / 60 |
| T3 | Qwen3 MoE 单层，TP1/CP4/EP4 | TP1/CP8/EP8 | 10 / 40 |
| D1 | Q32/KV2/D128，TP4/DCP2，smoke | Q32/KV4/D128，TP8/DCP2，formal | 20 / 30 |

T1/T2 batch 为 1、2、4、8、16，local length 为 `131072 / (batch × GPU数)`。
四卡 local length 依次为 32768、16384、8192、4096、2048；八卡为其一半。
T1/T3 使用实际 SM 数减 8 作为计算 SM；D1 保留 4 个通信 SM；T2 compute-only
使用全部计算 SM、0 个通信 SM。默认 seed=0。

T3 使用公共 `generate_dataset_length_cases`，五个数据集 arxiv、github、pile、
freelaw、prolong 各 20 cases。131072 是公共 sampler 的目标预算：采样和对齐
可能让原始长度之和稍超过目标，本版本 seed=0 的范围为 131072–132864。
保留公共 sampler 输出，不随 GPU 数裁剪或重采样；manifest 分别记录
`target_tokens`、`raw_tokens`、`raw_lengths`。策略执行 padding 另记，
不覆盖原始长度。同一次 seed 的 4/8 卡原始 manifest 完全一致。

`--case-limit N` 显式选择 T1/T2 前 N 个 batch、T3 **每数据集**前 N 个 case、
D1 前 N 个候选；`--warmup`、`--iters` 显式缩短次数。这些参数不缩短输入。
GPU 数本身不会减少 case 数或迭代数。

## 八卡完整实验脚本

完成项目环境安装和 CUDA extension 构建后，在分配了八张完整 SM90 GPU 的
CUDA 节点上执行。脚本默认使用仓库 `.venv/bin/python`，继承当前
`CUDA_VISIBLE_DEVICES`；可用 `PYTHON=python` 指定已经激活的项目环境。

```bash
# 只查看完整流程，不加载 CUDA、不创建结果目录。
bash motivation/run_full_8gpu.sh --dry-run

# 正式运行：保留调度系统分配的 CUDA_VISIBLE_DEVICES。
bash motivation/run_full_8gpu.sh

# 也可以指定一个尚不存在的结果目录。
bash motivation/run_full_8gpu.sh --output-dir benchmark_logs/motivation_v2/formal_8gpu
```

如需手动选卡，在命令前设置 `CUDA_VISIBLE_DEVICES` 为本次分配的八张完整
GPU 的编号或 UUID。脚本不改写这个变量，也不负责申请节点或安装环境。
长任务建议在 CUDA 节点的 tmux 会话中运行。

脚本依次完成：

1. **T1/T2**：全部五档 batch、128K tokens，warmup 40、测量 60。
2. **T3**：五数据集各 20 cases、四种 placement，共 400 个整层 FWD/BWD
   组合，附带同布局静态指标；warmup 10、测量 40。
3. **D1 候选**：默认 10 cases、三种 Graph 方案，三个独立进程运行；
   全部关闭 trace，warmup 20、测量 30。
4. 分别生成上述各次运行的 `summary.csv` 和 `figures/`。
5. 使用三次候选结果选择 decode-only 两个、mixed 两个稳定获益案例。
6. 对选中四个案例重新进行无 trace 计时，再额外采集 trace 诊断，
   单独生成这次运行的汇总和图。

脚本不使用 `--case-limit`，不缩短 token 或采样次数。默认创建
`benchmark_logs/motivation_v2/full_8gpu_<UTC timestamp>/`，其中包含
`run.log`、`training/`、`d1_run_0/`、`d1_run_1/`、`d1_run_2/`、
`d1_selected.json` 和 `d1_selected_trace/`。每个运行子目录独立保存
配置、原始数据、汇总和图，不把候选三次测量及选例后的重测混在一起。

任一步骤失败时脚本立即以非零状态退出，已写入的数据保留。特别是某类不足
两个稳定获益候选时，D1 选择器会停止流程，不生成选例或继续 trace；
此时完整候选结果及其汇总、图仍然可用，需要先根据结果决定是否扩充候选。
脚本不自动续跑；不要把完整脚本重新指向已有目录。

## 实验口径

- **T1**：Ring、AllGather 各有 comm-only、compute-only、serial、overlap。
  Python 控制循环复用主分支 backend、相同 causal zigzag 划分及 merge 数学。
  compute-only 预加载完整 K/V；AllGather 保留本地拷贝和打包开销。
  测量前与 Ring 输出对照，MegaRing 保留完整调用。co-run 在独立流和宿主线程上
  测量通信/计算竞争，其分项是诊断，不能当作原 overlap 调用的临界路径分解。
- **T2**：`step_external_reduce`、`step_fused_reduce`、`linear_queue_recycle`。
  完整 K/V 预加载后计时，计时关闭统计，再单独 probe。检查三组输出一致，
  `qo_visits`、`kv_tile_reads` 一致。`compute_only` 追加于已有 binding 参数末尾，
  默认 false，已有八卡 ablation 的位置参数语义保留。T1/T2 仅 forward。
- **T3**：All-CP、BR-PBS hybrid、Megatron-adapted、Zeppelin-adapted 全部使用
  `mega_ring_hybrid` 执行器。公共 planner 的 buddy 映射决定实际长度、顺序和
  ring metadata；静态指标和整层执行共用这个布局。记录 sample ID、native/mapped
  group、执行 padding、token/work imbalance、forward CP payload 和 KV tile work。
  通信静态统计不含层内 MoE dispatch 或输入重分布。
  单层复用 main 的 Qwen3 配置（hidden=2048、Q32/KV4/D128、128 experts、top-8、
  expert FFN=768）、均衡路由、autograd 和计时工具。
  与 main 一致跳过每轮 projected K/V 复制；arena 在测量前置零一次，保证读取内容
  已定义。完整层 FWD/BWD、core attention 和其余层计算均报告 main 的
  **每次选 CUDA critical rank 后取均值**，不是 T1/T2 的 p50。
  这是受控性能实验，梯度存在性检查不等同于完整训练数值等价验证。
- **D1**：仅 attention，显式比较 critical-wave、FA3-native/FIFO、本地 vLLM A2A，
  三者都使用 CUDA Graph。关闭 trace 的完整 Graph 延迟用于选择；`--d1-trace`
  另外构建带诊断的 graph，采集少量 replay。A2A 使用同一个 critical rank 的
  CUDA event 分项，不伪造 SM 轨迹。每个 graph 在测量前重复 replay，并检查输出。

T1/T2/D1 性能时间为 CUDA events 的每次 rank-max 分布（p50/p90）。T3 输出
沿用主分支 timing 字段，汇总工具通过 `statistic` 明确区分均值与 p50。

## D1 候选与选择

`d1_candidates.json` 来自 main 的 `dcp_test/mega_dcp_trace_cases.jsonl`，目前含
3 个 decode-only Q16、7 个 mixed 候选，状态是 `pending_gpu_measurement`。
它们是候选输入，尚未声称在 DCP2 有收益。可转换更大的现有 trace workload：

```bash
python -m motivation.prepare_d1_cases --source-jsonl path/to/cases.jsonl \
  --output-json path/to/candidates.json
```

在 GPU 阶段使用同一代码、硬件、候选和测量配置进行三次独立进程运行：

```bash
for run in 0 1 2; do
  python -m motivation.run --gpus 8 --experiments D1 --d1-run-id "$run" \
    --output-dir "benchmark_logs/motivation_v2/d1_run_$run"
done
python -m motivation.select_d1_cases \
  benchmark_logs/motivation_v2/d1_run_0/D1/cases.jsonl \
  benchmark_logs/motivation_v2/d1_run_1/D1/cases.jsonl \
  benchmark_logs/motivation_v2/d1_run_2/D1/cases.jsonl \
  --output-json benchmark_logs/motivation_v2/d1_selected.json
```

候选必须在每次运行中 critical-wave p50 都严格优于 native。每组从稳定获益
候选中选两个，优先接近该组获益候选的中位加速比，平局按数值 case ID。
所有原始候选测量留在各 run 目录；选择文件也保留全部候选的三次比值。
不够两个稳定获益候选则选择失败、继续待测，不补凑。四卡 smoke 数据拒绝用于
正式选择。选定后用 `--d1-manifest` 和 `--d1-trace` 采集正式诊断。

CTA trace 每个 block 固定五个阶段槽位，每行
`[start_ns, end_ns, cta_id, sm_id, phase_id]`。仅 thread 0 读 timer、写记录；
关闭 trace 使用独立的编译期特化，不执行新增采样。通信 CTA 每次覆盖阶段 0/3，
计算 CTA 每次覆盖 1/2/4，角色由 runner 固定。buffer 构造时置零，未用槽保持零，
有效槽以 `end_ns > 0` 判断，因此 replay 不清零。诊断阶段复制连续两次 replay 的
记录，检查有效槽位集合不变、时间戳全部向前推进。不能复用到改变角色的 launch。
记录是 thread-0 的 CTA **阶段检查点区间**，包含等待和同步，不是每个 CTA 的
有效工作结束时间或有效工作量；图中不填入全局任务数。不同 rank 的时间轴独立。
尚未给出 trace 开销下降比例或目标，需后续 GPU 实测关闭/开启路径。

## 结果与分析

默认结果保存在新的 `benchmark_logs/motivation_v2/<UTC timestamp>/`；
指定 `--output-dir` 时也要求目录尚不存在，避免覆盖历史实验。
`run.json` 记录基线、实际 HEAD、dirty 状态、启动命令；`uniform.json` 和
`t3_cases.json` 保存输入。T1/T2 每个 batch 一个 JSON，T3 每数据集一个 JSONL，
D1 一个候选 JSONL。各记录包含实际硬件、配置、策略/执行器、布局及计时语义。

```bash
python -m motivation.summarize path/to/run --output path/to/summary.csv
python -m motivation.plot path/to/run --output-dir path/to/figures
```

汇总只接受 `motivation.v2.*`；原始 JSON 是完整记录，CSV 是分析视图。
绘图需要项目中的 matplotlib，注明实际硬件、拓扑和统计口径；T3 数据集柱状图
为各 case 的等权算术均值。D1 图标注 trace 状态、选择状态和规则。
不会套用历史 H20/DCP4 标签，也不会把旧 schema 自动合并。

单独生成 T3 静态记录需要真实 CPU torch（公共模型/布局模块导入依赖它），不需要 CUDA：

```bash
python -m motivation.transformer_layer --gpus 4 --static-only \
  --manifest path/to/t3_cases.json --dataset arxiv --output-jsonl path/to/static.jsonl
```

## 验证与后续 GPU 阶段

```bash
python -m unittest motivation.test_config motivation.test_results motivation.test_interfaces -v
python -m unittest balancer.test_balancer ring_test.load_balance_bench.test_topology \
  scripts.test_min_fa3.test_dcp_topology scripts.test_min_fa3.test_dcp_mega_batch
git diff --check
```

第一组测试包括真实 CPU 配置、原始 manifest、选择规则、汇总/绘图输入和源码 ABI
对比；实际布局和 T2 hierarchy 检查需要 torch，缺失会明确 skip，不替换成 mock。
源码 ABI 检查不代表 CUDA extension 已编译。

初始本地实现和静态检查记录见 `LOCAL_VALIDATION.md`。后续经用户 SSH 授权完成的
H20 四卡共享 GPU 功能验证见 `REMOTE_VALIDATION.md`；这些 smoke 延迟不用于性能结论。
在可用 CUDA 节点沿用 `setup_fresh_environment.sh` 与 Makefile 构建后，可先执行：

```bash
python -m motivation.run --gpus 4 --case-limit 1 --warmup 1 --iters 2 --d1-trace
```

这仍使用完整长度，T3 为五数据集各一个 case、四种策略，D1 为 KVH2 smoke。
随后做八卡、三次 D1 候选测试与关闭/开启 trace 的实际开销测量，再冻结选例。
