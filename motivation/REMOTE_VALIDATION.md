# H20 四卡功能验证 — 2026-09-16

本次为**共享 GPU 上的功能 smoke**。用户允许共用物理 GPU 0、1、3、5，
明确不据此报告性能。下述原始记录中的延迟只用于检查采集链路，不用于加速比、
trace 开销优化结论或正式案例选择。

## 同步与环境

- 登录方式：`ssh H20-HKUSTGZ`，再 `ssh h20`；CUDA 主机名 `H20-2`。
- 用户指定共享仓库：`/home/LOCAL/shixuan/hongyu/tmp/test/min_fa3_demo`。
- 分支：`codex/motivation-v2`，通过 Git bundle 从本地同步，未向 GitHub push。
- 基线：`5b0bf71240d4adf0168f9d3d991ff4ed952b08e1`。
- 完整 smoke 使用 `da1ffd9b`；新增 trace 新鲜度检查使用 `45d89b6c`。
  两者之间仅修改 D1 诊断验证和对应说明，CUDA 实现未改变。
- 使用现有 `.venv`：Python 3.12.13、torch 2.11.0+cu128、CUDA toolkit 12.8、
  driver 570.133.20、Transformer Engine 2.17.1+4329ff84、matplotlib 3.11.0。
- 四张完整 H20 均为 78 SM。GPU 4、7 启用 MIG，因此通过物理卡 UUID 设置
  `CUDA_VISIBLE_DEVICES`，避免 CUDA 枚举编号与物理编号混淆。
- 主实验 SM 分配：T1/T3 为 70:8，T2 为 78:0，D1 为 74:4。

最初发现的外层 `~/hongyu/min_fa3_demo` 已恢复原 `main`，其既有 `results/`
未改动。后续构建、smoke 和产物均使用用户指定的测试仓库。

## 构建与测试

构建成功，8 个需要更新的 CUDA 编译单元全部完成，扩展成功链接并原地更新：

```bash
MAX_JOBS=8 NVCC_THREADS=2 CUDA_HOME=/usr/local/cuda-12.8 make PYTHON=.venv/bin/python
```

| 验证 | 结果 |
|---|---|
| motivation 针对性 CPU 测试 | 17/17 通过，无 skip |
| T3 全量静态布局 | CP4/CP8 × 五数据集 × 20 cases × 四策略，800 条通过 |
| 静态记录复核 | raw manifest 一致；执行 tokens、padding、rank token 总量一致 |
| T1 四卡 smoke | B1、131072 tokens；四模式、co-run、MegaRing 完整调用通过，计算输出对照通过 |
| T2 四卡 smoke | 同一 B1 输入；三 profile 输出对照及 Q/O visits、KV tile reads 一致性检查通过 |
| T3 四卡 smoke | 五数据集各第一个完整 case × 四策略，共 20 个整层 FWD/BWD 组合通过 |
| T3 梯度与计时 | main 的梯度存在性检查通过；FWD+BWD 合计恒等式、critical-rank 计数通过 |
| D1 mixed/decode-only | Q32/KV2/D128、TP4/DCP2；三种 Graph 路径重复 replay 和输出对照通过 |
| D1 CTA trace | 两种 Mega 调度器、两类 workload、全部四个 rank 均通过覆盖和新鲜度检查 |
| 正式选择隔离 | 实际四卡 smoke 记录被正式案例选择器拒绝 |
| 汇总/绘图链路 | 主 smoke 生成 261 行指标与 27 张图；渲染用途仅为管道检查 |

远端真实 torch 检查发现原布局单测使用了未经 sampler 对齐的长度。测试输入已改为
遵循公共 sampler 约束、同时保留额外 placement padding 的长度；未放宽公共校验，
未改动 planner 或模型实现。

同时运行现有四组 CPU 测试：`balancer.test_balancer`、
`ring_test.load_balance_bench.test_topology`、`scripts.test_min_fa3.test_dcp_topology`、
`scripts.test_min_fa3.test_dcp_mega_batch`。共 88 个测试，87 个通过；
`test_tp2_and_tp4_example_configs_load` 的两个子案例报错，因为仓库缺少：

- `dcp_test/configs/dcp_mega_tp2.json`
- `dcp_test/configs/dcp_mega_tp4.json`

已核对这两个文件也不在本次 main 基线中，属于既有测试夹具缺失。本次未扩展修改范围。

## Smoke 命令与 trace 复核

在 CUDA 节点、测试仓库根目录中：

```bash
task_gpu_uuids=$(nvidia-smi --id=0,1,3,5 --query-gpu=uuid --format=csv,noheader | paste -sd, -)
CUDA_VISIBLE_DEVICES="$task_gpu_uuids" .venv/bin/python -m motivation.run \
  --gpus 4 --case-limit 1 --warmup 1 --iters 2 --d1-trace
```

T1/T2 为 batch=1，global=131072、local=32768。T3 为五数据集各一个原始完整 case，
四种 placement 全部测试。未裁剪 token 长度。D1 首轮为 mixed `case_000000`，随后补测
decode-only `case_000002`。最后用包含这两个原始 case 的显式 manifest，运行
`--experiments D1 --case-limit 2 --warmup 1 --iters 2 --d1-trace` 验证新鲜度检查。

每个 rank 的有效 trace 槽位数为 `4 × 2 + 74 × 3 = 230`：

- 通信 CTA 每个覆盖 Q all-gather、receive 两个阶段。
- 计算 CTA 每个覆盖 attention、history combine、final combine 三个阶段。
- `(cta_id, phase_id)` 无重复，`end_ns >= start_ns > 0`，角色与阶段一致。
- 连续两次 Graph replay 的有效槽位集合相同，每个后一次区间的起点均晚于
  对应前一次区间终点，证明本次测试中所有有效槽都更新了。

该记录仍是包含等待的阶段检查点区间，不是逐 CTA 的有效计算工作量。

## 产物

远端相对仓库路径：

```text
benchmark_logs/motivation_v2/static_20260916T133436Z/
benchmark_logs/motivation_v2/smoke_20260916T134135Z/
benchmark_logs/motivation_v2/smoke_decode_20260916T134745Z/
benchmark_logs/motivation_v2/smoke_replay_20260916T135044Z/
.cache/mega_cp/logs/motivation_v2_*.log
```

这些结果和日志已下载到本地 `benchmark_logs/motivation_v2/h20_20260916/`。
`render_check_only/` 中的图是共享 GPU 上的 smoke 渲染检查，不能用于正式性能报告。

仍未执行：八卡 CUDA smoke、默认次数/全 case 的 GPU 实验、三次独立八卡 D1
正式测量、trace 性能开销评估和获益案例冻结。D1 正式选择状态继续为
`pending_gpu_measurement`。
