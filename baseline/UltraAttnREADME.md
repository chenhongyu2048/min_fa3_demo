# UltraAttn Graph baseline：8K 固定五 case

`UltraAttn` 基线已经被移动至 `ultraattn_baseline` 分支。

本文档记录 `ring_test` 中唯一受支持的 UltraAttn forward baseline：

```text
world_size = 8
global tokens = 131072
block_tokens = 8192
QH/KVH/D = 32/8/128
dtype = BF16
mode = causal
workloads = 1×128K, 2×64K, 4×32K, 8×16K, 16×8K
```

不支持 dataset sampler、256-token packing、非固定 workload、非因果模式、
backward 或 staged runtime，也不存在 all-CP、Buddy-ring 或 round-robin
runtime fallback。

## 1. 执行结构

离线阶段使用 UltraAttn 的 block-allocation ILP；运行时把 allocation 编译为
UltraAttn 风格的依赖图：

```text
packed document-causal mask
  -> UltraAttn Gurobi Q×K allocation
  -> portable pickle-free .npz
  -> per-rank input-Q / input-KV graph nodes
  -> fused full / diagonal-causal compute graph nodes
  -> partial-O/LSE return graph node
  -> owner merge graph nodes
```

具体后端为：

```text
communication = asynchronous torch.distributed ProcessGroupNCCL
compute       = in-repo min_fa3_op.forward_varlen
partial O     = FP32
partial LSE   = FP32
final O       = BF16
```

graph executor 按依赖执行三波计算：

1. 输入通信异步启动后，先执行只依赖本地 Q/K/V 的节点。
2. 远程 Q 到达后，执行只额外依赖远程 Q 的节点；远程 K/V 继续传输。
3. 远程 K/V 到达后，执行剩余节点。

同一 rank、同一 dependency wave、同一 causal 类型的多个计算节点会被打包
成 varlen batch，避免退化为每个 8K Q×K block 一次 Python launch。partial
通信启动后，owner 同时初始化本地 partial；远程 partial 到达后用
`logaddexp` 做稳定合并。

这里的 graph executor 是 dependent-kernel graph executor，不是 CUDA Graph
capture。运行时不导入 UltraAttn 的外部 `flash_attn`、PyNCCL wrapper 或
Gurobi。

## 2. 两个 Python 环境

benchmark 使用仓库已有环境：

```text
/home/hychen/min_fa3_demo/.venv
```

它不需要安装 Gurobi、外部 FlashAttention、UltraAttn PyNCCL 或自定义 NCCL。

离线 planner 使用独立环境：

```bash
cd /home/hychen/min_fa3_demo

python3 -m venv /home/hychen/.venvs/ultraattn-planner
/home/hychen/.venvs/ultraattn-planner/bin/python -m pip install --upgrade pip
/home/hychen/.venvs/ultraattn-planner/bin/python -m pip install \
  -r baseline/UltraAttn/packing/requirements-planner.txt
```

依赖文件包含：

```text
gurobipy==12.0.1
numpy
nvidia-ml-py
pulp
regex
sympy
torch
```

还需要可用的 Gurobi license：

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic

/home/hychen/.venvs/ultraattn-planner/bin/python -c \
  "import gurobipy as gp; print(gp.gurobi.version()); gp.Model(); print('license ok')"
```

如果旧环境安装了已废弃的 `pynvml` distribution，可替换为：

```bash
/home/hychen/.venvs/ultraattn-planner/bin/python -m pip uninstall -y pynvml
/home/hychen/.venvs/ultraattn-planner/bin/python -m pip install -U nvidia-ml-py
```

这只消除 FutureWarning，不改变计划。

## 3. 固定五个 workload

五个 case 的总 token 数和每 rank token 数相同：

```text
global tokens = 131072
tokens/rank   = 16384
block_tokens  = 8192
ParD          = 16
blocks/rank   = 2
```

| Case | `global_seqlens` | Mega Ring Hybrid topology | `ring_starts` |
| --- | --- | --- | --- |
| 1×128K | `[131072]` | `1 × G8` | `[0]` |
| 2×64K | `[65536,65536]` | `2 × G4` | `[0,4]` |
| 4×32K | `[32768] × 4` | `4 × G2` | `[0,2,4,6]` |
| 8×16K | `[16384] × 8` | `8 × G1` | `[0,1,2,3,4,5,6,7]` |
| 16×8K | `[8192] × 16` | 每 rank 两个 G1 | `[0..7,0..7]` |

Mega Ring Hybrid 使用表中的 `ring_sizes/ring_starts`。UltraAttn 不读取这些
ring metadata，而是使用相同 packed stream、8192-token document-causal
block mask 和 default contiguous `cmap`。Q×K compute rank 由 UltraAttn ILP
allocation 决定。

## 4. 一次性生成五份 plans

使用：

```text
baseline/UltraAttn/packing/generate_fixed_128k_plans.sh
```

完整命令：

```bash
cd /home/hychen/min_fa3_demo

BLOCK_TOKENS=8192 \
WORLD_SIZE=8 \
QHEAD=32 \
KVHEAD=8 \
HEADDIM=128 \
TIME_LIMIT=1800 \
SOLVER_SEED=0 \
GUROBI_NUM_THREADS=32 \
PLANNER_PY=/home/hychen/.venvs/ultraattn-planner/bin/python \
PLAN_DIR=/home/hychen/min_fa3_demo/baseline/UltraAttn/packing_plans \
baseline/UltraAttn/packing/generate_fixed_128k_plans.sh
```

脚本顺序生成：

```text
1×128K
2×64K
4×32K
8×16K
16×8K
```

`TIME_LIMIT` 是每个 case 的限制。五个 Gurobi 实例串行执行；单个实例最多
使用 `GUROBI_NUM_THREADS` 个线程，避免多个多线程 MIP 同时占用全部 CPU
和内存。

脚本会拒绝 `BLOCK_TOKENS != 8192`、`WORLD_SIZE != 8` 以及不兼容的 head
配置。

## 5. 单独生成一个 plan

例如只生成 `1×128K`：

```bash
cd /home/hychen/min_fa3_demo

GUROBI_NUM_THREADS=32 \
/home/hychen/.venvs/ultraattn-planner/bin/python \
  baseline/UltraAttn/packing/export_packed_causal_plan.py \
  --global-seqlens 131072 \
  --world-size 8 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --block-tokens 8192 \
  --time-limit 1800 --solver-seed 0 \
  --output-dir baseline/UltraAttn/packing_plans
```

例如只生成 `4×32K`：

```bash
GUROBI_NUM_THREADS=32 \
/home/hychen/.venvs/ultraattn-planner/bin/python \
  baseline/UltraAttn/packing/export_packed_causal_plan.py \
  --global-seqlens 32768,32768,32768,32768 \
  --world-size 8 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --block-tokens 8192 \
  --time-limit 1800 --solver-seed 0 \
  --output-dir baseline/UltraAttn/packing_plans
```

plan cache identity 包含 global sequence lengths、world size、head 配置、
block size 和 planner revision。benchmark 会在所有 rank 间验证完整 plan
content hash；计划缺失或 metadata 不匹配时硬失败。

## 6. 对比 UltraAttn Graph 与 Mega Ring Hybrid

使用：

```text
ring_test/ultraattn/benchmark_hybrid_fixed_forward.py
```

该前端不调用 dataset sampler。它只创建一次 8-rank NCCL process group，
并在同一进程组中连续运行五个 case。

正式命令：

```bash
cd /home/hychen/min_fa3_demo

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  ring_test/ultraattn/benchmark_hybrid_fixed_forward.py \
  --qhead 32 \
  --kvhead 8 \
  --headdim 128 \
  --methods ultraattn,mega_ring_hybrid \
  --ultraattn-plan-dir /home/hychen/min_fa3_demo/baseline/UltraAttn/packing_plans \
  --ultraattn-block-tokens 8192 \
  --ultraattn-workspace-mib 2048 \
  --sm-configs 128:4 \
  --warmup-iters 10 \
  --num-iters 40 \
  --no-check
```

`--ultraattn-block-tokens` 只接受 `8192`。`--ultraattn-workspace-mib`
限制同一 dependency wave 中 varlen compute-node fusion 的 workspace，不是
CUDA kernel shared memory。

## 7. Correctness 检查

128K correctness reference 不构造 dense FP32 score tensor。它先 gather 完整
K/V，然后让每个 local Q segment 对应到 document prefix，利用 min-FA3
bottom-right causal alignment 计算参考结果。因此检查覆盖：

- default contiguous packed token ownership；
- graph input communication；
- 跨 document causal boundary；
- UltraAttn allocation；
- full 与 diagonal-causal compute nodes；
- GQA `32/8`；
- distributed partial return；
- FP32 O/LSE stable merge。

`1×128K` 检查命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc_per_node=8 \
  ring_test/ultraattn/benchmark_hybrid_forward.py \
  --global-seqlens 131072 \
  --ring-sizes 8 --ring-starts 0 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods ultraattn \
  --ultraattn-plan-dir baseline/UltraAttn/packing_plans \
  --ultraattn-block-tokens 8192 \
  --ultraattn-workspace-mib 2048 \
  --sm-configs 128:4 \
  --warmup-iters 1 --num-iters 1 --check
```

实测结果：

```text
1×128K: Check=ok
2×64K:  Check=ok
```

`2×64K` 额外覆盖 packed document boundary。

## 8. 离线计划结果

当前五份计划使用：

```text
Gurobi 12.0.1
QH=32, KVH=8, D=128
world_size=8
block_tokens=8192
solver_seed=0
threads=32
time_limit=300 seconds/case
```

| Workload | ParD | Solver status | Solve time | Objective | Best bound | MIP gap |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1×128K | 16 | TIME_LIMIT | 300.024 s | 538,968,064 | 491,285,450.1 | 8.847% |
| 2×64K | 16 | OPTIMAL | 0.384 s | 336,592,896 | 336,592,896 | 0 |
| 4×32K | 16 | OPTIMAL | 0.029 s | 202,375,168 | 202,375,168 | 0 |
| 8×16K | 16 | OPTIMAL | 0.0001 s | 0 | 0 | 0 |
| 16×8K | 16 | OPTIMAL | 0.0001 s | 0 | 0 | 0 |

`1×128K` 是 time-limit 内的 feasible incumbent，并未证明全局最优。它可
正常执行；runtime Note 会显示：

```text
solver=TIME_LIMIT, gap=0.08847
```

`8×16K` 与 `16×8K` objective 为 0，是因为 default contiguous `cmap`
已经让每个 document 完全落在 owner rank 内。

## 9. 8-GPU 正式运行结果

测试环境：

```text
GPU               = 8 × NVIDIA H100 80GB
PyTorch           = 2.10.0+cu128
dtype             = BF16
QH/KVH/D          = 32/8/128
mode              = causal
UltraAttn block   = 8192
Ultra workspace   = 2048 MiB
Mega Ring SM      = 128 compute + 4 communication
warmup iterations = 10
measured iters    = 40
correctness       = disabled for the performance run
```

| Workload | Mega topology | Ultra Graph ms | Ultra Agg TFLOPS | Ultra Avg/GPU | Mega ms | Mega Agg TFLOPS | Mega Avg/GPU | Ultra/Mega latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1×128K | 1×G8 | 32.207 | 4369.8 | 546.2 | 30.167 | 4665.4 | 583.2 | 1.068× |
| 2×64K | 2×G4 | 17.830 | 3946.7 | 493.3 | 15.044 | 4677.6 | 584.7 | 1.185× |
| 4×32K | 4×G2 | 10.230 | 3439.3 | 429.9 | 7.510 | 4685.4 | 585.7 | 1.362× |
| 8×16K | 8×G1 | 4.782 | 3679.3 | 459.9 | 3.780 | 4654.2 | 581.8 | 1.265× |
| 16×8K | 每 rank 2×G1 | 2.455 | 3583.7 | 448.0 | 1.942 | 4530.0 | 566.2 | 1.264× |

跨 case summary：

| Method | Cases | Min ms | Mean ms | P50 ms | Max ms | Mean TFLOPS | Weighted TFLOPS | Weighted/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UltraAttn Graph | 5/5 | 2.455 | 13.501 | 10.230 | 32.207 | 3803.8 | 4039.6 | 504.9 |
| Mega Ring Hybrid | 5/5 | 1.942 | 11.688 | 7.510 | 30.167 | 4642.5 | 4665.9 | 583.2 |

五个 case 的 causal work 为 `sum_i L_i² / 2`，因此 sequence 数增加时理论
FLOPs 会下降。TFLOPS 使用每个 case 的实际 causal FLOPs。

UltraAttn timing 包含 input packing、异步 graph input communication、varlen
compute-node fusion、FP32 partial return 和 owner merge。Mega Ring Hybrid
timing 是 fused hierarchical kernel 的 end-to-end op 时间。两者都不包含
plan load、graph compile、buffer allocation 和 benchmark 外层同步。

## 10. Runtime Note

结果中的 Note 类似：

```text
UltraAttn ILP graph executor; async torch.distributed NCCL;
min_fa3 varlen; FP32 O/LSE merge; block=8192; nodes=5;
workspace=2048MiB; solver=TIME_LIMIT, gap=0.08847;
plan=448bd4eafc80
```

其中：

- `nodes` 是当前 rank 的 fused compute graph node 数；不同 rank 可能不同，
  Note 显示 rank 0 的数量。
- `plan` 是完整 plan content SHA-256 的前缀，不是文件名 cache key。
- graph runtime 没有 staged fallback。
- graph runtime 不使用 Mega Ring Hybrid topology。
- 显式请求 `--methods ultraattn` 时，计划缺失或 workload 不属于固定五 case
  会硬失败。
- 通过 `--methods all` 间接选择时，不兼容的 UltraAttn case 会被跳过。

## 11. 相关入口

| 文件 | 作用 |
| --- | --- |
| `baseline/UltraAttn/packing/plan_format.py` | mask、metadata、cache key、plan validation |
| `baseline/UltraAttn/packing/graph_executor.py` | runtime-only UltraAttn graph lowering 与 Execution Plan 类型 |
| `baseline/UltraAttn/packing/export_packed_causal_plan.py` | 离线 Gurobi allocation exporter |
| `baseline/UltraAttn/packing/generate_fixed_128k_plans.sh` | 固定五 case 计划生成 |
| `ring_test/ultraattn/ultraattn_forward.py` | 8K graph compiler、executor、Torch NCCL 和 min-FA3 adapter |
| `ring_test/ultraattn/benchmark_hybrid_fixed_forward.py` | 固定五 case 对比前端 |
| `ring_test/ultraattn/benchmark_hybrid_forward.py` | UltraAttn 单 workload benchmark 与 correctness frontend |
