# Hopper FlashAttention Backward 偶发 `misaligned address`：完整 GPU Debug 教程

> 案例来源：`min_fa3_demo` 的 8 卡 H100 MegaRing FlashAttention backward。
>
> 文档日期：2026-08-13。
>
> 目标读者：维护 CUDA/CUTLASS/CuTe、PyTorch CUDA Extension、分布式训练算子和 Hopper warp-specialized kernel 的开发者。

本文不是只保留成功路径的事后总结。它按实际排查过程记录：原始异步报错、三次没有抓到问题的 memcheck、带噪声的 racecheck/synccheck、coredump 参数错误、Python 导入失败、错误补丁位置、Ninja 没有重编头文件、NCCL 导致的 sanitizer 假阳性，以及最终如何用 core、SASS 和隔离实验收敛到一个最小修复。

本文使用三种证据等级：

- **已证实**：可由日志、CUDA coredump、error PC、SASS、源码控制流或重复实验直接验证。
- **强支持**：隔离实验稳定改变结果，并且与底层机制一致，但还没有独立 PTX 最小用例证明是编译器或硬件 erratum。
- **待验证**：合理假设或通用建议，不冒充本次案例的已证实事实。

---

## 目录

1. [问题概述](#1-问题概述)
2. [环境信息](#2-环境信息)
3. [复现步骤](#3-复现步骤)
4. [完整排查时间线](#4-完整排查时间线)
5. [根因分析](#5-根因分析)
6. [修复方案](#6-修复方案)
7. [验证方法和结果](#7-验证方法和结果)
8. [可复用 GPU 排查框架](#8-可复用-gpu-排查框架)
9. [常见 GPU Bug 分类与工具](#9-常见-gpu-bug-分类与工具)
10. [工具命令速查表](#10-工具命令速查表)
11. [预防建议与最佳实践](#11-预防建议与最佳实践)
12. [仍待补充的信息](#12-仍待补充的信息)
13. [10 分钟快速定位 GPU 问题](#13-10-分钟快速定位-gpu-问题)

---

## 1. 问题概述

### 1.1 症状

执行 `benchmark_uniform.sh` 的 causal backward 时，MegaRing fused backward kernel 偶发：

```text
CUDA error: misaligned address
```

它不是每次发生，也不固定在某一张 GPU、某一个 distributed rank 或某一种 context length：

| 证据 | workload | 首个可见失败位置 |
|---|---:|---|
| `20260813-132129` | `context=65536, B=16, S=4096` | rank 5 在 `torch.cuda.synchronize()` 报错 |
| `20260813-132714` | `context=262144, B=16, S=16384` | rank 6 的 NCCL watchdog 报错 |
| coredump stress | `context=65536, B=16, S=4096` | rank 0；kernel launch check 报错 |

原始日志：

- [`20260813-132129` 日志](../benchmark_logs/20260813-132129-uniform-backward/benchmark_uniform_backward.log)
- [`20260813-132714` 日志](../benchmark_logs/20260813-132714-uniform-backward/benchmark_uniform_backward.log)
- [生成 coredump 的 stress 日志](../benchmark_logs/coredump-log-stress-uniform-bwd.log)

### 1.2 完整关键报错

第一份日志中，错误在一次较晚的同步点被观察到：

```text
[rank5]:   File "/home/hychen/min_fa3_demo/ring_test/benchmark_topology_backward.py", line 629, in measure_backward_ms
[rank5]:     torch.cuda.synchronize()
[rank5]:   File "/home/hychen/min_fa3_demo/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py", line 1108, in synchronize
[rank5]:     return torch._C._cuda_synchronize()
[rank5]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]: torch.AcceleratorError: CUDA error: misaligned address
[rank5]: Search for `cudaErrorMisalignedAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
[rank5]: CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
[rank5]: For debugging consider passing CUDA_LAUNCH_BLOCKING=1
[rank5]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

第二份日志中，首先看到的是 NCCL watchdog：

```text
[rank6]:[E813 13:28:06.493579419 ProcessGroupNCCL.cpp:2093]
[PG ID 0 PG GUID 0(default_pg) Rank 6]
Process group watchdog thread terminated with exception: CUDA error: misaligned address

Exception raised from query at /pytorch/aten/src/ATen/cuda/CUDAEvent.h:108
...
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const
frame #3: c10d::ProcessGroupNCCL::WorkNCCL::isCompleted()
frame #4: c10d::ProcessGroupNCCL::Watchdog::runLoop()
```

启用同步 launch 和 GPU coredump 后，报错收敛到真正的 fused backward launch 边界：

```text
Starting GPU coredump generation, set the CUDA_COREDUMP_SHOW_PROGRESS environment variable to 1 to enable more detailed output
CUDA error (/home/hychen/min_fa3_demo/include/backward/min_fa3_bwd_launch.h:686): misaligned address
```

对应源码是：

```cpp
dim3 mega_grid(params.num_comp_sm + params.num_comm_sm, 1, 1);
kernel<<<mega_grid, block_dims, smem_size, stream>>>(mega_params);
C10_CUDA_KERNEL_LAUNCH_CHECK();  // 当时的第 686 行
```

### 1.3 影响范围

本案例已确认的影响面：

- SM90a Hopper MegaRing fused **backward**。
- causal attention。
- `torch.bfloat16`，head dim 128。
- `QH=32, KVH=8` 的 GQA。
- 8 卡显式拓扑 benchmark。
- `B=16`、所有 ring 均为 G1 的 uniform workload。
- 至少 `context=65536` 和 `context=262144` 可偶发触发。

没有证据表明以下路径受到同一个 bug 影响：

- forward-only kernel；
- 非 MegaRing 普通 backward；
- 非 SM90 GPU；
- fp16/fp8 或其他 head dim；
- 其他项目、其他 CUTLASS 版本。

### 1.4 触发条件和偶发性

高风险配置：

```text
world_size = 8
method = mega_ring_all_cp
mode = causal
B = 16
QH = 32
KVH = 8
D = 128
dtype = bfloat16
SM config = 128 compute CTAs + 4 communication CTAs
ring_sizes = [1, 1, ..., 1]  # 16 个 G1 batch
ring_starts = [0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]
```

该问题偶发意味着：

- 一次通过不能证明没有 bug；
- sanitizer 下通过不能直接证明没有 bug，因为插桩会改变代码生成、launch 时序和资源占用；
- 必须提高迭代次数，并用 `CUDA_LAUNCH_BLOCKING=1` 让错误尽可能靠近真实 launch 边界。

本次没有统计出严格的触发概率。因此，频率应记录为：

> **[待补充]** 修复前在固定空闲节点、固定二进制、固定 seed 下，运行 N 次出现 M 次的统计数据。需要提供一组自动循环 stress 的完整结果，才能给出可信的复现概率。

---

## 2. 环境信息

### 2.1 硬件与拓扑

| 项目 | 本次环境 |
|---|---|
| GPU | 8 × NVIDIA H100 80GB HBM3 |
| Compute capability | 9.0；构建目标为 `sm_90a` |
| 每卡显存 | 81559 MiB |
| GPU 间连接 | 全互联 `NV18` |
| MIG | Disabled |
| Compute mode | Exclusive Process |
| Power limit | 700 W/GPU |
| 修复后检查时 ECC | volatile/aggregate corrected/uncorrected 均为 0 |

采集命令：

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,memory.total,compute_cap --format=csv,noheader
nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv,noheader
```

注意：修复后 ECC 为 0 支持“不是明显硬件 ECC 故障”，但不能替代故障发生时的 Xid/ECC 日志。

### 2.2 软件栈

| 组件 | 版本 |
|---|---|
| OS | Ubuntu 22.04.4 LTS |
| Kernel | `5.15.0-185-generic #195-Ubuntu` |
| NVIDIA driver | 590.48.01 |
| `nvidia-smi` 显示 CUDA | 13.1 |
| 实际 CUDA toolkit / NVCC | 12.8.61 |
| PyTorch | 2.10.0+cu128 |
| `torch.version.cuda` | 12.8 |
| Python | 3.12.13 |
| cuDNN | 9.10.2（`torch.backends.cudnn.version() == 91002`） |
| NCCL | 2.27.5 |
| GCC | 11.4.0 |
| glibc | 2.35 |
| CUTLASS | header version 4.3.4；commit `7127592069c2fe01b041e174ba4345ef9b279671` |
| ThunderKittens | commit `34b15f7e7012de25ae162c8d9dc85296dd342676` |
| Compute Sanitizer | 2025.1.0.0, build 35351055 |
| cuda-gdb | 12.8, based on GDB 13.2 |
| Nsight Systems | 2024.6.2 |
| Nsight Compute | 2025.1.0.0 |
| 项目 HEAD | `585894c15556070b7c494dc65c2c6de76afe7604`，另有本文描述的未提交修复 |

#### 不要误读 `nvidia-smi` 的 CUDA Version

`nvidia-smi` 中的：

```text
Driver Version: 590.48.01      CUDA Version: 13.1
```

表示当前驱动支持的最高 CUDA driver API 兼容版本，不表示当前项目由 CUDA 13.1 编译。本项目实际为：

```text
nvcc release 12.8, V12.8.61
torch=2.10.0+cu128
torch.version.cuda=12.8
```

排查“驱动/CUDA 不兼容”时，必须同时采集 driver、NVCC、框架编译 CUDA 版本，不能只看 `nvidia-smi` 顶部一行。

### 2.3 构建关键参数

本项目 CUDA Extension 使用：

```text
-O3
-std=c++20
--use_fast_math
-lineinfo
-gencode arch=compute_90a,code=sm_90a
-DCUTLASS_ENABLE_GDC_FOR_SM90
-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
-Xptxas=-v
```

`-lineinfo` 很重要：它不会像 `-G` 那样完全改变优化模型，但能让 sanitizer/SASS 尽可能回映源码。对于偶发优化相关 bug，建议先保留 release 优化和 `-lineinfo`，再单独构建 `-G` 版本做对照；不要一开始只测 debug kernel，因为它可能不再复现。

### 2.4 环境信息缺口

- **[待补充] 故障时段的 kernel/Xid 日志**：当前用户无权读取完整 `dmesg` 或其他用户的 kernel journal：

  ```text
  dmesg: read kernel buffer failed: Operation not permitted
  journalctl: You are currently not seeing messages from other users and the system.
  ```

  需要管理员提供：

  ```bash
  sudo dmesg -T | grep -Ei 'NVRM|Xid|ECC|AER|PCIe'
  sudo journalctl -k --since '2026-08-13 13:20:00' --until '2026-08-13 14:15:00'
  ```

- **[待补充] 原始 Hopper FA3 provenance commit**：本目录是 copied + trimmed demo，但本次会话没有记录被复制的上游 FA3 精确 commit。若要向上游或 NVIDIA 提交问题，需要提供该 commit。

---

## 3. 复现步骤

### 3.1 前置条件

```bash
cd /home/hychen/min_fa3_demo
source .venv/bin/activate
make
```

要求：

- 8 张空闲 H100；
- GPU 0-7 可见且拓扑正常；
- `_min_fa3_op.so` 确实由当前源码重新编译；
- 不与其他训练/推理服务共享这些 GPU；
- 保留完整 stdout/stderr 和退出码。

先确认 GPU 空闲：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
```

输出为空才开始。共享节点上“显存还有余量”不等于适合复现偶发时序 bug；其他进程会改变 SM 调度、时钟、显存压力和复现概率。

### 3.2 原始 benchmark 复现

原始脚本默认覆盖三个 context 和四个 compute/communication SM 配置：

```bash
./benchmark_uniform.sh
```

日志中展开后的核心命令形如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_topology_backward.py \
  --global-seqlens 4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096 \
  --ring-sizes 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 \
  --ring-starts 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --mode causal --methods mega_ring_all_cp \
  --sm-configs 128:4,124:8,120:12,116:16 \
  --warmup-iters 10 --num-iters 40 \
  --seed 0 --no-check
```

### 3.3 更稳定、边界更清晰的 stress

优先锁定一次已失败的 workload，并打开同步 launch：

```bash
CUDA_LAUNCH_BLOCKING=1 \
CONTEXT_LENGTHS=65536 \
BATCH_SIZES=16 \
SM_CONFIGS=128:4 \
WARMUP_ITERS=50 \
NUM_ITERS=200 \
CHECK=0 \
LOG_DIR=benchmark_logs/repro-stress-uniform-bwd \
./benchmark_uniform.sh
```

另一个已失败规模：

```bash
CUDA_LAUNCH_BLOCKING=1 \
CONTEXT_LENGTHS=262144 \
BATCH_SIZES=16 \
SM_CONFIGS=128:4 \
WARMUP_ITERS=50 \
NUM_ITERS=200 \
CHECK=0 \
LOG_DIR=benchmark_logs/repro-262144-uniform-bwd \
./benchmark_uniform.sh
```

为什么用 `CUDA_LAUNCH_BLOCKING=1`：

- CUDA launch 默认异步；真正的 kernel 错误可能在后续 `cudaEventQuery`、`torch.cuda.synchronize()`、NCCL watchdog 或另一个无关 API 才被观察到。
- blocking 会在每个 launch 后更早同步，缩小“最后一个成功 launch”和“第一个失败 launch”之间的范围。
- 它会改变时序，因此“blocking 下不复现”不能排除 race；本案例中 blocking 下仍能复现，因而特别有价值。

### 3.4 输入数据和 shape

`context=65536, B=16` 时：

```text
每个 global sequence length = 65536 / 16 = 4096
ring_size = 1
每张 GPU 拥有两个 batch（starts 0..7 重复两次）
每 rank local_total = 2 * 4096 = 8192 tokens
q / dout: [8192, 32, 128], bfloat16
k / v:    [8192,  8, 128], bfloat16
```

`context=262144, B=16` 时：

```text
每个 global sequence length = 16384
每 rank local_total = 32768 tokens
q / dout: [32768, 32, 128], bfloat16
k / v:    [32768,  8, 128], bfloat16
```

数据由固定 seed 生成。MegaRing all-CP 的 Q/K/V 使用 `base_seed=args.seed + 101`，dO generator 使用：

```python
generator.manual_seed(args.seed + 20_260_817 + rank)
```

固定输入不能固定 GPU 调度，但可以排除“随机数内容变化导致”的混淆。

### 3.5 定向 memcheck 最小命令

对多进程应用，必须让 sanitizer 跟踪所有子进程，并过滤目标 kernel：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_LAUNCH_BLOCKING=1 \
compute-sanitizer \
  --tool memcheck \
  --target-processes all \
  --kernel-name kernel_substring=mega_ring_flash_attn_bwd_kernel \
  --report-api-errors no \
  --error-exitcode 99 \
  --force-blocking-launches \
  --log-file benchmark_logs/kernel-memcheck-uniform-bwd-%p.log \
  .venv/bin/torchrun --standalone --nproc_per_node=8 \
  ring_test/benchmark_topology_backward.py \
  --global-seqlens 4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096 \
  --ring-sizes 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 \
  --ring-starts 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7 \
  --qhead 32 --kvhead 8 --headdim 128 \
  --allgather-overlapping-heads-k-stride 4 \
  --mode causal --methods mega_ring_all_cp \
  --sm-configs 128:4 \
  --warmup-iters 0 --num-iters 1 \
  --seed 0 --no-check
```

为什么过滤 kernel：

- 分布式启动会运行 PyTorch、NCCL、forward preparation、IPC reset 和多个辅助 kernel。
- 不过滤时，输出可能被已知探测 API 或其他 kernel 的 hazard 淹没。
- `--launch-count` 只统计匹配 filter 的 launch，便于精确控制。

为什么这里用了 `--report-api-errors no`：见 4.16 节。它用于屏蔽 NCCL 初始化期间预期的 `cudaFuncGetAttributes` capability probing；不能用来掩盖目标应用真正依赖的 CUDA API 错误。应先看一次完整日志，确认噪声来源后再关闭。

---

## 4. 完整排查时间线

### 4.1 阅读两份原始日志：先判断“是否固定”

动作：比较失败 workload、rank 和错误观察点。

关键发现：

```text
日志 A: context=65536, rank 5, torch.cuda.synchronize()
日志 B: context=262144, rank 6, NCCL watchdog / cudaEventQuery
```

当时判断：

- 不像固定 GPU 地址坏块或固定 rank 的输入偏移错误。
- NCCL watchdog 很可能只是异步错误的观察者，不应立即把根因归到 NCCL。
- 需要把错误拉回真实 kernel launch。

类似问题怎么判断：

- 固定在同一 rank/shape/边界：优先查分片、offset、stride、owner mapping。
- 随 rank 漂移：优先查 race、未初始化状态、编译器调度脆弱点、硬件/系统错误。
- 总在 NCCL watchdog：先用 blocking 或显式同步证明是否 NCCL kernel 自身失败。

### 4.2 优先运行 memcheck：三次都是 0 errors

执行了普通、stress 和较长版本的 memcheck，日志分别是：

- [`memcheck-uniform-bwd-1929913.log`](../benchmark_logs/memcheck-uniform-bwd-1929913.log)
- [`memcheck-stress-uniform-bwd-1935664.log`](../benchmark_logs/memcheck-stress-uniform-bwd-1935664.log)
- [`memcheck-long-uniform-bwd-1945258.log`](../benchmark_logs/memcheck-long-uniform-bwd-1945258.log)

三份结果均为：

```text
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

为什么这样做：

- `misaligned address` 首先应查非法 global/shared/local memory access。
- memcheck 能检测越界、misaligned、use-after-free 和部分 API 错误。

为什么没有因此排除 bug：

- sanitizer 插桩显著改变 kernel 执行速度和 host/device 时序。
- 偶发 bug 可能只在 release codegen 或特定调度窗口出现。
- 0 errors 只说明“被检查到的那些运行没有触发”，不是全称证明。

可复用判断：

> sanitizer 通过 + 原生 stress 失败，通常要继续做 blocking、coredump 和 SASS；不要停在“memcheck 没报错”。

### 4.3 尝试 racecheck/synccheck：出现了相互矛盾的噪声

未充分过滤的 racecheck 出现：

```text
========= RACECHECK SUMMARY: 100 hazards displayed
========= (4528176 errors, 661824 warnings)
```

未过滤 synccheck 出现：

```text
========= Barrier error detected. Divergent thread(s) in block.
========= at flash::named_barrier_sync(...) in min_fa3_named_barrier.h:21
...
========= ERROR SUMMARY: 30976 errors
```

而定向到 backward kernel 的 racecheck 为：

```text
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

相关日志：

- [未过滤 racecheck](../benchmark_logs/racecheck-valid-uniform-bwd.log)
- [synccheck](../benchmark_logs/synccheck-uniform-bwd-1952099.log)
- [定向 backward racecheck](../benchmark_logs/racecheck-bwd-kernel-uniform-bwd.log)

当时判断：

- 数百万 hazard 涵盖 preparation/辅助 kernel，不能直接映射到本次 fault。
- Hopper warp-specialized kernel 使用 named barrier、不同 warp role 和部分线程参与，工具报告必须结合预期 participant count 解读。
- synccheck 报告不应被简单宣布为“假阳性”；它是一个独立信号，但 error PC 最终落在 dO TMA，而修复未改变 named barrier/fence。因此它没有被作为本次 misaligned 的直接根因。

可复用判断：

1. 先保留一份未过滤日志，知道全局有什么信号。
2. 再用 `--kernel-name` 和 `--launch-count` 缩到目标 kernel。
3. 看 device frame、block/thread、地址和源行是否与真实 fault 一致。
4. 工具报告数量大不等于因果关系强。
5. 对 named barrier 报告建立单独 issue，不能因主 bug 修复就默认它无害。

### 4.4 用 `CUDA_LAUNCH_BLOCKING=1` 拉回真实 launch

命令：

```bash
CUDA_LAUNCH_BLOCKING=1 \
CONTEXT_LENGTHS=65536 BATCH_SIZES=16 \
SM_CONFIGS=128:4 WARMUP_ITERS=50 NUM_ITERS=200 CHECK=0 \
./benchmark_uniform.sh
```

关键输出：

```text
CUDA error (/home/hychen/min_fa3_demo/include/backward/min_fa3_bwd_launch.h:686): misaligned address
```

确认：错误来自 `mega_ring_flash_attn_bwd_kernel` launch 之后的检查，不是稍后的 NCCL event query 自己生成。

为什么重要：

- PyTorch/NCCL stack trace 是“错误被观察到的位置”，不是一定是“错误发生的位置”。
- CUDA context 在严重 device fault 后进入 sticky error 状态，之后多个 API 都会重复返回错误。
- distributed launcher 会向其他 rank 发送 SIGTERM，因此其他 rank 的 `-15` 不是第二个根因。

### 4.5 检查拓扑控制流：排除 mega-copy fence

失败 workload 明确打印：

```text
B=16
Hybrid rings: sizes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
starts=[0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]
```

G1 的含义：每个 ring 只有 `ring_step=0`，没有远端 K/V step。远端 KV mega-copy 及其 KV-ready fence 分支不会执行。

当时判断：

- 如果不执行某段代码仍可复现，它不能是该复现的必要根因。
- 这比“注释掉 fence 后似乎不复现”更强，因为没有引入额外 codegen 变化。

类似问题怎么判断：

- 用业务参数推导实际分支，不要只看 kernel 源码中“存在某 fence”。
- 输出 topology/scheduler ticket，使每个异步阶段是否执行可审计。
- 对通信 kernel，分别设计 G1（无远端通信）和 G>1（真实通信）用例。

### 4.6 生成 GPU coredump：第一次参数有错误，但 core 仍生成

日志开头出现：

```text
WARNING: Ignoring unknown coredump flag 'log_only'
WARNING: Ignoring unknown coredump flag 'faulted_contexts_only'
Valid flags are: skip_nonrelocated_elf_images, skip_global_memory,
skip_shared_memory, skip_local_memory, skip_abort, skip_constbank_memory,
gzip_compress
```

这是一次失败尝试：使用了当前 CUDA 12.8 不支持的 coredump flag。

为什么仍继续：

- 警告明确说“忽略未知 flag”，不是 coredump 完全禁用。
- 随后日志出现 `Starting GPU coredump generation`，并生成约 1.5 GiB：

  ```text
  core_1786601204_zkrh-58_2012398.nvcudmp
  ```

正确做法：

```bash
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_FILE='core_%h_%p.nvcudmp'
# 只使用当前 toolkit --help/文档列出的 flags；不确定时先不设置。
```

提醒：完整 GPU core 可能非常大。先确认磁盘空间，避免在共享 home 或根分区无限生成。

### 4.7 用 cuda-gdb 打开 core：定位到 compute CTA

打开：

```bash
cuda-gdb -c core_1786601204_zkrh-58_2012398.nvcudmp
```

典型交互命令：

```gdb
info cuda devices
info cuda kernels
info cuda threads
print/x $errorpc
x/32i $errorpc-0x100
info registers
```

关键结果见：

- [coredump summary](../benchmark_logs/cuda-gdb-coredump-summary.log)
- [error PC 反汇编](../benchmark_logs/cuda-gdb-errorpc.log)

原样关键输出：

```text
CUDA Exception: Warp Misaligned Address
The exception was triggered at PC 0x7f539dfb3080
[Current focus set to CUDA kernel 0, grid 1111,
 block (54,0,0), thread (0,0,0), device 0, sm 36, warp 1, lane 0]

kernel<<<(132,1,1),(384,1,1)>>>
```

launch 分流源码：

```cpp
if (int(blockIdx.x) < params.num_comp_sm) {
    AttnKernel{}(params.mainloop, smem_buf);
    return;
}
int const comm_bid = int(blockIdx.x) - params.num_comp_sm;
run_mega_ring_bwd_kv_load(...);
run_mega_ring_bwd_dkv_store(...);
```

因为：

```text
blockIdx.x = 54
num_comp_sm = 128
54 < 128
```

所以 **已证实** fault 位于 compute CTA，不在 communication CTA。mega-copy/fence 假设进一步被排除。

### 4.8 error PC 精确落到第二条 dO TMA

error PC 附近：

```text
+129040: UMOV UR21,0x40
+129056: UTMALDG.4D desc[UR18][UR8][UR16]
+129072: UMOV UR23,UR9
+129088: STL [R1],R2
+129104: UMOV UR8,UR20
+129120: UMOV UR10,UR21
+129136: UMOV UR20,0x10
*> +129152: UTMALDG.4D desc[UR18][UR8][UR16]
+129168: UBLKCP.S.G [UR22][UR24],UR20
```

逻辑上的 `64 x 128` dO tile 被 TMA 拆为两次 4D load：

- 第一条：head-dim `D=0..63`，成功发出。
- 第二条：head-dim `D=64..127`，触发 Warp Misaligned Address。

从 descriptor/寄存器/源码 tile 映射还原的坐标约为：

```text
{D=64, token=8128, q_head=15, batch_mode=0}
```

对 `local_total=8192, QH=32, D=128` 而言，这个最后 tile 覆盖 token `8128..8191`，仍在范围内；head 15 也合法。dO TMA descriptor、预期 shared destination、mbarrier 地址的静态对齐检查同样合法。

当时判断：

- 不是简单的 tensor shape 越界。
- 不是 dO base pointer 天然未对齐，否则更可能稳定地在第一条或固定位置失败。
- 需要检查 TMA 近邻指令如何生成和复用 uniform registers。

### 4.9 提取 cubin 和 SASS

使用：

```bash
cuobjdump --list-elf _min_fa3_op.so
cuobjdump --extract-elf all _min_fa3_op.so
cuobjdump --dump-sass _min_fa3_op.so > min_fa3_bwd_launch.sass
# 或对提取出的 cubin：
nvdisasm -g -gi min_fa3_bwd_launch.sm_90a.cubin > min_fa3_bwd_launch.sass
```

本次产物：

- [`min_fa3_bwd_launch.sass`](../benchmark_logs/min_fa3_bwd_launch.sass)，约 27 MiB。
- [`min_fa3_bwd_launch.sm_90a.cubin`](../benchmark_logs/min_fa3_bwd_launch.sm_90a.cubin)，约 3.8 MiB。

为什么看 SASS 而不只看 C++/PTX：

- 真正执行的是 ptxas 生成的 SASS。
- Hopper TMA 使用 uniform register (`UR*`) 传 descriptor、坐标、shared destination 和 transaction 参数。
- C++ 中两个不相关表达式可能被编译器调度到同一局部区域，并复用同一个 UR 编号。
- 偶发 fault 的源代码行有时只是一条内联 PTX 包装，必须看它最终拿到什么寄存器。

### 4.10 检查 `scheduler_prefetch()`：先否定一个错误假设

一开始怀疑本地 copied + trimmed 版本把 `scheduler_prefetch()` 放到了不同于上游 FA3 的位置。对比后发现：上游普通 backward 同样在最后一次 dO TMA 之前调用 prefetch。

因此，“本地 prefetch 调用位置偏离上游”这个假设被否定。

真正的差异是 prefetch 的内容：

- 普通 scheduler 的 prefetch 很轻或为空。
- MegaRing scheduler 的 `prefetch_next_work()` 会：
  - `atomicAdd` claim 新 ticket；
  - warp `__shfl_sync`；
  - 调用 `claim_dynamic_tile()`；
  - 进入复杂的 `decode_work()`；
  - 做 warp-wide prefix scan、`__ballot_sync`、`__ffs` 和多次 shuffle；
  - 形成复杂的分支和 uniform-register 压力。

源代码核心：

```cpp
CUTLASS_DEVICE void prefetch_next_work(
        Params const& params,
        WorkTileInfo& current_work) const {
    int tile_idx = 0;
    if (int(threadIdx.x) % cutlass::NumThreadsPerWarp == 0) {
        tile_idx = atomicAdd(params.tile_count_semaphore, 1) + params.num_comp_sm;
    }
    tile_idx = __shfl_sync(0xffffffff, tile_idx, 0);
    current_work = claim_dynamic_tile(params, tile_idx, current_work);
}
```

这一步的通用经验：

> 对 copied kernel，不只比较调用点，还要比较 template specialization 后实际执行的 helper。相同 API 名称不代表相同 codegen 复杂度。

### 4.11 寄存器快照：有价值，但不能过度解读

旧 SASS 的 scheduler 区域包含：

```text
UMOV UR8, 0xffffffff
BRA.DIV UR8, ...
```

而 TMA 指令也使用 `UR8`：

```text
UTMALDG.4D [UR8], [UR16], desc[UR18]
```

core 当前 uniform-register 快照看到 `UR8=0xffffffff`。这高度可疑，因为 `UR8` 在 TMA 中承载 shared-memory destination 一类的 uniform operand。

但这里有一个必须保留的限制：

```text
errorpc = kernel + 129152
core 当前 pc = kernel + 129712（停止点晚于 errorpc）
```

GPU warp 抛出异常后，dump 的“当前 PC/当前寄存器”不一定就是 fault 指令消费时的精确状态。`UR8=0xffffffff` 可能来自 fault 后 scheduler 路径已经执行的指令。

因此正确结论是：

- **已证实**：fault 是第二条 dO TMA。
- **已证实**：正常静态坐标和预期地址合法。
- **强支持**：相邻复杂 scheduler decode 形成 UR codegen 调度脆弱区。
- **未证实**：fault 瞬间第二条 TMA 一定消费了 `0xffffffff`。
- **未证实**：这是某个已知 ptxas 或 H100 silicon erratum。

### 4.12 隔离实验：把 prefetch 移到最后 dO TMA 后

第一版实验直接交换：

```diff
-        scheduler_prefetch();
         if (lane_predicate) {
             // final dO/dPsum TMA
         }
+        scheduler_prefetch();
```

目的不是声称最终 API 设计应如此，而是回答一个可证伪问题：

> 如果复杂 scheduler decode 不再与最后两条 dO TMA 相邻，misaligned 是否消失？

新 SASS 顺序确认：

```text
0x1c3a0  UTMALDG.4D   # first dO half
0x1c400  UTMALDG.4D   # second dO half
0x1c470  scheduler prefetch starts
0x1c570  UMOV UR8, 0xffffffff
```

这证明源码修改实际改变了目标机器码区域，而不只是 C++ diff 看起来不同。

### 4.13 直接 `make` 成功，但后来发现第一次条件化构建没有重编

用户要求直接使用：

```bash
make
```

最初实验版本成功编译。ptxas 报告 MegaRing kernel：

```text
104 bytes stack frame
480 bytes spill stores
528 bytes spill loads
Used 168 registers
Used 16 barriers
```

旧 cubin stack 为 88 bytes。这说明修复不是通过“减少 spill/降低寄存器压力总量”起作用，而是通过隔离局部指令顺序。

后来把修改收窄为 MegaRing-only 后，再次 `make` 输出：

```text
ninja: no work to do.
```

这是一次关键失败：`.so` 被重新链接，但 CUDA object 很可能仍是上一版。时间戳和 Ninja dependency 查询确认：

```text
include/backward/min_fa3_bwd_mainloop.h  比 object 新
min_fa3_bwd_launch.o                    仍是旧时间
ninja -t deps .../min_fa3_bwd_launch.o:
#deps 0
```

修正方法：只清理受影响的生成 object，再直接 `make`：

```bash
ninja -C build/temp.linux-x86_64-cpython-312 \
  -t clean \
  /home/hychen/min_fa3_demo/build/temp.linux-x86_64-cpython-312/csrc/backward/min_fa3_bwd_launch.o
make
```

随后看到完整 NVCC 命令和 ptxas 输出，确认真正重编。

通用经验：

> “build command 返回 0”不等于“修改进入二进制”。对 header-heavy CUDA template，必须检查 depfile、object mtime、NVCC 输出或二进制 hash。

### 4.14 GPU 被外部服务占用：等待，而不是抢占

中途 8 张 GPU 被外部 `sglang` 服务占用，每卡约 46 GiB、利用率 94-100%。进程属于其他用户，父进程有：

```text
timeout --signal=TERM --kill-after=150 2400 ...
```

处理方式：

- 不终止他人进程。
- 不在高负载环境下运行 correctness/stress。
- 只读监测，等待 GPU 空闲。

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
```

为什么：

- 共享负载会污染性能和偶发性判断。
- Exclusive Process 模式下也可能直接无法创建新 context。
- 终止未知任务是破坏性操作，不属于调试权限范围。

### 4.15 Correctness 第一次失败：不是 CUDA 回归，而是 Python import

第一次运行：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=8 \
  scripts/test_mega_ring/mega_ring_test_min_fa3_varlen_backward_hybrid_multi_rank.py \
  ...
```

8 个 rank 都在 import 阶段失败：

```text
ModuleNotFoundError: No module named 'min_fa3_op'
```

为什么不是 kernel 回归：

- traceback 停在 Python module import。
- 没有创建目标 CUDA tensor，也没有 launch kernel。
- 多个 rank 的 SIGTERM 是 torchrun 终止同组进程的结果。

修正：显式加入项目根目录和 sibling test helper：

```bash
PYTHONPATH=/home/hychen/min_fa3_demo:/home/hychen/min_fa3_demo/scripts/test_mega_ring \
.venv/bin/torchrun --standalone --nproc_per_node=8 \
  scripts/test_mega_ring/mega_ring_test_min_fa3_varlen_backward_hybrid_multi_rank.py \
  --global-seqlens 2048,1024,512,256 \
  --ring-sizes 8,4,2,1 \
  --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --repeat 2 \
  --num-comp-sm 100 --num-comm-sm 16
```

结果：

```text
hierarchical mega-ring backward: ok
(global=[2048, 1024, 512, 256], rings=[8, 4, 2, 1],
 starts=[0, 4, 2, 7], QH=16, KVH=8, repeat=2)
```

### 4.16 第一次最终 memcheck 以 99 退出：32 条其实来自 NCCL 探测

第一次最终定向 memcheck 使用了：

```text
--error-exitcode 99
```

benchmark 本身完成，但 sanitizer 退出 99：

```text
========= Program hit cudaErrorNoKernelImageForDevice (error 209)
========= due to "no kernel image is available for execution on the device"
========= on CUDA API call to cudaFuncGetAttributes.
========= Host Frame: ncclInitKernelsForDevice(...) in enqueue.cc:40
...
========= ERROR SUMMARY: 32 errors
```

完整日志：[第一次最终 memcheck](../benchmark_logs/final-memcheck-uniform-bwd-2153894.log)。

判断依据：

- 32 条全部是 `cudaFuncGetAttributes`。
- host stack 全部在 `ncclInitKernelsForDevice`。
- 没有 `Invalid __global__ read/write`、`Misaligned Address` 或目标 kernel device frame。
- benchmark 正常完成。

NCCL 初始化会探测多个预编译 kernel image 是否适合当前设备；`cudaErrorNoKernelImageForDevice` 可以是 capability probing 的预期返回，但 memcheck 默认把显式 API 错误计入 summary。

修正：确认噪声来源后添加：

```text
--report-api-errors no
```

重跑结果：

```text
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

最终日志：[定向 kernel memcheck](../benchmark_logs/final-kernel-memcheck-uniform-bwd-2160742.log)。

通用经验：

> sanitizer 非零退出必须逐条分类。不要因为 summary 非零就修改 kernel，也不要因为错误来自依赖库就一律忽略。先确认 API、stack、device frame 和应用行为。

### 4.17 收窄最终修复时，补丁第一次匹配了错误位置

为了只影响 MegaRing，计划写成：

```cpp
if constexpr (!MegaRing) { scheduler_prefetch(); }
// final TMA
if constexpr (MegaRing) { scheduler_prefetch(); }
```

第一次 patch 因上下文太宽，匹配到了函数中更早的第一个 `if (lane_predicate)`，把普通 prefetch 移到了整个 Q/K/V 初始 load 之前。

这个错误没有进入最终版本，因为立即检查：

```bash
git diff -- include/backward/min_fa3_bwd_mainloop.h
nl -ba include/backward/min_fa3_bwd_mainloop.h | sed -n '638,670p'
```

发现位置不对后重新修正。

通用经验：

- 自动 patch 成功只说明文本可应用，不说明语义位置正确。
- 对重复结构多的 CUDA mainloop，修改后必须看带行号上下文。
- 构建和 GPU 测试不能代替 diff review；错误位置有时仍能编译并偶然通过。

---

## 5. 根因分析

### 5.1 最终结论

本次最准确的根因表述是：

> MegaRing backward producer warp 在最后一次 dO/dPsum TMA load 之前执行复杂、warp-wide 的 scheduler ticket claim/decode。该高复杂度代码与 Hopper TMA 的 uniform-register operand 区域紧邻，形成了一个对 ptxas 指令调度/UR 复用敏感的 codegen 脆弱区。偶发 fault 最终落在第二条 dO `UTMALDG.4D`。将 MegaRing scheduler prefetch 排到最后两条 dO TMA 之后后，SASS 中两段 UR 使用被严格隔离，原生 stress 和定向 memcheck 均通过。

证据等级：

- fault 指令、CTA 类型、合法静态坐标：**已证实**。
- 与 mega-copy fence 无关：**已证实**，因为 G1 不执行远端 copy，且 fault 在 compute CTA。
- scheduler/TMA 邻接是触发所需的 codegen 条件：**强支持**，由最小排序修改和前后 SASS/stress 支持。
- fault 瞬间 `UR8` 的具体错误值：**未直接证实**。
- ptxas bug 还是 H100 硬件 erratum：**未证实**。

### 5.2 Hopper TMA 为什么对 alignment 敏感

Hopper Tensor Memory Accelerator (TMA) 用 tensor map descriptor 描述多维 global tensor。一次 `UTMALDG.4D` 大致依赖：

- tensor map descriptor；
- 多维 coordinate；
- shared-memory destination；
- transaction barrier/mbarrier；
- copy shape、element type 和 swizzle/layout。

这些对象有严格的范围和对齐约束。例如，shared destination、barrier address、descriptor 本身以及 descriptor 推导出的 global address都必须满足指令要求。任一 operand 损坏，都可能产生 `Warp Misaligned Address`。

本案例的关键不是“bf16 元素地址必须 128-byte 对齐”这种简单规则，而是：一条异步 tensor load 的 descriptor/coordinate/shared destination 组合必须整体合法。

### 5.3 Hopper uniform registers 与 producer warp

Hopper SASS 使用 `UR0..URn` 这类 uniform register 保存 warp-uniform 值。它们适合 TMA descriptor、shared offset、branch target 和 warp-wide 控制信息。

本 kernel 是 warp-specialized：

- warp group 0 是 producer；
- producer 内 warp 0 负责 K/V、Q/dO TMA；
- 其他 producer warp/consumer warp 负责 dQ store 和 WGMMA；
- scheduler 在这些 role 之间通过 named barriers 发布 work。

`scheduler_prefetch()` 并不是 host 侧异步预取，而是在 producer warp 内 claim/decode 下一块 work，并修改 `work_tile_info`。对 MegaRing 来说，它执行大量 warp-wide scan 和分支，可能扩大 live range 和 UR 压力。

### 5.4 为什么错误看起来是“偶发”的

源码和 cubin 对同一个构建是确定的，但以下因素仍会让观察表现偶发：

- 不同 CTA/SM 的运行时进度不同；
- persistent scheduler 的 ticket claim 顺序依赖原子操作进度；
- 不同 work tile 走不同 decode hint/section 路径；
- TMA pipeline 和 producer/consumer 的相对进度不同；
- CUDA 错误异步上报，不同 rank 可能先在不同同步点观察到；
- sanitizer、blocking 和外部负载改变调度窗口。

“偶发”不一定意味着机器码随机变化，更常见的是确定机器码只在特定动态路径/状态组合下暴露问题。

### 5.5 为什么 NCCL stack 不是根因

PyTorch distributed 的 NCCL watchdog 周期性查询 CUDA event。当之前某个自定义 kernel 已让 context 进入 error 状态时，event query 会返回同一 sticky CUDA error。

因此：

```text
NCCL watchdog 报 misaligned
```

只证明 watchdog 最先观察到错误，不证明 NCCL kernel 发出了 misaligned access。`CUDA_LAUNCH_BLOCKING=1` 将其定位到 fused backward launch，core 的 kernel 名也属于 `_min_fa3_op.so`。

### 5.6 为什么不是显存 OOM、显存容量或设备不一致

- 错误类型是 `cudaErrorMisalignedAddress`，不是 `cudaErrorMemoryAllocation`。
- coredump 有精确 device exception 和 fault SASS。
- shape、dtype、device 在 benchmark 中固定且 kernel 已运行多次。
- fault 位于 TMA issuance，不是 host 到错误 device 的 tensor copy。
- 最终修复不改变 allocation 大小、显存池或 device mapping，却消除了问题。

仍应在通用排查中检查这些项目，因为不同 bug 可能在后续同步点表现相似。

### 5.7 为什么不是 mega-copy fence

三条独立证据：

1. G1 workload 没有远端 KV step，copy/fence 路径不执行。
2. `blockIdx.x=54 < num_comp_sm=128`，fault CTA 是 compute CTA。
3. error PC 是本地 dO TMA `UTMALDG.4D`，不是 communication load/store/fence 指令。

因此修改 fence、barrier ID 或 communication protocol 不但缺乏证据，还会扩大回归面。

### 5.8 为什么移动 prefetch 有效

修复前逻辑：

```text
循环内 Q/dO pipeline loads
    -> scheduler_prefetch(next work)
    -> 最后一次 dO TMA + dPsum bulk copy
```

修复后 MegaRing：

```text
循环内 Q/dO pipeline loads
    -> 最后一次 dO TMA + dPsum bulk copy
    -> scheduler_prefetch(next work)
```

它不改变：

- 取到的 next work ticket；
- 当前 tile 的 dO 坐标；
- pipeline state 的最终值；
- named barrier ID；
- mbarrier transaction bytes；
- 通信 fence；
- tensor layout。

它只改变 scheduler decode 与最后 TMA 的局部机器码邻接关系。新 SASS 证明 scheduler UR 写入出现在两条 TMA 之后。

### 5.9 为什么不能直接宣布“这是 ptxas bug”

要严谨证明 compiler bug，至少还需要：

1. 独立于整个 PyTorch/FlashAttention 的最小 PTX/CUDA 用例；
2. 明确展示语义合法输入产生错误 SASS 或错误 runtime 行为；
3. CUDA 12.8、其他 toolkit 版本的对照；
4. 多个 driver 和多台 H100 的对照；
5. 必要时 NVIDIA Compute Sanitizer/cuda-gdb 的完整 reproducer。

当前证据足以做工程修复，但不足以把责任确定到 ptxas 或 silicon。面向开发者应写“codegen scheduling hazard/workaround”，而不是未经验证的“CUDA compiler bug”。

---

## 6. 修复方案

### 6.1 临时 workaround

最初隔离实验把所有 backward specialization 的 prefetch 都移到最后 TMA 后：

```diff
-        scheduler_prefetch();
         if (lane_predicate) {
             PipelineState_dO smem_pipe_write_do_cur = ...;
             pipeline_do.producer_acquire(smem_pipe_write_do_cur);
             copy(params.tma_load_dO.with(...), ...);
             copy(bulk_copy.with(...), ...);
             ...
         }
+        scheduler_prefetch();
```

优点：

- 改动最小，适合快速验证因果关系。
- 新 SASS 明确隔离了目标区域。

缺点：

- 影响普通 backward，而普通路径没有观察到该问题。
- 偏离上游普通 FA3 的 prefetch 顺序。
- 扩大性能和正确性回归面。

所以它是实验性 workaround，不是最终提交方案。

### 6.2 最终修复

最终只对 `MegaRing=true` 延后 prefetch，普通 backward 保持原顺序。

文件：[`include/backward/min_fa3_bwd_mainloop.h`](../include/backward/min_fa3_bwd_mainloop.h)

完整 diff：

```diff
diff --git a/include/backward/min_fa3_bwd_mainloop.h b/include/backward/min_fa3_bwd_mainloop.h
index f7c80d2..b3d4c7e 100644
--- a/include/backward/min_fa3_bwd_mainloop.h
+++ b/include/backward/min_fa3_bwd_mainloop.h
@@ -648,7 +648,7 @@ struct CollectiveMainloopBwdSm90 {
                      gLSE(_, m_block + 1), sLSE(_, smem_pipe_write.index()));
             }
         }
-        scheduler_prefetch();
+        if constexpr (!MegaRing) { scheduler_prefetch(); }
         if (lane_predicate) {
             PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
             pipeline_do.producer_acquire(smem_pipe_write_do_cur);
@@ -659,6 +659,7 @@ struct CollectiveMainloopBwdSm90 {
             if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
             ++smem_pipe_write;
         }
+        if constexpr (MegaRing) { scheduler_prefetch(); }
         if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
     }
```

### 6.3 修复前后代码对比

修复前：

```cpp
// Upstream-style prefetch point. For MegaRing this executes complex decode_work().
scheduler_prefetch();

if (lane_predicate) {
    PipelineState_dO smem_pipe_write_do_cur =
        cute::conditional_return<Q_dO_same_stages>(
            smem_pipe_write, smem_pipe_write_do);
    pipeline_do.producer_acquire(smem_pipe_write_do_cur);
    copy(params.tma_load_dO.with(
             *pipeline_do.producer_get_barrier(smem_pipe_write_do_cur),
             mcast_mask_qdo,
             TMA::CacheHintSm90::EVICT_LAST),
         tdOgdO(_, m_block),
         tdOsdO(_, smem_pipe_write_do_cur.index()));
    copy(bulk_copy.with(
             *pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
         gdPsum(_, m_block),
         sdPsum(_, smem_pipe_write_do_cur.index()));
    if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
    ++smem_pipe_write;
}
```

修复后：

```cpp
// Preserve the copied upstream order for ordinary backward specializations.
if constexpr (!MegaRing) { scheduler_prefetch(); }

if (lane_predicate) {
    PipelineState_dO smem_pipe_write_do_cur =
        cute::conditional_return<Q_dO_same_stages>(
            smem_pipe_write, smem_pipe_write_do);
    pipeline_do.producer_acquire(smem_pipe_write_do_cur);
    copy(params.tma_load_dO.with(
             *pipeline_do.producer_get_barrier(smem_pipe_write_do_cur),
             mcast_mask_qdo,
             TMA::CacheHintSm90::EVICT_LAST),
         tdOgdO(_, m_block),
         tdOsdO(_, smem_pipe_write_do_cur.index()));
    copy(bulk_copy.with(
             *pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
         gdPsum(_, m_block),
         sdPsum(_, smem_pipe_write_do_cur.index()));
    if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
    ++smem_pipe_write;
}

// Isolate MegaRing's warp-wide decode from the final dO TMA instructions.
if constexpr (MegaRing) { scheduler_prefetch(); }
```

实际文件没有添加上面两段解释性注释，以保持 copied + trimmed 源码改动最小；这里的注释用于教程说明。

### 6.4 没有做的修改

本次没有：

- 修改 mega-copy fence；
- 修改 named barrier ID 或 participant count；
- 改 TMA tensor layout；
- 增加额外 `__syncthreads()`；
- 降级 CUDA/driver；
- 禁用优化或改成 `-G`；
- 改 dtype/head dim；
- 重写 scheduler；
- 用 host synchronization 掩盖问题。

这使最终修复保持在一个文件、三行 diff，符合最小变更原则。

### 6.5 若排序 workaround 仍失败，下一步是什么

本次没有执行，因为排序后已通过。可复用的下一隔离实验是：

1. 禁用 producer-side prefetch。
2. 在 `get_next_work()` 中同步 claim/decode。
3. 保持 barrier/fence 不变。
4. 比较 SASS 和性能。

这会牺牲 overlap，但能判断问题来自“预取的时间位置”还是 scheduler decode 本身。

---

## 7. 验证方法和结果

### 7.1 构建验证

最终真正重编后的 ptxas 资源：

```text
104 bytes stack frame
480 bytes spill stores
528 bytes spill loads
Used 168 registers
Used 16 barriers
```

检查：

```bash
stat -c '%y %n' \
  include/backward/min_fa3_bwd_mainloop.h \
  build/temp.linux-x86_64-cpython-312/csrc/backward/min_fa3_bwd_launch.o \
  _min_fa3_op.so
git diff --check
```

通过标准：

- NVCC 真实运行，不是 `ninja: no work to do`。
- object 和 `.so` 时间晚于 header。
- `git diff --check` 无 whitespace 错误。
- SASS 中最终 dO TMA 在 MegaRing scheduler decode 前。

### 7.2 8 卡 correctness

```bash
PYTHONPATH=/home/hychen/min_fa3_demo:/home/hychen/min_fa3_demo/scripts/test_mega_ring \
.venv/bin/torchrun --standalone --nproc_per_node=8 \
  scripts/test_mega_ring/mega_ring_test_min_fa3_varlen_backward_hybrid_multi_rank.py \
  --global-seqlens 2048,1024,512,256 \
  --ring-sizes 8,4,2,1 \
  --ring-starts 0,4,2,7 \
  --qhead 16 --kvhead 8 --repeat 2 \
  --num-comp-sm 100 --num-comm-sm 16
```

结果：通过。该用例覆盖：

- G8/G4/G2 的真实跨 rank 通信；
- G1 本地路径；
- forward 输出和 backward gradient reference；
- remote dK/dV owner completion；
- 重复执行。

### 7.3 原故障规模 stress

最终 `context=65536`：

```bash
CUDA_LAUNCH_BLOCKING=1 \
CONTEXT_LENGTHS=65536 BATCH_SIZES=16 \
SM_CONFIGS=128:4 WARMUP_ITERS=50 NUM_ITERS=200 CHECK=0 \
LOG_DIR=benchmark_logs/final-post-prefetch-stress-uniform-bwd \
./benchmark_uniform.sh
```

结果：通过，日志：[`final 65536`](../benchmark_logs/final-post-prefetch-stress-uniform-bwd/benchmark_uniform_backward.log)。

关键结果：

```text
mega_ring_all_cp 128:4
t0=6.684, t1=6.683, t2=6.650, t3=6.639,
t4=6.573, t5=6.540, t6=6.523, t7=6.510
max_across_ranks=6.704 ms
```

最终 `context=262144`：

```bash
CUDA_LAUNCH_BLOCKING=1 \
CONTEXT_LENGTHS=262144 BATCH_SIZES=16 \
SM_CONFIGS=128:4 WARMUP_ITERS=50 NUM_ITERS=200 CHECK=0 \
LOG_DIR=benchmark_logs/final-post-prefetch-262144-uniform-bwd \
./benchmark_uniform.sh
```

结果：通过，日志：[`final 262144`](../benchmark_logs/final-post-prefetch-262144-uniform-bwd/benchmark_uniform_backward.log)。

```text
mega_ring_all_cp 128:4
t0=26.800, t1=26.982, t2=26.756, t3=26.341,
t4=26.149, t5=25.651, t6=25.371, t7=25.255
max_across_ranks=27.020 ms
```

中间版本还覆盖 `context=131072`，同样完成 50 warmup + 200 iterations。

### 7.4 最终定向 memcheck

命令见 3.5 节，最终结果：

```text
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

为什么 memcheck 只跑一轮：sanitizer 下单次运行成本显著放大。它用于精确检查内存访问；偶发性则由原生 50+200 stress 覆盖。两类测试不能互相替代。

### 7.5 验证矩阵建议

以后修复类似问题，至少覆盖：

| 维度 | 最小集合 | 目的 |
|---|---|---|
| Correctness | 小 shape + CPU/PyTorch reference | 防止排序修复改变语义 |
| Boundary | 最后 tile 恰好对齐、差 1、空 tile | 查 offset/predicate |
| Topology | G1、G2、最大 ring | 分离本地和通信路径 |
| Context | 原失败最小/最大规模 | 覆盖资源和 scheduler path |
| SM split | 原失败配置 + 至少一个不同 split | 查 CTA role/scheduling |
| Iterations | 50 warmup + 200 或更多 | 查偶发性 |
| Sanitizer | 目标 kernel memcheck/racecheck | 查内存和 shared race |
| Codegen | `cuobjdump`/`nvdisasm` | 确认修改进入 SASS |
| Performance | 修复前后同机同负载 | 防止 overlap 退化 |

### 7.6 还不能声称什么

- 不能声称所有 H100/driver/toolkit 组合都修复。
- 不能声称严格复现概率变为 0；只能说本验证窗口未复现。
- 不能声称 synccheck 的 named-barrier 报告已解决；本次没有修改那部分。
- 不能声称已证明 NVIDIA compiler/hardware bug。

---

## 8. 可复用 GPU 排查框架

### 8.1 第 0 阶段：保护现场

- [ ] 保存完整 stdout/stderr、退出码和时间戳。
- [ ] 记录 Git commit、dirty diff、构建命令和 binary hash。
- [ ] 记录所有环境变量，尤其是 `CUDA_*`、`NCCL_*`、框架配置。
- [ ] 不立即清空 build、重启节点或覆盖 core。
- [ ] 如果 context 已 sticky error，保存证据后退出进程；不要继续相信同一进程的后续测试。
- [ ] 共享节点先确认进程归属，不终止未知任务。

为什么：GPU bug 的核心证据往往是短暂的；重建或重启会改变 cubin、地址和时序。

### 8.2 第 1 阶段：硬件和进程健康

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
nvidia-smi -q -d ECC,ERROR,POWER,TEMPERATURE,CLOCK,PERFORMANCE
sudo dmesg -T | grep -Ei 'NVRM|Xid|ECC|AER|PCIe'
```

- [ ] GPU 是否存在、型号是否符合目标架构。
- [ ] driver 是否加载，是否有 Xid。
- [ ] ECC 是否增长。
- [ ] 是否有其他 GPU 进程。
- [ ] 显存是否接近耗尽。
- [ ] 温度、功耗、时钟是否异常。
- [ ] PCIe/NVLink 拓扑是否变化。
- [ ] MIG/compute mode 是否符合预期。

判断：

- 同一物理卡反复 Xid/ECC：优先硬件/驱动。
- rank 漂移、ECC 为 0、精确 kernel exception：优先软件 kernel。
- 所有卡同时掉线：查 driver reset、电源、PCIe fabric。

### 8.3 第 2 阶段：确认真实错误边界

```bash
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 python repro.py
```

在 C++/CUDA extension 每个可疑 launch 后：

```cpp
kernel<<<grid, block, smem, stream>>>(...);
C10_CUDA_KERNEL_LAUNCH_CHECK();
// 隔离阶段可临时 cudaStreamSynchronize(stream)，正式代码不一定保留。
```

- [ ] 错误在哪一个 launch 后首次出现。
- [ ] 是 launch configuration error，还是 kernel 执行期 fault。
- [ ] NCCL/框架 stack 是根因还是观察点。
- [ ] 错误是否 sticky，后续 API 是否只是连带失败。

### 8.4 第 3 阶段：审计算子输入输出

对每个 tensor 记录：

```python
def dump_tensor(name, x):
    print(
        name,
        "shape=", tuple(x.shape),
        "stride=", x.stride(),
        "dtype=", x.dtype,
        "device=", x.device,
        "contiguous=", x.is_contiguous(),
        "storage_offset=", x.storage_offset(),
        "ptr=", hex(x.data_ptr()),
        "ptr_mod_16=", x.data_ptr() % 16,
        "ptr_mod_128=", x.data_ptr() % 128,
    )
```

- [ ] shape 与 kernel template 一致。
- [ ] stride/layout 与 descriptor 一致。
- [ ] dtype 与 element type 一致。
- [ ] tensor 在正确 device。
- [ ] storage lifetime 覆盖异步 kernel。
- [ ] base pointer 和每一维 stride 满足 vector/TMA alignment。
- [ ] offset、cu_seqlens、index tensor 的 host/device 副本一致。
- [ ] output buffer 没有 alias 不允许 alias 的 input。
- [ ] empty/last/partial tile predicate 正确。
- [ ] grid/block/shared-memory size 不越硬件限制。

### 8.5 第 4 阶段：建立最小可复现矩阵

- [ ] 单卡 vs 多卡。
- [ ] 单 rank vs 8 rank。
- [ ] G1（无通信）vs G>1（有通信）。
- [ ] forward vs backward。
- [ ] compute CTA vs communication CTA。
- [ ] 小 shape vs 原失败 shape。
- [ ] 固定 seed vs 多 seed。
- [ ] blocking vs non-blocking。
- [ ] release vs `-lineinfo` vs `-G`。
- [ ] 原生运行 vs sanitizer。

每次只改变一个维度，否则无法做因果判断。

### 8.6 第 5 阶段：sanitizer 分层

推荐顺序：

1. `memcheck`：越界、misaligned、use-after-free。
2. `initcheck`：未初始化 global memory。
3. `racecheck`：shared-memory hazard、部分 async copy race。
4. `synccheck`：barrier/warp synchronization misuse。

模板：

```bash
compute-sanitizer \
  --tool memcheck \
  --target-processes all \
  --kernel-name kernel_substring=my_kernel \
  --error-exitcode 99 \
  --log-file sanitizer-%p.log \
  python repro.py
```

检查原则：

- [ ] 看 summary，也看第一条完整 device frame。
- [ ] 区分 target kernel、依赖库和 API probing。
- [ ] 多进程用 `%p` 分日志。
- [ ] 用 kernel filter 降噪。
- [ ] sanitizer 通过后仍运行原生 stress。

### 8.7 第 6 阶段：coredump、cuda-gdb 和 SASS

当 bug 偶发、memcheck 抓不到，但原生运行能触发时：

- [ ] 启用 GPU coredump。
- [ ] 保存产生 core 的精确 `.so`/cubin。
- [ ] 记录 `$errorpc`、kernel、grid、block、thread、SM、warp、lane。
- [ ] 根据 `blockIdx` 映射 CTA role。
- [ ] 反汇编 error PC 前后至少 32 条指令。
- [ ] 检查 fault 指令 operand 的地址、descriptor、coordinate、predicate。
- [ ] 对照 source、PTX、SASS，避免只凭某一层下结论。
- [ ] 修改后重新提取 SASS，确认 codegen 真变了。

### 8.8 第 7 阶段：假设管理

建议维护表格：

| 假设 | 预测 | 实验 | 结果 | 状态 |
|---|---|---|---|---|
| mega-copy fence 错 | G1 不执行 copy 时不应失败 | G1 stress/core | 仍失败 | 排除 |
| 固定 GPU 硬件错 | 总在同一卡/rank，伴随 Xid/ECC | 多次日志/ECC | rank 0/5/6 漂移，当前 ECC 0 | 弱化；incident dmesg 待补 |
| dO 静态越界 | 坐标越界且稳定失败 | core 解码坐标 | `{64,8128,15,0}` 合法 | 排除简单越界 |
| scheduler/TMA 邻接 | 移开 decode 后 fault 消失 | source reorder + SASS + stress | 通过 | 强支持 |
| spill 总量导致 | 修复后 spill 应下降 | ptxas resource | stack/spill 未下降 | 排除“靠降 spill 修复” |

### 8.9 第 8 阶段：修复后验证

- [ ] 小 shape correctness reference。
- [ ] 原失败 shape。
- [ ] 原失败 rank/topology 配置。
- [ ] 多次 stress。
- [ ] sanitizer。
- [ ] 性能回归。
- [ ] 非目标 specialization 不改变 codegen/行为。
- [ ] clean/incremental build 都能包含修改。
- [ ] GPU/Xid/ECC 保持健康。

---

## 9. 常见 GPU Bug 分类与工具

| 类别 | 常见症状 | 优先检查 | 主要工具 |
|---|---|---|---|
| OOM | `CUDA out of memory`、allocation failed | allocated/reserved、碎片、峰值、其他进程 | `nvidia-smi`、framework memory summary、Nsight Systems |
| 非法/越界访问 | `illegal memory access`、error 700 | index、shape、stride、lifetime、last tile | memcheck、DSA、cuda-gdb、coredump |
| 未对齐访问 | `misaligned address`、Warp Misaligned Address | vector/TMA operand、base/stride/shared/barrier alignment | memcheck、cuda-gdb、SASS、pointer `% alignment` |
| Use-after-free | 异步、偶发、后续 API 报错 | stream lifetime、allocator reuse、IPC ownership | memcheck、stream sync、allocation logging |
| 数据竞争 | 偶发错误或精度漂移 | shared/global ownership、atomic、barrier、async copy | racecheck、synccheck、重复 deterministic test |
| 同步错误 | deadlock、barrier divergence、hang | participant count、warp role、phase、stream/event | synccheck、cuda-gdb、Nsight Systems |
| 未初始化内存 | NaN、随机输出、rank 漂移 | buffer reset、padding、predicate | initcheck、memcheck、fill sentinel |
| 精度异常 | loss/gradient NaN/Inf、误差放大 | dtype、accumulator、reduction order、overflow | reference、anomaly detection、check numerics、NCU |
| 驱动/toolkit 不兼容 | no kernel image、invalid device function、load failure | driver、SM target、fatbin、framework CUDA | `nvidia-smi`、`nvcc -V`、`cuobjdump -lelf` |
| ECC/硬件错误 | Xid、DBE、GPU fallen off bus | ECC、Xid、AER、固定物理卡 | `nvidia-smi -q`、`dmesg`、DCGM |
| 过热降频 | 性能抖动、throttle reason | 温度、clock、power cap | `nvidia-smi dmon`、`nvidia-smi -q -d PERFORMANCE` |
| 电源不足 | Xid、掉卡、重负载复位 | power draw、PSU/BMC、全机同时失败 | `nvidia-smi dmon`、BMC/IPMI、kernel log |
| Kernel 启动配置错 | invalid configuration、too many resources | grid/block、dynamic smem、register、cluster | launch check、occupancy API、ptxas、NCU |
| NCCL/网络错误 | timeout、unhandled system error、hang | rank mapping、NIC、NVLink、IB、async root error | `NCCL_DEBUG=INFO`、nsys、nccl-tests、topology |
| 编译器/codegen 敏感 | 只在 `-O3`/特定 toolkit 偶发 | PTX/SASS、register/spill、source reorder | cuobjdump、nvdisasm、toolkit matrix、最小 PTX |

### 9.1 OOM 不只是“总量不够”

还要区分：

- framework allocated vs reserved；
- 大块连续分配失败；
- CUDA graph/private pool；
- NCCL buffer 和 IPC allocation；
- 其他进程占用；
- 内存泄漏导致逐步增长。

PyTorch：

```python
print(torch.cuda.memory_summary())
print(torch.cuda.memory_stats())
```

### 9.2 精度问题的正确顺序

1. 固定 seed 和 deterministic 配置。
2. 用 FP32/CPU/框架 reference。
3. 分别比较 forward、dQ、dK、dV。
4. 检查 first bad layer/first bad tile，而不是只看最终 loss。
5. 检查 NaN/Inf、max abs、max relative、ULP。
6. 再用 NCU 分析 tensor core dtype、reduction 和 atomic 行为。

### 9.3 硬件故障的判据

更支持硬件问题的证据：

- 同一物理 GPU/PCI bus 持续失败；
- ECC aggregate/volatile 递增；
- `dmesg` 有 Xid 48/63/79 等或 PCIe AER；
- 与具体模型/shape 无关；
- 换卡后故障跟着卡走；
- DCGM diagnostics 失败。

更支持软件问题的证据：

- error PC 可稳定映射到自定义 kernel；
- shape/topology/codegen 改变可控制复现；
- 换 rank/SM 仍在同一 SASS 指令类型；
- 最小源码排序修改后消失；
- ECC/Xid 无异常。

---

## 10. 工具命令速查表

### 10.1 `nvidia-smi`

基本状态：

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

进程：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
nvidia-smi pmon -s um -d 1
```

持续监控功耗、利用率、时钟、显存、ECC、温度：

```bash
nvidia-smi dmon -s pucvmet -d 1
```

详细健康状态：

```bash
nvidia-smi -q -d ECC,ERROR,POWER,TEMPERATURE,CLOCK,PERFORMANCE
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used --format=csv
```

拓扑：

```bash
nvidia-smi topo -m
nvidia-smi nvlink --status
```

### 10.2 `dmesg` / kernel journal

```bash
sudo dmesg -T | grep -Ei 'NVRM|Xid|ECC|AER|PCIe|fallen off'
sudo journalctl -k -b | grep -Ei 'NVRM|Xid|ECC|AER|PCIe'
sudo journalctl -k --since '10 minutes ago'
```

重点：NVIDIA Xid、PCIe AER、GPU reset、ECC DBE、fallen off bus。

### 10.3 Compute Sanitizer

Memcheck：

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 python repro.py
```

多进程 + kernel filter：

```bash
compute-sanitizer \
  --tool memcheck \
  --target-processes all \
  --kernel-name kernel_substring=my_kernel \
  --log-file memcheck-%p.log \
  --error-exitcode 99 \
  torchrun --nproc_per_node=8 repro.py
```

Racecheck：

```bash
compute-sanitizer \
  --tool racecheck \
  --racecheck-report all \
  --kernel-name kernel_substring=my_kernel \
  python repro.py
```

Synccheck：

```bash
compute-sanitizer \
  --tool synccheck \
  --kernel-name kernel_substring=my_kernel \
  python repro.py
```

Initcheck：

```bash
compute-sanitizer --tool initcheck python repro.py
```

限定匹配 launch：

```bash
compute-sanitizer \
  --kernel-name kernel_substring=my_kernel \
  --launch-skip 10 \
  --launch-count 1 \
  python repro.py
```

### 10.4 cuda-gdb 与 GPU coredump

在线调试：

```bash
CUDA_DEVICE_WAITS_ON_EXCEPTION=1 cuda-gdb --args python repro.py
```

GPU core：

```bash
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_FILE='core_%h_%p.nvcudmp'
export CUDA_COREDUMP_SHOW_PROGRESS=1
python repro.py
cuda-gdb -c core_hostname_pid.nvcudmp
```

cuda-gdb 常用：

```gdb
set cuda api_failures stop
info cuda devices
info cuda contexts
info cuda kernels
info cuda blocks
info cuda threads
cuda kernel 0 block 0 thread 0
print/x $errorpc
x/32i $errorpc-0x100
info registers
bt
```

不同 cuda-gdb 版本的 CUDA focus 命令略有差异，以 `help cuda` 为准。

### 10.5 `cuobjdump` / `nvdisasm`

```bash
cuobjdump --list-elf extension.so
cuobjdump --dump-resource-usage extension.so
cuobjdump --dump-sass extension.so > extension.sass
cuobjdump --extract-elf all extension.so
nvdisasm -g -gi extracted.sm_90a.cubin > extracted.sass
nvdisasm -plr extracted.sm_90a.cubin > live_ranges.sass
```

用途：

- 确认 fatbin 是否包含目标 SM；
- 看 register、stack、spill、barrier；
- 映射 error PC；
- 比较修复前后机器码顺序。

### 10.6 Nsight Systems (`nsys`)

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,nccl \
  --sample=none \
  --force-overwrite=true \
  -o repro_timeline \
  python repro.py
```

适合：

- 找最后一个成功 kernel；
- 看 stream/event/NCCL overlap；
- 查同步和 host gap；
- 判断错误是否在 NCCL 前已经发生。

### 10.7 Nsight Compute (`ncu`)

```bash
ncu \
  --set full \
  --kernel-name regex:my_kernel \
  --launch-count 1 \
  -o my_kernel_report \
  python repro.py
```

适合：

- occupancy、register、spill；
- memory transaction、cache、bank conflict；
- source/SASS 对照；
- warp stall 和 barrier 行为。

不要一开始就对长时间多卡 benchmark 使用 `--set full`；先缩成单 launch，否则采集成本非常高。

### 10.8 PyTorch 调试

同步和 C++ stack：

```bash
CUDA_LAUNCH_BLOCKING=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
python repro.py
```

Autograd anomaly detection：

```python
import torch

torch.autograd.set_detect_anomaly(True)
# 或局部：
with torch.autograd.detect_anomaly(check_nan=True):
    loss.backward()
```

显存：

```python
print(torch.cuda.memory_summary())
print(torch.cuda.max_memory_allocated())
print(torch.cuda.max_memory_reserved())
```

分布式：

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
torchrun ...
```

注意：环境变量支持范围随 PyTorch/NCCL 版本变化，应记录实际版本。

### 10.9 TensorFlow 调试

数值检查：

```python
import tensorflow as tf

tf.debugging.enable_check_numerics()
```

TensorBoard Debugger V2 dump：

```python
tf.debugging.experimental.enable_dump_debug_info(
    "/tmp/tfdbg2",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1,
)
```

设备放置：

```python
tf.debugging.set_log_device_placement(True)
```

自定义 TensorFlow CUDA op 仍应结合 Compute Sanitizer、`CUDA_LAUNCH_BLOCKING=1`、coredump 和 SASS。

### 10.10 性能/硬件长期监控

可选工具：

- NVIDIA DCGM / `dcgmi`：健康诊断、ECC、Xid、field watch。
- Prometheus DCGM Exporter：集群告警。
- BMC/IPMI：整机电源、风扇、温度、电源冗余。
- `nccl-tests`：隔离 NCCL/NVLink/IB。
- `ibstat`、`ibv_devinfo`：InfiniBand 状态。

---

## 11. 预防建议与最佳实践

### 11.1 CUDA/CUTLASS 编码习惯

- 每个 custom kernel launch 后保留 launch check。
- 对 vector/TMA 输入做 host 侧 shape、dtype、stride、alignment 校验。
- 用 `static_assert` 固化 tile/head-dim/architecture 约束。
- last tile、empty tile、partial tile 单独写测试。
- descriptor shape 与实际 storage shape 在一个地方构造，避免重复推导。
- 对异步 TMA 明确记录 mbarrier transaction bytes 和 phase。
- 对 named barrier 建立唯一 ID/participant count 表，避免复用冲突。
- 对 producer/consumer role 写清线程范围，不让条件分支隐式改变参与者。
- 不在 TMA operand 构造附近无必要地内联大型 warp-wide decode；必要时通过函数边界、源级排序或 noinline 实验控制 codegen。
- 不把 `__syncthreads()` 当通用修复。它可能死锁 warp-specialized kernel，也可能只掩盖 bug。

### 11.2 单元测试

每个算子至少测试：

- supported dtype/shape 的最小值和最大值；
- 连续和允许的非连续 layout；
- GQA/MQA head ratio；
- causal/noncausal；
- fixed/varlen；
- sequence length 为 tile、tile±1；
- batch 中包含不同长度、空或极短 segment；
- 输出 buffer 复用和 alias contract；
- deterministic 模式重复一致；
- 多 seed；
- 多 GPU topology。

### 11.3 GPU CI

建议分层：

**每个 PR：**

- build from clean；
- incremental header rebuild test；
- 小 shape correctness；
- `git diff --check`；
- 至少一个目标 GPU smoke test。

**Nightly：**

- memcheck target kernel；
- racecheck/synccheck 的受控 filter；
- 200+ iteration stress；
- 多 seed、多个 SM split；
- performance threshold；
- ECC/Xid pre/post check。

**Release：**

- driver/toolkit/framework matrix；
- clean node 和真实分布式拓扑；
- coredump capability smoke；
- binary resource/SASS archive；
- 长时 soak。

### 11.4 修复构建依赖

本次暴露出 CUDA object 的 Ninja dependency record 为 `#deps 0`。建议单独修复构建系统：

- 确保 NVCC depfile 被 Ninja 正确读取；
- CI 中修改一个核心 header 后断言 CUDA object mtime/hash 改变；
- clean build 与 incremental build 的 `.so` hash 应一致；
- 构建日志保存完整 NVCC command。

否则开发者可能验证的是旧 cubin，得到完全错误的结论。

### 11.5 环境版本锁定

保存：

- driver package/version；
- CUDA toolkit patch version；
- PyTorch wheel build tag；
- NCCL/cuDNN version；
- CUTLASS/ThunderKittens commit；
- compiler 和 glibc；
- compile flags；
- GPU SKU/VBIOS；
- container digest 或 lockfile。

不要只写“CUDA 12”或“PyTorch latest”。GPU codegen bug 对 patch 版本和 flags 都可能敏感。

### 11.6 可观测性和告警

生产/集群至少告警：

- Xid；
- corrected/uncorrected ECC delta；
- GPU fallen off bus；
- 温度和 throttle reason；
- power limit/throttle；
- 显存持续增长；
- NCCL timeout；
- kernel error rate；
- 节点 GPU 型号/driver 漂移。

### 11.7 日志和 coredump 管理

- 每次运行使用唯一 timestamp/run ID。
- 记录 rank、local rank、GPU UUID/PCI bus。
- 错误时保存 input metadata，不必保存敏感完整 tensor。
- core 与对应 `.so`/cubin/commit 绑定归档。
- 设置磁盘配额和 core 生命周期。
- 对 1 GiB 以上 core 优先放到专用 scratch。

### 11.8 与 NVIDIA/上游沟通时的最小材料

1. 可运行 reproducer。
2. 精确环境矩阵。
3. fault coredump。
4. 对应 cubin 和 SASS。
5. `$errorpc`、CTA/thread/warp。
6. 输入 shape/stride/alignment。
7. 修复前后最小 diff。
8. 修复前后 SASS 对照。
9. 原生复现概率和 sanitizer 结果。
10. 是否跨 toolkit/driver/GPU 复现。

---

## 12. 仍待补充的信息

为了把本文升级为可对外提交的完整 incident report，还需要：

1. **[待补充] 修复前复现概率**

   需要在固定空闲节点、固定旧 `.so`、固定 seed 下循环至少 20 次，记录成功/失败次数、rank、GPU UUID 和 context。

2. **[待补充] 故障时段 Xid/ECC/AER 日志**

   需要管理员执行：

   ```bash
   sudo journalctl -k --since '2026-08-13 13:20:00' --until '2026-08-13 14:15:00'
   sudo dmesg -T | grep -Ei 'NVRM|Xid|ECC|AER|PCIe'
   ```

3. **[待补充] 上游 FA3 精确 commit**

   需要提供本目录 backward mainloop/scheduler 最初复制自哪个仓库和 commit，以便确认 upstream 是否已有对应修复。

4. **[待补充] 跨版本矩阵**

   至少测试 CUDA 12.8 当前 patch、另一个 12.x toolkit、另一个 driver。若只有 12.8 复现，compiler hypothesis 会更强。

5. **[待补充] 独立 PTX/CUDA 最小用例**

   当前 reproducer依赖 PyTorch、CUTLASS、NCCL 和整个 FlashAttention kernel。若要证明 compiler/hardware erratum，需要将 TMA + warp-wide decode 缩成单 kernel。

6. **[待补充] synccheck named-barrier 报告的独立结论**

   当前未过滤 synccheck 报告大量 divergence。它没有映射到本次 error PC，也没有因本修复而改变，但应建立单独的 participant-count/role 审计和定向 synccheck 任务。

---

## 13. 10 分钟快速定位 GPU 问题

下面按优先级给出一个不依赖本项目的快速流程。

### 0-1 分钟：保存现场，不要先重启

```bash
date
git rev-parse HEAD
git status --short
env | sort | grep -E '^(CUDA|NCCL|TORCH|TF|CUDNN)'
nvidia-smi
```

最先看：

1. 精确 CUDA error code/message。
2. 哪个 rank、哪张 GPU、哪个同步点。
3. 是否有其他进程/显存压力。
4. driver/GPU 是否正常可见。

### 1-2 分钟：查硬件信号

```bash
nvidia-smi -q -d ECC,ERROR,POWER,TEMPERATURE,CLOCK,PERFORMANCE
sudo dmesg -T | grep -Ei 'NVRM|Xid|ECC|AER|PCIe|fallen off'
```

优先级：

- 有 Xid/ECC/fallen off bus：先走硬件/驱动路径。
- 无硬件信号且有精确 kernel error：继续软件算子路径。

### 2-4 分钟：让异步错误靠近真实 launch

```bash
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 python repro.py
```

自定义 extension 在每个可疑 launch 后加 launch check。不要因为 stack 出现在 NCCL、allocator 或 `synchronize()` 就认定它们是根因。

### 4-6 分钟：打印 tensor contract

立即打印：

- shape；
- stride；
- dtype；
- device；
- storage offset；
- contiguous；
- `data_ptr % 16/128`；
- index/cu_seqlens 最小最大值；
- grid/block/dynamic shared memory。

优先检查 last tile、partial tile、负 index、32-bit overflow和生命周期。

### 6-8 分钟：运行定向 memcheck

```bash
compute-sanitizer \
  --tool memcheck \
  --target-processes all \
  --kernel-name kernel_substring=my_kernel \
  --launch-count 1 \
  --error-exitcode 99 \
  python repro.py
```

看第一条完整 device frame，不只看 summary。若 sanitizer 通过但原生仍偶发，继续，不要宣布修复。

### 8-10 分钟：缩小路径并决定下一工具

按顺序回答：

1. 单卡还失败吗？
2. 无通信路径还失败吗？
3. forward/backward 哪个失败？
4. compute/communication 哪类 CTA？
5. 固定 shape/seed 是否仍漂移 rank？
6. blocking 与 non-blocking 是否都失败？

下一步选择：

| 现象 | 下一步 |
|---|---|
| memcheck 给出精确越界 | 修 index/shape/lifetime，补 boundary test |
| 原生失败、memcheck 不失败 | GPU coredump + cuda-gdb + SASS |
| hang/barrier divergence | synccheck + Nsight Systems |
| 数值错但无 CUDA error | reference + anomaly/check numerics + race/initcheck |
| 固定物理卡 + Xid/ECC | 下线卡、DCGM、驱动/硬件支持 |
| no kernel image | 检查 driver、fatbin、SM target、framework CUDA |
| 只在 `-O3`/特定 toolkit | 保存 PTX/SASS，做版本矩阵和最小 codegen reproducer |

### 最重要的十项

1. **不要把异步报错 stack 当真实 fault site。**
2. **先看 error code，再分类。**
3. **记录 GPU UUID/rank/shape/seed/commit/binary。**
4. **检查 Xid/ECC 和其他 GPU 进程。**
5. **用 `CUDA_LAUNCH_BLOCKING=1` 缩边界。**
6. **审计 shape/stride/dtype/device/alignment/lifetime。**
7. **sanitizer 要过滤目标 kernel，多进程要 `--target-processes all`。**
8. **sanitizer 0 errors 不能排除偶发时序/codegen bug。**
9. **偶发原生 fault 要保存 core、cubin 和 `$errorpc`。**
10. **修复后确认 CUDA object 真的重编，并跑 correctness + 原生 stress + sanitizer。**

---

## 附录 A：本次证据文件索引

| 文件 | 内容 |
|---|---|
| [`20260813-132129` 原日志](../benchmark_logs/20260813-132129-uniform-backward/benchmark_uniform_backward.log) | 65536/rank 5 异步错误 |
| [`20260813-132714` 原日志](../benchmark_logs/20260813-132714-uniform-backward/benchmark_uniform_backward.log) | 262144/rank 6 NCCL watchdog 观察错误 |
| [coredump stress 日志](../benchmark_logs/coredump-log-stress-uniform-bwd.log) | blocking launch、rank 0、line 686 |
| [cuda-gdb summary](../benchmark_logs/cuda-gdb-coredump-summary.log) | Warp Misaligned、block 54、device/SM/warp |
| [cuda-gdb error PC](../benchmark_logs/cuda-gdb-errorpc.log) | 第二条 `UTMALDG.4D` |
| [旧 SASS](../benchmark_logs/min_fa3_bwd_launch.sass) | TMA/scheduler/UR 机器码 |
| [三次旧 memcheck](../benchmark_logs/memcheck-uniform-bwd-1929913.log) | 0 errors，但未排除偶发问题 |
| [未过滤 synccheck](../benchmark_logs/synccheck-uniform-bwd-1952099.log) | named barrier divergence 噪声/独立信号 |
| [定向 racecheck](../benchmark_logs/racecheck-bwd-kernel-uniform-bwd.log) | 0 hazards |
| [NCCL API probing memcheck](../benchmark_logs/final-memcheck-uniform-bwd-2153894.log) | 32 条 `cudaFuncGetAttributes` error 209 |
| [最终 kernel memcheck](../benchmark_logs/final-kernel-memcheck-uniform-bwd-2160742.log) | `ERROR SUMMARY: 0 errors` |
| [最终 65536 stress](../benchmark_logs/final-post-prefetch-stress-uniform-bwd/benchmark_uniform_backward.log) | 50+200 通过 |
| [最终 262144 stress](../benchmark_logs/final-post-prefetch-262144-uniform-bwd/benchmark_uniform_backward.log) | 50+200 通过 |

## 附录 B：一句话复盘

当一个 Hopper kernel 偶发 `misaligned address`、memcheck 又抓不到时，不要围着最后一个 Python/NCCL stack 猜：先用 blocking 找 launch，用 topology 控制流排除未执行路径，用 GPU core 找 error PC，再把 SASS 指令、CTA role、TMA descriptor/坐标和相邻 uniform-register codegen 对起来；修复后必须确认 header 真正触发了 CUDA object 重编，并同时通过 reference、原生 stress 和定向 sanitizer。
