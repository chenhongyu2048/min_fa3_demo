# Mega CP backward 偶发 `misaligned address` 问题记录

## 1. 文档目的

本文记录运行 `benchmark_uniform_backward.sh` 时出现的偶发 CUDA
`misaligned address` 错误，包括：

- benchmark 中可见的故障现象；
- CUDA core 定位到的 kernel、CTA、指令和源码位置；
- 故障 TMA 指令各操作数的恢复结果与对齐检查；
- Mega CP、Mega DCP 以及共享代码路径之间的关系；
- 已排除或证据较弱的原因；
- 当前最可能的根因和后续判别实验。

本文是阶段性分析记录，不把尚未通过对照实验验证的推断写成确定事实。

分析过程中没有运行 GPU kernel、没有重新编译，也没有修改 CUDA 源码。

## 2. 相关文件与环境

### 2.1 输入和产物

| 项目 | 路径或值 |
| --- | --- |
| benchmark 脚本 | `benchmark_uniform_backward.sh` |
| 原始故障日志 | `benchmark_logs/20260813-080737-uniform-backward/benchmark_uniform_backward.log` |
| CUDA core | `/tmp/min-fa3-core.zkrh-58.1353963.nvcudmp` |
| 与 core 匹配的二进制 | `_min_fa3_op.so` |
| 故障日期 | 2026-08-13 |
| 主机 | `zkrh-58` |

当前 `_min_fa3_op.so` 与 CUDA core 匹配。在完成后续 core/SASS 分析前，不能用新编译产物覆盖该文件，否则 CUDA core 中的 PC、符号和 cubin 将无法可靠对应。

本文使用两类证据，必须区分：

- 原始 benchmark 日志中的首个失败进程 PID 是 `1246013`；
- CUDA core 文件名中的 PID 是 `1353963`，来自后续捕获到的同类复现。

因此，原始日志可以证明 B=8 case 的 rank 6 在同步点观察到 `misaligned address`；CUDA core 可以精确证明后续复现故障位于 Mega backward 的第二条 dO TMA。两者的错误类型、运行对象和输入上下文相符，本文据此将它们作为同一问题族分析，但不会把后续 core 的 PC 当作原始 PID 1246013 现场的直接取证结果。

### 2.2 CUDA 环境

故障环境中记录到的工具链为：

```text
GPU:    NVIDIA H100 80GB HBM3 (SM90)
driver: 590.48.01
nvcc:   12.8.61
ptxas:  12.8.61
ptxas build date: 2025-01-15
```

### 2.3 用户提供的先验现象

1. Mega CP forward 可以连续正常运行 50 个迭代。
2. 加入 Mega DCP 之前，Mega CP backward 可以正常运行。
3. 错误不是每次必现，需要多次运行 Mega CP backward 才能遇到。
4. 初步怀疑对象是 `mega_ring_all_cp` 和 `mega_ring_hybrid` 两个 Mega CP backward 入口，以及它们与 Mega DCP 共享或受其影响的部分。

这些现象说明问题具有明显的 backward 特异性和偶发性。它们不直接证明 Mega DCP kernel 在运行时破坏了 Mega CP，但说明应重点检查加入 DCP 后的重新编译、fatbin/module 形态变化，以及 Mega backward 的共享 TMA/mainloop 路径。

## 3. 结论摘要

### 3.1 已经确认的事实

1. 后续 CUDA core 捕获的同类复现中，设备端异常发生在 `mega_ring_flash_attn_bwd_kernel<..., NumDevices=8>` 中。
2. 故障设备为 device 2，故障 CTA 为 `blockIdx.x=49`，而该次 launch 的配置为：

   ```text
   gridDim.x   = 132
   blockDim.x  = 384
   num_comp_sm = 128
   num_comm_sm = 4
   ```

3. CTA 49 满足 `49 < num_comp_sm`，因此它是 compute CTA，不会进入位于 grid 尾部的 Mega backward communication helper。故障不是 communication CTA 直接执行了未对齐的远端 load/store。
4. 精确故障指令是第二条 dO TMA load：

   ```text
   kernel + 0x1f8e0:
   UTMALDG.4D [UR8], [UR16], desc[UR18]
   ```

   对应 `include/backward/min_fa3_bwd_mainloop.h:655` 的最终 dO tile load。
5. 从 CUDA core 和 SASS 数据流恢复出的 descriptor、shared destination、mbarrier 和 tensor coordinates 都满足硬件对齐要求，访问的 tensor tile 也在合法范围内。
6. 故障前存在两条相邻的 dO TMA load。第一条加载 `D=0..63`，第二条加载 `D=64..127`；两条指令使用相同 descriptor、mbarrier、row/head/batch、cache hint 和 pipeline stage。第一条可以执行，而第二条偶发报 `Warp Misaligned Address`。
7. DCP 引入提交 `03679d6` 没有修改 backward kernel 源码；本次 backward benchmark 也不会调用 DCP kernel。
8. 当前 build object 与故障 `_min_fa3_op.so` 中提取的 backward cubin SHA-256 完全一致，排除了 host linker 或 DCP fatbin 聚合改写 backward cubin/SASS 的可能。

### 3.2 当前最可能的解释

当前最符合全部证据的解释是：CUDA 12.8.61 `ptxas` 在 Hopper TMA 指令生成或 uniform-register/scoreboard 依赖处理上存在缺陷。加入 Mega DCP 后触发了重新编译或改变了 CUDA module/fatbin 形态，使原本潜伏的 Mega backward TMA codegen 问题开始暴露；这不等价于 DCP kernel 在运行时直接污染 Mega CP。

这个判断的置信度为“较高但未最终证实”。最终确认仍需要至少一个编译器版本对照实验，例如用包含相应修复的更新 CUDA toolkit 编译同一源码并进行高次数复现。

### 3.3 根因候选排序

| 优先级 | 候选原因 | 当前判断 |
| --- | --- | --- |
| 1 | CUDA 12.8.61 `ptxas` 的 Hopper TMA codegen 或 uniform-register scoreboard bug | 最可能，与合法操作数、固定故障 PC 和偶发性同时吻合 |
| 2 | 加入 DCP 后重新编译/module 形态变化，触发上述 compiler bug | 很可能是变化发生后的触发条件，不是设备端直接污染 |
| 3 | backward scheduler/workspace 重写中的极端 race | 仍未完全排除，但 core 中 work 坐标合法，没有直接证据 |
| 4 | 大 `__grid_constant__` 参数或 descriptor staging 问题 | 理论上可能，当前 descriptor 内容和地址均正常，证据较弱 |
| 5 | 缺失 async proxy fence | 可能影响数值正确性，但不能解释本次合法 operand 上的 misaligned fault |
| 6 | 固定 shared/global 地址实际未对齐 | 已由 core 中操作数及对齐计算基本排除 |
| 7 | DCP kernel 在 benchmark 运行时直接污染 backward | DCP kernel 未被调用且每个 case 是独立进程，基本排除 |

## 4. Benchmark 可见现象

### 4.1 benchmark 配置

默认脚本在 8 张 GPU 上运行 15 个 case：

```text
world_size = 8
contexts   = 65536, 131072, 262144
batches    = 1, 2, 4, 8, 16
QH         = 32
KVH        = 8
D          = 128
mode       = causal backward
warmup     = 10
iters      = 40
check      = false
SM configs = 128:4, 124:8, 120:12, 116:16
```

执行的方法包括：

```text
allgather_attention
llama3_allgather_attention
fa3_ring
megatron_hybrid_cp
magi_attention
zeppelin
mega_ring_all_cp
mega_ring_hybrid
```

脚本对每个 workload 单独执行一次 `torchrun --standalone --nproc_per_node=8`。因此 B=1、B=2、B=4 和 B=8 是四个独立 CUDA 进程组，前面 case 的 CUDA 状态不会延续到故障 case。

### 4.2 前三个 case 正常完成

日志中的前三个 case 均完整打印了所有方法和所有 SM 配置的结果：

```text
[uniform_backward 1/15] context=65536, batch=1, seqlen=65536, hybrid=G8
[uniform_backward 2/15] context=65536, batch=2, seqlen=32768, hybrid=G4
[uniform_backward 3/15] context=65536, batch=4, seqlen=16384, hybrid=G2
```

例如第 3 个 case 的 Mega 结果为：

```text
mega_ring_all_cp                128:4 ... max_across_ranks=6.833 ...
mega_ring_all_cp                124:8 ... max_across_ranks=7.168 ...
mega_ring_all_cp               120:12 ... max_across_ranks=6.952 ...
mega_ring_all_cp               116:16 ... max_across_ranks=7.063 ...
mega_ring_hybrid                128:4 ... max_across_ranks=6.087 ...
mega_ring_hybrid                124:8 ... max_across_ranks=5.923 ...
mega_ring_hybrid               120:12 ... max_across_ranks=5.972 ...
mega_ring_hybrid               116:16 ... max_across_ranks=6.211 ...
```

这说明同一个 `_min_fa3_op.so` 中的 Mega backward 并非固定地址一经执行就必然失败。

### 4.3 第四个 case 发生故障

故障 case 是：

```text
[uniform_backward 4/15] context=65536, batch=8, seqlen=8192, hybrid=G1

Workload: explicit topology, B=8, global_tokens=65536,
global_seqlens=[8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192]

Mega-ring all-CP workload: alignment=2048, global_tokens=65536,
global_seqlens=[8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192]

Hybrid rings: sizes=[1, 1, 1, 1, 1, 1, 1, 1],
starts=[0, 1, 2, 3, 4, 5, 6, 7]
```

rank 6 首先观察到错误：

```text
[rank6]:   File ".../ring_test/benchmark_topology_backward.py", line 629,
[rank6]:     torch.cuda.synchronize()
[rank6]: torch.AcceleratorError: CUDA error: misaligned address
[rank6]: CUDA kernel errors might be asynchronously reported at some other API call,
[rank6]: so the stacktrace below might be incorrect.
```

随后异常清理过程再次 synchronize 并得到同一错误，rank 6 最终以 SIGABRT 退出，其他 rank 被 `torchrun` 发送 SIGTERM。

### 4.4 Python 栈不能确定具体迭代和设备端 PC

`measure_backward_ms()` 的核心流程是：

```python
for _ in range(warmup_iters):
    prepare()
    torch.cuda.synchronize()
    dist.barrier()
    launch()
    torch.cuda.synchronize()

for _ in range(num_iters):
    prepare()
    torch.cuda.synchronize()
    dist.barrier()
    launch()
    torch.cuda.synchronize()  # 日志中的 line 629
```

日志落在 timed loop 的 launch 后同步点，只说明同步时发现此前异步执行的 kernel 已出错。Python 栈本身不能确定 kernel 内部的真实 PC。

此外，benchmark 在一个 workload 的所有 method 完成后才集中打印结果表。第 4 个 case 没有打印 `Method` 表头，不能据此认定错误发生在第一个方法的准备阶段。结合后续同类复现 CUDA core 中的 kernel、`num_comp_sm=128` 和 `num_comm_sm=4`，原始故障最可能也发生在第一个 Mega sweep 配置 `mega_ring_all_cp 128:4` 的某次 warmup/timed backward，但这只是跨复现推断；仅凭原始日志不能严格区分是 `mega_ring_all_cp` 还是 `mega_ring_hybrid` 入口，也不能确定具体迭代。

## 5. 后续同类复现的 CUDA core 精确定位

### 5.1 加载 core

分析使用的只读命令为：

```bash
cd /home/hychen/min_fa3_demo
cuda-gdb -q -batch _min_fa3_op.so \
  -ex 'target cudacore /tmp/min-fa3-core.zkrh-58.1353963.nvcudmp'
```

### 5.2 故障 kernel 和执行上下文

后续同类复现的 CUDA core 将异常定位到：

```text
mega_ring_flash_attn_bwd_kernel<..., NumDevices=8>
```

故障上下文：

```text
device      = 2
blockIdx.x  = 49
gridDim.x   = 132
blockDim.x  = 384
num_comp_sm = 128
num_comm_sm = 4
```

融合 kernel 按 CTA 编号分工：

```cpp
if (int(blockIdx.x) < params.num_comp_sm) {
    // compute path
}

int const comm_bid = int(blockIdx.x) - params.num_comp_sm;
// communication path
```

因为 `49 < 128`，故障 CTA 明确位于 compute path。两个 Mega backward 入口共用的 compute mainloop 比 communication helper 更值得优先检查。

### 5.3 故障源码和指令

故障对应源码：

```cpp
// include/backward/min_fa3_bwd_mainloop.h:651-658
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
}
```

对应的精确 SASS PC：

```text
kernel + 0x1f8e0:
UTMALDG.4D [UR8], [UR16], desc[UR18]
```

这是最终 dO tile 的第二个 TMA transaction，负责加载 head dimension 的 `D=64..127`。

## 6. 故障 TMA 操作数审计

### 6.1 `UTMALDG.4D` 操作数含义

CUTLASS `SM90_TMA_LOAD_4D` 在该指令上的 operand 对应关系为：

```text
UR8       shared-memory destination
UR16:UR17 tensor-map descriptor pointer
UR9       shared mbarrier address
UR10:UR13 4D tensor coordinates
UR18:UR19 cache hint descriptor
```

从 core 和指令数据流恢复出的故障操作数为：

```text
shared destination = 0x22500
descriptor pointer = 0x00007f1f282a03c0
mbarrier            = 0x30d38
coordinates         = {64, 8128, 22, 0}
```

注意：CUDA-GDB 停住的 lane 已越过故障指令，异常现场直接显示的 `UR8` 可能已经被后续指令覆盖。`0x22500` 是结合 SASS 数据流、pipeline stage 和 shared-storage 基址恢复的故障指令输入，而不是盲目采用停住时的单个寄存器显示值。

### 6.2 对齐检查

逐项检查如下：

```text
shared destination = 0x22500
0x22500 % 128       = 0

descriptor pointer = 0x00007f1f282a03c0
descriptor % 64     = 0

mbarrier            = 0x30d38
0x30d38 % 8         = 0
```

因此：

- shared destination 满足该 TMA transaction 的 128-byte 对齐；
- tensor-map descriptor 满足 CUDA tensor map 的 64-byte 对齐；
- mbarrier 满足 8-byte 对齐。

没有发现通常能够直接导致 `misaligned address` 的 operand 对齐错误。

### 6.3 tensor coordinates 和边界检查

坐标含义为：

```text
D dimension       = 64
packed Q row      = 8128
Q head            = 22
batch             = 0
```

故障 workload 的 packed row shape 为 8192。该 TMA tile 实际访问：

```text
rows = 8128..8191
D    = 64..127
```

两者均位于合法范围内。head 22 也满足 `QH=32`。因此 core 中没有 scheduler 产生越界 row/head/batch 坐标的证据。

## 7. 两条相邻 dO TMA 的对比

### 7.1 故障附近 SASS

故障附近的关键 SASS 为：

```text
/*1f7a0*/ UIADD3 UR8,  UR20, 0x20000, URZ
/*1f800*/ UIADD3 UR20, UR20, 0x22000, URZ
...
/*1f830*/ UMOV UR10, URZ
/*1f840*/ UMOV UR18, 0x0
/*1f850*/ UMOV UR19, 0x14f00000
/*1f870*/ UMOV UR21, 0x40

/*1f880*/ UTMALDG.4D [UR8], [UR16], desc[UR18]
/*1f890*/ UMOV UR23, UR9
/*1f8a0*/ STL [R1], R2
/*1f8b0*/ UMOV UR8, UR20
/*1f8c0*/ UMOV UR10, UR21
/*1f8d0*/ UMOV UR20, 0x10
/*1f8e0*/ UTMALDG.4D [UR8], [UR16], desc[UR18]
```

融合 kernel 的 static shared size 是 `0x500`，动态 shared 从 `0x500` 开始。故障使用 stage 0，所以两条 TMA 的实际 destination 为：

```text
first  = 0x500 + 0x20000 = 0x20500
second = 0x500 + 0x22000 = 0x22500
```

### 7.2 唯一有意义的差异

两条 TMA 使用相同的：

- descriptor pointer；
- mbarrier；
- packed row、head 和 batch coordinates；
- cache hint；
- pipeline stage。

只改变：

```text
first:  destination=0x20500, D=0
second: destination=0x22500, D=64
```

两组 destination 和 coordinate 都满足对齐及范围要求。第一条执行，第二条在相同动态指令位置偶发异常。两条指令之间没有 scheduler、communication helper 或 fence 逻辑，只有 uniform register 更新和普通局部指令。

这个局部事实是当前怀疑编译器 codegen/scoreboard 问题的最强证据之一：如果 descriptor、tensor base 或 pipeline stage 本身错误，两条相邻 transaction 通常应共同受影响；如果 `0x22500` 是固定非法地址，则应稳定失败，而不应偶发失败。

### 7.3 对应 PTX

从 cubin 对应的原始 PTX 可以看到两条合法的 4D TMA：

```ptx
cp.async.bulk.tensor.4d.shared::cluster.global...
    [dst0], [desc, {0, row, head, batch}], [barrier], cache;

cp.async.bulk.tensor.4d.shared::cluster.global...
    [dst1], [desc, {64, row, head, batch}], [barrier], cache;
```

也就是说，高层 C++/CuTe 生成的 PTX operand 结构没有明显错误；异常出现在 `ptxas` 生成的 Hopper `UTMALDG.4D` 机器码执行阶段。

## 8. Shared-memory layout 与 swizzle

`SmemLayoutdO` 很可能使用 CUTLASS `Swizzle<3,4,3>`，其整体对齐辅助值为：

```cpp
alignment_for_swizzle(Swizzle<3,4,3>) == 1024
```

但不能因此得出 `0x22500 % 1024 != 0` 就是非法地址。PTX 对 128-byte swizzle TMA destination 的定义允许 `dstMem` 不落在 1024-byte 边界，硬件以 destination 计算 swizzle base offset：

```text
base offset = (dstMem / 128) % 8
```

所以 `0x22500` 的非零 swizzle base offset 是合法状态。这里真正需要满足的是 transaction/shared destination 的相应对齐，`0x22500 % 128 == 0` 已满足。

此外，`0x22500` 是固定 layout 产生的地址。如果它因为 1024-byte 对齐而硬性非法，同一个 kernel 应在执行这条固定指令时稳定失败，这也与问题的偶发性矛盾。

## 9. Tensor-map descriptor 审计

core 中恢复的 dO descriptor global base 为：

```text
dO global base = 0x00007f1e9c000000
Q  global base = 0x00007f1e94000000
```

Q 和 dO descriptor 除 global base 外逐字节一致，shape、stride、box size 和 swizzle 等字段没有发现异常。dO global base 本身也具有充足对齐。

故障 descriptor pointer 为：

```text
0x00007f1f282a03c0
```

CUDA tensor map 要求 64-byte alignment，而：

```text
0x00007f1f282a03c0 % 64 == 0
```

descriptor 内容和地址都不支持“host 侧构造了未对齐 tensor map”的解释。

另一个调试注意事项是，普通 CUDA-GDB 命令：

```gdb
x/16gx 0x22500
```

会把 `0x22500` 按 host virtual address space 解释，不能用来检查 kernel 的 shared memory 内容，也不能作为 shared memory 已损坏的证据。

## 10. Kernel 参数区检查

故障 specialization 的参数信息为：

```text
sizeof(MegaRingBwdKernelParams<..., 8>) = 0x2e80
alignof                                  = 128
KPARAM offset                            = 0x70
KPARAM size                              = 0x2e80
CBANK size                               = 0x2ef0
```

CUDA core 中保存的参数区覆盖完整 `0x2e80`，不存在“core 只保存了参数前 4 KiB，后部 descriptor 是无效调试数据”的问题。恢复出的 `num_comp_sm`、`num_comm_sm`、descriptor 和 shape 彼此一致。

大 `__grid_constant__` 参数或 descriptor staging 仍可作为低优先级候选，但当前没有参数截断、错位或 descriptor 被破坏的实证。

## 11. Scheduler 与 workspace 审计

提交：

```text
585894c perf(backward): reuse mega-ring workspace and compact auxiliary grids
```

修改了 backward scheduler、workspace 和辅助 grid。scheduler 发布区从单个整数扩展为两个对齐的 `int4`：

```cpp
struct alignas(16) SharedStorage {
    alignas(16) int4 block_head_batch_ticket;
    alignas(16) int4 step_level_valid_reserved;
};
```

scheduler 同步参与线程数为：

```text
load scheduler warp = 32
dQ scheduler warp   = 32
MMA consumers       = 256
total               = 320
NumSchedulerThreads = 320
```

参与数一致，没有发现 named barrier 计数不匹配。

core 中本次发布的 work 坐标是合法的最后一个 tile：

```text
packed row tile starts at 8128
head = 22
batch = 0
```

没有发现 ticket 撕裂、越界 head/batch 或无效 tile。scheduler/workspace race 仍不能仅靠一次 core 完全排除，但当前证据不支持它是首要根因。

## 12. Mega DCP 与 Mega CP 的关系

### 12.1 DCP 引入提交没有修改 backward 源码

提交：

```text
03679d6 add mega kernel for dcp
```

主要新增：

```text
csrc/dcp_mega_min_fa3_varlen_bindings.cu
csrc/dcp_mega_min_fa3_varlen_kernel_pack_*.cu
csrc/dcp_mega_min_fa3_varlen_launch.cu
include/dcp_mega_min_fa3_kernel.h
include/dcp_mega_min_fa3_varlen_launch.h
include/dcp_mega_min_fa3_varlen_params.h
include/dcp_mega_min_fa3_varlen_scheduler.h
```

以及 bindings、Python wrapper 和 `setup.py` source list。该提交没有改动 `include/backward/` 或 `csrc/backward/` 下的文件。

### 12.2 benchmark 不调用 DCP kernel

本次 `benchmark_uniform_backward.sh` 的方法列表不包含 Mega DCP，故障进程内没有 DCP kernel 与 Mega CP backward 并发执行。每次 backward launch 后还有显式 `torch.cuda.synchronize()`，进一步降低了其他异步 kernel 跨迭代污染现场的可能。

不同 workload 又由独立 `torchrun` 启动，因此不能用前三个 case 遗留的 DCP/CP CUDA 状态解释第 4 个 case。

### 12.3 backward cubin 未被链接过程改写

从当前 build object 和故障 `_min_fa3_op.so` 分别提取 backward cubin，SHA-256 完全一致：

```text
c412be26383df35df74dadd3e6d736f65e43abb6afc32546ef5d09b9eb6ba6d0
```

因此可以排除：

- host linker 改写 backward SASS；
- DCP fatbin 聚合过程改写 backward cubin；
- `.o` 和 `.so` 内的 backward constant layout 不一致。

### 12.4 更合理的因果链

当前证据支持的因果链是：

```text
加入 Mega DCP
    -> extension source/fatbin/module 发生变化并触发重新编译
    -> Mega backward 在 CUDA 12.8.61 下生成了存在潜在问题的 Hopper TMA SASS
    -> 特定运行时机下第二条 UTMALDG.4D 偶发 Warp Misaligned Address
```

这里的“相关变化”不等于“DCP kernel 运行时写坏 CP 内存”。后一种解释与 DCP 未执行、故障位于 compute CTA、TMA operand 合法等事实不符。

## 13. Fence 为什么不是本次异常的主要原因

`fence.proxy.async.*` 用于 generic proxy 与 TMA/async proxy 间的数据可见性。缺少 fence 可能产生：

- 读取旧数据；
- 数值错误；
- 动态更新的 descriptor 对 async proxy 不可见；
- 其他正确性问题。

但 fence 不会把已经发射的这条指令的以下 operand 自动变成另一个值：

- descriptor pointer；
- shared destination；
- mbarrier address；
- tensor coordinates。

本次 descriptor 不是在 kernel 运行过程中动态修改，core 中恢复的各 operand 又全部对齐且在界内。因此，缺少 fence 可以作为独立的正确性审计项，但不适合用来解释当前 `Warp Misaligned Address`。

用户关于“这些 fence 主要影响正确性，不应直接导致 misaligned address”的判断与当前现场证据一致。后续不应优先通过无目标地增加 fence 来处理这个错误。

## 14. 编译器问题的外部佐证

NVIDIA CUTLASS issue #1250 报告了相近现象：

```text
https://github.com/NVIDIA/cutlass/issues/1250
```

其共同点包括：

- Hopper；
- fault PC 位于 `UTMALDG`；
- CUDA 报 `misaligned address`；
- 用户侧检查不到实际 alignment 错误。

NVIDIA 在 2025-01-24 回复：

```text
This is a compiler issue. Compiler team has solved it and will include
the solution in its next release version.
```

本项目使用的 `ptxas 12.8.61` 构建于 2025-01-15，比该回复早 9 天，因此很可能不包含这一修复。

这条 issue 不能单独证明本项目遇到的是完全相同的编译器 bug，因为没有 NVIDIA 的内部 bug ID 和修复版本对应关系。但“合法 PTX、合法运行时 operand、Hopper、偶发在 `UTMALDG` 报 misaligned”与该类 compiler issue 高度吻合，可作为当前根因排序的重要旁证。

## 15. 已排除项与剩余不确定性

### 15.1 基本排除

| 假设 | 排除依据 |
| --- | --- |
| dO global pointer 未对齐 | descriptor global base 为 `0x...000000`，对齐充足 |
| tensor-map descriptor pointer 未对齐 | `0x...03c0 % 64 == 0` |
| shared destination 未对齐 | `0x22500 % 128 == 0` |
| mbarrier 未对齐 | `0x30d38 % 8 == 0` |
| dO tile 越界 | row `8128..8191`、D `64..127`、head 22 均合法 |
| communication helper 直接故障 | CTA 49 属于前 128 个 compute CTA |
| DCP kernel 并发污染 | benchmark 未调用 DCP kernel |
| 前面 workload 遗留 CUDA 状态 | 每个 workload 是独立 `torchrun` 进程 |
| linker/fatbin 改写 backward cubin | object 与 `.so` 中 backward cubin hash 一致 |
| swizzle 要求 destination 必须 1024-byte 对齐 | PTX 允许相应 128B swizzle base offset，且固定地址不会解释偶发性 |

### 15.2 尚未完全排除

1. scheduler/workspace 的低概率 race。一次 core 中坐标合法并不能数学上排除其他迭代可能发布坏状态。
2. 大 kernel parameter block 或 descriptor staging 的特殊编译器/ABI 问题。
3. 当前 `ptxas` 生成 SASS 后，是否确实遗漏了两条 TMA 间某种硬件依赖。Hopper SASS control word 并非公开 ISA，不能把 SM80 的第三方控制码解码直接套到 SM90 并当作定论。
4. CUTLASS #1250 的修复究竟首次包含在哪个 CUDA toolkit 版本，当前尚未通过 release note 或编译器版本实验确认。

## 16. 后续建议的判别实验

以下实验按信息增益排序。因为问题偶发，每个实验都应只保留两个 Mega CP backward 方法并进行足够多轮重复，不能只跑一次就下结论。

### 16.1 首选：更新 `ptxas` 对照

用同一源码分别使用：

- 当前 CUDA 12.8.61；
- CUDA 12.9、CUDA 13.x，或已确认包含相关 Hopper TMA 修复的 toolkit；

重新编译，仅测试 `mega_ring_all_cp` 和 `mega_ring_hybrid` 的 backward。保持驱动、GPU、输入、SM 配置和进程模型一致。

如果旧 `ptxas` 能在足够次数内复现，而新 `ptxas` 长时间不复现，这是支持 compiler bug 的最强判别证据。还应对比两版故障区域 SASS，而不只比较运行结果。

### 16.2 当前工具链下增加可靠指令间隔/依赖

在两条 dO TMA 之间加入能够真实改变指令调度或建立依赖的最小实验性改动，检查故障率是否显著变化。该实验用于验证 uniform-register/scoreboard 假设，不应作为未经解释的最终修复。

单纯增加 proxy fence 的信息量较低，因为它不针对当前故障 operand 或相邻 `UTMALDG` 的寄存器依赖。

### 16.3 区分重新编译与 module 聚合

保留完全相同的 backward object，只移除 DCP objects 后重新链接，再与包含 DCP objects 的 `.so` 做多轮对照：

- 如果 backward cubin 完全相同且故障率不变，进一步支持 backward 自身 codegen 问题；
- 如果仅 module 聚合就显著改变故障率，需要继续检查 module load、constant placement 或驱动层行为。

当前已经确认 object 与故障 `.so` 内 backward cubin 相同，所以这项实验优先级低于更新 `ptxas`。

### 16.4 对比 DCP 加入前的真实遗留 cubin

如果能找到加入 DCP 前构建并验证正常的 `_min_fa3_op.so` 或 backward cubin，应直接比较：

- PTX；
- SASS；
- register/uniform-register usage；
- kernel info、constant bank 和 shared-memory metadata；
- 故障附近两条 TMA 的指令序列与 control word。

只比较 Git 源码不足以证明旧、新 cubin 相同，因为重新编译工具链、编译顺序和编译选项都可能改变 codegen。

## 17. 阶段性判断

这是一个真实的设备端 `Warp Misaligned Address`，不是 PyTorch 栈本身的问题；Python 的 `torch.cuda.synchronize()` 只是异步错误的观察点。

然而，“misaligned address”这个异常名称不意味着用户 tensor 或 shared pointer 必然真的未对齐。当前 core 中，故障 `UTMALDG.4D` 的 descriptor、destination、mbarrier 和 tensor coordinates 均合法；紧邻的第一条等价 TMA 又能执行，而第二条只在某些运行中失败。固定 layout 错误、越界坐标、缺 fence 以及 DCP kernel 直接污染都不能很好解释这一组合。

因此，现阶段应把 CUDA 12.8.61 `ptxas` 的 Hopper TMA codegen/uniform-register scoreboard 缺陷作为主因调查。Mega DCP 的加入更可能通过重新编译或 module 变化暴露这个缺陷，而不是通过运行时共享内存路径直接写坏 Mega CP backward。

在更新编译器的 A/B 多轮实验完成前，这仍是证据充分的根因假设，而不是最终定案。

## 附录 A：原始 benchmark 故障日志摘录

以下内容直接摘自：

```text
benchmark_logs/20260813-080737-uniform-backward/benchmark_uniform_backward.log
```

### A.1 故障 case 启动和 workload

```text
================================================================================
[uniform_backward 4/15] context=65536, batch=8, seqlen=8192, hybrid=G1
================================================================================
CUDA_VISIBLE_DEVICES=0\,1\,2\,3\,4\,5\,6\,7 torchrun --standalone --nproc_per_node=8 ring_test/benchmark_topology_backward.py --global-seqlens 8192\,8192\,8192\,8192\,8192\,8192\,8192\,8192 --ring-sizes 1\,1\,1\,1\,1\,1\,1\,1 --ring-starts 0\,1\,2\,3\,4\,5\,6\,7 --qhead 32 --kvhead 8 --headdim 128 --allgather-overlapping-heads-k-stride 4 --mode causal --methods allgather_attention\,llama3_allgather_attention\,fa3_ring\,megatron_hybrid_cp\,magi_attention\,zeppelin\,mega_ring_all_cp\,mega_ring_hybrid --zeppelin-threshold 4096 --megatron-max-seqlen-per-rank 8192 --magi-overlap-degree 2 --sm-configs 128:4\,124:8\,120:12\,116:16 --warmup-iters 10 --num-iters 40 --seed 0 --no-check
Explicit-topology mega-ring causal backward: world_size=8, methods=['allgather_attention', 'llama3_allgather_attention', 'fa3_ring', 'megatron_hybrid_cp', 'magi_attention', 'zeppelin', 'mega_ring_all_cp', 'mega_ring_hybrid'], QH=32, KVH=8, D=128, allgather_overlapping_heads_k_stride=4, sm_configs=128:4,124:8,120:12,116:16, zeppelin_threshold=4096, megatron_max_seqlen_per_rank=8192, magi_overlap_degree=2, warmup=10, iters=40, check=False
Block baseline backend: in-repo min_fa3 fallback
Reusable backward IPC pools: cases=1, all_cp_rank_capacity=8192, all_cp_accum_numel=9437184, hybrid_rank_capacity=8192, hybrid_accum_numel=9437184

Workload: explicit topology, B=8, global_tokens=65536, global_seqlens=[8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192]
Mega-ring all-CP workload: alignment=2048, global_tokens=65536, global_seqlens=[8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192]
Hybrid rings: sizes=[1, 1, 1, 1, 1, 1, 1, 1], starts=[0, 1, 2, 3, 4, 5, 6, 7]
Zeppelin placement: L=4096, final_s0=4096, iterations=1, groups=[1, 1, 1, 1, 1, 1, 1, 1], execution_lengths=[8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192], padding=0, physical_rank_token_loads=[8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192]
Timing excludes forward preparation, opaque-workspace construction, owner-accumulator reset, and the pre-launch distributed barrier; reused step-workspace reset and method-internal phase barriers are included; reported time is the average of the per-iteration max-rank end-to-end backward op times.
```

### A.2 首次错误栈

```text
[rank6]: Traceback (most recent call last):
[rank6]:   File "/home/hychen/min_fa3_demo/ring_test/benchmark_topology_backward.py", line 1805, in main
[rank6]:     case_results = benchmark_topology(
[rank6]:                    ^^^^^^^^^^^^^^^^^^^
[rank6]:   File "/home/hychen/min_fa3_demo/ring_test/benchmark_topology_backward.py", line 1428, in benchmark_topology
[rank6]:     timing = measure_backward_ms(
[rank6]:              ^^^^^^^^^^^^^^^^^^^^
[rank6]:   File "/home/hychen/min_fa3_demo/ring_test/benchmark_topology_backward.py", line 629, in measure_backward_ms
[rank6]:     torch.cuda.synchronize()
[rank6]:   File "/home/hychen/min_fa3_demo/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py", line 1108, in synchronize
[rank6]:     return torch._C._cuda_synchronize()
[rank6]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank6]: torch.AcceleratorError: CUDA error: misaligned address
[rank6]: Search for `cudaErrorMisalignedAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
[rank6]: CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
[rank6]: For debugging consider passing CUDA_LAUNCH_BLOCKING=1
[rank6]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

### A.3 异常清理时再次报错

```text
[rank6]: During handling of the above exception, another exception occurred:

[rank6]: Traceback (most recent call last):
[rank6]:   File "/home/hychen/min_fa3_demo/ring_test/benchmark_topology_backward.py", line 1836, in <module>
[rank6]:     main()
[rank6]:   File "/home/hychen/min_fa3_demo/ring_test/benchmark_topology_backward.py", line 1827, in main
[rank6]:     torch.cuda.synchronize()
[rank6]:   File "/home/hychen/min_fa3_demo/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py", line 1108, in synchronize
[rank6]:     return torch._C._cuda_synchronize()
[rank6]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank6]: torch.AcceleratorError: CUDA error: misaligned address
[rank6]: Search for `cudaErrorMisalignedAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
[rank6]: CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
[rank6]: For debugging consider passing CUDA_LAUNCH_BLOCKING=1
[rank6]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: misaligned address
```

### A.4 torchrun 最终根故障

```text
E0813 08:21:22.980000 1245483 torch/distributed/elastic/multiprocessing/api.py:984] failed (exitcode: -6) local_rank: 6 (pid: 1246013) of binary: /home/hychen/min_fa3_demo/.venv/bin/python3

Root Cause (first observed failure):
[0]:
  time      : 2026-08-13_08:21:19
  host      : zkrh-58
  rank      : 6 (local_rank: 6)
  exitcode  : -6 (pid: 1246013)
  error_file: <N/A>
  traceback : Signal 6 (SIGABRT) received by PID 1246013
```

## 附录 B：CUDA core/SASS 分析关键输出

以下是本文结论直接依赖的 core/SASS 结果汇总。它们不在原始 Python benchmark 日志中，而是使用匹配的 `_min_fa3_op.so` 对后续同类复现的 CUDA core 做离线分析得到。

```text
fault kernel:
  mega_ring_flash_attn_bwd_kernel<..., NumDevices=8>

launch context:
  device      = 2
  blockIdx.x  = 49
  gridDim.x   = 132
  blockDim.x  = 384
  num_comp_sm = 128
  num_comm_sm = 4

fault PC:
  kernel + 0x1f8e0
  UTMALDG.4D [UR8], [UR16], desc[UR18]

source:
  include/backward/min_fa3_bwd_mainloop.h:655

recovered TMA operands:
  shared destination = 0x22500
  descriptor pointer = 0x00007f1f282a03c0
  mbarrier            = 0x30d38
  coordinates         = {64, 8128, 22, 0}

alignment/range checks:
  0x22500 % 128                  = 0
  0x00007f1f282a03c0 % 64       = 0
  0x30d38 % 8                   = 0
  accessed rows                 = 8128..8191 of 8192
  accessed D                    = 64..127 of 128
  accessed Q head               = 22 of 32
```

故障附近 SASS：

```text
/*1f7a0*/ UIADD3 UR8,  UR20, 0x20000, URZ
/*1f800*/ UIADD3 UR20, UR20, 0x22000, URZ
/*1f830*/ UMOV UR10, URZ
/*1f840*/ UMOV UR18, 0x0
/*1f850*/ UMOV UR19, 0x14f00000
/*1f870*/ UMOV UR21, 0x40
/*1f880*/ UTMALDG.4D [UR8], [UR16], desc[UR18]
/*1f890*/ UMOV UR23, UR9
/*1f8a0*/ STL [R1], R2
/*1f8b0*/ UMOV UR8, UR20
/*1f8c0*/ UMOV UR10, UR21
/*1f8d0*/ UMOV UR20, 0x10
/*1f8e0*/ UTMALDG.4D [UR8], [UR16], desc[UR18]
```

descriptor global bases：

```text
dO descriptor global base = 0x00007f1e9c000000
Q  descriptor global base = 0x00007f1e94000000
```

kernel parameter/constant bank：

```text
sizeof(params) = 0x2e80
alignof(params) = 128
KPARAM offset = 0x70
KPARAM size = 0x2e80
CBANK size = 0x2ef0
```

backward cubin identity：

```text
current build object backward cubin SHA-256:
c412be26383df35df74dadd3e6d736f65e43abb6afc32546ef5d09b9eb6ba6d0

faulting _min_fa3_op.so backward cubin SHA-256:
c412be26383df35df74dadd3e6d736f65e43abb6afc32546ef5d09b9eb6ba6d0
```
