# Hierarchical Hybrid Mega-Ring Forward Design

## Status

本文记录 `world_size=8` 下层级 hybrid mega-ring forward 的实现方案。
本文档本身不修改 CUDA、Python、构建或 Slurm 行为。

目标是在现有一次 fused mega-ring attention/communication launch 中同时支持：

- size 1：单 GPU 完整序列
- size 2：`0-1`、`2-3`、`4-5`、`6-7`
- size 4：`0-1-2-3`、`4-5-6-7`
- size 8：`0-1-2-3-4-5-6-7`

本次只覆盖 forward，保持当前 BF16、head dim 128、SM90、GQA/MQA 等约束。
Backward 不在本方案范围内。

## Selected Execution Model

每条全局序列显式指定：

```text
global length
ring size
ring start rank
```

所有 ring size 仍由同一个 persistent kernel 处理，不为 size 8、4、2、1
分别发起 attention kernel。

kernel 内部的逻辑 work stream 为：

```text
step 0: 所有本 rank 非空序列，包括 size 8/4/2/1
G=8:    replay ring step 1..7
G=4:    replay ring step 1..3
G=2:    replay ring step 1
G=1:    无 replay
```

这是一种“按 exact ring size 分组，组内按 ring step replay”的 fused 调度。
优先处理较大的 ring，使长 CP 序列尽早与通信 CTA 重叠。

## Public CUDA/Python Entry

计划将 Python API 固定为：

```python
forward_varlen_mega_ring(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    is_causal,
    *,
    cu_seqlens_q_host,
    cu_seqlens_k_host,
    remote_k,
    remote_v,
    remote_barrier,
    num_comp_sm,
    num_comm_sm,
    global_seqlens_host,
    ring_sizes_host,
    ring_starts_host,
    out=None,
    lse=None,
    return_lse=False,
)
```

删除现有 threshold hybrid 参数：

```text
cp_threshold
half_cu_seqlens
half_cu_seqlens_host
```

不保留 `global_seqlens_host + cp_threshold` 的 N/1 兼容分支。
全 size 8 或全 size 1 也通过显式 ring metadata 表达。

### Public tensor semantics

`q` 是当前 rank 的紧凑 Q：

```text
[local_total_q, QH, 128]
```

`k` 和 `v` 是 `TKParallelTensor` 对应的完整 rank-major arena：

```text
[world_size * rank_kv_capacity, KVH, 128]
```

当前 rank 的 owner-local K/V 放在：

```text
owner_base = global_rank * rank_kv_capacity

k[owner_base : owner_base + local_total_k]
v[owner_base : owner_base + local_total_k]
```

`q` 和输出 `o` 不需要按 capacity padding。只有 K/V arena 使用统一的
rank stride，以满足 `TKParallelTensor` 在所有进程上具有相同 shape 的要求。

`global_seqlens_host`、`ring_sizes_host` 和 `ring_starts_host` 均为：

```text
CPU, int32, contiguous, shape [B]
```

所有 rank 必须传入相同的三组全局 metadata。

## Input Layout Contract

### Ring metadata

每个 batch `b` 必须满足：

```text
ring_sizes_host[b] in {1, 2, 4, 8}
ring_starts_host[b] % ring_sizes_host[b] == 0
0 <= ring_starts_host[b]
ring_starts_host[b] + ring_sizes_host[b] <= 8
global_seqlens_host[b] % ring_sizes_host[b] == 0
```

本 rank 是否属于该序列的 ring：

```text
is_member = ring_start <= global_rank < ring_start + ring_size
```

期望的本地长度为：

```text
expected_local_len = global_len / ring_size  if is_member
                     0                       otherwise
```

本方案要求 self-attention，因此每个 batch 的本地 Q/K 长度相同。

### Batch ordering

全局 batch 必须按 ring size 非递增排列：

```text
8 -> 4 -> 2 -> 1
```

同一 ring size 内可以按 `ring_start` 和长度任意排列。
不属于本 rank 的序列在本地 `cu_seqlens` 中表现为零长度 batch。

这一顺序保证同一个 size=G 子环内的 rank 对该 G bucket 使用相同的物理
row offset。原因是同组 rank 对所有 size 大于或等于 G 的序列具有相同的
成员关系，而所有 size 小于 G 的序列都排列在该 bucket 之后。

因此通信可以继续使用相同的 source/destination `row_idx`，不需要增加
`[source_rank, batch]` 远端 offset 查找表。

### Capacity

各 rank 的有效 token 数可以不同。上层在创建 `TKParallelTensor` 前计算：

```text
rank_kv_capacity = max(local_total_k across all 8 ranks)
```

该 max 可以在输入准备阶段通过 `all_gather` 或 `all_reduce(MAX)` 得到。
它不是按 ring size 发起的 attention launch。

CUDA binding 从 arena shape 推导：

```text
rank_kv_capacity = k.size(0) / world_size
```

并验证：

```text
k.size(0) == v.size(0)
k.size(0) % world_size == 0
cu_seqlens_k_host[-1] <= rank_kv_capacity
```

## Host-Side Derived Metadata

CUDA binding 在 launch 前从 public 参数构造以下数据。

### Device ring sizes

`ring_sizes_host` 被复制为：

```text
ring_sizes: CUDA int32 [B]
```

mainloop、scheduler 和 epilogue 都通过 actual batch index 读取它。

`global_seqlens_host` 和 `ring_starts_host` 只用于 host validation。
对一个本 rank 非空的 batch，kernel 可以从固定对齐拓扑直接推导
`ring_base`，不需要 device `ring_starts`：

```text
G = ring_sizes[actual_batch]
ring_base = floor(global_rank / G) * G
ring_local_rank = global_rank - ring_base
```

### Internal half cu_seqlens

causal 模式内部生成 `half_cu_seqlens_host[B+1]`：

```text
half_len[b] = local_len[b] / 2  if ring_size[b] > 1 and this rank is a member
              0                 otherwise
```

然后复制成 CUDA int32 tensor。

size 2/4/8 的有效 batch 要求：

```text
local_len % 2 == 0
(local_len / 2) % 128 == 0
```

size 1 继续走普通 local causal attention，不增加 half-length 对齐约束。

noncausal 模式不读取 `half_cu_seqlens`。

### Ring-level descriptors

由于只支持 8、4、2、1 四个 level，使用固定数量 descriptor，不引入
通用动态 group abstraction。

建议的内部结构为：

```cpp
struct MegaRingLevelDesc {
    int ring_size;
    int batch_begin;
    int batch_end;

    int row_begin;
    int full_rows;
    int half_row_begin;
    int half_rows;

    int full_tiles;
    int half_tiles;

    int reduction_base;
    int kv_ready_base;
};
```

字段含义：

- `batch_begin/end`：该 exact ring size 在全局 batch 中的范围。
- `row_begin/full_rows`：该 level 在本 rank 紧凑 K/V 中的连续 row 范围。
- `half_row_begin/half_rows`：该 level 在内部 half stream 中的范围。
- `full_tiles/half_tiles`：该 level 的 full-Q/half-Q scheduler tile 数。
- `reduction_base`：该 level 在 `step_ready` 中的 tile 前缀。
- `kv_ready_base`：该 level 在通信 section counter 中的前缀。

tile 数使用当前 varlen kernel 的 `BlockM=128`：

```text
tiles_for_batch = ceil_div(local_q_len, 128) * QH
```

half tile 使用 `local_q_len / 2`。

### Scheduler metadata

现有 `prepare_varlen_num_blocks` 和 varlen batch sort 保留。

所有 ring metadata 查找必须使用 scheduler 的 actual batch index：

```text
actual_batch = varlen_batch_idx_ptr[virtual_batch]
G = ring_sizes[actual_batch]
```

不得用排序后的 virtual batch index 直接索引 `ring_sizes`。

## Scheduler Work Stream

### Noncausal

令：

```text
T_all = 本 rank 所有非空 batch 的 full-Q tile 数
T8    = 本 rank exact size 8 batch 的 full-Q tile 数
T4    = 本 rank exact size 4 batch 的 full-Q tile 数
T2    = 本 rank exact size 2 batch 的 full-Q tile 数
```

总 work tile 数为：

```text
T_all + 7*T8 + 3*T4 + T2
```

### Causal

对 exact size G，定义：

```text
rG = global_rank - floor(global_rank / G) * G
```

该 level 的 replay tile 数为：

```text
rG * Tfull[G] + (G - 1 - rG) * Thalf[G]
```

总 work tile 数为：

```text
T_all + sum over active G in {8,4,2}(
    rG * Tfull[G] + (G - 1 - rG) * Thalf[G]
)
```

### Tile decode

step 0 沿用基础 varlen tile decoder，覆盖所有非空 batch。

replay section 使用 exact-size predicate：

```text
ring_sizes[actual_batch] == target_G
```

不能只判断 `ring_size > 1`，否则不同 level 的 step 数和 source rank 会混淆。

`WorkTileInfo` 继续携带：

```text
batch/head/m-block
ring_step
global work tile id
step-local tile id
reduction tile id
```

不需要在 work tile 中额外保存 ring size；mainloop 可以通过 actual batch
读取 `ring_sizes`。

## Source Rank and K/V Addressing

对 batch `b` 的 ring step `s`：

```text
G = ring_sizes[b]
ring_base = floor(global_rank / G) * G
ring_local_rank = global_rank - ring_base
src_rank = ring_base + (ring_local_rank - s + G) % G
```

K/V arena 地址为：

```text
kv_row = src_rank * rank_kv_capacity + local_batch_row
```

通信 CTA 从 `remote_k[src_rank]` / `remote_v[src_rank]` 的 `kv_row`
读取，并写入当前 GPU 本地 arena 的相同 `kv_row`。

同组 source 和 destination 的 batch row offset 因 8->4->2->1 排列而一致。

## Causal Zigzag

causal 模式沿用现有 front/back zigzag，只将全局 ring rank 改为子环内 rank。

对 size=G、subgroup-local rank `r`：

```text
step 0:
    full Q
    local full KV
    local diagonal causal mask

step 1..r:
    full Q
    remote front-half KV

step r+1..G-1:
    back-half Q
    remote full KV
```

对应 mainloop 条件：

```text
q_use_half  = ring_step > ring_local_rank
kv_use_half = ring_step >= 1 && ring_step <= ring_local_rank
```

causal half-Q tile 必须映射回相同 full-Q tile 的后半部分 reduction index，
以便 O/LSE 按 ring step 原地归并。

## Communication Sections

communication CTA 使用和 compute scheduler 相同的 level 顺序：

```text
G8 step 1..7
G4 step 1..3
G2 step 1
```

noncausal 每个 section 复制该 level 的完整连续 row range。

causal：

- step 1..ring_local_rank 复制每条序列的 front half。
- 后续 step 复制完整 level row range。
- half-row 到物理 row 的映射复用内部 `half_cu_seqlens`。

首版保持当前 row-granular remote load，不在本任务中同时实施
`REMOTE_LOAD_OPTIMIZATION_PLAN.md` 的 chunk 化优化。

## Synchronization Metadata

### K/V readiness

当前按 source rank 计数的 `kv_ready_counts[world_size]` 不能直接复用。
同一个 source rank 可能同时是不同 ring size/step 的来源，聚合计数会造成
某个 level 被另一个 level 的完成量提前满足。

改为固定 11 个 section counter：

```text
index 0..6: G8 step 1..7
index 7..9: G4 step 1..3
index 10:   G2 step 1
```

首版 row-granular 通信下：

```text
full section ready target = 2 * full_rows[G]
half section ready target = 2 * half_rows[G]
```

乘 2 是因为 K 和 V 每完成一行分别贡献一次 release increment。

通信 CTA 必须在本地 global store 完成后才发布 ready，不能只等待 shared
memory read 完成。

step 0 使用 owner-local K/V，不等待 remote counter。

### O/LSE reduction ordering

`step_ready` 为每个 size 8/4/2 full-Q tile 分配一个 counter：

```text
step_ready_size = T8 + T4 + T2
```

size 1 不需要 reduction counter。

状态转换：

```text
初始值: 0
step 0 完成并写入 O/LSE: signal +1
step s > 0: wait counter >= s
完成该 step 的 O/LSE merge: signal +1
```

不同 Q tile 独立归并，不建立全局 step barrier。

## Fully Worked Example

给定：

```python
global_seqlens_host = tensor([8192, 4096, 2048, 1024], int32, cpu)
ring_sizes_host      = tensor([   8,    4,    2,    1], int32, cpu)
ring_starts_host     = tensor([   0,    4,    2,    7], int32, cpu)
```

### Global meaning

| Batch | Global length | Ring | Member ranks | Local length on member |
|---:|---:|---|---|---:|
| 0 | 8192 | size 8, start 0 | 0-7 | 1024 |
| 1 | 4096 | size 4, start 4 | 4-7 | 1024 |
| 2 | 2048 | size 2, start 2 | 2-3 | 1024 |
| 3 | 1024 | size 1, start 7 | 7 | 1024 |

### Per-rank local lengths and cu_seqlens

| Rank | Local lengths `[b0,b1,b2,b3]` | `cu_seqlens` | Local total |
|---:|---|---|---:|
| 0-1 | `[1024,0,0,0]` | `[0,1024,1024,1024,1024]` | 1024 |
| 2-3 | `[1024,0,1024,0]` | `[0,1024,1024,2048,2048]` | 2048 |
| 4-6 | `[1024,1024,0,0]` | `[0,1024,2048,2048,2048]` | 2048 |
| 7 | `[1024,1024,0,1024]` | `[0,1024,2048,2048,3072]` | 3072 |

因此：

```text
rank_kv_capacity = 3072
arena rows = 8 * 3072 = 24576
```

每个进程创建：

```text
remote_k.data_: [24576, KVH, 128]
remote_v.data_: [24576, KVH, 128]
```

### Owner-local arena placement

rank 7 的 owner block 起始位置：

```text
owner_base = 7 * 3072 = 21504
```

其本地数据布局为：

```text
batch 0, G8: rows [21504, 22528)
batch 1, G4: rows [22528, 23552)
batch 3, G1: rows [23552, 24576)
```

rank 2 的 owner block 起始位置：

```text
owner_base = 2 * 3072 = 6144
```

其本地数据布局为：

```text
batch 0, G8: rows [6144, 7168)
batch 2, G2: rows [7168, 8192)
padding:      rows [8192, 9216)
```

### Causal half metadata

| Rank | `half_cu_seqlens_host` |
|---:|---|
| 0-1 | `[0,512,512,512,512]` |
| 2-3 | `[0,512,512,1024,1024]` |
| 4-7 | `[0,512,1024,1024,1024]` |

batch 3 是 size 1，因此即使 rank 7 拥有它，也不进入 ring half stream。

### Level descriptors for rank 7

假设 `QH=16`：

```text
full tiles for 1024 rows = ceil_div(1024, 128) * 16 = 128
half tiles for 512 rows = ceil_div(512, 128) * 16 = 64
```

| G | Batch range | Local row range | Full/half rows | Full/half tiles | Reduction base | KV counter base |
|---:|---|---|---|---|---:|---:|
| 8 | `[0,1)` | `[0,1024)` | `1024/512` | `128/64` | 0 | 0 |
| 4 | `[1,2)` | `[1024,2048)` | `1024/512` | `128/64` | 128 | 7 |
| 2 | `[2,3)` | empty | `0/0` | `0/0` | 256 | 10 |
| 1 | `[3,4)` | `[2048,3072)` | `1024/0` | `128/0` | unused | unused |

### Level descriptors for rank 2

| G | Batch range | Local row range | Full/half rows | Full/half tiles | Reduction base | KV counter base |
|---:|---|---|---|---|---:|---:|
| 8 | `[0,1)` | `[0,1024)` | `1024/512` | `128/64` | 0 | 0 |
| 4 | `[1,2)` | empty | `0/0` | `0/0` | 128 | 7 |
| 2 | `[2,3)` | `[1024,2048)` | `1024/512` | `128/64` | 128 | 10 |
| 1 | `[3,4)` | empty | `0/0` | `0/0` | unused | unused |

### Scheduler tile counts with QH=16

step 0 tile 数：

| Rank | Active levels | `T_all` |
|---:|---|---:|
| 0-1 | G8 | 128 |
| 2-3 | G8, G2 | 256 |
| 4-6 | G8, G4 | 256 |
| 7 | G8, G4, G1 | 384 |

noncausal 总 work tile 数：

| Rank | Formula | Total |
|---:|---|---:|
| 0-1 | `128 + 7*128` | 1024 |
| 2-3 | `256 + 7*128 + 1*128` | 1280 |
| 4-6 | `256 + 7*128 + 3*128` | 1536 |
| 7 | `384 + 7*128 + 3*128` | 1664 |

causal 总 work tile 数：

| Rank | Total |
|---:|---:|
| 0 | 576 |
| 1 | 640 |
| 2 | 896 |
| 3 | 1024 |
| 4 | 1152 |
| 5 | 1280 |
| 6 | 1408 |
| 7 | 1664 |

例如 rank 2：

```text
step 0: 256

G8 local rank 2:
    2 * 128 full-Q tiles
    5 * 64 half-Q tiles
    subtotal = 576

G2 local rank 0:
    0 * 128 full-Q tiles
    1 * 64 half-Q tiles
    subtotal = 64

total = 256 + 576 + 64 = 896
```

### Reduction counters

`step_ready` 长度：

| Rank | Active CP levels | Counter count |
|---:|---|---:|
| 0-1 | G8 | 128 |
| 2-3 | G8, G2 | 256 |
| 4-7 | G8, G4 | 256 |

size 1 batch 不分配 counter。

### Source rank examples

rank 7：

```text
G8 step 1..7 source: 6,5,4,3,2,1,0
G4 step 1..3 source: 6,5,4
G1: no remote step
```

rank 2：

```text
G8 step 1..7 source: 1,0,7,6,5,4,3
G2 step 1 source:    3
```

rank 2 拉取 batch 2 时，source rank 3 上的 owner row 为：

```text
src owner base = 3 * 3072
batch 2 offset = 1024
remote row range = [3*3072 + 1024, 3*3072 + 2048)
                 = [10240, 11264)
```

数据被写入 rank 2 本地 arena 的相同 rank-major row range，之后 compute CTA
使用本地 HBM TMA 读取。

### Ready targets

本例每个有效 full level 有 1024 rows，half level 有 512 rows。

```text
full section target = 2 * 1024 = 2048
half section target = 2 * 512  = 1024
```

例如 causal rank 7：

```text
G8 local rank 7:
    step 1..7 use half KV, target 1024 each

G4 local rank 3:
    step 1..3 use half KV, target 1024 each
```

例如 causal rank 2：

```text
G8 local rank 2:
    step 1..2 use half KV, target 1024
    step 3..7 use full KV, target 2048

G2 local rank 0:
    step 1 uses full KV, target 2048
```

## Kernel Parameter Summary

最终 fused kernel 需要的数据分为四类。

原始 attention 数据：

```text
q
rank-major K/V arena
out / lse
cu_seqlens_q / cu_seqlens_k
prepared varlen scheduler metadata
```

拓扑与分组数据：

```text
global_rank
world_size = 8
ring_sizes[B]
rank_kv_capacity
MegaRingLevelDesc levels[4]
internal half_cu_seqlens, causal only
```

scheduler 数据：

```text
total_work_tiles
per-level full/half tile counts
per-level reduction prefixes
tile_count_semaphore
```

同步数据：

```text
kv_ready_counts[11]
step_ready[T8 + T4 + T2]
```

`global_seqlens_host` 和 `ring_starts_host` 不直接传给 device mainloop；它们
在 binding 中完成合法性和本 rank membership 验证。

## Validation Requirements

CUDA binding 必须拒绝：

- `world_size != 8`
- ring size 不属于 1/2/4/8
- ring start 未按 ring size 对齐
- ring 越过 rank 7
- global length 不能整除 ring size
- batch 未按 ring size 非递增排列
- member rank 的 local length 不等于 `global_len/ring_size`
- non-member rank 的 local length 非零
- Q/K 本地长度不同
- K/V arena 不能被 world size 整除
- local total K 超过 rank capacity
- causal size 2/4/8 的 half length 不是 128 对齐
- metadata dtype、device、shape 或 contiguous 属性不正确

跨 rank metadata 必须相同是调用方契约。correctness test 使用 distributed
collective 验证测试输入，但 CUDA op 不在热路径中增加 metadata collective。

## Implementation Sequence

1. 修改 Python/PyBind API，使用显式 global length、ring size、ring start。
2. 增加 host validation，并允许 K/V arena capacity 大于本 rank 有效 token 数。
3. 内部生成 device ring sizes、causal half cu_seqlens 和四个 level descriptor。
4. 在现有 `Ring_fwd_params` 中追加层级 metadata，保持 params provenance。
5. 将 mega-ring scheduler 扩展为 step 0 加 exact-size replay sections。
6. 用 subgroup-local rank 替换 mainloop 中的全局 ring-rank causal 判断。
7. 将通信 task decoder 改为按 `(G, step)` section 解码。
8. 将 K/V ready counter 改为 11 个 section counter。
9. 将 O/LSE reduction index 改为 level-prefix 加 level-local tile index。
10. 更新 8-rank correctness、benchmark、README 和 Slurm 参数。

不得为每个 ring size 增加独立 attention kernel launch，也不得重写现有
FA3 mainloop、epilogue 或 scheduler 基础结构。

## Verification Plan

构建：

- 重新编译所有 forward、ring 和 mega-ring 实例。
- 检查 causal/noncausal 模板实例的 kernel 参数布局和 dynamic shared memory。
- 确认普通 varlen forward 和单步 ring 路径仍可构建。

8-rank Hopper correctness：

- 本文完整示例，causal 和 noncausal。
- 所有合法 size 2 ring start：0、2、4、6。
- 所有合法 size 4 ring start：0、4。
- size 8 only，与原 all-CP 结果一致。
- size 1 only，与本地 varlen attention 一致。
- GQA：QH=16、KVH=8、D=128。
- 每个 rank 有不同 local total、共享 padded capacity。
- 检查已拉取 K/V row 与 distributed reference 完全一致。
- arena padding 填入 sentinel，确认通信不越界且不修改 padding。
- 连续多轮运行，检查无 deadlock 和 counter 串扰。

reference 计算：

- noncausal 只拼接当前 batch 所属 subgroup 的 K/V shard。
- causal 使用 subgroup-local rank 生成 front/back 全局位置。
- size 1 使用普通 local causal/noncausal reference。

错误输入测试覆盖 Validation Requirements 中的每一项。

性能与 launch 验证：

- profiler 中每次 op 只有一个 `mega_ring_flash_attn_varlen_kernel`。
- 出现更多 ring size 时不能增加 attention kernel launch 数量。
- 现有 varlen metadata preparation 只执行一次，不按 level 重复。
- 分别记录 compute、communication 和总 op 时间，避免只比较 kernel body。

