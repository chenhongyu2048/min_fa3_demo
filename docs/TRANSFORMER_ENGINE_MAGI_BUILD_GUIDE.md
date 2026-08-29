# Transformer Engine 与 MagiAttention 编译安装记录

本文记录 2026-08-29 为 Mega CP 单层 Transformer forward + backward benchmark 准备 Transformer Engine（TE）和 MagiAttention 的完整过程。内容包括环境约束、固定版本、实际执行命令、讨论中形成的技术决策、遇到的问题、修复方式、最终产物和后续复现建议。

这不是通用安装教程。文中的路径、版本和编译开关针对本仓库以及当前 H20 节点环境，目标是尽量冻结依赖并减少不必要的运行测试。

## 当前 canonical 布局（环境重构后）

本文后续仍保留 2026-08-29 首次安装过程、遇到的问题和判断依据，但当前环境已经从“仓库外独立 checkout + 根目录脚本”重构为父仓库固定的 Git 子模块：

```text
third_party/Megatron-LM       5eb0744c700e3791f3992fdce08cc41d5a469326
third_party/TransformerEngine 4329ff84bfbdaa778a33cba02a15fb0807c64689
third_party/MagiAttention      872717e1f88fa6938593e452a28a41597c849a00
```

父仓库 gitlink 是唯一版本真源。安装脚本不再维护另一套 commit 常量，也不再 clone 到仓库外绝对路径。所有源码应在联网节点一次递归准备：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
git submodule update --init --checkout --recursive
```

完整 fresh-environment 公共入口已经集中到根目录的
[`../setup_fresh_environment.sh`](../setup_fresh_environment.sh)；它会在刚完成
`git clone`、尚无 `.venv` 的环境中先准备仓库本地 `uv==0.12.4`，再调用
`third_party/setup_fresh_environment.sh` 完成实际环境编排。
共享文件系统的两节点环境只需分别执行：

```bash
# 联网节点：递归准备所有子模块，并精确重建 lock 对应的基础环境。
./setup_fresh_environment.sh prepare

# H20 CUDA 节点：构建 min-FA3、TE、Magi 和三个目标 FFA AOT kernel，随后验证。
CUDA_VISIBLE_DEVICES=0 ./setup_fresh_environment.sh install
```

`verify` 仅验证源码、依赖版本、原生扩展和 AOT 产物，不重新编译；同时具备联网和
H20 CUDA 环境的单机可以使用 `all`。下文的分步命令作为问题分析和人工排障参考，
脚本是后续 fresh 环境复现的 canonical 入口。

根 `uv.lock` 已对齐到 Linux x86_64、Python 3.12 和官方 PyTorch cu128 索引，canonical 环境为 `torch==2.11.0+cu128`。先精确同步基础环境，再在 CUDA 节点安装原生扩展：

```bash
UV_CACHE_DIR=.cache/uv \
uv sync --frozen --no-install-project \
  --group build --group transformer-layer

make PYTHON=.venv/bin/python
third_party/install_transformer_engine.sh install
third_party/install_magi_attention.sh install
third_party/precompile_magi_ffa_training.sh
```

TE/Magi 是脚本安装的非 editable 包，不作为 uv path dependency。完成原生安装后若再次同步基础依赖，使用 `uv sync --inexact`，否则精确同步会移除这些未声明包。

官方 FlashAttention 本轮没有新增为顶层子模块。已完成的单层 smoke 显示 `FA3 block backend=min_fa3`，八种方法可以共同使用仓库内 `min_fa3_op` fallback；Magi 自身记录的嵌套 FlashAttention 只随其递归子模块初始化，不安装为顶层 `flash-attn`。

PyTorch 2.11 cu128 wheel 将 CUDA runtime 放在 `nvidia/cuda_runtime`，而 TE 2.17.1 的 loader 仍严格探测 `nvidia/cuda_cudart`。TE 安装脚本会在本仓库 `.venv` 内创建 `cuda_cudart -> cuda_runtime` 兼容链接，不修改系统 CUDA、全局 Python 或 NVIDIA wheel 的库文件。

最短、持续维护的入口请优先参考 [`../third_party/README.md`](../third_party/README.md)。

## 1. 背景与目标

后续 benchmark 的目标配置为：

- 单节点 8 张 H20，Context Parallel（CP）大小为 8；
- 只测试单层 Transformer 的 forward + backward，不执行参数更新；
- QKV 投影、MLP 等非 CP 部分使用 Tensor Parallel（TP）大小 1；
- 上下文长度为 128K；
- attention 配置为 BF16、`qhead=32`、`kvhead=8`、`head_dim=128`，即普通 GQA 4:1；
- TE 主要为 Megatron Transformer layer 的非 CP 模块提供已有实现；
- MagiAttention 的 distributed Flexible FlashAttention（FFA）作为待测试的 CP attention 路径之一；
- 当前阶段关注端到端性能和逻辑连通性，不进行严格数值正确性验证。

这次环境准备的交付物分为三部分：

1. 固定提交的 Transformer Engine Python 包和 PyTorch 原生扩展；
2. 固定提交的 MagiAttention Python 包、通信扩展和主原生扩展；
3. 针对 SM90、BF16、head dimension 128、训练 forward + backward 的三个 Magi FFA AOT 内核。

## 2. 两节点约束与总体方案

实际环境中有两个角色不同的节点：

- 联网节点 `lab`：可以访问 GitHub，但没有可用 CUDA 编译环境；
- CUDA 节点 `H20-2`（可通过 `ssh h20` 进入）：有 CUDA 12.8 和 H20，但不依赖其联网能力。

两个节点可以访问相同的 `/home/LOCAL/shixuan/hongyu` 文件系统，因此最终采用“源码准备与 CUDA 编译分离”的方式：

1. 在联网节点 clone 固定提交及其递归 submodule；
2. 在 H20 节点从共享目录离线编译；
3. 安装时使用 `--offline --no-build-isolation --no-deps`，避免构建过程临时联网或改变现有 PyTorch/CUDA 依赖栈；
4. `fetch` 阶段不检查 `nvcc`，只有 `install`/FFA 预编译阶段需要 CUDA。

曾讨论过让自动化直接通过 `ssh h20` 完成安装，但这会把登录、环境激活、共享路径和命令转义耦合到一起。最终选择将流程封装为短脚本，由用户在对应节点手动执行并保留日志。这也更适合后续复现和审计。

### 2.1 讨论与操作时间线

本次讨论形成的关键决策按先后顺序如下：

1. 先确认现有 `.venv` 为 PyTorch 2.11.0+cu128、CUDA 12.8，再决定从源码安装与当前 Megatron commit 匹配的 TE 和 Magi；
2. 因联网能力与 CUDA 环境位于不同节点，将 `fetch` 和 `install` 拆开，并移除 `fetch` 阶段所有 `nvcc`/GPU 检查；
3. 曾考虑直接 `ssh h20` 自动安装，随后因为命令过长、环境切换复杂而放弃，改为三个可反复参考的短脚本；
4. TE 普通 HTTPS clone 遇到 `Empty reply from server`，使用 Git HTTP/1.1 和 partial clone 成功；
5. Magi 作为当前仓库 submodule，通过命令级 HTTPS URL override clone 成功，避免依赖 GitHub SSH identity；
6. TE 第一次安装实际已构建并安装 extension，但安装后 probe 因缺少 `pydantic` 失败；补依赖后单独 verify 成功，没有重复编译 TE；
7. TE verify 报告 flash-attn 2.8.4 超过声明上限 2.8.3，经判断只是当前路径的非阻塞兼容性 warning，没有为消除 warning 改动现有 FA3 环境；
8. 确认 TE 编译的 kernel 不能直接替代 Magi FFA，但两套 extension 和缓存目录互相独立，不会因为同时安装而自然发生覆盖冲突；
9. Magi 安装时发现 NVIDIA wheel 只提供 `libnvshmem_host.so.3`，因此不设置 `NVSHMEM_DIR`，改由 library search path 使用版本化动态库；
10. 根据 benchmark 的固定形状，决定只预编译 BF16、SM90、head dimension 128、训练 forward + backward 所需的三个 FFA spec；普通 GQA 4:1 不开启 PackGQA/CatGQA；
11. 第一版预编译入口错误地经过 Magi testing package，引入缺失的 `expecttest`；改用生产 JIT spec 接口后，三个 FFA 内核全部编译和 AOT 复制成功；
12. 最终只做了安装/import/AOT 产物层面的静态或轻量验证，把真正的 8 卡 CP=8、128K 运行验证留给后续单层 benchmark。

## 3. 首次安装环境快照（历史）

仓库及主要路径：

```text
工作目录:
  /home/LOCAL/shixuan/hongyu/min_fa3_demo

Python:
  /home/LOCAL/shixuan/hongyu/min_fa3_demo/.venv/bin/python

Megatron-LM:
  /home/LOCAL/shixuan/hongyu/min_fa3_demo/third_party/Megatron-LM
  commit 5eb0744c700e3791f3992fdce08cc41d5a469326

Transformer Engine source:
  /home/LOCAL/shixuan/hongyu/min_fa3_demo/third_party/TransformerEngine
  commit 4329ff84bfbdaa778a33cba02a15fb0807c64689

MagiAttention source:
  /home/LOCAL/shixuan/hongyu/min_fa3_demo/third_party/MagiAttention
  commit 872717e1f88fa6938593e452a28a41597c849a00

Magi FFA workspace:
  /home/LOCAL/shixuan/hongyu/min_fa3_demo/.cache/mega_cp/magi_ffa_sm90_bf16_hd128
```

最终安装版本：

```text
Python                         3.12
PyTorch                        2.11.0+cu128
PyTorch CUDA                   12.8
transformer-engine             2.17.1+4329ff84
magi-attention                 1.1.1.post16+g872717e1
flash-attn                     2.8.4（首次安装时存在；当前 canonical 环境不要求）
pydantic                       2.13.5
nvidia-cudnn-cu12              9.19.0.56
nvidia-nvshmem-cu12            3.4.5
```

CUDA toolkit 使用：

```text
/usr/local/cuda-12.8
```

cuDNN 和 NVSHMEM 来自当前虚拟环境中的 NVIDIA wheel：

```text
.venv/lib/python3.12/site-packages/nvidia/cudnn
.venv/lib/python3.12/site-packages/nvidia/nvshmem
```

## 4. 本次新增的辅助脚本和日志

脚本：

```text
third_party/setup_fresh_environment.sh
third_party/install_transformer_engine.sh
third_party/install_magi_attention.sh
third_party/precompile_magi_ffa_training.sh
```

日志：

```text
install_transformer_engine.log
verify_transformer_engine.log
install_magi_attention.log
precompile_magi_ffa_training.log
```

四个脚本均位于仓库的 `third_party/` 目录：

```text
/home/LOCAL/shixuan/hongyu/min_fa3_demo/third_party
```

`third_party/install_transformer_engine.sh` 和 `third_party/install_magi_attention.sh` 都提供三个动作：

```text
fetch    在联网节点准备固定提交及递归 submodule
install  在 CUDA 节点离线构建并安装
verify   验证已安装的 Python API 和原生扩展
```

## 5. Transformer Engine：源码准备

### 5.1 固定版本

固定 TE 提交为：

```text
4329ff84bfbdaa778a33cba02a15fb0807c64689
```

对应源码目录：

```text
/home/LOCAL/shixuan/hongyu/min_fa3_demo/third_party/TransformerEngine
```

固定提交而不是跟随默认分支，目的是使后续 Mega CP benchmark 可以复现同一套 Megatron/TE 接口和编译产物。

### 5.2 首选 fetch 命令

在联网节点执行：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
./third_party/install_transformer_engine.sh fetch 2>&1 | tee fetch_transformer_engine.log
```

脚本会让父仓库按已记录的 gitlink 初始化 TE/Megatron 及其递归
submodule，然后检查实际 HEAD 与父 gitlink 完全一致。脚本不会维护或
checkout 第二套硬编码 commit。

最终核对到的 TE submodule 包括：

```text
3rdparty/cudnn-frontend  e46d7082450264ce05cf898f8740011c4896f817
3rdparty/cutlass         57e3cfb47a2d9e0d46eb6335c3dc411498efa198
3rdparty/googletest      f8d7d77c06936315286eb55f8de22cd23c188571
3rdparty/nccl            a6b5de08b6af4f938cef541ae6e4d405632f89a4
```

### 5.3 遇到的问题：GitHub clone 返回 Empty reply

第一次使用脚本 clone 时出现：

```text
fatal: unable to access 'https://github.com/NVIDIA/TransformerEngine.git/':
Empty reply from server
```

这不是 CUDA 或 TE 源码问题，而是联网节点到 GitHub 的 HTTP 连接不稳定。
迁移前曾用独立 clone 绕过；当前布局应直接对父仓库的递归 submodule
命令设置 HTTP/1.1：

```bash
git -c http.version=HTTP/1.1 \
  submodule update --init --checkout --recursive \
  third_party/TransformerEngine third_party/Megatron-LM
```

中间曾手动中断过一次递归 submodule update。当前 `git submodule status --recursive` 的所有行均以空格开头，没有 `-` 或 `+`，说明最终已经完整拉取且提交匹配。复现时不要把 clone 成功等同于源码准备完成，必须再检查递归 submodule。

## 6. Transformer Engine：CUDA 节点编译与安装

### 6.1 安装前检查

在 H20 节点进入仓库并确认 PyTorch/CUDA：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
PY=.venv/bin/python

"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

预期：

```text
2.11.0+cu128 12.8
```

由于安装命令使用 `--no-deps`，Python 运行时依赖必须已经存在。本次缺失过的关键依赖是 `pydantic`，当前版本为 2.13.5。若新环境缺失，可先安装：

```bash
uv pip install --python "$PY" pydantic
```

如果 CUDA 节点不能联网，可在联网节点对共享 `.venv` 执行上述命令，或先下载 wheel 再离线安装。

### 6.2 最终采用的安装命令

cuDNN 来自 Python wheel，构建系统未必会自动从 site-packages 发现它，因此显式添加头文件和动态库路径。TE 的 NCCL Expert Parallel 可选扩展本次不需要，使用 `NVTE_WITH_NCCL_EP=0` 禁用：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo

CUDNN_ROOT=/home/LOCAL/shixuan/hongyu/min_fa3_demo/.venv/lib/python3.12/site-packages/nvidia/cudnn

(
  set -o pipefail
  NVTE_WITH_NCCL_EP=0 \
  CPATH="$CUDNN_ROOT/include${CPATH:+:$CPATH}" \
  LIBRARY_PATH="$CUDNN_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
  LD_LIBRARY_PATH="$CUDNN_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  MAX_JOBS=8 \
  ./third_party/install_transformer_engine.sh install \
    2>&1 | tee install_transformer_engine.log
)
rc=$?
echo "TE install exit code: $rc"
```

这里把 pipeline 放在 subshell 中，使 `pipefail` 可以保留安装脚本的真实退出码，又不会退出当前 SSH login shell。此前使用过末尾的 `exit "$rc"`，所以命令完成后终端显示了 `logout`；那不是编译错误，而是主动退出了远端 shell。后续命令不再这样写。

脚本内部固定了以下关键构建选项：

```text
TORCH_CUDA_ARCH_LIST=9.0
NVTE_CUDA_ARCHS=90
NVTE_FRAMEWORK=pytorch
MAX_JOBS=8
```

实际安装使用：

```text
uv pip install
  --offline
  --no-build-isolation
  --no-deps
  <固定的 TE 源码目录>
```

`NVTE_WITH_NCCL_EP=0` 只关闭 TE 自身可选的 NCCL Expert Parallel 扩展。它不会关闭 PyTorch distributed/NCCL，也不妨碍本次 CP=8 通信。当前 benchmark 的 EP 为 1，因此不需要该扩展。

### 6.3 遇到的问题：安装成功后 verify 因 pydantic 失败

第一次构建的关键信息为：

```text
Built transformer-engine
Installed transformer-engine==2.17.1+4329ff84
```

但安装脚本随后的 import 验证失败：

```text
ModuleNotFoundError: No module named 'pydantic'
```

错误发生在：

```text
transformer_engine.common.recipe
  -> from pydantic.dataclasses import dataclass
```

这说明 TE wheel/extension 已经完成构建和安装，失败的是安装后的 Python import probe。原因是我们有意使用 `--no-deps`，不会自动安装缺失依赖。修复方式是单独安装 `pydantic`，随后只执行 `verify`；不需要因为这一错误重新编译 TE。

注意：当前 `install_transformer_engine.log` 保留的是这次“构建成功、首次 verify 失败”的日志；最终成功状态记录在 `verify_transformer_engine.log`。

### 6.4 最终 verify

在 H20 节点执行：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo

CUDNN_ROOT=/home/LOCAL/shixuan/hongyu/min_fa3_demo/.venv/lib/python3.12/site-packages/nvidia/cudnn

(
  set -o pipefail
  LD_LIBRARY_PATH="$CUDNN_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./third_party/install_transformer_engine.sh verify \
    2>&1 | tee verify_transformer_engine.log
)
rc=$?
echo "TE verify exit code: $rc"
```

最终输出：

```text
PyTorch: 2.11.0+cu128
PyTorch CUDA: 12.8
Transformer Engine Python: .../transformer_engine/__init__.py
Transformer Engine extension: .../transformer_engine_torch.cpython-312-x86_64-linux-gnu.so
transformer-engine: 2.17.1+4329ff84
Megatron TE spec: TransformerLayerSubmodules
Transformer Engine installation: OK
TE verify exit code: 0
```

原生扩展已安装为：

```text
.venv/lib/python3.12/site-packages/transformer_engine/
  transformer_engine_torch.cpython-312-x86_64-linux-gnu.so
```

verify 不仅 import TE，还通过当前 Megatron-LM 调用了：

```python
get_gpt_layer_with_transformer_engine_submodules()
```

因此确认 Megatron 可以构造 TE layer spec。

### 6.5 flash-attn 2.8.4 警告的判断

verify 输出包含：

```text
Supported flash-attn versions are >= 2.1.1, <= 2.8.3.
Found flash-attn 2.8.4.
```

讨论中注意到本地 FA3 源码原本预期类似 `2.8.3.post*`，但 Python distribution metadata 实际报告 `flash-attn==2.8.4`。对 Python 包判断应以当前环境的 distribution metadata 为准，因此 TE/Megatron 识别为 2.8.4 是正常的。

这是兼容性 warning，不是 TE 安装失败：

- TE Python 包和原生扩展均可导入；
- Megatron TE spec 可以构造；
- verify 最终退出码为 0；
- 当前 Mega CP 路径不依赖用 TE 调用该本地 flash-attn 作为核心 CP attention backend。

为了减少环境扰动，本次没有仅为消除 warning 而降级 flash-attn。若后续 benchmark 明确走 TE 的 flash-attn backend，并出现真实的 API/运行错误，再单独将 flash-attn 固定到 TE 声明支持的 `<=2.8.3`。不要把这个 warning 与 Magi FFA 的编译产物混为一谈。

### 6.6 其他非阻塞 warning

verify 还出现：

```text
absl.logging is not installed
Apex is not installed. Falling back to Torch Norm
```

当前判断：

- `absl.logging` 缺失只导致 Megatron 使用 Python 标准 logging；
- Apex 缺失时 Megatron 回退到 Torch Norm；
- 两者均没有阻止 TE spec 构造；
- 本轮任务不需要为这些 warning 扩大依赖安装范围。

## 7. MagiAttention：源码准备

### 7.1 固定版本

固定 MagiAttention 提交为：

```text
872717e1f88fa6938593e452a28a41597c849a00
```

它是当前仓库的 submodule：

```text
third_party/MagiAttention
```

### 7.2 clone 命令

仓库当前直接记录 HTTPS URL，不要求 GitHub SSH identity：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo

git submodule update --init --checkout --recursive third_party/MagiAttention
```

也可以直接使用封装脚本：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
./third_party/install_magi_attention.sh fetch 2>&1 | tee fetch_magi_attention.log
```

最终核对到的主要递归 submodule 包括：

```text
magi_attention/csrc/cutlass
magi_attention/functional/flash-attention
magi_attention/functional/flash-attention/csrc/composable_kernel
magi_attention/functional/flash-attention/csrc/cutlass
magi_attention/functional/flash-attention/third_party/aiter
```

所有 submodule 均处于 Magi 固定提交记录的 revision。

## 8. MagiAttention：CUDA 节点编译与安装

### 8.1 最终采用的安装方式

在 H20 节点执行。关键点是提供 cuDNN/NVSHMEM 运行库路径，但不要设置 `NVSHMEM_DIR`：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo

CUDNN_ROOT=/home/LOCAL/shixuan/hongyu/min_fa3_demo/.venv/lib/python3.12/site-packages/nvidia/cudnn
NVSHMEM_ROOT=/home/LOCAL/shixuan/hongyu/min_fa3_demo/.venv/lib/python3.12/site-packages/nvidia/nvshmem

unset NVSHMEM_DIR

(
  set -o pipefail
  CPATH="$CUDNN_ROOT/include${CPATH:+:$CPATH}" \
  LIBRARY_PATH="$NVSHMEM_ROOT/lib:$CUDNN_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
  LD_LIBRARY_PATH="$NVSHMEM_ROOT/lib:$CUDNN_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  MAX_JOBS=8 \
  ./third_party/install_magi_attention.sh install \
    2>&1 | tee install_magi_attention.log
)
rc=$?
echo "Magi install exit code: $rc"
```

脚本内部使用：

```text
TORCH_CUDA_ARCH_LIST=9.0
MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=1
MAX_JOBS=8
```

并从固定源码目录执行离线、无 build isolation、无依赖解析的安装。

### 8.2 NVSHMEM 的问题与结论

当前 NVIDIA NVSHMEM wheel 提供的实际 host library 是版本化文件：

```text
.venv/lib/python3.12/site-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3
```

它没有无版本后缀的：

```text
libnvshmem_host.so
```

讨论和排查中确认：如果显式设置 `NVSHMEM_DIR`，Magi 的构建逻辑会按开发版 SDK 布局查找/链接无版本的 `libnvshmem_host.so`，从而失败。最终做法是：

- `unset NVSHMEM_DIR`；
- 通过 `LIBRARY_PATH` 和 `LD_LIBRARY_PATH` 暴露 wheel 的 `lib/`；
- 直接使用已经存在的 `libnvshmem_host.so.3`；
- 不创建额外的伪 symlink，不修改 wheel 安装目录。

这避免了对共享 Python 环境做不可追踪的库文件改动。

### 8.3 最终安装状态

`install_magi_attention.log` 记录：

```text
Built magi-attention
Installed magi-attention==1.1.1.post16+g872717e1
MagiAttention probe: True None
MagiAttention installation: OK
```

两个关键原生扩展均存在：

```text
.venv/lib/python3.12/site-packages/magi_attention/
  magi_attn_ext.cpython-312-x86_64-linux-gnu.so
  magi_attn_comm.cpython-312-x86_64-linux-gnu.so
```

其中：

- `magi_attn_ext` 是 Magi 的核心原生扩展；
- `magi_attn_comm` 包含其通信相关原生实现；
- 安装后的 probe 同时 import 这两个扩展，并调用仓库 benchmark 使用的 `probe_magi_attention()`。

### 8.4 独立 verify 命令

安装脚本末尾已经执行过 probe。若需要额外留档，可在 H20 上运行：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
MAGI_ATTENTION_WORKSPACE_BASE=/home/LOCAL/shixuan/hongyu/min_fa3_demo/.cache/mega_cp/magi_ffa_sm90_bf16_hd128 \
./third_party/install_magi_attention.sh verify \
  2>&1 | tee verify_magi_attention.log
```

如果新 shell 中因动态库查找失败，再补全：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo

NVSHMEM_ROOT=/home/LOCAL/shixuan/hongyu/min_fa3_demo/.venv/lib/python3.12/site-packages/nvidia/nvshmem
CUDNN_ROOT=/home/LOCAL/shixuan/hongyu/min_fa3_demo/.venv/lib/python3.12/site-packages/nvidia/cudnn

LD_LIBRARY_PATH="$NVSHMEM_ROOT/lib:$CUDNN_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MAGI_ATTENTION_WORKSPACE_BASE=/home/LOCAL/shixuan/hongyu/min_fa3_demo/.cache/mega_cp/magi_ffa_sm90_bf16_hd128 \
./third_party/install_magi_attention.sh verify \
  2>&1 | tee verify_magi_attention.log
```

## 9. TE 编译的 attention kernel 能否给 Magi FFA 复用

讨论结论是：不能把 TE 编译时生成的 attention kernel 当作 Magi FFA 内核直接复用。

原因不是二者都使用 CUDA 就可以共享二进制，而是它们的调用栈、代码生成参数、注册方式和 ABI 不同：

- TE 的扩展属于 `transformer_engine`，由 TE 的模块和 backend 调用；
- Magi benchmark 实际调用 `magi_attention.functional.flex_flash_attn`；
- Magi FFA 根据自己的 JIT spec 生成 URI、源码实例化、缓存目录和 `.so`；
- Magi distributed attention 需要自己的 partial output/gradient accumulation 语义；
- TE 安装完成只说明 TE extension 可用，不会自动产生 Magi FFA 期望的 AOT URI。

因此仍需单独编译 Magi 的 FFA kernel。

## 10. Magi FFA 与现有 flash-attn/FA3 是否冲突

讨论结论是：按本次布局不会直接冲突。

当前涉及的产物有不同的 Python namespace 和加载位置：

```text
现有 flash-attn:
  flash_attn Python package / 自己的 CUDA extension

Transformer Engine:
  transformer_engine/transformer_engine_torch*.so

Magi 原生扩展:
  magi_attention/magi_attn_ext*.so
  magi_attention/magi_attn_comm*.so

Magi FFA JIT/AOT:
  magi_attention/lib/flex_flash_attn_sm_90_*/...
  独立 MAGI_ATTENTION_WORKSPACE_BASE
```

Magi 源码树中虽然包含自己的 flash-attention submodule，但其 FFA JIT 产物由 Magi 的模块加载，不会覆盖仓库已有的 `flash-attn==2.8.4` distribution。为进一步降低冲突风险，本次：

- 没有卸载或覆盖现有 flash-attn；
- 没有复用 flash-attn 的 build/cache 目录；
- 为 Magi 指定独立 workspace；
- 没有创建全局 CUDA/NVSHMEM symlink；
- 没有把 Magi 的内部 flash-attention 路径加入全局 `PYTHONPATH`。

## 11. 为什么只定向预编译三个 FFA 内核

完整预编译 Magi FFA 的所有排列组合成本很高，而且本 benchmark 的形状已经固定。为了减少编译时间和无关测试，仅预编译当前训练路径需要的组合：

```text
GPU architecture       SM90/SM90a（H20）
compute dtype          BF16
head dimension         128
Q heads                32
KV heads               8
GQA ratio              4:1
direction              forward + backward
forward partial/output FP32
backward dQ/dK/dV      FP32 partial accumulation
range_merge            false
pack_gqa               false
cat_gqa                false
```

最终三种 spec 为：

1. forward，允许 FP32 atomic accumulation；
2. forward，禁用 atomic reduction，写出新的 FP32 output；
3. backward，FP32 dQ/dK/dV partial accumulation。

Magi distributed FFA 路径可能根据当前操作是累加 partial output，还是写出新 output，选择两个 forward variant 中的一个，所以两个都需要。

为确保 fresh 环境不会在安装 Magi package 时先编译其上游默认的通用组合矩阵，
`install_magi_attention.sh` 固定设置 `MAGI_ATTENTION_PREBUILD_FFA=0`。package
安装完成后才由下一节的脚本精确构造并编译这三个 spec。完整 fresh 脚本还会设置
`FORCE_REBUILD=1`，只删除这三个 URI 的旧 JIT/AOT 目录后重建，避免跨旧
PyTorch/CUDA 环境复用二进制；不会清理其他 Magi cache。

## 12. GQA 4:1 为什么不生成 `packgqa4` 专用 URI

本 benchmark 使用 `qhead=32`、`kvhead=8`，但普通 GQA 的 head 数和比率是运行时 tensor metadata，不是这三个 kernel URI 的静态模板维度。

当前 benchmark 没有启用 Magi 的实验性 PackGQA/CatGQA 重排算法，因此 spec 使用：

```text
pack_gqa=False
cat_gqa=False
pack_gqa_factor=1
```

这仍然支持普通 GQA 4:1。强行预编译 `packgqa4` 会改变所测算法和数据排列语义，而不只是为同一个算法补一个优化二进制，因此没有这样做。

## 13. 定向 FFA 预编译脚本

脚本路径：

```text
/home/LOCAL/shixuan/hongyu/min_fa3_demo/third_party/precompile_magi_ffa_training.sh
```

在 H20 节点只需执行短命令：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
./third_party/precompile_magi_ffa_training.sh
```

脚本会：

1. 检查 Python、`nvcc` 和 `libnvshmem_host.so.3`；
2. 设置 CUDA、cuDNN、NVSHMEM 的 include/library/runtime 路径；
3. 固定 `TORCH_CUDA_ARCH_LIST=9.0` 和 Magi compute capability 90；
4. 使用独立 workspace；
5. 调用 Magi 生产代码的 `get_ffa_jit_spec()` 构造三个 spec；
6. 调用每个 spec 的 `build()`；
7. 将 workspace 中的 `.so` 复制到已安装 Magi package 的 AOT `lib/`；
8. 检查每个目标目录中确实存在 `.so`；
9. 将完整输出写入 `.cache/mega_cp/logs/precompile_magi_ffa_training.log`。

关键环境变量：

```text
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=/usr/local/cuda-12.8
TORCH_CUDA_ARCH_LIST=9.0
MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90
MAGI_ATTENTION_KERNEL_BACKEND=ffa
MAGI_ATTENTION_RANGE_MERGE=0
MAGI_ATTENTION_CATGQA=0
MAGI_ATTENTION_NO_BUILD_CACHE=0
MAGI_ATTENTION_FORCE_JIT_BUILD=0
MAX_JOBS=8
NVCC_THREADS=2
```

预编译只需要一张可见 GPU/一套 CUDA 编译环境，不需要启动 8 卡 distributed job。

## 14. 遇到的问题：预编译脚本误引入测试依赖

第一版定向预编译脚本导入：

```python
magi_attention.testing.precompile
```

这会沿着 Magi 测试工具链引入额外依赖，最终失败：

```text
ModuleNotFoundError: No module named 'expecttest'
```

`expecttest` 是测试框架依赖，不是生产 FFA JIT 编译必须依赖。为此没有扩大环境、安装整套测试依赖，而是将脚本改为直接使用生产接口：

```python
from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_spec
```

修正后的脚本不依赖 `expecttest`、`pytest` 或 Magi testing package，并已成功生成三个内核。这一处理符合当前“减少测试次数、避免无关依赖”的原则。

## 15. 最终 FFA 预编译结果

日志：

```text
/home/LOCAL/shixuan/hongyu/min_fa3_demo/precompile_magi_ffa_training.log
mtime 2026-08-29 02:44:32 +0800
size  150268 bytes
```

成功标志：

```text
Targeted FFA kernel count: 3
[precompile 1/3] OK
[precompile 2/3] OK
[precompile 3/3] OK
Targeted Magi FFA training kernels: OK
```

脚本在 Python 编译 pipeline 成功后还会向终端输出：

```text
Magi FFA precompile: OK
```

该行位于内部 `tee "$LOG_FILE"` 之后，因此不会写入当前的 `precompile_magi_ffa_training.log`；日志内应以 `Targeted Magi FFA training kernels: OK` 作为最终成功标志。

三个目标 URI：

```text
flex_flash_attn_sm_90_bwd_128hd_compute_bfloat16_dq_float32_dkv_float32_atomic_mmunified_pr40_cr232

flex_flash_attn_sm_90_fwd_128hd_compute_bfloat16_out_float32_atomic_m128n128_pr40_cr232

flex_flash_attn_sm_90_fwd_128hd_compute_bfloat16_out_float32_m128n128_pr40_cr232
```

已安装到 Magi package 的三个 AOT `.so`：

```text
3,930,800 bytes  backward
3,903,984 bytes  forward atomic
3,895,680 bytes  forward non-atomic
```

目录结构为：

```text
.venv/lib/python3.12/site-packages/magi_attention/lib/
  flex_flash_attn_sm_90_bwd_.../
    flex_flash_attn_sm_90_bwd_....so
  flex_flash_attn_sm_90_fwd_..._atomic_.../
    flex_flash_attn_sm_90_fwd_..._atomic_....so
  flex_flash_attn_sm_90_fwd_.../
    flex_flash_attn_sm_90_fwd_....so
```

workspace 中共有 4 个 `.so`：

```text
1 个共享辅助库:
  cached_ops/128hd_common/128hd_common.so

3 个目标 FFA 内核:
  cached_ops/flex_flash_attn_sm_90_bwd_.../*.so
  cached_ops/flex_flash_attn_sm_90_fwd_..._atomic_.../*.so
  cached_ops/flex_flash_attn_sm_90_fwd_.../*.so
```

因此“workspace 里有 4 个 `.so`”与“目标内核数量为 3”并不矛盾；多出的一个是三个内核共同链接的 `128hd_common.so`。

## 16. FFA 编译 warning 的判断

日志中出现过以下 warning：

```text
128hd_common build.ninja file has been changed
capturing structured bindings is a C++20 feature
local memory used / 16-byte stack frame
no return statement in function returning non-void
```

当前判断为非阻塞 warning，依据是：

- 三次目标 kernel 的 CUDA 编译和链接均完成；
- 三个目标目录都生成非空 `.so`；
- 三个产物都被复制到 Magi AOT package 目录；
- 日志最终输出三个 `[precompile n/3] OK` 和总 `OK`；
- 没有 `FAILED`、异常 traceback 或非零退出状态留在最终日志。

这些 warning 值得在真正运行 kernel 时继续观察，但当前没有理由因此重复编译或修改 Magi 内核源码。

## 17. 当前验证边界

已经确认：

- TE 固定源码和递归 submodule 完整；
- TE Python 包和 PyTorch extension 可导入；
- 当前 Megatron 可以构造 TE Transformer layer spec；
- Magi 固定源码和递归 submodule 完整；
- Magi Python API、`magi_attn_ext` 和 `magi_attn_comm` 可导入；
- 仓库的 Magi probe 返回 `True`；
- 三个定向 FFA 内核完成编译、链接并安装到 AOT 目录。

尚未由这些安装步骤证明：

- 8 卡 CP=8 的进程组、通信和数据排列完整运行；
- 128K 输入下三个 FFA 内核是否全部按预期命中 AOT 而不触发新 JIT；
- 单层 Transformer forward + backward 的端到端显存和时间表现；
- 严格数值正确性。

这些属于后续 Mega CP benchmark 的职责。为了遵循“尽量减少测试次数”的要求，本次没有额外启动重复的多卡正确性/性能测试。

## 18. 推荐的最短复现顺序

### 联网节点

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
./setup_fresh_environment.sh prepare
```

若 GitHub clone 再次遇到 `Empty reply from server`，可对标准
`git submodule update --init --checkout --recursive` 临时设置
`git -c http.version=HTTP/1.1`；不要恢复迁移前仓库外的 TE clone 目录。

### H20 CUDA 节点

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
CUDA_VISIBLE_DEVICES=0 ./setup_fresh_environment.sh install
```

`install` 会先检查 `.venv` 中的固定 Python/Torch/CUDA wheel 版本，再依次构建
min-FA3、TE、Magi 和三个目标 FFA kernel，最后执行完整验证。已有当前产物时不要
重复执行 `install`；只需检查环境时运行：

```bash
CUDA_VISIBLE_DEVICES=0 ./setup_fresh_environment.sh verify
```

## 19. 何时必须重新编译

以下变化会使现有产物不再具有可比性，建议重新安装或重新预编译：

- PyTorch、CUDA major/minor、Python ABI 变化；
- TE 或 Magi commit 变化；
- GPU architecture 不再是 SM90；
- compute dtype 不再是 BF16；
- head dimension 不再是 128；
- 启用 PackGQA、CatGQA、RangeMerge、softcap、deterministic 或其他会改变 FFA spec URI 的选项；
- forward/backward partial accumulation dtype 改变；
- 已安装 Magi package 被重装，导致其 `lib/` 下 AOT 产物被覆盖。

以下变化本身通常不要求重新编译当前普通 GQA kernel：

- Q/KV head 数改变，但仍是 Magi 当前 kernel 支持的普通 GQA，且不启用 PackGQA/CatGQA；
- sequence length 或 batch 中 case 的排列改变；
- CP rank 对输入的重新分配或 padding 变化。

不过后一类变化仍必须通过真实 benchmark 验证运行时行为和性能。

## 20. 快速状态检查命令

检查版本：

```bash
cd /home/LOCAL/shixuan/hongyu/min_fa3_demo
.venv/bin/python - <<'PY'
from importlib.metadata import version
import torch

print("torch", torch.__version__, torch.version.cuda)
print("transformer-engine", version("transformer-engine"))
print("magi-attention", version("magi-attention"))
print("flash-attn", version("flash-attn"))
PY
```

检查固定提交：

```bash
git -C /home/LOCAL/shixuan/hongyu/min_fa3_demo/third_party/TransformerEngine rev-parse HEAD
git -C /home/LOCAL/shixuan/hongyu/min_fa3_demo/third_party/MagiAttention rev-parse HEAD
```

检查三个已安装 AOT kernel：

```bash
find /home/LOCAL/shixuan/hongyu/min_fa3_demo/.venv/lib/python3.12/site-packages/magi_attention/lib \
  -type f -name 'flex_flash_attn_sm_90_*.so' -print
```

检查预编译成功标记：

```bash
rg 'Targeted FFA kernel count|\[precompile [0-9]+/[0-9]+\] OK|Targeted Magi FFA training kernels' \
  /home/LOCAL/shixuan/hongyu/min_fa3_demo/.cache/mega_cp/logs/precompile_magi_ffa_training.log
```

预期恰好看到：

```text
Targeted FFA kernel count: 3
[precompile 1/3] OK: ...
[precompile 2/3] OK: ...
[precompile 3/3] OK: ...
Targeted Magi FFA training kernels: OK
```

## 21. 最终结论

截至本文记录时间：

- Transformer Engine 已成功安装并通过 Megatron TE spec 验证；
- MagiAttention 已成功安装，Python API 和两个原生扩展通过 probe；
- 针对 H20/SM90、BF16、head dimension 128、普通 GQA 4:1 训练路径的三个 FFA kernel 已全部预编译并放入 Magi AOT 目录；
- TE 的 `flash-attn==2.8.4` 版本提示和 FFA 编译 warning 均为当前已知的非阻塞项；
- 不需要再次安装 TE/Magi，也不需要重复预编译 FFA；
- 下一步应直接进入单层 Transformer、CP=8、128K 输入的端到端 benchmark，由该 benchmark 完成真正的多卡运行验证。
