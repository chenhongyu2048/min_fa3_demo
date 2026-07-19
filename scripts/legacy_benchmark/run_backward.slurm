#!/bin/bash
#SBATCH --job-name=min_fa3_backward_h200
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=01:30:00

set -euo pipefail

source /etc/profile.d/modules.sh
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate megatron
module load cuda/12.8 gcc/13.3.0 cmake/4.2.0 nccl/2.27_cuda12.8 cudnn/9.6.0.74_cuda12

export CUDA_HOME=/data/apps/cuda/12.8
export CUDNN_HOME=/data/apps/cudnn/9.6.0.74_cuda12
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}
export CPLUS_INCLUDE_PATH=$CUDNN_HOME/include:${CPLUS_INCLUDE_PATH:-}

cd /data/home/scyb683/run/min_fa3_demo

LLAMA3_HEADS_K_STRIDE=${LLAMA3_HEADS_K_STRIDE:-1}

# The login node can build the extension. Uncomment when the job should rebuild it.
# MAX_JOBS=2 make

echo "Running batched backward correctness: MHA"
python test_min_fa3_backward.py \
    --b 2 --seqlen 128,129 --qhead 8 --kvhead 8 --headdim 128 --mode both
python test_min_fa3_backward.py \
    --b 2 --seqlen 128,129 --qhead 8 --kvhead 8 --headdim 128 --mode both --deterministic

echo "Running batched backward correctness: GQA"
python test_min_fa3_backward.py \
    --b 2 --seqlen 128,129 --qhead 8 --kvhead 2 --headdim 128 --mode both
python test_min_fa3_backward.py \
    --b 2 --seqlen 128,129 --qhead 8 --kvhead 2 --headdim 128 --mode both --deterministic

echo "Running varlen backward correctness: MHA"
python test_min_fa3_varlen_backward.py \
    --b 3 --seqlen 128,129 --qhead 8 --kvhead 8 --headdim 128 --mode both
python test_min_fa3_varlen_backward.py \
    --b 3 --seqlen 128,129 --qhead 8 --kvhead 8 --headdim 128 --mode both --deterministic

echo "Running varlen backward correctness: GQA"
python test_min_fa3_varlen_backward.py \
    --b 3 --seqlen 128,129 --qhead 8 --kvhead 2 --headdim 128 --mode both
python test_min_fa3_varlen_backward.py \
    --b 3 --seqlen 128,129 --qhead 8 --kvhead 2 --headdim 128 --mode both --deterministic

echo "Benchmarking batched backward against FA3: MHA"
python benchmark_backward.py \
    --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 32 --headdim 128 \
    --mode both --warmup-iters 10 --num-iters 50
python benchmark_backward.py \
    --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 32 --headdim 128 \
    --mode both --deterministic --warmup-iters 10 --num-iters 50

echo "Benchmarking batched backward against FA3: GQA"
python benchmark_backward.py \
    --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 8 --headdim 128 \
    --mode both --warmup-iters 10 --num-iters 50
python benchmark_backward.py \
    --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 8 --headdim 128 \
    --mode both --deterministic --warmup-iters 10 --num-iters 50

echo "Benchmarking varlen backward against FA3: MHA"
python benchmark_varlen_backward.py \
    --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 32 --headdim 128 \
    --mode both --warmup-iters 10 --num-iters 50
python benchmark_varlen_backward.py \
    --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 32 --headdim 128 \
    --mode both --deterministic --warmup-iters 10 --num-iters 50

echo "Benchmarking varlen backward against FA3: GQA"
python benchmark_varlen_backward.py \
    --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 8 --headdim 128 \
    --mode both --warmup-iters 10 --num-iters 50
python benchmark_varlen_backward.py \
    --b 4 --seqlen 512,1024,2048,4096 --qhead 32 --kvhead 8 --headdim 128 \
    --mode both --deterministic --warmup-iters 10 --num-iters 50

echo "Checking explicit all-CP causal ring backward"
torchrun --standalone --nproc_per_node=2 \
    mega_ring_test_min_fa3_varlen_backward_multi_rank.py \
    --b 1 --seqlen 256 --qhead 16 --kvhead 8 --headdim 128 \
    --num-comp-sm 64 --num-comm-sm 8

echo "Benchmarking two-GPU all-CP causal ring backward"
torchrun --standalone --nproc_per_node=2 ring_test/benchmark_ring_backward.py \
    --b 4,4,4 --seqlen 256,512,1024 --qhead 32 --kvhead 8 --headdim 128 \
    --methods all --num-comp-sm 64 --num-comm-sm 8 \
    --llama3-heads-k-stride "$LLAMA3_HEADS_K_STRIDE" \
    --warmup-iters 5 --num-iters 20 --check

echo "Checking hierarchical G8/G4/G2/G1 causal backward"
torchrun --standalone --nproc_per_node=8 \
    mega_ring_test_min_fa3_varlen_backward_hybrid_multi_rank.py \
    --global-seqlens 2048,1024,512,256 \
    --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
    --qhead 16 --kvhead 8 --headdim 128 --repeat 2 \
    --num-comp-sm 100 --num-comm-sm 16

echo "Checking hierarchical backward host validation failures"
torchrun --standalone --nproc_per_node=8 \
    mega_ring_test_min_fa3_varlen_backward_validation_multi_rank.py

echo "Benchmarking hierarchical G8/G4/G2/G1 causal backward"
torchrun --standalone --nproc_per_node=8 \
    ring_test/benchmark_hybrid_backward.py \
    --b 1,4 --seqlen 256,256 \
    --ring-sizes 8,4,2,1 --ring-starts 0,4,2,7 \
    --qhead 16 --kvhead 8 --headdim 128 \
    --methods all \
    --num-comp-sm 100 --num-comm-sm 16 \
    --llama3-heads-k-stride "$LLAMA3_HEADS_K_STRIDE" \
    --warmup-iters 5 --num-iters 20

# Optional full dataset-shaped sweep. Keep --no-check for the 128K workload;
# the dense autograd reference has quadratic score memory.
# torchrun --standalone --nproc_per_node=8 \
#     ring_test/benchmark_hybrid_dataset_backward.py \
#     --dataset arxiv --target-tokens 131072 --seed 0 --num-cases 4 \
#     --qhead 32 --kvhead 8 --headdim 128 \
#     --methods all \
#     --sm-configs 128:4,124:8,120:12,116:16 \
#     --warmup-iters 10 --num-iters 40 --no-check

# Optional memory checks for the non-tile-aligned GQA paths:
# compute-sanitizer --tool memcheck python test_min_fa3_backward.py \
#     --b 2 --seqlen 129 --qhead 8 --kvhead 2 --headdim 128 --mode both --deterministic
# compute-sanitizer --tool memcheck python test_min_fa3_varlen_backward.py \
#     --b 3 --seqlen 129 --qhead 8 --kvhead 2 --headdim 128 --mode both --deterministic
