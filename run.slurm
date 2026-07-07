#!/bin/bash
#SBATCH --job-name=min_fa3_demo_h200
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00

source /etc/profile.d/modules.sh
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate megatron
module load cuda/12.8 gcc/13.3.0 cmake/4.2.0 nccl/2.27_cuda12.8 cudnn/9.6.0.74_cuda12

export CUDA_HOME=/data/apps/cuda/12.8
export CUDNN_HOME=/data/apps/cudnn/9.6.0.74_cuda12
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDNN_HOME/include:$CPLUS_INCLUDE_PATH
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128

cd /data/home/scyb683/run/min_fa3_demo

# make clean && make

# Flash Attention 3 (FA3) benchmark:
# echo "Running tests and benchmarks for BSHD style FA3"
# echo "========================================================="
# python test_min_fa3.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 8 --d 128 --mode both
# python benchmark.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 8 --d 128 --mode both
# python test_min_fa3.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 8 --d 128 --mode both --manual-block-count 128
# python benchmark.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 8 --d 128 --mode both --manual-block-count 128
# echo "========================================================="
# python test_min_fa3.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 32 --d 128 --mode both
# python benchmark.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 32 --d 128 --mode both
# echo "========================================================="

# echo "Running tests and benchmarks for variable-length FA3"
# echo "========================================================="
# python test_min_fa3_varlen.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --mode both
# python benchmark_varlen.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --mode both
# # echo "========================================================="
# python test_min_fa3_varlen.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 32 --headdim 128 --mode both
# python benchmark_varlen.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 32 --headdim 128 --mode both
# echo "========================================================="

# Remote-load benchmark note:
# This script requests 2 GPUs, so benchmark_parallel_remote_load.py can be run with:
# torchrun --nproc_per_node=2 benchmark_parallel_remote_load.py --shape 4096x4096 --src-rank 0

# Single-Step Ring-FlashAttention benchmark:
# echo "Running tests and benchmarks for Ring-FlashAttention"
# echo "========================================================="
# python test_min_fa3_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --num-comp-sm 128 --num-comm-sm 0 --mode both
# python benchmark_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 128 --num-comm-sm 0 --mode both
# python test_min_fa3_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --num-comp-sm 128 --num-comm-sm 4 --mode both
# python benchmark_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 128 --num-comm-sm 4 --mode both
# python benchmark_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 124 --num-comm-sm 8 --mode both
# python benchmark_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 116 --num-comm-sm 16 --mode both
# python benchmark_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 108 --num-comm-sm 24 --mode both

# Single-Step Mega-Ring-FlashAttention varlen benchmark:
# echo "Running tests and benchmarks for Mega-Ring-FlashAttention"
# echo "========================================================="
# python mega_ring_test_min_fa3_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --num-comp-sm 128 --num-comm-sm 0 --mode both
# python benchmark_varlen_mega_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 128 --num-comm-sm 0 --mode both
# python mega_ring_test_min_fa3_varlen_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --num-comp-sm 128 --num-comm-sm 4 --mode both
# python benchmark_varlen_mega_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 128 --num-comm-sm 4 --mode both
# python benchmark_varlen_mega_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 124 --num-comm-sm 8 --mode both
# python benchmark_varlen_mega_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 116 --num-comm-sm 16 --mode both
# python benchmark_varlen_mega_ring_local.py --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 108 --num-comm-sm 24 --mode both

# echo "========================================================="
# nsys profile -t cuda,nvtx,osrt -o my_report --stats=true python benchmark_varlen_mega_ring_local.py --profile --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 --num-comp-sm 116 --num-comm-sm 16 --mode both

# Multi-Step Multi-rank full Ring-FlashAttention forward benchmark:
# This script requests a 2-GPU allocation for these examples.
echo "Running multi-rank ring_test benchmark with correctness checks"
echo "========================================================="
torchrun --standalone --nproc_per_node=4 ring_test/benchmark_ring_forward.py \
    --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 \
    --mode both --methods all --num-comp-sm 128 --num-comm-sm 4 \
    --warmup-iters 20 --num-iters 30 --check

echo "Running multi-rank ring_test benchmark without correctness checks"
echo "========================================================="
torchrun --standalone --nproc_per_node=4 ring_test/benchmark_ring_forward.py \
    --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 \
    --mode both --methods all --num-comp-sm 128 --num-comm-sm 4 \
    --warmup-iters 20 --num-iters 30 --no-check
torchrun --standalone --nproc_per_node=4 ring_test/benchmark_ring_forward.py \
    --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 \
    --mode both --methods all --num-comp-sm 124 --num-comm-sm 8 \
    --warmup-iters 20 --num-iters 30 --no-check
torchrun --standalone --nproc_per_node=4 ring_test/benchmark_ring_forward.py \
    --b 16 --seqlen 512,1024,2048,4096,8192 --qhead 32 --kvhead 8 --headdim 128 \
    --mode both --methods all --num-comp-sm 116 --num-comm-sm 16 \
    --warmup-iters 20 --num-iters 30 --no-check

# Memory sanitizer runs:
# compute-sanitizer --tool memcheck python test_min_fa3_varlen_ring_local.py --b 16 --seqlen 1024,4096 --qhead 32 --kvhead 8 --num-comp-sm 128 --num-comm-sm 0 --mode both > memcheck-128-0.log 2>&1
# compute-sanitizer --tool memcheck python test_min_fa3_varlen_ring_local.py --b 16 --seqlen 1024,4096 --qhead 32 --kvhead 8 --num-comp-sm 128 --num-comm-sm 4 --mode both > memcheck-128-4.log 2>&1
