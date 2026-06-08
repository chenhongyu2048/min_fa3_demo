#!/bin/bash
#SBATCH --job-name=min_fa3_demo_h100
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:10:00

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# make clean && make
echo "Running tests and benchmarks for BSHD style FA3"
echo "========================================================="
python test_min_fa3.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 8 --d 128 --mode both
python benchmark.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 8 --d 128 --mode both
python test_min_fa3.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 8 --d 128 --mode both --manual-block-count 128
python benchmark.py --b 16 --seqlen 512,1024,2048,4096,8192 --qo-head 32 --kv-head 8 --d 128 --mode both --manual-block-count 128
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
# This script requests 1 GPU, so it cannot run benchmark_parallel_remote_load.py as-is.
# Use a 2+ GPU job and launch, for example:
# torchrun --nproc_per_node=2 benchmark_parallel_remote_load.py --shape 4096x4096 --src-rank 0
