#!/bin/bash
#SBATCH --job-name=min_fa3_hierarchical_ring
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
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

cd /data/home/scyb683/run/min_fa3_demo

make clean
make -j

NUM_COMP_SM=${NUM_COMP_SM:-116}
NUM_COMM_SM=${NUM_COMM_SM:-16}
REPEAT=${REPEAT:-20}

echo "== G2 + G1 hierarchical forward, causal and noncausal =="
torchrun --standalone --nproc_per_node=2 \
    mega_ring_test_min_fa3_varlen_hybrid_multi_rank.py \
    --global-seqlens 2048,1024 \
    --ring-sizes 2,1 \
    --ring-starts 0,1 \
    --qhead 16 \
    --kvhead 8 \
    --headdim 128 \
    --num-comp-sm "$NUM_COMP_SM" \
    --num-comm-sm "$NUM_COMM_SM" \
    --mode both \
    --check-arena \
    --repeat "$REPEAT"

echo "== G1 only with rank 0 empty, causal and noncausal =="
torchrun --standalone --nproc_per_node=2 \
    mega_ring_test_min_fa3_varlen_hybrid_multi_rank.py \
    --global-seqlens 1024 \
    --ring-sizes 1 \
    --ring-starts 1 \
    --qhead 16 \
    --kvhead 8 \
    --headdim 128 \
    --num-comp-sm "$NUM_COMP_SM" \
    --num-comm-sm 0 \
    --mode both \
    --check-arena \
    --repeat "$REPEAT"
