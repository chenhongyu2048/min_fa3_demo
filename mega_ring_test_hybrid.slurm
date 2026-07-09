#!/bin/bash
#SBATCH --job-name=min_fa3_demo_h200
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
export https_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128
export http_proxy=http://u-MS9MdQ:Qixfk8ku@10.248.0.7:3128

cd /data/home/scyb683/run/min_fa3_demo

# echo "== single-GPU hybrid local =="
# CUDA_VISIBLE_DEVICES=0 python mega_ring_test_min_fa3_varlen_hybrid_local.py \
#     --seqlens 1152,4096,1408 \
#     --all-local-seqlens 1152,1408 \
#     --qhead 16 \
#     --kvhead 8 \
#     --headdim 128 \
#     --num-comp-sm 1 \
#     --num-comm-sm 0 \
#     --cp-threshold 2048 \
#     --mode both
# echo "== single-GPU legacy mega-ring local =="
# CUDA_VISIBLE_DEVICES=0 python mega_ring_test_min_fa3_varlen_ring_local.py \
#     --b 1 \
#     --seqlen 256,512 \
#     --qhead 8 \
#     --kvhead 8 \
#     --headdim 128 \
#     --num-comp-sm 1 \
#     --num-comm-sm 1 \
#     --mode both
# echo "== single-GPU hybrid all-CP explicit =="
# CUDA_VISIBLE_DEVICES=0 python mega_ring_test_min_fa3_varlen_hybrid_local.py \
#     --seqlens 4096,4352 \
#     --all-local-seqlens 1152,1408 \
#     --qhead 16 \
#     --kvhead 8 \
#     --headdim 128 \
#     --num-comp-sm 1 \
#     --num-comm-sm 0 \
#     --cp-threshold 2048 \
#     --mode both
# echo "== two-GPU distributed mega-ring all-CP =="
# torchrun --standalone --nproc_per_node=2 mega_ring_test_min_fa3_varlen_ring_multi_rank.py \
#     --b 1 \
#     --seqlen 256,512 \
#     --qhead 8 \
#     --kvhead 8 \
#     --headdim 128 \
#     --num-comp-sm 1 \
#     --num-comm-sm 1 \
#     --mode both
# torchrun --standalone --nproc_per_node=2 mega_ring_test_min_fa3_varlen_ring_multi_rank.py \
#     --b 1 \
#     --seqlen 256,512 \
#     --qhead 16 \
#     --kvhead 8 \
#     --headdim 128 \
#     --num-comp-sm 1 \
#     --num-comm-sm 1 \
#     --mode both

# == single-GPU hybrid local ==
# mega ring hybrid local case causal=False: ok (lengths=[1152, 4096, 1408], cp_count=1, threshold=2048)
# mega ring hybrid local case causal=True: ok (lengths=[1152, 4096, 1408], cp_count=1, threshold=2048)
# mega ring hybrid local case causal=False: ok (lengths=[1152, 1408], cp_count=0, threshold=2048)
# mega ring hybrid local case causal=True: ok (lengths=[1152, 1408], cp_count=0, threshold=2048)
# == single-GPU legacy mega-ring local ==
# mega ring varlen local case causal=False: ok (B=1, S=256, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=0)
# mega ring varlen local case causal=True: ok (B=1, S=256, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=0)
# mega ring varlen local case causal=False: ok (B=1, S=256, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# mega ring varlen local case causal=True: ok (B=1, S=256, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# mega ring varlen local case causal=False: ok (B=1, S=512, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=0)
# mega ring varlen local case causal=True: ok (B=1, S=512, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=0)
# mega ring varlen local case causal=False: ok (B=1, S=512, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# mega ring varlen local case causal=True: ok (B=1, S=512, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# == single-GPU hybrid all-CP explicit ==
# mega ring hybrid local case causal=False: ok (lengths=[4096, 4352], cp_count=2, threshold=2048)
# mega ring hybrid local case causal=True: ok (lengths=[4096, 4352], cp_count=2, threshold=2048)
# mega ring hybrid local case causal=False: ok (lengths=[1152, 1408], cp_count=0, threshold=2048)
# mega ring hybrid local case causal=True: ok (lengths=[1152, 1408], cp_count=0, threshold=2048)
# == two-GPU distributed mega-ring all-CP ==
# distributed mega ring varlen case causal=False: ok (world_size=2, B=1, S=256, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=True: ok (world_size=2, B=1, S=256, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=False: ok (world_size=2, B=1, S=512, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=True: ok (world_size=2, B=1, S=512, QH=8, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=False: ok (world_size=2, B=1, S=256, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=True: ok (world_size=2, B=1, S=256, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=False: ok (world_size=2, B=1, S=512, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=True: ok (world_size=2, B=1, S=512, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)

# echo "== two-GPU distributed mega-ring all-CP regression =="
# torchrun --standalone --nproc_per_node=2 mega_ring_test_min_fa3_varlen_ring_multi_rank.py \
#     --b 1 \
#     --seqlen 256,512 \
#     --qhead 16 \
#     --kvhead 8 \
#     --headdim 128 \
#     --num-comp-sm 1 \
#     --num-comm-sm 1 \
#     --mode both
# echo "== two-GPU distributed hybrid mega-ring =="
# torchrun --standalone --nproc_per_node=2 mega_ring_test_min_fa3_varlen_hybrid_multi_rank.py \
#     --local-seqlens 1152,2048,1408 \
#     --global-seqlens 1152,4096,1408 \
#     --all-local-seqlens 1152,1408 \
#     --all-cp-local-seqlens 2048,2304 \
#     --all-cp-global-seqlens 4096,4608 \
#     --qhead 16 \
#     --kvhead 8 \
#     --headdim 128 \
#     --num-comp-sm 1 \
#     --num-comm-sm 1 \
#     --cp-threshold 2048 \
#     --mode both

# == two-GPU distributed mega-ring all-CP regression ==
# distributed mega ring varlen case causal=False: ok (world_size=2, B=1, S=256, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=True: ok (world_size=2, B=1, S=256, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=False: ok (world_size=2, B=1, S=512, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring varlen case causal=True: ok (world_size=2, B=1, S=512, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# == two-GPU distributed hybrid mega-ring ==
# distributed mega ring hybrid case causal=False: ok (world_size=2, local_lengths=[1152, 2048, 1408], global_lengths=[1152, 4096, 1408], cp_count=1, threshold=2048, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring hybrid case causal=True: ok (world_size=2, local_lengths=[1152, 2048, 1408], global_lengths=[1152, 4096, 1408], cp_count=1, threshold=2048, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring hybrid case causal=False: ok (world_size=2, local_lengths=[1152, 1408], global_lengths=[1152, 1408], cp_count=0, threshold=2048, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring hybrid case causal=True: ok (world_size=2, local_lengths=[1152, 1408], global_lengths=[1152, 1408], cp_count=0, threshold=2048, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring hybrid case causal=False: ok (world_size=2, local_lengths=[2048, 2304], global_lengths=[4096, 4608], cp_count=2, threshold=2048, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
# distributed mega ring hybrid case causal=True: ok (world_size=2, local_lengths=[2048, 2304], global_lengths=[4096, 4608], cp_count=2, threshold=2048, QH=16, KVH=8, D=128, num_comp_sm=1, num_comm_sm=1)
