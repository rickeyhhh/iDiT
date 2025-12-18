#!/bin/bash
set -x
# ========== Configurable Parameters ==========
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ib7s
export NCCL_DEBUG=WARN
echo nproc_per_node=$nproc_per_node
echo nnodes=$nnodes
echo node_rank=$node_rank
echo master_addr=$master_addr
echo master_port=$master_port

DEVICE_LIST="0,1,2,3,4,5,6,7"
num_processes=$((nnodes * nproc_per_node))
echo num_processes=${num_processes}

# ========== Set Environment Variables ==========

export CUDA_VISIBLE_DEVICES=${DEVICE_LIST}
source /root/.bashrc
conda activate idit

# ========== Launch Training ==========
torchrun \
  --nproc_per_node ${nproc_per_node} \
  --nnodes ${nnodes} \
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
  ./train.py \
  --quant_method complex_phase_v1 \
  --model DiT-S/2 \
  --data-path /root/datasets/imagenet1k \
  


