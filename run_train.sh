#!/bin/bash
#set -x
source model_conf
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export GLOG_v=1
export CUDA_VISIBLE_DEVICES=0

python ./test.py