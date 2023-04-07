#!/bin/bash
#set -x
source model_conf
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export GLOG_v=1
export CUDA_VISIBLE_DEVICES=0

python ./rl.py \
    --data_path ${DATA_PATH} \
    --dataset_name ${DATASET} \
    --llm_dir ${LLM_DIR} \
    --seed ${SEED} \
    --max_length 1024 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --sample_num 4\
    --warmup_ratio 0 \
    --weight_decay 0 \
    --output_dir ./output_rl