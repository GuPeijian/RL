#!/bin/bash
#set -x
source model_conf
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export CUDA_VISIBLE_DEVICES=0

for SEED in "${SEED_SET[@]}"; do
python ./rl.py \
    --data_path ${DATA_PATH} \
    --dataset_name ${DATASET} \
    --llm_dir ${LLM_DIR} \
    --seed ${SEED} \
    --max_length 1024 \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --sample_num 8 \
    --temperature ${temperature}\
    --warmup_ratio 0.06 \
    --weight_decay 0 \
    --output_dir ./output_rl

done