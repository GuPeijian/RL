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
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --sample_num 8 \
    --temperature 0.8\
    --warmup_ratio 0.06 \
    --weight_decay 0 \
    --output_dir ./output_rl

done