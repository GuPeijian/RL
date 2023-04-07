#!/bin/bash
#encoding=utf8

source ./model_conf
echo 'Start Task:'${JOB_NAME}
echo ${job_tags}
echo ${job_remark}
echo ${QUEUE_NAME}

conf_path=./config.ini

#paddlecloud job --ak ${ak} --sk ${sk} \
paddlecloud job \
    train \
    --job-name $JOB_NAME \
    --job-conf ${conf_path} \
    --group-name $QUEUE_NAME \
    --job-version "paddle-v2.4.1" \
    --file-dir ${file_dir} \
    --k8s-trainers ${k8s_trainers} \
    --k8s-gpu-cards ${k8s_gpu_cards} \
    --k8s-priority ${k8s_priority} \
    --algo-id ${algo_id} \
    --is-standalone 1 \
    --wall-time ${walltime:-"00:00:00"} \
    --start-cmd "bash ./job.sh" 
    # --is-auto-over-sell ${not_wait} \
    
    # --image-addr ${image_addr} \