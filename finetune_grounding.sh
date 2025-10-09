#!/bin/bash

### ==> TODO: 编写Visual Grounding训练流程脚本
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=6002 

MODEL="output/finetune/mllm_sft_training_batch32_lr2e-6_steps100_cosine"
 
DATA="data/grounding/train_minicpmv_grounding.json" 
EVAL_DATA="data/grounding/val_minicpmv_grounding.json"
MODEL_MAX_Length=2048
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
learning_rate=5e-6
per_device_train_batch_size=16
gradient_accumulation_steps=4
batch_size=512
torchrun $DISTRIBUTED_ARGS mllm/finetune.py  \
    --eval_data_path $EVAL_DATA \
    --do_eval \
    --eval_steps 50 \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision false \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps 1000 \
    --output_dir output/grounding/fintuned_model_lr${learning_rate}_batch${batch_size}_gradnorm1_grounding_lossmodified \
    --logging_dir output/grounding/fintuned_model_lr${learning_rate}_batch${batch_size}_gradnorm1_grounding_lossmodified/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 20 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate $learning_rate \
    --max_grad_norm 1.0 \
    --weight_decay 0.05 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.1\
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_config_zero2_autoscheduler.json \
    --report_to "tensorboard" \
    --task Grounding 

### <===