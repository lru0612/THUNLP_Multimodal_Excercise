#!/bin/bash

### ==> TODO: 编写Visual Grounding训练流程脚本
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6002 

MODEL="MLLM_sft_training"
 
DATA="data/train_minicpmv_grounding.json" 
EVAL_DATA="data/val_minicpmv_grounding.json"
MODEL_MAX_Length=2048
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
learning_rate=5e-5
per_device_train_batch_size=12
GLOBAL_BATCH_SIZE=160
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
    --tune_vision true \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --num_train_epochs 1.5 \
    --output_dir output/grounding/fintuned_model_lr${learning_rate}_batch${GLOBAL_BATCH_SIZE}_gradnorm1_normalize_box \
    --logging_dir output/grounding/fintuned_model_lr${learning_rate}_batch${GLOBAL_BATCH_SIZE}_gradnorm1_normalize_box/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 5 \
    --learning_rate $learning_rate \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.15\
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_config_zero2_autoscheduler.json \
    --report_to "tensorboard" \
    --task Grounding 

### <===

#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL=output/grounding/fintuned_model_lr${learning_rate}_batch${GLOBAL_BATCH_SIZE}_gradnorm1_normalize_box
DATA_DIR="data/preference/fintuned_model_lr${learning_rate}_batch${GLOBAL_BATCH_SIZE}_gradnorm1_normalize_boxng/logps"
DATA="data/preference_train.json"
REF_NAME=MLLM_sft_training

MODEL_MAX_Length=2048

deepspeed --master_port 29600 mllm/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --data_dir $DATA_DIR \
    --ref_name $REF_NAME \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision true \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --num_train_epochs 7 \
    --output_dir output/preference/finetuned_mllm_preference_training_batch40_lr5e-7_beta0.1 \
    --logging_dir output/preference/finetuned_mllm_preference_training_batch40_lr5e-7_beta0.1/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 10 \
    --learning_rate 5e-7 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_pref_config_zero2.json \
    --report_to "tensorboard" \
    --dataloader_num_workers 16 \
    --preference_use_average_logp True \
    --preference_beta 0.1 \
    --task Preference