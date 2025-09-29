#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL=MLLM_sft_training
DATA_DIR="data/preference/MLLM_sft_training/logps"
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
    --save_steps 200 \
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