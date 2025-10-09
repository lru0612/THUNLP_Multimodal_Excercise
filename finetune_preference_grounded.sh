#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=$((RANDOM % 10000 + 20000))
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
MODEL=/home/user-C/THUNLP_Multimodal_Excercise/output/grounding/fintuned_model_lr5e-6_batch512_gradnorm1_grounding_lossmodified
DATA_DIR="data/preference/Grounded_Model/logps"
DATA="data/preference_train.json"
REF_NAME=Finetuned_Model

MODEL_MAX_Length=2048
per_batch_size=5
accu_gra=4
batch_size=80
lr=2e-5
beta=0.3
deepspeed  mllm/finetune.py \
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
    --num_train_epochs 9 \
    --output_dir output/preference/Grounded_mllm_preference_training_batch${batch_size}_lr${lr}_beta${beta} \
    --logging_dir output/preference/Grounded_mllm_preference_training_batch${batch_size}_lr${lr}_beta${beta}/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size $per_batch_size \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps $accu_gra \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate $lr \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_pref_config_zero2.json \
    --report_to "tensorboard" \
    --dataloader_num_workers 16 \
    --preference_use_average_logp True \
    --preference_beta $beta \
    --task Preference