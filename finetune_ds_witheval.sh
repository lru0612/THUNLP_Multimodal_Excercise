# #!/bin/bash

# export PYTHONPATH=$PYTHONPATH:`realpath .`
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# # export NCCL_P2P_DISABLE="1"
# # export NCCL_IB_DISABLE="1"
# GPUS_PER_NODE=4
# NNODES=1
# NODE_RANK=0
# MASTER_ADDR=localhost
# MASTER_PORT=6001

# MODEL="MLLM_Excercise_Model"
# DATA="data/train.json"
# EVAL_DATA="data/test.json"
# MODEL_MAX_Length=2048 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "
# torchrun $DISTRIBUTED_ARGS mllm/finetune.py  \
#     --model_name_or_path $MODEL \
#     --data_path $DATA \
#     --eval_data_path $EVAL_DATA \
#     --remove_unused_columns false \
#     --label_names "labels" \
#     --prediction_loss_only false \
#     --bf16 true \
#     --bf16_full_eval true \
#     --fp16 false \
#     --fp16_full_eval false \
#     --do_train \
#     --do_eval \
#     --tune_vision true \
#     --tune_llm true \
#     --model_max_length $MODEL_MAX_Length \
#     --max_slice_nums 9 \
#     --max_steps 150 \
#     --eval_steps 10 \
#     --output_dir output/finetune/mllm_sft_training_batch72_lr2e-5_steps150_test \
#     --logging_dir output/finetune/mllm_sft_training_batch72_lr2e-5_steps150_test/log \
#     --logging_strategy "steps" \
#     --per_device_train_batch_size 12 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "steps" \
#     --save_strategy "steps" \
#     --save_steps 75 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0.1 \
#     --adam_beta2 0.95 \
#     --warmup_ratio 0.1 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --gradient_checkpointing true \
#     --deepspeed mllm/ds_config_zero2.json \
#     --report_to "tensorboard" \
#     --task LM

export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=1,2,3,4

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="MLLM_Excercise_Model"
DATA="data/train.json"
EVAL_DATA="data/test.json"
MODEL_MAX_Length=2048 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
learning_rate=2e-6
per_device_train_batch_size=8
batch_size=32
steps=100
torchrun $DISTRIBUTED_ARGS mllm/finetune.py  \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps $steps \
    --eval_steps 5 \
    --output_dir output/finetune/mllm_sft_training_batch${batch_size}_lr${learning_rate}_steps${steps}_cosine \
    --logging_dir output/finetune/mllm_sft_training_batch${batch_size}_lr${learning_rate}_steps${steps}_cosine/log \
    --logging_strategy "steps" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 10 \
    --learning_rate $learning_rate \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed mllm/ds_config_zero2_autoscheduler.json \
    --report_to "tensorboard" \
    --task LM