#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=1
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6004
# model_name_or_path="output/finetune/mllm_sft_training_batch72_lr2e-5_steps150/checkpoint-75"
model_name_or_path=/home/user-C/THUNLP_Multimodal_Excercise/output/finetune/mllm_sft_training_batch80_lr4e-5_steps200_uselora_cosine_warmup30
data_path=data/test.json
# save_path=output/finetune/mllm_sft_training_batch72_lr2e-5_steps150/checkpoint-75/eval/test_accu.jsonl
save_path="${model_name_or_path}/test_accu.jsonl"
python ./eval/finetune_eval.py \
--model-name-or-path $model_name_or_path \
--question-file $data_path \
--answers-file $save_path \
--sampling