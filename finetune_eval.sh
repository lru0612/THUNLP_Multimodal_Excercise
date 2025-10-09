#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=4,5,6,7
GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6004
# model_name_or_path="output/finetune/mllm_sft_training_batch72_lr2e-5_steps150/checkpoint-75"
model_name_or_path=${1:-/home/user-C/THUNLP_Multimodal_Excercise/MLLM_sft_training}
data_path=data/test.json
# save_path=output/finetune/mllm_sft_training_batch72_lr2e-5_steps150/checkpoint-75/eval/test_accu.jsonl
save_path="${model_name_or_path}/test_accu.jsonl"
python ./eval/finetune_eval.py \
--model-name-or-path $model_name_or_path \
--question-file $data_path \
--answers-file $save_path \
--sampling