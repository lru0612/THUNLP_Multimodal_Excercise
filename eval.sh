#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=3

model_name_or_path="/home/user-C/THUNLP_Multimodal_Excercise/MLLM_Excercise_Model"
data_path=./data/objhal_bench.jsonl
save_path=$model_name_or_path/objhal_bench_answer_greedy.jsonl

python ./eval/model_eval.py \
--model-name-or-path $model_name_or_path \
--question-file $data_path \
--answers-file $save_path 
