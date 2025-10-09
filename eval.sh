#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=6

model_name_or_path=${1:-"/home/user-C/THUNLP_Multimodal_Excercise/output/grounding/fintuned_model_lr2e-7_batch160_gradnorm1_normalize_box_specialtoken/checkpoint-450"}
data_path=./data/objhal_bench.jsonl
save_path=$model_name_or_path/objhal_bench_answer_sampling.jsonl

python ./eval/model_eval.py \
--model-name-or-path $model_name_or_path \
--question-file $data_path \
--answers-file $save_path \
--sampling