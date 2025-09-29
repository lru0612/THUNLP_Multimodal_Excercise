#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=3

model_name_or_path=/home/user-C/THUNLP_Multimodal_Excercise/output/grounding/fintuned_model_lr5e-5_batch160_gradnorm1_normalize_box/checkpoint-800

question_paths=(
    "data/grounding_test/refcoco_testA.json"
    "data/grounding_test/refcoco_testB.json"
    "data/grounding_test/refcoco_val.json"
    "data/grounding_test/refcocoA_testA.json"
    "data/grounding_test/refcocoA_testB.json"
    "data/grounding_test/refcocoA_val.json"
    "data/grounding_test/refcocog_test.json"
    "data/grounding_test/refcocog_val.json"
)

save_dir=$model_name_or_path/eval
mkdir -p "$save_dir"

for qpath in "${question_paths[@]}"; do

    echo "Evaluating $qpath"
    python ./eval/grounding_eval.py \
        --model-name-or-path "$model_name_or_path" \
        --question-file "$qpath" \
        --image-dir $model_name_or_path/grounding_images \
        --vis-nums 300
done 
