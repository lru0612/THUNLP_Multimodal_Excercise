#!/bin/bash
set -e # 任何命令失败立即退出

export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=0

model_name_or_paths=(
    "output/grounding/fintuned_model_lr5e-6_batch512_gradnorm1_grounding_lossmodified/checkpoint-800"
    "output/grounding/fintuned_model_lr5e-6_batch512_gradnorm1_grounding_lossmodified"
    "output/grounding/fintuned_model_lr2e-5_batch128_gradnorm1_grounding_lossmodified"
)
question_paths=(
    "data/grounding/grounding_test/refcoco_testA.json"
    "data/grounding/grounding_test/refcoco_testB.json"
    "data/grounding/grounding_test/refcoco_val.json"
    "data/grounding/grounding_test/refcocoA_testA.json"
    "data/grounding/grounding_test/refcocoA_testB.json"
    "data/grounding/grounding_test/refcocoA_val.json"
    "data/grounding/grounding_test/refcocog_test.json"
    "data/grounding/grounding_test/refcocog_val.json"
)

# 将所有输出重定向到日志文件
exec > >(tee -a ground_eval.log)

for model_name_or_path in "${model_name_or_paths[@]}"; do
    echo "========================================="
    echo "Processing model: $model_name_or_path"
    echo "========================================="
    for qpath in "${question_paths[@]}"; do
        
        test_name=$(basename "$qpath" .json)
        
        image_dir="$model_name_or_path/grounding_eval/grounding_images/$test_name/"
        save_path="$model_name_or_path/grounding_eval/$test_name.jsonl"
        
        echo "-----------------------------------------"
        echo "Evaluating on: $test_name"
        echo "Image dir: $image_dir"
        echo "Save path: $save_path"
        
        python ./eval/grounding_eval.py \
            --model-name-or-path "$model_name_or_path" \
            --question-file "$qpath" \
            --image-dir "$image_dir" \
            --save-path "$save_path" \
            --vis-nums 30
    done
done