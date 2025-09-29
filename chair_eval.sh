MODEL_NAME="/home/user-C/THUNLP_Multimodal_Excercise/output/preference/mllm_preference_training_batch40_lr5e-7_beta0.5"

echo "========================="
echo "MODEL: $MODEL_NAME"
echo "========================="

python eval/chair.py \
--coco_path data/annotations \
--cache data/chair_300.pkl \
--cap_file $MODEL_NAME/objhal_bench_answer_greedy.jsonl \
--save_path $MODEL_NAME/eval-chair-300_answer_greedy.json \
--caption_key answer
