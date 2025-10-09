MODEL_NAME=${1:-"/home/user-C/THUNLP_Multimodal_Excercise/output/grounding/fintuned_model_lr2e-7_batch160_gradnorm1_normalize_box_specialtoken/checkpoint-450"}
echo "========================="
echo "MODEL: $MODEL_NAME"
echo "========================="

python eval/chair.py \
--coco_path data/annotations \
--cache data/chair_300.pkl \
--cap_file $MODEL_NAME/objhal_bench_answer_sampling.jsonl \
--save_path $MODEL_NAME/eval-chair-300.json \
--caption_key answer
