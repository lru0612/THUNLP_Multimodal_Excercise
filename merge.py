import torch
from peft import PeftModel
from mllm.model import MLLMModel
from transformers import AutoTokenizer
import os

# --- 1. 定义模型路径 ---
# 基础模型路径 (LoRA微调前的原始模型)
base_model_path = "/home/user-C/THUNLP_Multimodal_Excercise/MLLM_Excercise_Model"
# LoRA 适配器路径 (LoRA微调后生成的 checkpoint)
lora_adapter_path = "/home/user-C/THUNLP_Multimodal_Excercise/output/finetune/mllm_sft_training_batch80_lr4e-5_steps200_uselora_cosine_warmup30"
# 合并后模型的保存路径
output_path = "/home/user-C/THUNLP_Multimodal_Excercise/MLLM_sft_training"

print(f"即将合并模型，请确认路径：")
print(f"  - 基础模型: {base_model_path}")
print(f"  - LoRA适配器: {lora_adapter_path}")
print(f"  - 输出目录: {output_path}")

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# --- 2. 加载基础模型和分词器 ---
print("\n正在加载基础模型和分词器...")

# 使用 device_map={"": 0} 将整个模型直接加载到 GPU 0
# 这避免了 'auto' 模式下的模块分割问题，同时利用了 GPU
model = MLLMModel.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0}
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
print("基础模型加载完成 (已加载到 GPU)。")

# --- 3. 加载并融合 LoRA 适配器 ---
print("\n正在加载 LoRA 适配器...")
# PeftModel 会自动将适配器加载到基础模型所在的设备上
model = PeftModel.from_pretrained(model, lora_adapter_path)
print("LoRA 适配器加载完成。")

print("\n正在合并 LoRA 权重...")
model = model.merge_and_unload()
print("权重合并完成。")

# --- 4. 保存合并后的完整模型 ---
print(f"\n正在将合并后的模型保存到: {output_path}")
# 保存时，模型会自动从 GPU 移回 CPU，无需手动操作
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print("\n🎉 合并后的模型和分词器已成功保存！")
