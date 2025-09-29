import os
import io
import copy
import json
import base64
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from mllm.model import MLLMModel
from mllm.model.processing import ModelProcessor
from mllm.model.image_processing import ModelImageProcessor
from utils.file_io import read_jsonlines, read_json



class MLLMEvalModel(MLLMModel):
    def chat(
        self,
        image,
        msgs,
        tokenizer,
        processor,
        max_new_tokens=1024,
        max_inp_length=2048,
        system_prompt="",
        sampling=True,
    ):
        prompts, images = self.prepare_chat_inputs(
            tokenizer, system_prompt, [msgs], [image]
        )
        print("prompts:", prompts)
        inputs = processor(
            prompts,
            images,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)
        inputs.pop("image_sizes")

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        with torch.inference_mode():
            out = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                decode_text=True,
                min_new_tokens=3,
                **generation_config,
            )
            print("out:", out)
        print("===================================")
        return {"content": out[0]}

    def prepare_chat_inputs(self, tokenizer, system_prompt, msgs_list, images_list):
        prompts, img_inputs = [], []
        for msgs, img in zip(msgs_list, images_list):
            dialog = [{"role": "system", "content": system_prompt}]
            for m in msgs:
                content = m["content"]
                if m["role"] == "user":
                    if "<image>" in content:                     
                        content = content.replace(
                            "<image>", "<image>./</image>", 1
                        )
                    else:                                        
                        content = "<image>./</image>\n" + content
                dialog.append({"role": m["role"], "content": content})
            prompt = tokenizer.apply_chat_template(
                dialog, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            img_inputs.append(img)
        return prompts, img_inputs


def eval_model(args):
    model = MLLMEvalModel.from_pretrained(        # 修改类名
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    img_processor_config = read_json("mllm/model/mllm_preprocessor_config.json")
    image_processor = ModelImageProcessor(**img_processor_config)
    processor = ModelProcessor(image_processor, tokenizer)

    model.eval().cuda()

    input_data = read_json(args.question_file)

    ans_file = open(args.answers_file, "w")

    with torch.inference_mode():
        i = 0
        accu = 0
        for item in tqdm(input_data):
            image = item["image"]
            msgs = [item["conversations"][0]]

            if len(image) > 1000:
                image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")

            answer = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=args.sampling,
                processor=processor,
            )
            msgs = item["conversations"]
            answer_dict = {
                "idx": i,
                "question": msgs[0]["content"],
                "answer": answer["content"],
                "target": msgs[1]["content"],
                "eval": answer["content"].strip() == msgs[1]["content"].strip(),
                "model": args.model_name_or_path,
                "metainfos": {
                    key: value
                    for key, value in item.items()
                    if key not in ["image", "conversations"]
                },
            }

            if msgs[1]["content"].strip() in answer["content"].strip():
                accu += 1

            if "image_id" in item.keys():
                answer_dict["image_id"] = item["image_id"]

            ans_file.write(json.dumps(answer_dict) + "\n")
            ans_file.flush()

            i += 1
        print(f"Accuracy: {accu/i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args()

    eval_model(args)
