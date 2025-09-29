import io
import os
import re
import json
import base64
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import textwrap

import torch
from torchvision.ops.boxes import box_area
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

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

GRID = 1000

def denorm_box(box, w, h, grid=GRID):
    """Map box from 0-grid range back to image size."""
    x0, y0, x1, y1 = box
    return [
            x0 / grid * w,
            y0 / grid * h,
            x1 / grid * w,
            y1 / grid * h,
        ]


def vis_boxes(img, boxes, expr, save_name="output.png"):
    ### ==> TODO: 可视化Visual Grounding结果，包括给定图像、针对图像中对象的描述和对应对象的坐标框

    img = img.copy()
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    box_info = [
        {"label": "Ground Truth", "color": "green"},
        {"label": "Prediction", "color": "red"},
    ]

    if boxes and not isinstance(boxes[0], list):
        boxes = [boxes]

    for i, box in enumerate(boxes):
        info = (
            box_info[i]
            if i < len(box_info)
            else {"label": f"Box {i+1}", "color": "yellow"}
        )
        color = info["color"]
        label = info["label"]

        draw.rectangle(box, outline=color, width=3)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        label_y_pos = box[1] - text_height - 2
        if label_y_pos < 0:
            label_y_pos = box[3] + 2

        label_bg_box = [
            box[0],
            label_y_pos,
            box[0] + text_width + 4,
            label_y_pos + text_height,
        ]
        draw.rectangle(label_bg_box, fill=color)
        draw.text((box[0] + 2, label_y_pos), label, fill="white", font=font)

    wrapped_expr = textwrap.wrap(
        f"Expression: {expr}", width=int(img.width / (font.size * 0.65))
    )

    if wrapped_expr:
        line_height = (
            draw.textbbox((0, 0), wrapped_expr[0], font=font)[3]
            - draw.textbbox((0, 0), wrapped_expr[0], font=font)[1]
        ) + 4
        text_block_height = line_height * len(wrapped_expr)

        # 将文字背景放在图片底部
        text_bg_y0 = img.height - text_block_height - 10

        # 创建一个半透明的黑色背景
        text_bg = Image.new("RGBA", (img.width, text_block_height + 10), (0, 0, 0, 150))
        img.paste(text_bg, (0, text_bg_y0), text_bg)

        y_offset = text_bg_y0 + 5
        for line in wrapped_expr:
            draw.text((5, y_offset), line, font=font, fill="white")
            y_offset += line_height

    img = img.convert("RGB")
    img.save(save_name)

    ### <===


def eval_model(args):
    model = MLLMEvalModel.from_pretrained(
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

    ### TODO: Implement inference loop
    with torch.no_grad():
        correct = total_cnt = 0
        for item in tqdm(input_data):
            # image = os.path.join(args.image_dir, item["img_path"])
            image = item["image"]
            conversations = item["conversations"]

            # --- 解析 GT bbox -------------------------------------------------
            content = conversations[1]["content"]
            # 更新正则表达式以匹配 <box>[(x1, y1), (x2, y2)]</box> 格式
            m = re.search(r"<box>\s*\[\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)\s*,\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)\s*\]\s*</box>", content)
            assert m, f"bbox not found in: {content}"
            # m.groups() 会返回 (x1, y1, x2, y2) 四个字符串
            bbox = [float(x) for x in m.groups()]
            # -------------------------------------------------------------------

            msgs = [conversations[0]]
            if len(image) > 1000:
                image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")
            w, h = image.size

            answer = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=args.sampling,
                processor=processor,
            )

            # Calculate acc
            ### ==> TODO: 实现Visual Grounding的结果准确率计算方法
            # 更新正则表达式以匹配 <box>[(x1, y1), (x2, y2)]</box> 格式
            pattern = re.compile(
                r"<box>\s*\[\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)\s*,\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)\s*\]\s*</box>"
            )
            match = pattern.search(answer["content"])

            # ② 备选：四个数字由空格 / 逗号分隔 (这个备选逻辑可能不再需要，但暂时保留以防万一)
            if match is None:
                pattern_alt = re.compile(
                    r"([\d\.]+)[, ]+([\d\.]+)[, ]+([\d\.]+)[, ]+([\d\.]+)"
                )
                match = pattern_alt.search(answer["content"])

            if match:
                # m.groups() 会返回 (x1, y1, x2, y2) 四个字符串
                coords = match.groups()
                predict_bbox = [float(c) for c in coords]
            else:
                predict_bbox = [0.0, 0.0, 0.0, 0.0]
            denorm_gt_bbox = denorm_box(bbox, w, h)
            denorm_pred_bbox = denorm_box(predict_bbox, w, h)

            # --- 将 list → Tensor[N,4] ---
            pred_tensor = torch.tensor([denorm_pred_bbox], dtype=torch.float32)
            gt_tensor   = torch.tensor([denorm_gt_bbox],         dtype=torch.float32)

            iou = box_iou(pred_tensor, gt_tensor)[0].item()
            print(f"IOU: {iou}")
            # -----------------------------

            total_cnt += 1
            if iou >= 0.5:
                correct += 1
            ### <===

            # Visualize VG results
            ### ==> TODO: 实现Visual Grounding结果的可视化
            os.makedirs(args.image_dir, exist_ok=True)
            if args.vis_nums > 0:
                vis_boxes(
                    image,
                    [denorm_gt_bbox, denorm_pred_bbox],
                    conversations[0]["content"],
                    save_name=f"{args.image_dir}/output_{total_cnt}.png",
                )
                args.vis_nums -= 1
            ### <===

    print(f"Evaluating {args.question_file} ...")
    print(f"Precision @ 1: {correct / total_cnt} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--vis-nums", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
