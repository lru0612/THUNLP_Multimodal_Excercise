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

    # 假设第一个框是 Ground Truth，第二个是 Prediction
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

        # 绘制矩形
        draw.rectangle(box, outline=color, width=3)

        # 在框的上方绘制文字标签（如 "Ground Truth"）
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 确定标签背景的位置，如果框太靠上，则把标签放在框的下方
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

    # --- 在图片底部添加完整的指代描述 (expr) ---
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

    # --- 保存最终的图片 ---
    img = img.convert("RGB")
    img.save(save_name)

    ### <===


def eval_model(args):
    model = MLLMModel.from_pretrained(
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

    input_data = read_jsonlines(args.question_file)

    ### TODO: Implement inference loop
    with torch.no_grad():
        correct = total_cnt = 0
        for item in tqdm(input_data):
            image = os.path.join(args.image_dir, item["img_path"])
            expr = item["expression"]
            bbox = item["bbox"]
            prompt = "Where is {} in image? answer in [x0,y0,x1,y1] format.".format(
                expr
            )

            msgs = [{"role": "user", "content": prompt}]

            if len(image) > 1000:
                image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")

            answer = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=args.sampling,
                processor=processor,
            )

            # Calculate acc
            ### ==> TODO: 实现Visual Grounding的结果准确率计算方法
            pattern = re.compile(
                r"\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]"
            )
            match1 = pattern.search(answer)
            coords = match1.groups()
            if len(coords) == 4:
                predict_bbox = [float(c) for c in coords]
            else:
                predict_bbox = (0.0, 0.0, 0.0, 0.0)
            iou, _ = box_iou(predict_bbox, bbox)
            iou = iou.item()
            total_cnt += 1
            if iou >= 0.5:
                correct += 1
            ### <===

            # Visualize VG results
            ### ==> TODO: 实现Visual Grounding结果的可视化
            if args.vis_nums > 0:
                vis_boxes(
                    image,
                    [bbox, predict_bbox],
                    expr,
                    save_name=f"{args.image_dir}/output_{total_cnt}.png",
                )
                args.vis_nums -= 1
            ### <===

    print(f"Evaluating {args.qannotation_file} ...")
    print(f"Precision @ 1: {correct / total_cnt} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--vis-nums", type=int, default=5)
    args = parser.parse_args()

    eval_model(args)
