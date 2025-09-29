from fileinput import filename
import os
import json
import numpy as np
from numpy import random as nprdm
import random
import tqdm
import multiprocessing
import argparse
import threading
import re

random.seed(71)
nprdm.seed(71)


IMAGE_PLACEHOLDER = "<image>"
BOXES_PLACEHOLDER = "<boxes>"
EXPR_PLACEHOLDER = "<expr>"
OBJS_PLACEHOLDER = "<objs>"
QUESTION_PLACEHOLDER = "<question>"
POINTS_PLACEHOLDER = "<points>"
PHRASE_ST_PLACEHOLDER = "<ph_st>"
PHRASE_ED_PLACEHOLDER = "<ph_ed>"
BOX_ST="<box>"
BOX_ED="</box>"
REF_ST="<ref>"
REF_ED="</ref>"

# ---- Normalise bbox to grid=1000 ----------------------------------
from PIL import Image
GRID = 1000  # target grid size

def norm_box(box, w, h, grid: int = GRID):
    """Map [x0,y0,x1,y1] into 0-grid range (keep 3 decimals)."""
    x0, y0, x1, y1 = box
    return [
        int(x0 / w * grid),
        int(y0 / h * grid),
        int(x1 / w * grid),
        int(y1 / h * grid),
    ]


class RECDataset:
    def __init__(
        self,
        filename="",
        template_file="",
        image_folders="",
        version="",
        total=None,
        ratio=None,
        shuffle=False,
    ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders
        self.version = version

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(
        self,
    ):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Referring Expression Comprehension数据集
        result = []
        if not os.path.exists(self.datafile):
            print(f"Warning: Data file not found at {self.datafile}")
            return result

        with open(self.datafile, "r") as f:
            lines = f.readlines()

        if self.shuffle:
            random.shuffle(lines)

        num_samples_to_process = len(lines)
        if self.total is not None:
            num_samples_to_process = min(self.total, len(lines))
        elif self.ratio is not None:
            num_samples_to_process = int(len(lines) * self.ratio)

        lines_to_process = lines[:num_samples_to_process]

        for i, line in tqdm.tqdm(enumerate(lines_to_process)):
            try:
                data = json.loads(line)
                image_path = data["img_path"]
                expression = data["expression"]
                expression = expression.lower().rstrip(".,!?;:")
                bbox = data["bbox"]

                expression = f"{REF_ST}{expression}{REF_ED}"

                template = self.get_template()
                question = template.replace(EXPR_PLACEHOLDER, expression)

                if self.version == "vg":
                    if "images2" in image_path:
                        img_fp = image_path.replace("images2", self.image_dirs)
                    else:
                        img_fp = image_path.replace("images", self.image_dirs)

                    try:
                        w, h = Image.open(img_fp).size
                    except Exception:
                        w = h = 1
                    nx0, ny0, nx1, ny1 = norm_box(bbox, w, h)
                    bbox_str = f"{BOX_ST}[(<Loc{nx0}>, <Loc{ny0}>), (<Loc{nx1}>, <Loc{ny1}>)]{BOX_ED}"
                    conversations = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"{bbox_str}"},
                    ]
                    sample = {
                        "id": f"REC_VG_{i}",
                        "image": img_fp,
                        "conversations": conversations,
                    }
                else:
                    img_fp = os.path.join(self.image_dirs, image_path)
                    try:
                        w, h = Image.open(img_fp).size
                    except Exception:
                        w = h = 1
                    nx0, ny0, nx1, ny1 = norm_box(bbox, w, h)
                    bbox_str = f"{BOX_ST}[(<Loc{nx0}>, <Loc{ny0}>), (<Loc{nx1}>, <Loc{ny1}>)]{BOX_ED}"
                    conversations = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"{bbox_str}"},
                    ]
                    sample = {
                        "id": f"REC_{i}",
                        "image": img_fp,
                        "conversations": conversations,
                    }
                result.append(sample)

            except (json.JSONDecodeError, KeyError) as e:
                raise e

        if return_dict is not None:
            return_dict[dict_key] = result
        return result
        ### <===


class GCDataset:
    def __init__(
        self,
        filename="",
        template_file="",
        image_folders="",
        total=None,
        ratio=None,
        shuffle=False,
    ):
        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(
        self,
    ):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Grounded Captioning数据集

        result = []
        if not os.path.exists(self.datafile):
            print(f"Warning: Data file not found at {self.datafile}")
            return result

        with open(self.datafile, "r") as f:
            lines = f.readlines()

        if self.shuffle:
            random.shuffle(lines)

        num_samples_to_process = len(lines)
        if self.total is not None:
            num_samples_to_process = min(self.total, len(lines))
        elif self.ratio is not None:
            num_samples_to_process = int(len(lines) * self.ratio)

        lines_to_process = lines[:num_samples_to_process]

        for i, line in tqdm.tqdm(enumerate(lines_to_process)):
            try:
                data = json.loads(line)
                image_path = data["img_path"]
                expression = data["expression"]
                bbox = data["bbox"]

                if "images2" in image_path:
                    img_fp = image_path.replace("images2", self.image_dirs)
                else:
                    img_fp = image_path.replace("images", self.image_dirs)

                try:
                    w, h = Image.open(img_fp).size
                except Exception:
                    w = h = 1

                nx0, ny0, nx1, ny1 = norm_box(bbox, w, h)
                bbox_str = (
                    f"{BOX_ST}[(<Loc{nx0}>, <Loc{ny0}>), (<Loc{nx1}>, <Loc{ny1}>)]{BOX_ED}"
                )
                expression = f"{REF_ST}{expression}{REF_ED}"

                template = self.get_template()
                question = template.replace(OBJS_PLACEHOLDER, bbox_str)

                conversations = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": expression},
                ]
                sample = {
                    "id": f"GC_{i}",
                    "image": img_fp,
                    "conversations": conversations,
                }
                result.append(sample)

            except (json.JSONDecodeError, KeyError) as e:
                raise e

        if return_dict is not None:
            return_dict[dict_key] = result
        return result


class REGDataset:
    def __init__(
        self,
        filename="",
        template_file="",
        image_folders="",
        total=None,
        ratio=None,
        shuffle=False,
    ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(
        self,
    ):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Referring Expression Generation数据集
        result = []
        if not os.path.exists(self.datafile):
            print(f"Warning: Data file not found at {self.datafile}")
            return result

        with open(self.datafile, "r") as f:
            lines = f.readlines()

        if self.shuffle:
            random.shuffle(lines)

        num_samples_to_process = len(lines)
        if self.total is not None:
            num_samples_to_process = min(self.total, len(lines))
        elif self.ratio is not None:
            num_samples_to_process = int(len(lines) * self.ratio)

        lines_to_process = lines[:num_samples_to_process]

        for i, line in tqdm.tqdm(enumerate(lines_to_process)):
            try:
                data = json.loads(line)
                image_path = data["img_path"]
                expression = data["expression"]
                bbox = data["bbox"]
                img_fp = f"{self.image_dirs}/{image_path}"
                try:
                    w, h = Image.open(img_fp).size
                except Exception:
                    w = h = 1
                nx0, ny0, nx1, ny1 = norm_box(bbox, w, h)
                bbox_str = (
                    f"{BOX_ST}[(<Loc{nx0}>, <Loc{ny0}>), (<Loc{nx1}>, <Loc{ny1}>)]{BOX_ED}"
                )
                expression = f"{REF_ST}{expression}{REF_ED}"

                template = self.get_template()
                question = template.replace(OBJS_PLACEHOLDER, bbox_str)

                conversations = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"{expression}"},
                ]
                sample = {
                    "id": f"REG_{i}",
                    "image": img_fp,
                    "conversations": conversations,
                }
                result.append(sample)

            except (json.JSONDecodeError, KeyError) as e:
                raise e

        if return_dict is not None:
            return_dict[dict_key] = result
        ### <===
        return result


class FlickrDataset:
    def __init__(
        self,
        filename="",
        template_file="",
        image_folders="",
        total=None,
        ratio=None,
        shuffle=False,
    ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(
        self,
    ):
        return nprdm.choice(self.templates, 1)[0]

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现Flik30K-entities数据集
        result = []
        if not os.path.exists(self.datafile):
            print(f"Warning: Data file not found at {self.datafile}")
            return result

        with open(self.datafile, "r") as f:
            lines = f.readlines()

        if self.shuffle:
            random.shuffle(lines)

        num_samples_to_process = len(lines)
        if self.total is not None:
            num_samples_to_process = min(self.total, len(lines))
        elif self.ratio is not None:
            num_samples_to_process = int(len(lines) * self.ratio)

        lines_to_process = lines[:num_samples_to_process]

        for i, line in tqdm.tqdm(enumerate(lines_to_process)):
            try:
                data = json.loads(line)
                image_id = data["image_id"]
                sentence = data["sentence"]
                boxes = data["boxes"]
                boxes_seq = data["boxes_seq"]

                parts = re.split(r"(<ph_st>.*?<ph_ed>)", sentence)
                img_fp = f"{self.image_dirs}/{image_id}.jpg"
                try:
                    w, h = Image.open(img_fp).size
                except Exception:
                    print(f"Warning: Image not found at {img_fp}")
                    w = h = 1
                # 检查解析出的短语数量是否与 boxes_seq 匹配
                num_phrases = len([p for p in parts if p.startswith("<ph_st>")])
                if num_phrases != len(boxes_seq):
                    print(
                        f"Skipping line {i+1} due to phrase/box mismatch.phrase_len:{num_phrases},box_seq_len:{len(boxes_seq)}\nparts:{parts}"
                    )
                    continue
                assistant_response = ""
                phrase_idx = 0
                for part in parts:
                    if part.startswith("<ph_st>"):
                        box_indices = boxes_seq[phrase_idx]
                        bbox_strs = []
                        for box_idx in box_indices:
                            nx0, ny0, nx1, ny1 = norm_box(boxes[box_idx], w, h)
                            bbox_str = f"{BOX_ST}[(<Loc{nx0}>, <Loc{ny0}>), (<Loc{nx1}>, <Loc{ny1}>)]{BOX_ED}"
                            bbox_strs.append(bbox_str)

                        part = part.replace(PHRASE_ST_PLACEHOLDER, REF_ST)
                        part = part.replace(PHRASE_ED_PLACEHOLDER, REF_ED + " ".join(bbox_strs))
                        assistant_response += part
                        phrase_idx += 1
                    else:
                        assistant_response += part

                question = self.get_template()

                conversations = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": assistant_response},
                ]

                sample = {
                    "id": f"flickr_{data['id']}",
                    "image": img_fp,
                    "conversations": conversations,
                }
                result.append(sample)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {i+1} due to error: {e}")
        ### <===

        if return_dict is not None:
            return_dict[dict_key] = result

        return result


class GPT4GenDataset:
    def __init__(
        self,
        filename="",
        template_file="",
        image_folders="",
        version="p",
        total=None,
        ratio=None,
        shuffle=False,
    ):

        self.datafile = filename
        self.templates = json.load(open(template_file))
        self.image_dirs = image_folders

        self.version = version
        assert version in ["a", "c", "bc", "RD_bc"]  # answer cot b_cot

        self.total = total
        self.ratio = ratio
        self.shuffle = shuffle

    def get_template(
        self,
    ):
        return nprdm.choice(self.templates, 1)[0]

    def _format_sentence(self, sentence, boxes, boxes_seq,w,h):
        """Helper for notation using <ph_st>...<ph_ed> tags."""
        parts = re.split(r"(<ph_st>.*?<ph_ed>)", sentence)
        num_phrases = len([p for p in parts if p.startswith("<ph_st>")])
        if num_phrases != len(boxes_seq):
            print(
                f"Skipping line due to phrase/box mismatch.phrase_len:{num_phrases},box_seq_len:{len(boxes_seq)}\nparts:{parts}\nboxes_seq:{boxes_seq}\nboxes:{boxes}"
            )
            return None  # Skip if mismatch

        response = ""
        phrase_idx = 0
        for part in parts:
            if part.startswith("<ph_st>"):
                clean_phrase = part.replace(PHRASE_ST_PLACEHOLDER, REF_ST).replace(
                    PHRASE_ED_PLACEHOLDER, REF_ED
                )
                response += clean_phrase

                box_indices = boxes_seq[phrase_idx]

                bbox_strs = []
                for box_idx in box_indices:
                    nx0, ny0, nx1, ny1 = norm_box(boxes[box_idx], w, h)
                    bbox_str = f"{BOX_ST}[(<Loc{nx0}>, <Loc{ny0}>), (<Loc{nx1}>, <Loc{ny1}>)]{BOX_ED}"
                    bbox_strs.append(bbox_str)
                response += " " + " ".join(bbox_strs)
                phrase_idx += 1
            else:
                response += part
        return response

    def _format_RD_sentence(self, sentence, boxes, boxes_seq,w,h):
        """Helper for notation using <ph_ed> as a separator."""
        parts = sentence.split(PHRASE_ED_PLACEHOLDER)
        if len(parts) != len(boxes_seq) + 1:
            return None  # Skip if mismatch

        response = ""
        for i, part in enumerate(parts[:-1]):
            response += part
            box_indices = boxes_seq[i]
            bbox_strs = []
            for box_idx in box_indices:
                nx0, ny0, nx1, ny1 = norm_box(boxes[box_idx], w, h)
                bbox_str = f"{BOX_ST}[(<Loc{nx0}>, <Loc{ny0}>), (<Loc{nx1}>, <Loc{ny1}>)]{BOX_ED}"
                bbox_strs.append(bbox_str)
            response += " " + " ".join(bbox_strs)
        response += parts[-1]
        return response

    def build(self, return_dict=None, dict_key="key"):
        ### ==> TODO: 实现GPT-4生成的数据集
        result = []
        if not os.path.exists(self.datafile):
            print(f"Warning: Data file not found at {self.datafile}")
            return result

        with open(self.datafile, "r") as f:
            lines = f.readlines()

        if self.shuffle:
            random.shuffle(lines)

        num_samples_to_process = len(lines)
        if self.total is not None:
            num_samples_to_process = min(self.total, len(lines))
        elif self.ratio is not None:
            num_samples_to_process = int(len(lines) * self.ratio)

        lines_to_process = lines[:num_samples_to_process]

        for i, line in tqdm.tqdm(enumerate(lines_to_process)):
            try:
                data = json.loads(line)
                image_path = data["img_path"]
                img_fp = f"{self.image_dirs}/{image_path}"
                try:
                    w, h = Image.open(img_fp).size
                except Exception:
                    w = h = 1
                user_content = ""
                assistant_content = ""
                template = self.get_template()
                if self.version == "bc":
                    if "RD_BoxCoT" in self.datafile:
                        user_content = self._format_RD_sentence(
                            data["question"], data["boxes"], data["question_boxes_seq"],w,h
                        )
                        user_content = template.replace(
                            QUESTION_PLACEHOLDER, user_content
                        )
                        assistant_content = self._format_RD_sentence(
                            data["cot_with_ans"],
                            data["boxes"],
                            data["answer_boxes_seq"],
                            w,h
                        )
                    else:
                        user_content = data["question"]
                        user_content = template.replace(
                            QUESTION_PLACEHOLDER, user_content
                        )
                        assistant_content = self._format_sentence(
                            data["cot_with_ans"],
                            data["boxes"],
                            data["answer_boxes_seq"],
                            w,h
                        )

                elif self.version in ["a", "c"]:
                    user_content = data["question"]
                    user_content = template.replace(QUESTION_PLACEHOLDER, user_content)
                    if self.version == "c":
                        assistant_content = self._format_sentence(
                            data["cot_with_ans"],
                            data["boxes"],
                            data["answer_boxes_seq"],
                            w,h
                        )
                    else:  # version == 'a'
                        assistant_content = data["answer"]

                if user_content is None or assistant_content is None:
                    continue

                conversations = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]

                sample = {
                    "id": f"gpt4gen_{self.version}_{i}",
                    "image": img_fp,
                    "conversations": conversations,
                }
                result.append(sample)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {i+1} due to error: {e}")
        ### <===

        if return_dict is not None:
            return_dict[dict_key] = result
        return result


if __name__ == "__main__":
    # GPTGEN_TRAIN_COMMON_CFG = dict(
    #     type="GPT4Gen",
    #     filename=r"data/VG/shikra_data/GPT4GEN_BoxCoT_train.jsonl",
    #     image_folder=r"data/VG/images/flickr30k-images",
    # )

    # DEFAULT_TRAIN_DATASET = dict(
    #     flickr=dict(
    #         type="FlickrDataset",
    #         filename=r"data/VG/shikra_data/CWB_flickr30k_train.jsonl",
    #         image_folder=r"data/VG/images/flickr30k-images",
    #         template_file=r"data/VG/template/flickr30k.json",
    #     ),
    #     rec=dict(
    #         type="RECDataset",
    #         filename=r"data/VG/shikra_data/REC_ref3_train.jsonl",
    #         image_folder=r"data/VG/images/train2014",
    #         template_file=r"data/VG/template/REC.json",
    #     ),
    #     recvg=dict(
    #         type="RECDataset",
    #         filename=r"data/VG/shikra_data/GC_genome196_train.jsonl",
    #         image_folder=r"data/VG/images",
    #         template_file=r"data/VG/template/REC.json",
    #     ),
    #     reg=dict(
    #         type="REGDataset",
    #         filename=r"data/VG/shikra_data/REC_ref3_train.jsonl",
    #         image_folder=r"data/VG/images/train2014/",
    #         template_file=r"data/VG/template/REG.json",
    #     ),
    #     gc=dict(
    #         type="GCDataset",
    #         filename=r"data/VG/shikra_data/GC_genome196_train.jsonl",
    #         image_folder=r"data/VG/images",
    #         template_file=r"data/VG/template/GC.json",
    #     ),
    #     GPT4GEN_QA=dict(
    #         **GPTGEN_TRAIN_COMMON_CFG,
    #         version="a",
    #         template_file=r"data/VG/template/VQA.json",
    #     ),
    #     GPT4GEN_QC=dict(
    #         **GPTGEN_TRAIN_COMMON_CFG,
    #         version="c",
    #         template_file=r"data/VG/template/VQA_CoT.json",
    #     ),
    #     GPT4GEN_QBC=dict(
    #         **GPTGEN_TRAIN_COMMON_CFG,
    #         version="bc",
    #         template_file=r"data/VG/template/VQA_BCoT.json",
    #     ),
    #     GPT4GEN_RD_QBC=dict(
    #         type=GPTGEN_TRAIN_COMMON_CFG["type"],
    #         image_folder=GPTGEN_TRAIN_COMMON_CFG["image_folder"],
    #         filename="data/VG/shikra_data/GPT4GEN_RD_BoxCoT_train.jsonl",
    #         version="bc",
    #         template_file=r"data/VG/template/VQA_BCoT.json",
    #     ),
    # )

    # datasets = [
    #     RECDataset(
    #         filename=DEFAULT_TRAIN_DATASET["recvg"]["filename"],
    #         version="vg",
    #         ratio=1 / 100,
    #         image_folders=DEFAULT_TRAIN_DATASET["recvg"]["image_folder"],
    #         template_file=DEFAULT_TRAIN_DATASET["recvg"]["template_file"],
    #     ),
    #     GCDataset(
    #         filename=DEFAULT_TRAIN_DATASET["gc"]["filename"],
    #         ratio=1 / 100,
    #         template_file=DEFAULT_TRAIN_DATASET["gc"]["template_file"],
    #         image_folders=DEFAULT_TRAIN_DATASET["gc"]["image_folder"],
    #     ),
    #     RECDataset(
    #         filename=DEFAULT_TRAIN_DATASET["rec"]["filename"],
    #         ratio=1 / 20,
    #         image_folders=DEFAULT_TRAIN_DATASET["rec"]["image_folder"],
    #         version="coco",
    #         template_file=DEFAULT_TRAIN_DATASET["rec"]["template_file"],
    #     ),
    #     REGDataset(
    #         filename=DEFAULT_TRAIN_DATASET["reg"]["filename"],
    #         ratio=1 / 20,
    #         template_file=DEFAULT_TRAIN_DATASET["reg"]["template_file"],
    #         image_folders=DEFAULT_TRAIN_DATASET["reg"]["image_folder"],
    #     ),
    #     FlickrDataset(
    #         filename=DEFAULT_TRAIN_DATASET["flickr"]["filename"],
    #         ratio=1 / 20,
    #         template_file=DEFAULT_TRAIN_DATASET["flickr"]["template_file"],
    #         image_folders=DEFAULT_TRAIN_DATASET["flickr"]["image_folder"],
    #     ),
    #     GPT4GenDataset(
    #         filename=DEFAULT_TRAIN_DATASET["GPT4GEN_QA"]["filename"],
    #         version="a",
    #         template_file=DEFAULT_TRAIN_DATASET["GPT4GEN_QA"]["template_file"],
    #         image_folders=DEFAULT_TRAIN_DATASET["GPT4GEN_QA"]["image_folder"],
    #     ),
    #     GPT4GenDataset(
    #         filename=DEFAULT_TRAIN_DATASET["GPT4GEN_QC"]["filename"],
    #         version="c",
    #         template_file=DEFAULT_TRAIN_DATASET["GPT4GEN_QC"]["template_file"],
    #         image_folders=DEFAULT_TRAIN_DATASET["GPT4GEN_QC"]["image_folder"],
    #     ),
    #     GPT4GenDataset(
    #         filename=DEFAULT_TRAIN_DATASET["GPT4GEN_QBC"]["filename"],
    #         version="bc",
    #         template_file=DEFAULT_TRAIN_DATASET["GPT4GEN_QBC"]["template_file"],
    #         image_folders=DEFAULT_TRAIN_DATASET["GPT4GEN_QBC"]["image_folder"],
    #     ),
    #     GPT4GenDataset(
    #         filename=DEFAULT_TRAIN_DATASET["GPT4GEN_RD_QBC"]["filename"],
    #         version="RD_bc",
    #         template_file=DEFAULT_TRAIN_DATASET["GPT4GEN_RD_QBC"]["template_file"],
    #         image_folders=DEFAULT_TRAIN_DATASET["GPT4GEN_RD_QBC"]["image_folder"],
    #     ),
    # ]
    # tot = 0
    # results = []
    # for dataset in datasets:
    #     results.extend(dataset.build())
    # tot = len(results)
    # ### <===
    # import random
    # random.shuffle(results)  
    # # save
    # with open("data/train_minicpmv_grounding.json", "w") as f:
    #     json.dump(results, f, indent=4)
    # print("Total # exmaples: %d" % tot)
    

    REC_TEST_COMMON_CFG = dict(
    type='RECDataset',
    template_file=r'data/VG/template/REC.json',
    image_folder=r'data/VG/images/train2014',
    max_dynamic_size=None,
    )

    DEFAULT_TEST_REC_VARIANT = dict(
        REC_REFCOCOG_UMD_TEST=dict(
            **REC_TEST_COMMON_CFG,
            filename=r'data/VG/shikra_data/REC_refcocog_umd_test.jsonl',
        ),
        REC_REFCOCOA_UNC_TESTA=dict(
            **REC_TEST_COMMON_CFG,
            filename=r'data/VG/shikra_data/REC_refcoco+_unc_testA.jsonl',
        ),
        REC_REFCOCOA_UNC_TESTB=dict(
            **REC_TEST_COMMON_CFG,
            filename=r'data/VG/shikra_data/REC_refcoco+_unc_testB.jsonl',
        ),
        REC_REFCOCO_UNC_TESTA=dict(
            **REC_TEST_COMMON_CFG,
            filename=r'data/VG/shikra_data/REC_refcoco_unc_testA.jsonl',
        ),
        REC_REFCOCO_UNC_TESTB=dict(
            **REC_TEST_COMMON_CFG,
            filename=r'data/VG/shikra_data/REC_refcoco_unc_testB.jsonl',
        ),
        REC_REFCOCOG_UMD_VAL=dict(
            **REC_TEST_COMMON_CFG,
            filename=r'data/VG/shikra_data/REC_refcocog_umd_val.jsonl',
        ),
        REC_REFCOCOA_UNC_VAL=dict(
            **REC_TEST_COMMON_CFG,
            filename=r'data/VG/shikra_data/REC_refcoco+_unc_val.jsonl',
        ),
        REC_REFCOCO_UNC_VAL=dict(
            **REC_TEST_COMMON_CFG,
            filename=r'data/VG/shikra_data/REC_refcoco_unc_val.jsonl',
        ),
    )

    # val_datasets = [
    #     RECDataset(
    #         filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_VAL"]["filename"],
    #         template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_VAL"]["template_file"],
    #         image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_VAL"]["image_folder"],
    #         ratio=1/5,
    #     ),
        
    #     RECDataset(
    #         filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_VAL"]["filename"],
    #         template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_VAL"]["template_file"],
    #         image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_VAL"]["image_folder"],
    #         ratio=1/5,
    #     ),
        
    #     RECDataset(
    #         filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_VAL"]["filename"],
    #         template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_VAL"]["template_file"],
    #         image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_VAL"]["image_folder"],
    #         ratio=1/5,
    #     ),
    # ]
    # # # ### ==> TODO: 实现用于Visual Grounding的指令微调数据集的构建
    # tot = 0
    # results = []
    # for dataset in val_datasets:
    #     results.extend(dataset.build())
    # tot = len(results)
    # ### <===
    # import random
    # random.shuffle(results)  
    # # save
    # with open("data/val_minicpmv_grounding.json", "w") as f:
    #     json.dump(results, f, indent=4)
    # print("Total # exmaples: %d" % tot)

    refcocog_test = RECDataset(
            filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_TEST"]["filename"],
            template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_TEST"]["template_file"],
            image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_TEST"]["image_folder"],
            ratio=1/5,
        ).build()
    with open("data/grounding_test/refcocog_test.json", "w") as f:
        json.dump(refcocog_test, f, indent=4)
    refcocoA_testA = RECDataset(
            filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_TESTA"]["filename"],
            template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_TESTA"]["template_file"],
            image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_TESTA"]["image_folder"],
            ratio=1/5,
        ).build()
    with open("data/grounding_test/refcocoA_testA.json", "w") as f:
        json.dump(refcocoA_testA, f, indent=4)
        
    refcocoA_testB=   RECDataset(
            filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_TESTB"]["filename"],
            template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_TESTB"]["template_file"],
            image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_TESTB"]["image_folder"],
            ratio=1/5,
        ).build()
    with open("data/grounding_test/refcocoA_testB.json", "w") as f:
        json.dump(refcocoA_testB, f, indent=4)
        
    refcoco_testA=   RECDataset(
            filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_TESTA"]["filename"],
            template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_TESTA"]["template_file"],
            image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_TESTA"]["image_folder"],
            ratio=1/5,
        ).build()
    with open("data/grounding_test/refcoco_testA.json", "w") as f:
        json.dump(refcoco_testA, f, indent=4)
        
    refcoco_testB=   RECDataset(
            filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_TESTB"]["filename"],
            template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_TESTB"]["template_file"],
            image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_TESTB"]["image_folder"],
            ratio=1/5,
        ).build()
    with open("data/grounding_test/refcoco_testB.json", "w") as f:
        json.dump(refcoco_testB, f, indent=4)
    refcocog_val=RECDataset(
            filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_VAL"]["filename"],
            template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_VAL"]["template_file"],
            image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOG_UMD_VAL"]["image_folder"],
            ratio=1/5,
        ).build()
    with open("data/grounding_test/refcocog_val.json", "w") as f:
        json.dump(refcocog_val, f, indent=4)
        
    refcocoA_val=RECDataset(
            filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_VAL"]["filename"],
            template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_VAL"]["template_file"],
            image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCOA_UNC_VAL"]["image_folder"],
            ratio=1/5,
        ).build()
    with open("data/grounding_test/refcocoA_val.json", "w") as f:
        json.dump(refcocoA_val, f, indent=4)
        
    refcoco_val=       RECDataset(
            filename=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_VAL"]["filename"],
            template_file=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_VAL"]["template_file"],
            image_folders=DEFAULT_TEST_REC_VARIANT["REC_REFCOCO_UNC_VAL"]["image_folder"],
            ratio=1/5,
        ).build()
    with open("data/grounding_test/refcoco_val.json", "w") as f:
        json.dump(refcoco_val, f, indent=4)
    