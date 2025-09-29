import os, re
import json
import pickle
import random
import torch
import logging
import pandas as pd
import os.path as op
import transformers
from torch.utils.data import Dataset
import math
import copy

from PIL import Image
from typing import Dict
from utils.file_io import read_json, bytes_to_PIL_image
from mllm.train.preprocess import preprocess
from mllm.train.inference_logp import get_dataset_inference_logp
from mllm.train.preprocess import find_best_resize

logger = logging.getLogger(__name__)


class GroundingSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning Visual Grounding."""

    GRID = 1000    

    def __init__(
        self,
        raw_data,
        transform,
        tokenizer,
        slice_config,
        patch_size=14,
        query_nums=64,
        batch_vision=False,
        max_length=2048,
    ):
        super(GroundingSupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.patch_size = patch_size
        self.query_nums = query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ### ==> TODO: Visual Grounding数据处理流程
        item = self.raw_data[i]
        image_path = item["image"]
        conversations = item["conversations"]

        try:
            images_dict = {"<image>": Image.open(image_path).convert("RGB")}
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            raise e

        data_dict = preprocess(
            images_dict=images_dict,
            conversations=conversations,
            tokenizer=self.tokenizer,
            transform=self.transform,
            query_nums=self.query_nums,
            slice_config=self.slice_config,
            patch_size=self.patch_size,
            batch_vision=self.batch_vision,
            max_length=self.max_length,
        )
        # print("data_dict[input_ids]:", data_dict["input_ids"])
        # print("data_dict[target]:", data_dict["target"])
        return dict(
            input_ids=data_dict["input_ids"],
            position_ids=data_dict["position_ids"],
            labels=data_dict["target"],
            attention_mask=torch.ones_like(data_dict["input_ids"], dtype=torch.bool),
            pixel_values=data_dict.get("pixel_values", None),
            tgt_sizes=data_dict.get("tgt_sizes", None),
            image_bound=data_dict["image_bound"],
        )
        ### <===
