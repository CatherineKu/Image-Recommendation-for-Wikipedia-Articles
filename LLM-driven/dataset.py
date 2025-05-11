import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer
import pandas as pd

class WikipediaDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length,
        transforms=None,
        image_feature_mapping_path=None,
        split='train',
        include_images=True
    ):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.transforms = transforms
        self.split = split
        self.include_images = include_images

        if self.include_images:
            assert image_feature_mapping_path is not None, "Must pass image_feature_mapping_path"
            with open(image_feature_mapping_path, 'r', encoding='utf-8') as f:
                nested_feats = json.load(f)
                self.img_feat_map = {}
                for category_dict in nested_feats.values():
                    for image_id, entry in category_dict.items():
                        if isinstance(entry, dict) and "features" in entry:
                            self.img_feat_map[image_id] = entry["features"]
      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()


        if self.include_images:
            image_id = self.data.at[index, "image_id"]
            img_feat = self.img_feat_map.get(image_id, None)
            if img_feat is None:
                raise KeyError(f"No image features for image '{image_id}'")
            img = torch.tensor(img_feat, dtype=torch.float32)
        else:
            img = None


        raw_text = self.data.at[index, "caption_title_and_reference_description"]
        if pd.isnull(raw_text):
            raw_text = ""
        raw_text = str(raw_text).replace("[SEP]", "</s>")

        caption_inputs = self.tokenizer.encode_plus(
            raw_text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        caption_ids = caption_inputs["input_ids"].squeeze(0)
        caption_mask = caption_inputs["attention_mask"].squeeze(0)


        url_inputs = self.tokenizer.encode_plus(
            raw_text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        url_ids = url_inputs["input_ids"].squeeze(0)
        url_mask = url_inputs["attention_mask"].squeeze(0)

        return img, url_ids, url_mask, caption_ids, caption_mask


def collate_fn_without_nones(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    if all(item[0] is None for item in batch):
        batch = [ (u, um, c, cm) for (_, u, um, c, cm) in batch ]
        return [None] + default_collate(batch)
    return default_collate(batch)
