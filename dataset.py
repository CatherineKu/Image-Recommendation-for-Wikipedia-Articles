import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer

class WikipediaDataset(Dataset):
    def __init__(
        self,
        data,                              # pandas DataFrame with columns including "folder_name" & "combined_text"
        tokenizer,                         # HuggingFace tokenizer
        max_length,                        # max token length
        transforms=None,                   # unused here, kept for compatibility
        image_feature_mapping_path=None,   # path to your mapped_image_features.json
        split='train',                     # "train", "trainval", or "test"
        include_images=True                # whether to load image features
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
                # Expecting { folder_name: { img_filename: { "features": [...] }, … }, … }
                self.img_feat_map = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # allow integer or tensor index
        if torch.is_tensor(index):
            index = index.tolist()

        # --- IMAGE FEATURES ---
        if self.include_images:
            passage_id = self.data.at[index, "folder_name"]
            feats_dict = self.img_feat_map.get(passage_id, {})
            if not feats_dict:
                raise KeyError(f"No image features for passage '{passage_id}'")
            # pick the first image’s features
            img_key = next(iter(feats_dict.keys()))
            img_feat = feats_dict[img_key]["features"]
            img = torch.tensor(img_feat, dtype=torch.float32)
        else:
            img = None

        # --- CAPTION (use combined_text) ---
        caption = self.data.at[index, "combined_text"]
        cap_inputs = self.tokenizer.encode_plus(
            caption,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        caption_ids  = cap_inputs["input_ids"].squeeze(0)      # shape: (max_len,)
        caption_mask = cap_inputs["attention_mask"].squeeze(0)

        # --- URL (also use combined_text as a stand-in) ---
        url_text = self.data.at[index, "combined_text"]
        url_inputs = self.tokenizer.encode_plus(
            url_text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        url_ids   = url_inputs["input_ids"].squeeze(0)
        url_mask  = url_inputs["attention_mask"].squeeze(0)

        return img, url_ids, url_mask, caption_ids, caption_mask


def collate_fn_without_nones(batch):
    # filter out any None samples
    batch = [b for b in batch if b is not None]
    # if every img is None, strip that column away
    if all(item[0] is None for item in batch):
        # remove the None image column
        batch = [ (u, um, c, cm) for (_, u, um, c, cm) in batch ]
        return [None] + default_collate(batch)
    # else collate normally (img tensor included)
    return default_collate(batch)


def collate_fn_without_nones(batch):
    batch = [d for d in batch if d is not None]
    # if images are all None, strip them off
    if all(b[0] is None for b in batch):
        batch = [d[1:] for d in batch]
        return [None] + default_collate(batch)
    return default_collate(batch)
