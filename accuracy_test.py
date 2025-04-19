from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import clip
import yaml
import torch

import utils
from dataset import WikipediaDataset, collate_fn_without_nones
from baseline_model_CLIP import MatchingModel
import evaluation

# 1) 载入训练好的 model + config
cfg = yaml.safe_load(open('config/baseline.yaml'))
model = MatchingModel(cfg)
ckpt = torch.load('runs/test/model_best_fold0.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

# 2) 构造 test DataLoader（只前10条示例的话 df.head(10)）
test_df = utils.create_test_pd('data_reformat', subfolder='original').head(10).reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained(cfg['text-model']['model-name'])
_, clip_tfm = clip.load(cfg['image-model']['model-name'])
test_ds = WikipediaDataset(
    test_df, tokenizer, max_length=80,
    transforms=clip_tfm,
    include_images=not cfg['image-model']['disabled'],
    image_feature_mapping_path='data_reformat/mapped_image_features.json'
)
loader = DataLoader(test_ds, batch_size=cfg['training']['bs'],
                    shuffle=False, collate_fn=collate_fn_without_nones)

# 3) 编码 & 计算 recall
with torch.no_grad():
    q_feats, c_feats, _ = evaluation.encode_data(model, loader)
metrics = evaluation.compute_recall(q_feats, c_feats)

print("Recall@1 (Top‑1 命中率) =", metrics['r1'], "%")
print("Recall@5 =", metrics['r5'], "%")
print("Recall@10 =", metrics['r10'], "%")
