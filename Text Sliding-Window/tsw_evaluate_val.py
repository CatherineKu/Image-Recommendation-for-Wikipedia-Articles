import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    matplotlib.use('Agg')

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import clip
import evaluation, utils
from dataset import WikipediaDataset, collate_fn_without_nones
from TS_CPM import MatchingModel


def main():
    # 1) Load config and model checkpoint
    cfg = yaml.safe_load(open('/content/drive/MyDrive/CLIP_train/config/MCProp_imgs_text.yaml'))
    ckpt = torch.load('/content/drive/MyDrive/CLIP_train1/runs/test/model_best_fold0.pt', map_location='cpu')
    model = MatchingModel(cfg)
    model.load_state_dict(ckpt['model'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # 2) Load validation dataset
    df = utils.create_train_pd('/content/drive/MyDrive/CLIP_train/data_reformat', subfolder='clean')
    tokenizer = AutoTokenizer.from_pretrained(cfg['text-model']['model-name'])
    _, clip_tfm = clip.load(cfg['image-model']['model-name'])

    val_ds = WikipediaDataset(
        df, tokenizer, max_length=80,
        split='trainval',
        transforms=clip_tfm,
        include_images=not cfg['image-model']['disabled'],
        image_feature_mapping_path='/content/drive/MyDrive/CLIP_train/data_reformat/mapped_image_features.json'
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['bs'],
        shuffle=False,
        collate_fn=collate_fn_without_nones
    )

    # 3) Extract alphas (fusion weights)
    all_alphas = []

    with torch.no_grad():
        for batch in val_loader:
            if batch[0] is None:
                _, url_ids, url_mask, caption_ids, caption_mask = batch
                image = None
            else:
                image, url_ids, url_mask, caption_ids, caption_mask = batch[:5]

            loss, alphas = model(image, url_ids, url_mask, caption_ids, caption_mask)
            all_alphas.append(alphas.cpu())

    all_alphas = torch.cat(all_alphas, dim=0).numpy()
    alpha_img = all_alphas[:, 0]
    alpha_txt = all_alphas[:, 1]

    plt.figure(figsize=(8, 6))
    plt.hist(alpha_img, bins=40, alpha=0.6, label="Image α", color='steelblue')
    plt.hist(alpha_txt, bins=40, alpha=0.6, label="Text α", color='darkorange')
    plt.xlabel("Alpha Weight Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Fusion Weights (α)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("alphas_distribution.png")
    plt.show()

    print(f"Average Image Weight: {alpha_img.mean():.4f}")
    print(f"Average Text Weight: {alpha_txt.mean():.4f}")

    # 5) Compute recall
    q_feats, c_feats, _ = evaluation.encode_data(model, val_loader)
    metrics = evaluation.compute_recall(q_feats, c_feats)
    print("=== Recall on CLEAN validation ===")
    print(f"Recall@1:  {metrics['r1']:.2f}%")
    print(f"Recall@5:  {metrics['r5']:.2f}%")
    print(f"Recall@10: {metrics['r10']:.2f}%")
    print(f"MedianR:  {metrics['medr']}")
    print(f"MeanR:    {metrics['meanr']:.2f}")


if __name__ == "__main__":
    main()