import yaml, torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import clip
import evaluation, utils
from dataset import WikipediaDataset, collate_fn_without_nones
from TS_CPM import MatchingModel

def main():
    cfg = yaml.safe_load(open('/content/drive/MyDrive/CLIP_train/config/MCProp_imgs_text.yaml'))
    ckpt = torch.load('runs/test/model_best_fold0.pt', map_location='cpu')
    model = MatchingModel(cfg)
    model.load_state_dict(ckpt['model'], strict=True)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()


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

    q_feats, c_feats, _ = evaluation.encode_data(model, val_loader)
    metrics = evaluation.compute_recall(q_feats, c_feats)
    print("=== Recall on CLEAN validation ===")
    print(f"Recall@1:  {metrics['r1']:.2f}%")
    print(f"Recall@5:  {metrics['r5']:.2f}%")
    print(f"Recall@10: {metrics['r10']:.2f}%")
    print(f"MedianR:  {metrics['medr']}")
    print(f"MeanR:    {metrics['meanr']:.2f}")
    

if __name__=="__main__":
    main()
