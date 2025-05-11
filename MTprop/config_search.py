import os
import tqdm
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
import clip
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
import logging
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray import train
from ray.tune.schedulers import HyperBandScheduler

from dataset import WikipediaDataset, collate_fn_without_nones
import utils
from baseline_model_CLIP import MatchingModel
import evaluation
from shutil import copyfile

def override_yaml_config(yaml_config, tune_config):
    for key, val in tune_config.items():
        keys = key.split(".")
        sub_config = yaml_config
        for k in keys[:-1]:
            sub_config = sub_config[k]
        sub_config[keys[-1]] = val
    return yaml_config

def ray_train(tune_config):
    import gc
    with open("/content/drive/MyDrive/CLIP_train1/config/MCProp_imgs_text.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    config = override_yaml_config(config, tune_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config['text-model']['model-name'])
    _, clip_transform = clip.load(config['image-model']['model-name'])

    image_feature_mapping_path = '/content/drive/MyDrive/CLIP_train1/data_reformat/mapped_image_features.json'
    train_df = utils.create_train_pd('/content/drive/MyDrive/CLIP_train1/data_reformat', subfolder='full')
    train_df = train_df.sample(frac=1, random_state=42)
    val_samples = config['dataset']['val-samples']
    all_idxs = np.arange(len(train_df))
    valid_idx = all_idxs[:val_samples]
    train_idx = all_idxs[val_samples:]
    train_df.loc[valid_idx, 'Fold'] = 0
    train_df.loc[train_idx, 'Fold'] = 1

    x_train, x_valid = train_df.query("Fold != 0"), train_df.query("Fold == 0")

    train_dataset = WikipediaDataset(
        x_train, tokenizer, max_length=80, split='trainval',
        transforms=clip_transform, include_images=not config['image-model']['disabled'],
        image_feature_mapping_path=image_feature_mapping_path
    )

    val_dataset = WikipediaDataset(
        x_valid, tokenizer, max_length=80, split='trainval',
        transforms=clip_transform, include_images=not config['image-model']['disabled'],
        image_feature_mapping_path=image_feature_mapping_path
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, collate_fn=collate_fn_without_nones)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['bs'], shuffle=False, collate_fn=collate_fn_without_nones)

    model = MatchingModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=config['training']['gamma'], milestones=config['training']['milestones'])

    model.train()
    best_r5 = 0
    for epoch in range(3):
        for it, data in enumerate(train_loader):
            data = [d.to(device) if isinstance(d, torch.Tensor) else d for d in data]
            optimizer.zero_grad()
            loss, _ = model(*data)
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

        metrics, _ = validate(val_loader, model)
        train.report({
          "r5": float(metrics.get("r5", 0.0)),
          "mean_loss": float(loss.item())
        })

        if metrics['r5'] > best_r5:
            best_r5 = metrics['r5']

    with torch.no_grad():
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()

def validate(val_loader, model):
    model.eval()
    query_feats, caption_feats, alphas = evaluation.encode_data(model, val_loader, log_step=1)
    metrics = evaluation.compute_recall(query_feats, caption_feats)
    model.train()
    return metrics, alphas

def run_tune():
    search_space = {
        "training.lr": tune.loguniform(1e-5, 1e-3),
        "training.bs": tune.choice([32, 64]),
        "training.margin": tune.uniform(0.1, 0.5),
        "matching.text-transformer-layers": tune.choice([1, 2, 3]),
    }

    tuner = tune.Tuner(
        tune.with_resources(ray_train, resources={"cpu": 2, "gpu": 1}),
        tune_config=tune.TuneConfig(
            scheduler=HyperBandScheduler(metric="r5", mode="max"),
            num_samples=5,
            max_concurrent_trials=1,
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    best_config = results.get_best_result(metric="r5", mode="max").config
    with open("/content/drive/MyDrive/CLIP_train1/best_config.yaml", "w") as f:
        yaml.dump(best_config, f)

    print("Best Result:", best_config)
    print("Saved best config to best_config.yaml")

if __name__ == '__main__':
    run_tune()
