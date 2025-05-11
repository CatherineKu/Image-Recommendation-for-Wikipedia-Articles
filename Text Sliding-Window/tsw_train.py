import os
import tqdm
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
import clip
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset import WikipediaDataset, collate_fn_without_nones
import utils
from TS_CPM import MatchingModel
import evaluation
from shutil import copyfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--data_dir', default='data_reformat')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--val_step', default=200, type=int)
    parser.add_argument('--test_step', default=100000000, type=int)
    parser.add_argument('--logger_name', default='runs/test')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--load_model', default='', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--img_cache', type=str, default=None)
    parser.add_argument('--train_subfolder', type=str, default='full')
    parser.add_argument('--max_samples', type=int, default=None)
    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    train_df = utils.create_train_pd(opt.data_dir, subfolder=opt.train_subfolder)

    if opt.cross_validation:
        num_folds = config['dataset']['n-folds']
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        for k, (train_idx, valid_idx) in enumerate(kfold.split(X=train_df, y=train_df['language'])):
            train_df.loc[valid_idx, 'Fold'] = k
        logging.info('Using {} folds'.format(num_folds))
        for fold in tqdm.trange(num_folds):
            train(opt, config, train_df, fold=fold)
    else:
        train_df = train_df.sample(frac=1, random_state=42)
        all_idxs = np.arange(len(train_df))
        val_samples = config['dataset']['val-samples']
        logging.info('Using {} samples for validating'.format(val_samples))
        valid_idx = all_idxs[:val_samples]
        train_idx = all_idxs[val_samples:]
        train_df.loc[valid_idx, 'Fold'] = 0
        train_df.loc[train_idx, 'Fold'] = 1
        train(opt, config, train_df, fold=0)

def train(opt, config, data_df, fold=0):
    epoch_train_losses = []
    epoch_val_losses = []
    patience = 3
    no_improve_epochs = 0
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    experiment_path = tb_logger.get_logdir()
    copyfile(opt.config, os.path.join(experiment_path, 'config.yaml'))

    _, clip_transform = clip.load(config['image-model']['model-name'])
    tokenizer = AutoTokenizer.from_pretrained(config['text-model']['model-name'])

    x_train, x_valid = data_df.query(f"Fold != {fold}"), data_df.query(f"Fold == {fold}")
    train_dataset = WikipediaDataset(x_train, tokenizer, max_length=80, split='trainval', transforms=clip_transform, include_images=not config['image-model']['disabled'], image_feature_mapping_path=opt.img_cache)
    val_dataset = WikipediaDataset(x_valid, tokenizer, max_length=80, split='trainval', transforms=clip_transform, include_images=not config['image-model']['disabled'], image_feature_mapping_path=opt.img_cache)

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, num_workers=opt.workers, collate_fn=collate_fn_without_nones)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['bs'], shuffle=False, num_workers=opt.workers, collate_fn=collate_fn_without_nones)

    model = MatchingModel(config)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    scheduler_name = config['training']['scheduler']
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=config['training']['gamma'], milestones=config['training']['milestones'])
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError('{} scheduler is not available'.format(scheduler_name))

    start_epoch = 0
    if opt.resume or opt.load_model:
        filename = opt.resume if opt.resume else opt.load_model
        if os.path.isfile(filename):
            print(("=> loading checkpoint '{}'".format(filename)))
            checkpoint = torch.load(filename, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available():
                model.cuda()
            if opt.resume:
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                if checkpoint['scheduler'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                print(("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, start_epoch)))
            else:
                print(("=> loaded only model from checkpoint '{}'".format(opt.load_model)))
        else:
            print(("=> no checkpoint found at '{}'".format(opt.resume)))

    model.train()
    best_r5 = 0

    for epoch in tqdm.trange(start_epoch, opt.num_epochs):
        progress_bar = tqdm.tqdm(train_dataloader)
        progress_bar.set_description('Train')
        running_loss = 0.0
        total_loss = 0.0

        for it, data in enumerate(progress_bar):
            global_iteration = epoch * len(train_dataloader) + it
            optimizer.zero_grad()
            loss, alphas = model(*data)
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            if global_iteration % opt.log_step == 0:
                mean_loss = running_loss / opt.log_step
                progress_bar.set_postfix(dict(loss='{:.2}'.format(mean_loss)))
                running_loss = 0

            if alphas is not None:
                alphas = alphas.mean(dim=0)
                alphas = {'img_alpha': alphas[0].item(), 'txt_alpha': alphas[1].item()}
                tb_logger.add_scalars("Training/Alphas", alphas, global_iteration)
            tb_logger.add_scalar("Training/Epoch", epoch, global_iteration)
            tb_logger.add_scalar("Training/Loss", loss.item(), global_iteration)
            tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], global_iteration)

        epoch_loss = total_loss / len(train_dataloader)
        epoch_train_losses.append(epoch_loss)

        metrics, alphas = validate(val_dataloader, model)
        val_loss = 1 - (metrics['r1'] + metrics['r5'] + metrics['r10']) / 300
        epoch_val_losses.append(val_loss)

        for m, v in metrics.items():
            tb_logger.add_scalar("Validation/{}".format(m), v, epoch)
        if alphas is not None:
            tb_logger.add_scalars("Validation/Alphas", alphas, epoch)
        print(metrics)

        if metrics['r5'] > best_r5:
            print('Saving best model...')
            checkpoint = {
                'cfg': config,
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            latest = os.path.join(experiment_path, 'model_best_fold{}.pt'.format(fold))
            torch.save(checkpoint, latest)
            best_r5 = metrics['r5']
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement in r5. Patience: {no_improve_epochs}/{patience}")
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_train_losses, label='Train Loss')
    plt.plot(epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(experiment_path, 'loss_plot.png'))
    print("✅ Saved loss plot at:", os.path.join(experiment_path, 'loss_plot.png'))

def validate(val_dataloader, model):
    model.eval()
    query_feats, caption_feats, alphas = evaluation.encode_data(model, val_dataloader)
    metrics = evaluation.compute_recall(query_feats, caption_feats)
    model.train()
    return metrics, alphas

if __name__ == '__main__':
    main()