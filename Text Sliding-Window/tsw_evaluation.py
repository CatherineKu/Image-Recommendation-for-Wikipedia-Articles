import time
import numpy as np
import torch
import tqdm


def encode_data(model, data_loader, log_step=100, logging=print):
    """Encode all images and captions loadable by `data_loader`."""

    model.eval()

    img_embs = None
    cap_embs = None
    alphas_list = []

    pbar = tqdm.tqdm(data_loader)
    pbar.set_description('Encoding inference data')

    device = next(model.parameters()).device

    for i, data in enumerate(pbar):
        data = [d.to(device) if isinstance(d, torch.Tensor) else d for d in data]

        with torch.no_grad():
            img_emb, cap_emb, alphas = model.compute_embeddings(*data)

            if img_embs is None:
                img_embs = img_emb.cpu()
                cap_embs = cap_emb.cpu()
            else:
                img_embs = torch.cat([img_embs, img_emb.cpu()], dim=0)
                cap_embs = torch.cat([cap_embs, cap_emb.cpu()], dim=0)

            if alphas is not None:
                alphas_list.append(alphas)

        if (i + 1) % log_step == 0:
            logging(f'Validation: [{i + 1}/{len(data_loader)}]')

    if len(alphas_list) > 0:
        alphas = torch.cat(alphas_list, dim=0)
        alphas = alphas.mean(dim=0)
        alphas = {'img_alpha': alphas[0].item(), 'txt_alpha': alphas[1].item()}
    else:
        alphas = None

    return img_embs, cap_embs, alphas


def compute_recall(queries, captions):

    npts = queries.shape[0]
    ranks = np.zeros(npts)

    pbar = tqdm.trange(npts)
    pbar.set_description('Validation')

    for index in pbar:
        query = queries[index].unsqueeze(0)  # [1, dim]
        d = torch.mm(query, captions.T).cpu().numpy().flatten()
        ordered = np.argsort(d)[::-1]
        ranks[index] = np.where(ordered == index)[0][0]

    # Recall@1/5/10，median rank，mean rank
    r1 = 100.0 * np.sum(ranks < 1) / npts
    r5 = 100.0 * np.sum(ranks < 5) / npts
    r10 = 100.0 * np.sum(ranks < 10) / npts
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return {'r1': r1, 'r5': r5, 'r10': r10, 'medr': medr, 'meanr': meanr}
