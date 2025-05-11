import torch
from torch import nn
from transformers import AutoModel
import clip
from torch.nn import functional as F
from loss import ContrastiveLoss


class TextExtractorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        text_model_name = config['text-model']['model-name']
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.finetune = config['text-model'].get('finetune', False)

        self.window_size = config['text-model'].get('window-size', 128)
        self.stride = config['text-model'].get('stride',
                                               self.window_size // 2)

    def forward(self, input_ids, attention_mask):
        # input_ids:      [B, L]
        # attention_mask: [B, L]
        B, L = input_ids.size()
        all_hidden = []  # will be list of lists: [ [layer0_w1,layer1_w1,...], [layer0_w2,...], ... ]

        for start in range(0, L, self.stride):
            end = start + self.window_size
            if end > L:
                break  

            w_ids = input_ids[:, start:end]  # [B, window]
            w_mask = attention_mask[:, start:end]  # [B, window]

            with torch.set_grad_enabled(self.finetune):
                out = self.text_model(
                    input_ids=w_ids,
                    attention_mask=w_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            # out.hidden_states: list[num_layers] of (B, window, D)
            all_hidden.append(out.hidden_states)

        if not all_hidden:
            with torch.set_grad_enabled(self.finetune):
                out = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            hidden_states = torch.stack(out.hidden_states, dim=0)
            return hidden_states

        num_layers = len(all_hidden[0])
        hidden_dim = all_hidden[0][0].size(-1)

        hidden_states_per_layer = []
        for layer in range(num_layers):
            layer_windows = [window[layer] for window in all_hidden]
            layer_cat = torch.cat(layer_windows, dim=1)
            hidden_states_per_layer.append(layer_cat)

        hidden_states = torch.stack(hidden_states_per_layer, dim=0)
        return hidden_states


class TransformerPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=input_dim,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = None

    def forward(self, token_features, attention_mask):
        src_key_padding_mask = (attention_mask == 0)
        token_features = token_features.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(token_features,
                                                      src_key_padding_mask=src_key_padding_mask)
        cls_representation = transformer_output[0]  # shape: [B, input_dim]
        if self.proj is not None:
            cls_representation = self.proj(cls_representation)
        return cls_representation



class ImageExtractorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        img_feat_dim = config['image-model']['dim']
        common_space_dim = config['matching']['common-space-dim']
        self.fc = nn.Sequential(
            nn.Linear(img_feat_dim, img_feat_dim),
            nn.ReLU(),
            nn.Linear(img_feat_dim, common_space_dim)
        )

    def forward(self, img_feature):
        feat = self.fc(img_feature)
        return feat

class DepthAggregatorModel(nn.Module):
    def __init__(self, aggr, input_dim=1024, output_dim=1024):
        super().__init__()
        self.aggr = aggr
        if self.aggr == 'gated':
            self.self_attn = nn.MultiheadAttention(input_dim, num_heads=4, dropout=0.1)
            self.gate_ffn = nn.Linear(input_dim, 1)
        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = None

    def forward(self, x, mask):
        # x: [depth, Batch, SeqLen, dim]
        if self.aggr is None:
            out = x[-1, :, 0, :]  
        elif self.aggr == 'mean':
            out = x[:, :, 0, :].mean(dim=0)  
        elif self.aggr == 'gated':
            mask_bool = mask.clone().bool()
            mask_bool = ~mask_bool  
            mask_bool = mask_bool.unsqueeze(1).expand(-1, x.shape[0], -1)
            mask_bool = mask_bool.reshape(-1, mask_bool.shape[2])
            orig = x
            bs = x.shape[1]
            x = x.view(-1, x.shape[2], x.shape[3]).permute(1, 0, 2)
            sa, _ = self.self_attn(x, x, x, key_padding_mask=mask_bool)
            scores = torch.sigmoid(self.gate_ffn(sa))
            scores = scores.permute(1, 0, 2).view(-1, bs, x.shape[0], 1)
            scores = scores[:, :, 0, :]  
            orig = orig[:, :, 0, :]     
            scores = scores.permute(1, 2, 0) 
            orig = orig.permute(1, 0, 2)        
            out = torch.matmul(scores, orig)    
            out = out.squeeze(1)
        if self.proj is not None:
            out = self.proj(out)
        return out


class FeatureFusionModel(nn.Module):
    def __init__(self, mode, img_feat_dim, txt_feat_dim, common_space_dim):
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            pass
        elif mode == 'weighted':
            self.alphas = nn.Sequential(
                nn.Linear(img_feat_dim + txt_feat_dim, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, 2)
            )
            self.img_proj = nn.Linear(img_feat_dim, common_space_dim)
            self.txt_proj = nn.Linear(txt_feat_dim, common_space_dim)
            self.post_process = nn.Sequential(
                nn.Linear(common_space_dim, common_space_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(common_space_dim, common_space_dim)
            )

    def forward(self, img_feat, txt_feat):
        concat_feat = torch.cat([img_feat, txt_feat], dim=1)
        alphas_raw = self.alphas(concat_feat)

        bias = torch.tensor([0.0, 0.3], device=alphas_raw.device)
        alphas = torch.sigmoid(alphas_raw + bias)

        img_feat_norm = F.normalize(self.img_proj(img_feat), p=2, dim=1)
        txt_feat_norm = F.normalize(self.txt_proj(txt_feat), p=2, dim=1)

        out_feat = img_feat_norm * alphas[:, 0].unsqueeze(1) + txt_feat_norm * alphas[:, 1].unsqueeze(1)
        out_feat = self.post_process(out_feat)

        return out_feat, alphas


class MatchingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        common_space_dim = config['matching']['common-space-dim']
        num_text_transformer_layers = config['matching']['text-transformer-layers']
        img_feat_dim = config['image-model']['dim']
        txt_feat_dim = config['text-model']['dim']
        image_disabled = config['image-model']['disabled']

        self.aggregate_tokens_depth = config['matching'].get('aggregate-tokens-depth', None)
        if self.aggregate_tokens_depth in ['null', 'None', 'none']:
            self.aggregate_tokens_depth = None

        self.fusion_mode = config['matching'].get('fusion-mode', 'concat')
        self.image_disabled = image_disabled

        self.txt_model = TextExtractorModel(config)

        if not image_disabled:
            self.img_model = ImageExtractorModel(config)
            self.image_fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(512, 512)
            )

            if self.fusion_mode == 'concat':
                self.process_after_concat = nn.Sequential(
                    nn.Linear(img_feat_dim + txt_feat_dim, common_space_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(common_space_dim, common_space_dim)
                )
            else:
                self.process_after_concat = FeatureFusionModel(
                    self.fusion_mode, img_feat_dim, txt_feat_dim, common_space_dim)

        self.caption_process = TransformerPooling(
            input_dim=txt_feat_dim,
            output_dim=common_space_dim,
            num_layers=num_text_transformer_layers)

        self.url_process = TransformerPooling(
            input_dim=txt_feat_dim,
            output_dim=txt_feat_dim if not image_disabled else common_space_dim,
            num_layers=num_text_transformer_layers)

        if self.aggregate_tokens_depth is not None:
            self.token_aggregator = DepthAggregatorModel(
                self.aggregate_tokens_depth,
                input_dim=txt_feat_dim,
                output_dim=common_space_dim)

        contrastive_margin = config['training']['margin']
        max_violation = config['training']['max-violation']
        self.matching_loss = ContrastiveLoss(margin=contrastive_margin, max_violation=max_violation)

    def compute_embeddings(self, img, url, url_mask, caption, caption_mask):
        alphas = None
        if torch.cuda.is_available():
            if img is not None:
                img = img.cuda()
            url = url.cuda()
            url_mask = url_mask.cuda()
            caption = caption.cuda()
            caption_mask = caption_mask.cuda()

        url_feats = self.txt_model(url, url_mask)
        url_feats_plus = self.url_process(url_feats[-1], url_mask)
        url_feats = url_feats_plus

        caption_feats = self.txt_model(caption, caption_mask)
        caption_feats_plus = self.caption_process(caption_feats[-1], caption_mask)
        caption_feats = caption_feats_plus

        if not self.image_disabled:
            img_feats = self.img_model(img).float()
            img_feats = self.image_fc(img_feats)
            if self.fusion_mode == 'concat':
                query_feats = torch.cat([img_feats, url_feats], dim=1)
                query_feats = self.process_after_concat(query_feats)
            else:
                query_feats, alphas = self.process_after_concat(img_feats, url_feats)
        else:
            query_feats = url_feats

        query_feats = F.normalize(query_feats, p=2, dim=1)
        caption_feats = F.normalize(caption_feats, p=2, dim=1)

        return query_feats, caption_feats, alphas

    def compute_loss(self, query_feats, caption_feats):
        return self.matching_loss(query_feats, caption_feats)

    def forward(self, img, url, url_mask, caption, caption_mask):
        query_feats, caption_feats, alphas = self.compute_embeddings(
            img, url, url_mask, caption, caption_mask)
        loss = self.compute_loss(query_feats, caption_feats)
        return loss, alphas