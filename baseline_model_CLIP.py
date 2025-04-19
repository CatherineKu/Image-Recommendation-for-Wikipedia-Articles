import torch
from torch import nn
from transformers import AutoModel
import clip
from torch.nn import functional as F
from loss import ContrastiveLoss

class TextExtractorModel(nn.Module):
    """
    针对你数据集的文本部分：
    输入的是经过 tokenizer 处理后的 token_ids 和 attention mask，
    该模型使用预训练的 Transformer 模型（比如 RoBERTa、BERT 等）来获取文本的隐藏状态。
    这里假设文本输入已经拼接了 title 和 summary 部分。
    """
    def __init__(self, config):
        super().__init__()
        # 从配置中读取预训练模型名称，例如："roberta-base"
        text_model_name = config['text-model']['model-name']
        # 是否 fine-tune 文本模型（True 则反向传播更新预训练模型的参数，否则只提取特征）
        self.finetune = config['text-model'].get('finetune', False)
        # 加载预训练模型，要求返回所有隐藏层状态
        self.text_model = AutoModel.from_pretrained(text_model_name)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: [B, L]，tokenized 的文本输入
        :param attention_mask: [B, L]，1 表示有效 token，0 表示 pad
        :return: 一个 tensor，形状为 [num_layers, B, L, hidden_dim]，包含所有 Transformer 层的隐藏状态
        """
        # 根据 self.finetune 决定是否计算梯度
        with torch.set_grad_enabled(self.finetune):
            outputs = self.text_model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True)
        # 将所有隐藏层状态堆叠在一起，输出形状为 [num_layers, Batch, SeqLen, HiddenDim]
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        return hidden_states


class TransformerPooling(nn.Module):
    """
    利用 Transformer 编码器对 token 级别的特征进行整合，
    并从中选取第一位置（通常对应 CLS token）的输出作为全局文本表示。
    """
    def __init__(self, input_dim, output_dim, num_layers=2, num_heads=4, dropout=0.1):
        """
        :param input_dim: 输入特征维度（通常与预训练模型的 hidden_dim 一致）
        :param output_dim: 输出特征的目标维度（公共空间维度）
        :param num_layers: Transformer 编码器的层数
        :param num_heads: 多头注意力的头数
        :param dropout: dropout 概率
        """
        super().__init__()
        # 构造一个 Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=input_dim,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 如果输入输出维度不一致，添加投影层
        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = None

    def forward(self, token_features, attention_mask):
        """
        :param token_features: [B, L, input_dim]，文本中每个 token 的特征（例如取自最后一层隐藏状态）
        :param attention_mask: [B, L]，1 表示有效 token，0 表示 pad
        :return: [B, output_dim]，整合后的文本全局特征表示
        """
        # Transformer Encoder 要求 key_padding_mask：True 表示需要屏蔽的 pad 部分
        src_key_padding_mask = (attention_mask == 0)
        # 变换维度为 [L, B, input_dim]，以满足 nn.TransformerEncoder 的要求
        token_features = token_features.permute(1, 0, 2)
        # 调用 TransformerEncoder 进行多层编码
        transformer_output = self.transformer_encoder(token_features,
                                                      src_key_padding_mask=src_key_padding_mask)
        # 通常选择第一位置作为整合输出（CLS token）
        cls_representation = transformer_output[0]  # shape: [B, input_dim]
        # 如果输入输出维度不同，通过投影调整为目标维度
        if self.proj is not None:
            cls_representation = self.proj(cls_representation)
        return cls_representation



class ImageExtractorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从配置中获取图像特征的原始维度（例如，1024）和匹配模型的公共空间维度
        img_feat_dim = config['image-model']['dim']
        common_space_dim = config['matching']['common-space-dim']
        # 构建一个简单的全连接网络，实现非线性变换并调整到公共空间维度
        self.fc = nn.Sequential(
            nn.Linear(img_feat_dim, img_feat_dim),
            nn.ReLU(),
            nn.Linear(img_feat_dim, common_space_dim)
        )

    def forward(self, img_feature):
        # 这里的 img_feature 是已经加载的预计算图像特征
        feat = self.fc(img_feature)
        return feat

#########################################
# Depth Aggregator 模块
#########################################
class DepthAggregatorModel(nn.Module):
    def __init__(self, aggr, input_dim=1024, output_dim=1024):
        """
        使用不同的方式融合文本模型中每一层 CLS token 表示。
        支持：None（仅取最后一层）、mean（平均）、gated（门控融合）。
        """
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
            out = x[-1, :, 0, :]  # 取最后一层 CLS
        elif self.aggr == 'mean':
            out = x[:, :, 0, :].mean(dim=0)  # 所有层的 CLS 求平均
        elif self.aggr == 'gated':
            mask_bool = mask.clone().bool()
            mask_bool = ~mask_bool  # Transformer 要求 True 表示 padding
            mask_bool = mask_bool.unsqueeze(1).expand(-1, x.shape[0], -1)
            mask_bool = mask_bool.reshape(-1, mask_bool.shape[2])
            orig = x
            bs = x.shape[1]
            x = x.view(-1, x.shape[2], x.shape[3]).permute(1, 0, 2)
            sa, _ = self.self_attn(x, x, x, key_padding_mask=mask_bool)
            scores = torch.sigmoid(self.gate_ffn(sa))
            scores = scores.permute(1, 0, 2).view(-1, bs, x.shape[0], 1)
            scores = scores[:, :, 0, :]  # 取 CLS 对应分数
            orig = orig[:, :, 0, :]      # 取 CLS 表示
            scores = scores.permute(1, 2, 0)  # [Batch, 1, depth]
            orig = orig.permute(1, 0, 2)        # [Batch, depth, dim]
            out = torch.matmul(scores, orig)    # 加权求和: [Batch, 1, dim]
            out = out.squeeze(1)
        if self.proj is not None:
            out = self.proj(out)
        return out

#########################################
# Feature Fusion 模块
#########################################
class FeatureFusionModel(nn.Module):
    def __init__(self, mode, img_feat_dim, txt_feat_dim, common_space_dim):
        """
        这里实现一种加权融合方法，将图像特征与文本特征分别投影到公共空间后，
        根据一个前馈网络产生的权重进行加权求和，再经过后处理。
        """
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            # 若使用拼接方式，暂未实现
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
        alphas = torch.sigmoid(self.alphas(concat_feat))  # 输出权重: [B, 2]
        img_feat_norm = F.normalize(self.img_proj(img_feat), p=2, dim=1)
        txt_feat_norm = F.normalize(self.txt_proj(txt_feat), p=2, dim=1)
        out_feat = img_feat_norm * alphas[:, 0].unsqueeze(1) + txt_feat_norm * alphas[:, 1].unsqueeze(1)
        out_feat = self.post_process(out_feat)
        return out_feat, alphas

#########################################
# Matching Model 主模块
#########################################
class MatchingModel(nn.Module):
    def __init__(self, config):
        """
        使用文本模块、图像模块、文本整合（TransformerPooling）和特征融合设计多模态匹配模型。
        注意：这里的 ImageExtractorModel 已经修改为直接接收预提取的图像特征。
        """
        super().__init__()
        common_space_dim = config['matching']['common-space-dim']
        num_text_transformer_layers = config['matching']['text-transformer-layers']
        img_feat_dim = config['image-model']['dim']
        txt_feat_dim = config['text-model']['dim']
        image_disabled = config['image-model']['disabled']
        self.aggregate_tokens_depth = config['matching'].get('aggregate-tokens-depth', None)
        self.fusion_mode = config['matching'].get('fusion-mode', 'concat')
        self.image_disabled = image_disabled

        # 文本特征提取器（预训练模型 + 输出所有层隐藏状态）
        self.txt_model = TextExtractorModel(config)
        if not image_disabled:
            # 这里调用我们的 ImageExtractorModel（直接处理预提取的图像特征）
            self.img_model = ImageExtractorModel(config)
            # 可选：再添加一个图像全连接层进行后续处理
            self.image_fc = nn.Sequential(
                nn.Linear(img_feat_dim, img_feat_dim),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(img_feat_dim, img_feat_dim)
            )
            if self.fusion_mode == 'concat':
                self.process_after_concat = nn.Sequential(
                    nn.Linear(img_feat_dim + txt_feat_dim, common_space_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(common_space_dim, common_space_dim)
                )
            else:
                self.process_after_concat = FeatureFusionModel(self.fusion_mode, img_feat_dim, txt_feat_dim, common_space_dim)
        # 文本部分后续整合
        self.caption_process = TransformerPooling(input_dim=txt_feat_dim,
                                                   output_dim=common_space_dim,
                                                   num_layers=num_text_transformer_layers)
        # 用于 URL（或其他文本）的处理，可适当调整
        self.url_process = TransformerPooling(input_dim=txt_feat_dim,
                                               output_dim=txt_feat_dim if not image_disabled else common_space_dim,
                                               num_layers=num_text_transformer_layers)
        if self.aggregate_tokens_depth is not None:
            self.token_aggregator = DepthAggregatorModel(self.aggregate_tokens_depth,
                                                           input_dim=txt_feat_dim,
                                                           output_dim=common_space_dim)

        # 对比损失设置
        contrastive_margin = config['training']['margin']
        max_violation = config['training']['max-violation']
        self.matching_loss = ContrastiveLoss(margin=contrastive_margin, max_violation=max_violation)

    def compute_embeddings(self, img, url, url_mask, caption, caption_mask):
        """
        对于输入的图像（预提取的特征）、URL（例如 passage 中提取的部分文本）和 caption，
        分别进行文本编码、Transformer 整合、以及图像特征处理，然后融合到同一公共空间。
        """
        alphas = None
        if torch.cuda.is_available():
            if img is not None:
                img = img.cuda()
            url = url.cuda()
            url_mask = url_mask.cuda()
            caption = caption.cuda()
            caption_mask = caption_mask.cuda()

        # 文本部分：先通过文本特征提取器获得所有层隐藏状态
        url_feats = self.txt_model(url, url_mask)
        # 利用 TransformerPooling 处理最后一层特征
        url_feats_plus = self.url_process(url_feats[-1], url_mask)
        if self.aggregate_tokens_depth:
            # 深度融合：将多个层信息融合
            url_feats_depth_aggregated = self.token_aggregator(url_feats, url_mask)
            url_feats = url_feats_plus + url_feats_depth_aggregated
        else:
            url_feats = url_feats_plus

        # 同理，处理 caption 部分
        caption_feats = self.txt_model(caption, caption_mask)
        caption_feats_plus = self.caption_process(caption_feats[-1], caption_mask)
        if self.aggregate_tokens_depth:
            caption_feats_depth_aggregated = self.token_aggregator(caption_feats, caption_mask)
            caption_feats = caption_feats_plus + caption_feats_depth_aggregated
        else:
            caption_feats = caption_feats_plus

        # 图像部分：这里的 img 是预提取的图像特征向量
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

        # L2 归一化，有助于余弦相似度计算
        query_feats = F.normalize(query_feats, p=2, dim=1)
        caption_feats = F.normalize(caption_feats, p=2, dim=1)

        return query_feats, caption_feats, alphas

    def compute_loss(self, query_feats, caption_feats):
        loss = self.matching_loss(query_feats, caption_feats)
        return loss

    def forward(self, img, url, url_mask, caption, caption_mask):
        # 通过 compute_embeddings 得到 query 和 caption 特征，再计算匹配损失
        query_feats, caption_feats, alphas = self.compute_embeddings(img, url, url_mask, caption, caption_mask)
        loss = self.compute_loss(query_feats, caption_feats)
        return loss, alphas
