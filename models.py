import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from transformers import AutoModel

from config import CFG


class VideoEncoder(nn.Module):
    def __init__(self, trainable=CFG.trainable):
        super().__init__()
        self.weights = R2Plus1D_18_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        pretrained = r2plus1d_18(weights=self.weights if CFG.pretrained else None)

        self.backbone = nn.Sequential(
            pretrained.stem,
            pretrained.layer1,
            pretrained.layer2,
            pretrained.layer3,
            pretrained.layer4,
        )
        self.avgpool = pretrained.avgpool

        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        x = self.preprocess(x)
        features = self.backbone(x)  # [B, 512, T', H', W']
        tokens = features.flatten(2).permute(0, 2, 1)  # [B, N, 512]
        pooled = self.avgpool(features).flatten(1)  # [B, 512]
        return tokens, pooled


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = output.last_hidden_state  # [B, L, 384]
        # Mean pooling (standard for sentence-transformers)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return token_embeddings, pooled  # [B, L, 384], [B, 384]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)


class TGA(nn.Module):
    """Text-Guided Attention pooling: text determines which visual regions matter.
    No FFN, no residual — just attention-weighted pooling. Output is pure visual."""
    def __init__(self, embed_dim=CFG.projection_dim, num_heads=4, dropout=CFG.dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, text_tokens, visual_tokens):
        """
        Args:
            text_tokens: [B, L, D] — text as Query
            visual_tokens: [B, N, D] — visual as Key/Value
        Returns:
            v_hat: [B, D] — text-guided visual representation
            attn_weights: [B, L, N] — attention map
        """
        _, attn_weights = self.cross_attn(
            query=text_tokens, key=visual_tokens, value=visual_tokens,
            need_weights=True,
        )
        # Relevance per visual token: average attention over text tokens
        relevance = attn_weights.mean(dim=1)  # [B, N]
        # Weighted pooling of visual tokens
        v_hat = (visual_tokens * relevance.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return v_hat, attn_weights


class VideoCLIPModel(nn.Module):
    def __init__(self, num_classes=None, use_tga=False, temperature=CFG.temperature):
        super().__init__()
        self.image_encoder = VideoEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=CFG.image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=CFG.text_embedding)
        self.temperature = temperature
        self.use_tga = use_tga

        if use_tga:
            self.visual_token_proj = nn.Linear(CFG.image_embedding, CFG.projection_dim)
            self.text_token_proj = nn.Linear(CFG.text_embedding, CFG.projection_dim)
            self.tga = TGA(embed_dim=CFG.projection_dim)

        # Classification head
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(CFG.projection_dim, CFG.projection_dim),
                nn.GELU(),
                nn.Dropout(CFG.dropout),
                nn.Linear(CFG.projection_dim, num_classes),
            )
        else:
            self.classifier = None

    def encode_image(self, clip):
        tokens, pooled = self.image_encoder(clip)
        image_embeddings = self.image_projection(pooled)
        return image_embeddings, tokens

    def encode_text(self, input_ids, attention_mask):
        tokens, pooled = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = self.text_projection(pooled)
        return text_embeddings, tokens, pooled

    def forward(self, clip, input_ids, attention_mask):
        image_embeddings, visual_tokens = self.encode_image(clip)
        text_embeddings, text_tokens, text_pooled = self.encode_text(input_ids, attention_mask)

        tga_features = None
        tga_text_anchor = None
        attn_weights = None
        if self.use_tga:
            vt = self.visual_token_proj(visual_tokens)   # [B, N, D]
            tt = self.text_token_proj(text_tokens)        # [B, L, D]
            tga_features, attn_weights = self.tga(tt, vt) # [B, D], [B, L, N]
            tga_text_anchor = self.text_token_proj(text_pooled)  # [B, D]

        # Classifier: fuse image_emb + v_hat when TGA enabled
        cls_logits = None
        if self.classifier is not None:
            if self.use_tga and tga_features is not None:
                cls_logits = self.classifier(image_embeddings + tga_features)
            else:
                cls_logits = self.classifier(image_embeddings)

        return image_embeddings, text_embeddings, cls_logits, tga_features, tga_text_anchor, attn_weights
