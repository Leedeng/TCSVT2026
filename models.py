import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from transformers import AutoModel

from config import CFG


class VideoEncoder(nn.Module):
    def __init__(self, output_layer="avgpool", trainable=CFG.trainable):
        super().__init__()
        self.weights = R2Plus1D_18_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        pretrained = r2plus1d_18(weights=self.weights if CFG.pretrained else None)

        layers = list(pretrained._modules.keys())
        layer_count = 0
        for name in layers:
            if name != output_layer:
                layer_count += 1
            else:
                break
        for i in range(1, len(layers) - layer_count):
            pretrained._modules.pop(layers[-i])

        self.net = nn.Sequential(*pretrained._modules.values())
        for p in self.net.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.net(self.preprocess(x))


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
        # Mean pooling (standard for sentence-transformers)
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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


class VideoCLIPModel(nn.Module):
    def __init__(self, num_classes=None, temperature=CFG.temperature):
        super().__init__()
        self.image_encoder = VideoEncoder(output_layer="avgpool")
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=CFG.image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=CFG.text_embedding)
        self.temperature = temperature

        # Classification head (used in end-to-end training)
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
        features = self.image_encoder(clip)
        return self.image_projection(features.squeeze(-1).squeeze(-1).squeeze(-1))

    def encode_text(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_projection(features)

    def forward(self, clip, input_ids, attention_mask):
        image_embeddings = self.encode_image(clip)
        text_embeddings = self.encode_text(input_ids, attention_mask)

        cls_logits = None
        if self.classifier is not None:
            cls_logits = self.classifier(image_embeddings)

        return image_embeddings, text_embeddings, cls_logits
