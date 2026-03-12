import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from transformers import DistilBertModel, DistilBertConfig

from config import CFG
from module_utils import prompt_utils


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
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]  # CLS token


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


class Classifier(nn.Module):
    """Two-layer projection head used as a classifier for finetuning."""
    def __init__(self, embedding_dim, num_classes, dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, num_classes)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(num_classes, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_classes)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return self.fc(x)


class VideoCLIPModel(nn.Module):
    def __init__(self, temperature=CFG.temperature):
        super().__init__()
        self.image_encoder = VideoEncoder(output_layer="avgpool")
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=CFG.image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=CFG.text_embedding)
        self.prompt_generator = prompt_utils.VideoSpecificPrompt(layers=1, embed_dim=256, alpha=0.1)
        self.temperature = temperature

    def encode_image(self, clip):
        features = self.image_encoder(clip)
        return self.image_projection(features.squeeze(-1).squeeze(-1).squeeze(-1))

    def encode_text(self, input_ids, attention_mask):
        features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_projection(features)

    def apply_prompt(self, image_embeddings, text_embeddings):
        img = image_embeddings.unsqueeze(0)
        txt = text_embeddings.unsqueeze(0)
        txt = self.prompt_generator(txt, img)
        return img.squeeze(0), txt.squeeze(0)
