import torch
from transformers import AutoTokenizer


class CFG:
    batch_size = 32
    num_workers = 8
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 10
    factor = 0.8
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_embedding = 512
    text_encoder_model = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding = 384
    max_length = 128

    pretrained = True
    trainable = True
    temperature = 0.05

    frame = 16
    height = 224
    width = 224

    projection_dim = 256
    dropout = 0.1

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
