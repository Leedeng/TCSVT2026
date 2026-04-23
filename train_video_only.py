"""Train a pure video-only baseline: R(2+1)D-18 + classifier head with CE loss.

No text encoder, no contrastive learning. Meant as the "(1) Video-only"
row in the component ablation so the gains from short-label MCL,
SFT-warmed LLM, and GRPO refinement are decomposed cleanly.

Usage:
  python train_video_only.py --dataset iMiGUE --epochs 50 --lr 1e-4 \
      --batch_size 32 --ckpt_dir ckpt_video_only/iMiGUE
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from config import CFG, AvgMeter, get_lr
from dataset import get_dataloader
from models import VideoEncoder, ProjectionHead


class VideoOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.image_encoder = VideoEncoder(output_layer="avgpool")
        self.image_projection = ProjectionHead(embedding_dim=CFG.image_embedding)
        self.classifier = nn.Sequential(
            nn.Linear(CFG.projection_dim, CFG.projection_dim),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.projection_dim, num_classes),
        )

    def encode_image(self, clip):
        features = self.image_encoder(clip)
        return self.image_projection(features.squeeze(-1).squeeze(-1).squeeze(-1))

    def forward(self, clip):
        emb = self.encode_image(clip)
        return self.classifier(emb)


def evaluate(model, loader, device):
    model.eval()
    correct_1, correct_5, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            clip = batch["clip"].to(device)
            label = batch["label"].to(device).argmax(dim=-1)
            logits = model(clip)
            top5 = logits.topk(5, dim=-1).indices
            pred = logits.argmax(dim=-1)
            correct_1 += (pred == label).sum().item()
            correct_5 += (top5 == label.unsqueeze(1)).any(dim=1).sum().item()
            total += label.size(0)
    return correct_1 / total * 100, correct_5 / total * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Directory to save best checkpoint; default = ckpt_video_only/<dataset>")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Tensorboard log dir; default = log/<dataset>_video_only")
    args = parser.parse_args()

    device = CFG.device
    dataset = args.dataset.rstrip("/")
    ckpt_dir = args.ckpt_dir or f"ckpt_video_only/{dataset}"
    log_dir = args.log_dir or f"log/{dataset}_video_only"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader = get_dataloader(
        data_dir=f"{dataset}/training_clips/", mode="training",
        batch_size=args.batch_size, num_workers=CFG.num_workers,
    )
    test_loader = get_dataloader(
        data_dir=f"{dataset}/testing_clips/", mode="testing",
        batch_size=args.batch_size, num_workers=CFG.num_workers,
    )

    # Infer num_classes from a peek at the first batch
    peek = next(iter(train_loader))
    num_classes = peek["label"].shape[-1]
    print(f"dataset={dataset}  num_classes={num_classes}  "
          f"train_batches={len(train_loader)}  test_batches={len(test_loader)}", flush=True)

    model = VideoOnlyModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CFG.factor, patience=CFG.patience,
    )
    scaler = GradScaler("cuda")
    ce = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir)

    best_acc1 = 0.0
    best_acc5 = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        loss_meter = AvgMeter()
        pbar = tqdm(train_loader, desc=f"ep {epoch+1}/{args.epochs}")
        for batch in pbar:
            clip = batch["clip"].to(device)
            label = batch["label"].to(device).argmax(dim=-1)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(clip)
                loss = ce(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), label.size(0))
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", get_lr(optimizer), global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        acc1, acc5 = evaluate(model, test_loader, device)
        writer.add_scalar("eval/acc1", acc1, epoch)
        writer.add_scalar("eval/acc5", acc5, epoch)
        lr_scheduler.step(acc1)

        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
            best_acc5 = acc5
            torch.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 "acc1": acc1, "acc5": acc5},
                os.path.join(ckpt_dir, "best.pt"),
            )

        print(
            f"ep {epoch+1}: train_loss={loss_meter.avg:.4f}  "
            f"acc@1={acc1:.2f}%  acc@5={acc5:.2f}%  "
            f"best@1={best_acc1:.2f}%",
            flush=True,
        )

    print(f"\nFinal best Acc@1={best_acc1:.2f}%  Acc@5={best_acc5:.2f}%", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
