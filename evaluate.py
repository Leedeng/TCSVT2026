"""Evaluate a saved checkpoint on test set.

Usage:
  python evaluate.py --dataset MA52 --checkpoint 0.6_MA52_desc_v2.pt
"""
import argparse
import os
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from config import CFG
from dataset import get_dataloader
from models import VideoCLIPModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset.rstrip("/")
    device = CFG.device

    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    num_classes = len(labels)

    model = VideoCLIPModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    test_loader = get_dataloader(dataset_name, mode="testing", label_names=labels)

    all_preds = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for batch in test_loader:
            image_emb, _, cls_logits = model(
                clip=batch["clip"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_logits.append(cls_logits.cpu().numpy())
            all_preds.extend(cls_logits.argmax(dim=-1).cpu().numpy())
            all_targets.extend(batch["label"].argmax(dim=-1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.concatenate(all_logits, axis=0)

    acc_1 = (all_preds == all_targets).mean() * 100
    top5 = np.argsort(all_logits, axis=-1)[:, -5:]
    acc_5 = np.mean([all_targets[i] in top5[i] for i in range(len(all_targets))]) * 100
    f1_macro = f1_score(all_targets, all_preds, average="macro") * 100
    f1_micro = f1_score(all_targets, all_preds, average="micro") * 100
    f1_weighted = f1_score(all_targets, all_preds, average="weighted") * 100

    print(f"{args.checkpoint}\tacc@1={acc_1:.2f}%\tacc@5={acc_5:.2f}%\tF1_macro={f1_macro:.2f}%\tF1_micro={f1_micro:.2f}%\tF1_weighted={f1_weighted:.2f}%")


if __name__ == "__main__":
    main()
