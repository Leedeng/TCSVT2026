"""Evaluate a saved checkpoint on test set.

Usage:
  python evaluate.py --dataset MA52 --checkpoint 0.6_MA52_desc_v2.pt
"""
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

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

    # Load labels
    import pandas as pd
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    num_classes = len(labels)

    # Load model
    model = VideoCLIPModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded {args.checkpoint}")

    # Test loader
    test_loader = get_dataloader(dataset_name, mode="testing", label_names=labels)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            image_emb, _, cls_logits = model(
                clip=batch["clip"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            pred = cls_logits.argmax(dim=-1).cpu().numpy()
            target = batch["label"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(pred)
            all_targets.extend(target)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc_1 = (all_preds == all_targets).mean() * 100
    f1_mean = f1_score(all_targets, all_preds, average="macro") * 100

    print(f"\nDataset: {dataset_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"acc@1 = {acc_1:.2f}%")
    print(f"F1_mean = {f1_mean:.2f}%")

    # Per-class F1
    f1_per_class = f1_score(all_targets, all_preds, average=None) * 100
    print(f"\nPer-class F1:")
    for i, label in enumerate(labels):
        print(f"  {label}: {f1_per_class[i]:.2f}%")


if __name__ == "__main__":
    main()
