"""Extract visual embeddings from a trained VideoCLIPModel checkpoint on a test split.

Saves a single .npz with fields:
  embeds: (N, D) float32
  labels: (N,) int32   # class index
  label_names: list of C strings (index -> class name)

Usage:
  python extract_embeds.py --dataset iMiGUE \
      --ckpt /path/to/0.66_iMiGUE_grpo_v2.pt \
      --out experiment/tsne/iMiGUE_ours.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import VideoTextDataset
from models import VideoCLIPModel


def load_checkpoint(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys; first few: {missing[:3]}", flush=True)
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys; first few: {unexpected[:3]}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)
    args = parser.parse_args()

    device = CFG.device
    dataset = args.dataset.rstrip("/")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    label_df = pd.read_csv(f"{dataset}/Clip_label.csv")
    label_names = list(label_df["name"].values)
    num_classes = len(label_names)
    label_to_idx = {name.strip(): i for i, name in enumerate(label_names)}
    print(f"dataset={dataset}  num_classes={num_classes}", flush=True)

    test_dir = f"{dataset}/testing_clips/"
    clip_df = pd.read_csv(test_dir + "clip.csv")
    ds = VideoTextDataset(
        test_dir, clip_df["clip"].values, clip_df["caption"].values,
        mode="testing", label_names=label_names,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False,
    )

    model = VideoCLIPModel(num_classes=num_classes).to(device)
    load_checkpoint(model, args.ckpt)
    model.eval()

    all_embeds, all_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            clip = batch["clip"].to(device)
            emb = model.encode_image(clip)                     # [B, D]
            emb = torch.nn.functional.normalize(emb, dim=-1)   # unit length for stable t-SNE
            label_idx = batch["label"].argmax(dim=-1).cpu().numpy()
            all_embeds.append(emb.cpu().numpy().astype(np.float32))
            all_labels.append(label_idx.astype(np.int32))
            if (i + 1) % 25 == 0 or i == 0:
                print(f"  batch {i+1}/{len(loader)}", flush=True)

    embeds = np.concatenate(all_embeds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    np.savez(
        args.out,
        embeds=embeds,
        labels=labels,
        label_names=np.array(label_names),
    )
    print(f"Saved {args.out}  embeds={embeds.shape}  labels={labels.shape}", flush=True)


if __name__ == "__main__":
    main()
