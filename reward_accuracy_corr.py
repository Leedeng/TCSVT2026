"""Per-class reward-accuracy correlation (rebuttal R1-2).

For each class we pair:
  - reward_c : the frozen judge's alignment reward R1 for that class's GRPO
               description(s) (how well the description matches the class's
               visual centroid), and
  - acc_c    : the final MCL model's per-class recall on the test set.
A positive correlation across classes indicates the reward tracks visual
recognizability class-by-class, i.e. it is a meaningful signal rather than an
arbitrary bias of the judge.

Usage:
  python reward_accuracy_corr.py --dataset iMiGUE \
      --reward_model  /.../ckpt/iMiGUE/desc_v2/0.66_iMiGUE_desc_v2.pt \
      --final_model   /.../ckpt/iMiGUE/grpo_desc/0.65_iMiGUE_grpo_desc.pt \
      --desc_file     descriptions/iMiGUE_grpo_descriptions.json \
      --output        reward_acc_iMiGUE.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG
from dataset import get_dataloader
from models import VideoCLIPModel
from grpo_train import pre_encode_videos, compute_mean_class_embs, compute_R1


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return pearson(rx, ry)


def per_class_reward(judge, labels, descriptions, mean_image_embs, device):
    """Mean alignment reward R1 per class over that class's descriptions."""
    judge.eval()
    tok = CFG.tokenizer
    rewards = np.full(len(labels), np.nan)
    with torch.no_grad():
        for c, name in enumerate(labels):
            descs = descriptions.get(name, [])
            if not descs:
                continue
            r1s = []
            for d in descs:
                enc = tok(d, return_tensors="pt", padding=True, truncation=True,
                          max_length=CFG.max_length).to(device)
                t = judge.encode_text(input_ids=enc["input_ids"],
                                      attention_mask=enc["attention_mask"])
                t = F.normalize(t, dim=-1)
                r1s.append(compute_R1(t, mean_image_embs.to(device), c))
            rewards[c] = float(np.mean(r1s))
    return rewards


def per_class_accuracy(model, test_loader, num_classes, device):
    """Per-class top-1 recall of the classifier head on the test set."""
    model.eval()
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test eval"):
            _, _, cls_logits = model(
                clip=batch["clip"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            pred = cls_logits.argmax(dim=-1).cpu().numpy()
            tgt = batch["label"].argmax(dim=-1).cpu().numpy()
            for p, t in zip(pred, tgt):
                total[t] += 1
                correct[t] += (p == t)
    acc = np.divide(correct, total, out=np.full(num_classes, np.nan), where=total > 0)
    return acc * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reward_model", required=True, help="Frozen judge checkpoint (.pt)")
    ap.add_argument("--final_model", required=True, help="Final recognition checkpoint (.pt)")
    ap.add_argument("--desc_file", default=None, help="GRPO descriptions JSON")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = args.dataset.rstrip("/")

    labels = list(pd.read_csv(f"{ds}/Clip_label.csv")["name"].values)
    num_classes = len(labels)
    desc_path = args.desc_file or f"descriptions/{ds}_grpo_descriptions.json"
    descriptions = json.load(open(desc_path))
    print(f"{ds}: {num_classes} classes, descriptions from {desc_path}")

    # --- per-class reward from the frozen judge ---
    print("Loading judge (reward model)...")
    judge = VideoCLIPModel(num_classes=num_classes).to(device)
    judge.load_state_dict(torch.load(args.reward_model, map_location=device))
    judge.eval()
    train_loader = get_dataloader(ds, mode="training", label_names=labels)
    image_embs, label_idx, _ = pre_encode_videos(judge, train_loader, device)
    mean_image_embs = compute_mean_class_embs(image_embs, label_idx, num_classes)
    rewards = per_class_reward(judge, labels, descriptions, mean_image_embs, device)
    del judge
    torch.cuda.empty_cache()

    # --- per-class accuracy from the final model ---
    print("Loading final recognition model...")
    final = VideoCLIPModel(num_classes=num_classes).to(device)
    final.load_state_dict(torch.load(args.final_model, map_location=device))
    test_loader = get_dataloader(ds, mode="testing", label_names=labels)
    accs = per_class_accuracy(final, test_loader, num_classes, device)

    # --- correlation ---
    mask = ~np.isnan(rewards) & ~np.isnan(accs)
    r_p = pearson(rewards[mask], accs[mask])
    r_s = spearman(rewards[mask], accs[mask])
    print(f"\n=== {ds}: reward-accuracy correlation over {mask.sum()} classes ===")
    print(f"Pearson r  = {r_p:.3f}")
    print(f"Spearman rho = {r_s:.3f}")

    df = pd.DataFrame({"class": labels, "reward_R1": rewards, "acc@1": accs})
    df = df.sort_values("reward_R1", ascending=False)
    out = args.output or f"reward_acc_{ds}.csv"
    df.to_csv(out, index=False)
    print(f"Saved per-class table to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
