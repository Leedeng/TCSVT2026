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
from grpo_train import pre_encode_videos, compute_mean_class_embs, compute_R3


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return pearson(rx, ry)


def per_class_components(judge, labels, texts_per_class, mean_image_embs, class_embs, device):
    """Per-class reward components, averaged over each class's text(s):
      R_cls  : cosine(text, class centroid)          -- class-level visual alignment
      R_intra: mean cosine(text, each member video)  -- intra-class visual grounding
      R_tok  : rule-based token quality (length/diversity) -- textual, non-visual
    `texts_per_class` maps stripped class name -> list of texts."""
    judge.eval()
    tok = CFG.tokenizer
    cen = mean_image_embs.to(device)
    C = len(labels)
    r_cls = np.full(C, np.nan)
    r_intra = np.full(C, np.nan)
    r_tok = np.full(C, np.nan)
    with torch.no_grad():
        for c, name in enumerate(labels):
            texts = texts_per_class.get(name.strip(), [])
            if not texts:
                continue
            members = class_embs.get(c)  # [M, D] normalized, or None
            cls_s, intra_s, tok_s = [], [], []
            for txt in texts:
                enc = tok(txt, return_tensors="pt", padding=True, truncation=True,
                          max_length=CFG.max_length).to(device)
                e = judge.encode_text(input_ids=enc["input_ids"],
                                      attention_mask=enc["attention_mask"])
                e = F.normalize(e, dim=-1)
                cls_s.append(F.cosine_similarity(e, cen[c:c + 1], dim=-1).item())
                if members is not None and len(members):
                    intra_s.append(F.cosine_similarity(e, members.to(device), dim=-1).mean().item())
                tok_s.append(compute_R3(txt, tok))
            r_cls[c] = float(np.mean(cls_s))
            if intra_s:
                r_intra[c] = float(np.mean(intra_s))
            r_tok[c] = float(np.mean(tok_s))
    return r_cls, r_intra, r_tok


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
    ap.add_argument("--baseline_model", default=None,
                    help="Short-label baseline checkpoint (.pt). If given, also correlate the "
                         "reward with the per-class accuracy GAIN (final - baseline), which "
                         "controls for intrinsic class difficulty.")
    ap.add_argument("--desc_file", default=None, help="GRPO descriptions JSON")
    ap.add_argument("--label_file", default=None,
                    help="Class label CSV (default: {dataset}/Clip_label.csv). Use the label "
                         "order that the final/judge models were trained with.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = args.dataset.rstrip("/")

    labels = list(pd.read_csv(args.label_file or f"{ds}/Clip_label.csv")["name"].values)
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
    # group member video embeddings by class (for the intra-class reward)
    class_embs = {}
    for i in range(len(image_embs)):
        class_embs.setdefault(label_idx[i].item(), []).append(image_embs[i])
    for c in class_embs:
        class_embs[c] = torch.stack(class_embs[c])
    # per-class reward components for the CroRR descriptions and for the short label
    desc_map = {k.strip(): v for k, v in descriptions.items()}
    label_map = {name.strip(): [name] for name in labels}
    R_cls, R_intra, R_tok = per_class_components(judge, labels, desc_map, mean_image_embs, class_embs, device)
    label_reward, _, _ = per_class_components(judge, labels, label_map, mean_image_embs, class_embs, device)
    del judge
    torch.cuda.empty_cache()

    # --- per-class accuracy from the final model ---
    print("Loading final recognition model...")
    final = VideoCLIPModel(num_classes=num_classes).to(device)
    final.load_state_dict(torch.load(args.final_model, map_location=device))
    test_loader = get_dataloader(ds, mode="testing", label_names=labels)
    accs = per_class_accuracy(final, test_loader, num_classes, device)
    del final
    torch.cuda.empty_cache()

    # --- optional: per-class accuracy of the short-label baseline (for the gain) ---
    base_accs = None
    if args.baseline_model:
        print(f"Loading short-label baseline: {args.baseline_model}")
        base = VideoCLIPModel(num_classes=num_classes).to(device)
        base.load_state_dict(torch.load(args.baseline_model, map_location=device))
        base_accs = per_class_accuracy(base, test_loader, num_classes, device)
        del base
        torch.cuda.empty_cache()

    # --- correlation ---
    def report(name, x, y):
        m = ~np.isnan(x) & ~np.isnan(y)
        print(f"\n=== {ds}: {name} over {m.sum()} classes ===")
        print(f"Pearson r    = {pearson(x[m], y[m]):.3f}")
        print(f"Spearman rho = {spearman(x[m], y[m]):.3f}")

    cols = {"class": labels,
            "R_cls": R_cls, "R_intra": R_intra, "R_tok": R_tok,
            "reward_label": label_reward, "method_acc@1": accs}
    if base_accs is not None:
        cols["baseline_acc@1"] = base_accs
        d_acc = accs - base_accs
        # the correct form: each reward component vs. per-class accuracy GAIN
        report("R_cls   vs. delta-acc", R_cls, d_acc)
        report("R_intra vs. delta-acc", R_intra, d_acc)
        report("R_tok   vs. delta-acc", R_tok, d_acc)

    df = pd.DataFrame(cols)
    df = df.sort_values("R_cls", ascending=False)
    out = args.output or f"reward_acc_{ds}.csv"
    df.to_csv(out, index=False)
    print(f"Saved per-class table to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
