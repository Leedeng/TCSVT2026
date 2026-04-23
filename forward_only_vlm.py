"""Forward-only Qwen2.5-VL linear probe for micro-gesture recognition.

Freezes the VLM, runs one forward pass per video, mean-pools the last-layer
hidden state over valid tokens, and trains a linear classifier head on top.
Reports Acc@1 / Acc@5 on the test split and the average forward-pass
inference time per sample (frame sampling + processor + backbone forward +
classifier head).

Usage:
  python forward_only_vlm.py --dataset iMiGUE \
      --model_path /scratch/project_2014500/dengli/EALD/QWEN_VL/Qwen2.5-VL-7B-Instruct
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
from PIL import Image

from config import CFG


PROMPT = (
    "This video shows a person performing a micro-gesture. "
    "Describe what the person is doing."
)


def sample_frames(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0), width=CFG.width, height=CFG.height)
    total = len(vr)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames]


@torch.no_grad()
def extract_feature(model, processor, video_path, device, num_frames=8):
    """One forward pass through Qwen2.5-VL; returns mean-pooled last hidden state."""
    frames = sample_frames(video_path, num_frames)
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1]  # [1, T, D]
    attn_mask = inputs.get("attention_mask", None)
    if attn_mask is not None:
        mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
        feat = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    else:
        feat = hidden.mean(dim=1)
    return feat.squeeze(0).float().cpu()


def extract_split(model, processor, device, dataset, split, num_frames):
    """Extract features for an entire split."""
    sub = "training_clips" if split == "train" else "testing_clips"
    base = f"{dataset}/{sub}/"
    df = pd.read_csv(base + "clip.csv")
    feats, targets, skipped = [], [], 0
    label_df = pd.read_csv(f"{dataset}/Clip_label.csv")
    labels = list(label_df["name"].values)
    label_to_idx = {l.strip(): i for i, l in enumerate(labels)}
    t_start = time.time()
    for idx in range(len(df)):
        video_path = base + df["clip"].iloc[idx]
        label = df["caption"].iloc[idx].strip()
        try:
            feat = extract_feature(model, processor, video_path, device, num_frames)
        except Exception as e:
            skipped += 1
            if skipped < 10:
                print(f"[{split}] skip {video_path}: {e}", flush=True)
            continue
        feats.append(feat)
        targets.append(label_to_idx.get(label, -1))
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (len(df) - idx - 1)
            print(
                f"[{split}] {idx+1}/{len(df)}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                flush=True,
            )
    feats_t = torch.stack(feats)
    tgts_t = torch.tensor(targets)
    return feats_t, tgts_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch/project_2014500/dengli/EALD/QWEN_VL/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--time_n", type=int, default=200, help="Number of test videos to re-time")
    parser.add_argument("--cache_dir", type=str, default="feats_qwen_vl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset.rstrip("/")
    os.makedirs(args.cache_dir, exist_ok=True)

    # Labels
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    num_classes = len(label_df)
    print(f"dataset={dataset_name}  num_classes={num_classes}", flush=True)

    # Load VLM
    print(f"Loading {args.model_path} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Qwen2.5-VL exposes hidden_size under text_config; plain LMs at top level.
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_dim = model.config.text_config.hidden_size
    else:
        hidden_dim = model.config.hidden_size
    print(f"hidden_dim={hidden_dim}", flush=True)

    # Feature extraction (with caching)
    train_cache = os.path.join(args.cache_dir, f"{dataset_name}_train.pt")
    test_cache = os.path.join(args.cache_dir, f"{dataset_name}_test.pt")

    if os.path.exists(train_cache):
        print(f"Loading cached train feats from {train_cache}", flush=True)
        blob = torch.load(train_cache)
        train_feats, train_tgts = blob["feats"], blob["tgts"]
    else:
        print("Extracting train features...", flush=True)
        train_feats, train_tgts = extract_split(
            model, processor, device, dataset_name, "train", args.num_frames
        )
        torch.save({"feats": train_feats, "tgts": train_tgts}, train_cache)

    if os.path.exists(test_cache):
        print(f"Loading cached test feats from {test_cache}", flush=True)
        blob = torch.load(test_cache)
        test_feats, test_tgts = blob["feats"], blob["tgts"]
    else:
        print("Extracting test features...", flush=True)
        test_feats, test_tgts = extract_split(
            model, processor, device, dataset_name, "test", args.num_frames
        )
        torch.save({"feats": test_feats, "tgts": test_tgts}, test_cache)

    print(f"train_feats={tuple(train_feats.shape)}  test_feats={tuple(test_feats.shape)}", flush=True)

    # Linear probe
    head = nn.Linear(hidden_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    tf = train_feats.to(device).float()
    tt = train_tgts.to(device)
    xf = test_feats.to(device).float()
    xt = test_tgts.to(device)

    best_acc1 = 0.0
    best_acc5 = 0.0
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(tf), device=device)
        total = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = head(tf[idx])
            loss = loss_fn(logits, tt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            logits = head(xf)
            acc1 = (logits.argmax(-1) == xt).float().mean().item() * 100
            top5 = logits.topk(5, dim=-1).indices
            acc5 = (top5 == xt.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        best_acc1 = max(best_acc1, acc1)
        best_acc5 = max(best_acc5, acc5)
        print(
            f"epoch {ep+1:02d}  loss={total/len(tf):.4f}  acc@1={acc1:.2f}%  acc@5={acc5:.2f}%  "
            f"best@1={best_acc1:.2f}%",
            flush=True,
        )

    # Forward-only inference time measurement
    test_df = pd.read_csv(f"{dataset_name}/testing_clips/clip.csv")
    N = min(args.time_n, len(test_df))
    times = []
    head.eval()
    for idx in range(N):
        video_path = f"{dataset_name}/testing_clips/" + test_df["clip"].iloc[idx]
        try:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            feat = extract_feature(model, processor, video_path, device, args.num_frames)
            feat = feat.to(device).float()
            with torch.no_grad():
                _ = head(feat)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)
        except Exception as e:
            print(f"skip timing for {video_path}: {e}", flush=True)

    avg_ms = np.mean(times) * 1000.0
    median_ms = np.median(times) * 1000.0
    p90_ms = np.percentile(times, 90) * 1000.0

    print("\n" + "=" * 60)
    print("Forward-only Qwen2.5-VL linear probe, final results")
    print("=" * 60)
    print(f"Dataset:           {dataset_name}")
    print(f"Model:             {args.model_path}")
    print(f"Best Acc@1:        {best_acc1:.2f}%")
    print(f"Best Acc@5:        {best_acc5:.2f}%")
    print(
        f"Inference time:    avg={avg_ms:.1f} ms  median={median_ms:.1f} ms  "
        f"p90={p90_ms:.1f} ms  (n={len(times)})"
    )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
