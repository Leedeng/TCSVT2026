"""For each class of a dataset, pick K videos and render:
  - 9 uniformly sampled frames (top row)
  - original GPT-seed description + GRPO-refined description (bottom)
Each (class, video) becomes one PNG for easy browsing.

Usage:
  python make_desc_examples.py --dataset iMiGUE --k_per_class 10 \
      --out_dir experiment/desc_examples
"""
import argparse
import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from matplotlib.gridspec import GridSpec


def sample_frames(video_path, n=9):
    vr = VideoReader(video_path, ctx=cpu(0))
    try:
        total = len(vr)
        if total <= n:
            idx = list(range(total)) + [total - 1] * (n - total)
        else:
            idx = np.linspace(0, total - 1, n, dtype=int).tolist()
        frames = vr.get_batch(idx).asnumpy().copy()
        return frames
    finally:
        # Decord's VideoReader holds a C++ handle; explicit del + gc keeps
        # resident memory bounded when processing hundreds of clips in a loop.
        del vr
        gc.collect()


def render(frames, class_name, orig, grpo, out_path, n_frames=9):
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, n_frames, figure=fig, height_ratios=[3, 1], hspace=0.15, wspace=0.05)
    # Top row: frames
    for i, f in enumerate(frames[:n_frames]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(f)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
    # Bottom: text block spanning all columns
    tax = fig.add_subplot(gs[1, :])
    tax.axis("off")
    tax.text(0.01, 0.85, f"Class: {class_name}",
             fontsize=14, fontweight="bold", va="top")
    tax.text(
        0.01, 0.55,
        f"Original: {orig}",
        fontsize=11, va="top", wrap=True,
    )
    tax.text(
        0.01, 0.2,
        f"GRPO:     {grpo}",
        fontsize=11, va="top", wrap=True, color="tab:blue",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def safe_name(s):
    return s.replace("/", "_").replace(" ", "_").replace(",", "").replace("，", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k_per_class", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=9)
    parser.add_argument("--videos_from", default="training_clips",
                        choices=["training_clips", "testing_clips"])
    parser.add_argument("--out_dir", default="experiment/desc_examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orig_json", default=None,
                        help="Defaults to descriptions/<dataset>_descriptions.json")
    parser.add_argument("--grpo_json", default=None,
                        help="Defaults to descriptions/<dataset>_grpo_descriptions.json")
    args = parser.parse_args()

    ds = args.dataset.rstrip("/")
    orig_path = args.orig_json or f"descriptions/{ds}_descriptions.json"
    grpo_path = args.grpo_json or f"descriptions/{ds}_grpo_descriptions.json"
    orig_desc_raw = json.load(open(orig_path))
    grpo_desc_raw = json.load(open(grpo_path))
    # Some iMiGUE keys have trailing whitespace (e.g. 'Rubbing eyes ') while
    # clip.csv captions are stripped. Normalise by stripping keys on both sides.
    orig_desc = {k.strip(): v for k, v in orig_desc_raw.items()}
    grpo_desc = {k.strip(): v for k, v in grpo_desc_raw.items()}

    clip_dir = f"{ds}/{args.videos_from}/"
    df = pd.read_csv(clip_dir + "clip.csv")
    rng = np.random.default_rng(args.seed)

    out_root = os.path.join(args.out_dir, ds)
    os.makedirs(out_root, exist_ok=True)

    classes = sorted(set(df["caption"].values))
    print(f"dataset={ds}  classes={len(classes)}  clips={len(df)}", flush=True)

    for cls in classes:
        sub = df[df["caption"] == cls].reset_index(drop=True)
        if len(sub) == 0:
            continue
        n = min(args.k_per_class, len(sub))
        picks = rng.choice(len(sub), size=n, replace=False)

        orig_list = orig_desc.get(cls.strip(), [])
        grpo_list = grpo_desc.get(cls.strip(), [])
        orig_text = orig_list[0] if orig_list else "(no original description)"
        grpo_text = grpo_list[0] if grpo_list else "(no grpo description)"

        cls_dir = os.path.join(out_root, safe_name(cls))
        os.makedirs(cls_dir, exist_ok=True)

        for i in picks:
            row = sub.iloc[int(i)]
            video_path = clip_dir + row["clip"]
            try:
                frames = sample_frames(video_path, args.n_frames)
            except Exception as e:
                print(f"  skip {video_path}: {e}", flush=True)
                continue
            out_png = os.path.join(
                cls_dir, os.path.splitext(row["clip"])[0].replace("/", "_") + ".png"
            )
            render(frames, cls, orig_text, grpo_text, out_png, args.n_frames)
            del frames
        gc.collect()

        print(f"  [{cls}] rendered {n} videos -> {cls_dir}", flush=True)


if __name__ == "__main__":
    main()
