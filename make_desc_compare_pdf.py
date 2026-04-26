"""Render a bilingual (EN+CN) description-comparison PDF per dataset.

For each (class, video) where we have a VLM caption, render one PDF page:
  Top:    9 uniformly sampled frames
  Header: Class (GT) + clip filename
  Body:   Four description sources, each in English + Chinese:
            Original (GPT-4o-mini, class-level)
            GRPO     (refined,      class-level)
            Qwen-VL  (per-video)
            Gemma 4  (per-video)

Usage:
  python make_desc_compare_pdf.py --dataset iMiGUE \
      --out_pdf experiment/desc_compare/iMiGUE_compare.pdf
"""
import argparse
import gc
import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


CN_FONTS = ["PingFang HK", "PingFang SC", "Heiti TC", "STHeiti",
            "Arial Unicode MS", "Songti SC", "Hiragino Sans GB",
            "Droid Sans Fallback", "Noto Sans CJK SC"]
CN_FONT_FILES = [
    "/usr/share/fonts/google-droid/DroidSansFallback.ttf",  # CSC default CJK font
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def setup_cn_font():
    # Try registering known TTF/TTC paths first (works even if fc-cache missed them)
    for path in CN_FONT_FILES:
        if os.path.exists(path):
            try:
                matplotlib.font_manager.fontManager.addfont(path)
            except Exception:
                pass
    for f in CN_FONTS:
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = ["sans-serif"]
            matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"Using CJK font: {f}", flush=True)
            return f
        except Exception:
            continue
    print("WARNING: no CJK font found; CN text may render as boxes")
    return None


def sample_frames(video_path, n=9):
    vr = VideoReader(video_path, ctx=cpu(0))
    try:
        total = len(vr)
        if total <= n:
            idx = list(range(total)) + [total - 1] * (n - total)
        else:
            idx = np.linspace(0, total - 1, n, dtype=int).tolist()
        return vr.get_batch(idx).asnumpy().copy()
    finally:
        del vr
        gc.collect()


def load_inputs(ds, args):
    orig_path = args.orig_json or f"descriptions/{ds}_descriptions.json"
    grpo_path = args.grpo_json or f"descriptions/{ds}_grpo_descriptions.json"
    qwen_path = args.qwen_json or f"captions/{ds}_qwen.json"
    gem_path  = args.gemma_json or f"captions/{ds}_gemma.json"
    zh_path   = args.zh_json or f"experiment/translations/{ds}_zh.json"

    orig = {k.strip(): v for k, v in json.load(open(orig_path)).items()}
    grpo = {k.strip(): v for k, v in json.load(open(grpo_path)).items()}
    qwen = json.load(open(qwen_path))
    gem  = json.load(open(gem_path))
    zh   = json.load(open(zh_path))

    return orig, grpo, qwen, gem, zh


def en_zh(text, zh_map):
    if not text:
        return "(none)", ""
    t = text.strip()
    cn = zh_map.get(t, "")
    return t, cn


def render_page(pdf, frames, header, blocks, n_frames=9):
    """blocks: list of (label, en_text, cn_text)"""
    fig = plt.figure(figsize=(18, 10))
    n_blocks = len(blocks)
    # Top row: frames; bottom: header + blocks
    gs = GridSpec(2, n_frames, figure=fig,
                  height_ratios=[3, 1 + 0.6 * n_blocks],
                  hspace=0.06, wspace=0.04)
    # frames
    for i, f in enumerate(frames[:n_frames]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(f)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
    # text panel
    tax = fig.add_subplot(gs[1, :])
    tax.axis("off")
    tax.set_xlim(0, 1); tax.set_ylim(0, 1)

    # header
    tax.text(0.005, 0.97, header, fontsize=13, fontweight="bold", va="top")

    # description blocks
    n = len(blocks)
    # Each block uses ~ (0.93/n) vertical fraction; reserve 2 lines per block (EN+CN)
    block_h = 0.92 / n
    colors = {"Original": "black", "GRPO": "tab:blue",
              "Qwen-VL": "tab:green", "Gemma 4": "tab:purple"}
    for i, (label, en, cn) in enumerate(blocks):
        y_top = 0.92 - i * block_h
        c = colors.get(label, "black")
        tax.text(0.005, y_top, f"[{label}] EN: {en}",
                 fontsize=10.5, va="top", color=c, wrap=True)
        if cn:
            tax.text(0.005, y_top - 0.45 * block_h, f"               CN: {cn}",
                     fontsize=10.5, va="top", color=c, wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def cls_idx_from_clip(clip, idx2name):
    """iMiGUE clip naming: <id>_<start>_<end>_<classidx>.mp4"""
    m = re.search(r"_(\d+)\.mp4$", clip)
    if not m:
        return None
    return idx2name.get(int(m.group(1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_pdf", required=True)
    parser.add_argument("--n_frames", type=int, default=9)
    parser.add_argument("--videos_from", default="training_clips")
    parser.add_argument("--orig_json", default=None)
    parser.add_argument("--grpo_json", default=None)
    parser.add_argument("--qwen_json", default=None)
    parser.add_argument("--gemma_json", default=None)
    parser.add_argument("--zh_json", default=None)
    args = parser.parse_args()

    setup_cn_font()
    ds = args.dataset.rstrip("/")
    orig, grpo, qwen, gem, zh = load_inputs(ds, args)

    clip_dir = f"{ds}/{args.videos_from}/"
    df = pd.read_csv(clip_dir + "clip.csv")
    # Build clip -> class via clip.csv (caption column has class name)
    clip2cls = dict(zip(df["clip"].astype(str), df["caption"].astype(str)))

    # Iterate every video that has a VLM caption (Qwen or Gemma)
    vlm_clips = sorted(set(qwen.keys()) | set(gem.keys()))
    # Group by class for ordered rendering
    by_cls = {}
    for clip in vlm_clips:
        cls = clip2cls.get(clip, "?").strip()
        by_cls.setdefault(cls, []).append(clip)

    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
    n_total = sum(len(v) for v in by_cls.values())
    print(f"dataset={ds} videos={n_total} classes={len(by_cls)} -> {args.out_pdf}",
          flush=True)

    n_done = 0
    with PdfPages(args.out_pdf) as pdf:
        for cls in sorted(by_cls.keys()):
            cls_clean = cls.strip()
            orig_list = orig.get(cls_clean, [])
            grpo_list = grpo.get(cls_clean, [])
            orig_text = orig_list[0] if orig_list else ""
            grpo_text = grpo_list[0] if grpo_list else ""

            for clip in by_cls[cls]:
                video_path = clip_dir + clip
                try:
                    frames = sample_frames(video_path, args.n_frames)
                except Exception as e:
                    print(f"  skip {clip}: {e}", flush=True)
                    continue

                qwen_text = qwen.get(clip, "")
                gem_text  = gem.get(clip, "")

                blocks = [
                    ("Original",) + en_zh(orig_text, zh),
                    ("GRPO",)     + en_zh(grpo_text, zh),
                    ("Qwen-VL",)  + en_zh(qwen_text, zh),
                    ("Gemma 4",)  + en_zh(gem_text, zh),
                ]
                header = f"Class: {cls}    |    Clip: {clip}"
                render_page(pdf, frames, header, blocks, args.n_frames)
                del frames
                n_done += 1
                if n_done % 10 == 0:
                    print(f"  [{n_done}/{n_total}] {cls} {clip}", flush=True)
            gc.collect()

    print(f"Saved {n_done} pages -> {args.out_pdf}", flush=True)


if __name__ == "__main__":
    main()
