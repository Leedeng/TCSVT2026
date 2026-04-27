"""Curate motivation-figure candidates and render to a single PDF.

For each (dataset, class) we score how good a motivation example it is:
  - Qwen-VL inconsistency across samples (per-sample chaos)
  - Qwen-VL hallucination markers (Nadal/Djokovic/'C '/'X')
  - Gemma echoing the class name (convergence to label)
  - Original is generic (lots of 'indicating', 'often', 'feelings of')
  - GRPO has discrimination markers ('distinct from', 'vs', 'whereas')

Top-N classes are rendered, one page per class, with 3 videos (6 frames each)
and all 4 description sources side-by-side for browsing.

Usage (run on CSC where videos live):
  python make_motivation_pdf.py \
      --datasets iMiGUE SMG MA52 \
      --top_n 25 \
      --out_pdf experiment/motivation_candidates.pdf
"""
import argparse
import gc
import json
import os
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from matplotlib.backends.backend_pdf import PdfPages


CN_FONT_FILES = ["/usr/share/fonts/google-droid/DroidSansFallback.ttf"]


def setup_font():
    for p in CN_FONT_FILES:
        if os.path.exists(p):
            try:
                matplotlib.font_manager.fontManager.addfont(p)
            except Exception:
                pass


def jaccard_dist(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    u = A | B
    return 1 - len(A & B) / len(u) if u else 0.0


def score_class(orig_text, grpo_text, qwen_caps, gemma_caps):
    """Heuristic motivation-figure-worthiness score."""
    s = 0.0
    notes = []

    # 1) Qwen diversity
    if len(qwen_caps) >= 2:
        d = []
        for i in range(len(qwen_caps)):
            for j in range(i+1, len(qwen_caps)):
                d.append(jaccard_dist(qwen_caps[i], qwen_caps[j]))
        avg = np.mean(d) if d else 0
        if avg > 0.5:
            s += 1.5
            notes.append(f"Qwen-divergent({avg:.2f})")

    # 2) Qwen hallucination markers
    halluc_pat = re.compile(r'\b(Nadal|Djokovic|Rafael|Federer|Murray|Williams)\b', re.I)
    placeholder_pat = re.compile(r'\b(C |C$|man X|woman X)')
    halluc = sum(1 for c in qwen_caps if halluc_pat.search(c))
    placeholder = sum(1 for c in qwen_caps if placeholder_pat.search(c))
    if halluc:
        s += 1.0; notes.append(f"halluc({halluc})")
    if placeholder:
        s += 0.3; notes.append(f"placeholder({placeholder})")

    # 3) Gemma convergence (low diversity = echoing the class label)
    if len(gemma_caps) >= 2:
        d = []
        for i in range(len(gemma_caps)):
            for j in range(i+1, len(gemma_caps)):
                d.append(jaccard_dist(gemma_caps[i], gemma_caps[j]))
        avg = np.mean(d) if d else 0
        if avg < 0.2:
            s += 0.5; notes.append(f"Gemma-echoes({avg:.2f})")

    # 4) Original is generic (lots of emotional hedges, no body-part anchors)
    if orig_text:
        hedges = sum(orig_text.lower().count(w) for w in
                     ['indicating', 'often', 'feelings of', 'desire', 'sense of'])
        if hedges >= 2:
            s += 0.5; notes.append(f"orig-generic({hedges})")

    # 5) GRPO has discrimination/grounding markers
    if grpo_text:
        markers = ['distinct from', 'whereas', 'vs.', 'unlike', 'specifically',
                   'as opposed to', 'rather than']
        hits = sum(1 for m in markers if m in grpo_text.lower())
        if hits:
            s += 0.7; notes.append(f"grpo-discrim({hits})")

    return s, notes


def sample_frames(video_path, n=6):
    vr = VideoReader(video_path, ctx=cpu(0))
    try:
        total = len(vr)
        idx = (np.linspace(0, total-1, n, dtype=int).tolist()
               if total > n else list(range(total)) + [total-1]*(n-total))
        frames = vr.get_batch(idx).asnumpy().copy()
        return frames
    finally:
        del vr; gc.collect()


def render_page(pdf, ds, cls, score, notes, items, orig_text, grpo_text, n_frames=6):
    """One page per class. items: list of (clip, frames, qwen_cap, gemma_cap)."""
    fig = plt.figure(figsize=(20, 12))
    n_videos = len(items)

    # Header
    fig.suptitle(
        f"[{ds}]  {cls}    score={score:.2f}    {' '.join(notes)}",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # Layout: per-video block (frames row + 2 caption rows). Then 2 rows for orig/grpo.
    # Use GridSpec.
    from matplotlib.gridspec import GridSpec
    rows_per_video = 1   # only frames row, captions in title-text
    gs = GridSpec(
        n_videos + 1, n_frames, figure=fig,
        height_ratios=[3]*n_videos + [3.5],
        hspace=0.35, wspace=0.05,
    )

    for vi, (clip, frames, qcap, gcap) in enumerate(items):
        for fi, fr in enumerate(frames[:n_frames]):
            ax = fig.add_subplot(gs[vi, fi])
            ax.imshow(fr)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_visible(False)
        # Title above the frame row with both captions
        title = (f"V{vi+1} ({clip[:35]})\n"
                 f"Qwen-VL: {qcap}\n"
                 f"Gemma:   {gcap}")
        ax_first = fig.add_subplot(gs[vi, 0])
        ax_first.set_title(title, fontsize=8, loc="left", pad=4, wrap=True)

    # Bottom: Original + GRPO descriptions
    tax = fig.add_subplot(gs[n_videos, :])
    tax.axis("off")
    tax.set_xlim(0, 1); tax.set_ylim(0, 1)
    tax.text(0.005, 0.95,
             f"Original (GPT-4o-mini): {orig_text}",
             fontsize=10, va="top", wrap=True, color="black")
    tax.text(0.005, 0.45,
             f"GRPO (Ours): {grpo_text}",
             fontsize=10, va="top", wrap=True, color="tab:blue")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def cls_idx_from_clip(clip, idx2name):
    m = re.search(r'_(\d+)\.mp4$', clip)
    return idx2name.get(int(m.group(1))) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["iMiGUE", "SMG", "MA52"])
    parser.add_argument("--top_n", type=int, default=25)
    parser.add_argument("--out_pdf", default="experiment/motivation_candidates.pdf")
    parser.add_argument("--n_frames", type=int, default=6)
    parser.add_argument("--videos_from", default="training_clips")
    args = parser.parse_args()

    setup_font()

    all_candidates = []  # (score, ds, cls, items, orig, grpo, notes)
    for ds in args.datasets:
        orig = {k.strip(): v for k, v in
                json.load(open(f"descriptions/{ds}_descriptions.json")).items()}
        grpo = {k.strip(): v for k, v in
                json.load(open(f"descriptions/{ds}_grpo_descriptions.json")).items()}
        qwen = json.load(open(f"captions/{ds}_qwen.json"))
        gem  = json.load(open(f"captions/{ds}_gemma.json"))

        clip_dir = f"{ds}/{args.videos_from}/"
        df = pd.read_csv(clip_dir + "clip.csv")
        clip2cls = dict(zip(df["clip"].astype(str), df["caption"].astype(str)))

        # Group by class
        by_cls = defaultdict(list)
        for clip in sorted(set(qwen.keys()) | set(gem.keys())):
            cls = clip2cls.get(clip, "?").strip()
            by_cls[cls].append(clip)

        for cls, clips in by_cls.items():
            qcaps = [qwen.get(c, "") for c in clips]
            gcaps = [gem.get(c, "") for c in clips]
            otext = (orig.get(cls.strip(), [""]) or [""])[0]
            gtext = (grpo.get(cls.strip(), [""]) or [""])[0]
            score, notes = score_class(otext, gtext, qcaps, gcaps)
            all_candidates.append((score, ds, cls, clips, otext, gtext, notes))

    # Sort & pick top
    all_candidates.sort(key=lambda x: -x[0])
    picks = all_candidates[: args.top_n]
    print(f"Selected top {len(picks)} candidates:", flush=True)
    for s, ds, cls, _, _, _, n in picks:
        print(f"  [{ds:8}] {cls[:40]:40}  score={s:.2f}  {' '.join(n)}", flush=True)

    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)

    with PdfPages(args.out_pdf) as pdf:
        for score, ds, cls, clips, otext, gtext, notes in picks:
            clip_dir = f"{ds}/{args.videos_from}/"
            qwen = json.load(open(f"captions/{ds}_qwen.json"))
            gem  = json.load(open(f"captions/{ds}_gemma.json"))
            items = []
            for c in clips[:3]:
                try:
                    frames = sample_frames(clip_dir + c, args.n_frames)
                except Exception as e:
                    print(f"  skip {c}: {e}", flush=True)
                    continue
                items.append((c, frames, qwen.get(c, ""), gem.get(c, "")))
                del frames
            if not items:
                continue
            render_page(pdf, ds, cls, score, notes, items, otext, gtext, args.n_frames)
            gc.collect()
            print(f"  rendered [{ds}] {cls}", flush=True)

    print(f"Saved -> {args.out_pdf}", flush=True)


if __name__ == "__main__":
    main()
