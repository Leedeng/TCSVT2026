"""Generate one free-form caption per video using a VLM (Qwen2.5-VL or Gemma 4).

For each (dataset, class) pair, sample K videos and ask the VLM to describe
the micro-gesture in one sentence. Saves a JSON dict keyed by relative
clip path: { "clip_rel_path": "caption text" }.

Usage:
  python vlm_caption.py --dataset iMiGUE --model qwen \
      --model_path /scratch/.../Qwen2.5-VL-7B-Instruct \
      --k_per_class 3 --out captions/iMiGUE_qwenvl.json
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from decord import VideoReader, cpu


PROMPT = (
    "This video shows a person performing a micro-gesture. "
    "Describe what the person is doing in ONE concise sentence focused on "
    "the body part(s) and the motion."
)


def sample_frames(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    try:
        total = len(vr)
        if total <= num_frames:
            idx = list(range(total)) + [total - 1] * (num_frames - total)
        else:
            idx = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames = vr.get_batch(idx).asnumpy().copy()
        return [Image.fromarray(f) for f in frames]
    finally:
        del vr
        gc.collect()


def caption_qwen(model, processor, video_path, num_frames, device, max_new=80):
    from qwen_vl_utils import process_vision_info
    frames = sample_frames(video_path, num_frames)
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": PROMPT})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def caption_gemma(model, processor, video_path, num_frames, device, max_new=80):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": os.path.abspath(video_path)},
            {"type": "text", "text": PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    raw = processor.decode(out[0][input_len:], skip_special_tokens=False)
    try:
        parsed = processor.parse_response(raw)
        if isinstance(parsed, dict):
            return (parsed.get("content") or parsed.get("text") or str(parsed)).strip()
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
        return str(parsed).strip()
    except Exception:
        return raw.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=["qwen", "gemma"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--k_per_class", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--videos_from", default="training_clips")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = args.dataset.rstrip("/")
    clip_dir = f"{ds}/{args.videos_from}/"
    df = pd.read_csv(clip_dir + "clip.csv")

    rng = np.random.default_rng(args.seed)
    classes = sorted(set(df["caption"].values))
    picks = []
    for cls in classes:
        sub = df[df["caption"] == cls].reset_index(drop=True)
        if len(sub) == 0:
            continue
        n = min(args.k_per_class, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        for i in idx:
            picks.append((cls, sub.iloc[int(i)]["clip"]))

    print(f"dataset={ds} model={args.model} videos={len(picks)} "
          f"({len(classes)} classes x {args.k_per_class})", flush=True)

    print(f"Loading {args.model_path} ...", flush=True)
    if args.model == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
        cap_fn = caption_qwen
    else:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path)
        if hasattr(processor, "video_processor") and processor.video_processor is not None:
            processor.video_processor.num_frames = args.num_frames
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_path, dtype="auto", device_map="auto",
        )
        cap_fn = caption_gemma
    model.eval()
    print("Model loaded.", flush=True)

    captions = {}
    if os.path.exists(args.out):
        captions = json.load(open(args.out))
        print(f"Resuming with {len(captions)} cached captions", flush=True)

    t_start = time.time()
    for i, (cls, clip_rel) in enumerate(picks):
        if clip_rel in captions:
            continue
        video_path = clip_dir + clip_rel
        try:
            cap = cap_fn(model, processor, video_path, args.num_frames, device,
                        max_new=args.max_new_tokens)
        except Exception as e:
            cap = f"(generation failed: {e})"
            print(f"  [{i+1}/{len(picks)}] FAIL {clip_rel}: {e}", flush=True)
        captions[clip_rel] = cap
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = (time.time() - t_start) / 60
            eta = elapsed / (i + 1) * (len(picks) - i - 1)
            print(f"  [{i+1}/{len(picks)}] {cls}: {cap[:80]}  "
                  f"elapsed={elapsed:.1f}m eta={eta:.1f}m", flush=True)
        # Periodically dump so we can resume on timeout
        if (i + 1) % 20 == 0:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(captions, f, indent=2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {len(captions)} captions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
