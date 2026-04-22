"""Zero-shot evaluation of Gemma 4 E4B on micro-gesture recognition.

Adapted from eval_vlm.py for Gemma 4 (Gemma4ForConditionalGeneration).
Uses the multimodal chat template with multiple sampled frames as image
content. Reports Acc@1, Acc@5, and avg generation time per sample.

Note: Gemma 4 requires transformers >= 5.5.0.dev0. If the installed
transformers does not know about Gemma 4, install from main:
  pip install --user --upgrade git+https://github.com/huggingface/transformers@main

Usage:
  python eval_vlm_gemma.py --dataset iMiGUE \
      --model_path /scratch/project_2014500/dengli/gemma-4-E4B
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from decord import VideoReader, cpu
from PIL import Image

from config import CFG


def sample_frames(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0), width=CFG.width, height=CFG.height)
    total = len(vr)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames]


def build_prompt(labels):
    label_list = ", ".join([f'"{l}"' for l in labels])
    return (
        f"This video shows a person performing a micro-gesture. "
        f"Classify it into exactly one of the following categories: {label_list}. "
        f"Reply with only the category name, nothing else."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    dataset_name = args.dataset.rstrip("/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    labels_lower = {l.strip().lower(): l for l in labels}
    prompt_text = build_prompt(labels)

    print(f"Loading {args.model_path}...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    print(f"Model loaded. Device: {device}", flush=True)

    test_dir = f"{dataset_name}/testing_clips/"
    clip_df = pd.read_csv(test_dir + "clip.csv")
    total = len(clip_df)
    print(f"Test split: {total} clips, {len(labels)} classes", flush=True)

    correct_1, correct_5 = 0, 0
    seen = 0
    times = []

    for idx in range(total):
        video_path = test_dir + clip_df["clip"].iloc[idx]
        gt_label = clip_df["caption"].iloc[idx].strip()

        try:
            frames = sample_frames(video_path, args.num_frames)
        except Exception as e:
            print(f"skip {video_path}: {e}", flush=True)
            continue

        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            if hasattr(inputs, "to"):
                pass
            input_len = inputs["input_ids"].shape[1]

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)

            generated = output_ids[0][input_len:]
            response = processor.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"generate failed on {video_path}: {e}", flush=True)
            continue

        response_lower = response.lower().strip().strip('"').strip("'")
        pred_label = None
        for lkey, lval in labels_lower.items():
            if lkey in response_lower or response_lower in lkey:
                pred_label = lval
                break

        seen += 1
        if pred_label and pred_label.strip() == gt_label.strip():
            correct_1 += 1
            correct_5 += 1
        else:
            if gt_label.strip().lower() in response_lower:
                correct_5 += 1

        if seen % 10 == 0 or seen == 1 or idx == total - 1:
            print(
                f"[{idx+1}/{total}] seen={seen} "
                f"acc@1={correct_1/seen*100:.2f}% "
                f"acc@5={correct_5/seen*100:.2f}% "
                f"avg_time={np.mean(times):.2f}s "
                f"pred='{response[:50]}' gt='{gt_label}'",
                flush=True,
            )

    acc_1 = correct_1 / max(seen, 1) * 100
    acc_5 = correct_5 / max(seen, 1) * 100
    avg_time_ms = np.mean(times) * 1000.0 if times else 0.0
    median_time_ms = np.median(times) * 1000.0 if times else 0.0

    print("\n" + "=" * 60)
    print(f"Gemma 4 E4B zero-shot on {dataset_name}")
    print("=" * 60)
    print(f"Seen samples:     {seen} / {total}")
    print(f"Acc@1:            {acc_1:.2f}%")
    print(f"Acc@5:            {acc_5:.2f}%")
    print(f"Avg time:         {avg_time_ms:.1f} ms")
    print(f"Median time:      {median_time_ms:.1f} ms")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
