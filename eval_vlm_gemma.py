"""Zero-shot evaluation of Gemma 4 E4B (IT) on micro-gesture recognition.

Uses the canonical Gemma 4 multimodal API per the model card:
  - AutoModelForMultimodalLM
  - apply_chat_template(messages, tokenize=True, return_dict=True, ...)
  - {"type": "video", "video": <path>} for direct video input (Gemma's
    video processor samples 32 frames internally)

Usage:
  python eval_vlm_gemma.py --dataset iMiGUE \
      --model_path /scratch/project_2014500/dengli/gemma-4-E4B-it
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


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
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    dataset_name = args.dataset.rstrip("/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    labels_lower = {l.strip().lower(): l for l in labels}
    prompt_text = build_prompt(labels)

    print(f"Loading {args.model_path} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )
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
        rel_clip = clip_df["clip"].iloc[idx]
        video_path = os.path.abspath(test_dir + rel_clip)
        gt_label = clip_df["caption"].iloc[idx].strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)

            response_raw = processor.decode(
                outputs[0][input_len:], skip_special_tokens=False
            )
            try:
                response = processor.parse_response(response_raw)
            except Exception:
                response = response_raw
        except Exception as e:
            print(f"generate failed on {video_path}: {e}", flush=True)
            continue

        if isinstance(response, dict):
            response = response.get("content", "") or response.get("text", "") or str(response)
        elif isinstance(response, list) and response:
            response = response[0] if isinstance(response[0], str) else str(response[0])
        response = str(response).strip()

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
                f"pred='{response[:60]}' gt='{gt_label}'",
                flush=True,
            )

    acc_1 = correct_1 / max(seen, 1) * 100
    acc_5 = correct_5 / max(seen, 1) * 100
    avg_time_ms = np.mean(times) * 1000.0 if times else 0.0
    median_time_ms = np.median(times) * 1000.0 if times else 0.0

    print("\n" + "=" * 60)
    print(f"Gemma 4 E4B (IT) zero-shot on {dataset_name}")
    print("=" * 60)
    print(f"Seen samples:     {seen} / {total}")
    print(f"Acc@1:            {acc_1:.2f}%")
    print(f"Acc@5:            {acc_5:.2f}%")
    print(f"Avg time:         {avg_time_ms:.1f} ms")
    print(f"Median time:      {median_time_ms:.1f} ms")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
