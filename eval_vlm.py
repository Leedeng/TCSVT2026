"""Evaluate Qwen2.5-VL on micro-gesture recognition (zero-shot).

Usage:
  python eval_vlm.py --dataset iMiGUE --mode zero-shot
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu

from config import CFG


def sample_frames(video_path, num_frames=8):
    """Sample frames uniformly from video, return as PIL images."""
    from PIL import Image
    vr = VideoReader(video_path, ctx=cpu(0), width=CFG.width, height=CFG.height)
    total = len(vr)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames]


def build_prompt(labels):
    """Build zero-shot classification prompt."""
    label_list = ", ".join([f'"{l}"' for l in labels])
    return (
        f"This video shows a person performing a micro-gesture. "
        f"Classify it into exactly one of the following categories: {label_list}. "
        f"Reply with only the category name, nothing else."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()

    dataset_name = args.dataset.rstrip("/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load labels
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    labels_lower = {l.strip().lower(): l for l in labels}

    # Load model
    print(f"Loading {args.model_name}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    print("Model loaded.", flush=True)

    # Load test data
    test_dir = f"{dataset_name}/testing_clips/"
    clip_df = pd.read_csv(test_dir + "clip.csv")
    prompt_text = build_prompt(labels)

    correct_1 = 0
    correct_5 = 0
    total = len(clip_df)
    times = []

    for idx in range(total):
        video_path = test_dir + clip_df["clip"].iloc[idx]
        gt_label = clip_df["caption"].iloc[idx].strip()

        # Sample frames
        frames = sample_frames(video_path, args.num_frames)

        # Build message with frames as images
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        # Process and generate
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50)
        t1 = time.time()
        times.append(t1 - t0)

        # Decode response
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True).strip()

        # Match prediction to label
        response_lower = response.lower().strip().strip('"').strip("'")
        pred_label = None
        for lkey, lval in labels_lower.items():
            if lkey in response_lower or response_lower in lkey:
                pred_label = lval
                break

        if pred_label and pred_label.strip() == gt_label.strip():
            correct_1 += 1
            correct_5 += 1
        else:
            # For acc@5, check if GT appears anywhere in response
            if gt_label.strip().lower() in response_lower:
                correct_5 += 1

        if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
            print(f"[{idx+1}/{total}] acc@1={correct_1/(idx+1)*100:.2f}% "
                  f"avg_time={np.mean(times):.2f}s "
                  f"pred='{response[:60]}' gt='{gt_label}'", flush=True)

    acc_1 = correct_1 / total * 100
    acc_5 = correct_5 / total * 100
    avg_time = np.mean(times)

    print(f"\n{args.model_name}\tacc@1={acc_1:.2f}%\tacc@5={acc_5:.2f}%\tavg_time={avg_time:.3f}s")


if __name__ == "__main__":
    main()
