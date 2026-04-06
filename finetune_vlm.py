"""Fine-tune Qwen2.5-VL with LoRA for micro-gesture classification.

Usage:
  python finetune_vlm.py --dataset iMiGUE --model_name Qwen2.5-VL-7B-Instruct --epochs 5
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, TaskType
from decord import VideoReader, cpu
from PIL import Image

from config import CFG


def sample_frames(video_path, num_frames=16):
    """Sample frames uniformly from video."""
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
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = args.dataset.rstrip("/")

    # Load labels
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    labels_lower = {l.strip().lower(): l for l in labels}
    prompt_text = build_prompt(labels)

    # Load training and test data
    train_dir = f"{dataset_name}/training_clips/"
    test_dir = f"{dataset_name}/testing_clips/"
    train_df = pd.read_csv(train_dir + "clip.csv")
    test_df = pd.read_csv(test_dir + "clip.csv")

    # Load model
    print(f"Loading {args.model_name}...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Classes: {len(labels)}", flush=True)
    print(f"Epochs: {args.epochs}, lr: {args.lr}, LoRA r: {args.lora_r}", flush=True)

    # Training
    for epoch in range(args.epochs):
        model.train()
        indices = list(range(len(train_df)))
        random.shuffle(indices)
        total_loss = 0
        count = 0

        for i, idx in enumerate(indices):
            video_path = train_dir + train_df["clip"].iloc[idx]
            gt_label = train_df["caption"].iloc[idx].strip()

            try:
                frames = sample_frames(video_path, args.num_frames)
            except Exception:
                continue

            # Build message
            content = []
            for frame in frames:
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": prompt_text})

            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": gt_label}]},
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(device)

            # Create labels (mask everything except the answer)
            input_ids = inputs["input_ids"]
            labels_ids = input_ids.clone()
            # Find where assistant answer starts and mask everything before
            answer_tokens = processor.tokenizer.encode(gt_label, add_special_tokens=False)
            answer_len = len(answer_tokens)
            labels_ids[:, :-answer_len] = -100

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels_ids)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1} [{i+1}/{len(indices)}] loss={total_loss/count:.4f}", flush=True)

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}", flush=True)

        # Evaluate
        model.eval()
        correct = 0
        total = len(test_df)

        for idx in range(total):
            video_path = test_dir + test_df["clip"].iloc[idx]
            gt_label = test_df["caption"].iloc[idx].strip()

            try:
                frames = sample_frames(video_path, args.num_frames)
            except Exception:
                continue

            content = []
            for frame in frames:
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50)

            generated = output_ids[0][inputs.input_ids.shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True).strip()

            response_lower = response.lower().strip().strip('"').strip("'")
            pred_label = None
            for lkey, lval in labels_lower.items():
                if lkey in response_lower or response_lower in lkey:
                    pred_label = lval
                    break

            if pred_label and pred_label.strip() == gt_label.strip():
                correct += 1

            if (idx + 1) % 100 == 0:
                print(f"Eval [{idx+1}/{total}] acc@1={correct/(idx+1)*100:.2f}%", flush=True)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}: acc@1={acc:.2f}%", flush=True)

        # Save best
        save_path = f"vlm_ft_{dataset_name}"
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"Saved to {save_path}/", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
