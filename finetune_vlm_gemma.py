"""LoRA fine-tune Gemma 4 E4B (IT) on micro-gesture classification.

Mirrors finetune_vlm.py but uses the canonical Gemma 4 multimodal API
from the model card:
  - AutoModelForMultimodalLM
  - apply_chat_template(messages, tokenize=True, return_dict=True, ...)
  - {"type": "video", "video": <path>} for direct video input

The supervised target is the label string as the assistant turn. We
mask everything except the last `len(answer_tokens)` positions in the
labels tensor so that the loss is computed only on the response tokens.

Usage:
  python finetune_vlm_gemma.py --dataset iMiGUE \
      --model_path /scratch/project_2014500/dengli/gemma-4-E4B-it
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType


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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target module suffixes",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per video for Gemma's video processor (Gemma default is 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="vlm_ft_gemma_iMiGUE")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset.rstrip("/")

    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    labels_lower = {l.strip().lower(): l for l in labels}
    prompt_text = build_prompt(labels)

    print(f"Loading {args.model_path} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    # iMiGUE clips are short (~30 frames at 30 FPS); reduce frame count
    # so the video processor doesn't fail on the shortest clips and to
    # cut sequence length for faster training.
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        processor.video_processor.num_frames = args.num_frames
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
    )

    # Enable gradient checkpointing for memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    train_dir = f"{dataset_name}/training_clips/"
    test_dir = f"{dataset_name}/testing_clips/"
    train_df = pd.read_csv(train_dir + "clip.csv")
    test_df = pd.read_csv(test_dir + "clip.csv")

    print(
        f"Train={len(train_df)}, Test={len(test_df)}, Classes={len(labels)} "
        f"epochs={args.epochs} lr={args.lr} lora_r={args.lora_r}",
        flush=True,
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_df)))
        random.shuffle(order)
        total_loss, count = 0.0, 0
        ep_t0 = time.time()

        for i, idx in enumerate(order):
            video_path = train_dir + train_df["clip"].iloc[idx]
            gt_label = train_df["caption"].iloc[idx].strip()

            # Build full conversation with assistant turn = ground-truth label
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": gt_label}],
                },
            ]

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                ).to(model.device)
            except Exception as e:
                print(f"skip {video_path}: {e}", flush=True)
                continue

            input_ids = inputs["input_ids"]
            labels_ids = input_ids.clone()
            answer_tokens = processor.tokenizer.encode(gt_label, add_special_tokens=False)
            ans_len = max(len(answer_tokens), 1)
            labels_ids[:, :-ans_len] = -100

            optimizer.zero_grad()
            try:
                outputs = model(**inputs, labels=labels_ids)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM on {video_path}: {e}", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"step failed on {video_path}: {e}", flush=True)
                continue

            total_loss += loss.item()
            count += 1
            if (i + 1) % 50 == 0:
                ep_elapsed = time.time() - ep_t0
                eta = ep_elapsed / (i + 1) * (len(order) - i - 1)
                print(
                    f"ep {epoch+1} [{i+1}/{len(order)}] "
                    f"loss={total_loss/max(count,1):.4f} "
                    f"elapsed={ep_elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )

        print(
            f"ep {epoch+1} done: train_loss={total_loss/max(count,1):.4f} "
            f"({(time.time()-ep_t0)/60:.1f} min)",
            flush=True,
        )

        # Evaluation
        model.eval()
        correct = 0
        seen = 0
        eval_t0 = time.time()
        for j in range(len(test_df)):
            video_path = test_dir + test_df["clip"].iloc[j]
            gt_label = test_df["caption"].iloc[j].strip()
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
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                    )
                response_raw = processor.decode(out[0][input_len:], skip_special_tokens=False)
                try:
                    response = processor.parse_response(response_raw)
                except Exception:
                    response = response_raw
            except Exception as e:
                print(f"eval skip {video_path}: {e}", flush=True)
                continue

            if isinstance(response, dict):
                response = response.get("content", "") or response.get("text", "") or str(response)
            elif isinstance(response, list) and response:
                response = response[0] if isinstance(response[0], str) else str(response[0])
            response = str(response).strip()
            response_lower = response.lower().strip().strip('"').strip("'")

            pred = None
            for lkey, lval in labels_lower.items():
                if lkey in response_lower or response_lower in lkey:
                    pred = lval
                    break
            seen += 1
            if pred and pred.strip() == gt_label.strip():
                correct += 1
            if (j + 1) % 100 == 0:
                print(
                    f"  eval [{j+1}/{len(test_df)}] acc@1={correct/max(seen,1)*100:.2f}%",
                    flush=True,
                )

        acc = correct / max(seen, 1) * 100
        print(
            f"ep {epoch+1}: acc@1={acc:.2f}% ({(time.time()-eval_t0)/60:.1f}m for eval)",
            flush=True,
        )

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(args.save_dir)
            processor.save_pretrained(args.save_dir)
            print(f"  saved best to {args.save_dir} (acc@1={best_acc:.2f}%)", flush=True)

    print(f"\nBest Acc@1: {best_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
