"""Generate descriptions using GRPO-optimized LLM for each class.

Usage:
  python generate_grpo_descriptions.py --dataset iMiGUE \
    --sft_model_path grpo_iMiGUE_Qwen2.5-0.5B \
    --llm_base Qwen/Qwen2.5-0.5B \
    --num_per_class 15
"""
import argparse
import json

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPT_TEMPLATE = "Describe the micro gesture called '{}' in one sentence:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sft_model_path", type=str, required=True, help="GRPO or SFT LoRA adapter directory")
    parser.add_argument("--llm_base", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_per_class", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    dataset_name = args.dataset.rstrip("/")
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    print(f"Dataset: {dataset_name}, {len(labels)} classes")

    # Load model
    print(f"Loading model from {args.sft_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.llm_base, torch_dtype=torch.float16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.sft_model_path)
    model = model.to(device)
    model.eval()

    # Generate descriptions
    descriptions = {}
    for label in labels:
        prompt = PROMPT_TEMPLATE.format(label)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        descs = set()
        # Generate greedy (best single description)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        greedy_desc = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        descs.add(greedy_desc)

        # Generate diverse descriptions with sampling
        while len(descs) < args.num_per_class:
            with torch.no_grad():
                batch_size = min(args.num_per_class - len(descs) + 5, 20)
                outs = model.generate(
                    **inputs, max_new_tokens=80,
                    do_sample=True, temperature=args.temperature, top_p=0.9,
                    num_return_sequences=batch_size,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for i in range(outs.shape[0]):
                desc = tokenizer.decode(outs[i][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                if len(desc) > 10:
                    descs.add(desc)
                if len(descs) >= args.num_per_class:
                    break

        descriptions[label] = list(descs)[:args.num_per_class]
        print(f"  [{label}] {len(descriptions[label])} descriptions, greedy: {greedy_desc[:80]}...")

    # Save
    output_path = args.output or f"descriptions/{dataset_name}_grpo_descriptions.json"
    with open(output_path, "w") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {sum(len(v) for v in descriptions.values())} descriptions to {output_path}")


if __name__ == "__main__":
    main()
