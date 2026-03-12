"""
Step 1: Generate micro-gesture descriptions using GPT-5 mini.

Usage:
    python generate_descriptions.py --dataset SMG --num_per_angle 5
    python generate_descriptions.py --dataset iMiGUE --num_per_angle 5

Requires: OPENAI_API_KEY environment variable.
Output: descriptions/{dataset}_descriptions.json
"""

import argparse
import json
import os
import time

import pandas as pd
from openai import OpenAI


def build_prompts(label: str, all_labels: list[str]) -> list[dict]:
    """Build prompts from three angles for a given label."""
    return [
        {
            "angle": "action",
            "system": "You are an expert in human body language and micro-gesture analysis.",
            "user": (
                f"Describe the micro gesture '{label}' in one sentence, "
                f"focusing on which body parts are involved and the specific motion pattern."
            ),
        },
        {
            "angle": "discriminative",
            "system": "You are an expert in human body language and micro-gesture analysis.",
            "user": (
                f"Given these micro gesture categories: {', '.join(all_labels)}, "
                f"describe '{label}' in one sentence that clearly distinguishes it "
                f"from the most similar categories."
            ),
        },
        {
            "angle": "emotion",
            "system": "You are an expert in human body language and micro-gesture analysis.",
            "user": (
                f"Describe the micro gesture '{label}' in one sentence, "
                f"focusing on what emotional state or psychological condition it might indicate."
            ),
        },
    ]


def generate_descriptions(client: OpenAI, label: str, all_labels: list[str],
                          num_per_angle: int = 5, model: str = "gpt-5-mini") -> list[str]:
    """Generate descriptions for one label from multiple angles."""
    prompts = build_prompts(label, all_labels)
    descriptions = []

    for prompt_info in prompts:
        for i in range(num_per_angle):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt_info["system"]},
                        {"role": "user", "content": prompt_info["user"]},
                    ],
                    max_completion_tokens=100,
                )
                desc = response.choices[0].message.content.strip()
                if desc and 10 < len(desc) < 500:
                    descriptions.append(desc)
            except Exception as e:
                print(f"  ERROR: {e}")
                time.sleep(1)

    # Deduplicate
    seen = set()
    unique = []
    for d in descriptions:
        d_lower = d.lower()
        if d_lower not in seen:
            seen.add(d_lower)
            unique.append(d)

    return unique


def main():
    parser = argparse.ArgumentParser(description="Generate MG descriptions via GPT-5 mini")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (SMG or iMiGUE)")
    parser.add_argument("--num_per_angle", type=int, default=5, help="Number of descriptions per angle")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="OpenAI model name")
    parser.add_argument("--output_dir", type=str, default="descriptions", help="Output directory")
    args = parser.parse_args()

    client = OpenAI()  # uses OPENAI_API_KEY env var

    label_df = pd.read_csv(f"{args.dataset}/Clip_label.csv")
    labels = list(label_df["name"].values)
    print(f"Dataset: {args.dataset}, {len(labels)} classes")

    all_descriptions = {}
    for i, label in enumerate(labels):
        print(f"[{i+1}/{len(labels)}] Generating for: {label}")
        descs = generate_descriptions(client, label, labels,
                                      num_per_angle=args.num_per_angle, model=args.model)
        all_descriptions[label] = descs
        print(f"  Got {len(descs)} unique descriptions")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}_descriptions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_descriptions, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in all_descriptions.values())
    print(f"\nDone! Total {total} descriptions saved to {output_path}")


if __name__ == "__main__":
    main()
