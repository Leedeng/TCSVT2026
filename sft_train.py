"""SFT: Fine-tune Qwen2.5-0.5B with LoRA to generate micro-gesture descriptions.

Training data: GPT-5 mini generated descriptions from descriptions/*.json
Prompt format: "Describe the micro gesture called '{label}' in one sentence:"
Target: One of the GPT-generated descriptions for that label.
"""
import argparse
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.amp import autocast, GradScaler
from tqdm import tqdm


PROMPT_TEMPLATE = "Describe the micro gesture called '{}' in one sentence:"


class SFTDataset(Dataset):
    """Each sample is a (prompt, description) pair for causal LM training."""
    def __init__(self, descriptions_dict, tokenizer, max_length=256):
        self.samples = []
        for label, descs in descriptions_dict.items():
            prompt = PROMPT_TEMPLATE.format(label)
            for desc in descs:
                self.samples.append((prompt, desc))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, desc = self.samples[idx]
        # Format: "<prompt> <description><eos>"
        full_text = prompt + " " + desc + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels: mask the prompt tokens with -100 so loss is only on the description
        prompt_encoded = self.tokenizer(
            prompt + " ",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_len = len(prompt_encoded["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def evaluate(model, eval_loader, device):
    """Compute average loss on eval set."""
    model.eval()
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_count += input_ids.size(0)
    model.train()
    return total_loss / total_count


def generate_samples(model, tokenizer, labels, device, num_samples=3):
    """Generate sample descriptions for a few random labels."""
    model.eval()
    sampled = random.sample(labels, min(num_samples, len(labels)))
    for label in sampled:
        prompt = PROMPT_TEMPLATE.format(label)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  [{label}] {generated.strip()}")
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. iMiGUE")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Base LLM model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load descriptions
    desc_path = f"descriptions/{args.dataset}_descriptions.json"
    with open(desc_path) as f:
        descriptions = json.load(f)
    labels = list(descriptions.keys())
    print(f"Loaded {sum(len(v) for v in descriptions.values())} descriptions for {len(labels)} classes")

    # Split: 90% train, 10% eval (by description, not by class)
    all_samples = []
    for label, descs in descriptions.items():
        for desc in descs:
            all_samples.append((label, desc))
    random.shuffle(all_samples)
    split = int(0.9 * len(all_samples))
    train_descs = {}
    eval_descs = {}
    for label, desc in all_samples[:split]:
        train_descs.setdefault(label, []).append(desc)
    for label, desc in all_samples[split:]:
        eval_descs.setdefault(label, []).append(desc)
    print(f"Train: {sum(len(v) for v in train_descs.values())} samples, Eval: {sum(len(v) for v in eval_descs.values())} samples")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # Datasets
    train_dataset = SFTDataset(train_descs, tokenizer, max_length=args.max_length)
    eval_dataset = SFTDataset(eval_descs, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler("cuda")

    print(f"\nTraining: {args.epochs} epochs, {len(train_loader)} steps/epoch, lr={args.lr}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Model: {args.model_name}\n")

    best_eval_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in tqdm_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            tqdm_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = total_loss / len(train_loader)
        eval_loss = evaluate(model, eval_loader, device)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, eval_loss={eval_loss:.4f}")

        # Generate sample descriptions
        print("Sample generations:")
        generate_samples(model, tokenizer, list(descriptions.keys()), device)

        # Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_path = f"sft_{args.dataset}_{args.model_name.split('/')[-1]}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved best model to {save_path}/ (eval_loss={eval_loss:.4f})")

    print(f"\nDone! Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    main()
