"""GRPO: Group Relative Policy Optimization for micro-gesture description generation.

Per-class optimization: generate descriptions per class (not per sample),
evaluate against all samples of that class, average advantages.
~430x faster than per-sample.

Uses verifiable rewards (no reward model) to optimize LLM description quality:
  R1 (alignment): cosine_sim(image_emb, text_emb) in frozen contrastive space
  R2 (discrimination): margin between correct class and nearest wrong class
  R3 (quality): rule-based length/repetition checks
"""
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from config import CFG
from dataset import get_dataloader
from models import VideoCLIPModel

PROMPT_TEMPLATE = "Describe the micro gesture called '{}' in one sentence:"


# ============================================================
# Reward Functions
# ============================================================

def compute_R1(text_emb, mean_image_embs, correct_idx):
    """Ranking-based reward: 1/(rank+1) of correct class among all classes.
    Rank 0 (top-1) → 1.0, rank 1 → 0.5, rank 4 → 0.2, etc.
    Much sharper signal than raw cosine similarity."""
    sims = F.cosine_similarity(text_emb, mean_image_embs, dim=-1)  # [num_classes]
    correct_sim = sims[correct_idx]
    rank = (sims > correct_sim).sum().item()
    return 1.0 / (rank + 1)


def compute_R2(text_emb, image_emb, mean_image_embs, correct_idx):
    """Per-sample cosine similarity (continuous fine-grained signal)."""
    return F.cosine_similarity(text_emb, image_emb, dim=-1).item()


def compute_R3(description, tokenizer, min_len=10, max_len=80):
    """Quality reward: rule-based checks on generation quality."""
    tokens = tokenizer.encode(description)
    score = 1.0
    if len(tokens) < min_len or len(tokens) > max_len:
        score -= 0.5
    words = description.lower().split()
    if len(words) > 0 and len(set(words)) / len(words) < 0.5:
        score -= 0.5
    return max(score, 0.0)


def compute_reward(text_emb, image_emb, mean_image_embs, correct_idx,
                   description, tokenizer, alpha=1.0, beta=0.3, gamma=0.2):
    """Total verifiable reward. R1 (ranking) dominates."""
    r1 = compute_R1(text_emb, mean_image_embs, correct_idx)
    r2 = compute_R2(text_emb, image_emb, mean_image_embs, correct_idx)
    r3 = compute_R3(description, tokenizer)
    total = alpha * r1 + beta * r2 + gamma * r3
    return total, r1, r2, r3


# ============================================================
# Pre-encoding
# ============================================================

def pre_encode_videos(reward_model, dataloader, device):
    """Pre-encode all training videos with frozen visual encoder.
    Returns: image_embs [N, D], labels [N] (class indices), label_names [N]."""
    reward_model.eval()
    all_embs = []
    all_labels = []
    all_label_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Pre-encoding videos"):
            clip = batch["clip"].to(device)
            image_emb = reward_model.encode_image(clip)
            image_emb = F.normalize(image_emb, dim=-1)
            all_embs.append(image_emb.cpu())
            label_idx = batch["label"].argmax(dim=-1)
            all_labels.append(label_idx)
            all_label_names.extend(batch["caption"])

    return torch.cat(all_embs), torch.cat(all_labels), all_label_names


def compute_mean_class_embs(image_embs, label_indices, num_classes):
    """Compute mean image embedding per class for R2."""
    mean_embs = torch.zeros(num_classes, image_embs.shape[1])
    counts = torch.zeros(num_classes)
    for i in range(len(image_embs)):
        c = label_indices[i].item()
        mean_embs[c] += image_embs[i]
        counts[c] += 1
    for c in range(num_classes):
        if counts[c] > 0:
            mean_embs[c] /= counts[c]
    return F.normalize(mean_embs, dim=-1)


# ============================================================
# Log prob computation
# ============================================================

def compute_log_probs(model, tokenizer, prompt, description, device):
    """Compute per-token log probabilities of description given prompt."""
    full_text = prompt + " " + description + tokenizer.eos_token
    prompt_text = prompt + " "

    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = prompt_enc["input_ids"].shape[1]

    with autocast("cuda"):
        outputs = model(**full_enc)
        logits = outputs.logits

    shift_logits = logits[:, prompt_len-1:-1, :]
    shift_labels = full_enc["input_ids"][:, prompt_len:]

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum()


# ============================================================
# Per-class GRPO step
# ============================================================

def grpo_class_step(llm, ref_lora_state, tokenizer, reward_model, text_encoder_tokenizer,
                    class_image_embs, label_name, correct_idx, mean_image_embs,
                    device, G=8, temperature=0.8, max_new_tokens=80,
                    epsilon=0.2, beta_kl=0.04,
                    alpha=1.0, beta_r=0.3, gamma=0.2):
    """One GRPO step for an entire class.

    Generates G descriptions once, evaluates against all samples of the class,
    averages advantages, then computes loss.

    Args:
        class_image_embs: [M, D] pre-cached image embeddings for this class
        Other args same as before.
    """
    prompt = PROMPT_TEMPLATE.format(label_name)
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # 1. Sample G descriptions (once per class)
    llm.eval()
    with torch.no_grad():
        outputs = llm.generate(
            **prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=G,
            pad_token_id=tokenizer.pad_token_id,
        )
    descriptions = []
    for i in range(G):
        desc = tokenizer.decode(outputs[i][prompt_ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        descriptions.append(desc)

    # 2. Encode descriptions through reward model (once per class)
    text_embs = []
    r3_scores = []
    reward_model.eval()
    with torch.no_grad():
        for desc in descriptions:
            desc_tokens = text_encoder_tokenizer(
                desc, return_tensors="pt", padding=True, truncation=True, max_length=CFG.max_length
            ).to(device)
            text_emb = reward_model.encode_text(
                input_ids=desc_tokens["input_ids"],
                attention_mask=desc_tokens["attention_mask"],
            )
            text_emb = F.normalize(text_emb, dim=-1)
            text_embs.append(text_emb)
            r3_scores.append(compute_R3(desc, tokenizer))

    # 3. Compute rewards for all samples × all descriptions
    M = class_image_embs.shape[0]  # number of samples in this class
    all_rewards = torch.zeros(M, G)  # [M, G]
    all_r1s = torch.zeros(M, G)

    mean_embs_device = mean_image_embs.to(device)
    for g in range(G):
        # R1 (ranking): same for all samples of this class (uses mean_image_embs)
        r1 = compute_R1(text_embs[g], mean_embs_device, correct_idx)
        for m in range(M):
            img_emb = class_image_embs[m:m+1].to(device)
            r2 = compute_R2(text_embs[g], img_emb, mean_embs_device, correct_idx)
            total = alpha * r1 + beta_r * r2 + gamma * r3_scores[g]
            all_rewards[m, g] = total
            all_r1s[m, g] = r1

    # 4. Per-sample advantage, then average across samples
    # For each sample: normalize rewards across G descriptions
    advantages = torch.zeros(G)
    for m in range(M):
        sample_rewards = all_rewards[m]
        if sample_rewards.std() > 1e-8:
            sample_adv = (sample_rewards - sample_rewards.mean()) / sample_rewards.std()
        else:
            sample_adv = torch.zeros(G)
        advantages += sample_adv
    advantages = advantages / M  # average advantage per description

    # 5. Compute log probs and GRPO loss
    llm.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    valid_count = 0

    for g in range(G):
        if len(descriptions[g].strip()) == 0:
            continue

        # Current policy log prob
        log_prob_theta = compute_log_probs(llm, tokenizer, prompt, descriptions[g], device)

        # Reference log prob: swap in frozen SFT weights
        current_lora = {}
        for name, param in llm.named_parameters():
            if name in ref_lora_state:
                current_lora[name] = param.data.clone()
                param.data.copy_(ref_lora_state[name])

        with torch.no_grad():
            log_prob_ref = compute_log_probs(llm, tokenizer, prompt, descriptions[g], device)

        for name, param in llm.named_parameters():
            if name in current_lora:
                param.data.copy_(current_lora[name])

        # Clipped surrogate objective
        ratio = torch.exp(log_prob_theta - log_prob_ref.detach())
        A = advantages[g].detach().to(device)
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A
        policy_loss = -torch.min(surr1, surr2)

        # KL penalty
        kl = log_prob_theta - log_prob_ref.detach()
        total_loss = total_loss + policy_loss + beta_kl * kl
        valid_count += 1

    total_loss = total_loss / max(valid_count, 1)

    mean_reward = all_rewards.mean().item()
    mean_r1 = all_r1s.mean().item()
    # Back-compute mean R2 from total = alpha*R1 + beta*R2 + gamma*R3.
    # Use the actual reward weights so the logged R2 matches what the
    # advantage was based on. Previously this hardcoded /0.5 which gave
    # an incorrect (sometimes negative) R2 in tensorboard.
    mean_r2 = (mean_reward - alpha * mean_r1 - gamma * np.mean(r3_scores)) / max(beta_r, 1e-8)
    best_desc = descriptions[advantages.argmax().item()]

    return total_loss, mean_reward, {
        "r1": mean_r1, "r2": mean_r2, "r3": np.mean(r3_scores),
    }, best_desc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True, help="Frozen baseline model .pt")
    parser.add_argument("--sft_model_path", type=str, required=True, help="SFT LoRA adapter directory")
    parser.add_argument("--llm_base", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--G", type=int, default=8, help="Group size")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_suffix", type=str, default=None, help="Suffix for save directory")
    parser.add_argument("--alpha", type=float, default=1.0, help="R1 weight")
    parser.add_argument("--beta_r", type=float, default=0.3, help="R2 weight")
    parser.add_argument("--gamma", type=float, default=0.2, help="R3 weight")
    parser.add_argument("--ckpt_dir", type=str, default=".", help="Directory to save checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = args.dataset.rstrip("/")
    writer = SummaryWriter(f"./log/{dataset_name}_grpo")

    # Load label info
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    num_classes = len(labels)

    # Load frozen reward model
    print("Loading frozen reward model...")
    reward_model = VideoCLIPModel(num_classes=num_classes).to(device)
    reward_model.load_state_dict(torch.load(args.reward_model_path, map_location=device))
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad = False

    # Pre-encode all training videos
    print("Pre-encoding training videos...")
    train_loader = get_dataloader(dataset_name, mode="training", label_names=labels)
    image_embs, label_indices, label_names_list = pre_encode_videos(reward_model, train_loader, device)
    mean_image_embs = compute_mean_class_embs(image_embs, label_indices, num_classes)
    print(f"Cached {len(image_embs)} image embeddings, {num_classes} classes")

    # Group image embeddings by class
    class_embs = {}
    for i in range(len(image_embs)):
        c = label_indices[i].item()
        if c not in class_embs:
            class_embs[c] = []
        class_embs[c].append(image_embs[i])
    for c in class_embs:
        class_embs[c] = torch.stack(class_embs[c])
    print(f"Samples per class: {', '.join(f'{labels[c]}:{len(v)}' for c, v in sorted(class_embs.items())[:5])}...")

    # Free reward model's visual encoder from GPU (only need text encoder for reward)
    # Keep reward_model on device for encode_text

    # Load SFT-trained LLM with LoRA
    print("Loading SFT LLM...")
    llm_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.llm_base, torch_dtype=torch.float16, trust_remote_code=True,
    )
    llm = PeftModel.from_pretrained(base_model, args.sft_model_path, is_trainable=True)
    llm = llm.to(device)
    llm.print_trainable_parameters()

    # Save SFT LoRA weights as frozen reference policy
    ref_lora_state = {}
    for name, param in llm.named_parameters():
        if param.requires_grad:
            ref_lora_state[name] = param.data.clone()
    print(f"Saved {len(ref_lora_state)} reference LoRA parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in llm.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    text_encoder_tokenizer = CFG.tokenizer

    print(f"\nGRPO Training (per-class): {args.epochs} epochs, G={args.G}, lr={args.lr}")
    print(f"Classes: {num_classes}, ~{len(image_embs)//num_classes} samples/class\n")

    global_step = 0
    best_reward = -float("inf")

    for epoch in range(args.epochs):
        class_order = list(range(num_classes))
        random.shuffle(class_order)
        epoch_rewards = []
        epoch_r1s, epoch_r2s, epoch_r3s = [], [], []

        pbar = tqdm(class_order, desc=f"Epoch {epoch+1}/{args.epochs}")
        for c in pbar:
            loss, mean_reward, reward_dict, best_desc = grpo_class_step(
                llm, ref_lora_state, llm_tokenizer, reward_model, text_encoder_tokenizer,
                class_embs[c], labels[c], c, mean_image_embs,
                device, G=args.G, temperature=args.temperature,
                epsilon=args.epsilon, beta_kl=args.beta_kl,
                alpha=args.alpha, beta_r=args.beta_r, gamma=args.gamma,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_rewards.append(mean_reward)
            epoch_r1s.append(reward_dict["r1"])
            epoch_r2s.append(reward_dict["r2"])
            epoch_r3s.append(reward_dict["r3"])

            pbar.set_postfix(
                R=f"{mean_reward:.3f}",
                R1=f"{reward_dict['r1']:.3f}",
                loss=f"{loss.item():.4f}",
            )

            global_step += 1
            writer.add_scalar("reward/total", mean_reward, global_step)
            writer.add_scalar("reward/R1_align", reward_dict["r1"], global_step)
            writer.add_scalar("reward/R2_discrim", reward_dict["r2"], global_step)
            writer.add_scalar("loss", loss.item(), global_step)

        # Epoch summary
        avg_reward = np.mean(epoch_rewards)
        print(f"Epoch {epoch+1}: avg_reward={avg_reward:.4f}, "
              f"R1={np.mean(epoch_r1s):.4f}, R2={np.mean(epoch_r2s):.4f}, R3={np.mean(epoch_r3s):.4f}")

        # Generate sample descriptions (greedy)
        print("Sample descriptions:")
        llm.eval()
        for label in labels[:3]:
            prompt = PROMPT_TEMPLATE.format(label)
            inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = llm.generate(
                    **inputs, max_new_tokens=80, do_sample=False,
                    pad_token_id=llm_tokenizer.pad_token_id,
                )
            desc = llm_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"  [{label}] {desc}")

        # Save best
        if avg_reward > best_reward:
            best_reward = avg_reward
            suffix = args.save_suffix or args.llm_base.split('/')[-1]
            save_path = f"{args.ckpt_dir}/grpo_{dataset_name}_{suffix}"
            llm.save_pretrained(save_path)
            llm_tokenizer.save_pretrained(save_path)
            print(f"Saved best model (avg_reward={avg_reward:.4f})")

    writer.close()
    print(f"\nDone! Best avg reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
