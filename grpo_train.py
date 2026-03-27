"""GRPO: Group Relative Policy Optimization for micro-gesture description generation.

Uses verifiable rewards (no reward model) to optimize LLM description quality:
  R1 (alignment): cosine_sim(image_emb, text_emb) in frozen contrastive space
  R2 (discrimination): margin between correct class and nearest wrong class
  R3 (quality): rule-based length/repetition checks
"""
import argparse
import json
import os

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

def compute_R1(text_emb, image_emb):
    """Alignment reward: cosine similarity between generated text and video."""
    return F.cosine_similarity(text_emb, image_emb, dim=-1).item()


def compute_R2(text_emb, image_emb_correct, mean_image_embs, correct_idx):
    """Discrimination reward: margin between correct class and nearest wrong class."""
    sims = F.cosine_similarity(text_emb, mean_image_embs, dim=-1)  # [num_classes]
    correct_sim = sims[correct_idx].item()
    # Mask correct class, find max of wrong classes
    sims[correct_idx] = -1.0
    max_wrong_sim = sims.max().item()
    return correct_sim - max_wrong_sim


def compute_R3(description, tokenizer, min_len=10, max_len=80):
    """Quality reward: rule-based checks on generation quality."""
    tokens = tokenizer.encode(description)
    score = 1.0

    # Length check
    if len(tokens) < min_len or len(tokens) > max_len:
        score -= 0.5

    # Repetition check
    words = description.lower().split()
    if len(words) > 0 and len(set(words)) / len(words) < 0.5:
        score -= 0.5

    return max(score, 0.0)


def compute_reward(text_emb, image_emb, mean_image_embs, correct_idx,
                   description, tokenizer, alpha=1.0, beta=0.5, gamma=0.2):
    """Total verifiable reward."""
    r1 = compute_R1(text_emb, image_emb)
    r2 = compute_R2(text_emb, image_emb, mean_image_embs, correct_idx)
    r3 = compute_R3(description, tokenizer)
    total = alpha * r1 + beta * r2 + gamma * r3
    return total, r1, r2, r3


# ============================================================
# Pre-encoding
# ============================================================

def pre_encode_videos(reward_model, dataloader, device):
    """Pre-encode all training videos with frozen visual encoder.
    Returns: image_embs [N, D], labels [N] (class indices)."""
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

            # Get class index from one-hot label
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
# GRPO Core
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
        logits = outputs.logits  # [1, seq_len, vocab]

    # Shift: predict next token
    shift_logits = logits[:, prompt_len-1:-1, :]  # [1, desc_len, vocab]
    shift_labels = full_enc["input_ids"][:, prompt_len:]  # [1, desc_len]

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, desc_len]

    return token_log_probs.sum()  # scalar: total log prob of description


def grpo_step(llm, llm_ref_forward, tokenizer, reward_model, text_encoder_tokenizer,
              image_emb, label_name, correct_idx, mean_image_embs,
              device, G=8, temperature=0.8, max_new_tokens=80,
              epsilon=0.2, beta_kl=0.04):
    """One GRPO step for a single sample.

    Args:
        llm: current policy (trainable LoRA)
        llm_ref_forward: function to compute log_probs with reference policy
        tokenizer: LLM tokenizer
        reward_model: frozen VideoCLIPModel for encoding text
        text_encoder_tokenizer: CFG.tokenizer for encoding descriptions
        image_emb: [1, D] pre-cached image embedding
        label_name: class label string
        correct_idx: class index
        mean_image_embs: [num_classes, D] mean embeddings per class
        device: cuda device
        G: group size
        temperature: sampling temperature
        max_new_tokens: max generation length
        epsilon: clip range
        beta_kl: KL penalty coefficient

    Returns:
        loss: scalar GRPO loss
        mean_reward: average reward across group
        rewards_dict: dict with r1, r2, r3 averages
    """
    prompt = PROMPT_TEMPLATE.format(label_name)
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # 1. Sample G descriptions
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

    # 2. Compute rewards
    rewards = []
    r1s, r2s, r3s = [], [], []
    text_embs = []
    reward_model.eval()
    with torch.no_grad():
        for desc in descriptions:
            # Encode description through reward model's text encoder
            desc_tokens = text_encoder_tokenizer(
                desc, return_tensors="pt", padding=True, truncation=True, max_length=CFG.max_length
            ).to(device)
            text_emb = reward_model.encode_text(
                input_ids=desc_tokens["input_ids"],
                attention_mask=desc_tokens["attention_mask"],
            )
            text_emb = F.normalize(text_emb, dim=-1)
            text_embs.append(text_emb)

            r, r1, r2, r3 = compute_reward(
                text_emb, image_emb.to(device), mean_image_embs.to(device),
                correct_idx, desc, tokenizer,
            )
            rewards.append(r)
            r1s.append(r1)
            r2s.append(r2)
            r3s.append(r3)

    rewards = torch.tensor(rewards, device=device)

    # 3. Group normalize → advantages
    if rewards.std() > 1e-8:
        advantages = (rewards - rewards.mean()) / rewards.std()
    else:
        advantages = torch.zeros_like(rewards)

    # 4. Compute log probs and GRPO loss
    llm.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for g in range(G):
        if len(descriptions[g].strip()) == 0:
            continue

        log_prob_theta = compute_log_probs(llm, tokenizer, prompt, descriptions[g], device)

        # Reference log prob (disable LoRA)
        llm.disable_adapter_layers()
        with torch.no_grad():
            log_prob_ref = compute_log_probs(llm, tokenizer, prompt, descriptions[g], device)
        llm.enable_adapter_layers()

        # Policy ratio
        ratio = torch.exp(log_prob_theta - log_prob_ref.detach())

        # Clipped surrogate
        A = advantages[g].detach()
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A
        policy_loss = -torch.min(surr1, surr2)

        # KL penalty
        kl = log_prob_theta - log_prob_ref.detach()

        total_loss = total_loss + policy_loss + beta_kl * kl

    total_loss = total_loss / max(G, 1)

    return total_loss, rewards.mean().item(), {
        "r1": np.mean(r1s), "r2": np.mean(r2s), "r3": np.mean(r3s),
    }, descriptions[rewards.argmax().item()]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True, help="Frozen baseline model .pt")
    parser.add_argument("--sft_model_path", type=str, required=True, help="SFT LoRA adapter directory")
    parser.add_argument("--llm_base", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--G", type=int, default=8, help="Group size")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
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
    image_embs, label_indices, label_names = pre_encode_videos(reward_model, train_loader, device)
    mean_image_embs = compute_mean_class_embs(image_embs, label_indices, num_classes)
    print(f"Cached {len(image_embs)} image embeddings, {num_classes} classes")

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in llm.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Text encoder tokenizer (for encoding generated descriptions through reward model)
    text_encoder_tokenizer = CFG.tokenizer

    print(f"\nGRPO Training: {args.epochs} epochs, G={args.G}, lr={args.lr}")
    print(f"Samples: {len(image_embs)}, Classes: {num_classes}\n")

    global_step = 0
    best_reward = -float("inf")

    for epoch in range(args.epochs):
        # Shuffle sample order
        perm = torch.randperm(len(image_embs))
        epoch_rewards = []
        epoch_r1s, epoch_r2s, epoch_r3s = [], [], []

        pbar = tqdm(range(len(perm)), desc=f"Epoch {epoch+1}/{args.epochs}")
        for i in pbar:
            idx = perm[i].item()
            image_emb = image_embs[idx:idx+1]  # [1, D]
            label_name = label_names[idx]
            correct_idx = labels.index(label_name)

            loss, mean_reward, reward_dict, best_desc = grpo_step(
                llm, None, llm_tokenizer, reward_model, text_encoder_tokenizer,
                image_emb, label_name, correct_idx, mean_image_embs,
                device, G=args.G, temperature=args.temperature,
                epsilon=args.epsilon, beta_kl=args.beta_kl,
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
                R2=f"{reward_dict['r2']:.3f}",
                loss=f"{loss.item():.4f}",
            )

            global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("reward/total", mean_reward, global_step)
                writer.add_scalar("reward/R1_align", reward_dict["r1"], global_step)
                writer.add_scalar("reward/R2_discrim", reward_dict["r2"], global_step)
                writer.add_scalar("reward/R3_quality", reward_dict["r3"], global_step)
                writer.add_scalar("loss", loss.item(), global_step)

        # Epoch summary
        avg_reward = np.mean(epoch_rewards)
        print(f"Epoch {epoch+1}: avg_reward={avg_reward:.4f}, "
              f"R1={np.mean(epoch_r1s):.4f}, R2={np.mean(epoch_r2s):.4f}, R3={np.mean(epoch_r3s):.4f}")

        # Generate sample descriptions
        print("Sample descriptions (best in group):")
        for label in labels[:3]:
            prompt = PROMPT_TEMPLATE.format(label)
            inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
            llm.eval()
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
            save_path = f"grpo_{dataset_name}_{args.llm_base.split('/')[-1]}"
            llm.save_pretrained(save_path)
            llm_tokenizer.save_pretrained(save_path)
            print(f"Saved best model (avg_reward={avg_reward:.4f})")

    writer.close()
    print(f"\nDone! Best avg reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
