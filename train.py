import argparse
import itertools
import multiprocessing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from config import CFG, AvgMeter, get_lr
from dataset import get_dataloader
from models import VideoCLIPModel
from module_utils.loss_utils import KLLoss


class AdaptiveLossWeighter(nn.Module):
    """Uncertainty-based adaptive loss weighting (Kendall et al., 2018).
    Learns log(sigma^2) for each task, effective weight = 1/(2*sigma^2)."""
    def __init__(self, num_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, *losses):
        total = 0
        for i, loss in enumerate(losses):
            total = total + 0.5 * torch.exp(-self.log_vars[i]) * loss + 0.5 * self.log_vars[i]
        return total

    def get_weights(self):
        with torch.no_grad():
            return [0.5 * torch.exp(-lv).item() for lv in self.log_vars]


def train_epoch(model, train_loader, optimizer, lr_scheduler, step,
                contrastive_loss_fn, cls_loss_fn, loss_weighter, scaler, use_tga=False):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        caption = batch["caption"]
        optimizer.zero_grad()

        with autocast("cuda"):
            image_emb, text_emb, cls_logits, tga_features, _ = model(
                clip=batch["clip"].to(CFG.device),
                input_ids=batch["input_ids"].to(CFG.device),
                attention_mask=batch["attention_mask"].to(CFG.device),
            )

            # Global contrastive loss
            text_logits = (text_emb @ image_emb.T) / CFG.temperature
            image_logits = (image_emb @ text_emb.T) / CFG.temperature
            target = torch.tensor(
                [[1.0 if caption[i] == c else 0.0 for c in caption] for i in range(len(caption))],
                dtype=image_emb.dtype, device=CFG.device,
            )
            loss_con_global = (contrastive_loss_fn(text_logits, target) + contrastive_loss_fn(image_logits, target)) / 2

            # Classification loss
            label = batch["label"].to(CFG.device)
            loss_cls = cls_loss_fn(cls_logits, label)

            # TGA contrastive loss
            if use_tga and tga_features is not None:
                tga_text_logits = (tga_features @ text_emb.T) / CFG.temperature
                tga_image_logits = (text_emb @ tga_features.T) / CFG.temperature
                loss_con_tga = (contrastive_loss_fn(tga_text_logits, target) + contrastive_loss_fn(tga_image_logits, target)) / 2
                loss = loss_weighter(loss_con_global, loss_con_tga, loss_cls)
            else:
                loss_con_tga = None
                loss = loss_weighter(loss_con_global, loss_cls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == "batch":
            lr_scheduler.step()

        weights = loss_weighter.get_weights()
        loss_meter.update(loss.item(), len(caption))
        postfix = dict(
            loss=loss_meter.avg,
            L_con=f"{loss_con_global.item():.3f}",
            L_cls=f"{loss_cls.item():.3f}",
            lr=get_lr(optimizer),
        )
        if loss_con_tga is not None:
            postfix["L_tga"] = f"{loss_con_tga.item():.3f}"
        tqdm_object.set_postfix(**postfix)

    return loss_meter


def valid_epoch(model, valid_loader, test_loader,
                contrastive_loss_fn, cls_loss_fn, loss_weighter, labels, label_tokens,
                use_tga=False):
    loss_meter = AvgMeter()

    # --- Validation loss ---
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for batch in tqdm_object:
            caption = batch["caption"]
            image_emb, text_emb, cls_logits, tga_features, _ = model(
                clip=batch["clip"].to(CFG.device),
                input_ids=batch["input_ids"].to(CFG.device),
                attention_mask=batch["attention_mask"].to(CFG.device),
            )

            text_logits = (text_emb @ image_emb.T) / CFG.temperature
            image_logits = (image_emb @ text_emb.T) / CFG.temperature
            target = torch.tensor(
                [[1.0 if caption[i] == c else 0.0 for c in caption] for i in range(len(caption))],
                dtype=image_emb.dtype, device=CFG.device,
            )
            loss_con_global = (contrastive_loss_fn(text_logits, target) + contrastive_loss_fn(image_logits, target)) / 2

            label = batch["label"].to(CFG.device)
            loss_cls = cls_loss_fn(cls_logits, label)

            if use_tga and tga_features is not None:
                tga_text_logits = (tga_features @ text_emb.T) / CFG.temperature
                tga_image_logits = (text_emb @ tga_features.T) / CFG.temperature
                loss_con_tga = (contrastive_loss_fn(tga_text_logits, target) + contrastive_loss_fn(tga_image_logits, target)) / 2
                loss = loss_weighter(loss_con_global, loss_con_tga, loss_cls)
            else:
                loss = loss_weighter(loss_con_global, loss_cls)

            loss_meter.update(loss.item(), batch["clip"].size(0))
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    # --- Pre-encode all label texts ---
    with torch.no_grad():
        label_text_emb, label_text_tokens = model.encode_text(
            input_ids=label_tokens["input_ids"].to(CFG.device),
            attention_mask=label_tokens["attention_mask"].to(CFG.device),
        )
        # Pre-compute projected tokens for TGA evaluation
        if use_tga:
            label_text_proj = model.text_token_proj(label_text_tokens)  # [N_classes, L, D]

    # --- Test accuracy ---
    acc_1_zs, acc_5_zs = 0, 0
    acc_1_cls, acc_5_cls = 0, 0
    acc_1_tga, acc_5_tga = 0, 0
    total_number = len(test_loader)
    num_classes = len(labels)
    tqdm_object = tqdm(test_loader, total=total_number)

    with torch.no_grad():
        for batch in tqdm_object:
            caption = batch["caption"]
            clip_tensor = batch["clip"].to(CFG.device)

            # encode_image returns (image_embeddings [B,D], raw_tokens [B,N,512])
            image_emb, visual_tokens = model.encode_image(clip_tensor)
            cls_logits = model.classifier(image_emb) if model.classifier is not None else None

            # --- Zero-shot accuracy (global contrastive) ---
            dot_similarity = image_emb @ label_text_emb.T
            _, indices_pred = torch.topk(dot_similarity, 5, dim=-1)
            indices_pred = indices_pred.detach().cpu().numpy()
            acc_1_zs += (labels[indices_pred[0][0]] == caption[0])
            for a in indices_pred[0]:
                acc_5_zs += (labels[a] == caption[0])

            # --- ZS+TGA accuracy ---
            if use_tga:
                proj_visual = model.visual_token_proj(visual_tokens)  # [1, N, D]
                tga_scores = []
                for c in range(num_classes):
                    text_tok_c = label_text_proj[c:c+1]  # [1, L, D]
                    v_hat_c, _ = model.tga(text_tok_c, proj_visual)  # [1, D]
                    score_c = F.cosine_similarity(v_hat_c, label_text_emb[c:c+1])
                    tga_scores.append(score_c.item())
                tga_scores = np.array(tga_scores)
                top1_tga = np.argmax(tga_scores)
                top5_tga = np.argpartition(tga_scores, len(tga_scores) - 5)[-5:]
                gt_idx = labels.index(caption[0])
                acc_1_tga += (gt_idx == top1_tga)
                acc_5_tga += (gt_idx in top5_tga)

            # --- Classifier accuracy ---
            logits_np = cls_logits.detach().cpu().numpy()
            target_np = batch["label"].detach().cpu().numpy()
            top1 = np.argmax(logits_np[0])
            top1_target = np.argmax(target_np[0])
            top5 = np.argpartition(logits_np[0], len(logits_np[0]) - 5)[-5:]
            acc_1_cls += (top1_target == top1)
            acc_5_cls += (top1_target in top5)

    print(f"Zero-shot  acc@1 == {acc_1_zs / total_number * 100:.2f}%  acc@5 == {acc_5_zs / total_number * 100:.2f}%")
    if use_tga:
        print(f"ZS+TGA     acc@1 == {acc_1_tga / total_number * 100:.2f}%  acc@5 == {acc_5_tga / total_number * 100:.2f}%")
    print(f"Classifier acc@1 == {acc_1_cls / total_number * 100:.2f}%  acc@5 == {acc_5_cls / total_number * 100:.2f}%")
    acc_tga_ratio = acc_1_tga / total_number if use_tga else 0
    return loss_meter, acc_1_zs / total_number, acc_1_cls / total_number, acc_tga_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory, e.g. SMG or iMiGUE")
    parser.add_argument("--train_csv", type=str, default="clip.csv", help="Training CSV filename")
    parser.add_argument("--save_suffix", type=str, default=None, help="Model save suffix")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--use_tga", action="store_true", help="Enable Text-Guided Attention")
    args = parser.parse_args()

    dataset_name = args.dataset.rstrip("/")
    tga_tag = "_tga" if args.use_tga else ""
    save_suffix = args.save_suffix or f"{dataset_name}{tga_tag}"
    log_dir = args.log_dir or f"./log/{dataset_name}_e2e{tga_tag}"
    writer = SummaryWriter(log_dir)

    # Load label names and prepare label tokens
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    num_classes = len(labels)
    encoded_labels = CFG.tokenizer(labels, padding=True, truncation=True, max_length=CFG.max_length)
    label_tokens = {key: torch.tensor(values) for key, values in encoded_labels.items()}

    contrastive_loss_fn = KLLoss()
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = get_dataloader(dataset_name, mode="training", train_csv=args.train_csv, label_names=labels)
    valid_loader = get_dataloader(dataset_name, mode="validation", label_names=labels)
    test_loader = get_dataloader(dataset_name, mode="testing", label_names=labels)

    model = VideoCLIPModel(num_classes=num_classes, use_tga=args.use_tga).to(CFG.device)
    num_tasks = 3 if args.use_tga else 2
    loss_weighter = AdaptiveLossWeighter(num_tasks=num_tasks).to(CFG.device)

    print(f"TGA: {'enabled' if args.use_tga else 'disabled'}")

    # Build param groups
    head_params = [
        model.image_projection.parameters(),
        model.text_projection.parameters(),
        model.classifier.parameters(),
    ]
    if args.use_tga:
        head_params.extend([
            model.visual_token_proj.parameters(),
            model.text_token_proj.parameters(),
            model.tga.parameters(),
        ])

    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(*head_params), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay},
        {"params": loss_weighter.parameters(), "lr": CFG.head_lr},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor,
    )
    scaler = GradScaler("cuda")

    best_acc = 0
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        loss_weighter.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch",
                                 contrastive_loss_fn, cls_loss_fn, loss_weighter, scaler,
                                 use_tga=args.use_tga)
        writer.add_scalar("train_loss", train_loss.avg, epoch)

        weights = loss_weighter.get_weights()
        if args.use_tga:
            w_con, w_tga, w_cls = weights
            writer.add_scalar("weight/contrastive", w_con, epoch)
            writer.add_scalar("weight/tga", w_tga, epoch)
            writer.add_scalar("weight/classifier", w_cls, epoch)
            gate_val = torch.sigmoid(model.tga.res_gate).item()
            writer.add_scalar("tga/res_gate", gate_val, epoch)
            print(f"Adaptive weights: w_con={w_con:.4f}, w_tga={w_tga:.4f}, w_cls={w_cls:.2f}, tga_gate={gate_val:.4f}")
        else:
            w_con, w_cls = weights
            writer.add_scalar("weight/contrastive", w_con, epoch)
            writer.add_scalar("weight/classifier", w_cls, epoch)
            print(f"Adaptive weights: w_con={w_con:.4f}, w_cls={w_cls:.2f}")

        model.eval()
        loss_weighter.eval()
        with torch.no_grad():
            valid_loss, acc_zs, acc_cls, acc_tga = valid_epoch(
                model, valid_loader, test_loader,
                contrastive_loss_fn, cls_loss_fn, loss_weighter, labels, label_tokens,
                use_tga=args.use_tga,
            )
            writer.add_scalar("valid_loss", valid_loss.avg, epoch)
            writer.add_scalar("acc@1_zeroshot", acc_zs, epoch)
            writer.add_scalar("acc@1_classifier", acc_cls, epoch)
            if args.use_tga:
                writer.add_scalar("acc@1_zs_tga", acc_tga, epoch)

        # Use the better accuracy for model selection
        acc = max(acc_zs, acc_cls, acc_tga) if args.use_tga else max(acc_zs, acc_cls)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{round(best_acc, 2)}_{save_suffix}.pt")
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

    writer.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
