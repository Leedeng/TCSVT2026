import argparse
import itertools
import multiprocessing

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
from transformers import DistilBertTokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from config import CFG, AvgMeter, get_lr
from dataset import get_dataloader
from models import VideoCLIPModel
from module_utils.loss_utils import KLLoss


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, loss_fn, scaler):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        caption = batch["caption"]
        optimizer.zero_grad()

        with autocast("cuda"):
            image_embeddings = model.encode_image(batch["clip"].to(CFG.device))
            text_embeddings = model.encode_text(
                input_ids=batch["input_ids"].to(CFG.device),
                attention_mask=batch["attention_mask"].to(CFG.device),
            )
            image_embeddings, text_embeddings = model.apply_prompt(image_embeddings, text_embeddings)

            text_logits = (text_embeddings @ image_embeddings.T) / CFG.temperature
            image_logits = (image_embeddings @ text_embeddings.T) / CFG.temperature

            target = torch.tensor(
                [[1.0 if caption[i] == c else 0.0 for c in caption] for i in range(len(caption))],
                dtype=image_embeddings.dtype, device=CFG.device,
            )
            loss = (loss_fn(text_logits, target) + loss_fn(image_logits, target)) / 2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == "batch":
            lr_scheduler.step()

        loss_meter.update(loss.item(), len(caption))
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, valid_loader, test_loader, loss_fn, labels, label_tokens):
    loss_meter = AvgMeter()

    # --- Validation loss ---
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for batch in tqdm_object:
            caption = batch["caption"]
            image_embeddings = model.encode_image(batch["clip"].to(CFG.device))
            text_embeddings = model.encode_text(
                input_ids=batch["input_ids"].to(CFG.device),
                attention_mask=batch["attention_mask"].to(CFG.device),
            )
            image_embeddings, text_embeddings = model.apply_prompt(image_embeddings, text_embeddings)

            text_logits = (text_embeddings @ image_embeddings.T) / CFG.temperature
            image_logits = (image_embeddings @ text_embeddings.T) / CFG.temperature

            target = torch.tensor(
                [[1.0 if caption[i] == c else 0.0 for c in caption] for i in range(len(caption))],
                dtype=image_embeddings.dtype, device=CFG.device,
            )
            loss = (loss_fn(text_logits, target) + loss_fn(image_logits, target)) / 2
            loss_meter.update(loss.item(), batch["clip"].size(0))
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    # --- Test accuracy (zero-shot) ---
    acc_1, acc_5 = 0, 0
    total_number = len(test_loader)
    tqdm_object = tqdm(test_loader, total=total_number)

    with torch.no_grad():
        for batch in tqdm_object:
            caption = batch["caption"]
            image_embeddings = model.encode_image(batch["clip"].to(CFG.device))
            # Repeat image features for all labels
            image_embeddings_repeated = image_embeddings.repeat(len(labels), 1, 1, 1, 1) if image_embeddings.dim() == 5 else image_embeddings.repeat(len(labels), 1)

            # Encode all label texts
            text_embeddings = model.encode_text(
                input_ids=label_tokens["input_ids"].to(CFG.device),
                attention_mask=label_tokens["attention_mask"].to(CFG.device),
            )

            # For zero-shot: encode image once, repeat for each label
            image_features = model.image_encoder(batch["clip"].to(CFG.device))
            image_features = image_features.repeat(len(labels), 1, 1, 1, 1)
            image_features = image_features.squeeze(-1).squeeze(-1).squeeze(-1)
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(
                model.text_encoder(
                    input_ids=label_tokens["input_ids"].to(CFG.device),
                    attention_mask=label_tokens["attention_mask"].to(CFG.device),
                )
            )
            image_embeddings, text_embeddings = model.apply_prompt(image_embeddings, text_embeddings)

            dot_similarity = image_embeddings @ text_embeddings.T
            _, indices_pred = torch.topk(dot_similarity.squeeze(0), 5)
            indices_pred = indices_pred.detach().cpu().numpy()

            acc_1 += (labels[indices_pred[0][0]] == caption[0])
            for a in indices_pred[0]:
                acc_5 += (labels[a] == caption[0])

    print(f"acc@1 == {acc_1 / total_number * 100:.2f}%")
    print(f"acc@5 == {acc_5 / total_number * 100:.2f}%")
    return loss_meter, acc_1 / total_number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory, e.g. SMG or iMiGUE")
    parser.add_argument("--train_csv", type=str, default="clip.csv", help="Training CSV filename")
    parser.add_argument("--save_suffix", type=str, default=None, help="Model save suffix, e.g. SMG or iMiGUE")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory")
    args = parser.parse_args()

    dataset_name = args.dataset.rstrip("/")
    save_suffix = args.save_suffix or dataset_name
    log_dir = args.log_dir or f"./log/{dataset_name}_adaptiveprompting"
    writer = SummaryWriter(log_dir)

    # Load label names and prepare label tokens
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    labels = list(label_df["name"].values)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_labels = tokenizer(labels, padding=True, truncation=True, max_length=CFG.max_length)
    label_tokens = {key: torch.tensor(values) for key, values in encoded_labels.items()}

    loss_fn = KLLoss()
    train_loader = get_dataloader(dataset_name, mode="training", train_csv=args.train_csv)
    valid_loader = get_dataloader(dataset_name, mode="validation")
    test_loader = get_dataloader(dataset_name, mode="testing")

    model = VideoCLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(),
            model.text_projection.parameters(),
            model.prompt_generator.parameters(),
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay},
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
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch", loss_fn, scaler)
        writer.add_scalar("train_loss", train_loss.avg, epoch)

        model.eval()
        with torch.no_grad():
            valid_loss, acc = valid_epoch(model, valid_loader, test_loader, loss_fn, labels, label_tokens)
            writer.add_scalar("valid_loss", valid_loss.avg, epoch)
            writer.add_scalar("acc@1", acc, epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{round(best_acc, 2)}_{save_suffix}.pt")
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

    writer.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
