import argparse
import multiprocessing

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from config import CFG, AvgMeter, get_lr
from dataset import get_dataloader
from models import VideoCLIPModel, Classifier


def train_epoch(model, classifier, train_loader, optimizer, lr_scheduler, step, loss_fn, scaler):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        optimizer.zero_grad()
        with autocast("cuda"):
            image_embeddings = model.encode_image(batch["clip"].to(CFG.device))
            logits = classifier(image_embeddings)
            target = batch["label"].to(CFG.device)
            loss = loss_fn(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == "batch":
            lr_scheduler.step()

        loss_meter.update(loss.item(), len(batch["caption"]))
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, classifier, valid_loader, test_loader, loss_fn):
    # --- Test accuracy ---
    acc_1, acc_5 = 0, 0
    total_number = len(test_loader)
    tqdm_object = tqdm(test_loader, total=total_number)

    with torch.no_grad():
        for batch in tqdm_object:
            image_embeddings = model.encode_image(batch["clip"].to(CFG.device))
            logits = classifier(image_embeddings).detach().cpu().numpy()
            target = batch["label"].detach().cpu().numpy()

            top1 = np.argmax(logits[0])
            top1_target = np.argmax(target[0])
            top5 = np.argpartition(logits[0], len(logits[0]) - 5)[-5:]

            acc_1 += (top1_target == top1)
            acc_5 += (top1_target in top5)

    # --- Validation loss ---
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for batch in tqdm_object:
            image_embeddings = model.encode_image(batch["clip"].to(CFG.device))
            logits = classifier(image_embeddings)
            target = batch["label"].to(CFG.device)
            loss = loss_fn(logits, target)
            loss_meter.update(loss.item(), batch["clip"].size(0))
            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    print(f"acc@1 == {acc_1 / total_number * 100:.2f}%")
    print(f"acc@5 == {acc_5 / total_number * 100:.2f}%")
    return loss_meter, acc_1 / total_number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory, e.g. SMG")
    parser.add_argument("--checkpoint", type=str, required=True, help="Pretrained VideoCLIP model checkpoint")
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--save_suffix", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    dataset_name = args.dataset.rstrip("/")
    save_suffix = args.save_suffix or f"{dataset_name}_classifier_finetuned"
    log_dir = args.log_dir or f"./log/{dataset_name}_adaptiveprompting_finetuned"
    writer = SummaryWriter(log_dir)

    # Load labels
    label_df = pd.read_csv(f"{dataset_name}/Clip_label.csv")
    label_names = list(label_df["name"].values)
    num_classes = len(label_names)

    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = get_dataloader(dataset_name, mode="training", label_names=label_names)
    valid_loader = get_dataloader(dataset_name, mode="validation", label_names=label_names)
    test_loader = get_dataloader(dataset_name, mode="testing", label_names=label_names)

    # Load pretrained model and freeze
    model = VideoCLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=CFG.device))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    classifier = Classifier(embedding_dim=CFG.projection_dim, num_classes=num_classes).to(CFG.device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=CFG.head_lr, weight_decay=CFG.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience, factor=CFG.factor,
    )
    scaler = GradScaler("cuda")

    best_acc = 0
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        classifier.train()
        train_loss = train_epoch(model, classifier, train_loader, optimizer, lr_scheduler, "epoch", loss_fn, scaler)
        writer.add_scalar("train_loss", train_loss.avg, epoch)

        classifier.eval()
        with torch.no_grad():
            valid_loss, acc = valid_epoch(model, classifier, valid_loader, test_loader, loss_fn)
            writer.add_scalar("valid_loss", valid_loss.avg, epoch)
            writer.add_scalar("acc@1", acc, epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), f"{round(best_acc, 2)}_{save_suffix}.pt")
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

    writer.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
