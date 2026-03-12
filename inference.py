import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from config import CFG
from dataset import sample_frame_indices
from models import VideoCLIPModel, Classifier


def predict_video_class(video_path, model, classifier, labels):
    vr = VideoReader(video_path, ctx=cpu(0), width=CFG.width, height=CFG.height)

    if len(vr) > CFG.frame:
        indices = sample_frame_indices(CFG.frame, seg_len=len(vr))
    else:
        indices = np.sort(np.random.choice(len(vr), CFG.frame, replace=True))

    clip = vr.get_batch(indices).asnumpy()
    clip = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(CFG.device)

    with torch.no_grad():
        image_embeddings = model.encode_image(clip)
        logits = classifier(image_embeddings)
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()

    return labels[pred_idx], probs[0, pred_idx].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="VideoCLIP model checkpoint")
    parser.add_argument("--classifier_checkpoint", type=str, required=True, help="Classifier checkpoint")
    parser.add_argument("--label_csv", type=str, required=True, help="Path to Clip_label.csv")
    parser.add_argument("--gt_csv", type=str, default=None, help="Optional ground truth CSV")
    args = parser.parse_args()

    label_df = pd.read_csv(args.label_csv)
    labels = list(label_df["name"].values)
    num_classes = len(labels)

    model = VideoCLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=CFG.device))
    model.eval()

    classifier = Classifier(embedding_dim=CFG.projection_dim, num_classes=num_classes).to(CFG.device)
    classifier.load_state_dict(torch.load(args.classifier_checkpoint, map_location=CFG.device))
    classifier.eval()

    pred_label, confidence = predict_video_class(args.video, model, classifier, labels)
    print(f"Predicted: {pred_label} (confidence: {confidence:.4f})")

    if args.gt_csv:
        gt_df = pd.read_csv(args.gt_csv)
        video_name = os.path.basename(args.video)
        match = gt_df[gt_df["clip"] == video_name]
        if not match.empty:
            print(f"Ground Truth: {match.iloc[0]['caption']}")


if __name__ == "__main__":
    main()
