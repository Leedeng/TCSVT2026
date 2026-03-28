import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu

from config import CFG


def sample_frame_indices(clip_len, seg_len):
    boundaries = np.linspace(0, seg_len - 1, num=clip_len + 1)
    indices = [np.random.randint(int(boundaries[i]), int(boundaries[i + 1])) for i in range(len(boundaries) - 1)]
    return np.clip(indices, 0, seg_len).astype(np.int64)


class VideoTextDataset(Dataset):
    def __init__(self, file_dir, video_filenames, captions, mode="training",
                 label_names=None, descriptions=None):
        self.file_dir = file_dir
        self.video_filenames = list(video_filenames)
        self.captions = captions  # original short labels
        self.mode = mode
        self.label_names = label_names
        self.descriptions = descriptions  # {label: [desc1, desc2, ...]} or None

        # For testing/validation or when no descriptions: pre-tokenize short labels
        if descriptions is None or mode != "training":
            self.encoded_captions = CFG.tokenizer(
                list(self.captions), padding=True, truncation=True, max_length=CFG.max_length
            )
            self.use_dynamic_text = False
        else:
            self.encoded_captions = None
            self.use_dynamic_text = True

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, idx):
        label_name = self.captions[idx]

        # Text input: sample a random description during training, or use short label
        if self.use_dynamic_text:
            descs = self.descriptions.get(label_name)
            if descs:
                text = random.choice(descs)
            else:
                text = label_name
            encoded = CFG.tokenizer(
                text, padding="max_length", truncation=True, max_length=CFG.max_length,
                return_tensors="pt",
            )
            item = {key: val.squeeze(0) for key, val in encoded.items()}
            item["caption"] = label_name  # always return original label for loss matching
        else:
            item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}
            item["caption"] = label_name

        video_path = self.file_dir + self.video_filenames[idx]
        vr = VideoReader(video_path, ctx=cpu(0), width=CFG.width, height=CFG.height)

        if len(vr) > CFG.frame + 1:
            indices = sample_frame_indices(clip_len=CFG.frame, seg_len=len(vr))
            clip = vr.get_batch(indices).asnumpy()
        else:
            sampled = np.sort(np.random.choice(len(vr), CFG.frame, replace=True))
            clip = vr.get_batch(sampled).asnumpy()

        item["clip"] = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2)

        if self.label_names is not None:
            label = np.zeros(len(self.label_names))
            label[self.label_names.index(label_name)] = 1.0
            item["label"] = label

        return item


def get_dataloader(dataset_name, mode, train_csv="clip.csv", label_names=None,
                   use_descriptions=False, desc_file=None):
    if mode == "training":
        file_dir = f"{dataset_name}/training_clips/"
        clip_df = pd.read_csv(file_dir + train_csv)
    else:
        file_dir = f"{dataset_name}/testing_clips/"
        clip_df = pd.read_csv(file_dir + "clip.csv")

    # Load descriptions for training if requested
    descriptions = None
    if use_descriptions and mode == "training":
        desc_path = desc_file or f"descriptions/{dataset_name}_descriptions.json"
        with open(desc_path) as f:
            descriptions = json.load(f)
        print(f"Loaded {sum(len(v) for v in descriptions.values())} descriptions for {len(descriptions)} classes")

    dataset = VideoTextDataset(
        file_dir, clip_df["clip"].values, clip_df["caption"].values,
        mode=mode, label_names=label_names, descriptions=descriptions,
    )

    if mode == "training":
        return torch.utils.data.DataLoader(
            dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers,
            shuffle=True, drop_last=True,
        )
    if mode == "testing":
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=CFG.num_workers, shuffle=False,
        )
    return torch.utils.data.DataLoader(
        dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers,
        shuffle=False, drop_last=True,
    )
