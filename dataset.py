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
    def __init__(self, file_dir, video_filenames, captions, mode="training", label_names=None):
        self.file_dir = file_dir
        self.video_filenames = list(video_filenames)
        self.captions = captions
        self.mode = mode
        self.label_names = label_names
        self.encoded_captions = CFG.tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=CFG.max_length
        )

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}

        video_path = self.file_dir + self.video_filenames[idx]
        vr = VideoReader(video_path, ctx=cpu(0), width=CFG.width, height=CFG.height)

        if len(vr) > CFG.frame + 1:
            indices = sample_frame_indices(clip_len=CFG.frame, seg_len=len(vr))
            clip = vr.get_batch(indices).asnumpy()
        else:
            sampled = np.sort(np.random.choice(len(vr), CFG.frame, replace=True))
            clip = vr.get_batch([sampled]).asnumpy()

        item["clip"] = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2)
        item["caption"] = self.captions[idx]

        if self.label_names is not None:
            label = np.zeros(len(self.label_names))
            label[self.label_names.index(self.captions[idx])] = 1.0
            item["label"] = label

        return item


def get_dataloader(dataset_name, mode, train_csv="clip.csv", label_names=None):
    if mode == "training":
        file_dir = f"{dataset_name}/training_clips/"
        clip_df = pd.read_csv(file_dir + train_csv)
    else:
        file_dir = f"{dataset_name}/testing_clips/"
        clip_df = pd.read_csv(file_dir + "clip.csv")

    dataset = VideoTextDataset(
        file_dir, clip_df["clip"].values, clip_df["caption"].values,
        mode=mode, label_names=label_names,
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
