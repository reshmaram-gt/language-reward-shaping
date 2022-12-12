import os
import re
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class VideoLanguageDataset(Dataset):
    def __init__(self, mode="train", use_video_embedding=True):
        self.mode = mode
        self.use_video_embedding = use_video_embedding
        self.train_frames = {275, 281, 283, 324, 546, 559, 563, 658, 1199, 1256, 1340, 1381, 1383, 1431, 1458}
        self.dev_frames = {249, 300, 958, 1289, 1473}

        with open(os.path.join("data", "atari-lang", "annotations.txt")) as f:
            self.frames_language_pairs = f.readlines()

        self.frames_language_pairs_train = []
        self.frames_language_pairs_dev = []

        for p in self.frames_language_pairs:
            if int(p.split("\t")[0].split("/")[0]) in self.dev_frames:
                self.frames_language_pairs_dev.append(p + "\t1")
            else:
                self.frames_language_pairs_train.append(p + "\t1")
        
        if mode == "train":
            self.frames_language_pairs = self.frames_language_pairs_train
        else:
            self.frames_language_pairs = self.frames_language_pairs_dev
        
        # sample negative examples
        np.random.seed(24)
        self.frames_language_pairs_neg = []
        
        language_per_frame = dict()
        for p in self.frames_language_pairs:
            video = p.split("\t")[0]
            frame_address = int(video.split("/")[0])
            if frame_address not in language_per_frame.keys():
                language_per_frame[frame_address] = list()
            language_per_frame[frame_address].append(p.split("\t")[1].strip())

        
        for p in self.frames_language_pairs:
            video = p.split("\t")[0]
            language_to_choose_from = language_per_frame[int(video.split("/")[0])]
            orig_idx = language_to_choose_from.index(p.split("\t")[1].strip())
            random_idx = int(np.random.choice(np.concatenate((np.arange(0, orig_idx), np.arange(orig_idx + 1, len(language_to_choose_from)))), size=1))
            negative_example = language_to_choose_from[random_idx]
            self.frames_language_pairs_neg.append(f"{video}\t{negative_example}\t0")

        self.frames_language_pairs.extend(self.frames_language_pairs_neg)

    def __len__(self):
        return len(self.frames_language_pairs)

    def __getitem__(self, idx):
        regex_result = re.findall("[0-9]+", self.frames_language_pairs[idx].split("\t")[0])
        frames_address = regex_result[0]
        frame_idx_start = int(regex_result[1])
        frame_idx_end = int(regex_result[2])

        language = self.frames_language_pairs[idx].split("\t")[1].strip()
        label = int(self.frames_language_pairs[idx].split("\t")[2])

        if not self.use_video_embedding:
            # retrieve frames from disk
            video = torch.ones(frame_idx_end - frame_idx_start + 1, 3, 210, 160)
            for frame_idx in range(frame_idx_start, frame_idx_end + 1):
                try:
                    img = torchvision.io.read_image(os.path.join("data", "atari-lang", frames_address, f"{frame_idx}.png"))
                except Exception as e:
                    print(e)
                    img = torchvision.io.read_image(os.path.join("data", "atari-lang", frames_address, f"{frame_idx - 1}.png"))
                video[frame_idx - frame_idx_start] = img

            
            data = {"video": video,
                    "language": language,
                    "frames_address": frames_address,
                    "frame_idx_start": frame_idx_start,
                    "frame_idx_end": frame_idx_end,
                    "label": label
            }

            return data
        else:
            embedding = torch.load(os.path.join("data", "embedding", f"{frames_address}-{frame_idx_start}-{frame_idx_end}.pt"))
            return {"video_embedding": embedding, "language": language, "label": label}

class VideoLanguageContrastiveDataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        self.train_frames = {275, 281, 283, 324, 546, 559, 563, 658, 1199, 1256, 1340, 1381, 1383, 1431, 1458}
        self.dev_frames = {249, 300, 958, 1289, 1473}

        with open(os.path.join("data", "atari-lang", "annotations.txt")) as f:
            self.frames_language_pairs = f.readlines()

        self.frames_language_pairs_train = []
        self.frames_language_pairs_dev = []

        for p in self.frames_language_pairs:
            if int(p.split("\t")[0].split("/")[0]) in self.dev_frames:
                self.frames_language_pairs_dev.append(p + "\t1")
            else:
                self.frames_language_pairs_train.append(p + "\t1")
        
        if mode == "train":
            self.frames_language_pairs = self.frames_language_pairs_train
        else:
            self.frames_language_pairs = self.frames_language_pairs_dev
        
        
        self.language_per_frame = dict()
        for p in self.frames_language_pairs:
            video = p.split("\t")[0]
            frame_address = int(video.split("/")[0])
            if frame_address not in self.language_per_frame.keys():
                self.language_per_frame[frame_address] = list()
            self.language_per_frame[frame_address].append(p.split("\t")[1].strip())

    def __len__(self):
        return len(self.frames_language_pairs)

        
    def __getitem__(self, idx):
        regex_result = re.findall("[0-9]+", self.frames_language_pairs[idx].split("\t")[0])
        frames_address = regex_result[0]
        frame_idx_start = int(regex_result[1])
        frame_idx_end = int(regex_result[2])

        postive_example = self.frames_language_pairs[idx].split("\t")[1].strip()
        video = self.frames_language_pairs[idx].split("\t")[0]
        language_to_choose_from = self.language_per_frame[int(video.split("/")[0])]
        orig_idx = language_to_choose_from.index(self.frames_language_pairs[idx].split("\t")[1].strip())
        random_idx = int(np.random.choice(np.concatenate((np.arange(0, orig_idx), np.arange(orig_idx + 1, len(language_to_choose_from)))), size=1))
        negative_example = language_to_choose_from[random_idx]
        negative_example1 = language_to_choose_from[random_idx - 1]

        embedding = torch.load(os.path.join("data", "embedding", f"{frames_address}-{frame_idx_start}-{frame_idx_end}.pt"))
        
        return {"video_embedding": embedding, "positive_example": postive_example, "negative_example": negative_example}

# HOW TO USE
if __name__ == "__main__":
    dataset = VideoLanguageDataset(mode="dev", use_video_embedding=True)
    dataloader = DataLoader(dataset=dataset, batch_size=5)

    for batch in dataloader:
        pass