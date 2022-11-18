import os
import re
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

class VideoLanguageDataset(Dataset):
    def __init__(self, mode="train", use_video_embedding=True):
        self.mode = mode
        self.use_video_embedding = use_video_embedding

        with open(os.path.join("data", "atari-lang", "annotations.txt")) as f:
            self.frames_language_pairs = f.readlines()

        self.frames_language_pairs_train = []
        self.frames_language_pairs_dev = []

        for p in self.frames_language_pairs:
            if p.split("\t")[0].split("/")[0] in {249, 300, 958, 1280, 1473}:
                self.frames_language_pairs_dev.append(p + "\t1")
            else:
                self.frames_language_pairs_train.append(p + "\t1")
        
        if mode == "train":
            self.frames_language_pairs = self.frames_language_pairs_train
        else:
            self.frames_language_pairs = self.frames_language_pairs_dev
        
        self.frames_language_pairs_neg = []
        for p in self.frames_language_pairs:
            video = p.split("\t")[0]
            language = self.frames_language_pairs[torch.randint(0, len(self.frames_language_pairs), size=(1,))].split("\t")[0].split("/")[1]
            self.frames_language_pairs_neg.append(f"{video}\t{language}\t0")

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
                    "frame_idx_end": frame_idx_end
            }

            return data
        else:
            embedding = torch.load(os.path.join("data", "embedding", f"{frames_address}-{frame_idx_start}-{frame_idx_end}.pt"))
            return {"video_embedding": embedding, "language": language, "label": label}


# HOW TO USE
if __name__ == "__main__":
    dataset = VideoLanguageDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=5)

    for batch in dataloader:
        pass