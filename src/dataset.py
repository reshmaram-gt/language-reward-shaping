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
        

    def __len__(self):
        return len(self.frames_language_pairs)

    def __getitem__(self, idx):
        regex_result = re.findall("[0-9]+", self.frames_language_pairs[idx].split("\t")[0])
        frames_address = regex_result[0]
        frame_idx_start = int(regex_result[1])
        frame_idx_end = int(regex_result[2])

        language = self.frames_language_pairs[idx].split("\t")[1].strip()

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
            return {"video_embedding": embedding, "language": language}


# HOW TO USE
if __name__ == "__main__":
    dataset = VideoLanguageDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=5)

    for batch in dataloader:
        pass