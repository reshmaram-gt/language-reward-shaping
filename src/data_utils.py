import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import DataLoader

from dataset import VideoLanguageDataset

def generate_embedding(batch_size, device="cpu"):
    os.makedirs(os.path.join("data", "embedding"), exist_ok=True)

    dataset = VideoLanguageDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*(list(resnet.children())[:-1])).to(device)

    for batch in tqdm(dataloader, total=6870 // batch_size):
        with torch.no_grad():
            batch_video = batch["video"]
            frames_address = batch["frames_address"]
            frame_idx_start = batch["frame_idx_start"]
            frame_idx_end = batch["frame_idx_end"]

            frame_embeddings = torch.zeros((batch_video.shape[0], batch_video.shape[1], 512)).to(device)
            for frame_idx in range(batch_video.shape[1]):
                embedding = feature_extractor(batch_video[:, frame_idx, :, :, :].to(device))
                frame_embeddings[:, frame_idx, :] = embedding.squeeze()
            
            for batch_idx in range(batch_video.shape[0]):
                torch.save(frame_embeddings[batch_idx],
                    os.path.join("data", "embedding", f"{frames_address[batch_idx]}-{frame_idx_start[batch_idx]}-{frame_idx_end[batch_idx]}.pt"))