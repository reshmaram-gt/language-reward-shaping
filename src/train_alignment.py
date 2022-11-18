from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from transformers import BertTokenizer, BertModel

from dataset import VideoLanguageDataset
from models import VideoLanguageAligner
from train_utils import compute_metrics

# device for training
device = "cuda"

dataset = VideoLanguageDataset(mode="train", use_video_embedding=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

with torch.no_grad():
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = BertModel.from_pretrained("bert-base-uncased").to(device)

model = VideoLanguageAligner(d_image_embedding=512, d_model=768, d_ff=768, n_heads=3, n_layers=4, max_len=500).to(device)
optimizer = Adam(params=model.parameters())

with tqdm(dataloader, unit="batch") as pbar:

    for batch in pbar:
        tokenied_text = text_tokenizer(batch["language"], padding=True, return_tensors="pt").to(device)
        text_encoding = text_encoder(**tokenied_text).last_hidden_state
        output = model(batch["video_embedding"].to(device), text_encoding.to(device))
        optimizer.zero_grad()
        loss = BCEWithLogitsLoss()(output, batch["label"].reshape(-1, 1).float().to(device))
        batch_accuray = torch.sum(torch.where(output > 0.5, 1, 0).reshape(-1).to(device) == batch["label"].reshape(-1).to(device)) / 64
        pbar.set_postfix(loss=loss.item(), batch_accuracy=batch_accuray.item())
        loss.backward()
        optimizer.step()

dev_dataset = VideoLanguageDataset(mode="dev", use_video_embedding=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=True)

print("Dev accuracy: ", compute_metrics(dev_dataloader, text_tokenizer, text_encoder, model, device))