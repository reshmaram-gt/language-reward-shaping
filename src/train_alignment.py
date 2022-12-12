from tqdm import tqdm
import numpy as np
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
batch_size = 100
accum_iter = 10

dataset = VideoLanguageDataset(mode="train", use_video_embedding=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

with torch.no_grad():
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = BertModel.from_pretrained("bert-base-uncased").to(device)

model = VideoLanguageAligner(d_image_embedding=512, d_word_embedding=768, d_model=512, d_ff=512, n_heads=4, n_layers=3).to(device)
optimizer = Adam(params=model.parameters(), lr=1e-5)

for epoch in range(50):
    print(f"Epoch: {epoch + 1}")
    correct = 0
    with tqdm(dataloader, unit="batch") as pbar:
        for idx, batch in enumerate(pbar):
            with torch.no_grad():
                tokenied_text = text_tokenizer(batch["language"], padding=True, return_tensors="pt").to(device)
                text_encoding = text_encoder(**tokenied_text).last_hidden_state
            video_encoding = batch["video_embedding"][:, np.arange(0, 150, 10), :]
            output = model(video_encoding.to(device), text_encoding.to(device))
            # print(batch["label"])
            loss = BCEWithLogitsLoss()(output, batch["label"].reshape(-1, 1).float().to(device)) / accum_iter
            curr_correct = torch.sum(torch.where(output > 0, 1, 0).reshape(-1).to(device) == batch["label"].reshape(-1).to(device))
            correct += curr_correct
            batch_accuray = curr_correct / batch_size
            pbar.set_postfix(loss=loss.item(), batch_accuracy=batch_accuray.item())
            loss.backward()
            
            if ((idx + 1) % accum_iter == 0) or (idx + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
    print(f"Accuracy: {correct / len(dataset)}")
dev_dataset = VideoLanguageDataset(mode="dev", use_video_embedding=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=True)

print("Dev accuracy: ", compute_metrics(dev_dataloader, text_tokenizer, text_encoder, model, device))