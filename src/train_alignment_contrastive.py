from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from transformers import BertTokenizer, BertModel

from dataset import VideoLanguageContrastiveDataset
from models import VideoLanguageAligner
from loss import NCELoss
from train_utils import compute_metrics

# device for training
device = "cuda"
batch_size = 128
accum_iter = 10

dataset = VideoLanguageContrastiveDataset(mode="train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

with torch.no_grad():
    text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_encoder = BertModel.from_pretrained("bert-base-uncased").to(device)

model = VideoLanguageAligner(d_image_embedding=512, d_word_embedding=768, d_model=768, d_ff=512, n_heads=4, n_layers=3).to(device)
optimizer = Adam(params=model.parameters(), lr=1e-4)

model.load_state_dict(torch.load("weights/e50.pt"))

for epoch in range(50):
    print(f"Epoch: {epoch + 1}")
    correct = 0
    total_loss = 0
    with tqdm(dataloader, unit="batch") as pbar:
        for idx, batch in enumerate(pbar):
            with torch.no_grad():
                tokenied_pos = text_tokenizer(batch["positive_example"], padding=True, return_tensors="pt").to(device)
                pos_encoding = text_encoder(**tokenied_pos).last_hidden_state
                tokenied_neg = text_tokenizer(batch["negative_example"], padding=True, return_tensors="pt").to(device)
                neg_encoding = text_encoder(**tokenied_neg).last_hidden_state
                
            video_encoding = batch["video_embedding"][:, np.arange(0, 150, 10), :]
            pos_scores = model(video_encoding.to(device), pos_encoding.to(device))
            neg_scores = model(video_encoding.to(device), neg_encoding.to(device))
            # print(torch.sum(pos_scores > neg_scores) / 128)
            loss = NCELoss()(pos_scores, neg_scores) / accum_iter
            total_loss += loss * accum_iter
            pbar.set_postfix(loss=loss.item())
            loss.backward()
            
            if ((idx + 1) % accum_iter == 0) or (idx + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
    print(f"Loss: {total_loss / len(dataset)}")
    dev_dataset = VideoLanguageContrastiveDataset(mode="dev")
    dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=True)
    dev_loss = compute_metrics(dev_dataloader, text_tokenizer, text_encoder, model, NCELoss(), device)
    
    print("Dev loss: ", dev_loss)
    
    torch.save(model.state_dict(), f"weights/e{epoch + 1}.pt")