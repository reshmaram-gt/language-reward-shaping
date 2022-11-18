import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from transformers import BertTokenizer, BertModel

from dataset import VideoLanguageDataset
from models import VideoLanguageAligner

dataset = VideoLanguageDataset(mode="train", use_video_embedding=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_encoder = BertModel.from_pretrained("bert-base-uncased").to("cuda")

model = VideoLanguageAligner(d_image_embedding=512, d_model=768, d_ff=768, n_heads=3, n_layers=4, max_len=500).to("cuda")
optimizer = Adam(params=model.parameters())

for batch in dataloader:
    tokenied_text = text_tokenizer(batch["language"], padding=True, return_tensors="pt").to("cuda")
    text_encoding = text_encoder(**tokenied_text).last_hidden_state
    output = model(batch["video_embedding"].to("cuda"), text_encoding.to("cuda"))
    optimizer.zero_grad()
    loss = BCEWithLogitsLoss()(output, batch["label"].reshape(-1, 1).float().to("cuda"))
    print(torch.sum(torch.where(output > 0.5, 1, 0).reshape(-1).to("cuda") == batch["label"].reshape(-1).to("cuda")) / 64)
    print(loss.item())
    loss.backward()
    optimizer.step()

dev_dataset = VideoLanguageDataset(mode="dev", use_video_embedding=True)
dev_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
with torch.no_grad():
    for batch in dev_dataloader:
        tokenied_text = text_tokenizer(batch["language"], padding=True, return_tensors="pt").to("cuda")
        text_encoding = text_encoder(**tokenied_text).last_hidden_state
        output = model(batch["video_embedding"].to("cuda"), text_encoding.to("cuda"))
        # optimizer.zero_grad()
        loss = BCEWithLogitsLoss()(output, batch["label"].reshape(-1, 1).float().to("cuda"))
        print(torch.sum(torch.where(output > 0.5, 1, 0).reshape(-1).to("cuda") == batch["label"].reshape(-1).to("cuda")) / 64)
        print(loss.item())
        # loss.backward()
        # optimizer.step()