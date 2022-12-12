import torch
import numpy as np

def compute_metrics(dev_dataloader, text_tokenizer, text_encoder, model, loss, device):
    with torch.no_grad():
        total_loss = 0
        for batch in dev_dataloader:
            tokenied_pos = text_tokenizer(batch["positive_example"], padding=True, return_tensors="pt").to(device)
            pos_encoding = text_encoder(**tokenied_pos).last_hidden_state
            tokenied_neg = text_tokenizer(batch["negative_example"], padding=True, return_tensors="pt").to(device)
            neg_encoding = text_encoder(**tokenied_neg).last_hidden_state
            video_encoding = batch["video_embedding"][:, np.arange(0, 150, 10), :]
            pos_scores = model(video_encoding.to(device), pos_encoding.to(device))
            neg_scores = model(video_encoding.to(device), neg_encoding.to(device))
            total_loss += loss(pos_scores, neg_scores) / batch["video_embedding"].shape[0]

        return {"Loss": total_loss.item() / len(dev_dataloader)}