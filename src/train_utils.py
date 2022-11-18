import torch

def compute_metrics(dev_dataloader, text_tokenizer, text_encoder, model):
    with torch.no_grad():
        num_data_points = 0
        correct = 0
        for batch in dev_dataloader:
            tokenied_text = text_tokenizer(batch["language"], padding=True, return_tensors="pt").to("cuda")
            text_encoding = text_encoder(**tokenied_text).last_hidden_state
            output = model(batch["video_embedding"].to("cuda"), text_encoding.to("cuda"))

            num_data_points += len(batch["label"])
            correct += torch.sum(torch.where(output > 0.5, 1, 0).reshape(-1).to("cuda") == batch["label"].reshape(-1).to("cuda"))

        return {"accuracy": (correct / num_data_points).item()}