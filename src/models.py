import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from transformer import TransformerEncoder

class VideoLanguageAligner(nn.Module):
    def __init__(self, d_image_embedding: int, d_word_embedding: int, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1, 
                    dropout: float = 0.1, max_len: int = 500):
        
        super(VideoLanguageAligner, self).__init__()
        # self.dummy_param = nn.Parameter(torch.empty(0))
        # pretrained_cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # self.image_feature_extractor = nn.Sequential(*(list(pretrained_cnn.children())[:-1]))
        self.image_size_transformation = nn.Sequential(nn.Linear(d_image_embedding, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.word_size_transformation = nn.Sequential(nn.Linear(d_word_embedding, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        
        self.video_transformer = TransformerEncoder(d_model, d_ff, n_heads, n_layers, dropout, max_len)
        self.language_transformer = TransformerEncoder(d_model, d_ff, n_heads, n_layers, dropout, max_len)
        self.prediction = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, frame_embeddings, text_embedding):
        # frame_embeddings = torch.zeros((video.shape[0], video.shape[1], 512)).to(self.dummy_param.device)
        # for frame_idx in range(0, video.shape[1], 10):
        #     image_embeddings = self.image_feature_extractor(video[:, frame_idx, :, :, :])
        #     frame_embeddings[:, frame_idx, :] = image_embeddings.squeeze()
        
        transformed_image_embeddings = self.image_size_transformation(frame_embeddings)
        transformed_text_embedding = self.word_size_transformation(text_embedding)
        
        video_embedding = self.video_transformer(transformed_image_embeddings, None)
        language_embedding = self.language_transformer(transformed_text_embedding, None)
        
        return torch.bmm(video_embedding.unsqueeze(1), language_embedding.unsqueeze(2))