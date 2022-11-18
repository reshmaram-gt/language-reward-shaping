import torch
import torch.nn as nn

from transformer import TransformerEncoder

class VideoLanguageAligner(nn.Module):
    def __init__(self, d_image_embedding: int, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1, 
                    dropout: float = 0.1, max_len: int = 100):
        
        super(VideoLanguageAligner, self).__init__()
        self.image_size_transformation = nn.Linear(d_image_embedding, d_model, bias=False)
        self.encoder = TransformerEncoder(d_model, d_ff, n_heads, n_layers, dropout, max_len)

    def forward(self, image_embeddings, text_embedding):
        transformed_image_embeddings = self.image_size_transformation(image_embeddings)
        concatenated_input = torch.concatenate((transformed_image_embeddings, text_embedding), dim=1)
        return self.encoder(concatenated_input, None)