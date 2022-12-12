import math
import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer
from .feed_forward import FeedForward, Prediction
from .layer_norm import LayerNorm
from .multi_head_attention import MultiHeadAttention
from .utils import clones

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)

        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerEncoder(nn.Module):
    """The encoder of transformer

    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
        `dropout`: dropout rate, default 0.1
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1,
                 dropout: float = 0.1, max_len: int = 100):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.encoder_layer = EncoderLayer(d_model, self.multi_headed_attention, self.feed_forward, dropout)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.prediction = Prediction(d_model, dropout)
        self.position_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        last_hidden_states = self.encoder(self.position_encoding(x).to("cuda"), mask)
        averaged_last_hidden_states = torch.mean(last_hidden_states, dim=1)
        # output = self.prediction(averaged_last_hidden_states)
        
        return averaged_last_hidden_states 
