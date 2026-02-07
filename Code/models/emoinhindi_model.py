import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_transformer import ContextAttentionTransformer
from models.fourier_transformer import ContextFourierTransformer
from models.prediction_head import PredictionHead
from models.positional_encoding import PositionalEncoding

class EmoInHindiModel(nn.Module):
    def __init__(
        self,
        embed_dim=300,
        num_emotions=16,
        model_type="fourier"  # "attention" or "fourier"
    ):
        super().__init__()

        if model_type == "attention":
            self.context_encoder = ContextAttentionTransformer(embed_dim)
        elif model_type == "fourier":
            self.context_encoder = ContextFourierTransformer(embed_dim)
        else:
            raise ValueError("model_type must be 'attention' or 'fourier'")

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.prediction_head = PredictionHead(embed_dim, num_emotions)

    def forward(self, x):
        """
        x: (B, C, D)
        """
        h = self.context_encoder(x)
        h = h.transpose(1, 2)          # (B, D, C)
        h = self.pooling(h).squeeze(-1)
        return self.prediction_head(h)
