import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=300, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        """
        x: (B, C, D) → context-level utterance embeddings
        """
        x = self.positional_encoding(x)
        return self.transformer(x)
