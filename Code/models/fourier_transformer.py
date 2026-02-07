import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierMixing(nn.Module):
    def forward(self, x):
        # x: (B, C, D)
        x_fft = torch.fft.fft(x, dim=1)
        x_fft = torch.fft.fft(x_fft, dim=2)
        return x_fft.real
class ContextFourierTransformer(nn.Module):
    def __init__(self, embed_dim=300, dropout=0.3):
        super().__init__()
        self.fourier = FourierMixing()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, C, D)
        """
        x = self.fourier(x)
        x = self.layer_norm(x + self.ffn(x))
        return x
