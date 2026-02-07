import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictionHead(nn.Module):
    def __init__(self, embed_dim, num_emotions, num_intensity=4, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.emotion_head = nn.Linear(embed_dim, num_emotions)
        self.intensity_head = nn.Linear(embed_dim, num_intensity)

    def forward(self, x):
        """
        x: (B, D) → pooled context representation
        """
        x = self.fc(x)
        emotions = torch.sigmoid(self.emotion_head(x))   # multi-label
        intensity = F.log_softmax(self.intensity_head(x), dim=-1)
        return emotions, intensity
