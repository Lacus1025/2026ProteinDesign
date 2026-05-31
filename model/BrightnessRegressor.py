import torch
import torch.nn as nn


class BrightnessRegressor(nn.Module):
    def __init__(self, embed_dim=2560):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)
