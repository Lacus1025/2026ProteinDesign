import torch.nn as nn


class BrightnessRegressor(nn.Module):
    def __init__(self, seq_len=250, embed_dim=2560):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024,512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        # x: [batch, seq_len, embed_dim] -> [batch, embed_dim, seq_len]
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x).squeeze(-1)
        return x
