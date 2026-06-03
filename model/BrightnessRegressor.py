import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear(x)
        out = self.bn(out)
        out = self.gelu(out)
        out = self.dropout(out)
        return out + residual


class BrightnessRegressor(nn.Module):
    def __init__(self, embed_dim=12800):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.resblock0 = ResidualBlock(4096, 4096, dropout=0.2)
        self.resblock1 = ResidualBlock(4096, 512, dropout=0.2)
        self.resblock2 = ResidualBlock(512, 256, dropout=0.1)
        self.resblock3 = ResidualBlock(256, 128, dropout=0.1)
        self.resblock4 = ResidualBlock(128, 64)

        self.output_proj = nn.Linear(64, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.resblock0(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        return self.output_proj(x).squeeze(-1)
