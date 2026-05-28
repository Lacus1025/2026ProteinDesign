import torch.nn as nn


class BrightnessRegressor(nn.Module):
    def __init__(self, seq_len=250, embed_dim=1152):
        super().__init__()

        self.conv_net = nn.Sequential(
            # [batch, 1152, 250] -> [batch, 512, 125]
            nn.Conv1d(embed_dim, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            # [batch, 512, 125] -> [batch, 256, 63]
            nn.Conv1d(512, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            # [batch, 256, 63] -> [batch, 128, 32]
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 32, 1024),
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
        x = self.conv_net(x)
        x = x.flatten(1)
        x = self.fc(x).squeeze(-1)
        return x
