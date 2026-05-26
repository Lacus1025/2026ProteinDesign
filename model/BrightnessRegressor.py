import torch.nn as nn

class BrightnessRegressor(nn.Module):
    def __init__(self, input_dim, seq_len=500):
        super().__init__()

        # 1D CNN 层（处理序列局部特征）
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1152, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
        )

        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # 残差连接
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: [batch, 1152] 或 [batch, 500, 1152]

        if x.dim() == 3:
            # 如果输入是 [batch, seq_len, features]，需要转置用于Conv1d
            x_cnn = x.permute(0, 2, 1)  # [batch, features, seq_len]
            cnn_out = self.cnn(x_cnn).squeeze(-1)  # [batch, 512]
        else:
            # 如果已经是聚合的特征，跳过CNN
            cnn_out = x

        mlp_out = self.mlp(cnn_out).squeeze(-1)
        residual_out = self.proj(x.mean(dim=1) if x.dim() == 3 else x).squeeze(-1)

        return mlp_out + residual_out
