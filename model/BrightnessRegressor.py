import torch.nn as nn

class BrightnessRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            # 第一层：降维
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),

            # 第二层：学习高阶特征
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),

            # 第三层：精炼
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            # 第四层：输出
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.proj = nn.Linear(input_dim, 1)

    def forward(self, x):
        main_out = self.network(x).squeeze(-1)
        res_out = self.proj(x).squeeze(-1)
        return main_out + res_out
