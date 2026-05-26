import torch.nn as nn

class BrightnessRegressor(nn.Module):
    def __init__(self, input_dim=1152):
        super().__init__()

        self.network = nn.Sequential(
            # 第一层：漏斗式降维 (1152 -> 512)
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),  # 第一层特征最多，加大一点 Dropout 防止死记硬背

            # 第二层：特征浓缩 (512 -> 128)
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),

            # 第三层：高阶组合 (128 -> 32)
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),

            # 输出层
            nn.Linear(32, 1)
        )

        # 残差连接
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, x):
        main_out = self.network(x).squeeze(-1)
        res_out = self.proj(x).squeeze(-1)

        # 主网络(拟合复杂非线性) + 旁路网络(提供稳健线性基线)
        return main_out + res_out
