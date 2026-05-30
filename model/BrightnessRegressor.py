import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    def __init__(self, embed_dim, num_heads=8, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn = nn.Linear(embed_dim, num_heads)
        self.value = nn.Linear(embed_dim, num_heads * head_dim)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        batch, seq_len, _ = x.shape

        attn = self.attn(x)                              # [batch, seq_len, num_heads]
        attn = attn.permute(0, 2, 1)                     # [batch, num_heads, seq_len]
        attn = F.softmax(attn, dim=-1)

        v = self.value(x)                                 # [batch, seq_len, num_heads*head_dim]
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)                        # [batch, num_heads, seq_len, head_dim]

        out = torch.einsum("bhs,bhsd->bhd", attn, v)     # [batch, num_heads, head_dim]
        out = out.flatten(1)                              # [batch, num_heads*head_dim]
        return out


class BrightnessRegressor(nn.Module):
    def __init__(self, seq_len=250, embed_dim=2560, num_heads=8, head_dim=64):
        super().__init__()

        self.attn_pool = AttentionPool1d(embed_dim, num_heads, head_dim)
        pool_dim = num_heads * head_dim

        self.fc = nn.Sequential(
            nn.Linear(pool_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        main_out = self.attn_pool(x)          # [batch, pool_dim]
        main_out = self.fc(main_out).squeeze(-1)
        res_out = self.proj(x.mean(dim=1)).squeeze(-1)
        return main_out + res_out
