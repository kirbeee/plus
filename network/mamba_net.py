import torch
import torch.nn as nn
import torch.nn.functional as F
# 假設 WarpedSS2D 定義在 mamba_logits.py 裡
from mamba_logits import WarpedSS2D


class Mamba_Hash_Net(nn.Module):
    def __init__(self, in_channels=1, embedding_size=1024, drop_rate=0.1):
        super().__init__()

        # 1. Stem Conv + DWConv (把單通道影像轉換成 Mamba 需要的通道數，假設為 64)
        d_model = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model)  # DWConv
        )

        # 2. Hybrid Local-Mamba Stage ×4[cite: 1]
        # index 參數可以用來控制 WarpedSS2D 裡面的池化行為[cite: 2]
        self.stages = nn.Sequential(
            WarpedSS2D(d_model=d_model, index=0, drop_rate=drop_rate),
            WarpedSS2D(d_model=d_model, index=1, drop_rate=drop_rate),
            WarpedSS2D(d_model=d_model, index=2, drop_rate=drop_rate),
            WarpedSS2D(d_model=d_model, index=3, drop_rate=drop_rate)
        )

        # 3. Global Average Pooling[cite: 1]
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 4. 1024-d Identity Embedding + BatchNorm[cite: 1]
        self.embedding = nn.Linear(d_model, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # 輸入 x 的形狀: [B, C, H, W]
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x)  # 變成 [B, d_model, 1, 1]
        x = torch.flatten(x, 1)  # 展平為 [B, d_model]

        # 映射到 1024 維度
        x = self.embedding(x)
        x = self.bn(x)

        # L2 Norm[cite: 1]
        x = F.normalize(x, p=2, dim=1)
        return x