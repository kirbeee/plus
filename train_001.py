import os
import pickle
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import configs
import datasets

from model.utils import UNet, dct_transform
from model.model import MinusGenerativeModel

# ==========================================
# 2. 輕量化辨識模型 (Lightweight CNN for Vein)
# ==========================================
class LightVeinCNN(nn.Module):
    """
    針對指靜脈特徵設計的輕量級 CNN，替換原本龐大的 IR-50。
    輸入通道設定為 192 以匹配 DCT 輸出。
    """

    def __init__(self, in_channels=192, embedding_size=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        )
        self.fc = nn.Linear(256, embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# ==========================================
# 3. ArcFace 損失函數 (Margin Penalty)
# ==========================================
class ArcFace(nn.Module):
    """
    實作 ArcFace 損失函數，用於增強身分特徵的類間分離度與類內緊湊度。
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # 餘弦相似度
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)

        # 加上 margin
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 建立 One-hot 標籤
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 僅在 Ground Truth 類別加上 margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# ==========================================
# 4. 主要訓練流程
# ==========================================
def train(args):
    # --- 超參數設定 ---
    configs.setup_seed(args.seed)
    batch_size = args.batch_size
    epochs = 50
    lr = args.lr
    alpha = 5.0  # L1 生成損失權重
    beta = 1.0  # ArcFace 辨識損失權重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 資料加載 ---
    train_dataset = datasets.ImagesDataset(args=args, data_type='LED', phase='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True, drop_last=True )

    # 計算總類別數 (供 ArcFace 使用)
    num_classes =len(set(item['label'] for item in train_dataset.data))
    print(f"總訓練樣本數: {len(train_dataset)}, 總類別數: {num_classes}")

    # --- 模型初始化 ---
    # 1. 生成模型 (U-Net) -> 根據 MinusGenerativeModel，會自動使用 UNet(192, 192)
    generator = MinusGenerativeModel(mode='stage1', backbone=UNet).to(device)

    # 2. 辨識模型 (LightCNN) -> 192 輸入通道
    recognizer = LightVeinCNN(in_channels=192, embedding_size=512).to(device)

    # 3. ArcFace 分類頭
    arcface_head = ArcFace(in_features=512, out_features=num_classes).to(device)

    # --- 損失函數與優化器 ---
    criterion_gen = nn.L1Loss()
    criterion_fr = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {'params': generator.parameters(), 'lr': lr,'weight_decay': args.weight_decay},
        {'params': recognizer.parameters(), 'lr': lr, 'weight_decay': args.weight_decay},
        {'params': arcface_head.parameters(), 'lr': lr, 'weight_decay': args.weight_decay}
    ])

    # --- 訓練迴圈 ---
    best_loss = float('inf')
    for epoch in range(epochs):
        generator.train()
        recognizer.train()
        arcface_head.train()

        total_loss = 0.0
        total_loss_gen = 0.0
        total_loss_fr = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 1. 轉換至頻域 (Spatial -> Frequency)
            # x_freq 維度: (B, 192, H_block, W_block)
            x_freq = dct_transform(imgs, ratio=8)

            # 2. 生成模型重建
            x_encode, x_latent = generator(x_freq)

            # 3. 高維度特徵相減 (Residue Calculation)
            # 公式: r = x - x'
            r = x_freq - x_encode

            # 4. 辨識模型特徵提取與 ArcFace 計算
            embeddings = recognizer(r)
            outputs = arcface_head(embeddings, labels)

            # 5. 計算損失函數
            # L_gen: 生成特徵必須逼近原始頻域特徵
            loss_gen = criterion_gen(x_encode, x_freq)
            # L_fr: 殘差必須能被辨識出正確的身分
            loss_fr = criterion_fr(outputs, labels)

            # L_minus = alpha * L_gen + beta * L_fr
            loss = alpha * loss_gen + beta * loss_fr

            # 6. 反向傳播與參數更新
            loss.backward()
            optimizer.step()

            # 記錄 Loss
            total_loss += loss.item()
            total_loss_gen += loss_gen.item()
            total_loss_fr += loss_fr.item()

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'L_gen': f"{loss_gen.item():.4f}",
                'L_fr': f"{loss_fr.item():.4f}"
            })

        print(f"Epoch [{epoch + 1}/{epochs}] - Avg Loss: {total_loss / len(train_loader):.4f} "
              f"| Avg L_gen: {total_loss_gen / len(train_loader):.4f} "
              f"| Avg L_fr: {total_loss_fr / len(train_loader):.4f}")

        # ==========================================
        # 新增：保存最佳模型權重
        # ==========================================
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"--> 發現更低的 Loss ({best_loss:.4f})，正在保存模型...")
            os.makedirs('weights', exist_ok=True)

            # 保存各個模組的權重
            torch.save(generator.state_dict(), 'weights/best_generator.pth')
            torch.save(recognizer.state_dict(), 'weights/best_recognizer.pth')
            torch.save(arcface_head.state_dict(), 'weights/best_arcface.pth')


if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    train(args)