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

from model.utils import UNet, dct_transform
from model.model import MinusGenerativeModel


# ==========================================
# 1. 資料集準備 (Dataset)
# ==========================================
class PLUSVeinDataset(Dataset):
    def __init__(self, pkl_file, mode='train', sensor='LED', img_size=(256, 768)):
        super().__init__()
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # 讀取對應感測器與模式的資料集
        self.samples = data[sensor][f'{mode}_set']
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]['path']
        label = self.samples[idx]['label']

        # 讀取影像並轉換為 RGB
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 調整大小以確保可以被 DCT 區塊大小整除
        img = cv2.resize(img, self.img_size)

        # 轉換為 Tensor (C, H, W) 並正規化到 [-1, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        img_tensor = (img_tensor / 127.5) - 1.0

        return img_tensor, label


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
    batch_size = args.bat
    epochs = 50
    lr = args.lr
    alpha = 5.0  # L1 生成損失權重
    beta = 1.0  # ArcFace 辨識損失權重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    annot_file = 'annotations_plusvein.pkl'  # 請替換為 000_make_data_set.py 輸出的檔案名稱

    if not os.path.exists(annot_file):
        raise FileNotFoundError(f"找不到 {annot_file}，請先執行 000_make_data_set.py")

    # --- 資料加載 ---
    train_dataset = PLUSVeinDataset(pkl_file=annot_file, mode='train', sensor='LED')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 計算總類別數 (供 ArcFace 使用)
    num_classes = len(set([sample['label'] for sample in train_dataset.samples]))
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
            x_freq = dct_transform(imgs, ratio=1)

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


if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    train(args)