import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import configs
import datasets
from torchkit.head.localfc.arcface import ArcFace
from model.utils import  dct_transform
from model.model import MinusBackbone


# ==========================================
# 4. 主要訓練流程
# ==========================================
def train(args):
    """
    stage1
    stage2
    """

    # --- 超參數設定 ---
    configs.setup_seed(args.seed)
    epochs = 50
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
    generator = MinusBackbone(mode='stage1').to(device)

    # 2. 辨識模型 (LightCNN) -> 192 輸入通道
    # recognizer = LightVeinCNN(in_channels=192, embedding_size=512).to(device)

    # 3. ArcFace 分類頭
    arcface_head = ArcFace(in_features=512, out_features=num_classes).to(device)

    # --- 損失函數與優化器 ---
    criterion_gen = nn.L1Loss()
    criterion_fr = nn.CrossEntropyLoss()

    optimizer = optim.Adam([
        {'params': generator.parameters(), 'lr': args.lr,'weight_decay': args.weight_decay},
        {'params': arcface_head.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
    ])

    # --- 訓練迴圈 ---
    best_loss = float('inf')
    for epoch in range(epochs):
        generator.train()
        arcface_head.train()

        total_loss = 0.0
        total_loss_gen = 0.0
        total_loss_fr = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 2. 生成模型重建
            x_encode, x_residue, x_feature, x_latent = generator(imgs)

            x_freq = dct_transform(imgs, ratio=8)

            # 4. 辨識模型特徵提取與 ArcFace 計算
            outputs = arcface_head(x_feature, labels)

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
            torch.save(generator.generator.state_dict(), 'weights/best_generator.pth')
            torch.save(generator.recognizer.state_dict(), 'weights/best_recognizer.pth')


if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    train(args)