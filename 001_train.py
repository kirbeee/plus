import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

import configs
import datasets
from network.fsb_hash_net import FSB_Hash_Net, Hash_Generator
from network.logits import ArcFace


def train(args):
    # --- data loading ---
    train_dataset = datasets.ImagesDataset(args=args, data_type='LED', phase='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )
    num_classes = len(set(item['label'] for item in train_dataset.data))
    print(f"總訓練樣本數: {len(train_dataset)}, 總類別數: {num_classes}")

    # --- model initialize ---
    # 1. 初始化特徵提取器 (Feature Extractor) 與雜湊生成器 (Hash Generator)
    feature_extractor = FSB_Hash_Net(embedding_size=args.dim, do_prob=args.dropout).to(args.device)
    generator = Hash_Generator(embedding_size=args.dim, do_prob=args.dropout, device=args.device,
                               out_embedding_size=args.hash_dim).to(args.device)

    # 2. 初始化兩個 ArcFace 分類頭 (一個接特徵、一個接雜湊碼)
    feat_fc = ArcFace(in_features=args.dim, out_features=num_classes, s=64.0, m=0.35, device=args.device).to(
        args.device)
    hash_fc = ArcFace(in_features=args.hash_dim, out_features=num_classes, s=128.0, m=0.35, device=args.device).to(
        args.device)

    # --- Optimizer & Criterion ---
    # 針對不同的網路元件設定學習率 (FC 層通常需要較大的學習率)
    optimizer = optim.AdamW([
        {'params': feature_extractor.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': generator.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': feat_fc.parameters(), 'lr': args.lr * 10, 'weight_decay': args.weight_decay},
        {'params': hash_fc.parameters(), 'lr': args.lr * 10, 'weight_decay': args.weight_decay}
    ])

    # 使用標準的 CrossEntropyLoss 來處理 ArcFace 的輸出
    criterion = nn.CrossEntropyLoss().to(args.device)

    # --- 訓練迴圈 ---
    best_loss = float('inf')
    for epoch in range(args.epochs):
        feature_extractor.train()
        generator.train()
        feat_fc.train()
        hash_fc.train()

        total_loss = 0.0
        total_loss_feat = 0.0
        total_loss_hash = 0.0

        prefetcher = datasets.data_prefetcher(train_loader)
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}")

        imgs, labels = prefetcher.next()
        while imgs is not None:
            # 確保資料在正確的裝置上
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()

            # 3. 模型前向傳播：提取特徵與雜湊碼
            features = feature_extractor(imgs)
            hash_codes = generator(features, labels, training=True)

            # 4. 通過 ArcFace 分類頭計算 Logits
            logits_feat = feat_fc(features, labels)
            logits_hash = hash_fc(hash_codes, labels)

            # 5. 計算損失 (Feature Loss + Hash Loss)
            loss_feat = criterion(logits_feat, labels)
            loss_hash = criterion(logits_hash, labels)
            loss = loss_feat + loss_hash

            # 6. 反向傳播與參數更新
            loss.backward()
            optimizer.step()

            # 記錄 Loss
            total_loss += loss.item()
            total_loss_feat += loss_feat.item()
            total_loss_hash += loss_hash.item()

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'L_feat': f"{loss_feat.item():.4f}",
                'L_hash': f"{loss_hash.item():.4f}"
            })
            pbar.update(1)

            imgs, labels = prefetcher.next()

        pbar.close()

        # 每個 Epoch 結束後顯示平均 Loss
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Avg Loss: {avg_loss:.4f} "
              f"| Avg L_feat: {total_loss_feat / len(train_loader):.4f} "
              f"| Avg L_hash: {total_loss_hash / len(train_loader):.4f}")

        # save model weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"--> find new Loss ({best_loss:.4f}) Saving...")
            save_dir = 'weights/fsb_hashnet'
            os.makedirs(save_dir, exist_ok=True)

            torch.save(feature_extractor.state_dict(), os.path.join(save_dir, 'best_feature_extractor.pth'))
            torch.save(generator.state_dict(), os.path.join(save_dir, 'best_generator.pth'))
            torch.save(feat_fc.state_dict(), os.path.join(save_dir, 'best_feat_fc.pth'))
            torch.save(hash_fc.state_dict(), os.path.join(save_dir, 'best_hash_fc.pth'))


if __name__ == '__main__':
    args = configs.get_all_params()

    # --- 加入原本 params.py 中的網路結構超參數 ---
    args.dim = 1024
    args.hash_dim = 512
    args.dropout = 0.1

    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)

    # 開始訓練
    train(args)