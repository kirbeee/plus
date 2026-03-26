import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import configs
import datasets
from torchkit.head.localfc.arcface import ArcFace
from model.model import MinusBackbone, MinusLoss

def train():
    # --- hyperparameter ---
    configs.setup_seed(args.seed)

    # --- data loading ---
    train_dataset = datasets.ImagesDataset(args=args, data_type='LED', phase='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True, drop_last=True )
    num_classes =len(set(item['label'] for item in train_dataset.data))
    print(f"總訓練樣本數: {len(train_dataset)}, 總類別數: {num_classes}")

    # --- model initialize ---
    model = MinusBackbone(mode=args.mode).to(args.device)
    arcface_head = ArcFace(in_features=512,scale=32, out_features=num_classes).to(args.device)

    if args.mode == 'stage2':
        # 1. load Stage 1 Generator weights
        if os.path.exists('weights/best_generator.pth'):
            model.generator.load_state_dict(torch.load('weights/best_generator.pth'))
            print("Successfully loaded pre-trained Generator from Stage 1.")
        else:
            print("Warning: No Stage 1 weights found! Stage 2 will start from scratch (Not Recommended).")

        # 3. 凍結生成器 (Stage 2 核心要求)
        for param in model.generator.parameters():
            param.requires_grad = False
        model.generator.eval()  # 確保 BatchNorm/Dropout 狀態固定

    # --- Optimizer ---

    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr,'weight_decay': args.weight_decay},
        {'params': arcface_head.parameters(), 'lr':args.lr, 'weight_decay': args.weight_decay}
    ])

    criterion = MinusLoss(mode=args.mode).to(args.device)

    # --- 訓練迴圈 ---
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        arcface_head.train()

        total_loss = 0.0
        total_loss_gen = 0.0
        total_loss_fr = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            x_encode, x_residue, x_feature, x_latent = model(imgs)

            # 4. 辨識模型特徵提取與 ArcFace 計算
            outputs = arcface_head(x_feature, labels)

            loss, loss_gen, loss_fr, loss_ls = criterion(imgs, x_encode, x_latent, outputs[0], labels)

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

        # save model weights
        avg_loss = total_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"--> find new Loss ({best_loss:.4f}) Saving...")
            os.makedirs('weights', exist_ok=True)
            torch.save(arcface_head.state_dict(), 'weights/best_arcface_head.pth')
            torch.save(model.generator.state_dict(), 'weights/best_generator.pth')
            torch.save(model.recognizer.state_dict(), 'weights/best_recognizer.pth')


if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    epochs = 25
    args.mode = "stage1"
    train()
    epochs = 30
    args.mode = "stage2"
    train()