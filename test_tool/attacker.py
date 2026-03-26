import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# 引入你的 UNet 作為攻擊者的還原模型
from model.utils import UNet


def denormalize(tensor):
    """將 [-1, 1] 的 Tensor 轉回 [0, 1] 的 Numpy 用於計算指標與存圖"""
    return ((tensor + 1.0) / 2.0).clamp(0, 1).cpu().numpy()


def train_attacker(args, model, train_loader, epochs=10, save_path='weights/attacker_unet.pth'):
    """訓練白箱攻擊者的還原模型 (U-Net)"""
    print("\n--- 準備訓練攻擊者模型 ---")
    attacker_model = UNet(in_channels=3, out_channels=3).to(args.device)
    optimizer = optim.Adam(attacker_model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()  # 論文提到使用 L1 距離作為最佳化目標

    # PPFR 系統模型不應該被更新
    model.eval()
    attacker_model.train()

    # 確保儲存權重的資料夾存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in tqdm(train_loader, desc=f"Attacker Training Epoch {epoch + 1}/{epochs}"):
            imgs = imgs.to(args.device)

            with torch.no_grad():
                # 攻擊者攔截保護表示 X_p (即 x_residue_shuffle)
                _, _, _, _, _, x_residue_up = model.obtain_residue(imgs)
                x_residue_shuffle = model.shuffle(x_residue_up)

            # 訓練攻擊模型：試圖從 X_p 還原回原始影像 X
            optimizer.zero_grad()
            recovered_imgs = attacker_model(x_residue_shuffle)

            loss = criterion(recovered_imgs, imgs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], L1 Loss: {epoch_loss / len(train_loader):.4f}")

    # 儲存攻擊者權重
    torch.save(attacker_model.state_dict(), save_path)
    print(f"攻擊者模型已儲存至 '{save_path}'")
    return attacker_model


def attack_evaluation(args, model, attacker_model, test_loader):
    """計算攻擊者重建影像的 PSNR 與 SSIM"""
    all_psnr = []
    all_ssim = []

    model.eval()
    attacker_model.eval()

    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Attacker Reconstruction Evaluation"):
            imgs = imgs.to(args.device)

            # 取得保護表示 X_p
            _, _, _, _, _, x_residue_up = model.obtain_residue(imgs)
            x_residue_shuffle = model.shuffle(x_residue_up)

            # 攻擊者進行還原
            recovered_imgs = attacker_model(x_residue_shuffle)

            # 轉回 [0, 1] 的 Numpy 並調整維度以計算指標 (B, H, W, C)
            orig_np = denormalize(imgs).transpose(0, 2, 3, 1)
            recons_np = denormalize(recovered_imgs).transpose(0, 2, 3, 1)

            for i in range(orig_np.shape[0]):
                p = psnr(orig_np[i], recons_np[i], data_range=1.0)
                s = ssim(orig_np[i], recons_np[i], data_range=1.0, channel_axis=-1)
                all_psnr.append(p)
                all_ssim.append(s)

    return np.mean(all_psnr), np.mean(all_ssim)