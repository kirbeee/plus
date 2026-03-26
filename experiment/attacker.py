import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
# 引入專案的模組
sys.path.append(str(Path(__file__).resolve().parent.parent))
import configs
import datasets
from model.model import MinusBackbone
from model.utils import UNet
from test_tool.attacker import denormalize


def load_models_for_vis(args, weights_dir='weights'):
    """載入 PPFR 模型 (Stage 2) 與 攻擊者模型用於視覺化"""
    print("正在載入模型權重進行視覺化展示...")

    # 1. 載入 PPFR 骨幹 (確保在 Stage 2 模式)
    model = MinusBackbone(mode="stage2").to(args.device)
    gen_path = os.path.join(weights_dir, 'best_generator.pth')
    rec_path = os.path.join(weights_dir, 'best_recognizer.pth')

    if not (os.path.exists(gen_path) and os.path.exists(rec_path)):
        raise FileNotFoundError(f"找不到 PPFR 系統權重於 {weights_dir}。請確保已完成系統訓練。")

    model.generator.load_state_dict(torch.load(gen_path, map_location=args.device))
    model.recognizer.load_state_dict(torch.load(rec_path, map_location=args.device))
    model.eval()

    # 2. 載入攻擊者 UNet
    attacker_model = UNet(in_channels=3, out_channels=3).to(args.device)
    attack_path = os.path.join(weights_dir, 'attacker_unet.pth')

    if not os.path.exists(attack_path):
        raise FileNotFoundError(f"找不到攻擊者權重檔 '{attack_path}'。請先執行 002_test.py 訓練攻擊者。")

    attacker_model.load_state_dict(torch.load(attack_path, map_location=args.device))
    attacker_model.eval()

    return model, attacker_model


def save_comparison_plot(args, model, attacker_model, test_loader, save_dir='vis_comparison', num_samples=10):
    """產生並儲存 [原始 vs 洗牌殘差 vs 攻擊還原] 的對比圖"""
    os.makedirs(save_dir, exist_ok=True)
    print(f"正在產生對比圖片並儲存至 '{save_dir}' 資料夾 (預計展示 {num_samples} 張)...")

    count = 0
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Visualizing"):
            if count >= num_samples:
                break

            imgs = imgs.to(args.device)
            B = imgs.size(0)

            # --- PPFR 系統處理 ---
            # 取得 Stage 2 洗牌後的殘差影像 X_p (這是攻擊者攔截到的)
            _, _, _, _, _, x_residue_up = model.obtain_residue(imgs)
            x_residue_shuffle = model.shuffle(x_residue_up)

            # --- 攻擊者還原 ---
            recovered_imgs = attacker_model(x_residue_shuffle)

            # --- 轉換為 Numpy 用於繪圖 [0, 1] (B, H, W, C) ---
            orig_np = denormalize(imgs).transpose(0, 2, 3, 1)
            # 為了讓殘差影像更清晰，通常會看未洗牌前的，但既然你要展示攻擊者看到的，我就展示 shuffle 過的。
            # 註：Shuffle 過的殘差影像看起來會像是有靜脈紋理的雜訊或是邊緣，且背景是灰色的(0.5)。
            shuffled_np = denormalize(x_residue_shuffle).transpose(0, 2, 3, 1)
            recons_np = denormalize(recovered_imgs).transpose(0, 2, 3, 1)

            # 遍歷 Batch 中的每一張圖
            for i in range(B):
                if count >= num_samples:
                    break

                # 建立畫布：3 欄 (原始, 攔截到的保護影像, 攻擊者還原圖)
                # 雖然你只說要比後面兩張，但加上原始圖能更清楚看出還原程度。
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 1. 原始 RGB 靜脈圖
                axes[0].imshow(orig_np[i])
                axes[0].set_title(f'Sample {count + 1}\nOriginal Vein (RGB)')
                axes[0].axis('off')

                # 2. 攔截到的保護影像 (Stage 2 Residue Shuffled)
                # 這是你要求的：Shuffle 過的那一張
                axes[1].imshow(shuffled_np[i])
                axes[1].set_title(f'Interepted Info ($X_p$)\n(Residue Shuffled - Stage 2)')
                axes[1].axis('off')

                # 3. 攻擊者重建圖
                # 這是你要求的：攻擊者重建的那一張
                axes[2].imshow(recons_np[i])
                axes[2].set_title(f'Attacker Recovered ($\hat{X}$)\nfrom $X_p$ using U-Net')
                axes[2].axis('off')

                # 儲存圖片
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'attack_vis_{count:03d}.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

                count += 1

def main():
    # 1. 配置與裝置設定
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"  # 預設使用你的靜脈資料集
    args = configs.get_dataset_params(args)

    # 使用小 batch size 方便視覺化迭代
    args.batch_size = 4

    # 2. 載入測試資料
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    try:
        # 3. 載入模型
        ppfr_model, attacker_model = load_models_for_vis(args)

        # 4. 產生對比圖 (預設存 10 張)
        save_comparison_plot(args, ppfr_model, attacker_model, test_loader, num_samples=10)

    except FileNotFoundError as e:
        print(f"\n[錯誤] {e}")
        print("請確保 'weights/' 資料夾中有 'best_generator.pth', 'best_recognizer.pth', 和 'attacker_unet.pth'。")


if __name__ == '__main__':
    main()