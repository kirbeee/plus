import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import configs
import datasets
from model.model import MinusBackbone
from model.utils import UNet
from testkit.attacker import denormalize


def load_models_for_vis(args, weights_dir='../weights'):
    model = MinusBackbone(mode="stage2").to(args.device)
    gen_path = os.path.join(weights_dir, 'best_generator.pth')
    rec_path = os.path.join(weights_dir, 'best_recognizer.pth')
    model.generator.load_state_dict(torch.load(gen_path, map_location=args.device))
    model.recognizer.load_state_dict(torch.load(rec_path, map_location=args.device))
    model.eval()

    # load attacker
    attacker_model = UNet(in_channels=3, out_channels=3).to(args.device)
    attack_path = os.path.join(weights_dir, 'attacker_unet.pth')
    attacker_model.load_state_dict(torch.load(attack_path, map_location=args.device))
    attacker_model.eval()

    return model, attacker_model


def comparison_plot(args, model, attacker_model, test_loader):
    """產生並儲存 [原始 vs 洗牌殘差 vs 攻擊還原] 的對比圖"""

    count = 0
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Visualizing"):

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

            # 建立畫布：3 欄 (原始, 攔截到的保護影像, 攻擊者還原圖)
            # 雖然你只說要比後面兩張，但加上原始圖能更清楚看出還原程度。
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 1. 原始 RGB 靜脈圖
            axes[0].imshow(orig_np[0])
            axes[0].set_title(f'Sample {count + 1}\nOriginal Vein (RGB)')
            axes[0].axis('off')

            # 2. 攔截到的保護影像 (Stage 2 Residue Shuffled)
            # 這是你要求的：Shuffle 過的那一張
            axes[1].imshow(shuffled_np[0])
            axes[1].set_title(f'Interepted Info ($X_p$)\n(Residue Shuffled - Stage 2)')
            axes[1].axis('off')

            # 3. 攻擊者重建圖
            # 這是你要求的：攻擊者重建的那一張
            axes[2].imshow(recons_np[0])
            axes[2].set_title('Attacker Recovered ($\\hat{X}$)\nfrom $X_p$ using U-Net')
            axes[2].axis('off')

            plt.show()
            break

def main():
    # 1. 配置與裝置設定
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)

    # 使用小 batch size 方便視覺化迭代
    args.batch_size = 4

    # 2. 載入測試資料
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 3. 載入模型
    ppfr_model, attacker_model = load_models_for_vis(args)

    # 4. 產生對比圖 (預設存 10 張)
    comparison_plot(args, ppfr_model, attacker_model, test_loader)

if __name__ == '__main__':
    main()