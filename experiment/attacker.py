import torch
import os
import matplotlib.pyplot as plt
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


def comparison_plot(args, model, attacker_model, img_tensor):
    """產生並儲存 [原始 vs 洗牌殘差 vs 攻擊還原] 的對比圖"""
    with torch.no_grad():
        imgs = img_tensor.to(args.device)

        # Stage 2 (Shuffle)
        _, _, _, _, _, x_residue = model(imgs)

        recovered_imgs = attacker_model(x_residue)

        # --- 轉換為 Numpy 用於繪圖 [0, 1] (B, H, W, C) ---
        orig_np = denormalize(imgs).transpose(0, 2, 3, 1)
        # 為了讓殘差影像更清晰，通常會看未洗牌前的，但既然你要展示攻擊者看到的，我就展示 shuffle 過的。
        # 註：Shuffle 過的殘差影像看起來會像是有靜脈紋理的雜訊或是邊緣，且背景是灰色的(0.5)。
        shuffled_np = denormalize(x_residue).transpose(0, 2, 3, 1)
        recons_np = denormalize(recovered_imgs).transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. 原始 RGB 靜脈圖
    axes[0].imshow(orig_np[0])
    axes[0].set_title(f'Original Vein (RGB)')

    # 2. 攔截到的保護影像 (Stage 2 Residue Shuffled)
    axes[1].imshow(shuffled_np[0])
    axes[1].set_title(f'Interepted Info ($X_p$)\n(Residue Shuffled - Stage 2)')

    # 3. 攻擊者重建圖
    axes[2].imshow(recons_np[0])
    axes[2].set_title('Attacker Recovered ($\\hat{X}$)\nfrom $X_p$ using U-Net')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # 1. 配置與裝置設定
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)


    # 2. 載入測試資料
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    img_tensor, label = test_dataset[0]

    # 3. 載入模型
    ppfr_model, attacker_model = load_models_for_vis(args)

    # 4. 產生對比圖
    comparison_plot(args, ppfr_model, attacker_model, img_tensor)

if __name__ == '__main__':
    main()