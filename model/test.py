import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import dct_transform, idct_transform

def test_dct_dimension_transform(image_path):
    # 1. 讀取指靜脈影像 (RGB)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("錯誤：找不到影像檔案")
        return

    # 調整大小至 112x112 (符合你 model.py 中的預設尺寸)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (112, 112))

    # 轉換為 Tensor (B, C, H, W) 並正規化到 [-1, 1]
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 127.5) - 1.0
    img_tensor = img_tensor.unsqueeze(0)  # 增加 Batch 維度

    print(f"原始影像維度: {img_tensor.shape} (Batch, RGB, H, W)")

    # 2. 執行升維 (Spatial -> Frequency)
    # 使用你 utils.py 中的 dct_transform
    # ratio=8 會將 112x112 放大後再做 DCT
    with torch.no_grad():
        x_freq = dct_transform(img_tensor, ratio=8)

    print(f"升維後維度: {x_freq.shape} (Batch, 192 Channels, H_freq, W_freq)")
    # 註：192 = 3 (RGB) * 64 (DCT coefficients)

    # 3. 執行降維 (Frequency -> Spatial)
    # 使用你 utils.py 中的 idct_transform
    with torch.no_grad():
        x_reconstructed = idct_transform(x_freq, ratio=8)

    print(f"還原影像維度: {x_reconstructed.shape} (Batch, RGB, H, W)")

    # 4. 視覺化結果
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original (RGB)")
    plt.imshow(img_resized)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Frequency Channels (First 3 of 192)")
    # 抽取前三個頻域通道觀察特徵
    freq_vis = x_freq[0, :3].permute(1, 2, 0).numpy()
    freq_vis = (freq_vis - freq_vis.min()) / (freq_vis.max() - freq_vis.min())
    plt.imshow(freq_vis)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed (After IDCT)")
    recon_vis = x_reconstructed[0].permute(1, 2, 0).numpy()
    plt.imshow(recon_vis)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_dct_dimension_transform('../data/PLUSVein-FV3/PLUSVein-FV3-ROI_combined/ROI/PLUS-FV3-Laser/DORSAL/01/001/PLUS-FV3-Laser_DORSAL_01_001_02_01.png')