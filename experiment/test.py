import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import dct_transform, idct_transform


def test_dct_with_original_size(image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 736x192
    # 轉換為 Tensor (B, C, H, W) 並正規化到 [-1, 1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    img_tensor = (img_tensor / 127.5) - 1.0
    img_tensor = img_tensor.unsqueeze(0)

    # 升維 (Spatial -> Frequency)
    # 如果你的電腦記憶體 (RAM/VRAM) 不夠，可以將 ratio 改小 (例如 1)
    with torch.no_grad():
        # 注意: ratio=8 會消耗大量記憶體，若報錯請試著改為 ratio=1
        x_freq = dct_transform(img_tensor, ratio=8)

    print(f"升維後維度: {x_freq.shape}")
    # 預期輸出: (1, 192, 192, 736)

    # 3. 執行降維 (Frequency -> Spatial)
    with torch.no_grad():
        x_reconstructed = idct_transform(x_freq, ratio=8)

    print(f"還原影像維度: {x_reconstructed.shape}")

    # 4. 視覺化比較 (保持比例)
    plt.figure(figsize=(15, 6))

    plt.subplot(3, 1, 1)
    plt.title("Original Vein Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.title("Frequency Domain (Channel 0-2 combined)")
    # 抽取前三個頻域通道觀察血管特徵在頻域的樣子
    freq_vis = x_freq[0, :3].permute(1, 2, 0).numpy()
    freq_vis = (freq_vis - freq_vis.min()) / (freq_vis.max() - freq_vis.min())
    plt.imshow(freq_vis)
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.title("Reconstructed Image")
    # 將數值從 [0, 1] 轉回顯示格式
    recon_vis = x_reconstructed[0].permute(1, 2, 0).numpy()
    plt.imshow(np.clip(recon_vis, 0, 1))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_dct_with_original_size('PLUS-FV3-Laser_DORSAL_01_001_02_01.png')