import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from model.model import MinusGenerativeModel,LightVeinCNN
from model.utils import UNet, dct_transform, idct_transform
import datasets
import configs

def visualize_results(orig, encoded, residue_img, save_path):
    """視覺化對比：原圖 vs 生成特徵還原圖 vs 隱私保護殘差圖"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original (X)")
    plt.imshow(orig, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Generated (X')")
    plt.imshow(encoded, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Protected Residue (X_p)")
    plt.imshow(residue_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def channel_shuffle(x_freq, seed=42):
    """
    任務五：隨機通道洗牌 (Random Channel Shuffling)
    對頻域特徵的通道進行打亂以保護隱私
    """
    B, C, H, W = x_freq.shape
    torch.manual_seed(seed)
    # 產生隨機排列的索引
    idx = torch.randperm(C).to(x_freq.device)
    # 根據索引重新排列通道
    shuffled_x = x_freq[:, idx, :, :]
    return shuffled_x


def test(args):
    configs.setup_seed(args.seed)

    # 1. 載入測試資料
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True)

    # 2. 初始化並載入訓練好的模型
    generator = MinusGenerativeModel(mode='stage1', backbone=UNet).to(args.device)
    recognizer = LightVeinCNN(in_channels=192, embedding_size=512).to(args.device)

    print("正在載入模型權重...")
    generator.load_state_dict(torch.load('weights/best_generator.pth', map_location=args.device))
    recognizer.load_state_dict(torch.load('weights/best_recognizer.pth', map_location=args.device))

    generator.eval()
    recognizer.eval()

    all_embeddings = []
    all_labels = []
    ssim_scores = []
    psnr_scores = []

    print(f"開始測試... 總樣本數: {len(test_dataset)}")
    os.makedirs('test_results', exist_ok=True)

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            # --- 核心推理流程 ---
            # a. 轉頻域
            print(imgs.shape)
            x_freq = dct_transform(imgs, ratio=8)

            # b. 生成特徵
            x_encode, _ = generator(x_freq)

            # c. 計算殘差 (高維度相減)
            r = x_freq - x_encode

            # d. 任務五：通道洗牌
            r_shuffled = channel_shuffle(r, seed=args.seed)

            # e. 提取身分特徵 (用於辨識)
            x_feature = recognizer(r_shuffled)

            # f. 轉回空間域得到隱私保護影像 X_p
            x_p = idct_transform(r_shuffled)
            x_encode_spatial = idct_transform(x_encode)

            all_embeddings.append(x_feature.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # --- 視覺化與任務六：隱私評估 ---
            # 將 Tensor 轉為 numpy (H, W)，假設指靜脈已轉為單通道或取其灰階
            orig_np = ((imgs[0, 0].cpu().numpy() + 1) * 127.5).astype(np.uint8)
            xp_np = ((x_p[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            encode_np = ((x_encode_spatial[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

            # 計算 SSIM 與 PSNR (評估原圖與保護圖的差異，分數越低表示隱私保護越好)
            score_ssim = ssim(orig_np, xp_np, data_range=255)
            score_psnr = psnr(orig_np, xp_np, data_range=255)
            ssim_scores.append(score_ssim)
            psnr_scores.append(score_psnr)

            if i % 50 == 0:
                visualize_results(orig_np, encode_np, xp_np, f'test_results/sample_{i}.png')

    # --- 任務六：辨識準確度評估 (1:N) ---
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    np.fill_diagonal(sim_matrix, -1)  # 排除自己
    preds = labels[np.argmax(sim_matrix, axis=1)]
    accuracy = np.mean(preds == labels)

    print(f"\n=== 測試報告 ===")
    print(f"Top-1 辨識準確率: {accuracy * 100:.2f}%")
    print(f"平均 SSIM (隱私保護程度，越低越好): {np.mean(ssim_scores):.4f}")
    print(f"平均 PSNR (隱私保護程度，越低越好): {np.mean(psnr_scores):.2f} dB")
    print(f"視覺化結果已存至: ./test_results/")


if __name__ == '__main__':
    args = configs.get_all_params()
    args.datasets = 'PLUSVein-FV3'
    args = configs.get_dataset_params(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test(args)