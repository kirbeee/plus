import os
import torch
import cv2
import numpy as np
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 匯入您提供的模型組件
from model.model import MinusGenerativeModel, MinusBackbone
from model.utils import UNet, idct_transform
from model.model_irse_dct import IR_50
from

001
_train
import PLUSVeinDataset  # 複用訓練集的 Dataset 類


def visualize_results(orig, encoded, residue_img, save_path):
    """
    視覺化對比：原始圖 vs 生成圖 vs 最終殘差圖 (X_p)
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original (X)")
    plt.imshow(orig, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Generated (X')")
    plt.imshow(encoded, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Protected Residue (X_p)")
    plt.imshow(residue_img, cmap='gray')

    plt.savefig(save_path)
    plt.close()


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 載入測試資料 (使用 LASER 或 LED 感測器)
    test_dataset = PLUSVeinDataset(pkl_file='annotations_plusvein.pkl', mode='test', sensor='LED')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. 初始化模型 (必須與訓練時的參數一致)
    # 我們直接使用 MinusBackbone 來封裝整個流程
    model = MinusBackbone(mode='stage1').to(device)

    # 假設您已經有訓練好的權重，請取消註解下方行
    # model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    correct = 0
    total = 0
    all_embeddings = []
    all_labels = []

    print(f"開始測試... 總樣本數: {len(test_dataset)}")

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)

            # 透過 Backbone 取得中間產物
            # x_encode: 生成器重建的頻域特徵轉回空間域
            # x_residue: 殘差特徵轉回空間域 (即隱私保護後的 X_p)
            # x_feature: 辨識模型提取的特徵向量
            x_encode, x_residue, x_feature, _ = model(imgs)

            # 收集特徵用於計算準確度 (這裡簡化為最高分比對)
            # 在實際指靜脈場景中，通常會比對 Embedding 的餘弦相似度
            all_embeddings.append(x_feature.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # 3. 每隔一段時間存一張視覺化圖片
            if i % 50 == 0:
                # 將 Tensor 轉回可顯示的影像格式 [0, 255]
                orig_np = ((imgs[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                encode_np = ((x_encode[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                # 殘差影像通常較暗，我們做正規化方便觀察
                res_np = x_residue[0].permute(1, 2, 0).cpu().numpy()
                res_np = ((res_np - res_np.min()) / (res_np.max() - res_np.min()) * 255).astype(np.uint8)

                os.makedirs('test_results', exist_ok=True)
                visualize_results(orig_np, encode_np, res_np, f'test_results/sample_{i}.png')

    # 4. 簡單的檢索測試 (1:N 辨識)
    # 這裡演示如何評估特徵的區分度
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    # 計算特徵間的餘弦相似度矩陣
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)

    # 排除自己比對自己
    np.fill_diagonal(sim_matrix, -1)
    preds = labels[np.argmax(sim_matrix, axis=1)]

    accuracy = np.mean(preds == labels)
    print(f"\n--- 測試報告 ---")
    print(f"Top-1 辨識準確率: {accuracy * 100:.2f}%")
    print(f"視覺化結果已存至: ./test_results/")


if __name__ == '__main__':
    # 這裡可以手動指定參數或從 configs 讀取
    class DummyArgs:
        def __init__(self):
            self.seed = 42
            self.bat = 1
            self.lr = 0.0001
            self.weight_decay = 5e-4


    args = DummyArgs()
    test(args)