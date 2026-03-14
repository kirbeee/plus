import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import configs
import datasets
from model.model import MinusBackbone

def tensor_to_np(tensor):
    """將 Tensor 轉換為可顯示的 Numpy 影像 [0, 1]"""
    img = tensor.detach().cpu().numpy()[0] # 抓取 Batch 中的第一張
    if img.shape[0] == 192: # 如果是頻域特徵，只取前 3 個通道示意
        img = img[:3]

    # 轉置為 (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # 正規化到 [0, 1] 供 matplotlib 顯示
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def visualize_stages():
    # 1. 初始化環境與模型
    args = configs.get_all_params()
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 載入單一影像樣本
    train_dataset = datasets.ImagesDataset(args=args, data_type='LED', phase='train')
    img_tensor, label = train_dataset[0]
    img_tensor = img_tensor.unsqueeze(0).to(device) # 增加 Batch 維度

    # 3. 模擬 Stage 1 流程
    model_s1 = MinusBackbone(mode='stage1').to(device)
    # 若有權重，請在此載入: model_s1.generator.load_state_dict(torch.load('weights/best_generator.pth'))
    model_s1.eval()

    with torch.no_grad():
        x_encode_s1, x_residue_s1, _, _ = model_s1(img_tensor)

    # 4. 模擬 Stage 2 流程 (包含 Shuffle)
    model_s2 = MinusBackbone(mode='stage2').to(device)
    model_s2.eval()
    with torch.no_grad():
        x_encode_s2, x_residue_s2, _, _ = model_s2(img_tensor)

    # 5. 使用 Matplotlib 繪圖
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))

    axes[0].imshow(tensor_to_np(img_tensor))
    axes[0].set_title("Original Image (Input)")

    axes[1].imshow(tensor_to_np(x_encode_s1))
    axes[1].set_title("Stage 1: Reconstructed (x_encode)")

    axes[2].imshow(tensor_to_np(x_residue_s1))
    axes[2].set_title("Stage 1: Residue (Features)")

    axes[3].imshow(tensor_to_np(x_residue_s2))
    axes[3].set_title("Stage 2: Shuffled (Privacy)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_stages()
