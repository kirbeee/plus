import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import configs
import datasets
from model.model import MinusBackbone


def tensor_to_np(tensor, is_residue=False):
    """將 Tensor 轉換為可顯示的 Numpy 影像 [0, 1]"""
    img = tensor.detach().cpu().numpy()[0]

    # 取前三個通道 (如果 n_duplicate > 1，shuffle 後可能會有更多通道)
    if img.shape[0] >= 3:
        img = img[:3]

    # 轉置為 (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    if is_residue:
        # 對於殘差，因為範圍是 [-1, 1]，我們用固定的反正規化，而不是 min-max stretch
        # 這樣才能真實反映出它跟原圖相比有多 "暗"
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
    else:
        # 一般圖片的反正規化 (或者你原本的寫法也可以)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img

def visualize_stages():
    # 2. 載入單一影像樣本
    train_dataset = datasets.ImagesDataset(args=args, data_type='LED', phase='train')
    img_tensor, label = train_dataset[0]
    img_tensor = img_tensor.unsqueeze(0).to(device) # 增加 Batch 維度

    # 3. 模擬 Stage 1 流程
    model_s1 = MinusBackbone(mode='stage1').to(device)
    model_s1.generator.load_state_dict(torch.load('../weights/best_generator.pth'))
    model_s1.recognizer.load_state_dict(torch.load('../weights/best_recognizer.pth'))
    model_s1.eval()

    with torch.no_grad():
        x_encode_s1, x_residue_s1, _, _ = model_s1(img_tensor)

    # 4. 模擬 Stage 2 流程 (包含 Shuffle)
    model_s2 = MinusBackbone(mode='stage2').to(device)
    model_s2.generator.load_state_dict(torch.load('../weights/best_generator.pth'))
    model_s2.recognizer.load_state_dict(torch.load('../weights/best_recognizer.pth'))
    model_s2.eval()
    with torch.no_grad():
        x_encode_s2, x_residue_s2, _, _ = model_s2(img_tensor)

    # 5. 使用 Matplotlib 繪圖
    fig, axes = plt.subplots(1, 4)
    print(img_tensor.size())
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
    args = configs.get_all_params()
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_stages()
