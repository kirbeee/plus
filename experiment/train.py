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
    # 1. 移除 Batch 維度 (1, 3, 112, 112) -> (3, 112, 112)
    img = tensor.squeeze(0)

    # 將 Tensor 轉換為可顯示的 Numpy 影像 [0, 1]
    img = img.detach().cpu().numpy()

    # 轉置為 (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    if is_residue:
        gain = 5.0
        img = img * gain
        img = np.abs((img / 2.0) - 1.0)
        img = np.clip(img, 0, 1)
    else:
        img = ((img - 1.0) / 2.0)* -1
        img = np.clip(img, 0, 1)

    return img

def visualize_stages():
    gen_path = '../weights/best_generator.pth'
    rec_path = '../weights/best_recognizer.pth'

    # 2. 載入單一影像樣本
    train_dataset = datasets.ImagesDataset(args=args, data_type='LED', phase='train')
    img_tensor, label = train_dataset[0]
    img_tensor = img_tensor.unsqueeze(0).to(args.device) # 增加 Batch 維度
    print(img_tensor.shape)

    # 3. 模擬 Stage 1 流程
    model_s1 = MinusBackbone(mode='stage1').to(args.device)
    model_s1.generator.load_state_dict(torch.load(gen_path, map_location=args.device))
    model_s1.eval()

    with torch.no_grad():
        x_encode_s1, x_residue_s1, _, _ = model_s1(img_tensor)

    # 4. 模擬 Stage 2 流程 (包含 Shuffle)
    model_s2 = MinusBackbone(mode='stage2').to(args.device)
    model_s2.generator.load_state_dict(torch.load(gen_path, map_location=args.device))
    model_s2.recognizer.load_state_dict(torch.load(rec_path, map_location=args.device))
    model_s2.eval()
    with torch.no_grad():
        x_encode_s2, x_residue_s2, _, _ = model_s2(img_tensor)

    # 5. 使用 Matplotlib 繪圖
    fig, axes = plt.subplots(1, 6, figsize=(16, 4))
    # print(img_tensor.size(),x_encode_s1.size(),x_residue_s1.size(),x_encode_s2.size(),x_residue_s2.size())
    axes[0].imshow(tensor_to_np(img_tensor))
    axes[0].set_title("Original Image (Input)")

    axes[1].imshow(tensor_to_np(x_encode_s1))
    axes[1].set_title("Stage 1: Reconstructed (x_encode)")

    axes[2].imshow(tensor_to_np(x_residue_s1,is_residue=True))
    axes[2].set_title("Stage 1: Residue (Features)")

    axes[3].imshow(tensor_to_np(x_residue_s2,is_residue=True))
    axes[3].set_title("Stage 2: Shuffled (Privacy)")

    axes[4].imshow(tensor_to_np(torch.zeros(1, 3, 112, 112)))
    axes[4].set_title("tensor with all zero")

    axes[5].imshow(tensor_to_np(torch.full((1, 3, 112, 112), -1.0)))
    axes[5].set_title("tensor with all -1")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = configs.get_all_params()
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    visualize_stages()
