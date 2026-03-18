import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sklearn.metrics as skm
import configs
import datasets
from model.model import MinusBackbone
from torchkit.head.localfc.arcface import ArcFace
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_backbone(args):
    model = MinusBackbone(mode=args.mode).to(args.device)
    arcface_head = ArcFace(in_features=512, out_features=360).to(args.device)
    gen_path = 'weights/best_generator.pth'
    rec_path = 'weights/best_recognizer.pth'
    arcface_path = 'weights/best_arcface_head.pth'
    model.generator.load_state_dict(torch.load(gen_path, map_location=args.device))
    model.recognizer.load_state_dict(torch.load(rec_path, map_location=args.device))
    arcface_head.load_state_dict(torch.load(arcface_path, map_location=args.device))
    model.eval()
    arcface_head.eval()
    return model, arcface_head

def denormalize(tensor):
    """將 [-1, 1] 的 Tensor 轉回 [0, 1] 的 Numpy 用於計算指標"""
    return ((tensor + 1.0) / 2.0).clamp(0, 1).cpu().numpy()

def calculate_eer(labels, scores):
    fpr, tpr, thresholds = skm.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold

def evaluate_with_classification(args, model, arcface_head, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            # 1. 提取特徵
            _, _, x_feature, _ = model(imgs)

            # 2. 通過 ArcFace 獲取分類結果
            # 測試時 label 不重要，傳入 dummy labels 即可
            dummy_labels = torch.zeros(labels.size(0)).long().to(args.device)
            outputs = arcface_head(x_feature, dummy_labels)

            # outputs[0] 通常是相似度分數 (logits)
            predictions = torch.argmax(outputs[0], dim=1)

            # 3. 計算準確率
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    classification_acc = correct / total
    return classification_acc


def evaluate_metrics(args, model, arcface_head, test_loader):
    correct = 0
    total = 0
    all_psnr = []
    all_ssim = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            # 1. 提取特徵與重建影像
            x_encode, _, x_feature, _ = model(imgs)

            # 2. 分類準確度計算
            dummy_labels = torch.zeros(labels.size(0)).long().to(args.device)
            outputs = arcface_head(x_feature, dummy_labels)
            predictions = torch.argmax(outputs[0], dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # 3. PSNR & SSIM 計算 (原始影像 vs 重建影像 x_encode)
            # 轉換為 Numpy 並調整維度為 (B, H, W, C)
            orig_np = denormalize(imgs).transpose(0, 2, 3, 1)
            recons_np = denormalize(x_encode).transpose(0, 2, 3, 1)

            for i in range(orig_np.shape[0]):
                p = psnr(orig_np[i], recons_np[i], data_range=1.0)
                s = ssim(orig_np[i], recons_np[i], data_range=1.0, multichannel=True)
                all_psnr.append(p)
                all_ssim.append(s)

    metrics = {
        'ACC': correct / total,
        'EER': 0,
        'PSNR': np.mean(all_psnr),
        'SSIM': np.mean(all_ssim)
    }
    return metrics


def main():
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model, arc_face = load_backbone(args)

    res = evaluate_metrics(args, model, arc_face, test_loader)

    print(f"\n================測試結果================")
    print(f"ACC: {res['ACC'] * 100:.2f}%")
    print(f"PSNR: {res['PSNR']:.4f} dB")
    print(f"SSIM: {res['SSIM']:.4f}")
    print(f"=========================================")

if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    args.mode = "stage2"
    main()