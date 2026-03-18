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
    arcface_head = ArcFace(in_features=512, out_features=360, scale=32).to(args.device)
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

def evaluate_metrics(args, model, arcface_head, test_loader):
    correct = 0
    total = 0
    all_psnr = []
    all_ssim = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        # make data to batch img*32 , label *32
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            # Calculate Acc
            x_encode, _, x_feature, _ = model(imgs)
            dummy_labels = torch.zeros(labels.size(0)).long().to(args.device)
            outputs = arcface_head(x_feature, dummy_labels)
            predictions = torch.argmax(outputs[0], dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # 3. PSNR & SSIM 計算 (原始影像 vs 重建影像 x_encode)
            # 轉換為 Numpy 並調整維度為 (B, H, W, C)
            # orig_np = denormalize(imgs).transpose(0, 2, 3, 1)
            # recons_np = denormalize(x_encode).transpose(0, 2, 3, 1)

            # for i in range(orig_np.shape[0]):
            #     p = psnr(orig_np[i], recons_np[i], data_range=1.0)
            #     s = ssim(orig_np[i], recons_np[i], data_range=1.0)
            #     all_psnr.append(p)
            #     all_ssim.append(s)


            # --- EER 資料收集 ---
            logits = outputs[0]
            # 2. 將真實標籤存起來
            all_labels.extend(labels.cpu().numpy())

            # 3. 獲取分數。假設我們要看 class 1 的 EER，需先做 softmax 轉為機率
            probs = torch.softmax(logits, dim=1)
            scores_for_eer = probs[:, 1]  # 取出所有樣本屬於 class 1 的機率
            all_scores.extend(scores_for_eer.cpu().numpy())

            # 現在 all_labels 和 all_scores 有資料了，不會報錯
        eer_val, _ = calculate_eer(np.array(all_labels), np.array(all_scores))

    metrics = {
        'ACC': correct / total,
        'EER': eer_val,
        # 'PSNR': np.mean(all_psnr),
        # 'SSIM': np.mean(all_ssim)
    }
    return metrics


def main():
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model, arc_face = load_backbone(args)

    res = evaluate_metrics(args, model, arc_face, test_loader)

    print(f"\n================測試結果================")
    print(f"ACC: {res['ACC'] * 100:.2f}%")
    # print(f"PSNR: {res['PSNR']:.4f} dB")
    # print(f"SSIM: {res['SSIM']:.4f}")
    print(f"=========================================")

if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    args.mode = "stage2"
    main()