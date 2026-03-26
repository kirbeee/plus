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
import torch.nn.functional as F

def load_backbone(args):
    model = MinusBackbone(mode=args.mode).to(args.device)
    gen_path = 'weights/best_generator.pth'
    rec_path = 'weights/best_recognizer.pth'
    model.generator.load_state_dict(torch.load(gen_path, map_location=args.device))
    model.recognizer.load_state_dict(torch.load(rec_path, map_location=args.device))
    model.eval()
    return model

def denormalize(tensor):
    """將 [-1, 1] 的 Tensor 轉回 [0, 1] 的 Numpy 用於計算指標"""
    return ((tensor + 1.0) / 2.0).clamp(0, 1).cpu().numpy()

def eer_calculation(args, model, test_loader):
    embeds_list = []
    targets_list = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(args.device)
            _, _, x_feature, _ = model(imgs)
            x_feature = F.normalize(x_feature, p=2, dim=1)

            embeds_list.append(x_feature.cpu())
            targets_list.append(labels.cpu())

    embeddings = torch.cat(embeds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    # 使用矩陣乘法快速計算 Cosine Similarity: (N, 512) @ (512, N) -> (N, N)
    sim_matrix = torch.mm(embeddings, embeddings.t()).numpy()

    # 建立標籤矩陣 (N, N), 1 表示身分相同，0 表示不同
    targets_np = targets.numpy()
    label_matrix = (targets_np[:, None] == targets_np[None, :]).astype(int)

    # 排除對角線（自己跟自己比）
    mask = np.ones_like(sim_matrix, dtype=bool)
    np.fill_diagonal(mask, 0)

    scores = sim_matrix[mask]
    actual_labels = label_matrix[mask]

    fpr, tpr, thresholds = skm.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    # 計算特定閾值下的準確度
    preds = (scores >= threshold).astype(int)
    acc = skm.accuracy_score(actual_labels, preds)

    return acc, eer

def psnr_ssim_calculation(args, model, test_loader):
    all_psnr = []
    all_ssim = []

    with torch.no_grad():
        # make data to batch img*32 , label *32
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            # PSNR & SSIM 計算 (原始影像 vs 重建影像 x_encode)
            # 轉換為 Numpy 並調整維度為 (B, H, W, C)
            x_encode, _, x_feature, _ = model(imgs)
            orig_np = denormalize(imgs).transpose(0, 2, 3, 1)
            recons_np = denormalize(x_encode).transpose(0, 2, 3, 1)

            for i in range(orig_np.shape[0]):
                p = psnr(orig_np[i], recons_np[i], data_range=1.0)
                s = ssim(orig_np[i], recons_np[i], data_range=1.0)
                all_psnr.append(p)
                all_ssim.append(s)
    return {
        'PSNR': np.mean(all_psnr),
        'SSIM': np.mean(all_ssim)
    }

def print_results(res):
    print(f"\n================Test Result================")
    print(f"ACC: {res["ACC"]}%")
    print(f"EER: {res['EER']}")
    print(f"PSNR: {res['PSNR']:.4f} dB")
    print(f"SSIM: {res['SSIM']:.4f}")
    print(f"=========================================")

def main():
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = load_backbone(args)
    result = psnr_ssim_calculation(args, model, test_loader)
    aac,eer_number = eer_calculation(args, model, test_loader)
    result['ACC'] = aac
    result['EER'] = eer_number
    print_results(result)
    return None

if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    args.mode = "stage2"
    main()