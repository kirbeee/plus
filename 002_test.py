import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sklearn.metrics as skm
import configs
import datasets
from model.model import MinusBackbone
import torch.nn.functional as F
import testkit.attacker as attacker
from model.utils import UNet
import os
from testkit.unlinkability_metric import UnlinkabilityMetric

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
        for imgs, labels in tqdm(test_loader,desc="EER Calculation"):
            imgs = imgs.to(args.device)
            _, _, x_feature, _ = model(imgs)
            x_feature = F.normalize(x_feature, p=2, dim=1)

            embeds_list.append(x_feature.cpu())
            targets_list.append(labels.cpu())

    embeddings = torch.cat(embeds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    # Cosine Similarity (N, 512) @ (512, N) -> (N, N)
    sim_matrix = torch.mm(embeddings, embeddings.t()).numpy()

    # 建立標籤矩陣 (N, N), 1 表示身分相同，0 表示不同
    targets_np = targets.numpy()
    label_matrix = (targets_np[:, None] == targets_np[None, :]).astype(int)

    # 排除對角線（自己跟自己比）
    mask = np.ones_like(sim_matrix, dtype=bool)
    np.fill_diagonal(mask, 0)

    scores = sim_matrix[mask]
    actual_labels = label_matrix[mask]

    fpr, tpr, thresholds = skm.roc_curve(actual_labels, scores, pos_label=1)
    fnr = 1 - tpr
    threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    # 計算特定閾值下的準確度
    preds = (scores >= threshold).astype(int)
    acc = skm.accuracy_score(actual_labels, preds)

    return acc, eer

def psnr_ssim_calculation(args, model, test_loader):
    attacker_model = UNet(in_channels=3, out_channels=3).to(args.device)

    if os.path.exists(attacker_weight_path):
        print("\n[note] find model ! ")
        attacker_model.load_state_dict(torch.load(attacker_weight_path, map_location=args.device))
    else:
        print("\n[note] model not found, start training attacker model ! ")
        train_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='train')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        attacker_model = attacker.train_attacker(args, model, train_loader, epochs=10, save_path=attacker_weight_path)

    att_psnr, att_ssim = attacker.attack_evaluation(args, model, attacker_model, test_loader)
    return {'Attack_PSNR': att_psnr, 'Attack_SSIM': att_ssim}

def print_results(res):
    print(f"\n================Test Result================")
    print(f"ACC: {res['AAC']}%")
    print(f"EER: {res['EER']}")
    print(f"Attacker Recovered PSNR: {res['Attack_PSNR']:.4f} dB")
    print(f"Attacker Recovered SSIM: {res['Attack_SSIM']:.4f}")
    print(f"Unlinkability D_sys: {res['Dsys']}")
    print(f"=========================================")

def unlinkability_calculation(args, model, test_loader):
    embeds_list = []
    targets_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="EER Calculation"):
            imgs = imgs.to(args.device)
            _, _, x_feature, _ = model(imgs)
            x_feature = F.normalize(x_feature, p=2, dim=1)

            embeds_list.append(x_feature.cpu())
            targets_list.append(labels.cpu())

    embeddings = torch.cat(embeds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    # Cosine Similarity (N, 512) @ (512, N) -> (N, N)
    sim_matrix = torch.mm(embeddings, embeddings.t()).numpy()

    # 建立標籤矩陣 (N, N), 1 表示身分相同，0 表示不同
    targets_np = targets.numpy()
    label_matrix = (targets_np[:, None] == targets_np[None, :]).astype(int)

    # 取得上三角矩陣的索引 (k=1 代表排除主對角線，也就是自己跟自己的比較)
    row_idx, col_idx = np.triu_indices(len(targets_np), k=1)

    # 透過索引，攤平所有不重複的「兩兩配對」分數與標籤
    pair_scores = sim_matrix[row_idx, col_idx]
    pair_labels = label_matrix[row_idx, col_idx]

    # 利用標籤過濾出 mated (同身分) 與 non_mated (不同身分) 的分數
    mated_scores = pair_scores[pair_labels == 1]
    non_mated_scores = pair_scores[pair_labels == 0]

    print(f"Mated pairs: {len(mated_scores)}, Non-Mated pairs: {len(non_mated_scores)}")
    metric = UnlinkabilityMetric(mated_scores, non_mated_scores)
    dsys_score = metric.evaluate()
    metric.plot(figure_file="unlinkability_dist.png")
    return dsys_score

def main():
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = load_backbone(args)
    result = psnr_ssim_calculation(args, model, test_loader)
    aac,eer_number = eer_calculation(args, model, test_loader)
    Dsys = unlinkability_calculation(args, model, test_loader)
    result['AAC'] = aac
    result['EER'] = eer_number
    result['Dsys'] = Dsys
    print_results(result)
    return None

if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    attacker_weight_path = 'weights/attacker_unet.pth'
    args = configs.get_dataset_params(args)
    args.mode = "stage2"
    args.attacker_weight_path = "weights/attacker_unet.pth"
    main()