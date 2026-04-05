import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def compute_dsys_metric(mated_scores, non_mated_scores, omega=1.0, n_bins=100):
    """
    計算全域不可連結性指標 D_sys (Global Measure)
    參考自 General Framework to Evaluate Unlinkability (TIFS18)
    """
    # 決定分數範圍並建立直方圖
    min_s = min(np.min(mated_scores), np.min(non_mated_scores))
    max_s = max(np.max(mated_scores), np.max(non_mated_scores))
    bin_edges = np.linspace(min_s, max_s, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 計算機率密度函數 (PDF)
    y_mated, _ = np.histogram(mated_scores, bins=bin_edges, density=True)
    y_non_mated, _ = np.histogram(non_mated_scores, bins=bin_edges, density=True)

    # 防止除以 0
    with np.errstate(divide='ignore', invalid='ignore'):
        LR = np.divide(y_mated, y_non_mated)
        # 根據論文公式: D = 2 * (omega * LR / (1 + omega * LR)) - 1
        D = 2 * (omega * LR / (1 + omega * LR)) - 1
        D[omega * LR <= 1] = 0  # 只有當 LR > 1/omega 時才具備可連結性
        D[y_non_mated == 0] = 1  # 如果 Non-mated 機率為 0，視為完全可連結

    # 全域指標 D_sys = 積分 (D(s) * P(s|mated) ds)
    dsys = np.trapz(D * y_mated, bin_centers)
    return np.clip(dsys, 0, 1)


def test_unlinkability(args, model, attacker_model, test_loader):
    """
    評估 Attacker 生成影像與 Ground Truth 的不可連結性
    """
    model.eval()
    attacker_model.eval()

    gt_features = []
    recovered_features = []
    labels = []

    with torch.no_grad():
        for imgs, target in tqdm(test_loader, desc="提取特徵進行 Unlinkability 測試"):
            imgs = imgs.to(args.device)

            # 1. 取得 Ground Truth 特徵
            _, _, feat_gt, _ = model(imgs)
            gt_features.append(F.normalize(feat_gt, p=2, dim=1).cpu())

            # 2. 取得 Attacker 還原影像
            _, _, _, _, _, x_res = model.obtain_residue(imgs)
            x_input = model.shuffle(x_res)
            recovered_img = attacker_model(x_input)

            # 3. 取得還原影像的特徵
            _, _, feat_rec, _ = model(recovered_img)
            recovered_features.append(F.normalize(feat_rec, p=2, dim=1).cpu())
            labels.append(target)

    gt_f = torch.cat(gt_features, dim=0)
    rec_f = torch.cat(recovered_features, dim=0)
    lbls = torch.cat(labels, dim=0).numpy()

    # 計算餘弦相似度矩陣 (N x N)
    sim_matrix = torch.mm(gt_f, rec_f.t()).numpy()

    # 區分 Mated (同一人) 與 Non-Mated (不同人)
    label_mat = (lbls[:, None] == lbls[None, :])
    mated_scores = sim_matrix[label_mat]
    non_mated_scores = sim_matrix[~label_mat]

    dsys = compute_dsys_metric(mated_scores, non_mated_scores, omega=args.ul_omega, n_bins=args.ul_bins)
    return dsys

