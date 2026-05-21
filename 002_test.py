import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve

import configs
import datasets
from network.fsb_hash_net import FSB_Hash_Net, Hash_Generator

from testkit.unlinkability_metric import UnlinkabilityMetric


def load_models(args):
    """載入訓練好的 Feature Extractor 與 Hash Generator 權重"""
    fe = FSB_Hash_Net(embedding_size=args.dim, do_prob=0.0).to(args.device)
    gen = Hash_Generator(embedding_size=args.dim, do_prob=0.0, device=args.device, out_embedding_size=args.hash_dim).to(
        args.device)

    fe_path = 'weights/fsb_hashnet/best_feature_extractor.pth'
    gen_path = 'weights/fsb_hashnet/best_generator.pth'

    if os.path.exists(fe_path) and os.path.exists(gen_path):
        fe.load_state_dict(torch.load(fe_path, map_location=args.device))
        gen.load_state_dict(torch.load(gen_path, map_location=args.device))
        print("成功載入 FSB-HashNet 模型權重！")
    else:
        print("警告: 找不到模型權重，請確認 weights/fsb_hashnet/ 目錄下是否有 .pth 檔案。")

    fe.eval()
    gen.eval()
    return fe, gen


def compute_eer(genuine_scores, imposter_scores):
    """計算 EER (Equal Error Rate)"""
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    y_scores = np.concatenate([genuine_scores, imposter_scores])

    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    min_index = np.argmin(np.abs(fpr - fnr))
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return np.around(eer, 4)


def get_pairwise_scores(embeds_A, embeds_B, labels):
    """計算 Cosine Similarity 並區分 Genuine/Mated 與 Imposter/Non-Mated 分數"""
    embeds_A = torch.nn.functional.normalize(embeds_A, p=2, dim=1)
    embeds_B = torch.nn.functional.normalize(embeds_B, p=2, dim=1)
    sim_matrix = torch.mm(embeds_A, embeds_B.t()).numpy()

    targets_np = labels.numpy()
    label_matrix = (targets_np[:, None] == targets_np[None, :]).astype(int)

    # 排除對角線 (僅用於 EER 驗證；跨 Token 時不排除)
    mask = np.ones_like(sim_matrix, dtype=bool)
    np.fill_diagonal(mask, 0)

    same_id_scores = sim_matrix[(label_matrix == 1) & mask]
    diff_id_scores = sim_matrix[(label_matrix == 0) & mask]

    return same_id_scores, diff_id_scores


def run_test(args):
    # --- data loading ---
    test_dataset = datasets.ImagesDataset(args=args, data_type='LED', phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # --- model initialize ---
    fe, gen = load_models(args)

    print("\n[1/3] 提取指靜脈測試集特徵與生成雜湊碼 (Hash Codes)...")
    hash_user_list = []
    hash_stolen_list = []
    hash_renewed_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Feature Extraction"):
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            # 1. Base 特徵
            features = fe(imgs)

            # 2. User-Specific Token (正常註冊)
            h_user = gen(features, labels, training=False)

            # 3. Stolen Token (雜湊金鑰遭竊，假設攻擊者使用全0 Token)
            h_stolen = gen(features, torch.zeros_like(labels), training=False)

            # 4. Renewed Token (使用者註銷舊金鑰並重新配發)
            h_renewed = gen(features, labels + 1000, training=False)

            hash_user_list.append(h_user.cpu())
            hash_stolen_list.append(h_stolen.cpu())
            hash_renewed_list.append(h_renewed.cpu())
            labels_list.append(labels.cpu())

    hash_user = torch.cat(hash_user_list, dim=0)
    hash_stolen = torch.cat(hash_stolen_list, dim=0)
    hash_renewed = torch.cat(hash_renewed_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    print("\n[2/3] 計算安全與驗證指標 (EER)...")
    # --- 1. User-Specific 驗證 ---
    user_gen_scores, user_imp_scores = get_pairwise_scores(hash_user, hash_user, labels)
    user_eer = compute_eer(user_gen_scores, user_imp_scores)

    # --- 2. Stolen Token 驗證 ---
    stolen_gen_scores, stolen_imp_scores = get_pairwise_scores(hash_stolen, hash_stolen, labels)
    stolen_eer = compute_eer(stolen_gen_scores, stolen_imp_scores)

    print("\n[3/3] 呼叫官方套件計算不可連結性 (Unlinkability D_sys)...")
    # --- 3. 不可連結性 (Unlinkability / Revocability) ---
    sim_matrix_cross = torch.mm(
        torch.nn.functional.normalize(hash_user, p=2, dim=1),
        torch.nn.functional.normalize(hash_renewed, p=2, dim=1).t()
    ).numpy()

    targets_np = labels.numpy()
    label_matrix = (targets_np[:, None] == targets_np[None, :]).astype(int)

    mated_scores = sim_matrix_cross[label_matrix == 1]
    non_mated_scores = sim_matrix_cross[label_matrix == 0]

    # --- 準備匯出目錄 ---
    out_dir = './graphs/analysis_privacy_security/fsb_hashnet'
    os.makedirs(out_dir, exist_ok=True)

    # 直接呼叫你提供的 UnlinkabilityMetric 類別
    metric = UnlinkabilityMetric(mated_scores, non_mated_scores)
    dsys = metric.evaluate()

    # 自動匯出分析圖表
    fig_path = os.path.join(out_dir, 'image_unlinkability.pdf')
    metric.plot(figure_file=fig_path)

    # 儲存分數，確保相容性
    np.savetxt(os.path.join(out_dir, 'genuine.txt'), user_gen_scores)
    np.savetxt(os.path.join(out_dir, 'imposter.txt'), user_imp_scores)
    np.savetxt(os.path.join(out_dir, 'mated.txt'), mated_scores)
    np.savetxt(os.path.join(out_dir, 'nonmated.txt'), non_mated_scores)

    # --- 輸出最終結果 ---
    print(f"\n================ FSB-HashNet 評估結果 ================")
    print(f"1. 驗證效能 (User-Specific Token EER) : {user_eer * 100:.4f}%")
    print(f"2. 驗證效能 (Stolen Token EER)        : {stolen_eer * 100:.4f}%")
    print(f"3. 系統不可連結性 (Unlinkability D_sys): {dsys:.4f}")
    print(f"======================================================")
    print(f"[Info] 不可連結性分析圖表已自動匯出至 {fig_path}。")
    print(f"[Info] 所有特徵分數已存至 {out_dir} 目錄。")


if __name__ == '__main__':
    args = configs.get_all_params()
    args.dim = 1024
    args.hash_dim = 512
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)

    run_test(args)