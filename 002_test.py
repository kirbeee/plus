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


def load_models(args,data_type):
    feature_extractor = FSB_Hash_Net(embedding_size=args.dim, do_prob=0.0).to(args.device)
    hash_generator = Hash_Generator(embedding_size=args.dim, do_prob=0.0, device=args.device, out_embedding_size=args.hash_dim).to(
        args.device)
    save_path = os.path.join(args.root_model, str(data_type))
    fe_path = os.path.join(save_path, 'best_feature_extractor.pth')
    gen_path = os.path.join(save_path, 'best_generator.pth')
    if os.path.exists(fe_path) and os.path.exists(gen_path):
        feature_extractor.load_state_dict(torch.load(fe_path, map_location=args.device))
        hash_generator.load_state_dict(torch.load(gen_path, map_location=args.device))
        print("load model")
    else:
        print("no model find! check dir")
        exit(0)

    feature_extractor.eval()
    hash_generator.eval()
    return feature_extractor, hash_generator


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def compute_eer_and_save_roc(genuine_scores, imposter_scores, save_path=None):
    """
    計算 EER 並可選擇性將 ROC 曲線儲存成圖片
    :param genuine_scores: 正樣本（真實匹配）的分數列表/陣列
    :param imposter_scores: 負樣本（冒充匹配）的分數列表/陣列
    :param save_path: 圖片儲存路徑（例如 'roc_curve.png'），若為 None 則不儲存
    """
    # 1. 建立 Ground Truth 與 分數
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    y_scores = np.concatenate([genuine_scores, imposter_scores])

    # 2. 計算 ROC 曲線節點與 AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 3. 計算 EER
    fnr = 1 - tpr
    min_index = np.argmin(np.abs(fpr - fnr))
    eer = np.mean((fpr[min_index], fnr[min_index]))

    # 計算最佳準確率 (Best Accuracy)
    P = len(genuine_scores)  # 正樣本總數
    N = len(imposter_scores)  # 負樣本總數
    # 利用 TPR = TP/P -> TP = TPR * P ； FPR = FP/N -> TN = (1 - FPR) * N
    acc_list = (tpr * P + (1 - fpr) * N) / (P + N)
    best_acc = np.max(acc_list)

    # 4. 繪製並儲存 ROC 曲線
    if save_path is not None:
        plt.figure(figsize=(6, 6))

        # 畫出 ROC 曲線
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')

        # 畫出 對角對稱線（隨機猜測基證線）
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # 標記出 EER 的工作點 (FPR, TPR = 1 - FNR)
        plt.plot(fpr[min_index], tpr[min_index], 'ro', label=f'EER = {eer:.4f}')

        # 設定圖表標籤與範圍
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)

        # --- 核心儲存步驟 ---
        # bbox_inches='tight' 可以防止標籤被切到
        # dpi=300 可以大幅提升圖片清晰度，適合放進論文或報告
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 關閉畫布釋放記憶體，避免連續訓練/測試時記憶體洩漏
        print(f"ROC 曲線已成功儲存至: {save_path}")

    return eer, best_acc


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

def compute_unlinkability(labels, hash_user, hash_renewed, out_dir=None):
    # --- 1. 不可連結性 (Unlinkability / Revocability) ---
    sim_matrix_cross = torch.mm(
        torch.nn.functional.normalize(hash_user, p=2, dim=1),
        torch.nn.functional.normalize(hash_renewed, p=2, dim=1).t()
    ).numpy()

    targets_np = labels.numpy()
    label_matrix = (targets_np[:, None] == targets_np[None, :]).astype(int)

    mated_scores = sim_matrix_cross[label_matrix == 1]
    non_mated_scores = sim_matrix_cross[label_matrix == 0]
    print(f"Mated pairs: {len(mated_scores)}, Non-Mated pairs: {len(non_mated_scores)}")

    # prepare dir
    metric = UnlinkabilityMetric(mated_scores, non_mated_scores)
    dsys = metric.evaluate()

    if out_dir is not None:
        # make plot
        fig_path = os.path.join(out_dir, 'image_unlinkability.pdf')
        metric.plot(figure_file=fig_path)

    return dsys

def run_test(args):
    configs.setup_seed(args.seed)
    for data_type in args.data_type:
        test_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers)

        # 取得訓練集類別數，用來計算 token offset
        # 訓練 token 範圍: 0 ~ num_train_classes-1
        # 測試 token 必須完全在此範圍之外
        train_dataset_tmp = datasets.ImagesDataset(args=args, data_type=data_type, phase='train')
        num_train_classes = len(set(item['label'] for item in train_dataset_tmp.data))
        del train_dataset_tmp
        print(f"訓練類別數: {num_train_classes}, Token offset 起點: {num_train_classes}")

        # ---------------------------------------------------------------
        # Token 設計（論文 Section III-C）
        # 訓練: token = label (0 ~ N-1)
        # 測試: token 必須與訓練完全不同，才能驗證零樣本泛化能力
        #
        # USER   token: labels + offset_1  → 每位使用者有唯一 token
        # STOLEN token: 固定錯誤值         → 攻擊者不知道正確 token
        # RENEWED token: labels + offset_2 → 撤銷舊 token 後重新配發
        # ---------------------------------------------------------------
        OFFSET_USER    = num_train_classes + 10000
        OFFSET_RENEWED = num_train_classes + 20000
        STOLEN_TOKEN   = num_train_classes + 99999  # 固定錯誤 token（非 0，避免與 label=0 混淆）

        feature_extractor, hash_generator = load_models(args, data_type)

        hash_user_list, hash_stolen_list, hash_renewed_list, labels_list = [], [], [], []

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Feature Extraction"):
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                features = feature_extractor(imgs)

                # User-Specific Token（核心修正）
                # 論文: "distinct key sets for training and inference"
                h_user = hash_generator(features, labels + OFFSET_USER, training=False)

                # Stolen Token：攻擊者對所有人用同一個錯誤 token
                h_stolen = hash_generator(
                    features,
                    torch.full_like(labels, STOLEN_TOKEN),
                    training=False
                )

                # Renewed Token：使用者撤銷後取得全新 token
                h_renewed = hash_generator(features, labels + OFFSET_RENEWED, training=False)

                hash_user_list.append(h_user.cpu())
                hash_stolen_list.append(h_stolen.cpu())
                hash_renewed_list.append(h_renewed.cpu())
                labels_list.append(labels.cpu())

        hash_user    = torch.cat(hash_user_list, dim=0)
        hash_stolen  = torch.cat(hash_stolen_list, dim=0)
        hash_renewed = torch.cat(hash_renewed_list, dim=0)
        labels       = torch.cat(labels_list, dim=0)

        # 驗證指標（同原本邏輯）
        user_gen, user_imp = get_pairwise_scores(hash_user, hash_user, labels)
        user_eer, user_acc = compute_eer_and_save_roc(
            user_gen, user_imp,
            save_path=os.path.join(args.root_model, str(data_type), 'roc_user.png')
        )

        stolen_gen, stolen_imp = get_pairwise_scores(hash_stolen, hash_stolen, labels)
        stolen_eer, stolen_acc = compute_eer_and_save_roc(
            stolen_gen, stolen_imp,
            save_path=os.path.join(args.root_model, str(data_type), 'roc_stolen.png')
        )

        d_sys = compute_unlinkability(
            labels, hash_user, hash_renewed,
            out_dir=os.path.join(args.root_model, str(data_type))
        )

        print(f"\n================ FSB-HashNet 評估結果 [{data_type}] ================")
        print(f"1. 驗證效能 (User-Specific Token EER) : {user_eer * 100:.4f}%")
        print(f"   驗證效能 (User-Specific Token ACC) : {user_acc * 100:.4f}%")
        print(f"2. 驗證效能 (Stolen Token EER)        : {stolen_eer * 100:.4f}%")
        print(f"   驗證效能 (Stolen Token ACC)        : {stolen_acc * 100:.4f}%")
        print(f"3. 系統不可連結性 (Unlinkability D_sys): {d_sys:.4f}")
        print(f"======================================================")

if __name__ == '__main__':
    args = configs.get_all_params()
    args.dim = 1024
    args.hash_dim = 512
    for dataset in ['FV-USM', 'PLUSVein-FV3', 'UTFVP']:
        args.datasets = dataset
        args = configs.get_dataset_params(args)
        run_test(args)