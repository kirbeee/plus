import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import configs
import datasets
from network.fsb_hash_net import FSB_Hash_Net, Hash_Generator
from network.mamba_net import Mamba_Hash_Net
from testkit.unlinkability_metric import UnlinkabilityMetric
import numpy as np
from sklearn.metrics import roc_curve

def load_models(args):
    feature_extractor = Mamba_Hash_Net(embedding_size=args.dim).to(args.device)
    hash_generator = Hash_Generator(embedding_size=args.dim, do_prob=0.0, device=args.device, out_embedding_size=args.hash_dim).to(
        args.device)
    fe_path = os.path.join(args.model_root, f'{args.dataset}_feature_extractor.pth')
    gen_path = os.path.join(args.model_root,  f'{args.dataset}_best_generator.pth')
    if os.path.exists(fe_path) and os.path.exists(gen_path):
        feature_extractor.load_state_dict(torch.load(fe_path, map_location=args.device))
        hash_generator.load_state_dict(torch.load(gen_path, map_location=args.device))
        print("load model")
    else:
        raise ModuleNotFoundError

    feature_extractor.eval()
    hash_generator.eval()
    return feature_extractor, hash_generator

def compute_eer_and_save_roc(genuine_scores, imposter_scores):
    """
    計算 EER 並可選擇性將 ROC 曲線儲存成圖片
    :param genuine_scores: 正樣本（真實匹配）的分數列表/陣列
    :param imposter_scores: 負樣本（冒充匹配）的分數列表/陣列
    :param save_path: 圖片儲存路徑（例如 'roc_curve.png'），若為 None 則不儲存
    """
    # 1. 建立 Ground Truth 與 分數
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    y_scores = np.concatenate([genuine_scores, imposter_scores])

    # 2. 計算 ROC 曲線節點與 AUC 這是一個陣列
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

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

    return eer, best_acc

def get_pairwise_scores_split(embeddings, labels):
    n = embeddings.shape[0]
    idx = torch.randperm(n)

    half = n // 2
    idx_A = idx[:half]
    idx_B = idx[half:half*2]

    embeds_A = embeddings[idx_A]
    embeds_B = embeddings[idx_B]
    labels_A = labels[idx_A]
    labels_B = labels[idx_B]

    embeds_A = torch.nn.functional.normalize(embeds_A, p=2, dim=1)
    embeds_B = torch.nn.functional.normalize(embeds_B, p=2, dim=1)

    sim_matrix = torch.mm(embeds_A, embeds_B.t()).cpu().numpy()

    labels_A = labels_A.cpu().numpy()
    labels_B = labels_B.cpu().numpy()

    same = (labels_A[:, None] == labels_B[None, :])

    genuine = sim_matrix[same]
    imposter = sim_matrix[~same]

    return genuine, imposter

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

def main(args):
    configs.setup_seed(args.seed)
    test_dataset = datasets.ImagesDataset(args=args, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    feature_extractor, hash_generator = load_models(args)
    hash_user_list, labels_list = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Feature Extraction"):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            features = feature_extractor(imgs)
            h_user = hash_generator(features, labels, training=False)
            hash_user_list.append(h_user.cpu())
            labels_list.append(labels.cpu())

    hash_user    = torch.cat(hash_user_list, dim=0)
    labels       = torch.cat(labels_list, dim=0)

    user_gen, user_imp = get_pairwise_scores_split(hash_user, labels)
    user_eer, user_acc = compute_eer_and_save_roc(
        user_gen, user_imp)


    print(f"\n================ FSB-HashNet 評估結果 [{args.dataset}] ================")
    print(f"1. 驗證效能 (User-Specific Token EER) : {user_eer * 100:.4f}%")
    print(f"   驗證效能 (User-Specific Token ACC) : {user_acc * 100:.4f}%")
    print(f"======================================================")

if __name__ == '__main__':
    args = configs.get_all_params()
    args.dim = 1024
    args.hash_dim = 512
    for dataset in ['FV-USM', 'PLUSVein-FV3-LED','PLUSVein-FV3-LASER', 'UTFVP']:
        args.dataset = dataset
        args = configs.get_dataset_params(args)
        main(args)