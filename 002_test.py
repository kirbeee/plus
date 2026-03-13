import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sklearn.metrics as skm
import configs
import datasets
from model.model import MinusBackbone


def load_backbone(args):
    """加載模型並載入 Stage 2 訓練後的權重"""
    model = MinusBackbone(mode='stage2').to(args.device)

    # 根據 001_train.py 的保存邏輯載入權重
    gen_path = 'weights/best_generator.pth'
    rec_path = 'weights/best_recognizer.pth'

    if os.path.exists(gen_path) and os.path.exists(rec_path):
        model.generator.load_state_dict(torch.load(gen_path, map_location=args.device))
        model.recognizer.load_state_dict(torch.load(rec_path, map_location=args.device))
        print("成功載入 Generator 與 Recognizer 權重")
    else:
        print("警告：找不到權重檔案，將使用隨機初始化模型")

    model.eval()
    return model


def calculate_eer(labels, scores):
    """計算等錯誤率 (EER)"""
    fpr, tpr, thresholds = skm.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold


def evaluate(args, model, test_loader):
    embeds_list = []
    targets_list = []

    print("正在提取特徵...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(args.device)
            # 取得 stage 2 的混淆後殘差特徵
            _, _, x_feature, _ = model(imgs)
            # 正規化特徵以進行 Cosine Similarity 計算
            x_feature = F.normalize(x_feature, p=2, dim=1)

            embeds_list.append(x_feature.cpu())
            targets_list.append(labels.cpu())

    embeddings = torch.cat(embeds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    print("正在計算相似度矩陣...")
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

    print("正在計算指標...")
    eer, threshold = calculate_eer(actual_labels, scores)

    # 計算特定閾值下的準確度
    preds = (scores >= threshold).astype(int)
    acc = skm.accuracy_score(actual_labels, preds)

    return eer, acc


def main():
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)

    # 針對 LED 和 LASER 兩種光源分別測試
    results = {}
    for data_type in ['LED', 'LASER']:
        print(f"\n--- 測試數據類型: {data_type} ---")
        test_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        model = load_backbone(args)
        eer, acc = evaluate(args, model, test_loader)

        results[data_type] = {'EER': f"{eer * 100:.2f}%", 'ACC': f"{acc * 100:.2f}%"}
        print(f"結果 [{data_type}]: EER = {eer * 100:.2f}%, ACC = {acc * 100:.2f}%")

    print("\n最終總結:", results)


if __name__ == '__main__':
    main()