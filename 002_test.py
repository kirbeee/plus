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
import matplotlib.pyplot as plt
from torchkit.head.localfc.arcface import ArcFace


def load_backbone(args):

    model = MinusBackbone(mode=args.mode).to(args.device)
    arcface_head = ArcFace(in_features=512, out_features=360).to(args.device)

    gen_path = 'weights/best_generator.pth'
    rec_path = 'weights/best_recognizer.pth'
    arcface_path = 'weights/best_arcface_head.pth'

    if os.path.exists(gen_path) and os.path.exists(rec_path):
        model.generator.load_state_dict(torch.load(gen_path, map_location=args.device))
        model.recognizer.load_state_dict(torch.load(rec_path, map_location=args.device))
        arcface_head.load_state_dict(torch.load(arcface_path, map_location=args.device))
    else:
        raise("import model failed!")

    model.eval()
    arcface_head.eval()
    return model, arcface_head


def calculate_eer(labels, scores):
    """計算等錯誤率 (EER)"""
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

def main():
    results = {}
    print(f"\n--- 測試數據類型: LED ---")
    test_dataset = datasets.ImagesDataset(args=args, data_type="LED", phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model, arc_face = load_backbone(args)
    acc = evaluate_with_classification(args, model,arc_face, test_loader)

    results["LED"] = {'ACC': f"{acc * 100:.2f}%"}
    print(f"結果 [LED]: ACC = {acc * 100:.2f}%")


if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.datasets = "PLUSVein-FV3"
    args = configs.get_dataset_params(args)
    args.mode = "stage2"
    main()