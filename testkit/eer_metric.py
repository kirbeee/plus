import abc

from sklearn.model_selection import KFold
import numpy as np
import sklearn.metrics as skm
from tqdm.contrib import tzip
import random
import torch
dist_type='cosine'

# 來自學長程式碼
def calculate_metrics(distances, labels, threshold):
    if dist_type == 'cosine':
        preds = np.greater(distances, threshold)
    elif dist_type == 'euclidean':
        preds = np.less(distances, threshold)

    tn, fp, fn, tp = skm.confusion_matrix(labels, preds).ravel()
    fpr = float(fp) / (tn + fp) * 100
    fnr = float(fn) / (tp + fn) * 100
    acc = float(tp + tn) / distances.size * 100
    return acc, fpr, fnr
def calculate_average_metrics(dists, labels, num_folds=5):
    dist_min, dist_max = np.min(dists), np.max(dists)
    thresholds = np.arange(0, np.ceil(dist_max), 0.01)
    print(f'Distance Min: {dist_min} Max: {dist_max}')
    eer_list = []
    acc_list = []
    folds = KFold(n_splits=num_folds, shuffle=True)
    for train_set, test_set in folds.split(labels):
        _acc_fold = []
        _fpr_fold = []
        _fnr_fold = []
        for threshold in thresholds:
            acc, fpr, fnr = calculate_metrics(dists[train_set], labels[train_set], threshold)
            _acc_fold.append(acc)
            _fpr_fold.append(fpr)
            _fnr_fold.append(fnr)
        eer_idx = np.nanargmin(np.absolute((np.array(_fnr_fold) - np.array(_fpr_fold))))
        eer = (_fpr_fold[eer_idx] + _fnr_fold[eer_idx]) / 2

        best_threshold = thresholds[np.argmax(_acc_fold)]
        acc, fpr, fnr = calculate_metrics(dists[test_set], labels[test_set], best_threshold)

        eer_list.append(eer)
        acc_list.append(acc)
    return np.mean(acc_list), np.mean(eer_list)
# ------------
class BaseMetric(object):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def compute(self, embeddings, labels, **kwargs):
        pass

class EERMetric(BaseMetric):
    def __init__(self, name="EER & Best ACC"):
        super(EERMetric, self).__init__(name)

    def compute(self, embeddings, labels, **kwargs):
        # Convert PyTorch tensors to NumPy arrays if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        dists = []
        pair_labels = []  # Use a separate list to avoid overwriting the input parameter

        for embed_A, target_A in tzip(embeddings, labels):
            for embed_B, target_B in zip(embeddings, labels):
                if dist_type == 'cosine':
                    dist = np.dot(embed_A, embed_B) / (np.linalg.norm(embed_A) * np.linalg.norm(embed_B))
                    dist = (dist + 1) / 2
                elif dist_type == 'euclidean':
                    dist = np.sum((embed_A - embed_B) ** 2) ** 0.5

                label_val = int(target_A == target_B)
                dists.append(dist)
                pair_labels.append(label_val)

        dists = np.array(dists, dtype=np.float32)
        pair_labels = np.array(pair_labels, dtype=np.float32)

        dists_1 = dists[pair_labels == 1]
        dists_2 = dists[pair_labels == 0]
        random.shuffle(dists_2)
        dists_2 = dists_2[:len(dists_1)]
        dists = np.hstack([dists_1, dists_2])

        labels_1 = pair_labels[pair_labels == 1]
        labels_2 = pair_labels[pair_labels == 0]
        random.shuffle(labels_2)
        labels_2 = labels_2[:len(labels_1)]
        pair_labels = np.hstack([labels_1, labels_2])

        acc, eer = calculate_average_metrics(dists, pair_labels)

        # Return a dictionary to match the iteration logic in main()
        return {"EER": eer, "ACC": acc}