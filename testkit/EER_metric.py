import abc

from sklearn.model_selection import KFold
import numpy as np
import sklearn.metrics as skm

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

class BaseMetric(object):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def compute(self, embeddings, labels, **kwargs):
        pass

class EERMetric(BaseMetric):
    def __init__(self, name):
        super(EERMetric, self).__init__("EER & Best ACC")

    def compute(self, embeddings, labels, **kwargs):
        print(embeddings.shape)
        print(labels.shape)
        return