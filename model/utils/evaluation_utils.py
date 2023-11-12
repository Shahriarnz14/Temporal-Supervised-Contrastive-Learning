import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def get_evaluation_metrics(y_true, y_pred):
    auroc = get_auroc(y_true, y_pred)
    auprc = get_auprc(y_true, y_pred)
    return {"AUROC": auroc, "AUPRC": auprc}

def get_auroc(y_true, y_pred):
    y_true_one_hot = to_one_hot(y_true)
    auroc = np.array([roc_auc_score(y_true_one_hot[:, y_idx], y_pred[:, y_idx]) for y_idx in range(y_true_one_hot.shape[1])])
    return auroc

def get_auprc(y_true, y_pred):
    y_true_one_hot = to_one_hot(y_true)
    auprc = np.array([average_precision_score(y_true_one_hot[:, y_idx], y_pred[:, y_idx]) for y_idx in range(y_true_one_hot.shape[1])])
    return auprc


def to_one_hot(labels):
    max_class = len(np.unique(labels))
    one_hot_labels = np.zeros((len(labels), max_class))
    for i, lb in enumerate(labels):
        one_hot_labels[i, int(lb)] = 1
    return one_hot_labels