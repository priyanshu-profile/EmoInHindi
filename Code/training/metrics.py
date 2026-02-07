import torch
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, accuracy_score


def binarize_predictions(preds, threshold=0.5):
    """
    Convert probabilities to binary predictions
    """
    return (preds >= threshold).astype(int)


def compute_multilabel_metrics(y_true, y_pred, threshold=0.5):
    """
    Metrics for multi-label emotion classification
    """
    y_pred_bin = binarize_predictions(y_pred, threshold)

    metrics = {
        "micro_f1": f1_score(y_true, y_pred_bin, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred_bin),
        "jaccard_index": jaccard_score(y_true, y_pred_bin, average="samples", zero_division=0),
    }
    return metrics


def compute_single_label_metrics(y_true, y_pred):
    """
    Metrics for single-label emotion or intensity classification
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }
