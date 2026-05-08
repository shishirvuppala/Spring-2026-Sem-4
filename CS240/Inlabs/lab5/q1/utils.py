"""
utils.py
========
Lab 5, Q1: Utility functions for data loading and evaluation metrics.

All functions must be implemented using only NumPy.
Do NOT import any other libraries.
"""

import numpy as np
from typing import Tuple


# ============================================
# DATA LOADING
# ============================================

def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CSV file where the last column is the target label.
    The first row is assumed to be a header and is skipped.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X (np.ndarray): Feature matrix of shape (N, D).
            - y (np.ndarray): Label vector of shape (N,).
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


# ============================================
# EVALUATION METRICS
# ============================================

def get_true_positives(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Count samples where both y_true and y_pred are 1 (True Positives).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        int: Number of true positive samples.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return int(tp)


def get_false_positives(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Count samples where y_true=0 but y_pred=1 (False Positives).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        int: Number of false positive samples.
    """
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return int(fp)


def get_false_negatives(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Count samples where y_true=1 but y_pred=0 (False Negatives).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        int: Number of false negative samples.
    """
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return int(fn)

def get_true_negatives(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Count samples where both y_true and y_pred are 0 (True Negatives).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        int: Number of true negative samples.
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return int(tn)


def get_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision: TP / (TP + FP).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        float: Precision score. Returns 0.0 if TP + FP == 0.
    """
    tp = get_true_positives(y_true, y_pred)
    fp = get_false_positives(y_true, y_pred)
    denom = tp + fp
    if denom == 0:
        return 0.0
    return float(tp / denom)


def get_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall (sensitivity): TP / (TP + FN).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        float: Recall score. Returns 0.0 if TP + FN == 0.
    """
    tp = get_true_positives(y_true, y_pred)
    fn = get_false_negatives(y_true, y_pred)
    denom = tp + fn
    if denom == 0:
        return 0.0
    return float(tp / denom)


def get_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the F1 score: 2 * (precision * recall) / (precision + recall).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        float: F1 score. Returns 0.0 if both precision and recall are zero.
    """
    prec = get_precision(y_true, y_pred)
    rec = get_recall(y_true, y_pred)
    denom = prec + rec
    if denom == 0:
        return 0.0
    return float(2.0 * prec * rec / denom)

def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy: fraction of correctly classified samples.
    Works for both binary and multi-class labels.

    Args:
        y_true (np.ndarray): True labels of shape (N,).
        y_pred (np.ndarray): Predicted labels of shape (N,).

    Returns:
        float: Accuracy in [0, 1].
    """
    return float(np.mean(y_true == y_pred))