"""
utils.py
========
Lab 5, Q2: Utility functions for data loading and evaluation metrics.

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

    Hint:
        - Use np.sum with boolean/logical operations.
        - (y_true == 1) produces a boolean array; & performs element-wise AND
          on two boolean arrays.  So (y_true == 1) & (y_pred == 1) is True at
          positions where both conditions hold.
        - np.sum on a boolean array counts the number of True values.
        - Wrap the result with int() to return a plain Python int.
    """
    # TODO: Implement
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return int(tp)
    # END TODO


def get_false_positives(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Count samples where y_true=0 but y_pred=1 (False Positives).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        int: Number of false positive samples.

    Hint:
        - Similar to get_true_positives but check (y_true == 0) & (y_pred == 1).
        - Wrap with int() so the return type is a plain Python integer.
    """
    # TODO: Implement
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return int(fp)
    # END TODO


def get_false_negatives(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Count samples where y_true=1 but y_pred=0 (False Negatives).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        int: Number of false negative samples.

    Hint:
        - Check (y_true == 1) & (y_pred == 0).
        - Wrap with int() so the return type is a plain Python integer.
    """
    # TODO: Implement
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return int(fn)
    # END TODO


def get_true_negatives(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Count samples where both y_true and y_pred are 0 (True Negatives).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        int: Number of true negative samples.

    Hint:
        - Check (y_true == 0) & (y_pred == 0).
        - Wrap with int() so the return type is a plain Python integer.
    """
    # TODO: Implement
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return int(tn)
    # END TODO


def get_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute precision: TP / (TP + FP).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        float: Precision score. Returns 0.0 if TP + FP == 0.

    Hint:
        - Use get_true_positives and get_false_positives (call the functions
          you already implemented).
        - Handle the edge case where the denominator is zero to avoid
          ZeroDivisionError: check if denom == 0 before dividing.
        - Wrap the result with float() to return a plain Python float.
    """
    # TODO: Implement
    tp = get_true_positives(y_true, y_pred)
    fp = get_false_positives(y_true, y_pred)
    denom = tp + fp
    if denom == 0:
        return 0.0
    return float(tp / denom)
    # END TODO


def get_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall (sensitivity): TP / (TP + FN).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        float: Recall score. Returns 0.0 if TP + FN == 0.

    Hint:
        - Use get_true_positives and get_false_negatives (call the functions
          you already implemented).
        - Handle the edge case where the denominator is zero.
        - Wrap the result with float().
    """
    # TODO: Implement
    tp = get_true_positives(y_true, y_pred)
    fn = get_false_negatives(y_true, y_pred)
    denom = tp + fn
    if denom == 0:
        return 0.0
    return float(tp / denom)
    # END TODO


def get_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the F1 score: 2 * (precision * recall) / (precision + recall).

    Args:
        y_true (np.ndarray): True binary labels of shape (N,). Values in {0, 1}.
        y_pred (np.ndarray): Predicted binary labels of shape (N,). Values in {0, 1}.

    Returns:
        float: F1 score. Returns 0.0 if both precision and recall are zero.

    Hint:
        - Use get_precision and get_recall (call the functions you already
          implemented).
        - Handle the edge case where precision + recall == 0.
        - Wrap the result with float().
    """
    # TODO: Implement
    prec = get_precision(y_true, y_pred)
    rec = get_recall(y_true, y_pred)
    denom = prec + rec
    if denom == 0:
        return 0.0
    return float(2.0 * prec * rec / denom)
    # END TODO


def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy: fraction of correctly classified samples.
    Works for both binary and multi-class labels.

    Args:
        y_true (np.ndarray): True labels of shape (N,).
        y_pred (np.ndarray): Predicted labels of shape (N,).

    Returns:
        float: Accuracy in [0, 1].

    Hint:
        - (y_true == y_pred) gives a boolean array that is True wherever
          the prediction matches the ground truth.
        - np.mean on a boolean array returns the fraction of True values,
          which is exactly the accuracy.
        - Wrap the result with float().
    """
    # TODO: Implement
    return float(np.mean(y_true == y_pred))
    # END TODO
