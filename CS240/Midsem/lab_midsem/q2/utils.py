"""
utils.py — Evaluation and diagnostic utilities for LinearRegressionTree.

All functions are pure NumPy — no sklearn, no scipy.

Contents
--------
  compute_mse     : Mean Squared Error
  compute_rmse    : Root Mean Squared Error
  compute_mae     : Mean Absolute Error
  compute_r2      : Coefficient of Determination  (R^2)
  compute_metrics : Compute all four metrics at once → dict
  print_metrics   : Pretty-print a metrics dict
  count_tree_nodes: DFS node/leaf counter
  get_leaf_params : Collect (x_bar, y_bar, w) from every leaf
"""

import numpy as np


# ======================================================================
#  Regression Metrics
# ======================================================================

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) or (N, 1)
    y_pred : np.ndarray, shape (N,) or (N, 1)

    Returns
    -------
    float
        MSE = (1/N) * sum_i (y_true_i - y_pred_i)^2
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) or (N, 1)
    y_pred : np.ndarray, shape (N,) or (N, 1)

    Returns
    -------
    float
        RMSE = sqrt(MSE)
    """
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) or (N, 1)
    y_pred : np.ndarray, shape (N,) or (N, 1)

    Returns
    -------
    float
        MAE = (1/N) * sum_i |y_true_i - y_pred_i|
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination  (R^2).

    R^2 = 1  -  SS_res / SS_tot

    where  SS_res = sum_i (y_true_i - y_pred_i)^2
           SS_tot = sum_i (y_true_i - mean(y_true))^2

    A value of 1.0 indicates a perfect fit.  A value <= 0 means the
    model is no better (or worse) than predicting the mean.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) or (N, 1)
    y_pred : np.ndarray, shape (N,) or (N, 1)

    Returns
    -------
    float
        R^2 score in (-inf, 1].
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        # Degenerate case: constant target
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MSE, RMSE, MAE, and R^2 in a single call.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,) or (N, 1)
    y_pred : np.ndarray, shape (N,) or (N, 1)

    Returns
    -------
    dict with keys: "mse", "rmse", "mae", "r2"
    """
    return {
        "mse":  compute_mse(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mae":  compute_mae(y_true, y_pred),
        "r2":   compute_r2(y_true, y_pred),
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    """
    Pretty-print a metrics dict returned by ``compute_metrics``.

    Parameters
    ----------
    metrics : dict
        Keys: "mse", "rmse", "mae", "r2"
    label : str
        Optional header string (e.g. "Train" or "Test").
    """
    header = f"  {label}" if label else ""
    print(f"{header}")
    print(f"    MSE  : {metrics['mse']:>12.6f}")
    print(f"    RMSE : {metrics['rmse']:>12.6f}")
    print(f"    MAE  : {metrics['mae']:>12.6f}")
    print(f"    R^2  : {metrics['r2']:>12.6f}")


# ======================================================================
#  Tree Inspection Helpers
# ======================================================================

def count_tree_nodes(root) -> dict:
    """
    Count total nodes and leaf nodes in the tree via DFS.

    Parameters
    ----------
    root : Node
        Root node of a trained LinearRegressionTree.

    Returns
    -------
    dict with keys:
        "total"  : int — total number of nodes
        "leaves" : int — number of leaf nodes
        "internal" : int — number of internal (split) nodes
        "depth"  : int — maximum depth of any leaf (root = depth 0)
    """
    def _dfs(node, depth):
        if node is None:
            return 0, 0, 0
        if node.is_leaf:
            return 1, 1, depth
        lt, ll, ld = _dfs(node.left, depth + 1)
        rt, rl, rd = _dfs(node.right, depth + 1)
        return 1 + lt + rt, ll + rl, max(ld, rd)

    total, leaves, max_depth = _dfs(root, 0)
    return {
        "total":    total,
        "leaves":   leaves,
        "internal": total - leaves,
        "depth":    max_depth,
    }


def get_leaf_params(root) -> list:
    """
    Collect the local linear model parameters from every leaf node.

    Parameters
    ----------
    root : Node
        Root node of a trained LinearRegressionTree.

    Returns
    -------
    list of dict, one entry per leaf:
        {
          "x_bar" : np.ndarray, shape (d,),
          "y_bar" : float,
          "w"     : np.ndarray, shape (d,),
        }
    """
    params = []

    def _dfs(node):
        if node is None:
            return
        if node.is_leaf:
            params.append({
                "x_bar": node.x_bar,
                "y_bar": node.y_bar,
                "w":     node.w,
            })
        else:
            _dfs(node.left)
            _dfs(node.right)

    _dfs(root)
    return params
