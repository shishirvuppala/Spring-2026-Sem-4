#!/usr/bin/env python3
"""
q1_solve.py

Reference solution.
Students were expected to implement these functions.
"""

import numpy as np
from typing import List, Tuple


# -------------------------
# Utilities
# -------------------------
def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Add bias (column of ones) as first column.
    """
    N = X.shape[0]
    ones = np.ones((N, 1))
    return np.hstack([ones, X])


def mse(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean squared error.
    """
    return np.mean((y - y_pred) ** 2)


def standardize_train(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training statistics.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_std = (X - mean) / std
    return X_std, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply training standardization.
    """
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return (X - mean) / std_safe


# -------------------------
# Ridge Regression
# -------------------------
def ridge_regression_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form ridge regression:
        (X^T X + λD) w = X^T y
    where D[0,0] = 0 (bias not regularized).
    """
    N, D = X.shape

    XtX = X.T @ X
    Xty = X.T @ y

    Dmat = np.eye(D)
    Dmat[0, 0] = 0.0

    A = XtX + lam * Dmat
    w = np.linalg.solve(A, Xty)
    return w


# -------------------------
# Cross-validation
# -------------------------
def k_fold_split(N: int, k: int) -> List[np.ndarray]:
    indices = np.arange(N)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    return [fold for fold in folds]


def ridge_cv(X: np.ndarray, y: np.ndarray, lam: float, k: int) -> np.ndarray:
    """
    k-fold CV MSE for ridge.
    """
    N = X.shape[0]
    folds = k_fold_split(N, k)
    mses = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        w = ridge_regression_closed_form(X_train, y_train, lam)
        y_pred = X_val @ w
        mses.append(mse(y_val, y_pred))

    return np.mean(mses)


# -------------------------
# Hyperparameter search
# -------------------------
def grid_search_lambdas(
    X: np.ndarray, y: np.ndarray,
    lambdas: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate each λ using CV.
    """
    mses = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        mses[i] = ridge_cv(X, y, lam, k)

    return lambdas, mses


def random_search_lambdas(
    X: np.ndarray, y: np.ndarray,
    n_iter: int,
    low_exp: float,
    high_exp: float,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample λ = 10^u, where u ~ Uniform(low_exp, high_exp).
    """
    us = np.random.uniform(low_exp, high_exp, size=n_iter)
    lambdas = 10.0 ** us
    mses = np.zeros(n_iter)

    for i in range(n_iter):
        mses[i] = ridge_cv(X, y, lambdas[i], k)

    return lambdas, mses
