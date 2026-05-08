#!/usr/bin/env python3
"""
q1_solve.py

Implement ALL functions below.
Do NOT import sklearn.
"""

import numpy as np
from typing import List, Tuple


# -------------------------
# Utilities
# -------------------------
def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Add bias (column of ones) as first column.
    See main.py for usage.
    """
    # TODO
    raise NotImplementedError


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    # TODO
    raise NotImplementedError


def standardize_train(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training statistics.
    Returns standardized X, mean, and stddev.
    See main.py for usage.
    """
    # TODO
    raise NotImplementedError


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply training standardization.
    See main.py for usage.
    """
    # TODO
    raise NotImplementedError


# -------------------------
# Ridge Regression
# -------------------------
def ridge_regression_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form ridge regression:
        (X^T X + λD) w = X^T y
    where D[0,0] = 0 (bias not regularized).
    """
    # TODO
    raise NotImplementedError


# -------------------------
# Cross-validation
# -------------------------
def k_fold_split(N: int, k: int) -> List[np.ndarray]:
    """
        k-fold split after shuffling
        Returns list of k arrays of indices.
    """
    # TODO
    raise NotImplementedError


def ridge_cv(X: np.ndarray, y: np.ndarray, lam: float, k: int) -> float:
    """
    k-fold CV MSE for ridge.
    Use the k_fold_split function above to get the folds
    Use ridge_regression_closed_form to fit the model.
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        lam: regularization parameter
        k: number of folds
    Returns average MSE across folds.
    """
    # TODO
    raise NotImplementedError


# -------------------------
# Hyperparameter search
# -------------------------
def grid_search_lambdas(
    X: np.ndarray, y: np.ndarray,
    lambdas: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate each λ using CV.
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        lambdas: (M,) array of λ values to evaluate
        k: number of folds
    Returns:
        lambdas: (M,) same as input
        mses: (M,) average CV MSE for each λ
    """
    # TODO
    raise NotImplementedError


def random_search_lambdas(
    X: np.ndarray, y: np.ndarray,
    n_iter: int,
    low_exp: float,
    high_exp: float,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample λ = 10^u, where u ~ Uniform(low_exp, high_exp).
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        n_iter: number of λ values to sample
        low_exp: lower bound of exponent
        high_exp: upper bound of exponent
        k: number of folds
    Returns:
        lambdas: (n_iter,) sampled λ values
        mses: (n_iter,) average CV MSE for each λ
    """
    # TODO
    raise NotImplementedError
