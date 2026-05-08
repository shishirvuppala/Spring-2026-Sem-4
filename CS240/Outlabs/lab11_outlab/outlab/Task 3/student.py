"""
Task 3 — Open-Ended Ensemble Fusion
=====================================
You have two baseline predictors from Tasks 1 and 2.
Combine them (and anything else you like) to get the lowest MAE on test data.

Rules
-----
* Edit THIS file only. Do not rename task3_predict or FUSION_LEARNERS.
* No pre-built boosting libraries (XGBoost, LightGBM, GradientBoostingRegressor, …).
* Allowed base learners: same weak-learner constraint as Task 2.
* Do NOT use test labels at any point.
"""

import numpy as np
from typing import List
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ------------------------------------------------------------------
# Re-import your Task 1 and Task 2 systems.
# Adjust import paths to match your submission structure.
# ------------------------------------------------------------------
# from task1_student import GradientBoostingEnsemble, REGRESSION_LEARNERS
# from task2_student import KNORAURegressor, REGRESSION_MODELS


# ===========================================================================
# FUSION_LEARNERS — pool of weak learners available to your fusion
# ===========================================================================
# Rules:
#   • ≥ 5 unfitted estimators
#   • Only allowed base learners (same as Task 2)
#   • No pre-built boosting models

ALLOWED_MODEL_SPECS = {
    DecisionTreeRegressor: {
        "max_depth": {1, 2, 3}
    },
    LinearRegression: {},
    Ridge: {
        "alpha": (1e-4, 1e3)
    },
    Lasso: {
        "alpha": (1e-4, 1e1)
    },
    KNeighborsRegressor: {
        "n_neighbors": {5, 10}
    },
    SVR: {
        "kernel": {"rbf"},
        "C": (0.1, 10.0),
        "epsilon": (0.01, 1.0),
    },
}

FUSION_LEARNERS: List[BaseEstimator] = [
    # TODO: add at least 5 weak learners
]


# ===========================================================================
# task3_predict — entry point called by the autograder
# ===========================================================================

def task3_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
) -> np.ndarray:
    """
    Train your fusion strategy and return predictions for X_test.

    Parameters
    ----------
    X_train : (n_train, n_features)
    y_train : (n_train,)
    X_test  : (n_test,  n_features)

    Returns
    -------
    np.ndarray of shape (n_test,)

    Fusion ideas (pick one or invent your own)
    ------------------------------------------
    A) Simple weighted average of Task 1 and Task 2 predictions
         y_hat = w * y_task1 + (1 - w) * y_task2

    B) Learned weighted average — fit a Ridge on [y_task1, y_task2] using a
       validation split of the training data, then predict with it.

    C) Stacking — train a meta-learner on predictions from all FUSION_LEARNERS
       using cross-validation, then predict on test.

    D) Dynamic per-sample weighting — weight models based on local error
       estimates in the neighborhood of each test point.

    E) Anything else you can think of or find in the literature (or not in the literature ;) ).
    """
    # TODO: implement your fusion strategy here.
    raise NotImplementedError
