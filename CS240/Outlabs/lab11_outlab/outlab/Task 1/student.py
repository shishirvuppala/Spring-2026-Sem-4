"""
Read every docstring carefully before implementing.
Do NOT rename any class, method, or variable — the autograder depends on them.
"""

import numpy as np
from typing import List
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# ===========================================================================
# Training data loader (students only get TRAIN data)
# ===========================================================================

def load_training_data(path="train.npz"):
    """
    Load training data from disk.
    """
    data = np.load(path)
    X_train = data["X"]
    y_train = data["y"]
    return X_train, y_train


# ===========================================================================
# PART A  —  Loss functions
# ===========================================================================

def huber_loss(residuals: np.ndarray, delta: float) -> float:
    """
    Compute the MEAN Huber loss over an array of residuals.
    
    Returns
    -------
    float — mean Huber loss across all residuals
    """
    # TODO
    raise NotImplementedError


def huber_pseudo_residuals(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           delta: float) -> np.ndarray:
    """
    Compute the negative gradient of the Huber loss w.r.t. y_pred.

    Returns
    -------
    np.ndarray — pseudo-residuals, same shape as y_true
    """
    # TODO
    raise NotImplementedError


# ===========================================================================
# PART B  —  GradientBoostingEnsemble class
# ===========================================================================

class GradientBoostingEnsemble:
    """
    A from-scratch gradient boosting ensemble supporting regression
    using the Huber loss.

    The model has the form:
        f(x) = sum_{m=1}^{M} learning_rate(m) * F_m(x)

    where:
        - M is the number of boosting stages
        - F_m is the m-th weak learner fitted on pseudo-residuals
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta
        self.models_: List[BaseEstimator] = []
        self.initial_prediction_: float = 0.0
        self.scaler_ = StandardScaler()  # For feature scaling (if needed)

    # ------------------------------------------------------------------
    # B1 · learning_rate
    # ------------------------------------------------------------------
    def learning_rate(self, m: int) -> float:
        """
        Return the coefficient (learning rate) for the m-th boosting stage.

        Returns
        -------
        float — positive learning rate for stage m
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # B2 · _compute_pseudo_residuals
    # ------------------------------------------------------------------
    def _compute_pseudo_residuals(self,
                                  y_true: np.ndarray,
                                  f_pred: np.ndarray) -> np.ndarray:
        """
        Compute the pseudo-residuals (negative gradients) for the current stage.

        Returns
        -------
        np.ndarray — pseudo-residuals
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # B3 · fit
    # ------------------------------------------------------------------
    def fit(self,
            weak_learners: List[BaseEstimator],
            X: np.ndarray,
            y: np.ndarray) -> "GradientBoostingEnsemble":
        """
        Fit the boosting ensemble using the provided list of weak learners.

        Algorithm
        ---------
        1. Initialise f(x) = mean(y)
        2. For each weak learner:
             a. Compute pseudo-residuals
             b. Fit learner on (X, residuals)
             c. Update ensemble prediction
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # B4 · predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate final predictions for new data.

        Returns
        -------
        np.ndarray — predictions, shape (n_samples,)
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------------------------------
    # B5 · staged_loss
    # ------------------------------------------------------------------
    def staged_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the loss after each boosting stage on dataset (X, y).

        Returns
        -------
        np.ndarray — loss after each stage
        """
        # TODO
        raise NotImplementedError


# ===========================================================================
# PART C  —  Weak learner design
# ===========================================================================
#
# You may choose and reorder weak learners freely,
# subject to the constraints below.
#
# - At least 5 learners
# - Must be sklearn regressors
# - Must obey ALLOWED_MODEL_SPECS
# - No pre-built boosting models allowed
#

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ===========================================================================
# Allowed weak learner specifications (DO NOT MODIFY)
# ===========================================================================

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

# ===========================================================================
# Define your weak learners here
# ===========================================================================

REGRESSION_LEARNERS: List[BaseEstimator] = [
    # TODO: populate with at least 5 valid weak learners
]
