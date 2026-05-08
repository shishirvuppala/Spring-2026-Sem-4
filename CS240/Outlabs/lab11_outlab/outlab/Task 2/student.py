"""
KNORA-U — Dynamic Ensemble Selection for Regression (with optional Bagging)

Read every docstring carefully before implementing.
Do NOT rename any class, method, or variable — the autograder depends on them.
"""

import numpy as np
from typing import List, Tuple
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# Training data loader (students only get TRAIN data)
# ===========================================================================

def load_training_data(path="../train.npz"):
    """
    Load training data.

    NOTE:
    - Students only get access to TRAIN data.
    - DSEL must be constructed from training data only.
    """
    data = np.load(path)
    return data["X"], data["y"]


# ===========================================================================
# Utility functions
# ===========================================================================

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.

    Parameters
    ----------
    a, b : np.ndarray of shape (n_features,)

    Returns
    -------
    float
    """
    # TODO:
    # Return ||a - b||_2
    raise NotImplementedError


def knn_indices(X: np.ndarray, x: np.ndarray, K: int) -> np.ndarray:
    """
    Find indices of the K nearest neighbors of x in X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    x : np.ndarray, shape (n_features,)
    K : int

    Returns
    -------
    np.ndarray — indices of K nearest neighbors
    """
    # TODO:
    # Compute distances from x to all rows of X
    # Return indices of K smallest distances
    raise NotImplementedError


# ===========================================================================
# Bagging utilities
# ===========================================================================

def bootstrap_sample(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a bootstrap sample from (X, y).

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    random_state : int

    Returns
    -------
    X_boot, y_boot : np.ndarray
    """
    # TODO:
    # Sample len(X) points WITH replacement
    # Use np.random.default_rng(random_state)
    raise NotImplementedError


def fit_with_optional_bagging(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    use_bagging: bool,
    random_state: int
) -> BaseEstimator:
    """
    Fit a model either normally or using bagging.

    Parameters
    ----------
    model : BaseEstimator (unfitted)
    X, y : training data
    use_bagging : bool
        Whether to apply bagging
    random_state : int

    Returns
    -------
    fitted model
    """
    # TODO:
    # 1. Clone the model
    # 2. If use_bagging:
    #       - draw bootstrap sample
    #       - fit on bootstrap data
    #    else:
    #       - fit on full data
    raise NotImplementedError


# ===========================================================================
# KNORA-U Regressor
# ===========================================================================

class KNORAURegressor:
    """
    K-Nearest Oracles Union (KNORA-U) for regression.

    Dynamic ensemble selection based on local competence.
    """

    def __init__(self, K: int = 7, epsilon: float = 1.0):
        """
        Parameters
        ----------
        K : int
            Initial neighborhood size
        epsilon : float
            Local error tolerance
        """
        self.K = K
        self.epsilon = epsilon

        self.models_: List[BaseEstimator] = []
        self.X_dsel_: np.ndarray = None
        self.y_dsel_: np.ndarray = None
        self.scaler_ = StandardScaler()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        models: List[BaseEstimator],
        X_dsel: np.ndarray,
        y_dsel: np.ndarray
    ) -> "KNORAURegressor":
        """
        Store the fitted model pool and DSEL dataset.

        Parameters
        ----------
        models : list of FITTED regressors
        X_dsel : np.ndarray
        y_dsel : np.ndarray

        Returns
        -------
        self
        """
        # TODO:
        # Store models, X_dsel, y_dsel
        raise NotImplementedError

    # ------------------------------------------------------------------
    # _select_models
    # ------------------------------------------------------------------
    def _select_models(self, x: np.ndarray) -> List[BaseEstimator]:
        """
        Select locally competent models for a single test point x
        using the KNORA-U algorithm.

        Algorithm
        ---------
        1. Find K nearest neighbors of x in DSEL
        2. For each model:
             - compute MAE on these neighbors
        3. Select all models with MAE ≤ epsilon
        4. If none selected:
             - select the single best model (lowest MAE)

        Returns
        -------
        list of selected models (non-empty)
        """
        # TODO:
        # Implement KNORA-U exactly as described
        raise NotImplementedError

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using dynamic model selection.

        For each test point:
        - select models using _select_models
        - return average prediction

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        # TODO:
        # Loop over samples, apply KNORA-U, average predictions
        raise NotImplementedError


# ===========================================================================
# Allowed base learners (provided, not implemented by students)
# ===========================================================================

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

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
# PART D — Student-defined model pool
# ===========================================================================
#
# Rules enforced by autograder:
# - ≥ 5 models
# - Each model must be fitted
# - If same architecture appears multiple times:
#     → bagging MUST be used
#

REGRESSION_MODELS: List[BaseEstimator] = []

# TODO:
# 1. Load training data
# 2. Define base (unfitted) models
# 3. Decide which models use bagging
# 4. Fit all models using fit_with_optional_bagging
# 5. Populate REGRESSION_MODELS