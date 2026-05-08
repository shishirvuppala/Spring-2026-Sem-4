import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Part A: Data Loading and Preprocessing
# ---------------------------------------------------------------------------

def load_data(path: str = "dataset.npz") -> dict:
    """
    Load the dataset and apply standard scaling.

    Parameters
    ----------
    path : str
        Path to the .npz file containing the keys
        'X_train', 'y_train', 'X_test', 'y_test'.

    Returns
    -------
    dict with the following keys and value types:
        "X_train" : np.ndarray of shape (n_train, 2)   — scaled training features
        "y_train" : np.ndarray of shape (n_train,)      — training labels (0 or 1)
        "X_test"  : np.ndarray of shape (n_test, 2)     — scaled test features
        "y_test"  : np.ndarray of shape (n_test,)        — test labels (0 or 1)
        "scaler"  : sklearn.preprocessing.StandardScaler — the fitted scaler

    Notes
    -----
    - Fit the StandardScaler on the training set only, then transform both
      training and test sets.
    """
    # TODO: implement this function
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part B: Training and Evaluation Helpers
# ---------------------------------------------------------------------------

def train_svm(X_train: np.ndarray,
              y_train: np.ndarray,
              kernel: str = "rbf",
              C: float = 1.0,
              gamma = "scale",
              degree: int = 3) -> SVC:
    """
    Train an SVM classifier with the given hyperparameters.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, 2)
    y_train : np.ndarray of shape (n_train,)
    kernel  : str   — one of 'linear', 'rbf', 'poly'
    C       : float — regularisation parameter
    gamma   : float | str — kernel coefficient ('scale', 'auto', or a float)
    degree  : int   — degree for the polynomial kernel (ignored by other kernels)

    Returns
    -------
    sklearn.svm.SVC — the fitted SVM model
    """
    # TODO: implement this function
    raise NotImplementedError


def evaluate_model(model: SVC,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    """
    Evaluate a trained SVM and return performance metrics.

    Parameters
    ----------
    model  : sklearn.svm.SVC — a trained SVM classifier
    X_test : np.ndarray of shape (n_test, 2)
    y_test : np.ndarray of shape (n_test,)

    Returns
    -------
    dict with the following keys and value types:
        "accuracy"           : float — classification accuracy on the test set
        "n_support_vectors"  : int   — total number of support vectors
        "n_support_per_class": list[int] — number of SVs per class
    """
    return {
        "accuracy": None,
        "n_support_vectors": None,
        "n_support_per_class": None,
    }
    # TODO: implement this function
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part C: Hyperparameter Search
# ---------------------------------------------------------------------------

def find_best_config(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     accuracy_threshold: float = 0.95) -> dict:
    """
    Search over kernels, C, and gamma to find the configuration that
    achieves accuracy >= accuracy_threshold on the test set while
    using the fewest support vectors.

    You may use any search strategy you like (grid search, random search,
    manual tuning, etc.).  Only scikit-learn's SVC is permitted.
    You can set unused parameters to None in the returned dict.

    Parameters
    ----------
    X_train, y_train : training data (already scaled)
    X_test, y_test   : test data (already scaled)
    accuracy_threshold : float — minimum required accuracy

    Returns
    -------
    dict with the following keys and value types:
        "kernel"            : str   — best kernel name
        "C"                 : float — best C value
        "gamma"             : float | str — best gamma value
        "degree"            : int   — best degree (only relevant for 'poly')
        "accuracy"          : float — achieved test accuracy
        "n_support_vectors" : int   — total number of support vectors
        "model"             : sklearn.svm.SVC — the best trained model
    """
    # TODO: implement this function
    return {
        "kernel": None,
        "C": None,
        "gamma": None,
        "degree": None,
        "accuracy": None,
        "n_support_vectors": None,
        "model": None,
    }
    raise NotImplementedError
