"""
setup_data.py — Generate and persist the Segment regression dataset.

The Segment dataset models a piecewise-linear terrain surface defined
over a 2-D feature space [0, 10) x [0, 10), partitioned into four
Cartesian quadrants.  Each quadrant is governed by a distinct linear
model (different slopes and intercept).  Additive Gaussian noise
simulates observation error.

Run this script once to create the data/ directory structure and
save train/test CSV files that main.py will consume.
"""

import numpy as np
import os


# ======================================================================
#  Data Generation
# ======================================================================

def generate_segment_dataset(n_samples: int = 2000,
                             noise_std: float = 1.5) -> tuple:
    """
    Generate a synthetic piecewise-linear regression dataset on 2-D input.

    The feature space is divided into four quadrants with boundaries at
    x1 = 5 and x2 = 5.  The true data-generating process is:

        Q1  (x1 <= 5, x2 <= 5):  y =  3 x1 + 4 x2 + 10
        Q2  (x1 >  5, x2 <= 5):  y = -2 x1 + 8 x2 + 35
        Q3  (x1 <= 5, x2 >  5):  y =  7 x1 - 3 x2 + 45
        Q4  (x1 >  5, x2 >  5):  y = -5 x1 - 6 x2 + 120

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    noise_std : float
        Standard deviation of the additive Gaussian noise N(0, sigma).

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
        Feature matrix with columns [x1, x2], sampled uniformly
        from [0, 10).
    y : np.ndarray, shape (n_samples, 1)
        Target column vector including noise.
    """
    np.random.seed(42)

    # 2-D features uniformly distributed over [0, 10)
    X = np.random.uniform(0, 10, (n_samples, 2))
    y = np.zeros((n_samples, 1))

    x1 = X[:, 0]
    x2 = X[:, 1]

    # Quadrant 1 — lower-left
    m1 = (x1 <= 5.0) & (x2 <= 5.0)
    y[m1, 0] = 3.0 * x1[m1] + 4.0 * x2[m1] + 10.0

    # Quadrant 2 — lower-right
    m2 = (x1 > 5.0) & (x2 <= 5.0)
    y[m2, 0] = -2.0 * x1[m2] + 8.0 * x2[m2] + 35.0

    # Quadrant 3 — upper-left
    m3 = (x1 <= 5.0) & (x2 > 5.0)
    y[m3, 0] = 7.0 * x1[m3] - 3.0 * x2[m3] + 45.0

    # Quadrant 4 — upper-right
    m4 = (x1 > 5.0) & (x2 > 5.0)
    y[m4, 0] = -5.0 * x1[m4] - 6.0 * x2[m4] + 120.0

    # Inject Gaussian observation noise
    y += np.random.normal(0, noise_std, (n_samples, 1))

    return X, y


# ======================================================================
#  Train / Test Split
# ======================================================================

def train_test_split(X: np.ndarray,
                     y: np.ndarray,
                     test_ratio: float = 0.2,
                     seed: int = 123) -> tuple:
    """
    Randomly partition the dataset into training and test splits.

    A separate RandomState is used so that the split permutation is
    independent of the data-generation seed.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix.
    y : np.ndarray, shape (N, 1) or (N,)
        Target vector / column vector.
    test_ratio : float
        Fraction of samples reserved for testing  (0 < test_ratio < 1).
    seed : int
        Random seed for the permutation.

    Returns
    -------
    X_train : np.ndarray, shape (N_train, d)
    y_train : np.ndarray, shape (N_train, ...) — same trailing dims as y
    X_test  : np.ndarray, shape (N_test, d)
    y_test  : np.ndarray, shape (N_test,  ...) — same trailing dims as y
    """
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    idx = rng.permutation(N)
    split = int(N * (1 - test_ratio))
    return (X[idx[:split]], y[idx[:split]],
            X[idx[split:]], y[idx[split:]])


# ======================================================================
#  Persistence helpers
# ======================================================================

def save_dataset(X_train, y_train, X_test, y_test,
                 data_dir: str = "data/segment") -> None:
    """
    Persist train/test splits as CSV files with header ``x1,x2,y``.

    Parameters
    ----------
    X_train, X_test : np.ndarray, shape (N, 2)
    y_train, y_test : np.ndarray, shape (N, 1) or (N,)
    data_dir : str
        Directory in which to create ``train.csv`` and ``test.csv``.
    """
    os.makedirs(data_dir, exist_ok=True)
    header = "x1,x2,y"

    for name, Xm, ym in [("train.csv", X_train, y_train),
                          ("test.csv",  X_test,  y_test)]:
        data = np.column_stack([Xm, np.asarray(ym).ravel()])
        np.savetxt(os.path.join(data_dir, name), data,
                   delimiter=",", header=header, comments="")


def load_dataset(data_dir: str = "data/segment") -> tuple:
    """
    Load previously saved train/test CSV files.

    Parameters
    ----------
    data_dir : str
        Directory containing ``train.csv`` and ``test.csv``.

    Returns
    -------
    X_train : np.ndarray, shape (N_train, 2)
    y_train : np.ndarray, shape (N_train, 1)
    X_test  : np.ndarray, shape (N_test, 2)
    y_test  : np.ndarray, shape (N_test, 1)
    """
    train = np.loadtxt(os.path.join(data_dir, "train.csv"),
                       delimiter=",", skiprows=1)
    test  = np.loadtxt(os.path.join(data_dir, "test.csv"),
                       delimiter=",", skiprows=1)

    X_train, y_train = train[:, :2], train[:, 2:3]
    X_test,  y_test  = test[:, :2],  test[:, 2:3]

    return X_train, y_train, X_test, y_test


# ======================================================================
#  Main — generate, split, save
# ======================================================================

if __name__ == "__main__":
    X, y = generate_segment_dataset()
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    save_dataset(X_train, y_train, X_test, y_test)

    print(f"Feature matrix X shape strictly evaluated as: {X.shape}")
    print(f"Target vector y shape strictly evaluated as:  {y.shape}")
    print(f"Train split: X={X_train.shape}, y={y_train.shape}")
    print(f"Test split:  X={X_test.shape},  y={y_test.shape}")
    print("Dataset saved to data/segment/")
