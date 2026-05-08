# do not modify the imports below
import json
import numpy as np

# ============================================================
# OLS implementations
# ============================================================

def ols_with_intercept(X, y):
    """
    OLS with intercept
    X : (N, d)
    y : (N,)
    Returns:
        w  : (d,) slope vector
        w0 : scalar intercept
    """

    N = X.shape[0]
    X_aug = np.hstack([np.ones((N, 1)), X])  # [1, X]

    theta = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
    w0 = theta[0]
    w = theta[1:]

    return w, w0


def ols_no_intercept(X, y):
    """
    OLS without intercept
    X : (N, d)
    y : (N,)
    Returns:
        w : (d,) slope vector
    """

    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    return w


# ============================================================
# Metrics
# ============================================================

def predict_with_intercept(X, w, w0):
    return X @ w + w0


def predict_no_intercept(X, w):
    return X @ w

def compute_metrics(y, y_hat):

    # TODO: Compute mean squared error
    mse = np.mean((y - y_hat) ** 2)

    # TODO: Compute correlation between y and y_hat
    corr = np.corrcoef(y, y_hat)[0, 1]

    # TODO: Compute squared correlation
    corr_sq = corr ** 2

    # TODO: Compute R^2
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "mse": float(mse),
        "corr": float(corr),
        "corr_sq": float(corr_sq),
        "r2": float(r2),
    }


# ============================================================
# Data loading
# ============================================================

def load_data():
    """
    Load all datasets required for the lab.

    Returns
    -------
    X_train, y_train (q1_train.csv)
    X_test, y_test (q1_test.csv)
    X_train_outlier, y_train_outlier (q1_train.csv appeanded by q1_outliers.csv)
    """
    # TODO: Load data from CSV files and return numpy arrays

    train = np.loadtxt("q1_train.csv", delimiter=",", skiprows=1)
    test = np.loadtxt("q1_test.csv", delimiter=",", skiprows=1)
    outliers = np.loadtxt("q1_outliers.csv", delimiter=",", skiprows=1)

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    X_train_outlier = np.vstack([X_train, outliers[:, :-1]])
    y_train_outlier = np.concatenate([y_train, outliers[:, -1]])

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_outlier,
        y_train_outlier,
    )

# ============================================================
# Main experiment (DO NOT MODIFY, AUTOGRADER TESTS WILL RUN THIS)
# ============================================================

if __name__ == "__main__": 
    X_train, y_train, X_test, y_test, X_train_outlier, y_train_outlier = load_data()

    # ---------- Standard OLS ----------
    w, w0 = ols_with_intercept(X_train, y_train)

    yhat_train = predict_with_intercept(X_train, w, w0)
    yhat_test = predict_with_intercept(X_test, w, w0)

    standard_train_metrics = compute_metrics(y_train, yhat_train)
    standard_test_metrics = compute_metrics(y_test, yhat_test)

    # ---------- OLS with outliers ----------
    w_o, w0_o = ols_with_intercept(X_train_outlier, y_train_outlier)

    yhat_train_outlier = predict_with_intercept(X_train_outlier, w_o, w0_o)
    yhat_test_outlier = predict_with_intercept(X_test, w_o, w0_o)

    outlier_train_metrics = compute_metrics(y_train_outlier, yhat_train_outlier)
    outlier_test_metrics = compute_metrics(y_test, yhat_test_outlier)

    # ---------- No-intercept OLS ----------
    w_no = ols_no_intercept(X_train, y_train)

    yhat_train_no = predict_no_intercept(X_train, w_no)
    yhat_test_no = predict_no_intercept(X_test, w_no)

    no_intercept_train_metrics = compute_metrics(y_train, yhat_train_no)
    no_intercept_test_metrics = compute_metrics(y_test, yhat_test_no)

    # ------------- Dump metrics -------------

    metrics = {
        "standard_ols": {
            "train": standard_train_metrics,
            "test": standard_test_metrics,
        },
        "outlier_ols": {
            "train": outlier_train_metrics,
            "test": outlier_test_metrics,
        },
        "no_intercept_ols": {
            "train": no_intercept_train_metrics,
            "test": no_intercept_test_metrics,
        },
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)