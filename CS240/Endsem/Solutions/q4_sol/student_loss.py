"""SOLUTION FILE — not distributed to students."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(p="dataset.npz"):
    d = np.load(p)
    return {k: d[k] for k in d.files}


def estimate_y_true(x_cal, y_pred_cal, loss_cal, x_eval):
    """
    Key insight: the hidden loss is minimised when y_pred = y_true.
    So within each local x-region, the calibration point with the
    LOWEST loss has a prediction closest to y_true.

    Strategy:
      1. Partition the x-axis into n_bins equal-width bins.
      2. In each bin, pick the prediction that achieved the lowest loss
         — that prediction is our best estimate of y_true in that region.
      3. Linearly interpolate between bin estimates to get values at x_eval.
    """
    n_bins   = 60
    bin_edges = np.linspace(x_cal.min(), x_cal.max(), n_bins + 1)

    bin_centers  = []
    best_preds   = []

    for i in range(n_bins):
        lo, hi  = bin_edges[i], bin_edges[i + 1]
        # Include the right edge in the last bin so no points are missed
        if i < n_bins - 1:
            in_bin = (x_cal >= lo) & (x_cal < hi)
        else:
            in_bin = (x_cal >= lo) & (x_cal <= hi)

        if in_bin.sum() == 0:
            continue  # empty bin — skip

        # Prediction with smallest loss is closest to y_true
        best_idx = np.argmin(loss_cal[in_bin])
        bin_centers.append((lo + hi) / 2.0)
        best_preds.append(y_pred_cal[in_bin][best_idx])

    # Linear interpolation between bin estimates
    return np.interp(x_eval, bin_centers, best_preds)


def rbf_kernel(X1, X2, gamma):
    """RBF kernel: K[i,j] = exp(-gamma * ||X1[i] - X2[j]||^2)"""
    # Squared distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq_dists = (np.sum(X1 ** 2, axis=1, keepdims=True)
                - 2 * X1 @ X2.T
                + np.sum(X2 ** 2, axis=1))
    return np.exp(-gamma * np.maximum(sq_dists, 0))


def nw_predict(Z_new, Z_train, y_train, gamma):
    """Nadaraya-Watson: weighted average of y_train, weighted by RBF kernel."""
    K = rbf_kernel(Z_new, Z_train, gamma)
    # Clamp denominator away from zero to avoid division by zero
    return (K @ y_train) / np.maximum(K.sum(axis=1), 1e-12)


def model_loss(r_cal, x_cal, loss_cal, gamma=1.0):
    """
    Build a Nadaraya-Watson model for the hidden loss as a function of
    (residual, x). Features are scaled before building the model so that
    r and x contribute equally regardless of their raw magnitudes.
    """
    # Stack (r, x) into a 2-D feature matrix and standardise
    X_raw = np.column_stack([r_cal, x_cal])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return {
        "Z_train": X_scaled,
        "losses":  loss_cal.copy(),
        "gamma":   gamma,
        "scaler":  scaler,
    }


def predict_loss(model_dict, r, x):
    """Query the NW loss model at new (r, x) pairs."""
    r = np.atleast_1d(r)
    x = np.atleast_1d(x)
    Z_new = model_dict["scaler"].transform(np.column_stack([r, x]))
    return nw_predict(Z_new, model_dict["Z_train"],
                      model_dict["losses"], model_dict["gamma"])


def fit_linear_predictor(x_train, y_train, model_dict):
    """
    Find y_hat = a*x + b that minimises the mean learned loss on training data.
    Uses gradient descent with numerical gradients.
    Initialised from the OLS solution (good starting point).
    """
    def mean_loss(a, b):
        residuals = np.abs(y_train - (a * x_train + b))
        return float(np.mean(predict_loss(model_dict, residuals, x_train)))

    # Initialise from OLS
    coeffs = np.polyfit(x_train, y_train, 1)
    a, b   = float(coeffs[0]), float(coeffs[1])

    lr  = 0.05
    eps = 1e-5
    for _ in range(200):
        da = (mean_loss(a + eps, b) - mean_loss(a - eps, b)) / (2 * eps)
        db = (mean_loss(a, b + eps) - mean_loss(a, b - eps)) / (2 * eps)
        a -= lr * da
        b -= lr * db

    return {"a": a, "b": b}