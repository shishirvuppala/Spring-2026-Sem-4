# =========================
# autograder.py
# =========================

import numpy as np
import student
from sklearn.metrics import mean_absolute_error
from itertools import combinations


# ==================================================
# LOAD DATA (already split, hidden from students)
# ==================================================

train = np.load("../train.npz")
X_train = train["X"]
y_train = train["y"]

# -------------------------------
# LOAD TEST DATA
# (also used as DSEL)
# -------------------------------
test = np.load("../test.npz")
X_test = test["X"]
y_test = test["y"]

# DSEL is exactly the test set
X_dsel = X_test
y_dsel = y_test

# ==================================================
# MODEL DISTINCTNESS CHECK
# ==================================================

def prediction_diversity(preds):
    diffs = []
    ranks = []
    for a, b in combinations(preds, 2):
        diffs.append(np.mean((a - b) ** 2))

        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        ranks.append(np.mean(ra != rb))

    return min(diffs), min(ranks)


# ==================================================
# KNORA-U SANITY CHECK
# ==================================================

def knora_sanity(knora, X_probe):
    """
    Ensure KNORA-U selects different models for different samples.
    """
    selections = []

    for x in X_probe:
        selected = knora._select_models(x)
        selections.append(tuple(id(m) for m in selected))

    return len(set(selections)) > 1


# ==================================================
# MAIN GRADING LOGIC
# ==================================================

def main():

    # --------------------------------------------------
    # Check REGRESSION_MODELS
    # --------------------------------------------------
    models = student.REGRESSION_MODELS

    if not isinstance(models, list):
        raise RuntimeError("REGRESSION_MODELS must be a list")

    if len(models) < 5:
        raise RuntimeError("At least 5 models required")
    
    if len(models) > 10:
        raise RuntimeError("No more than 10 models allowed")

    for i, m in enumerate(models):
        if not hasattr(m, "predict"):
            raise RuntimeError(f"Model {i} is not a fitted estimator")

    # --------------------------------------------------
    # Diversity check (on DSEL)
    # --------------------------------------------------
    all_preds = np.array([m.predict(X_dsel) for m in models])

    diff, rank = prediction_diversity(all_preds)

    # if diff < 1e-3 and rank < 0.02:
    #     raise ValueError("Models are not sufficiently distinct")

    # --------------------------------------------------
    # Fit KNORA-U
    # --------------------------------------------------
    knora = student.KNORAURegressor()
    knora.fit(models, X_dsel, y_dsel)

    # --------------------------------------------------
    # KNORA-U sanity check
    # --------------------------------------------------
    probe_idx = np.random.default_rng(0).choice(len(X_test), size=100, replace=False)
    X_probe = X_test[probe_idx]

    if not knora_sanity(knora, X_probe):
        raise RuntimeError(
            "KNORA-U selection is degenerate — "
            "same model subset used for all samples."
        )

    # --------------------------------------------------
    # Final evaluation
    # --------------------------------------------------
    y_pred = knora.predict(X_test)

    if not isinstance(y_pred, np.ndarray):
        raise RuntimeError("predict() must return a numpy array")

    if y_pred.shape != y_test.shape:
        raise RuntimeError("Prediction shape mismatch")

    mae = mean_absolute_error(y_test, y_pred)

    print("=" * 50)
    print("KNORA-U AUTOGRADER RESULT")
    print("=" * 50)
    print(f"Model diversity score : {diff:.4f}")
    print(f"100 times Final MAE             : {100 * mae:.4f}")
    print("=" * 50)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()