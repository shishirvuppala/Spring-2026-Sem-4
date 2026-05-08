# =============================================================================
# autograder_task3.py  —  Task 3: Open-Ended Ensemble Fusion
# =============================================================================

import sys
import traceback
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator
import student as student

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
train = np.load("../train.npz")
X_train, y_train = train["X"], train["y"]

test = np.load("../test.npz")
X_test, y_test = test["X"], test["y"]

# ---------------------------------------------------------------------------
# Weak-learner constraint (same as Task 2)
# ---------------------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

ALLOWED_CLASSES = (
    DecisionTreeRegressor, Ridge, Lasso,
    LinearRegression, KNeighborsRegressor, SVR,
)
FORBIDDEN_NAMES = {
    "GradientBoostingRegressor", "HistGradientBoostingRegressor",
    "XGBRegressor", "LGBMRegressor", "CatBoostRegressor",
}
ALLOWED_SPECS = {
    DecisionTreeRegressor: {"max_depth": {1, 2, 3}},
    LinearRegression:      {},
    Ridge:                 {"alpha": (1e-4, 1e3)},
    Lasso:                 {"alpha": (1e-4, 1e1)},
    KNeighborsRegressor:   {"n_neighbors": {5, 10}},
    SVR:                   {"kernel": {"rbf"}, "C": (0.1, 10.0), "epsilon": (0.01, 1.0)},
}


def check_weak_learner(model, idx):
    name = type(model).__name__

    # Must be an allowed class
    if type(model) not in student.ALLOWED_MODEL_SPECS:
        raise RuntimeError(
            f"FUSION_LEARNERS[{idx}]: {name} is not an allowed model type."
        )

    spec = student.ALLOWED_MODEL_SPECS[type(model)]
    params = model.get_params()

    for param, constraint in spec.items():
        value = params.get(param)
        if isinstance(constraint, set):
            if value not in constraint:
                raise RuntimeError(
                    f"FUSION_LEARNERS[{idx}]: {name}.{param}={value!r} "
                    f"must be one of {constraint}."
                )
        elif isinstance(constraint, tuple):
            lo, hi = constraint
            if not (lo <= value <= hi):
                raise RuntimeError(
                    f"FUSION_LEARNERS[{idx}]: {name}.{param}={value} "
                    f"must be in [{lo}, {hi}]."
                )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SEP = "=" * 52

def main():
    print(SEP)
    print("  TASK 3 AUTOGRADER — Ensemble Fusion")
    print(SEP)

    errors = []

    # 1. FUSION_LEARNERS checks
    # print("\n[1] Checking FUSION_LEARNERS ...")
    learners = getattr(student, "FUSION_LEARNERS", None)
    if learners is None or not isinstance(learners, list):
        # print("  FAIL — FUSION_LEARNERS must be a list")
        sys.exit(1)

    if len(learners) < 5:
        errors.append(f"FUSION_LEARNERS has {len(learners)} entries; need ≥ 5")

    if len(learners) > 10:
        errors.append(f"FUSION_LEARNERS has {len(learners)} entries; must be ≤ 10")

    else:
        # print(f"  PASS — {len(learners)} learners found")
        pass

    for i, m in enumerate(learners):
        err = check_weak_learner(m, i)
        if err:
            errors.append(f"learner[{i}]: {err}")
        else:
            # print(f"  PASS — learner[{i}] {type(m).__name__}")
            pass

    # 2. task3_predict checks
    # print("\n[2] Checking task3_predict ...")
    if not callable(getattr(student, "task3_predict", None)):
        print("  FAIL — task3_predict not found or not callable")
        sys.exit(1)
    # print("  PASS — task3_predict is callable")

    # 3. Run and evaluate
    print("\n[3] Running task3_predict ...")
    try:
        y_pred = student.task3_predict(X_train, y_train, X_test)
    except NotImplementedError:
        print("  FAIL — task3_predict is not implemented yet")
        sys.exit(1)
    except Exception:
        print("  FAIL — task3_predict raised an exception:")
        traceback.print_exc()
        sys.exit(1)

    if not isinstance(y_pred, np.ndarray):
        errors.append(f"task3_predict must return np.ndarray, got {type(y_pred).__name__}")
    elif y_pred.shape != y_test.shape:
        errors.append(f"Shape mismatch: got {y_pred.shape}, expected {y_test.shape}")
    elif not np.all(np.isfinite(y_pred)):
        errors.append("Predictions contain NaN or Inf")
    else:
        # print("  PASS — output shape and type are correct")
        pass

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # 4. MAE
    mae = mean_absolute_error(y_test, y_pred)
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, np.mean(y_train)))

    print(f"\n{SEP}")
    print("  RESULTS")
    print(SEP)
    print(f"  Mean-prediction baseline MAE : {baseline_mae:.4f}")
    print(f"  Your fusion MAE              : {mae:.4f}")
    beat = mae < baseline_mae
    print(f"  Beats baseline               : {'YES' if beat else 'NO'}")
    print(SEP)


if __name__ == "__main__":
    main()
