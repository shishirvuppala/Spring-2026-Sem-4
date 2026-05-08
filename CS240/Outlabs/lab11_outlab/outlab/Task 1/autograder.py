# ===========================================================================
# AUTOGRADER — Gradient Boosting with Huber Loss (Regression)
# ===========================================================================
#
# Assumptions:
# - Student submission is in file: student.py
# - student.py defines:
#       * GradientBoostingEnsemble
#       * REGRESSION_LEARNERS
#       * ALLOWED_MODEL_SPECS
# - Students ONLY see train.npz
# - Autograder sees test.npz
#
# Output:
# - Prints MAE on hidden test data
# - Raises RuntimeError on rule violations
# ===========================================================================

import importlib.util
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator


# ---------------------------------------------------------------------------
# Load student module dynamically
# ---------------------------------------------------------------------------
def load_student_module(path="student.py"):
    spec = importlib.util.spec_from_file_location("student", path)
    student = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student)
    return student


# ---------------------------------------------------------------------------
# Estimator validation
# ---------------------------------------------------------------------------
def is_allowed_model(model: BaseEstimator, allowed_specs: dict) -> bool:
    cls = model.__class__
    if cls not in allowed_specs:
        return False

    params = model.get_params()
    constraints = allowed_specs[cls]

    for param, rule in constraints.items():
        if param not in params:
            return False

        value = params[param]

        # Discrete set constraint
        if isinstance(rule, set):
            if value not in rule:
                return False

        # Continuous range constraint
        elif isinstance(rule, tuple):
            lo, hi = rule
            if not (lo <= value <= hi):
                return False

    return True


# ---------------------------------------------------------------------------
# Hidden test data loader (AUTOGRADER ONLY)
# ---------------------------------------------------------------------------
def load_test_data(path="../test.npz"):
    data = np.load(path)
    X_test = data["X"]
    y_test = data["y"]
    return X_test, y_test


# ---------------------------------------------------------------------------
# Main grading routine
# ---------------------------------------------------------------------------
def main():
    # -----------------------------
    # Load student submission
    # -----------------------------
    student = load_student_module("student.py")

    # -----------------------------
    # Required symbols
    # -----------------------------
    required_symbols = [
        "GradientBoostingEnsemble",
        "REGRESSION_LEARNERS",
        "ALLOWED_MODEL_SPECS",
        "load_training_data",
    ]

    for name in required_symbols:
        if not hasattr(student, name):
            raise RuntimeError(f"student.py is missing required symbol: {name}")

    learners = student.REGRESSION_LEARNERS
    allowed_specs = student.ALLOWED_MODEL_SPECS

    # -----------------------------
    # Validate weak learners
    # -----------------------------
    if not isinstance(learners, list):
        raise RuntimeError("REGRESSION_LEARNERS must be a list")

    if len(learners) < 5:
        raise RuntimeError("At least 5 weak learners are required")
    
    if len(learners) > 10:
        raise RuntimeError("No more than 10 weak learners are allowed")

    for i, model in enumerate(learners):
        if not isinstance(model, BaseEstimator):
            raise RuntimeError(f"Learner at index {i} is not an sklearn estimator")

        if not is_allowed_model(model, allowed_specs):
            raise RuntimeError(
                f"Invalid weak learner at index {i}:\n"
                f"  {model.__class__.__name__}\n"
                f"  Params: {model.get_params()}"
            )

    # -----------------------------
    # Load training data (student API)
    # -----------------------------
    X_train, y_train = student.load_training_data("../train.npz")

    if X_train.shape[0] != y_train.shape[0]:
        raise RuntimeError("Mismatch between X_train and y_train sizes")

    # -----------------------------
    # Load hidden test data
    # -----------------------------
    X_test, y_test = load_test_data("../test.npz")

    # -----------------------------
    # Train ensemble
    # -----------------------------
    model = student.GradientBoostingEnsemble()
    model.fit(learners, X_train, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = model.predict(X_test)

    if not isinstance(y_pred, np.ndarray):
        raise RuntimeError("predict() must return a numpy array")

    if y_pred.shape != y_test.shape:
        raise RuntimeError(
            f"Prediction shape mismatch: expected {y_test.shape}, got {y_pred.shape}"
        )

    # -----------------------------
    # Evaluate
    # -----------------------------
    mae = mean_absolute_error(y_test, y_pred)
    print(f"100 times the MAE on hidden test set: {mae * 100:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()