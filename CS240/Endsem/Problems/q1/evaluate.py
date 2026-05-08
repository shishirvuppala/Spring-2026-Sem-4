"""
Public evaluator for Q2: Efficient SVM Classification.

Run:
    python3 evaluate.py
"""
import time
import numpy as np
from student_svm import load_data, train_svm, evaluate_model, find_best_config

# ---- thresholds ----
TIMEOUT = 60.0                 # seconds
ACCURACY_THRESHOLD = 0.95      # minimum test accuracy
SV_EXCELLENT = 75              # n_sv <= this  => full marks on Part C
SV_GOOD = 100                  # n_sv <= this  => good marks on Part C
SV_PARTIAL = 150               # n_sv <= this  => partial marks on Part C


def sep(title):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def check_part_a():
    sep("Part A: load_data")
    data = load_data("dataset.npz")
    required = {"X_train", "y_train", "X_test", "y_test", "scaler"}
    missing = required - set(data.keys())
    if missing:
        print(f"  FAIL — missing keys: {missing}")
        return None

    print(f"  X_train shape : {data['X_train'].shape}")
    print(f"  X_test  shape : {data['X_test'].shape}")

    mu = np.abs(data["X_train"].mean(axis=0))
    if np.all(mu < 0.1):
        print("  Scaling check : PASS (train mean ~ 0)")
    else:
        print(f"  Scaling check : WARN (train mean = {mu})")

    print("  Part A: OK")
    return data


def check_part_b(data):
    sep("Part B: train_svm + evaluate_model")
    model = train_svm(data["X_train"], data["y_train"],
                      kernel="rbf", C=1.0, gamma="scale")
    metrics = evaluate_model(model, data["X_test"], data["y_test"])

    required = {"accuracy", "n_support_vectors", "n_support_per_class"}
    missing = required - set(metrics.keys())
    if missing:
        print(f"  FAIL — missing keys in evaluate_model output: {missing}")
        return

    print(f"  Baseline RBF (C=1, gamma=scale):")
    print(f"    accuracy          = {metrics['accuracy']:.4f}")
    print(f"    n_support_vectors = {metrics['n_support_vectors']}")
    print(f"    n_support/class   = {metrics['n_support_per_class']}")
    print("  Part B: OK")


def check_part_c(data):
    sep("Part C: find_best_config")
    start = time.time()
    result = find_best_config(
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        accuracy_threshold=ACCURACY_THRESHOLD,
    )
    elapsed = time.time() - start

    if result is None:
        print("  FAIL — find_best_config returned None")
        return

    required = {"kernel", "C", "gamma", "accuracy",
                "n_support_vectors", "model"}
    missing = required - set(result.keys())
    if missing:
        print(f"  FAIL — missing keys: {missing}")
        return

    acc = result["accuracy"]
    nsv = result["n_support_vectors"]

    print(f"  kernel           = {result['kernel']}")
    print(f"  C                = {result['C']}")
    print(f"  gamma            = {result['gamma']}")
    print(f"  degree           = {result.get('degree', 'N/A')}")
    print(f"  accuracy         = {acc:.4f}")
    print(f"  n_support_vectors= {nsv}")
    print(f"  search time      = {elapsed:.2f} sec")

    if elapsed > TIMEOUT:
        print("  TIMEOUT — search exceeded time limit!")
        return

    if acc < ACCURACY_THRESHOLD:
        print(f"  FAIL — accuracy {acc:.4f} < {ACCURACY_THRESHOLD}")
    elif nsv <= SV_EXCELLENT:
        print(f"  EXCELLENT — accuracy OK and n_sv ({nsv}) <= {SV_EXCELLENT}")
    elif nsv <= SV_GOOD:
        print(f"  GOOD — accuracy OK and n_sv ({nsv}) <= {SV_GOOD}")
    elif nsv <= SV_PARTIAL:
        print(f"  PARTIAL — accuracy OK and n_sv ({nsv}) <= {SV_PARTIAL}")
    else:
        print(f"  WEAK — accuracy OK but n_sv ({nsv}) > {SV_PARTIAL}")


def main():
    data = check_part_a()
    if data is None:
        return
    check_part_b(data)
    check_part_c(data)


if __name__ == "__main__":
    main()
