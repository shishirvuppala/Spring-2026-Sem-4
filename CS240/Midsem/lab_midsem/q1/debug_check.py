import numpy as np
import os
import json

from model import LogisticModel, train, accuracy, give_optimal_hyperparameter

TEST_DIR = "tests"


# =========================================================
# Metrics
# =========================================================

def mean_abs_diff(a, b):
    return float(np.mean(np.abs(a - b)))


# =========================================================
# Run single test case
# =========================================================

def run_case(case_path):
    # ---------- load data ----------
    Xtr = np.load(os.path.join(case_path, "X_train.npy"))
    ytr = np.load(os.path.join(case_path, "y_train.npy"))

    Xval = np.load(os.path.join(case_path, "X_val.npy"))
    yval = np.load(os.path.join(case_path, "y_val.npy"))

    Xte = np.load(os.path.join(case_path, "X_test.npy"))
    yte = np.load(os.path.join(case_path, "y_test.npy"))

    y_ref = np.load(os.path.join(case_path, "y_pred_ref.npy"))

    with open(os.path.join(case_path, "meta.json")) as f:
        meta = json.load(f)

    # ---------- student pipeline ----------
    np.random.seed(0)
    lam = give_optimal_hyperparameter(Xtr, ytr, Xval, yval)

    model = LogisticModel(Xtr.shape[1], lam=lam)
    train(model, Xtr, ytr)

    y_pred = model.predict(Xte)

    # ---------- metrics ----------
    acc = accuracy(yte, y_pred)
    mad = mean_abs_diff(y_pred, y_ref)

    acc_ok = acc >= meta["accuracy_threshold"]
    mad_ok = mad <= meta["mad_threshold"]

    passed = acc_ok and mad_ok

    return {
        "passed": passed,
        "accuracy": acc,
        "accuracy_ref": meta["accuracy_ref"],
        "mad": mad,
        "lambda": lam,
        "lambda_ref": meta["lambda_ref"],
        "dataset": meta["dataset_name"],
    }


# =========================================================
# Run all tests
# =========================================================

def main():
    if not os.path.exists(TEST_DIR):
        print("tests/ directory not found.")
        return

    all_pass = True

    test_cases = sorted(d for d in os.listdir(TEST_DIR)
                        if d.startswith("test"))

    print("\nRunning debug checks\n")
    

    for case in test_cases:
        case_path = os.path.join(TEST_DIR, case)
        result = run_case(case_path)

        status = "PASS" if result["passed"] else "FAIL"

        print(f"{case} ({result['dataset']})")
        print(f"  accuracy: {result['accuracy']:.3f} "
              f"(ref {result['accuracy_ref']:.3f})")
        print(f"  MAD:      {result['mad']:.4f}")
        print(f"  λ found:  {result['lambda']:.2f} "
              f"(ref {result['lambda_ref']:.2f})")
        print(f"  → {status}\n")

        all_pass &= result["passed"]

    print("="*40)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("="*40)


if __name__ == "__main__":
    main()