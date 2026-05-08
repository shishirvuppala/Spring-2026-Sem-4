import numpy as np
import glob
import importlib

student = importlib.import_module("student_solution")

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

dataset_files = sorted(glob.glob("datasets/dataset_*.npz"))
total_pass = 0

for idx, path in enumerate(dataset_files, 1):
    data = np.load(path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val     = data["X_val"], data["y_val"]
    X_test, y_test   = data["X_test"], data["y_test"]
    lam, gamma, threshold = float(data["lam"]), float(data["gamma"]), float(data["r2_threshold"])

    print(f"\n=== Dataset {idx} ===")

    # Train (Student code should handle its own internal normalization)
    w = student.train_weighted_krr(X_train, y_train, X_val, y_val, lam=lam, gamma=gamma)
    print("Learned w:", w)

    # Evaluate: MUST normalize using training stats for consistency
    X_tr_norm, _, X_te_norm = student.normalize_dataset(X_train, X_val, X_test)

    K_train = student.weighted_rbf_kernel(X_tr_norm, X_tr_norm, w)
    alpha = student.solve_alpha(K_train, y_train, lam)

    K_test = student.weighted_rbf_kernel(X_te_norm, X_tr_norm, w)
    y_pred = K_test @ alpha

    r2 = r2_score(y_test, y_pred)
    print(f"Test R2 : {r2:.4f} | Threshold: {threshold:.4f}")

    if r2 >= threshold:
        print("Result: PASS")
        total_pass += 1
    else:
        print("Result: FAIL")

print("\n=====================")
print(f"Passed {total_pass} / {len(dataset_files)} datasets")
print("=====================")