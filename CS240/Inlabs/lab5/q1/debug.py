"""
debug.py
========
Lab 5, Q1: Debugging / sanity-check script for students.

Run this file to verify that your implementations in classifiers.py and
utils.py behave correctly on small, hand-crafted inputs.

Usage:
    python debug.py
"""

import numpy as np

# ------------------------------------------------------------------
# Import student implementations
# ------------------------------------------------------------------
from classifiers import LDAClassifier, LogisticRegression
from utils import (
    load_data,
    get_true_positives,
    get_false_positives,
    get_false_negatives,
    get_true_negatives,
    get_precision,
    get_recall,
    get_f1_score,
    get_accuracy,
)

SEP = "-" * 55


# ==================================================================
# 1. Metric helpers
# ==================================================================
def test_metrics() -> None:
    print(SEP)
    print("Testing evaluation metrics (utils.py)")
    print(SEP)

    y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    # TP=3  FP=1  FN=1  TN=3

    tp = get_true_positives(y_true, y_pred)
    fp = get_false_positives(y_true, y_pred)
    fn = get_false_negatives(y_true, y_pred)
    tn = get_true_negatives(y_true, y_pred)

    print(f"  TP = {tp}  (expected 3)")
    print(f"  FP = {fp}  (expected 1)")
    print(f"  FN = {fn}  (expected 1)")
    print(f"  TN = {tn}  (expected 3)")
    assert tp == 3, f"TP mismatch: got {tp}"
    assert fp == 1, f"FP mismatch: got {fp}"
    assert fn == 1, f"FN mismatch: got {fn}"
    assert tn == 3, f"TN mismatch: got {tn}"

    prec = get_precision(y_true, y_pred)
    rec = get_recall(y_true, y_pred)
    f1 = get_f1_score(y_true, y_pred)
    acc = get_accuracy(y_true, y_pred)

    print(f"  Precision = {prec:.4f}  (expected 0.7500)")
    print(f"  Recall    = {rec:.4f}  (expected 0.7500)")
    print(f"  F1        = {f1:.4f}  (expected 0.7500)")
    print(f"  Accuracy  = {acc:.4f}  (expected 0.7500)")
    assert abs(prec - 0.75) < 1e-6
    assert abs(rec - 0.75) < 1e-6
    assert abs(f1 - 0.75) < 1e-6
    assert abs(acc - 0.75) < 1e-6

    # Edge case: all zeros predicted
    y_all_neg = np.zeros_like(y_pred)
    prec_zero = get_precision(y_true, y_all_neg)
    rec_zero = get_recall(y_true, y_all_neg)
    f1_zero = get_f1_score(y_true, y_all_neg)
    print(f"\n  [Edge] All-zero preds → Prec={prec_zero:.4f}, Rec={rec_zero:.4f}, F1={f1_zero:.4f}")
    assert prec_zero == 0.0
    assert rec_zero == 0.0
    assert f1_zero == 0.0

    print("  ✓ All metric tests passed.\n")


# ==================================================================
# 2. Sigmoid numerical stability
# ==================================================================
def test_sigmoid() -> None:
    print(SEP)
    print("Testing LogisticRegression.sigmoid (numerical stability)")
    print(SEP)

    z = np.array([-1.0, 0.0, 1.0, 1e9, -1e9, 5e8])
    out = LogisticRegression.sigmoid(z)
    print(f"  Input:  {z}")
    print(f"  Output: {out}")
    # Expected: ~[0.2689, 0.5, 0.7311, 1.0, 0.0, 1.0]
    assert out.shape == z.shape, f"Shape mismatch: {out.shape} vs {z.shape}"
    assert not np.any(np.isnan(out)), "Sigmoid produced NaN — check clipping!"
    assert not np.any(np.isinf(out)), "Sigmoid produced Inf — check clipping!"
    assert np.isclose(out[1], 0.5, atol=1e-6), "sigmoid(0) should be 0.5"
    assert np.isclose(out[3], 1.0, atol=1e-6), "sigmoid(1e9) should be ~1.0"
    assert np.isclose(out[4], 0.0, atol=1e-6), "sigmoid(-1e9) should be ~0.0"
    print("  ✓ Sigmoid is numerically stable.\n")


# ==================================================================
# 3. LDA quick check
# ==================================================================
def test_lda() -> None:
    print(SEP)
    print("Testing LDAClassifier on toy data")
    print(SEP)

    np.random.seed(0)
    # 2D toy problem, well-separated Gaussians
    X0 = np.random.randn(50, 2) + np.array([-2, -2])
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50, dtype=float)

    lda = LDAClassifier()
    lda.fit(X, y)
    w, b = lda.get_params()
    print(f"  w = {w}")
    print(f"  b = {b:.4f}")
    assert w.shape == (2,), f"w shape mismatch: {w.shape}"

    preds = lda.predict(X)
    acc = get_accuracy(y, preds)
    print(f"  Train accuracy = {acc:.4f}  (should be close to 1.0)")
    assert acc > 0.9, f"LDA accuracy too low: {acc}"
    print("  ✓ LDA basic test passed.\n")


# ==================================================================
# 4. Logistic Regression gradient check (small)
# ==================================================================
def test_logreg_loss_and_gradient() -> None:
    print(SEP)
    print("Testing LogisticRegression loss & gradient")
    print(SEP)

    model = LogisticRegression()

    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    loss = model.compute_loss(y_true, y_pred)
    print(f"  Loss = {loss:.6f}  (expected ~0.1642)")
    # Manual: -(1/4)[ln(0.9)+ln(0.9)+ln(0.8)+ln(0.8)] ≈ 0.1642
    assert abs(loss - 0.164252) < 0.01, f"Loss mismatch: {loss}"

    # Gradient shape check
    X = np.random.randn(4, 3)
    grad_w, grad_b = model.compute_gradient(X, y_true, y_pred)
    print(f"  grad_w shape = {grad_w.shape}  (expected (3,))")
    print(f"  grad_b = {grad_b:.6f}")
    assert grad_w.shape == (3,)
    print("  ✓ Loss & gradient tests passed.\n")


# ==================================================================
# 5. Full pipeline on real data (quick smoke test)
# ==================================================================
def test_full_pipeline() -> None:
    print(SEP)
    print("Full pipeline smoke test on ALL datasets")
    print(SEP)

    datasets = [
        ("A — Iris",          "data/iris"),
        ("B — Breast Cancer", "data/cancer"),
        ("C — Sonar",         "data/sonar"),
    ]

    for name, data_dir in datasets:
        print(f"\n  >> Dataset {name}")
        try:
            X_train, y_train = load_data(f"{data_dir}/train.csv")
            X_test,  y_test  = load_data(f"{data_dir}/test.csv")
        except FileNotFoundError:
            print(f"    !! {data_dir}/ not found — skipping.")
            continue

        print(f"    Train: {X_train.shape}  Test: {X_test.shape}")

        # LDA
        lda = LDAClassifier()
        lda.fit(X_train, y_train)
        w_gen, b_gen = lda.get_params()
        preds_lda = lda.predict(X_test)
        acc_lda = get_accuracy(y_test, preds_lda)
        print(f"    LDA  test accuracy = {acc_lda:.4f}")

        # Logistic Regression (warm start, 200 epochs)
        lr = LogisticRegression(lr=0.01, max_iter=200,
                                batch_size=X_train.shape[0])
        lr.fit(X_train, y_train, init_params=(w_gen, b_gen))
        preds_lr = lr.predict(X_test)
        acc_lr = get_accuracy(y_test, preds_lr)
        print(f"    LR (warm, 200 ep) test accuracy = {acc_lr:.4f}")
        print(f"    Loss history length = {len(lr.loss_history)} (expected 200)")
        assert len(lr.loss_history) == 200
        print(f"    ✓ {name} passed.")

    print("\n  ✓ Pipeline smoke tests passed.\n")


# ==================================================================
# Run all tests
# ==================================================================
if __name__ == "__main__":
    test_metrics()
    test_sigmoid()
    test_lda()
    test_logreg_loss_and_gradient()
    test_full_pipeline()
    print("=" * 55)
    print("  All debug checks passed!")
    print("=" * 55)
