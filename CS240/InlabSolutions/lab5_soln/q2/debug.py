"""
debug.py
========
Lab 5, Q2: Debugging / sanity-check script for students.

Run this file to verify that your implementations in cart.py and
utils.py behave correctly on small, hand-crafted inputs.

Usage:
    python debug.py
"""

import numpy as np

# ------------------------------------------------------------------
# Import student implementations
# ------------------------------------------------------------------
from cart import (
    gini_impurity,
    find_best_split,
    DecisionTree,
    LeafNode,
    DecisionNode,
)
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
# 2. Gini Impurity
# ==================================================================
def test_gini_impurity() -> None:
    print(SEP)
    print("Testing gini_impurity (cart.py)")
    print(SEP)

    # Pure node — all class 0
    y_pure = np.array([0, 0, 0, 0])
    g_pure = gini_impurity(y_pure)
    print(f"  Pure node [0,0,0,0]    → Gini = {g_pure:.4f}  (expected 0.0000)")
    assert abs(g_pure - 0.0) < 1e-6, f"Gini mismatch: got {g_pure}"

    # Perfectly balanced binary
    y_half = np.array([0, 0, 1, 1])
    g_half = gini_impurity(y_half)
    print(f"  Balanced   [0,0,1,1]   → Gini = {g_half:.4f}  (expected 0.5000)")
    assert abs(g_half - 0.5) < 1e-6, f"Gini mismatch: got {g_half}"

    # 3-class balanced
    y_three = np.array([0, 1, 2, 0, 1, 2])
    g_three = gini_impurity(y_three)
    expected = 1.0 - 3 * (1.0 / 3.0) ** 2  # = 2/3
    print(f"  3-class    [0,1,2,…]   → Gini = {g_three:.4f}  (expected {expected:.4f})")
    assert abs(g_three - expected) < 1e-6, f"Gini mismatch: got {g_three}"

    # Empty array
    y_empty = np.array([])
    g_empty = gini_impurity(y_empty)
    print(f"  Empty      []          → Gini = {g_empty:.4f}  (expected 0.0000)")
    assert abs(g_empty - 0.0) < 1e-6, f"Gini mismatch: got {g_empty}"

    # Imbalanced binary: 1 out of 4 is class 1
    y_imb = np.array([0, 0, 0, 1])
    g_imb = gini_impurity(y_imb)
    expected_imb = 1.0 - (0.75 ** 2 + 0.25 ** 2)  # = 0.375
    print(f"  Imbalanced [0,0,0,1]   → Gini = {g_imb:.4f}  (expected {expected_imb:.4f})")
    assert abs(g_imb - expected_imb) < 1e-6, f"Gini mismatch: got {g_imb}"

    print("  ✓ Gini impurity tests passed.\n")


# ==================================================================
# 3. Find Best Split
# ==================================================================
def test_find_best_split() -> None:
    print(SEP)
    print("Testing find_best_split (cart.py)")
    print(SEP)

    # Simple 1D dataset: [1, 2, 3, 4] with labels [0, 0, 1, 1]
    # Best split should be at feature 0, threshold = 2 (cost = 0.0)
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([0, 0, 1, 1])

    result = find_best_split(X, y)
    assert result is not None, "find_best_split returned None for a splittable dataset"
    print(f"  1D separable → feature={result['feature_index']}, "
          f"threshold={result['threshold']:.2f}, cost={result['cost']:.4f}")
    assert result["feature_index"] == 0, f"Expected feature 0, got {result['feature_index']}"
    assert abs(result["cost"] - 0.0) < 1e-6, f"Expected cost 0.0, got {result['cost']}"

    # 2D dataset
    X2 = np.array([
        [1.0, 5.0],
        [2.0, 4.0],
        [3.0, 3.0],
        [4.0, 2.0],
    ])
    y2 = np.array([0, 0, 1, 1])
    result2 = find_best_split(X2, y2)
    assert result2 is not None, "find_best_split returned None for a 2D dataset"
    print(f"  2D separable → feature={result2['feature_index']}, "
          f"threshold={result2['threshold']:.2f}, cost={result2['cost']:.4f}")
    assert abs(result2["cost"] - 0.0) < 1e-6, f"Expected cost 0.0, got {result2['cost']}"

    # Unsplittable: single unique value per feature
    X_single = np.array([[1], [1], [1]], dtype=float)
    y_single = np.array([0, 1, 0])
    result_single = find_best_split(X_single, y_single)
    # There is only one unique value → all samples go left, right is empty.
    # Depending on implementation, this should return None.
    print(f"  Unsplittable (single value) → result = {result_single}")

    print("  ✓ find_best_split tests passed.\n")


# ==================================================================
# 4. Decision Tree on toy data
# ==================================================================
def test_decision_tree_toy() -> None:
    print(SEP)
    print("Testing DecisionTree on toy data")
    print(SEP)

    # XOR-like 2D problem
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.3, 0.3],
    ])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

    tree = DecisionTree(max_depth=5, min_samples_split=2)
    tree.fit(X, y)

    depth = tree.get_depth()
    n_leaves = tree.get_n_leaves()
    print(f"  Tree depth  = {depth}")
    print(f"  Num leaves  = {n_leaves}")
    assert depth >= 1, "Tree should have depth >= 1"
    assert n_leaves >= 2, "Tree should have at least 2 leaves"

    preds = tree.predict(X)
    acc = get_accuracy(y, preds)
    print(f"  Train accuracy = {acc:.4f}")
    assert preds.shape == y.shape, f"Shape mismatch: {preds.shape} vs {y.shape}"

    # Test with max_depth=1 (stump)
    stump = DecisionTree(max_depth=1, min_samples_split=2)
    stump.fit(X, y)
    stump_depth = stump.get_depth()
    stump_leaves = stump.get_n_leaves()
    print(f"\n  Stump (max_depth=1): depth={stump_depth}, leaves={stump_leaves}")
    assert stump_depth <= 1, f"Stump depth should be <= 1, got {stump_depth}"
    assert stump_leaves <= 2, f"Stump should have <= 2 leaves, got {stump_leaves}"

    preds_stump = stump.predict(X)
    assert preds_stump.shape == y.shape, f"Shape mismatch: {preds_stump.shape}"

    print("  ✓ Decision tree toy tests passed.\n")


# ==================================================================
# 5. Linearly separable quick check
# ==================================================================
def test_linearly_separable() -> None:
    print(SEP)
    print("Testing DecisionTree on linearly separable data")
    print(SEP)

    np.random.seed(0)
    # Two well-separated Gaussian clusters
    X0 = np.random.randn(50, 2) + np.array([-3, -3])
    X1 = np.random.randn(50, 2) + np.array([3, 3])
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    tree = DecisionTree(max_depth=5, min_samples_split=2)
    tree.fit(X, y)
    preds = tree.predict(X)
    acc = get_accuracy(y, preds)
    print(f"  Train accuracy = {acc:.4f}  (should be close to 1.0)")
    assert acc > 0.95, f"Accuracy too low: {acc}"

    print(f"  Tree depth  = {tree.get_depth()}")
    print(f"  Num leaves  = {tree.get_n_leaves()}")
    print("  ✓ Linearly separable test passed.\n")


# ==================================================================
# 6. Full pipeline smoke test on real datasets
# ==================================================================
def test_full_pipeline() -> None:
    print(SEP)
    print("Full pipeline smoke test on ALL datasets")
    print(SEP)

    datasets = [
        ("A — Moons",    "data/moons"),
        ("B — Iris",     "data/iris"),
        ("C — Circles",  "data/circles"),
    ]

    for name, data_dir in datasets:
        print(f"\n  >> Dataset {name}")
        try:
            X_train, y_train = load_data(f"{data_dir}/train.csv")
            X_test,  y_test  = load_data(f"{data_dir}/test.csv")
        except FileNotFoundError:
            print(f"    !! {data_dir}/ not found — run setup_data.py first. Skipping.")
            continue

        print(f"    Train: {X_train.shape}  Test: {X_test.shape}")

        # Fit decision tree
        tree = DecisionTree(max_depth=5, min_samples_split=2)
        tree.fit(X_train, y_train.astype(int))
        preds = tree.predict(X_test)
        acc = get_accuracy(y_test, preds)
        print(f"    Test accuracy (depth=5) = {acc:.4f}")
        print(f"    Tree depth  = {tree.get_depth()}")
        print(f"    Num leaves  = {tree.get_n_leaves()}")
        assert preds.shape == y_test.shape, "Prediction shape mismatch"
        print(f"    ✓ {name} passed.")

    print("\n  ✓ Pipeline smoke tests passed.\n")


# ==================================================================
# Run all tests
# ==================================================================
if __name__ == "__main__":
    test_metrics()
    test_gini_impurity()
    test_find_best_split()
    test_decision_tree_toy()
    test_linearly_separable()
    test_full_pipeline()
    print("=" * 55)
    print("  All debug tests passed!")
    print("=" * 55)
