"""
main.py
=======
Lab 5, Q2: Main experiment script (provided to students — DO NOT MODIFY).

This script runs the Decision Tree experiment on THREE datasets:
  A) Two Moons     — Non-linear boundary (2D, binary)
  B) Iris           — Multi-class classification (4D, 3 classes)
  C) Circles        — Concentric circles (2D, binary)

For each dataset it:
  1. Trains decision trees at multiple depths (Task 3).
  2. Evaluates each tree on the test set with accuracy, precision, recall, F1.
  3. For 2D datasets (Moons, Circles): plots decision boundaries at
     max_depth=3 and max_depth=10 to visualise the "staircase" effect.
  4. Reports tree statistics (depth, number of leaves).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cart import DecisionTree
from utils import (
    load_data,
    get_accuracy,
    get_precision,
    get_recall,
    get_f1_score,
)


# ------------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------------
# (label, data_dir, description, is_2d, is_binary)
DATASETS = [
    ("A", "data/moons",   "Two Moons (Non-Linear 2D)",   True,  True),
    ("B", "data/iris",    "Iris (Multi-Class 4D)",        False, False),
    ("C", "data/circles", "Circles (Concentric 2D)",      True,  True),
]

# Depths to evaluate
EVAL_DEPTHS = [1, 3, 5, 10, None]

# Depths at which to plot decision boundaries (2D datasets only)
PLOT_DEPTHS = [3, 10]


# ------------------------------------------------------------------
# Helpers (not student-implemented)
# ------------------------------------------------------------------

def print_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    """Pretty-print a per-class evaluation table plus overall accuracy."""
    print(f"\n{title}")
    print("-" * 65)
    print(f"{'Class':<8} | {'Precision':<12} | {'Recall':<12} | {'F1':<12}")
    print("-" * 65)

    classes = sorted(np.unique(y_true).astype(int))
    for k in classes:
        y_true_k = (y_true == k).astype(int)
        y_pred_k = (y_pred == k).astype(int)
        prec = get_precision(y_true_k, y_pred_k)
        rec = get_recall(y_true_k, y_pred_k)
        f1 = get_f1_score(y_true_k, y_pred_k)
        print(f"  {k:<6} | {prec:<12.4f} | {rec:<12.4f} | {f1:<12.4f}")

    print("-" * 65)
    acc = get_accuracy(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}\n")


def plot_decision_boundary(
    tree: DecisionTree,
    X: np.ndarray,
    y: np.ndarray,
    dataset_label: str,
    dataset_desc: str,
    max_depth_label: str,
    resolution: float = 0.02,
) -> None:
    """
    Plot the decision boundary of a 2D decision tree using contourf.

    Args:
        tree:            Fitted DecisionTree instance.
        X:               Feature matrix (N, 2) used for axis extent.
        y:               True labels for scatter plot.
        dataset_label:   Short label (e.g. "A").
        dataset_desc:    Description string.
        max_depth_label: String label for the depth used.
        resolution:      Mesh grid resolution.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = tree.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=20,
                cmap=plt.cm.RdYlBu)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Dataset {dataset_label}: {dataset_desc}\n"
              f"Decision Boundary — max_depth={max_depth_label}")
    plt.tight_layout()
    fname = f"boundary_{dataset_label}_depth{max_depth_label}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[Saved] {fname}")


def plot_accuracy_vs_depth(
    depths: list,
    train_accs: list,
    test_accs: list,
    dataset_label: str,
    dataset_desc: str,
) -> None:
    """Plot train vs test accuracy across tree depths."""
    depth_labels = [str(d) if d is not None else "None" for d in depths]

    plt.figure(figsize=(8, 5))
    plt.plot(depth_labels, train_accs, "o-", label="Train Accuracy", linewidth=1.5)
    plt.plot(depth_labels, test_accs,  "s--", label="Test Accuracy", linewidth=1.5)
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title(f"Dataset {dataset_label}: {dataset_desc}\n"
              f"Train vs Test Accuracy by Tree Depth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    fname = f"accuracy_vs_depth_{dataset_label}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[Saved] {fname}")


# ------------------------------------------------------------------
# Per-dataset experiment
# ------------------------------------------------------------------

def run_experiment(
    dataset_label: str,
    data_dir: str,
    dataset_desc: str,
    is_2d: bool,
    is_binary: bool,
) -> None:
    """Run the full decision tree experiment on one dataset."""

    print("\n" + "=" * 65)
    print(f"  Dataset {dataset_label}: {dataset_desc}")
    print("=" * 65)

    # ---- Load data ----
    try:
        X_train, y_train = load_data(f"{data_dir}/train.csv")
        X_test,  y_test  = load_data(f"{data_dir}/test.csv")
    except FileNotFoundError:
        print(f"ERROR: {data_dir}/ not found. Run setup_data.py first.")
        return

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    N, D = X_train.shape
    n_classes = len(np.unique(y_train))
    print(f"Train: ({N}, {D})  |  Test: {X_test.shape}  |  Classes: {n_classes}")

    # ==================================================================
    # TASK 3 — Train and Evaluate at multiple depths
    # ==================================================================
    train_accs = []
    test_accs = []
    fitted_trees = {}

    for depth in EVAL_DEPTHS:
        depth_str = str(depth) if depth is not None else "None"
        print(f"\n[Task 3] Training DecisionTree with max_depth={depth_str} …")

        tree = DecisionTree(max_depth=depth, min_samples_split=2)
        tree.fit(X_train, y_train)

        preds_train = tree.predict(X_train)
        preds_test = tree.predict(X_test)

        train_acc = get_accuracy(y_train, preds_train)
        test_acc = get_accuracy(y_test, preds_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        fitted_trees[depth] = tree

        print(f"  Tree depth (actual) = {tree.get_depth()}, "
              f"Leaves = {tree.get_n_leaves()}")

        print_metrics_report(
            y_test, preds_test,
            f"Decision Tree (max_depth={depth_str}) — "
            f"Dataset {dataset_label} (Test Set)",
        )

    # ==================================================================
    # Accuracy vs Depth plot
    # ==================================================================
    plot_accuracy_vs_depth(EVAL_DEPTHS, train_accs, test_accs,
                           dataset_label, dataset_desc)

    # ==================================================================
    # Decision boundary plots (2D datasets only)
    # ==================================================================
    if is_2d:
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

        for pd_depth in PLOT_DEPTHS:
            if pd_depth in fitted_trees:
                tree = fitted_trees[pd_depth]
            else:
                tree = DecisionTree(max_depth=pd_depth, min_samples_split=2)
                tree.fit(X_train, y_train)
            plot_decision_boundary(
                tree, X_all, y_all,
                dataset_label, dataset_desc,
                max_depth_label=str(pd_depth),
            )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  LAB 5  Q2 — Decision Trees: The Geometry of Splitting")
    print("=" * 65)

    for label, data_dir, desc, is_2d, is_binary in DATASETS:
        run_experiment(label, data_dir, desc, is_2d, is_binary)

    print("\n" + "=" * 65)
    print("  All experiments complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
