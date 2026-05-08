"""
main.py
=======
Lab 5, Q1: Main experiment script (provided to students — DO NOT MODIFY).

This script runs the Generative-Discriminative hybrid experiment on THREE
datasets (Iris, Breast Cancer, Sonar).

For each dataset it:
  1. Trains an LDA model (Task 1 — closed-form).
  2. Trains two Logistic Regression models (Task 2 / Task 3):
       • Model A — Random (cold) initialisation.
       • Model B — Warm-start with LDA weights.
  3. Evaluates all three models (LDA, A, B) on the test set with
     accuracy, precision, recall, and F1.
  4. Plots Training Loss vs. Epochs for Models A and B.
  5. Reports initial and final loss values for convergence analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

from classifiers import LDAClassifier, LogisticRegression
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
# (label, data_dir, description, lr, max_iter)
DATASETS = [
    ("A", "data/iris",   "Iris (Sanity Check)",              0.01, 1000),
    ("B", "data/cancer", "Breast Cancer (Convergence)",      0.01, 1000),
    ("C", "data/sonar",  "Sonar (High-Dim Regularisation)",  0.01, 1000),
]


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


def plot_loss_comparison(
    loss_random: list,
    loss_warm: list,
    dataset_label: str,
    dataset_desc: str,
) -> None:
    """Plot training loss curves for cold-start vs warm-start."""
    epochs = np.arange(1, len(loss_random) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_random, label="Model A — Random Init", linewidth=1.5)
    plt.plot(epochs, loss_warm,   label="Model B — Warm Start (LDA)", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (BCE)")
    plt.title(f"Dataset {dataset_label}: {dataset_desc}\n"
              f"Convergence Comparison — Cold Start vs Warm Start")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"loss_comparison_{dataset_label}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"[Saved] {fname}")


# ------------------------------------------------------------------
# Per-dataset experiment
# ------------------------------------------------------------------

def run_experiment(
    dataset_label: str,
    data_dir: str,
    dataset_desc: str,
    lr: float,
    max_iter: int,
) -> None:
    """Run the full LDA → LR cold/warm experiment on one dataset."""

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

    N, D = X_train.shape
    print(f"Train: ({N}, {D})  |  Test: {X_test.shape}")

    # ==================================================================
    # TASK 1 — LDA (closed-form)
    # ==================================================================
    print("\n[Task 1] Fitting LDA …")
    lda = LDAClassifier()
    lda.fit(X_train, y_train)
    w_gen, b_gen = lda.get_params()

    preds_lda = lda.predict(X_test)
    print_metrics_report(y_test, preds_lda,
                         f"LDA Results — Dataset {dataset_label} (Test Set)")

    # ==================================================================
    # TASK 2 + 3 — Logistic Regression: Cold vs Warm
    # ==================================================================
    LR_PARAMS = dict(lr=lr, max_iter=max_iter, batch_size=N)
    # batch_size = N → full-batch GD (as specified in the problem)

    # --- Model A: Random initialisation ---
    print(f"[Task 3] Training Model A (Random Init) on dataset {dataset_label} …")
    model_a = LogisticRegression(**LR_PARAMS)
    model_a.fit(X_train, y_train, init_params=None)

    preds_a = model_a.predict(X_test)
    print_metrics_report(y_test, preds_a,
                         f"Model A — Random Init — Dataset {dataset_label} (Test Set)")

    # --- Model B: Warm start with LDA weights ---
    print(f"[Task 3] Training Model B (Warm Start) on dataset {dataset_label} …")
    model_b = LogisticRegression(**LR_PARAMS)
    model_b.fit(X_train, y_train, init_params=(w_gen, b_gen))

    preds_b = model_b.predict(X_test)
    print_metrics_report(y_test, preds_b,
                         f"Model B — Warm Start — Dataset {dataset_label} (Test Set)")

    # ==================================================================
    # Convergence analysis
    # ==================================================================
    print("-" * 65)
    print(f"Convergence Analysis — Dataset {dataset_label}")
    print("-" * 65)
    print(f"  Model A (Random) — Initial Loss (epoch 0): "
          f"{model_a.loss_history[0]:.6f}")
    print(f"  Model B (Warm)   — Initial Loss (epoch 0): "
          f"{model_b.loss_history[0]:.6f}")
    print(f"  Model A (Random) — Final   Loss:           "
          f"{model_a.loss_history[-1]:.6f}")
    print(f"  Model B (Warm)   — Final   Loss:           "
          f"{model_b.loss_history[-1]:.6f}")

    # --- Plot ---
    plot_loss_comparison(model_a.loss_history, model_b.loss_history,
                         dataset_label, dataset_desc)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  LAB 5  Q1 — Generative ↔ Discriminative Hybrid Experiment")
    print("=" * 65)

    for label, data_dir, desc, lr, mi in DATASETS:
        run_experiment(label, data_dir, desc, lr, mi)

    print("\n" + "=" * 65)
    print("  All experiments complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
