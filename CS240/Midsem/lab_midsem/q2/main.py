"""
main.py — Training and evaluation pipeline for LinearRegressionTree.

Workflow
--------
  1. Load the pre-generated Segment dataset from data/segment/.
  2. Train a LinearRegressionTree with the optimal hyperparameters
     defined in give_optimal_hyperparameters() inside tree.py.
  3. Report train / test metrics (MSE, RMSE, MAE, R^2).
  4. Produce four diagnostic plots to help students tune max_depth
     and lambda:

        Figure 1 — Surface comparison
            True terrain surface  |  Tree prediction surface
        Figure 2 — Prediction diagnostics
            Predicted vs Actual scatter  |  Residuals vs Predicted

Only dependencies: NumPy and Matplotlib.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" if you want
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from tree        import LinearRegressionTree, give_optimal_hyperparameters
from setup_data  import load_dataset
from utils       import compute_metrics, print_metrics, count_tree_nodes, get_leaf_params


# ======================================================================
#  I/O helpers
# ======================================================================

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ======================================================================
#  Surface grid helper
# ======================================================================

def prediction_grid(tree, n=100, x1_range=(0, 10), x2_range=(0, 10)):
    """
    Evaluate the tree on a dense 2-D grid over [x1_min, x1_max] x [x2_min, x2_max].

    Parameters
    ----------
    tree : LinearRegressionTree  (must be fitted)
    n    : int   — grid resolution per axis
    x1_range, x2_range : tuple of (min, max)

    Returns
    -------
    x1_grid : np.ndarray, shape (n, n)
    x2_grid : np.ndarray, shape (n, n)
    z_grid  : np.ndarray, shape (n, n)   — predicted values
    """
    x1 = np.linspace(*x1_range, n)
    x2 = np.linspace(*x2_range, n)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    z_grid = tree.predict(X_grid).reshape(n, n)
    return x1_grid, x2_grid, z_grid


def true_surface_grid(n=100, x1_range=(0, 10), x2_range=(0, 10)):
    """
    Evaluate the true noiseless quadrant terrain on a dense grid.

    Returns
    -------
    x1_grid, x2_grid, z_grid : np.ndarray, each shape (n, n)
    """
    x1 = np.linspace(*x1_range, n)
    x2 = np.linspace(*x2_range, n)
    x1g, x2g = np.meshgrid(x1, x2)
    z = np.zeros_like(x1g)
    z[(x1g <= 5) & (x2g <= 5)] = (3*x1g[(x1g <= 5) & (x2g <= 5)]
                                  + 4*x2g[(x1g <= 5) & (x2g <= 5)] + 10)
    z[(x1g >  5) & (x2g <= 5)] = (-2*x1g[(x1g >  5) & (x2g <= 5)]
                                  + 8*x2g[(x1g >  5) & (x2g <= 5)] + 35)
    z[(x1g <= 5) & (x2g >  5)] = (7*x1g[(x1g <= 5) & (x2g >  5)]
                                  - 3*x2g[(x1g <= 5) & (x2g >  5)] + 45)
    z[(x1g >  5) & (x2g >  5)] = (-5*x1g[(x1g >  5) & (x2g >  5)]
                                  - 6*x2g[(x1g >  5) & (x2g >  5)] + 120)
    return x1g, x2g, z


# ======================================================================
#  Plot helpers
# ======================================================================

def _heatmap(ax, x1g, x2g, z, title, cmap="RdYlBu_r", vmin=None, vmax=None):
    vm = dict(vmin=vmin, vmax=vmax) if vmin is not None else {}
    im = ax.contourf(x1g, x2g, z, levels=40, cmap=cmap, **vm)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.axvline(5, color="white", lw=1, ls="--", alpha=0.6)
    ax.axhline(5, color="white", lw=1, ls="--", alpha=0.6)
    return im


# ======================================================================
#  Figure 1 — Surface comparison
# ======================================================================

def fig_surface_comparison(tree, X_test, y_test, metrics_test):
    """Two-panel 2-D colour-map: true terrain vs tree prediction."""
    x1g, x2g, z_true = true_surface_grid()
    _, _, z_pred = prediction_grid(tree)

    vmin = min(z_true.min(), z_pred.min())
    vmax = max(z_true.max(), z_pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Surface Comparison — Quadrant Terrain", fontsize=12, fontweight="bold")

    im0 = _heatmap(axes[0], x1g, x2g, z_true, "True surface (noiseless)",
                   vmin=vmin, vmax=vmax)
    im1 = _heatmap(axes[1], x1g, x2g, z_pred,
                   f"Tree prediction  (depth={tree.max_depth}, λ={tree.lam})",
                   vmin=vmin, vmax=vmax)

    # overlay test scatter coloured by residual magnitude
    resid = (tree.predict(X_test) - y_test.ravel())
    sc = axes[1].scatter(X_test[:, 0], X_test[:, 1],
                         c=np.abs(resid), cmap="hot_r", s=6, alpha=0.5,
                         vmin=0, vmax=np.percentile(np.abs(resid), 95))
    fig.colorbar(sc, ax=axes[1], label="Residual |y−ŷ|", shrink=0.8)
    fig.colorbar(im0, ax=axes[0], label="y", shrink=0.8)

    plt.tight_layout()
    _save(fig, "fig1_surface_comparison.png")


# ======================================================================
#  Figure 2 — Prediction diagnostics
# ======================================================================

def fig_prediction_diagnostics(tree, X_train, y_train, X_test, y_test):
    """Predicted-vs-Actual and Residuals-vs-Predicted for train & test."""
    ytr_pred = tree.predict(X_train)
    yte_pred = tree.predict(X_test)
    ytr_true = y_train.ravel()
    yte_true = y_test.ravel()

    all_lo = min(ytr_true.min(), yte_true.min())
    all_hi = max(ytr_true.max(), yte_true.max())
    diag   = np.array([all_lo, all_hi])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Prediction Diagnostics", fontsize=12, fontweight="bold")

    # Panel A — Predicted vs Actual
    ax = axes[0]
    ax.scatter(ytr_true, ytr_pred, s=6,  alpha=0.35, color="#2196F3", label="Train")
    ax.scatter(yte_true, yte_pred, s=8,  alpha=0.55, color="#F44336", label="Test")
    ax.plot(diag, diag, "k--", lw=1.2, label="Perfect fit")
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted ŷ")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=8)

    # Panel B — Residuals vs Predicted
    ax = axes[1]
    ax.scatter(ytr_pred, ytr_true - ytr_pred, s=6,  alpha=0.35, color="#2196F3", label="Train")
    ax.scatter(yte_pred, yte_true - yte_pred, s=8,  alpha=0.55, color="#F44336", label="Test")
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Predicted ŷ")
    ax.set_ylabel("Residual  (y − ŷ)")
    ax.set_title("Residuals vs Predicted")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "fig2_prediction_diagnostics.png")



# ======================================================================
#  Main pipeline
# ======================================================================

def main():
    # ------------------------------------------------------------------
    # 1.  Load data
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  LinearRegressionTree — Segment Dataset Evaluation")
    print("="*60)

    X_train, y_train, X_test, y_test = load_dataset()
    print(f"\n[Data]  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"        Test:  X={X_test.shape},  y={y_test.shape}")

    # ------------------------------------------------------------------
    # 2.  Train with optimal hyperparameters
    # ------------------------------------------------------------------
    max_depth, lam = give_optimal_hyperparameters()
    print(f"\n[Hyperparameters]  max_depth={max_depth},  λ={lam}")

    tree = LinearRegressionTree(max_depth=max_depth, lam=lam)
    tree.fit(X_train, y_train.ravel())

    info = count_tree_nodes(tree.root)
    print(f"\n[Tree Structure]")
    print(f"    Total nodes    : {info['total']}")
    print(f"    Leaf nodes     : {info['leaves']}")
    print(f"    Internal nodes : {info['internal']}")
    print(f"    Max leaf depth : {info['depth']}")

    # ------------------------------------------------------------------
    # 3.  Compute metrics
    # ------------------------------------------------------------------
    m_train = compute_metrics(y_train, tree.predict(X_train))
    m_test  = compute_metrics(y_test,  tree.predict(X_test))

    print(f"\n[Metrics — Train]")
    print_metrics(m_train, label="Train")

    print(f"\n[Metrics — Test]")
    print_metrics(m_test,  label="Test")

    # ------------------------------------------------------------------
    # 4.  Diagnostic plots
    # ------------------------------------------------------------------
    print(f"\n[Plots]  Saving to '{PLOTS_DIR}/'")

    print("  Generating Figure 1: Surface comparison ...")
    fig_surface_comparison(tree, X_test, y_test, m_test)

    print("  Generating Figure 2: Prediction diagnostics ...")
    fig_prediction_diagnostics(tree, X_train, y_train, X_test, y_test)

    # print("  Generating Figure 3: max_depth sweep ...")
    # fig_depth_sweep(X_train, y_train, X_test, y_test, lam=lam)

    # print("  Generating Figure 4: lambda sweep ...")
    # fig_lambda_sweep(X_train, y_train, X_test, y_test, max_depth=max_depth)

    # print("  Generating Figure 5: Leaf parameters ...")
    # fig_leaf_params(tree)

    # ------------------------------------------------------------------
    # 5.  Summary table
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  {'Metric':<10} {'Train':>12}  {'Test':>12}")
    print("  " + "-"*36)
    for key in ("mse", "rmse", "mae", "r2"):
        print(f"  {key.upper():<10} {m_train[key]:>12.6f}  {m_test[key]:>12.6f}")
    print("="*60)
    print(f"\n  Optimal: max_depth={max_depth},  λ={lam}")
    print(f"  Noise floor: σ² = {1.5**2:.4f}  →  Test MSE / σ² = "
          f"{m_test['mse'] / 1.5**2:.4f}x\n")


if __name__ == "__main__":
    main()
