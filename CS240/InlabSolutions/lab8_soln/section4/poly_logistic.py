"""
Section 4 – torch.nn.Module: The Model Engine
===============================================
poly_logistic.py

  4.2  Polynomial Logistic Regression        [INSPECTION]

Run:
    python poly_logistic.py

The two-moons dataset is not linearly separable: no single straight
line can separate the two classes.  This file demonstrates that
nn.Module can encapsulate non-learnable preprocessing alongside
learnable parameters.  PolynomialLogisticRegression expands x ∈ R²
to all monomials x1^a * x2^b with a+b ≤ degree, then fits a single
linear layer on the expanded features.  With degree=2 the boundary
becomes a conic section, which is enough to separate two moons.

Nothing to implement — read the code, run the file, observe the plot.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

# ── data ──────────────────────────────────────────────────────────────────────
SEED     = 0
N_EPOCHS = 1000
LR       = 0.1
DEGREE   = 2

torch.manual_seed(SEED)

def load_moons_dataset_from_hardcoded_path():
    """Load moons dataset from a fixed local .npz file path."""
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "moons_dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Moons dataset not found at hardcoded path: {dataset_path}. "
            "Run aux/data/load_data.py first to generate moons_dataset.npz."
        )

    npz_data = np.load(dataset_path)
    x_np = npz_data["x_train"]
    y_np = npz_data["y_train"]
    return x_np, y_np


X_np, y_np = load_moons_dataset_from_hardcoded_path()
X_np = X_np.astype(np.float32)
y_np = y_np.astype(np.float32)

X_t = torch.tensor(X_np)
y_t = torch.tensor(y_np)


# ── model ─────────────────────────────────────────────────────────────────────


class PolynomialLogisticRegression(nn.Module):
    """
    Logistic regression with polynomial feature expansion.

    Architecture:
        1. _expand_features(x)  — non-learnable feature map R^{B×2} → R^{B×M}
        2. nn.Linear(M, 1)      — learnable linear layer
        3. sigmoid              — squash to probability

    The feature expansion is NOT part of the parameter set.
    list(model.parameters()) returns only the weight and bias of
    the linear layer: two tensors, M+1 scalars total.

    Parameters
    ----------
    degree : int
        Maximum total monomial degree.  M = (degree+2)*(degree+1)//2.
        degree=1 → plain logistic regression (3 params for 2-d input).
        degree=2 → adds x1², x1·x2, x2² (6 params total).
    """

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree
        M = (degree + 2) * (degree + 1) // 2    # number of monomials
        self.linear = nn.Linear(M, 1)

    # ── non-learnable feature map ──────────────────────────────────────────

    def _expand_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enumerate all (a, b) with a ≥ 0, b ≥ 0, a+b ≤ degree in
        lexicographic order (a outer, b inner) and compute x1^a * x2^b.

        Returns tensor of shape (B, M).

        Example, degree=2:
            (0,0) → 1
            (0,1) → x2
            (0,2) → x2²
            (1,0) → x1
            (1,1) → x1·x2
            (2,0) → x1²
        """
        x1 = x[:, 0:1]    # (B, 1)
        x2 = x[:, 1:2]    # (B, 1)
        features = []
        for a in range(self.degree + 1):
            for b in range(self.degree + 1 - a):
                features.append(x1 ** a * x2 ** b)   # (B, 1)
        return torch.cat(features, dim=1)              # (B, M)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand → linear → sigmoid.  Returns probabilities of shape (B,).
        """
        phi  = self._expand_features(x)    # (B, M)
        out  = self.linear(phi)            # (B, 1)
        return torch.sigmoid(out).squeeze(1)  # (B,)


# ── training loop ─────────────────────────────────────────────────────────────


def train(model: nn.Module,
          X: torch.Tensor,
          y: torch.Tensor,
          lr: float = 0.1,
          n_epochs: int = 1000) -> list:
    """Standard full-batch training loop."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn   = nn.BCELoss()
    losses    = []
    for _ in range(n_epochs):
        y_hat = model(X)
        loss  = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# ── unit tests ────────────────────────────────────────────────────────────────


def test_expand_shape():
    """_expand_features must return shape (B, M) where M = (d+2)(d+1)//2."""
    for degree in [1, 2, 3]:
        model = PolynomialLogisticRegression(degree=degree)
        M_expected = (degree + 2) * (degree + 1) // 2
        phi = model._expand_features(X_t)
        assert phi.shape == (len(X_t), M_expected), (
            f"4.2 FAIL – _expand_features shape incorrect for degree={degree}\n"
            f"  got: {tuple(phi.shape)}   expected: ({len(X_t)}, {M_expected})"
        )
    print("  ✓  test_expand_shape passed")


def test_constant_column():
    """The first expanded feature (a=0, b=0) must be all-ones (bias term)."""
    model = PolynomialLogisticRegression(degree=2)
    phi   = model._expand_features(X_t)
    assert torch.allclose(phi[:, 0], torch.ones(len(X_t))), (
        "4.2 FAIL – first column of expanded features is not all-ones\n"
        "  (0,0) monomial = x1^0 * x2^0 = 1 for all samples"
    )
    print("  ✓  test_constant_column passed")


def test_parameter_count():
    """Only the linear layer should have learnable parameters."""
    model  = PolynomialLogisticRegression(degree=DEGREE)
    M      = (DEGREE + 2) * (DEGREE + 1) // 2
    params = list(model.parameters())
    # two tensors: weight (1, M) and bias (1,)
    assert len(params) == 2, (
        "4.2 FAIL – unexpected number of parameter tensors\n"
        f"  got: {len(params)}   expected: 2"
    )
    total = sum(p.numel() for p in params)
    assert total == M + 1, (
        "4.2 FAIL – total parameter count incorrect\n"
        f"  got: {total}   expected: {M + 1}  (M={M} weights + 1 bias)"
    )
    print("  ✓  test_parameter_count passed")


def test_degree1_is_linear():
    """
    With degree=1 the decision boundary is linear (same as plain logistic
    regression).  Training should still converge.
    """
    torch.manual_seed(SEED)
    model  = PolynomialLogisticRegression(degree=1)
    losses = train(model, X_t, y_t, lr=LR, n_epochs=500)
    assert losses[-1] < losses[0], (
        "4.2 FAIL – degree-1 model loss did not decrease\n"
        f"  epoch 1: {losses[0]:.4f}   epoch 500: {losses[-1]:.4f}"
    )
    print("  ✓  test_degree1_is_linear passed")


def test_degree2_accuracy():
    """Degree-2 model must reach > 85 % accuracy (two-moons is separable at degree 2)."""
    torch.manual_seed(SEED)
    model = PolynomialLogisticRegression(degree=2)
    train(model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)
    with torch.no_grad():
        preds = (model(X_t) >= 0.5).float()
    acc = (preds == y_t).float().mean().item()
    assert acc > 0.85, (
        "4.2 FAIL – degree-2 model accuracy too low\n"
        f"  got: {acc:.4f}   expected: > 0.85"
    )
    print("  ✓  test_degree2_accuracy passed")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Section 4.2 — Polynomial Logistic Regression  [INSPECTION]")
    print("=" * 60)

    for fn in [test_expand_shape, test_constant_column,
               test_parameter_count, test_degree1_is_linear,
               test_degree2_accuracy]:
        try:
            fn()
        except NotImplementedError as e:
            print(f"  (skipped – not implemented yet: {e})")
        except AssertionError as e:
            print(f"  FAIL: {e}")

    # ── train degree-1 and degree-2 for comparison ────────────────────────────
    print("\nTraining models for comparison...")

    results = {}
    for deg in [1, 2]:
        torch.manual_seed(SEED)
        model  = PolynomialLogisticRegression(degree=deg)
        losses = train(model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)
        with torch.no_grad():
            preds = (model(X_t) >= 0.5).float()
        acc = (preds == y_t).float().mean().item()
        results[deg] = {"model": model, "losses": losses, "acc": acc}
        M = (deg + 2) * (deg + 1) // 2
        print(f"  degree={deg}  M={M:2d} features  "
              f"accuracy={acc:.3f}  final_loss={losses[-1]:.4f}")

    # ── decision boundary plot ────────────────────────────────────────────────
    h = 0.02
    x_min, x_max = X_np[:, 0].min() - 0.4, X_np[:, 0].max() + 0.4
    y_min, y_max = X_np[:, 1].min() - 0.4, X_np[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid   = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = torch.tensor(grid)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plt.suptitle("Section 4.2 — Polynomial Logistic Regression", fontsize=12)

    for ax, deg in zip(axes[:2], [1, 2]):
        m = results[deg]["model"]
        with torch.no_grad():
            Z = m(grid_t).numpy().reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, levels=20, cmap="RdBu")
        ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np,
                   cmap="RdBu", edgecolors="k", s=18, linewidths=0.5)
        M = (deg + 2) * (deg + 1) // 2
        ax.set_title(f"degree={deg}  ({M} features)  acc={results[deg]['acc']:.2f}")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    axes[2].plot(results[1]["losses"], label="degree=1", linewidth=1.5)
    axes[2].plot(results[2]["losses"], label="degree=2", linewidth=1.5, linestyle="--")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("BCE loss")
    axes[2].set_title("Training curves")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("4_2_polynomial_logistic.png", dpi=130)
    print("\nPlot saved → 4_2_polynomial_logistic.png")