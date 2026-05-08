"""
Section 4 – torch.nn.Module: The Model Engine
===============================================
log_reg_module.py

  4.0  Logistic Regression as an nn.Module    [INSPECTION]

Run:
    python log_reg_module.py

Trains logistic regression on the two-moons dataset using two
implementations side by side:
  (A) a hand-written NumPy class that manages its own parameters,
  (B) a PyTorch nn.Module that delegates parameter storage and
      gradient bookkeeping to the framework.

Running this file prints per-run accuracies, runs unit tests,
and saves a side-by-side decision boundary plot.
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
N_EPOCHS = 300
LR       = 0.1

rng = np.random.default_rng(SEED)
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

# ── shared helpers ─────────────────────────────────────────────────────────────


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    """Element-wise logistic function."""
    return 1.0 / (1.0 + np.exp(-z))


def bce_np(y_hat: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy loss."""
    return float(-np.mean(
        y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
    ))


# ── NumPy implementation ───────────────────────────────────────────────────────


class LogisticRegressionNumPy:
    """
    Logistic regression trained with full-batch gradient descent in NumPy.

    Model:   p(y=1 | x) = sigmoid(x @ w + b)
    Loss:    binary cross-entropy  L(w, b)
    Update:  gradient descent with the analytical gradients:

        dL/dw = (1/N) X^T (y_hat - y)
        dL/db = (1/N) sum(y_hat - y)

    This class manages its own parameters explicitly as plain NumPy arrays.
    Compare the amount of bookkeeping against LogisticRegressionTorch below.
    """

    def __init__(self, input_dim: int, lr: float = 0.1, seed: int = 0):
        _rng = np.random.default_rng(seed)
        self.w  = _rng.normal(0, 0.01, size=(input_dim,)).astype(np.float32)
        self.b  = np.float32(0.0)
        self.lr = lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities, shape (N,)."""
        return sigmoid_np(X @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 or 1), shape (N,)."""
        return (self.predict_proba(X) >= 0.5).astype(np.float32)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_epochs: int = 300) -> "LogisticRegressionNumPy":
        """Full-batch gradient descent for n_epochs. Returns self."""
        N = len(y)
        self.losses_ = []
        for _ in range(n_epochs):
            # ── forward ──────────────────────────────────────────────────────
            y_hat = self.predict_proba(X)
            loss  = bce_np(y_hat, y)
            self.losses_.append(loss)

            # ── backward (hand-coded gradients) ──────────────────────────────
            err = y_hat - y                    # (N,)
            dw  = (X.T @ err) / N             # (D,)
            db  = err.mean()                  # scalar

            # ── update ───────────────────────────────────────────────────────
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self


# ── PyTorch implementation ─────────────────────────────────────────────────────


class LogisticRegressionTorch(nn.Module):
    """
    The same logistic regression model, expressed as an nn.Module.

    The key difference from LogisticRegressionNumPy:
      - Parameters (w, b) live inside nn.Linear and are registered
        automatically as nn.Parameter objects — no manual tracking.
      - forward() defines only the computation graph — no derivative code.
      - Training (gradient zeroing, backward pass, step) is handled
        outside the class, in the training loop below.

    nn.Linear(in_features, out_features) stores:
      - weight of shape (out_features, in_features)
      - bias   of shape (out_features,)
    and computes  y = x @ weight.T + bias.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)   # one output unit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: linear → sigmoid.

        nn.Linear produces shape (N, 1); we squeeze the trailing
        dimension to return shape (N,), matching the NumPy convention.
        """
        logit = self.linear(x)               # (N, 1)
        prob  = torch.sigmoid(logit)         # (N, 1), values in (0, 1)
        return prob.squeeze(1)               # (N,)


# ── training loop (PyTorch) ────────────────────────────────────────────────────


def train_torch(model: nn.Module,
                X: torch.Tensor,
                y: torch.Tensor,
                lr: float = 0.1,
                n_epochs: int = 300) -> list:
    """
    Full-batch gradient descent using PyTorch autograd.

    The five canonical lines inside the loop map directly to the NumPy
    loop in LogisticRegressionNumPy.fit():

        model(X)             ↔  y_hat = predict_proba(X)
        loss_fn(y_hat, y)    ↔  loss  = bce_np(y_hat, y)
        optimizer.zero_grad() ↔  (implicit in NumPy — w.grad = 0)
        loss.backward()      ↔  dw, db = hand_coded_gradients(...)
        optimizer.step()     ↔  w -= lr * dw;  b -= lr * db

    Returns a list of per-epoch losses.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn   = nn.BCELoss()
    losses    = []

    for _ in range(n_epochs):
        y_hat = model(X)                   # forward pass
        loss  = loss_fn(y_hat, y)          # compute loss
        optimizer.zero_grad()              # clear stale gradients
        loss.backward()                    # compute new gradients
        optimizer.step()                   # update parameters
        losses.append(loss.item())

    return losses


# ── unit tests ────────────────────────────────────────────────────────────────


def test_numpy_trains():
    """NumPy model reaches > 80 % accuracy on two-moons after 300 epochs."""
    model = LogisticRegressionNumPy(input_dim=2, lr=LR, seed=SEED)
    model.fit(X_np, y_np, n_epochs=N_EPOCHS)
    acc = model.accuracy(X_np, y_np)
    assert acc > 0.80, (
        "4.0 FAIL – NumPy model accuracy too low\n"
        f"  got: {acc:.4f}   expected: > 0.80"
    )
    print("  ✓  test_numpy_trains passed")


def test_torch_trains():
    """PyTorch model reaches > 80 % accuracy on two-moons after 300 epochs."""
    torch.manual_seed(SEED)
    model = LogisticRegressionTorch(input_dim=2)
    train_torch(model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)
    with torch.no_grad():
        preds = (model(X_t) >= 0.5).float()
    acc = (preds == y_t).float().mean().item()
    assert acc > 0.80, (
        "4.0 FAIL – PyTorch model accuracy too low\n"
        f"  got: {acc:.4f}   expected: > 0.80"
    )
    print("  ✓  test_torch_trains passed")


def test_parameter_count():
    """LogisticRegressionTorch must have exactly 2 parameter tensors: weight + bias."""
    model  = LogisticRegressionTorch(input_dim=2)
    params = list(model.parameters())
    assert len(params) == 2, (
        "4.0 FAIL – wrong parameter count\n"
        f"  got: {len(params)}   expected: 2"
    )
    print("  ✓  test_parameter_count passed")


def test_output_shape():
    """forward() must return shape (N,), not (N, 1)."""
    model = LogisticRegressionTorch(input_dim=2)
    out   = model(X_t)
    assert out.shape == (len(X_t),), (
        "4.0 FAIL – output shape incorrect\n"
        f"  got: {tuple(out.shape)}   expected: ({len(X_t)},)"
    )
    print("  ✓  test_output_shape passed")


def test_output_range():
    """All outputs must be strictly in (0, 1) — sigmoid guarantees this."""
    model = LogisticRegressionTorch(input_dim=2)
    out   = model(X_t)
    assert out.min().item() > 0.0 and out.max().item() < 1.0, (
        "4.0 FAIL – outputs not in (0, 1)\n"
        f"  min: {out.min().item():.6f}   max: {out.max().item():.6f}"
    )
    print("  ✓  test_output_range passed")


def test_loss_decreases():
    """Training loss at epoch 300 must be lower than at epoch 1."""
    torch.manual_seed(SEED)
    model  = LogisticRegressionTorch(input_dim=2)
    losses = train_torch(model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)
    assert losses[-1] < losses[0], (
        "4.0 FAIL – loss did not decrease\n"
        f"  epoch 1 loss: {losses[0]:.4f}   epoch 300 loss: {losses[-1]:.4f}"
    )
    print("  ✓  test_loss_decreases passed")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Section 4.0 — Logistic Regression as nn.Module  [INSPECTION]")
    print("=" * 60)

    for fn in [test_numpy_trains, test_torch_trains,
               test_parameter_count, test_output_shape,
               test_output_range, test_loss_decreases]:
        try:
            fn()
        except NotImplementedError as e:
            print(f"  (skipped – not implemented yet: {e})")
        except AssertionError as e:
            print(f"  FAIL: {e}")

    # ── train both models for comparison ─────────────────────────────────────
    print("\nTraining both models for 300 epochs...")

    np_model = LogisticRegressionNumPy(input_dim=2, lr=LR, seed=SEED)
    np_model.fit(X_np, y_np, n_epochs=N_EPOCHS)
    print(f"  NumPy  accuracy: {np_model.accuracy(X_np, y_np):.3f}")

    torch.manual_seed(SEED)
    torch_model  = LogisticRegressionTorch(input_dim=2)
    torch_losses = train_torch(torch_model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)
    with torch.no_grad():
        torch_preds = (torch_model(X_t) >= 0.5).float()
    torch_acc = (torch_preds == y_t).float().mean().item()
    print(f"  PyTorch accuracy: {torch_acc:.3f}")

    # ── decision boundary plot ────────────────────────────────────────────────
    h = 0.02
    x_min, x_max = X_np[:, 0].min() - 0.4, X_np[:, 0].max() + 0.4
    y_min, y_max = X_np[:, 1].min() - 0.4, X_np[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid   = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = torch.tensor(grid)

    Z_np = np_model.predict_proba(grid).reshape(xx.shape)
    with torch.no_grad():
        Z_torch = torch_model(grid_t).numpy().reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plt.suptitle("Section 4.0 — Logistic Regression: NumPy class vs. nn.Module",
                 fontsize=12)

    for ax, Z, title in zip(axes[:2], [Z_np, Z_torch],
                            ["NumPy class", "PyTorch nn.Module"]):
        ax.contourf(xx, yy, Z, alpha=0.3, levels=20, cmap="RdBu")
        ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np,
                   cmap="RdBu", edgecolors="k", s=18, linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    # training curves
    axes[2].plot(np_model.losses_,    label="NumPy",   linewidth=1.5)
    axes[2].plot(torch_losses,        label="PyTorch",  linewidth=1.5, linestyle="--")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("BCE loss")
    axes[2].set_title("Training curves")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("4_0_logistic_regression_module.png", dpi=130)
    print("\nPlot saved → 4_0_logistic_regression_module.png")
