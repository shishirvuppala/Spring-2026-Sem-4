"""
Section 2.3 – Logistic Regression Without Any Abstractions
============================================================
logistic_regression_raw.py                               [INSPECTION]

Two complete logistic regression trainers on the same two-moons dataset:

  Version A  ──  Pure NumPy   (~35 lines of logic)
  Version B  ──  Raw PyTorch  (~25 lines of logic, autograd replaces backward)

TASK: Read both versions carefully.  Run the file.  Verify that the two
decision boundaries are (essentially) the same.  Understand what changed
and what didn't.

         Forward code:   ~identical
         Backward code:  disappeared   (replaced by loss.backward())
         Update code:    slightly different (.data / .grad.zero_())

At the end of the file, a teaser asks: can we do better?
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Shared hyper-parameters ───────────────────────────────────────────────────
SEED      = 0
LR        = 0.5
N_EPOCHS  = 1000

# ── Shared data (NumPy arrays; Version B will convert to tensors later) ───────
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
X_np = (X_np - X_np.mean(axis=0)) / X_np.std(axis=0)   # standardise

rng = np.random.default_rng(SEED)
W_INIT = rng.standard_normal(2) * 0.01   # same starting point for both

# ═════════════════════════════════════════════════════════════════════════════
# VERSION A — Pure NumPy
# ═════════════════════════════════════════════════════════════════════════════

def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _bce_np(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy  L = −(1/N) Σ [ y log ŷ + (1−y) log(1−ŷ) ]"""
    eps = 1e-9
    y_hat = np.clip(y_hat, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


# Initialise parameters
w_A = W_INIT.copy()   # shape (2,)
b_A = 0.0             # scalar

losses_A = []

for epoch in range(N_EPOCHS):

    # ── Forward pass ─────────────────────────────────────────────────────────
    logits = X_np @ w_A + b_A          # (N,)
    y_hat  = _sigmoid_np(logits)       # (N,)  predictions in (0, 1)
    loss   = _bce_np(y_hat, y_np)      # scalar

    # ── Backward pass (HAND-CODED GRADIENTS — 4 lines) ───────────────────────
    #   ∂L/∂w = (1/N) Xᵀ (ŷ − y)
    #   ∂L/∂b = (1/N) Σᵢ (ŷᵢ − yᵢ)
    err = y_hat - y_np                 # (N,)  residuals
    dw  = (X_np.T @ err) / len(y_np)  # (D,)
    db  = err.mean()                   # scalar

    # ── Parameter update ─────────────────────────────────────────────────────
    w_A -= LR * dw
    b_A -= LR * db

    losses_A.append(loss)

acc_A = float(np.mean((_sigmoid_np(X_np @ w_A + b_A) >= 0.5) == y_np))
print(f"Version A (NumPy)    epoch={N_EPOCHS}  "
      f"loss={losses_A[-1]:.4f}  acc={acc_A:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# VERSION B — Raw PyTorch (no nn.Module, no optim — autograd only)
# ═════════════════════════════════════════════════════════════════════════════

# Convert data once
X_t = torch.tensor(X_np, dtype=torch.float32)
y_t = torch.tensor(y_np, dtype=torch.float32)

# Same starting parameters — note requires_grad=True
w_B = torch.tensor(W_INIT.copy(), dtype=torch.float32, requires_grad=True)
b_B = torch.tensor(0.0,           dtype=torch.float32, requires_grad=True)

losses_B = []

for epoch in range(N_EPOCHS):

    # ── Forward pass (compare line-by-line with Version A above) ─────────────
    logits = X_t @ w_B + b_B                            # (N,)
    y_hat  = torch.sigmoid(logits)                      # (N,)
    eps    = 1e-9
    loss   = -torch.mean(
        y_t * torch.log(y_hat + eps)
        + (1 - y_t) * torch.log(1 - y_hat + eps)
    )                                                   # scalar tensor

    # ── Backward pass (AUTOGRAD — replaces the 4 gradient lines in A) ────────
    if w_B.grad is not None:
        w_B.grad.zero_()
    if b_B.grad is not None:
        b_B.grad.zero_()
    loss.backward()                    # ← one call; PyTorch does the calculus

    # ── Parameter update (use .data to avoid tracking the update itself) ─────
    with torch.no_grad():
        w_B -= LR * w_B.grad
        b_B -= LR * b_B.grad

    losses_B.append(loss.item())

with torch.no_grad():
    preds_B = (torch.sigmoid(X_t @ w_B + b_B) >= 0.5).numpy()
acc_B = float(np.mean(preds_B == y_np))
print(f"Version B (PyTorch)  epoch={N_EPOCHS}  "
      f"loss={losses_B[-1]:.4f}  acc={acc_B:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# Plots
# ═════════════════════════════════════════════════════════════════════════════

def _decision_surface(w, b, numpy_mode: bool) -> tuple:
    """Return meshgrid + predicted probability surface for plotting."""
    x0 = np.linspace(X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5, 350)
    x1 = np.linspace(X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5, 350)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    if numpy_mode:
        Z = _sigmoid_np(grid @ w + b)
    else:
        g_t = torch.tensor(grid)
        with torch.no_grad():
            Z = torch.sigmoid(g_t @ w + b).numpy()
    return xx, yy, Z.reshape(xx.shape)


def _scatter(ax):
    ax.scatter(X_np[:, 0], X_np[:, 1], c=y_np,
               cmap="RdBu", edgecolors="k", s=18, linewidths=0.4, zorder=3)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1 — Version A boundary
xx, yy, Z = _decision_surface(w_A, b_A, numpy_mode=True)
axes[0].contourf(xx, yy, Z, levels=40, cmap="RdBu", alpha=0.65)
_scatter(axes[0])
axes[0].set_title(f"Version A — NumPy\nacc={acc_A:.3f}", fontsize=10)
axes[0].set_xlabel("x₁"); axes[0].set_ylabel("x₂")

# Panel 2 — Version B boundary
xx, yy, Z = _decision_surface(w_B, b_B, numpy_mode=False)
axes[1].contourf(xx, yy, Z, levels=40, cmap="RdBu", alpha=0.65)
_scatter(axes[1])
axes[1].set_title(f"Version B — PyTorch (autograd)\nacc={acc_B:.3f}", fontsize=10)
axes[1].set_xlabel("x₁")

# Panel 3 — Training curves
axes[2].plot(losses_A, lw=2,   label="Version A (NumPy)",   color="tomato")
axes[2].plot(losses_B, lw=2, linestyle="--",
             label="Version B (PyTorch)", color="steelblue")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("BCE Loss")
axes[2].set_title("Training curves"); axes[2].legend(fontsize=8)

plt.suptitle("Section 2.3 — Logistic Regression Without Any Abstractions",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("2_3_logistic_regression_raw.png", dpi=130)
print("Plot saved → 2_3_logistic_regression_raw.png")


# ═════════════════════════════════════════════════════════════════════════════
# Sanity checks
# ═════════════════════════════════════════════════════════════════════════════

assert acc_A > 0.75, f"Version A accuracy too low: {acc_A:.3f}"
assert acc_B > 0.75, f"Version B accuracy too low: {acc_B:.3f}"
assert abs(acc_A - acc_B) < 0.07, (
    f"Versions diverged: A={acc_A:.3f}  B={acc_B:.3f}  "
    "(started from the same weights — something is wrong)"
)
print(f"\n✓  Both versions reach similar accuracy "
      f"(A={acc_A:.3f}, B={acc_B:.3f})")
print("  Decision boundaries in the plot should be (nearly) identical.\n")


# ═════════════════════════════════════════════════════════════════════════════
# ── Teaser ───────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

print("""
──────────────────────────────────────────────────────────────
Notice what Version B still does manually:

  • w_B.grad.zero_()  ← zero gradients before each backward pass
  • loss.backward()   ← ok, autograd handles this
  • w_B -= lr * w_B.grad  ← the update rule is still hand-written
  • requires_grad=True on every parameter by hand
  • no .parameters() — we can't easily swap optimisers
  • no .to(device)  — moving to GPU would require manual .cuda() calls

We have raw tensors floating around.  Every ML pipeline has the same
bookkeeping mess.  Can we do better?

→ Section 3 names the abstractions.
→ Sections 4–6 implement them.
──────────────────────────────────────────────────────────────
""")