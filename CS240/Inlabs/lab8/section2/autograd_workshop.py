"""
Section 2 – No One Likes Derivatives
======================================
autograd_workshop.py

  2.1  Warm-up: A Scalar Function        [INSPECTION  – nothing to implement]
  2.2  Exercise: Loss Function Gradient  [TODO        – fill in two blanks]

Run:
    python autograd_workshop.py

Each section prints a header, runs its unit test, and (for 2.1) saves a plot.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # headless – swap to "TkAgg" / "Qt5Agg" locally
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 2.1  Warm-up: A Scalar Function                                  [INSPECTION]
#
#      f(x) = sin(x²) · e^{-x} / (1 + x²)
#
#      Four things are implemented for you:
#        ① NumPy forward
#        ② NumPy hand-coded derivative  (quotient-rule + chain-rule)
#        ③ PyTorch forward              (notice: identical to ①)
#        ④ PyTorch autograd derivative  (notice: 2 lines instead of ~8)
#
#      Your task: read, run, look at the plot, and understand WHY the two
#      derivative curves coincide perfectly.
# ══════════════════════════════════════════════════════════════════════════════

# ── ① NumPy forward ───────────────────────────────────────────────────────────
def f_numpy(x: np.ndarray) -> np.ndarray:
    """f(x) = sin(x²) · e^{−x} / (1 + x²)  element-wise."""
    return np.sin(x**2) * np.exp(-x) / (1.0 + x**2)


# ── ② NumPy hand-coded derivative ────────────────────────────────────────────
def df_numpy(x: np.ndarray) -> np.ndarray:
    """
    Quotient rule on  f = u/v,  then chain rule inside u.

        u  = sin(x²) · e^{−x}
        u' = e^{−x} · (2x cos(x²) − sin(x²))       [product rule + chain rule]

        v  = 1 + x²
        v' = 2x

        f' = (u'v − uv') / v²
    """
    u  = np.sin(x**2) * np.exp(-x)
    du = np.exp(-x) * (2*x * np.cos(x**2) - np.sin(x**2))
    v  = 1.0 + x**2
    dv = 2*x
    return (du * v - u * dv) / v**2


# ── ③ PyTorch forward  (spot-the-difference from ①: torch.* instead of np.*) ─
def f_torch(x: torch.Tensor) -> torch.Tensor:
    """Same maths as f_numpy, using torch ops so autograd can track it."""
    return torch.sin(x**2) * torch.exp(-x) / (1.0 + x**2)


# ── ④ PyTorch autograd derivative ────────────────────────────────────────────
def df_torch_scalar(x_val: float) -> float:
    """
    Ask PyTorch to differentiate  f  at a single point.

    Step 1 – create a leaf tensor with requires_grad=True so PyTorch
             starts recording operations.
    Step 2 – run the *exact same forward code* as f_torch.
    Step 3 – call .backward() → PyTorch walks the graph in reverse,
             accumulates dL/dx in x.grad.
    Step 4 – read x.grad.
    """
    x = torch.tensor(x_val, dtype=torch.float64, requires_grad=True)
    y = f_torch(x)
    y.backward()
    return x.grad.item()


def _df_torch_vectorised(xs: np.ndarray) -> np.ndarray:
    """Convenience: autograd over a whole grid for plotting."""
    x_t = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
    f_torch(x_t).sum().backward()   # sum() → scalar so .backward() works
    return x_t.grad.detach().numpy()


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_2_1():
    xs = np.linspace(-2.0, 2.0, 500)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    axes[0].plot(xs, f_numpy(xs), lw=2, color="steelblue")
    axes[0].set_title(r"$f(x)=\sin(x^2)\,e^{-x}\,/\,(1+x^2)$")
    axes[0].set_xlabel("x")

    axes[1].plot(xs, df_numpy(xs),             lw=3,
                 label="NumPy hand-coded f′(x)", color="tomato")
    axes[1].plot(xs, _df_torch_vectorised(xs), lw=2, linestyle="--",
                 label="PyTorch autograd f′(x)", color="navy")
    axes[1].set_title("Derivative comparison — do the curves overlap?")
    axes[1].set_xlabel("x")
    axes[1].legend(fontsize=8)

    plt.suptitle("Section 2.1 – Scalar Function", fontsize=11)
    plt.tight_layout()
    plt.savefig("2_1_derivative_comparison.png", dpi=130)
    print("  Plot saved → 2_1_derivative_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.2  Exercise: Loss Function Gradient                                  [TODO]
#
#      L(w) = (1/N) Σᵢ (σ(wᵀxᵢ) − yᵢ)²       σ(z) = 1/(1+e^{−z})
#
#      Given:
#        • NumPy forward  loss_numpy(w, X, y)
#        • NumPy hand-coded gradient  grad_loss_numpy(w, X, y)
#        • A unit test comparing the two
#
#      You implement:
#        • loss_and_grad_torch  (two TODOs below)
# ══════════════════════════════════════════════════════════════════════════════

# ── NumPy helpers (GIVEN – read and understand) ───────────────────────────────

def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def loss_numpy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    L(w) = (1/N) Σᵢ (σ(wᵀxᵢ) − yᵢ)²

    Parameters
    ----------
    w : shape (D,)
    X : shape (N, D)
    y : shape (N,)   binary labels in {0, 1}
    """
    y_hat = _sigmoid_np(X @ w)
    return float(np.mean((y_hat - y)**2))


def grad_loss_numpy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    ∂L/∂w  — derived by hand with the chain rule.

    Let  s = σ(Xw)  (shape N).  Then:

        ∂L/∂wⱼ = (2/N) Σᵢ (sᵢ − yᵢ) · σ′(sᵢ) · Xᵢⱼ

    where  σ′(z) = σ(z)(1 − σ(z)).  In matrix form:

        ∂L/∂w = (2/N) Xᵀ [ (s − y) ⊙ s ⊙ (1 − s) ]
    """
    s     = _sigmoid_np(X @ w)
    delta = (s - y) * s * (1.0 - s)
    return (2.0 / len(y)) * (X.T @ delta)


# ── PyTorch skeleton (fill in the two TODOs) ──────────────────────────────────

def loss_and_grad_torch(
        w: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute L(w) and ∂L/∂w using PyTorch autograd.

    Parameters
    ----------
    w : shape (D,), dtype=float32, requires_grad=True  ← the caller sets this up
    X : shape (N, D), dtype=float32
    y : shape (N,),   dtype=float32

    Returns
    -------
    loss : scalar tensor
    grad : tensor of shape (D,)  — the gradient ∂L/∂w
    """

    # ── TODO 1 ───────────────────────────────────────────────────────────────
    # Compute the loss L(w) in PyTorch.
    # Translate loss_numpy line-by-line: use torch.sigmoid and torch.mean.
    # Store the result in `loss` (it should be a scalar tensor).
    #
    # Hint: the forward code is ~2 lines, identical in structure to loss_numpy.
    # ─────────────────────────────────────────────────────────────────────────
    loss = None  # replace this
    raise NotImplementedError("TODO 1: compute the loss L(w) — see the docstring above")
    # ─────────────────────────────────────────────────────────────────────────

    # ── TODO 2 ───────────────────────────────────────────────────────────────
    # Obtain ∂L/∂w via autograd.
    # (a) Zero any gradient left over from a previous call:
    #         if w.grad is not None:  w.grad.zero_()
    # (b) Call  loss.backward()
    # (c) Copy the gradient out:  grad = w.grad.clone()
    # ─────────────────────────────────────────────────────────────────────────
    grad = None  # replace this
    raise NotImplementedError("TODO 2: use autograd to get ∂L/∂w — see the docstring above")
    # ─────────────────────────────────────────────────────────────────────────

    return loss, grad


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests
# ══════════════════════════════════════════════════════════════════════════════

def _make_synthetic(seed: int = 7, N: int = 60, D: int = 5):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((N, D)).astype(np.float32)
    y   = (rng.uniform(size=N) > 0.5).astype(np.float32)
    w   = (rng.standard_normal(D) * 0.1).astype(np.float32)
    return X, y, w


def test_2_1_derivative_match():
    """
    Hand-coded NumPy derivative == PyTorch autograd derivative on a grid.
    If this passes the two curves in the plot will be indistinguishable.
    """
    xs    = np.linspace(-2.0, 2.0, 300)
    df_np = df_numpy(xs)
    df_pt = _df_torch_vectorised(xs)
    assert np.allclose(df_np, df_pt, atol=1e-6), (
        "2.1 FAIL – NumPy gradient ≠ torch autograd gradient\n"
        f"  max |diff| = {np.abs(df_np - df_pt).max():.2e}"
    )
    print("  ✓  test_2_1_derivative_match passed")


def test_2_2_loss_value():
    """PyTorch loss value ≈ NumPy loss value."""
    X, y, w = _make_synthetic()
    loss_ref = loss_numpy(w, X, y)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32, requires_grad=True)

    loss_t, _ = loss_and_grad_torch(w_t, X_t, y_t)
    assert abs(loss_t.item() - loss_ref) < 1e-5, (
        f"2.2 FAIL – loss: torch={loss_t.item():.10f}  numpy={loss_ref:.10f}"
    )
    print("  ✓  test_2_2_loss_value passed")


def test_2_2_gradient_match():
    """PyTorch autograd gradient ≈ NumPy hand-coded gradient."""
    X, y, w = _make_synthetic()
    grad_ref = grad_loss_numpy(w, X, y)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32, requires_grad=True)

    _, grad_t = loss_and_grad_torch(w_t, X_t, y_t)
    assert torch.allclose(
        grad_t,
        torch.tensor(grad_ref, dtype=torch.float32),
        atol=1e-4,
    ), (
        "2.2 FAIL – gradient mismatch\n"
        f"  max |diff| = {(grad_t - torch.tensor(grad_ref, dtype=torch.float32)).abs().max():.2e}"
    )
    print("  ✓  test_2_2_gradient_match passed")


def test_2_2_grad_different_from_zero():
    """Sanity: gradient should not be the zero vector for typical data."""
    X, y, w = _make_synthetic(seed=99)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32, requires_grad=True)
    _, grad_t = loss_and_grad_torch(w_t, X_t, y_t)
    assert grad_t.norm().item() > 1e-6, "2.2 FAIL – gradient is (nearly) zero"
    print("  ✓  test_2_2_grad_different_from_zero passed")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ─── Section 2.1 ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Section 2.1 — Scalar Function  [INSPECTION]")
    print("=" * 60)
    test_2_1_derivative_match()
    plot_2_1()
    print()

    # ─── Section 2.2 ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Section 2.2 — Loss Function Gradient  [TODO]")
    print("=" * 60)
    for fn in [test_2_2_loss_value, test_2_2_gradient_match,
               test_2_2_grad_different_from_zero]:
        try:
            fn()
        except NotImplementedError as e:
            print(f"  (skipped – not implemented yet: {e})")
        except AssertionError as e:
            print(f"  FAIL: {e}")