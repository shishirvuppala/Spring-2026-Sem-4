"""
Section 4 – torch.nn.Module: The Model Engine
===============================================
fcnn.py

  4.3  Feed-Forward Neural Network           [TODO — ungraded]

Run:
    python fcnn.py

A two-hidden-layer fully connected neural network for binary
classification.  The NumPy reference (FeedForwardNumPy) shows the
manual matrix operations.  Your task is to implement the identical
architecture using nn.Module (FeedForwardNN).

NumPy reference:  complete — read to understand the architecture.
PyTorch TODO:     fill in __init__ and forward in FeedForwardNN.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

# ── data ──────────────────────────────────────────────────────────────────────
SEED       = 0
N_EPOCHS   = 500
LR         = 0.05
HIDDEN_DIM = 16

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


# ── NumPy reference implementation ────────────────────────────────────────────


class FeedForwardNumPy:
    """
    Two-hidden-layer fully connected network — reference implementation.

    Architecture (matches FeedForwardNN exactly):
        x  ∈ R^{B × input_dim}
        h1 = relu( x  @ W1.T + b1 )   ∈ R^{B × hidden_dim}
        h2 = relu( h1 @ W2.T + b2 )   ∈ R^{B × hidden_dim}
        out =      h2 @ W3.T + b3     ∈ R^{B × output_dim}

    Note: weights are stored transposed relative to the mathematical
    convention so that the computation matches nn.Linear, which stores
    weight of shape (out_features, in_features) and computes x @ W.T + b.

    Parameters
    ----------
    input_dim  : D — number of input features
    hidden_dim : H — width of each hidden layer
    output_dim : C — number of output units (1 for binary classification)
    seed       : random seed for weight initialisation
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int, seed: int = 0):
        _rng   = np.random.default_rng(seed)
        scale  = np.sqrt(2.0 / input_dim)    # He initialisation for ReLU

        # Layer 1: input_dim → hidden_dim
        self.W1 = _rng.normal(0, scale, (hidden_dim, input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Layer 2: hidden_dim → hidden_dim
        scale2  = np.sqrt(2.0 / hidden_dim)
        self.W2 = _rng.normal(0, scale2, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        # Layer 3: hidden_dim → output_dim
        self.W3 = _rng.normal(0, scale2, (output_dim, hidden_dim)).astype(np.float32)
        self.b3 = np.zeros(output_dim, dtype=np.float32)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Rectified linear unit: max(0, x)."""
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : shape (B, input_dim)

        Returns
        -------
        out : shape (B, output_dim) — raw logits, no activation applied.
        """
        h1  = self.relu(x  @ self.W1.T + self.b1)   # (B, hidden_dim)
        h2  = self.relu(h1 @ self.W2.T + self.b2)   # (B, hidden_dim)
        out = h2 @ self.W3.T + self.b3               # (B, output_dim)
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid of the single output unit, shape (B,)."""
        logit = self.forward(x)[:, 0]
        return 1.0 / (1.0 + np.exp(-logit))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.float32)


# ── PyTorch exercise ───────────────────────────────────────────────────────────


class FeedForwardNN(nn.Module):
    """
    Two-hidden-layer fully connected network — your implementation.

    Implement the identical architecture as FeedForwardNumPy:
        fc1: input_dim  → hidden_dim   (nn.Linear)
        fc2: hidden_dim → hidden_dim   (nn.Linear)
        fc3: hidden_dim → output_dim   (nn.Linear)

    forward() must compute:
        fc1 → relu → fc2 → relu → fc3
    and return raw logits of shape (B, output_dim).
    Do NOT apply sigmoid in forward — that is the caller's responsibility.

    Parameters
    ----------
    input_dim  : int
    hidden_dim : int
    output_dim : int
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # ── TODO ─────────────────────────────────────────────────────────────
        # Define three linear layers as attributes named fc1, fc2, fc3:
        #   fc1 maps input_dim  → hidden_dim
        #   fc2 maps hidden_dim → hidden_dim
        #   fc3 maps hidden_dim → output_dim
        # Each is created with nn.Linear(in_features, out_features).
        # Naming them fc1/fc2/fc3 matters — the unit tests check these names.
        # ─────────────────────────────────────────────────────────────────────
        self.fc1 = None  # <- replace
        raise NotImplementedError("TODO: define fc1, fc2, fc3 as nn.Linear layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: fc1 → relu → fc2 → relu → fc3.

        Use F.relu(tensor) for the activation function (imported above
        as torch.nn.functional).  Return the output of fc3 directly —
        no sigmoid, no squeeze.

        Parameters
        ----------
        x : shape (B, input_dim)

        Returns
        -------
        logits : shape (B, output_dim)
        """
        # TODO: implement the forward pass.
        # Sequence: fc1 → ReLU → fc2 → ReLU → fc3
        # Use F.relu(tensor) for activations.
        # Do NOT apply sigmoid or squeeze — return raw logits of shape (B, output_dim).
        # Hint: h1 = F.relu(self.fc1(x))   shape: (B, hidden_dim)
        out = None  # <- replace
        raise NotImplementedError("TODO: implement fc1 → relu → fc2 → relu → fc3")
        return out


# ── training loop ─────────────────────────────────────────────────────────────


def train(model: nn.Module,
          X: torch.Tensor,
          y: torch.Tensor,
          lr: float = 0.05,
          n_epochs: int = 500) -> list:
    """Full-batch gradient descent with BCEWithLogitsLoss."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn   = nn.BCEWithLogitsLoss()
    losses    = []
    for _ in range(n_epochs):
        logits = model(X).squeeze(1)        # (B,)
        loss   = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# ── unit tests ────────────────────────────────────────────────────────────────


def test_numpy_output_shape():
    """NumPy forward must return shape (B, output_dim)."""
    model = FeedForwardNumPy(2, HIDDEN_DIM, 1)
    out   = model.forward(X_np)
    assert out.shape == (len(X_np), 1), (
        "4.3 FAIL – NumPy output shape incorrect\n"
        f"  got: {out.shape}   expected: ({len(X_np)}, 1)"
    )
    print("  ✓  test_numpy_output_shape passed")


def test_numpy_relu_applied():
    """
    ReLU must be active: hidden activations should have non-negative values
    and at least some zeros (dead units on random input).
    """
    model  = FeedForwardNumPy(2, HIDDEN_DIM, 1, seed=42)
    x_test = np.random.default_rng(1).normal(0, 1, (50, 2)).astype(np.float32)
    h1     = model.relu(x_test @ model.W1.T + model.b1)
    assert h1.min() >= 0.0, "4.3 FAIL – ReLU output has negative values"
    print("  ✓  test_numpy_relu_applied passed")


def test_torch_output_shape():
    """FeedForwardNN forward must return shape (B, output_dim)."""
    model = FeedForwardNN(2, HIDDEN_DIM, 1)
    out   = model(X_t)
    assert out.shape == (len(X_t), 1), (
        "4.3 FAIL – PyTorch output shape incorrect\n"
        f"  got: {tuple(out.shape)}   expected: ({len(X_t)}, 1)"
    )
    print("  ✓  test_torch_output_shape passed")


def test_parameter_count():
    """
    Total parameter count must match:
        input_dim*H + H  +  H*H + H  +  H*output_dim + output_dim
    """
    D, H, C = 2, HIDDEN_DIM, 1
    expected = D * H + H  +  H * H + H  +  H * C + C
    model    = FeedForwardNN(D, H, C)
    got      = sum(p.numel() for p in model.parameters())
    assert got == expected, (
        "4.3 FAIL – parameter count incorrect\n"
        f"  got: {got}   expected: {expected}"
    )
    print("  ✓  test_parameter_count passed")


def test_layer_names():
    """Model must expose attributes fc1, fc2, fc3 (used by the auto-grader)."""
    model = FeedForwardNN(2, HIDDEN_DIM, 1)
    for name in ["fc1", "fc2", "fc3"]:
        assert hasattr(model, name) and isinstance(getattr(model, name), nn.Linear), (
            f"4.3 FAIL – model.{name} is not an nn.Linear layer"
        )
    print("  ✓  test_layer_names passed")


def test_torch_matches_numpy():
    """
    When PyTorch weights are copied to the NumPy model, both forward
    passes must agree to within 1e-5.
    """
    torch.manual_seed(SEED)
    torch_model = FeedForwardNN(2, HIDDEN_DIM, 1)
    numpy_model = FeedForwardNumPy(2, HIDDEN_DIM, 1)

    # copy PyTorch weights → NumPy
    numpy_model.W1 = torch_model.fc1.weight.detach().numpy().copy()
    numpy_model.b1 = torch_model.fc1.bias.detach().numpy().copy()
    numpy_model.W2 = torch_model.fc2.weight.detach().numpy().copy()
    numpy_model.b2 = torch_model.fc2.bias.detach().numpy().copy()
    numpy_model.W3 = torch_model.fc3.weight.detach().numpy().copy()
    numpy_model.b3 = torch_model.fc3.bias.detach().numpy().copy()

    with torch.no_grad():
        out_torch = torch_model(X_t).numpy()
    out_numpy = numpy_model.forward(X_np)

    assert np.allclose(out_torch, out_numpy, atol=1e-5), (
        "4.3 FAIL – PyTorch and NumPy outputs differ with identical weights\n"
        f"  max absolute difference: {np.abs(out_torch - out_numpy).max():.2e}"
    )
    print("  ✓  test_torch_matches_numpy passed")


def test_torch_trains():
    """PyTorch model must reach > 80 % accuracy after 500 epochs."""
    torch.manual_seed(SEED)
    model  = FeedForwardNN(2, HIDDEN_DIM, 1)
    losses = train(model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_t).squeeze(1)) >= 0.5).float()
    acc = (preds == y_t).float().mean().item()
    assert acc > 0.80, (
        "4.3 FAIL – model accuracy too low after training\n"
        f"  got: {acc:.4f}   expected: > 0.80"
    )
    print("  ✓  test_torch_trains passed")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Section 4.3 — Feed-Forward Network  [TODO — ungraded]")
    print("=" * 60)

    # NumPy tests (always run — nothing to implement here)
    print("\n--- NumPy reference ---")
    for fn in [test_numpy_output_shape, test_numpy_relu_applied]:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")

    # PyTorch tests (skip gracefully if not yet implemented)
    print("\n--- PyTorch TODO ---")
    for fn in [test_torch_output_shape, test_parameter_count,
               test_layer_names, test_torch_matches_numpy, test_torch_trains]:
        try:
            fn()
        except NotImplementedError as e:
            print(f"  (skipped – not implemented yet: {e})")
        except AssertionError as e:
            print(f"  FAIL: {e}")

    # ── plot (only if PyTorch model is implemented) ───────────────────────────
    try:
        torch.manual_seed(SEED)
        torch_model = FeedForwardNN(2, HIDDEN_DIM, 1)
        losses      = train(torch_model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)

        h = 0.02
        x_min, x_max = X_np[:, 0].min() - 0.4, X_np[:, 0].max() + 0.4
        y_min, y_max = X_np[:, 1].min() - 0.4, X_np[:, 1].max() + 0.4
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid   = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        grid_t = torch.tensor(grid)

        with torch.no_grad():
            Z = torch.sigmoid(torch_model(grid_t).squeeze(1)).numpy().reshape(xx.shape)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        plt.suptitle("Section 4.3 — Feed-Forward Network", fontsize=12)

        axes[0].contourf(xx, yy, Z, alpha=0.3, levels=20, cmap="RdBu")
        axes[0].scatter(X_np[:, 0], X_np[:, 1], c=y_np,
                        cmap="RdBu", edgecolors="k", s=18, linewidths=0.5)
        axes[0].set_title("Decision boundary (PyTorch FCNN)")
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")

        axes[1].plot(losses, linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("BCE loss")
        axes[1].set_title("Training curve")

        plt.tight_layout()
        plt.savefig("4_3_fcnn_exercise.png", dpi=130)
        print("\nPlot saved → 4_3_fcnn_exercise.png")

    except NotImplementedError:
        print("\n(plot skipped — implement FeedForwardNN first)")