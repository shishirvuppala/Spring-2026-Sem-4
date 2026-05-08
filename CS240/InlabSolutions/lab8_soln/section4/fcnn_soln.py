"""
Section 4 - torch.nn.Module: The Model Engine
===============================================
fcnn_soln.py

  4.3  Feed-Forward Neural Network           [solution]

Run:
    python fcnn_soln.py

A two-hidden-layer fully connected neural network for binary
classification.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

# -- data ---------------------------------------------------------------------
SEED = 0
N_EPOCHS = 500
LR = 0.05
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


# -- NumPy reference implementation --------------------------------------------


class FeedForwardNumPy:
    """
    Two-hidden-layer fully connected network - reference implementation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
        _rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_dim)

        self.W1 = _rng.normal(0, scale, (hidden_dim, input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W2 = _rng.normal(0, scale2, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        self.W3 = _rng.normal(0, scale2, (output_dim, hidden_dim)).astype(np.float32)
        self.b3 = np.zeros(output_dim, dtype=np.float32)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = self.relu(x @ self.W1.T + self.b1)
        h2 = self.relu(h1 @ self.W2.T + self.b2)
        out = h2 @ self.W3.T + self.b3
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logit = self.forward(x)[:, 0]
        return 1.0 / (1.0 + np.exp(-logit))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.float32)


# -- PyTorch implementation -----------------------------------------------------


class FeedForwardNN(nn.Module):
    """
    Two-hidden-layer fully connected network.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out


# -- training loop --------------------------------------------------------------


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, lr: float = 0.05, n_epochs: int = 500) -> list:
    """Full-batch gradient descent with BCEWithLogitsLoss."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(n_epochs):
        logits = model(X).squeeze(1)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# -- unit tests ----------------------------------------------------------------


def test_numpy_output_shape():
    model = FeedForwardNumPy(2, HIDDEN_DIM, 1)
    out = model.forward(X_np)
    assert out.shape == (len(X_np), 1), (
        "4.3 FAIL - NumPy output shape incorrect\n"
        f"  got: {out.shape}   expected: ({len(X_np)}, 1)"
    )
    print("  check test_numpy_output_shape passed")


def test_numpy_relu_applied():
    model = FeedForwardNumPy(2, HIDDEN_DIM, 1, seed=42)
    x_test = np.random.default_rng(1).normal(0, 1, (50, 2)).astype(np.float32)
    h1 = model.relu(x_test @ model.W1.T + model.b1)
    assert h1.min() >= 0.0, "4.3 FAIL - ReLU output has negative values"
    print("  check test_numpy_relu_applied passed")


def test_torch_output_shape():
    model = FeedForwardNN(2, HIDDEN_DIM, 1)
    out = model(X_t)
    assert out.shape == (len(X_t), 1), (
        "4.3 FAIL - PyTorch output shape incorrect\n"
        f"  got: {tuple(out.shape)}   expected: ({len(X_t)}, 1)"
    )
    print("  check test_torch_output_shape passed")


def test_parameter_count():
    D, H, C = 2, HIDDEN_DIM, 1
    expected = D * H + H + H * H + H + H * C + C
    model = FeedForwardNN(D, H, C)
    got = sum(p.numel() for p in model.parameters())
    assert got == expected, (
        "4.3 FAIL - parameter count incorrect\n"
        f"  got: {got}   expected: {expected}"
    )
    print("  check test_parameter_count passed")


def test_layer_names():
    model = FeedForwardNN(2, HIDDEN_DIM, 1)
    for name in ["fc1", "fc2", "fc3"]:
        assert hasattr(model, name) and isinstance(getattr(model, name), nn.Linear), (
            f"4.3 FAIL - model.{name} is not an nn.Linear layer"
        )
    print("  check test_layer_names passed")


def test_torch_matches_numpy():
    torch.manual_seed(SEED)
    torch_model = FeedForwardNN(2, HIDDEN_DIM, 1)
    numpy_model = FeedForwardNumPy(2, HIDDEN_DIM, 1)

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
        "4.3 FAIL - PyTorch and NumPy outputs differ with identical weights\n"
        f"  max absolute difference: {np.abs(out_torch - out_numpy).max():.2e}"
    )
    print("  check test_torch_matches_numpy passed")


def test_torch_trains():
    torch.manual_seed(SEED)
    model = FeedForwardNN(2, HIDDEN_DIM, 1)
    losses = train(model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_t).squeeze(1)) >= 0.5).float()
    acc = (preds == y_t).float().mean().item()
    assert acc > 0.80, (
        "4.3 FAIL - model accuracy too low after training\n"
        f"  got: {acc:.4f}   expected: > 0.80"
    )
    print("  check test_torch_trains passed")


# -- main ----------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Section 4.3 - Feed-Forward Network  [solution]")
    print("=" * 60)

    print("\n--- NumPy reference ---")
    for fn in [test_numpy_output_shape, test_numpy_relu_applied]:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")

    print("\n--- PyTorch solution ---")
    for fn in [
        test_torch_output_shape,
        test_parameter_count,
        test_layer_names,
        test_torch_matches_numpy,
        test_torch_trains,
    ]:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")

    torch.manual_seed(SEED)
    torch_model = FeedForwardNN(2, HIDDEN_DIM, 1)
    losses = train(torch_model, X_t, y_t, lr=LR, n_epochs=N_EPOCHS)

    h = 0.02
    x_min, x_max = X_np[:, 0].min() - 0.4, X_np[:, 0].max() + 0.4
    y_min, y_max = X_np[:, 1].min() - 0.4, X_np[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = torch.tensor(grid)

    with torch.no_grad():
        Z = torch.sigmoid(torch_model(grid_t).squeeze(1)).numpy().reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plt.suptitle("Section 4.3 - Feed-Forward Network", fontsize=12)

    axes[0].contourf(xx, yy, Z, alpha=0.3, levels=20, cmap="RdBu")
    axes[0].scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="RdBu", edgecolors="k", s=18, linewidths=0.5)
    axes[0].set_title("Decision boundary (PyTorch FCNN)")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    axes[1].plot(losses, linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE loss")
    axes[1].set_title("Training curve")

    plt.tight_layout()
    plt.savefig("4_3_fcnn_exercise.png", dpi=130)
    print("\nPlot saved -> 4_3_fcnn_exercise.png")
