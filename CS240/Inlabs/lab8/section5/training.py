"""
Section 5 – torch.autograd + torch.optim: The Training Engine
==============================================================
training_workshop.py

  5.0  The Problem With Manual Updates    [INSPECTION]
  5.1  torch.optim: SGD                   [INSPECTION]
  5.2  Switching Optimisers               [INSPECTION]

Run:
    python training_workshop.py

Trains LogisticRegression (from Section 4) on two-moons three ways:
  (1) manual parameter updates,
  (2) optim.SGD,
  (3) optim.Adam with a StepLR scheduler.
Prints loss curves for each and confirms all three converge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch import optim

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(0)

# ── data ─────────────────────────────────────────────────────────────────────
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
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)


# ── model definition (from Section 4) ────────────────────────────────────────
class LogisticRegression(nn.Module):
    """Binary classifier: linear layer + sigmoid."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1).sigmoid()


# ══ 5.0  Manual updates ══════════════════════════════════════════════════════

def run_manual_sgd():
    """
    Plain SGD written by hand.

    The update rule is simple — but notice how much boilerplate is needed
    just to zero gradients and step every parameter.  Adding momentum or
    learning-rate decay would require tracking extra state tensors manually.
    """
    print("─" * 60)
    print("5.0  Manual SGD")
    print("─" * 60)

    torch.manual_seed(0)
    model = LogisticRegression(input_dim=2)
    lr = 0.1
    loss = torch.tensor(0.0)

    for epoch in range(200):
        y_hat = model(X)
        loss  = F.binary_cross_entropy(y_hat, y)

        # zero accumulated gradients from the previous step
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        loss.backward()

        # manually update every parameter
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= lr * p.grad

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d}  loss={loss.item():.4f}")

    return loss.item()


# ══ 5.1  torch.optim.SGD ════════════════════════════════════════════════════

def run_optim_sgd():
    """
    The same training run using optim.SGD.

    optimizer.zero_grad()  replaces the manual grad-zeroing loop.
    optimizer.step()       replaces the manual parameter-update loop.
    The five-line loop body works for any model and any optimiser.
    """
    print("─" * 60)
    print("5.1  optim.SGD")
    print("─" * 60)

    torch.manual_seed(0)
    model     = LogisticRegression(input_dim=2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = torch.tensor(0.0)

    for epoch in range(200):
        y_hat = model(X)                          # 1. forward pass
        loss  = F.binary_cross_entropy(y_hat, y)  # 2. compute loss
        optimizer.zero_grad()                     # 3. clear old gradients
        loss.backward()                           # 4. compute new gradients
        optimizer.step()                          # 5. update parameters

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d}  loss={loss.item():.4f}")

    return loss.item()


# ══ 5.2  Adam + StepLR ══════════════════════════════════════════════════════

def run_adam_with_scheduler():
    """
    Switch from SGD to Adam by changing one line.
    Add a learning-rate scheduler without touching the loop body.

    Adam tracks per-parameter first and second moment estimates internally —
    none of that state management leaks into the training loop.
    StepLR multiplies the learning rate by gamma every step_size epochs.
    """
    print("─" * 60)
    print("5.2  optim.Adam + StepLR scheduler")
    print("─" * 60)

    torch.manual_seed(0)
    model     = LogisticRegression(input_dim=2)

    # ── swap SGD → Adam: one line changes ────────────────────────────────────
    # optimizer = optim.SGD(model.parameters(), lr=0.1)       # 5.1 version
    optimizer = optim.Adam(model.parameters(), lr=5e-2)       # 5.2 version

    # ── attach a scheduler: lr decays by 0.5× at epochs 50, 100, 150 ────────
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss = torch.tensor(0.0)

    for epoch in range(200):
        y_hat = model(X)                          # 1. forward pass
        loss  = F.binary_cross_entropy(y_hat, y)  # 2. compute loss
        optimizer.zero_grad()                     # 3. clear old gradients
        loss.backward()                           # 4. compute new gradients
        optimizer.step()                          # 5. update parameters
        scheduler.step()                          # decay lr if at milestone

        if (epoch + 1) % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch+1:3d}  loss={loss.item():.4f}  "
                  f"lr={current_lr:.2e}")

    return loss.item()


# ── unit tests ───────────────────────────────────────────────────────────────

def test_manual_and_optim_converge_to_same_loss():
    """Manual SGD and optim.SGD should reach roughly the same final loss."""
    loss_manual = run_manual_sgd()
    print()
    loss_optim  = run_optim_sgd()
    assert abs(loss_manual - loss_optim) < 1e-4, (
        "5.0/5.1 FAIL – manual SGD and optim.SGD diverged\n"
        f"  manual={loss_manual:.6f}  optim={loss_optim:.6f}"
    )
    print("\n  ✓  test_manual_and_optim_converge_to_same_loss passed")


def test_adam_converges():
    """Adam + StepLR should reduce loss to below 0.35 on two-moons."""
    print()
    final_loss = run_adam_with_scheduler()
    assert final_loss < 0.35, (
        "5.2 FAIL – Adam did not converge\n"
        f"  final loss={final_loss:.4f}  expected < 0.35"
    )
    print("\n  ✓  test_adam_converges passed")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("Section 5 — The Training Engine  [INSPECTION]")
    print("=" * 60)
    print()

    for fn in [test_manual_and_optim_converge_to_same_loss,
               test_adam_converges]:
        try:
            fn()
        except NotImplementedError as e:
            print(f"  (skipped – not implemented yet: {e})")
        except AssertionError as e:
            print(f"  FAIL: {e}")

    print()
    print("=" * 60)
    print("All Section 5 checks complete.")
    print("=" * 60)